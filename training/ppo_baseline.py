"""
PPO Baseline — shared-policy self-play on Micro-4X.

Both players share a single policy ("shared_policy") so the agent
learns to play from either side of the board.

Usage:
    python training/ppo_baseline.py                     # fresh run
    python training/ppo_baseline.py --resume latest     # resume
    python training/ppo_baseline.py --iterations 200    # longer run
"""

import numpy as np
import argparse
import glob

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def train_ppo(resume_checkpoint=None, num_iterations=100):
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env

    from env.action_mask_model import GridWiseActionMaskModel
    from training.rllib_wrapper import Micro4XMultiAgentEnv
    from env.micro4x_env import Micro4XEnv

    # ── Register custom model ──
    ModelCatalog.register_custom_model(
        "grid_action_mask", GridWiseActionMaskModel
    )

    # ── Register environment (MultiAgentEnv wrapper) ──
    register_env("micro4x", lambda cfg: Micro4XMultiAgentEnv(cfg))

    ray.init(
        ignore_reinit_error=True,
        num_cpus=8,
        num_gpus=1,
    )

    grid_h, grid_w = 24, 24

    # ── Observation / action spaces for policy spec ──
    dummy_env = Micro4XEnv(grid_h=grid_h, grid_w=grid_w)
    obs_space = dummy_env.observation_space("player_0")
    act_space = dummy_env.action_space("player_0")

    policies = {
        "shared_policy": (None, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"

    # ── PPO Config ──
    config = (
        PPOConfig()
        .environment(
            "micro4x",
            env_config={
                "grid_h": grid_h,
                "grid_w": grid_w,
                "max_turns": 500,
            },
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            lr=2e-4,
            gamma=0.995,
            lambda_=0.95,

            train_batch_size=8000,
            minibatch_size=1024,
            num_epochs=3,

            clip_param=0.2,
            vf_clip_param=10.0,
            grad_clip=0.5,

            kl_coeff=0.2,
            kl_target=0.01,

            entropy_coeff_schedule=[
                (0, 0.05),
                (300_000, 0.02),
                (800_000, 0.005),
            ],

            model={
                "custom_model": "grid_action_mask",
                "custom_model_config": {
                    "grid_h": grid_h,
                    "grid_w": grid_w,
                    "num_action_types": 31,
                },
                "vf_share_layers": True,
            },
        )
        .resources(num_gpus=1)
    )

    # ── Build / restore ──
    if resume_checkpoint:
        print(f"Restoring from checkpoint: {resume_checkpoint}")
        algo = config.build_algo()
        algo.restore(resume_checkpoint)
        print("  Checkpoint restored successfully.")
    else:
        algo = config.build_algo()

    best_reward = float("-inf")

    checkpoint_dir = os.path.join(
        os.path.dirname(__file__), "..", "checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        for i in range(num_iterations):
            result = algo.train()

            # Extract metrics — handle different result structures
            env_runners = result.get("env_runners", result)
            mean_reward = (
                env_runners.get("episode_reward_mean")
                or result.get("episode_reward_mean", 0.0)
            )
            episode_len = (
                env_runners.get("episode_len_mean")
                or result.get("episode_len_mean", 0.0)
            )

            if mean_reward is None or np.isnan(mean_reward):
                mean_reward = 0.0
            if episode_len is None or np.isnan(episode_len):
                episode_len = 0.0

            print(
                f"Iter {i + 1:4d} | "
                f"reward_mean={mean_reward:8.2f} | "
                f"ep_len_mean={episode_len:6.1f}"
            )

            # Save best
            if mean_reward > best_reward:
                best_reward = mean_reward
                save_path = algo.save(checkpoint_dir)
                print(f"  -> New best! Saved to {save_path}")

            # Periodic save
            if (i + 1) % 10 == 0:
                save_path = algo.save(checkpoint_dir)
                print(f"  -> Periodic save: {save_path}")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        save_path = algo.save(checkpoint_dir)
        print(f"Saved to {save_path}")

    finally:
        algo.stop()
        ray.shutdown()

    print(f"\nTraining complete. Best reward: {best_reward:.2f}")


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recently modified checkpoint directory."""
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*"))
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    return latest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Micro-4X PPO Training")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to an RLlib checkpoint directory to resume from. "
             "Use --resume latest to auto-find the most recent checkpoint."
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Number of training iterations."
    )
    args = parser.parse_args()

    resume_path = args.resume
    if resume_path == "latest":
        checkpoint_dir = os.path.join(
            os.path.dirname(__file__), "..", "checkpoints"
        )
        resume_path = find_latest_checkpoint(checkpoint_dir)
        if resume_path:
            print(f"Auto-detected latest checkpoint: {resume_path}")
        else:
            print("No existing checkpoints found — starting fresh.")
            resume_path = None

    train_ppo(resume_checkpoint=resume_path, num_iterations=args.iterations)