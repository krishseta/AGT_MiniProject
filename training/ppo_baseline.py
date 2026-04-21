import numpy as np
import gymnasium
import argparse
import glob

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from env.micro4x_env import Micro4XEnv
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1


def train_ppo(resume_checkpoint=None):
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env

    from env.action_mask_model import GridWiseActionMaskModel

    # Register custom model
    ModelCatalog.register_custom_model(
        "grid_action_mask", GridWiseActionMaskModel
    )

    # ---- ENV CREATOR ----
    def env_creator(config):
        env = Micro4XEnv(
            grid_h=config.get("grid_h", 24),
            grid_w=config.get("grid_w", 24),
            max_turns=500
        )
        env = pettingzoo_env_to_vec_env_v1(env)

        # ⚠️ keep this small to avoid oversubscription
        env = concat_vec_envs_v1(env, 4)

        return env

    register_env("micro4x_vec", env_creator)

    ray.init(
        ignore_reinit_error=True,
        num_cpus=8,
        num_gpus=1,
    )

    grid_h, grid_w = 24, 24

    config = (
        PPOConfig()
        .environment(
            "micro4x_vec",
            env_config={
                "grid_h": grid_h,
                "grid_w": grid_w,
            },
        )
        # ✅ REQUIRED for custom_model
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .framework("torch")
        .training(
            # Learning
            lr=2e-4,
            gamma=0.995,
            lambda_=0.95,

            train_batch_size=8000,
            minibatch_size=1024,     # ✅ FIXED (was sgd_minibatch_size)
            num_epochs=3,            # ✅ FIXED (was num_sgd_iter)

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

    # ❌ REMOVE env_runners (conflicts with old API)
    # ❌ DO NOT add rollouts either (deprecated mess)

    # ---- CHECKPOINT RESUME ----
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

    num_iterations = 100

    try:
        for i in range(num_iterations):
            result = algo.train()

            mean_reward = result.get("episode_reward_mean", 0.0)
            episode_len = result.get("episode_len_mean", 0.0)

            if np.isnan(mean_reward):
                mean_reward = 0.0
            if np.isnan(episode_len):
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
    # RLlib saves checkpoints as directories like checkpoint_000001/
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

    train_ppo(resume_checkpoint=resume_path)