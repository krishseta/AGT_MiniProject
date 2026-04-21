"""
MAPPO Training — Multi-Agent PPO with per-player or shared policies.

Modes:
    mappo    — separate policy per player (independent learning)
    selfplay — single shared policy for both players

Usage:
    python training/mappo_train.py --mode mappo
    python training/mappo_train.py --mode selfplay
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from env.micro4x_env import Micro4XEnv
from env.action_mask_model import GridWiseActionMaskModel


def train_mappo():
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env
    from training.rllib_wrapper import Micro4XMultiAgentEnv

    ModelCatalog.register_custom_model(
        "grid_action_mask", GridWiseActionMaskModel
    )

    register_env("micro4x", lambda cfg: Micro4XMultiAgentEnv(cfg))

    ray.init(
        ignore_reinit_error=True,
        num_cpus=8,
        num_gpus=1,
    )

    grid_h, grid_w = 24, 24
    num_players = 2

    # ── Per-player policies ──
    dummy_env = Micro4XEnv(grid_h=grid_h, grid_w=grid_w, num_players=num_players)
    policies = {
        f"player_{i}": (
            None,
            dummy_env.observation_space(f"player_{i}"),
            dummy_env.action_space(f"player_{i}"),
            {},
        )
        for i in range(num_players)
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return agent_id

    config = (
        PPOConfig()
        .environment(
            "micro4x",
            env_config={
                "seed": 42,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "num_players": num_players,
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
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            train_batch_size=4096,
            minibatch_size=512,
            num_epochs=5,
            model={
                "custom_model": "grid_action_mask",
                "custom_model_config": {
                    "grid_h": grid_h,
                    "grid_w": grid_w,
                    "num_obs_channels": 7,
                    "num_action_types": 31,
                },
                "vf_share_layers": True,
                "_disable_preprocessor_api": True,
            },
        )
        .resources(num_gpus=1)
    )

    algo = config.build_algo()

    checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints_mappo")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_reward = float("-inf")
    num_iterations = 1000

    try:
        for i in range(num_iterations):
            result = algo.train()

            env_runners = result.get("env_runners", result)
            p0_reward = env_runners.get("policy_reward_mean", {}).get("player_0")
            p1_reward = env_runners.get("policy_reward_mean", {}).get("player_1")
            ep_len = env_runners.get("episode_len_mean")

            if p0_reward is None or np.isnan(p0_reward):
                p0_reward = 0.0
            if p1_reward is None or np.isnan(p1_reward):
                p1_reward = 0.0
            if ep_len is None or np.isnan(ep_len):
                ep_len = 0.0

            print(
                f"Iter {i + 1:4d} | "
                f"P0={p0_reward:8.2f} | "
                f"P1={p1_reward:8.2f} | "
                f"ep_len={ep_len:6.1f}"
            )

            combined = p0_reward + p1_reward
            if combined > best_reward:
                best_reward = combined
                save_path = algo.save(checkpoint_dir)
                print(f"  -> New best! Saved to {save_path}")

            if (i + 1) % 50 == 0:
                save_path = algo.save(checkpoint_dir)
                print(f"  -> Periodic save: {save_path}")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        save_path = algo.save(checkpoint_dir)
        print(f"Saved to {save_path}")

    finally:
        algo.stop()
        ray.shutdown()

    print(f"\nMAPPO training complete. Best combined reward: {best_reward:.2f}")


def train_self_play():
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env
    from training.rllib_wrapper import Micro4XMultiAgentEnv

    ModelCatalog.register_custom_model(
        "grid_action_mask", GridWiseActionMaskModel
    )

    register_env("micro4x", lambda cfg: Micro4XMultiAgentEnv(cfg))

    ray.init(
        ignore_reinit_error=True,
        num_cpus=8,
        num_gpus=1,
    )

    grid_h, grid_w = 24, 24
    num_players = 2

    dummy_env = Micro4XEnv(grid_h=grid_h, grid_w=grid_w, num_players=num_players)

    policies = {
        "shared_policy": (
            None,
            dummy_env.observation_space("player_0"),
            dummy_env.action_space("player_0"),
            {},
        ),
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"

    config = (
        PPOConfig()
        .environment(
            "micro4x",
            env_config={
                "seed": 42,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "num_players": num_players,
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
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.02,
            vf_loss_coeff=0.5,
            train_batch_size=4096,
            minibatch_size=512,
            num_epochs=5,
            model={
                "custom_model": "grid_action_mask",
                "custom_model_config": {
                    "grid_h": grid_h,
                    "grid_w": grid_w,
                    "num_obs_channels": 7,
                    "num_action_types": 31,
                },
                "vf_share_layers": True,
                "_disable_preprocessor_api": True,
            },
        )
        .resources(num_gpus=1)
    )

    algo = config.build_algo()

    checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints_selfplay")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_reward = float("-inf")
    num_iterations = 1000

    try:
        for i in range(num_iterations):
            result = algo.train()
            env_runners = result.get("env_runners", result)
            mean_reward = env_runners.get("episode_reward_mean")
            ep_len = env_runners.get("episode_len_mean")

            if mean_reward is None or np.isnan(mean_reward):
                mean_reward = 0.0
            if ep_len is None or np.isnan(ep_len):
                ep_len = 0.0

            print(
                f"Iter {i + 1:4d} | "
                f"reward_mean={mean_reward:8.2f} | "
                f"ep_len={ep_len:6.1f}"
            )

            if mean_reward > best_reward:
                best_reward = mean_reward
                save_path = algo.save(checkpoint_dir)
                print(f"  -> New best! Saved to {save_path}")

            if (i + 1) % 50 == 0:
                save_path = algo.save(checkpoint_dir)
                print(f"  -> Periodic save: {save_path}")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        save_path = algo.save(checkpoint_dir)
        print(f"Saved to {save_path}")

    finally:
        algo.stop()
        ray.shutdown()

    print(f"\nSelf-play training complete. Best reward: {best_reward:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["mappo", "selfplay"], default="mappo"
    )
    args = parser.parse_args()

    if args.mode == "mappo":
        train_mappo()
    else:
        train_self_play()
