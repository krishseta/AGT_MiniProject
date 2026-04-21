import numpy as np
import gymnasium
from gymnasium.spaces import Box, Dict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from env.micro4x_env import Micro4XEnv
from training.rule_bot import RuleBot


class SingleAgentWrapper(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_config=None):
        super().__init__()
        config = env_config or {}

        seed = config.get("seed", 42)
        grid_h = config.get("grid_h", 16)
        grid_w = config.get("grid_w", 16)
        max_turns = config.get("max_turns", 200)

        self.base_env = Micro4XEnv(
            seed=seed,
            grid_h=grid_h,
            grid_w=grid_w,
            num_players=2,
            max_turns=max_turns,
        )

        self.bot = RuleBot(1, self.base_env.engine)

        self.learning_agent = "player_0"
        
        # Override observation space manually
        self.observation_space = Dict({
            "observation": Box(-np.inf, np.inf, (16, 16, 7), dtype=np.float32),
            "action_mask": Box(0.0, 1.0, (7936,), dtype=np.float32)
        })
        self.action_space = self.base_env.action_space(self.learning_agent)

        self._last_obs = None

    def _sanitize_obs(self, obs):
        observation = np.asarray(obs["observation"], dtype=np.float32)
        action_mask = np.asarray(obs["action_mask"], dtype=np.float32)

        # Clip mask to exactly 0 or 1
        action_mask = np.clip(action_mask, 0.0, 1.0)

        # Replace NaNs or Infs
        np.nan_to_num(observation, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(action_mask, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Validation assertions
        assert not np.isnan(observation).any()
        assert not np.isinf(observation).any()
        assert not np.isnan(action_mask).any()
        assert not np.isinf(action_mask).any()
        assert action_mask.shape == (7936,)

        return {
            "observation": observation,
            "action_mask": action_mask
        }

    def reset(self, *, seed=None, options=None):
        obs, infos = self.base_env.reset(seed=seed)
        self._last_obs = self._sanitize_obs(obs[self.learning_agent])
        return self._last_obs, infos.get(self.learning_agent, {})

    def step(self, action):
        bot_action = self.bot.get_action().flatten()

        actions = {
            self.learning_agent: action,
            "player_1": bot_action,
        }

        obs, rewards, terms, truncs, infos = self.base_env.step(actions)

        if self.learning_agent in obs:
            self._last_obs = self._sanitize_obs(obs[self.learning_agent])

        reward = rewards.get(self.learning_agent, 0.0)
        terminated = terms.get(self.learning_agent, False)
        truncated = truncs.get(self.learning_agent, False)
        info = infos.get(self.learning_agent, {})

        return self._last_obs, reward, terminated, truncated, info


def train_ppo():
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog

    from env.action_mask_model import GridWiseActionMaskModel

    # Register custom model
    ModelCatalog.register_custom_model(
        "grid_action_mask", GridWiseActionMaskModel
    )

    ray.init(
        ignore_reinit_error=True,
        num_cpus=4,
        num_gpus=1,
        object_store_memory=2 * 1024 * 1024 * 1024,
    )

    grid_h, grid_w = 16, 16

    config = (
        PPOConfig()
        .environment(
            SingleAgentWrapper,
            env_config={
                "seed": 42,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "max_turns": 200,
            },
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            sample_timeout_s=1200,
        )
        .framework("torch")
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            train_batch_size=2048,
            minibatch_size=256,
            num_epochs=10,
            model={
                "custom_model": "grid_action_mask",
                "custom_model_config": {
                    "grid_h": grid_h,
                    "grid_w": grid_w,
                    "num_obs_channels": 7,
                    "num_action_types": 31,
                },
            },
        )
        .resources(
            num_gpus=1,
        )
    )
    # ✅ new correct builder
    algo = config.build_algo()

    best_reward = float("-inf")

    checkpoint_dir = os.path.join(
        os.path.dirname(__file__), "..", "checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_iterations = 500

    for i in range(num_iterations):
        result = algo.train()

        env_runners = result.get("env_runners", result)
        mean_reward = env_runners.get("episode_reward_mean")
        episode_len = env_runners.get("episode_len_mean")

        if mean_reward is None or np.isnan(mean_reward):
            mean_reward = 0.0
        if episode_len is None or np.isnan(episode_len):
            episode_len = 0.0

        print(
            f"Iter {i + 1:4d} | "
            f"reward_mean={mean_reward:8.2f} | "
            f"ep_len_mean={episode_len:6.1f}"
        )

        if mean_reward > best_reward:
            best_reward = mean_reward
            save_path = algo.save(checkpoint_dir)
            print(f"  -> New best! Saved to {save_path}")

    algo.stop()
    ray.shutdown()

    print(f"\nTraining complete. Best reward: {best_reward:.2f}")


if __name__ == "__main__":
    train_ppo()