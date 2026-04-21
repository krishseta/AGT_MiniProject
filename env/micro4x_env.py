import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.engine import GameEngine
from core.config import (
    NUM_OBS_CHANNELS, NUM_ACTION_TYPES, NUM_TERRAIN_TYPES,
    NUM_UNIT_TYPES, MAX_TURNS
)


class Micro4XEnv(ParallelEnv):
    """PettingZoo Parallel env for Micro-4X.
    
    Observation is a flat Box vector: [action_mask | observation_flat].
    The custom model splits it back into the mask and the 3-D grid tensor.
    This avoids RLlib DictFlatteningPreprocessor bugs.
    """
    metadata = {"name": "micro4x_v0", "render_modes": ["human"], "is_parallelizable": True}

    def __init__(self, seed=42, grid_h=24, grid_w=24, num_players=2, max_turns=MAX_TURNS, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_players = num_players
        self.max_turns = max_turns
        self._seed = seed

        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents = list(self.possible_agents)

        self.engine = GameEngine(seed=seed, h=grid_h, w=grid_w, num_players=num_players)

        self._mask_size = grid_h * grid_w * NUM_ACTION_TYPES
        self._obs_size = grid_h * grid_w * NUM_OBS_CHANNELS
        total_size = self._mask_size + self._obs_size

        # Flat Box: first _mask_size floats = action mask, rest = observation
        self._obs_space = spaces.Box(
            low=-1.0,
            high=100.0,
            shape=(total_size,),
            dtype=np.float32,
        )

        self._action_space = spaces.MultiDiscrete(
            np.full(grid_h * grid_w, NUM_ACTION_TYPES, dtype=np.int32)
        )

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._action_space

    def _pack_obs(self, obs_3d, mask_3d):
        """Pack observation grid and action mask into a single flat vector."""
        mask_flat = mask_3d.reshape(-1).astype(np.float32)
        obs_flat = obs_3d.reshape(-1).astype(np.float32)
        return np.concatenate([mask_flat, obs_flat])

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        self.engine.reset(seed=self._seed)
        self.agents = list(self.possible_agents)

        observations = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            obs = self.engine.get_observation(i)
            mask = self.engine.get_action_mask(i)
            observations[agent] = self._pack_obs(obs, mask)
            infos[agent] = {}

        return observations, infos

    def step(self, actions):
        engine_actions = {}
        for i in range(self.num_players):
            agent = f"player_{i}"
            if agent in actions:
                flat = np.asarray(actions[agent], dtype=np.int32)
                engine_actions[i] = flat.reshape(self.grid_h, self.grid_w)
            else:
                engine_actions[i] = np.zeros(
                    (self.grid_h, self.grid_w), dtype=np.int32
                )

        obs_dict, rewards_arr, terminal, infos_dict = self.engine.step(engine_actions)

        truncated_all = self.engine.turn >= self.max_turns

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for i, agent in enumerate(self.possible_agents):
            if agent not in self.agents:
                continue

            terminated = not self.engine.player_alive[i]

            observations[agent] = self._pack_obs(
                obs_dict[i],
                infos_dict[i]["action_mask"],
            )
            rewards[agent] = float(rewards_arr[i])
            terminations[agent] = bool(terminated)
            truncations[agent] = bool(truncated_all and not terminated)
            infos[agent] = {
                "income": infos_dict[i]["income"],
                "currency": infos_dict[i]["currency"],
            }

        self.agents = [
            a
            for a in self.agents
            if not terminations.get(a, False) and not truncations.get(a, False)
        ]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    def close(self):
        pass
