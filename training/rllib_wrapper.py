"""
Wraps the Micro4XEnv (PettingZoo ParallelEnv) into a Ray RLlib
MultiAgentEnv so that RLlib's rollout workers can sample from it directly.

Usage:
    from training.rllib_wrapper import Micro4XMultiAgentEnv
    register_env("micro4x", lambda cfg: Micro4XMultiAgentEnv(cfg))
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces

from env.micro4x_env import Micro4XEnv


class Micro4XMultiAgentEnv(MultiAgentEnv):
    """Thin adapter: PettingZoo ParallelEnv → Ray MultiAgentEnv."""

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.par_env = Micro4XEnv(
            seed=config.get("seed", 42),
            grid_h=config.get("grid_h", 24),
            grid_w=config.get("grid_w", 24),
            num_players=config.get("num_players", 2),
            max_turns=config.get("max_turns", 500),
        )
        self._agent_ids = set(self.par_env.possible_agents)

        # RLlib reads these
        self.observation_space = self.par_env.observation_space("player_0")
        self.action_space = self.par_env.action_space("player_0")

    # Expose engine for external inspection (e.g. interactive client)
    @property
    def engine(self):
        return self.par_env.engine

    def reset(self, *, seed=None, options=None):
        obs, infos = self.par_env.reset(seed=seed, options=options)
        return obs, infos

    def step(self, action_dict):
        # Fill missing agents with NO_OP so engine doesn't crash
        full_actions = {}
        for agent in self.par_env.agents:
            if agent in action_dict:
                full_actions[agent] = action_dict[agent]
            else:
                full_actions[agent] = np.zeros(
                    self.par_env.grid_h * self.par_env.grid_w, dtype=np.int32
                )

        obs, rewards, terms, truncs, infos = self.par_env.step(full_actions)

        # Ray requires __all__ sentinel keys
        terms["__all__"] = len(self.par_env.agents) == 0
        truncs["__all__"] = all(
            truncs.get(a, False) for a in self.par_env.possible_agents
        )

        return obs, rewards, terms, truncs, infos
