import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.micro4x_env import Micro4XEnv
from training.rule_bot import RuleBot
from training.ppo_baseline import SingleAgentWrapper
from core.config import NUM_OBS_CHANNELS, NUM_ACTION_TYPES


def test_pettingzoo_reset():
    print("=" * 60)
    print("TEST: PettingZoo Reset")
    print("=" * 60)

    env = Micro4XEnv(seed=42, grid_h=16, grid_w=16, num_players=2)
    observations, infos = env.reset()

    assert set(observations.keys()) == {"player_0", "player_1"}
    for agent in observations:
        obs = observations[agent]
        assert "observation" in obs
        assert "action_mask" in obs
        assert obs["observation"].shape == (16, 16, NUM_OBS_CHANNELS)
        assert obs["action_mask"].shape == (16 * 16 * NUM_ACTION_TYPES,)
        assert obs["action_mask"].dtype == np.float32
        print(f"  {agent}: obs={obs['observation'].shape}, mask_sum={int(np.sum(obs['action_mask']))}")

    print("  PASSED\n")


def test_pettingzoo_step():
    print("=" * 60)
    print("TEST: PettingZoo Step Cycle")
    print("=" * 60)

    env = Micro4XEnv(seed=42, grid_h=16, grid_w=16, num_players=2)
    observations, infos = env.reset()

    for t in range(5):
        actions = {}
        for agent in env.agents:
            mask = observations[agent]["action_mask"]
            flat_mask = mask.reshape(16 * 16, NUM_ACTION_TYPES)
            action = np.zeros(16 * 16, dtype=np.int32)
            for tile_idx in range(16 * 16):
                valid = np.where(flat_mask[tile_idx] > 0)[0]
                action[tile_idx] = np.random.choice(valid)
            actions[agent] = action

        observations, rewards, terminations, truncations, infos = env.step(actions)
        alive = [a for a in env.agents]
        print(
            f"  Step {t + 1}: alive={alive}, "
            f"rewards={[f'{rewards.get(a, 0):.2f}' for a in env.possible_agents]}"
        )

        for agent in env.agents:
            if agent in observations:
                obs = observations[agent]
                obs_space = env.observation_space(agent)
                assert obs_space.contains(obs), f"Obs out of space for {agent}"

    print("  PASSED\n")


def test_action_mask_validity():
    print("=" * 60)
    print("TEST: Action Mask Validity (No Illegal Actions)")
    print("=" * 60)

    env = Micro4XEnv(seed=42, grid_h=16, grid_w=16, num_players=2)
    observations, _ = env.reset()

    for agent in env.agents:
        mask = observations[agent]["action_mask"]
        flat_mask = mask.reshape(16 * 16, NUM_ACTION_TYPES)

        for tile_idx in range(16 * 16):
            valid = flat_mask[tile_idx]
            assert valid[0] == 1.0, f"NO_OP must always be valid at tile {tile_idx}"
            assert np.sum(valid) >= 1, f"At least one action must be valid"

    print("  All tiles have valid NO_OP and at least one valid action")
    print("  PASSED\n")


def test_rule_bot_integration():
    print("=" * 60)
    print("TEST: RuleBot Integration")
    print("=" * 60)

    env = Micro4XEnv(seed=42, grid_h=16, grid_w=16, num_players=2)
    env.reset()

    bot = RuleBot(1, env.engine)

    for t in range(15):
        bot_action = bot.get_action()
        assert bot_action.shape == (16, 16)

        mask = env.engine.get_action_mask(1)
        for y in range(16):
            for x in range(16):
                chosen = bot_action[y, x]
                assert mask[y, x, chosen], (
                    f"Illegal action {chosen} at ({y},{x}) on turn {t + 1}"
                )

        actions = {
            "player_0": np.zeros(16 * 16, dtype=np.int32),
            "player_1": bot_action.flatten(),
        }
        observations, rewards, terms, truncs, infos = env.step(actions)

        p1_units = env.engine.units.count_player_alive(1)
        p1_currency = float(env.engine.economy.currency[1])
        print(f"  Turn {t + 1}: Bot units={p1_units}, currency={p1_currency:.1f}")

        if not env.agents:
            break

    print("  All bot actions were legal across 15 turns")
    print("  PASSED\n")


def test_single_agent_wrapper():
    print("=" * 60)
    print("TEST: SingleAgentWrapper (Gymnasium API)")
    print("=" * 60)

    env = SingleAgentWrapper(env_config={
        "seed": 42, "grid_h": 16, "grid_w": 16, "max_turns": 50,
    })

    obs, info = env.reset()
    assert "observation" in obs
    assert "action_mask" in obs
    print(f"  Reset: obs_shape={obs['observation'].shape}")

    total_reward = 0.0
    for t in range(20):
        mask = obs["action_mask"].reshape(16 * 16, NUM_ACTION_TYPES)
        action = np.zeros(16 * 16, dtype=np.int32)
        for tile_idx in range(16 * 16):
            valid = np.where(mask[tile_idx] > 0)[0]
            action[tile_idx] = np.random.choice(valid)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"  Episode ended at step {t + 1}: reward={total_reward:.2f}")
            break
    else:
        print(f"  20 steps completed: total_reward={total_reward:.2f}")

    print("  PASSED\n")


def test_obs_space_containment():
    print("=" * 60)
    print("TEST: Observation Space Containment")
    print("=" * 60)

    env = SingleAgentWrapper(env_config={
        "seed": 42, "grid_h": 16, "grid_w": 16, "max_turns": 50,
    })
    obs, _ = env.reset()

    assert env.observation_space.contains(obs), "Initial obs not in space"
    print(f"  Initial obs in space: True")

    for t in range(5):
        mask = obs["action_mask"].reshape(16 * 16, NUM_ACTION_TYPES)
        action = np.zeros(16 * 16, dtype=np.int32)
        for tile_idx in range(16 * 16):
            valid = np.where(mask[tile_idx] > 0)[0]
            action[tile_idx] = np.random.choice(valid)

        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), f"Obs not in space at step {t + 1}"
        if terminated or truncated:
            break

    print(f"  All observations within declared space bounds")
    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MARL MICRO-4X — PHASE 2/3 TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        test_pettingzoo_reset,
        test_pettingzoo_step,
        test_action_mask_validity,
        test_rule_bot_integration,
        test_single_agent_wrapper,
        test_obs_space_containment,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)
