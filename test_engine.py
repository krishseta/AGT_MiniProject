import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine import GameEngine
from core.config import (
    TerrainType, UnitType, ActionType, TechType,
    NUM_TERRAIN_TYPES, NUM_OBS_CHANNELS, NUM_ACTION_TYPES,
    UNIT_MAX_HP, UNIT_ATTACK_POWER, TERRAIN_DEF_MOD,
)


def test_terrain_generation():
    print("=" * 60)
    print("TEST: Terrain Generation")
    print("=" * 60)
    engine = GameEngine(seed=42, h=16, w=16, num_players=2)

    assert engine.terrain.shape == (16, 16)
    counts = np.bincount(engine.terrain.flatten(), minlength=NUM_TERRAIN_TYPES)
    for tt in TerrainType:
        print(f"  {tt.name}: {counts[tt]} tiles")

    assert counts[TerrainType.CITY] >= 2
    print(f"  Start positions: {engine.start_positions.tolist()}")
    print(f"  Resource caches: {engine.resource_caches.tolist()}")

    for i in range(len(engine.start_positions)):
        sy, sx = engine.start_positions[i]
        assert engine.terrain[sy, sx] == TerrainType.CITY
    print("  PASSED\n")


def test_observation_space():
    print("=" * 60)
    print("TEST: Observation Space")
    print("=" * 60)
    engine = GameEngine(seed=42, h=16, w=16, num_players=2)

    obs = engine.get_observation(0)
    assert obs.shape == (16, 16, NUM_OBS_CHANNELS)
    print(f"  Observation shape: {obs.shape}")

    fog_mask = obs[:, :, 6]
    visible_count = int(np.sum(fog_mask > 0))
    print(f"  Visible tiles (P0): {visible_count}")
    assert visible_count > 0

    invisible = fog_mask == 0
    for c in range(6):
        leaked = np.any(obs[:, :, c][invisible] != -1.0)
        assert not leaked, f"Data leaked on channel {c}"
    print("  Fog of war masking: No data leakage")
    print("  PASSED\n")


def test_action_mask():
    print("=" * 60)
    print("TEST: Action Mask")
    print("=" * 60)
    engine = GameEngine(seed=42, h=16, w=16, num_players=2)

    mask = engine.get_action_mask(0)
    assert mask.shape == (16, 16, NUM_ACTION_TYPES)
    print(f"  Action mask shape: {mask.shape}")

    total_valid = int(np.sum(mask))
    print(f"  Total valid actions: {total_valid}")
    assert total_valid > 0

    noop_count = int(np.sum(mask[:, :, ActionType.NO_OP]))
    assert noop_count == 16 * 16
    print(f"  NO_OP valid everywhere: {noop_count} tiles")

    for at in ActionType:
        count = int(np.sum(mask[:, :, at]))
        if count > 0 and at != ActionType.NO_OP:
            print(f"  {at.name}: {count}")
    print("  PASSED\n")


def test_movement():
    print("=" * 60)
    print("TEST: Unit Movement")
    print("=" * 60)
    engine = GameEngine(seed=42, h=16, w=16, num_players=2)

    p0_units = engine.units.get_player_units(0)
    assert len(p0_units) > 0, "Player 0 has no units"
    uid = p0_units[0]
    old_pos = engine.units.positions[uid].copy()
    print(f"  Unit {uid} at {old_pos.tolist()}")

    mask = engine.get_action_mask(0)
    uy, ux = int(old_pos[0]), int(old_pos[1])
    moved = False
    chosen_dir = -1
    for d in range(8):
        if mask[uy, ux, ActionType.MOVE_N + d]:
            chosen_dir = d
            break

    if chosen_dir >= 0:
        actions = {
            p: np.zeros((engine.h, engine.w), dtype=np.int32)
            for p in range(engine.num_players)
        }
        actions[0][uy, ux] = ActionType.MOVE_N + chosen_dir
        obs, rewards, terminal, infos = engine.step(actions)
        new_pos = engine.units.positions[uid]
        print(f"  Moved to {new_pos.tolist()}")
        assert not np.array_equal(old_pos, new_pos)
        moved = True

    assert moved, "No valid move found"
    print("  PASSED\n")


def test_combat():
    print("=" * 60)
    print("TEST: Deterministic Combat + Retaliation")
    print("=" * 60)
    engine = GameEngine(seed=42, h=16, w=16, num_players=2)

    plains_mask = engine.terrain == TerrainType.PLAINS
    unit_grid = engine.units.build_unit_grid(engine.h, engine.w)
    plains_yx = np.argwhere(plains_mask)

    combat_y, combat_x = -1, -1
    for i in range(len(plains_yx)):
        y, x = int(plains_yx[i, 0]), int(plains_yx[i, 1])
        if (x + 1 < engine.w and
                engine.terrain[y, x + 1] == TerrainType.PLAINS and
                unit_grid[y, x] < 0 and
                unit_grid[y, x + 1] < 0):
            combat_y, combat_x = y, x
            break

    assert combat_y >= 0, "No adjacent plains tiles found"
    print(f"  Combat at ({combat_y},{combat_x}) -> ({combat_y},{combat_x + 1})")

    attacker = engine.units.spawn(0, UnitType.INFANTRY, combat_y, combat_x)
    defender = engine.units.spawn(1, UnitType.INFANTRY, combat_y, combat_x + 1)
    assert attacker >= 0 and defender >= 0

    atk_hp_before = float(engine.units.hp[attacker])
    def_hp_before = float(engine.units.hp[defender])
    print(f"  Attacker HP: {atk_hp_before}, Defender HP: {def_hp_before}")

    expected_dmg = float(UNIT_ATTACK_POWER[UnitType.INFANTRY]) / float(TERRAIN_DEF_MOD[TerrainType.PLAINS])
    print(f"  Expected damage: {expected_dmg:.2f}")

    actions = {
        p: np.zeros((engine.h, engine.w), dtype=np.int32)
        for p in range(engine.num_players)
    }
    actions[0][combat_y, combat_x] = ActionType.ATTACK_E

    engine.units.reset_turn()
    obs, rewards, terminal, infos = engine.step(actions)

    def_hp_after = float(engine.units.hp[defender])
    atk_hp_after = float(engine.units.hp[attacker])
    actual_dmg = def_hp_before - def_hp_after

    print(f"  Damage dealt: {actual_dmg:.2f}")
    print(f"  Defender HP after: {def_hp_after:.2f}")
    print(f"  Attacker HP after (retaliation): {atk_hp_after:.2f}")

    assert actual_dmg > 0, "No damage dealt"

    if engine.units.alive[defender]:
        assert atk_hp_after < atk_hp_before, "No retaliation damage"
        print("  Retaliation confirmed")
    else:
        print("  Defender killed (no retaliation)")

    print("  PASSED\n")


def test_economy():
    print("=" * 60)
    print("TEST: Economy Tick")
    print("=" * 60)
    engine = GameEngine(seed=42, h=16, w=16, num_players=2)

    currency_before = float(engine.economy.currency[0])
    print(f"  P0 currency before: {currency_before:.1f}")

    actions = {
        p: np.zeros((engine.h, engine.w), dtype=np.int32)
        for p in range(engine.num_players)
    }
    obs, rewards, terminal, infos = engine.step(actions)

    currency_after = float(engine.economy.currency[0])
    income = infos[0]["income"]
    print(f"  P0 income: {income:.1f}")
    print(f"  P0 currency after: {currency_after:.1f}")
    assert currency_after > currency_before
    print("  PASSED\n")


def test_multi_turn_simulation():
    print("=" * 60)
    print("TEST: Multi-Turn Simulation (10 turns)")
    print("=" * 60)
    engine = GameEngine(seed=42, h=16, w=16, num_players=2)

    for t in range(10):
        actions = {
            p: np.zeros((engine.h, engine.w), dtype=np.int32)
            for p in range(engine.num_players)
        }

        mask = engine.get_action_mask(0)
        p0_units = engine.units.get_player_units(0)
        if len(p0_units) > 0:
            uid = p0_units[0]
            uy, ux = int(engine.units.positions[uid, 0]), int(engine.units.positions[uid, 1])
            for d in range(8):
                if mask[uy, ux, ActionType.MOVE_N + d]:
                    actions[0][uy, ux] = ActionType.MOVE_N + d
                    break

        obs, rewards, terminal, infos = engine.step(actions)
        print(f"  Turn {engine.turn}: P0=${infos[0]['currency']:.1f} P1=${infos[1]['currency']:.1f} | terminal={terminal}")
        if terminal:
            break

    assert engine.turn >= 10 or terminal
    print("  PASSED\n")


def test_json_logging():
    print("=" * 60)
    print("TEST: JSON State Logging")
    print("=" * 60)
    engine = GameEngine(seed=42, h=16, w=16, num_players=2)

    actions = {
        p: np.zeros((engine.h, engine.w), dtype=np.int32)
        for p in range(engine.num_players)
    }
    for _ in range(3):
        engine.step(actions)

    log_path = "test_game_log.json"
    engine.save_log(log_path)

    with open(log_path, "r") as f:
        log = json.load(f)

    assert len(log) == 3
    print(f"  Log entries: {len(log)}")

    last = log[-1]
    assert "turn" in last
    assert "terrain" in last
    assert "units" in last
    assert "economy" in last
    assert "events" in last
    print(f"  Last turn: {last['turn']}")
    print(f"  Units logged: {len(last['units'])}")
    print(f"  Events logged: {len(last['events'])}")

    os.remove(log_path)
    print("  PASSED\n")


def test_artillery_no_retaliation():
    print("=" * 60)
    print("TEST: Artillery No-Retaliation Mechanic")
    print("=" * 60)
    engine = GameEngine(seed=42, h=16, w=16, num_players=2)

    plains_yx = np.argwhere(engine.terrain == TerrainType.PLAINS)
    unit_grid = engine.units.build_unit_grid(engine.h, engine.w)

    pos = None
    for i in range(len(plains_yx)):
        y, x = int(plains_yx[i, 0]), int(plains_yx[i, 1])
        if (x + 2 < engine.w and
                engine.terrain[y, x + 2] == TerrainType.PLAINS and
                unit_grid[y, x] < 0 and
                unit_grid[y, x + 2] < 0):
            all_clear = True
            for cx in range(x, x + 3):
                if unit_grid[y, cx] >= 0:
                    all_clear = False
            if all_clear:
                pos = (y, x)
                break

    if pos is None:
        print("  SKIPPED (no suitable position)\n")
        return

    y, x = pos
    artillery = engine.units.spawn(0, UnitType.ARTILLERY, y, x)
    target = engine.units.spawn(1, UnitType.INFANTRY, y, x + 2)
    art_hp_before = float(engine.units.hp[artillery])
    print(f"  Artillery at ({y},{x}), Target at ({y},{x + 2})")

    actions = {
        p: np.zeros((engine.h, engine.w), dtype=np.int32)
        for p in range(engine.num_players)
    }
    actions[0][y, x] = ActionType.ATTACK_E

    engine.units.reset_turn()
    engine.step(actions)

    art_hp_after = float(engine.units.hp[artillery])
    print(f"  Artillery HP: {art_hp_before} -> {art_hp_after}")
    assert art_hp_after == art_hp_before, "Artillery took retaliation damage"
    print("  No retaliation confirmed")
    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MARL MICRO-4X ENGINE — PHASE 1 TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        test_terrain_generation,
        test_observation_space,
        test_action_mask,
        test_movement,
        test_combat,
        test_economy,
        test_multi_turn_simulation,
        test_json_logging,
        test_artillery_no_retaliation,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)
