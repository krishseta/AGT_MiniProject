import numpy as np
from .config import (
    GRID_HEIGHT, GRID_WIDTH, NUM_PLAYERS, NUM_ACTION_TYPES,
    NUM_OBS_CHANNELS, NUM_UNIT_TYPES, NUM_TECHS, MAX_TURNS, GAMMA,
    TerrainType, UnitType, ActionType, TechType,
    DIRECTION_DELTAS, TERRAIN_MOVE_COST, TERRAIN_VISION,
    UNIT_MAX_HP, UNIT_COST, UNIT_RANGE, UNIT_MOUNTAIN_IMPASSABLE,
    UNIT_MOVE_AFTER_ATTACK, UNIT_PERSIST_ON_KILL, UNIT_NO_RETALIATION,
    TECH_COST, STARTING_CURRENCY,
    REWARD_DESTROY_UNIT_SCALE, REWARD_CAPTURE_CITY,
    REWARD_UNIT_SURVIVES, REWARD_ECON_GROWTH,
    REWARD_WIN, REWARD_LOSS
)
from .terrain import generate_terrain
from .units import UnitManager
from .combat import resolve_attack
from .economy import EconomyManager
from .game_theory import (
    detect_border_contacts, resolve_ipd, resolve_stag_hunt,
    compute_ownership_grid
)
from .fog import compute_visibility
from .logger import serialize_state, save_turn_log


class GameEngine:

    def __init__(self, seed=42, h=GRID_HEIGHT, w=GRID_WIDTH, num_players=NUM_PLAYERS):
        self.h = h
        self.w = w
        self.num_players = num_players
        self.seed = seed
        self.turn = 0
        self.terrain = None
        self.start_positions = None
        self.resource_caches = None
        self.units = None
        self.economy = None
        self.player_alive = None
        self.capital_cities = None
        self.turn_log = []
        self._build(seed)

    def _build(self, seed):
        self.terrain, self.start_positions, self.resource_caches = generate_terrain(
            self.h, self.w, seed, self.num_players
        )
        self.units = UnitManager()
        self.economy = EconomyManager(self.num_players, self.h, self.w)
        self.player_alive = np.ones(self.num_players, dtype=np.bool_)
        self.capital_cities = np.full((self.num_players, 2), -1, dtype=np.int32)
        self.turn = 0
        self.turn_log = []
        self._initialize_players()

    def _initialize_players(self):
        for p in range(self.num_players):
            sy, sx = int(self.start_positions[p, 0]), int(self.start_positions[p, 1])
            self.economy.initialize_city(sy, sx, p)
            self.capital_cities[p] = [sy, sx]
            self.economy.currency[p] = STARTING_CURRENCY

            candidates = np.array([sy, sx]) + DIRECTION_DELTAS[1:]
            valid = (
                (candidates[:, 0] >= 0) & (candidates[:, 0] < self.h) &
                (candidates[:, 1] >= 0) & (candidates[:, 1] < self.w)
            )
            valid_cands = candidates[valid]
            if len(valid_cands) > 0:
                terrain_at = self.terrain[valid_cands[:, 0], valid_cands[:, 1]]
                walkable = (
                    (terrain_at != TerrainType.WATER) &
                    (terrain_at != TerrainType.MOUNTAIN) &
                    (terrain_at != TerrainType.CITY)
                )
                walkable_cands = valid_cands[walkable]
                if len(walkable_cands) > 0:
                    self.units.spawn(p, UnitType.INFANTRY, int(walkable_cands[0, 0]), int(walkable_cands[0, 1]))

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self._build(self.seed)
        observations = {}
        action_masks = {}
        for p in range(self.num_players):
            observations[p] = self.get_observation(p)
            action_masks[p] = self.get_action_mask(p)
        return observations, {p: {"action_mask": action_masks[p]} for p in range(self.num_players)}

    def step(self, actions):
        self.turn += 1
        self.units.reset_turn()

        old_potential = self._compute_potential()
        rewards = np.zeros(self.num_players, dtype=np.float32)
        events = []

        unit_grid = self.units.build_unit_grid(self.h, self.w)
        self._process_spawns(actions, unit_grid, rewards, events)

        unit_grid = self.units.build_unit_grid(self.h, self.w)
        self._process_moves(actions, unit_grid, events)

        self._process_attacks(actions, unit_grid, rewards, events)

        unit_grid = self.units.build_unit_grid(self.h, self.w)
        self._check_city_capture(unit_grid, rewards, events)

        self._process_investments(actions, events)
        self._process_research(actions, events)

        income = self.economy.tick(self.terrain)

        self._process_ipd(actions, rewards, events)
        self._process_stag_hunt(actions, rewards, events)

        new_potential = self._compute_potential()
        pbrs = GAMMA * new_potential - old_potential
        rewards += pbrs

        terminal = self._check_terminal(rewards)

        observations = {}
        action_masks = {}
        for p in range(self.num_players):
            observations[p] = self.get_observation(p)
            action_masks[p] = self.get_action_mask(p)

        state = serialize_state(self, events)
        self.turn_log.append(state)

        infos = {}
        for p in range(self.num_players):
            infos[p] = {
                "action_mask": action_masks[p],
                "income": float(income[p]),
                "currency": float(self.economy.currency[p]),
                "alive": bool(self.player_alive[p]),
            }

        return observations, rewards, terminal, infos

    def get_observation(self, player_id):
        obs = np.full((self.h, self.w, NUM_OBS_CHANNELS), -1.0, dtype=np.float16)

        vis = compute_visibility(player_id, self.units, self.economy, self.terrain, self.h, self.w)

        obs[:, :, 6] = vis.astype(np.float16)
        obs[:, :, 0] = np.where(vis, self.terrain.astype(np.float16), -1.0)

        infrastructure = self.economy.infrastructure.astype(np.float16)
        obs[:, :, 1] = np.where(vis, infrastructure, -1.0)

        ownership = compute_ownership_grid(self.economy, self.h, self.w)
        obs[:, :, 2] = np.where(vis, ownership.astype(np.float16), -1.0)

        unit_grid = self.units.build_unit_grid(self.h, self.w)
        has_unit = unit_grid >= 0

        unit_present = has_unit.astype(np.float16)
        obs[:, :, 3] = np.where(vis, unit_present, -1.0)

        unit_types = np.full((self.h, self.w), -1.0, dtype=np.float16)
        if np.any(has_unit):
            ids = unit_grid[has_unit]
            unit_types[has_unit] = self.units.unit_type[ids].astype(np.float16)
        obs[:, :, 4] = np.where(vis, unit_types, -1.0)

        unit_health = np.full((self.h, self.w), -1.0, dtype=np.float16)
        if np.any(has_unit):
            ids = unit_grid[has_unit]
            unit_health[has_unit] = self.units.hp[ids] / self.units.max_hp[ids]
        obs[:, :, 5] = np.where(vis, unit_health, -1.0)

        return obs

    def get_action_mask(self, player_id):
        mask = np.zeros((self.h, self.w, NUM_ACTION_TYPES), dtype=np.bool_)
        mask[:, :, ActionType.NO_OP] = True

        if not self.player_alive[player_id]:
            return mask

        unit_grid = self.units.build_unit_grid(self.h, self.w)

        player_unit_mask = np.zeros((self.h, self.w), dtype=np.bool_)
        movement_grid = np.zeros((self.h, self.w), dtype=np.int32)
        range_grid = np.zeros((self.h, self.w), dtype=np.int32)
        attacked_grid = np.ones((self.h, self.w), dtype=np.bool_)
        utype_grid = np.full((self.h, self.w), -1, dtype=np.int32)

        p_alive = (self.units.owner == player_id) & self.units.alive
        if np.any(p_alive):
            ids = np.where(p_alive)[0]
            ys = self.units.positions[ids, 0]
            xs = self.units.positions[ids, 1]
            player_unit_mask[ys, xs] = True
            movement_grid[ys, xs] = self.units.movement_remaining[ids]
            range_grid[ys, xs] = self.units.attack_range[ids]
            attacked_grid[ys, xs] = self.units.has_attacked[ids]
            utype_grid[ys, xs] = self.units.unit_type[ids]

        can_move_base = player_unit_mask & ~attacked_grid & (movement_grid > 0)
        can_attack_base = player_unit_mask & ~attacked_grid

        enemy_present = np.zeros((self.h, self.w), dtype=np.bool_)
        all_alive_ids = np.where(self.units.alive)[0]
        if len(all_alive_ids) > 0:
            a_ys = self.units.positions[all_alive_ids, 0]
            a_xs = self.units.positions[all_alive_ids, 1]
            is_enemy = self.units.owner[all_alive_ids] != player_id
            enemy_ids = all_alive_ids[is_enemy]
            if len(enemy_ids) > 0:
                e_ys = self.units.positions[enemy_ids, 0]
                e_xs = self.units.positions[enemy_ids, 1]
                enemy_present[e_ys, e_xs] = True

        yy_grid = np.arange(self.h).reshape(-1, 1)
        xx_grid = np.arange(self.w).reshape(1, -1)

        has_sailing = bool(self.economy.tech_unlocked[player_id, TechType.SAILING])
        mountain_blocked_types = np.where(UNIT_MOUNTAIN_IMPASSABLE)[0]
        max_unit_range = int(np.max(UNIT_RANGE))

        for d_idx in range(8):
            dy = int(DIRECTION_DELTAS[d_idx + 1, 0])
            dx = int(DIRECTION_DELTAS[d_idx + 1, 1])

            ty = yy_grid + dy
            tx = xx_grid + dx
            in_bounds = (ty >= 0) & (ty < self.h) & (tx >= 0) & (tx < self.w)
            ty_c = np.clip(ty, 0, self.h - 1)
            tx_c = np.clip(tx, 0, self.w - 1)

            target_terrain = self.terrain[ty_c, tx_c]
            target_cost = TERRAIN_MOVE_COST[target_terrain]
            target_free = unit_grid[ty_c, tx_c] < 0

            can_afford = movement_grid >= target_cost
            water_ok = has_sailing | (target_terrain != TerrainType.WATER)
            mountain_ok = ~(
                np.isin(utype_grid, mountain_blocked_types) &
                (target_terrain == TerrainType.MOUNTAIN)
            )

            move_valid = can_move_base & in_bounds & target_free & can_afford & water_ok & mountain_ok
            mask[:, :, ActionType.MOVE_N + d_idx] = move_valid

            attack_valid = np.zeros((self.h, self.w), dtype=np.bool_)
            for r in range(1, max_unit_range + 1):
                rty = yy_grid + r * dy
                rtx = xx_grid + r * dx
                r_in_bounds = (rty >= 0) & (rty < self.h) & (rtx >= 0) & (rtx < self.w)
                rty_c = np.clip(rty, 0, self.h - 1)
                rtx_c = np.clip(rtx, 0, self.w - 1)
                has_enemy_at = enemy_present[rty_c, rtx_c] & r_in_bounds
                in_unit_range = range_grid >= r
                attack_valid |= (can_attack_base & has_enemy_at & in_unit_range)

            mask[:, :, ActionType.ATTACK_N + d_idx] = attack_valid

        player_cities = (self.terrain == TerrainType.CITY) & (self.economy.city_owner == player_id)
        mask[:, :, ActionType.INVEST] = player_cities

        city_empty = player_cities & (unit_grid < 0)
        for ut in range(NUM_UNIT_TYPES):
            can_afford_unit = self.economy.currency[player_id] >= UNIT_COST[ut]
            mask[:, :, ActionType.SPAWN_INFANTRY + ut] = city_empty & can_afford_unit

        for tech_id in range(NUM_TECHS):
            if not self.economy.tech_unlocked[player_id, tech_id]:
                can_afford_tech = self.economy.currency[player_id] >= TECH_COST[tech_id]
                mask[:, :, ActionType.RESEARCH_SAILING + tech_id] = player_cities & can_afford_tech

        ownership = compute_ownership_grid(self.economy, self.h, self.w)
        contacts = detect_border_contacts(ownership, self.num_players)
        has_contact = np.any(contacts[player_id])
        player_territory = ownership == player_id
        if has_contact:
            mask[:, :, ActionType.IPD_COOPERATE] = player_territory
            mask[:, :, ActionType.IPD_DEFECT] = player_territory

        for ci in range(len(self.resource_caches)):
            cy, cx = int(self.resource_caches[ci, 0]), int(self.resource_caches[ci, 1])
            if cy < 0:
                continue
            dist_to_cache = np.abs(yy_grid - cy) + np.abs(xx_grid - cx)
            near_cache = (dist_to_cache <= 2) & player_unit_mask
            mask[:, :, ActionType.STAG_ATTACK] |= near_cache
            mask[:, :, ActionType.STAG_RETREAT] |= near_cache

        return mask

    def _process_spawns(self, actions, unit_grid, rewards, events):
        for p in range(self.num_players):
            if not self.player_alive[p]:
                continue
            for ut in range(NUM_UNIT_TYPES):
                spawn_action = ActionType.SPAWN_INFANTRY + ut
                spawn_tiles = actions[p] == spawn_action
                if not np.any(spawn_tiles):
                    continue
                spawn_yx = np.argwhere(spawn_tiles)
                for i in range(len(spawn_yx)):
                    sy, sx = int(spawn_yx[i, 0]), int(spawn_yx[i, 1])
                    if (self.terrain[sy, sx] != TerrainType.CITY or
                            self.economy.city_owner[sy, sx] != p or
                            unit_grid[sy, sx] >= 0 or
                            self.economy.currency[p] < UNIT_COST[ut]):
                        continue
                    uid = self.units.spawn(p, ut, sy, sx)
                    if uid >= 0:
                        self.economy.currency[p] -= UNIT_COST[ut]
                        unit_grid[sy, sx] = uid
                        events.append({
                            "type": "spawn",
                            "player": p,
                            "unit": int(uid),
                            "unit_type": ut,
                            "pos": [sy, sx],
                        })

    def _process_moves(self, actions, unit_grid, events):
        for p in range(self.num_players):
            if not self.player_alive[p]:
                continue
            for d_idx in range(8):
                move_action = ActionType.MOVE_N + d_idx
                move_tiles = actions[p] == move_action
                if not np.any(move_tiles):
                    continue
                move_yx = np.argwhere(move_tiles)
                dy = int(DIRECTION_DELTAS[d_idx + 1, 0])
                dx = int(DIRECTION_DELTAS[d_idx + 1, 1])
                for i in range(len(move_yx)):
                    y, x = int(move_yx[i, 0]), int(move_yx[i, 1])
                    uid = unit_grid[y, x]
                    if uid < 0 or self.units.owner[uid] != p or not self.units.alive[uid]:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < self.h and 0 <= nx < self.w):
                        continue
                    if unit_grid[ny, nx] >= 0:
                        continue
                    terrain_cost = int(TERRAIN_MOVE_COST[self.terrain[ny, nx]])
                    if self.units.movement_remaining[uid] < terrain_cost:
                        continue
                    self.units.positions[uid] = [ny, nx]
                    self.units.movement_remaining[uid] -= terrain_cost
                    unit_grid[y, x] = -1
                    unit_grid[ny, nx] = uid
                    events.append({
                        "type": "move",
                        "unit": int(uid),
                        "player": p,
                        "from": [y, x],
                        "to": [ny, nx],
                    })

    def _process_attacks(self, actions, unit_grid, rewards, events):
        for p in range(self.num_players):
            if not self.player_alive[p]:
                continue
            for d_idx in range(8):
                attack_action = ActionType.ATTACK_N + d_idx
                attack_tiles = actions[p] == attack_action
                if not np.any(attack_tiles):
                    continue
                atk_positions = np.argwhere(attack_tiles)
                dy = int(DIRECTION_DELTAS[d_idx + 1, 0])
                dx = int(DIRECTION_DELTAS[d_idx + 1, 1])

                for i in range(len(atk_positions)):
                    ay, ax = int(atk_positions[i, 0]), int(atk_positions[i, 1])
                    uid = unit_grid[ay, ax]
                    if uid < 0 or self.units.owner[uid] != p:
                        continue
                    if self.units.has_attacked[uid] or not self.units.alive[uid]:
                        continue

                    utype = int(self.units.unit_type[uid])
                    atk_range = int(self.units.attack_range[uid])

                    target_uid = -1
                    target_y, target_x = -1, -1
                    for r in range(1, atk_range + 1):
                        ty, tx = ay + r * dy, ax + r * dx
                        if 0 <= ty < self.h and 0 <= tx < self.w:
                            t = unit_grid[ty, tx]
                            if t >= 0 and self.units.owner[t] != p:
                                target_uid = t
                                target_y, target_x = ty, tx
                                break

                    if target_uid < 0:
                        continue

                    target_type = int(self.units.unit_type[target_uid])

                    dmg, ret_dmg, killed = resolve_attack(
                        self.units, uid, target_uid, self.terrain
                    )
                    self.units.has_attacked[uid] = True

                    event = {
                        "type": "attack",
                        "attacker": int(uid),
                        "defender": int(target_uid),
                        "attacker_player": p,
                        "damage": float(dmg),
                        "retaliation": float(ret_dmg),
                        "killed": killed,
                        "attacker_pos": [ay, ax],
                        "defender_pos": [target_y, target_x],
                    }
                    events.append(event)

                    if killed:
                        unit_grid[target_y, target_x] = -1
                        rewards[p] += float(REWARD_DESTROY_UNIT_SCALE[target_type])

                        if UNIT_PERSIST_ON_KILL[utype] and self.units.alive[uid]:
                            self.units.has_attacked[uid] = False
                            self.units.positions[uid] = [target_y, target_x]
                            unit_grid[ay, ax] = -1
                            unit_grid[target_y, target_x] = uid
                            events.append({
                                "type": "persist_advance",
                                "unit": int(uid),
                                "from": [ay, ax],
                                "to": [target_y, target_x],
                            })

                    if not killed and UNIT_MOVE_AFTER_ATTACK[utype] and self.units.alive[uid]:
                        ry, rx = ay - dy, ax - dx
                        if 0 <= ry < self.h and 0 <= rx < self.w and unit_grid[ry, rx] < 0:
                            self.units.positions[uid] = [ry, rx]
                            unit_grid[ay, ax] = -1
                            unit_grid[ry, rx] = uid
                            events.append({
                                "type": "cavalry_retreat",
                                "unit": int(uid),
                                "from": [ay, ax],
                                "to": [ry, rx],
                            })

                    if not self.units.alive[uid]:
                        unit_grid[ay, ax] = -1

    def _check_city_capture(self, unit_grid, rewards, events):
        city_yx = np.argwhere(self.terrain == TerrainType.CITY)
        for i in range(len(city_yx)):
            cy, cx = int(city_yx[i, 0]), int(city_yx[i, 1])
            uid = unit_grid[cy, cx]
            if uid < 0:
                continue
            unit_owner = int(self.units.owner[uid])
            city_owner = int(self.economy.city_owner[cy, cx])
            if city_owner >= 0 and unit_owner != city_owner:
                self.economy.capture_city(cy, cx, unit_owner)
                rewards[unit_owner] += REWARD_CAPTURE_CITY
                rewards[city_owner] -= REWARD_CAPTURE_CITY
                events.append({
                    "type": "capture_city",
                    "pos": [cy, cx],
                    "old_owner": city_owner,
                    "new_owner": unit_owner,
                })

    def _process_investments(self, actions, events):
        for p in range(self.num_players):
            if not self.player_alive[p]:
                continue
            invest_tiles = actions[p] == ActionType.INVEST
            if not np.any(invest_tiles):
                continue
            invest_yx = np.argwhere(invest_tiles)
            for i in range(len(invest_yx)):
                y, x = int(invest_yx[i, 0]), int(invest_yx[i, 1])
                if (self.terrain[y, x] == TerrainType.CITY and
                        self.economy.city_owner[y, x] == p):
                    if self.economy.invest_in_city(p, y, x):
                        events.append({
                            "type": "invest",
                            "player": p,
                            "pos": [y, x],
                            "new_level": int(self.economy.infrastructure[y, x]),
                        })

    def _process_research(self, actions, events):
        for p in range(self.num_players):
            if not self.player_alive[p]:
                continue
            for tech_id in range(NUM_TECHS):
                action = ActionType.RESEARCH_SAILING + tech_id
                if np.any(actions[p] == action):
                    if self.economy.research_tech(p, tech_id):
                        events.append({
                            "type": "research",
                            "player": p,
                            "tech": tech_id,
                            "tech_name": TechType(tech_id).name,
                        })
                        break

    def _process_ipd(self, actions, rewards, events):
        ownership = compute_ownership_grid(self.economy, self.h, self.w)
        contacts = detect_border_contacts(ownership, self.num_players)

        for pa in range(self.num_players):
            for pb in range(pa + 1, self.num_players):
                if not contacts[pa, pb]:
                    continue
                if not self.player_alive[pa] or not self.player_alive[pb]:
                    continue

                pa_territory = ownership == pa
                pb_territory = ownership == pb

                def _get_ipd_choice(player_actions, territory):
                    defect = player_actions == ActionType.IPD_DEFECT
                    coop = player_actions == ActionType.IPD_COOPERATE
                    if np.any(defect & territory):
                        return 1
                    if np.any(coop & territory):
                        return 0
                    return 0

                choice_a = _get_ipd_choice(actions[pa], pa_territory)
                choice_b = _get_ipd_choice(actions[pb], pb_territory)

                ra, rb = resolve_ipd(choice_a, choice_b)
                rewards[pa] += ra
                rewards[pb] += rb
                events.append({
                    "type": "ipd",
                    "players": [pa, pb],
                    "choices": [choice_a, choice_b],
                    "rewards": [float(ra), float(rb)],
                })

    def _process_stag_hunt(self, actions, rewards, events):
        yy = np.arange(self.h).reshape(-1, 1)
        xx = np.arange(self.w).reshape(1, -1)

        for ci in range(len(self.resource_caches)):
            cy, cx = int(self.resource_caches[ci, 0]), int(self.resource_caches[ci, 1])
            if cy < 0:
                continue

            near = (np.abs(yy - cy) + np.abs(xx - cx)) <= 2
            participants = {}

            for p in range(self.num_players):
                if not self.player_alive[p]:
                    continue
                stag_atk = (actions[p] == ActionType.STAG_ATTACK) & near
                stag_ret = (actions[p] == ActionType.STAG_RETREAT) & near
                if np.any(stag_atk):
                    participants[p] = 0
                elif np.any(stag_ret):
                    participants[p] = 1

            if len(participants) >= 2:
                players = sorted(participants.keys())[:2]
                pa, pb = players[0], players[1]
                ca, cb = participants[pa], participants[pb]
                ra, rb = resolve_stag_hunt(ca, cb)
                rewards[pa] += ra
                rewards[pb] += rb
                events.append({
                    "type": "stag_hunt",
                    "cache": ci,
                    "players": [pa, pb],
                    "choices": [ca, cb],
                    "rewards": [float(ra), float(rb)],
                })
                if ca == 0 and cb == 0:
                    self.resource_caches[ci] = [-1, -1]

    def _check_terminal(self, rewards):
        capital_captured = False
        winner_by_capture = -1

        for p in range(self.num_players):
            if not self.player_alive[p]:
                continue
            cy, cx = self.capital_cities[p]
            owner = self.economy.city_owner[cy, cx]
            if owner != p and owner != -1:
                capital_captured = True
                winner_by_capture = owner
                self.player_alive[p] = False
                break

        alive_count = int(np.sum(self.player_alive))
        terminal = capital_captured or (alive_count <= 1) or (self.turn >= MAX_TURNS)

        if terminal:
            if capital_captured:
                for p in range(self.num_players):
                    if p == winner_by_capture:
                        rewards[p] += REWARD_WIN
                    else:
                        rewards[p] += REWARD_LOSS
            elif alive_count == 1:
                winner = int(np.where(self.player_alive)[0][0])
                for p in range(self.num_players):
                    if p == winner:
                        rewards[p] += REWARD_WIN
                    else:
                        rewards[p] += REWARD_LOSS
            else:
                scores = np.zeros(self.num_players)
                for p in range(self.num_players):
                    if self.player_alive[p]:
                        scores[p] = np.sum(self.economy.infrastructure[self.economy.city_owner == p])
                winner = int(np.argmax(scores))
                for p in range(self.num_players):
                    if p == winner and self.player_alive[p]:
                        rewards[p] += REWARD_WIN
                    else:
                        rewards[p] += REWARD_LOSS

        return terminal

    def _compute_potential(self):
        potential = np.zeros(self.num_players, dtype=np.float32)
        income = self.economy.compute_income(self.terrain)
        potential += REWARD_ECON_GROWTH * income
        for p in range(self.num_players):
            num_cities = self.economy.player_city_count(p)
            potential[p] += REWARD_CAPTURE_CITY * num_cities
            num_units = self.units.count_player_alive(p)
            potential[p] += REWARD_UNIT_SURVIVES * num_units
        return potential

    def save_log(self, filepath):
        save_turn_log(self.turn_log, filepath)
