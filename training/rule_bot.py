import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.config import (
    ActionType, UnitType, NUM_UNIT_TYPES,
    DIRECTION_DELTAS, UNIT_COST
)


class RuleBot:

    def __init__(self, player_id, engine):
        self.player_id = player_id
        self.engine = engine

    def get_action(self):
        h, w = self.engine.h, self.engine.w
        mask = self.engine.get_action_mask(self.player_id)
        actions = np.zeros((h, w), dtype=np.int32)
        assigned = np.zeros((h, w), dtype=np.bool_)

        self._assign_attacks(mask, actions, assigned)
        self._assign_moves(mask, actions, assigned, h, w)
        self._assign_spawns(mask, actions, assigned)
        self._assign_investments(mask, actions, assigned)

        return actions

    def _assign_attacks(self, mask, actions, assigned):
        for d in range(8):
            atk = ActionType.ATTACK_N + d
            can = mask[:, :, atk] & ~assigned
            if np.any(can):
                yx = np.argwhere(can)
                for i in range(len(yx)):
                    y, x = int(yx[i, 0]), int(yx[i, 1])
                    actions[y, x] = atk
                    assigned[y, x] = True

    def _assign_moves(self, mask, actions, assigned, h, w):
        center_y, center_x = h // 2, w // 2

        enemy_cities = np.argwhere(
            (self.engine.economy.city_owner >= 0) &
            (self.engine.economy.city_owner != self.player_id)
        )

        for d in range(8):
            move = ActionType.MOVE_N + d
            can = mask[:, :, move] & ~assigned
            if not np.any(can):
                continue
            dy = int(DIRECTION_DELTAS[d + 1, 0])
            dx = int(DIRECTION_DELTAS[d + 1, 1])
            yx = np.argwhere(can)
            for i in range(len(yx)):
                y, x = int(yx[i, 0]), int(yx[i, 1])
                ny, nx = y + dy, x + dx

                if len(enemy_cities) > 0:
                    dists_old = np.abs(enemy_cities[:, 0] - y) + np.abs(enemy_cities[:, 1] - x)
                    dists_new = np.abs(enemy_cities[:, 0] - ny) + np.abs(enemy_cities[:, 1] - nx)
                    if np.min(dists_new) < np.min(dists_old):
                        actions[y, x] = move
                        assigned[y, x] = True
                        continue

                old_dist = abs(y - center_y) + abs(x - center_x)
                new_dist = abs(ny - center_y) + abs(nx - center_x)
                if new_dist < old_dist:
                    actions[y, x] = move
                    assigned[y, x] = True

    def _assign_spawns(self, mask, actions, assigned):
        num_units = self.engine.units.count_player_alive(self.player_id)
        max_desired = 6

        if num_units >= max_desired:
            return

        priority = [UnitType.INFANTRY, UnitType.CAVALRY, UnitType.ARTILLERY]
        for ut in priority:
            spawn_action = ActionType.SPAWN_INFANTRY + ut
            can = mask[:, :, spawn_action] & ~assigned
            if np.any(can):
                yx = np.argwhere(can)
                y, x = int(yx[0, 0]), int(yx[0, 1])
                actions[y, x] = spawn_action
                assigned[y, x] = True
                return

    def _assign_investments(self, mask, actions, assigned):
        currency = float(self.engine.economy.currency[self.player_id])
        if currency < 15:
            return

        can = mask[:, :, ActionType.INVEST] & ~assigned
        if np.any(can):
            yx = np.argwhere(can)
            y, x = int(yx[0, 0]), int(yx[0, 1])
            actions[y, x] = ActionType.INVEST
            assigned[y, x] = True
