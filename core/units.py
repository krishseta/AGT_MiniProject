import numpy as np
from .config import (
    MAX_UNITS, UNIT_MAX_HP, UNIT_ATTACK_POWER,
    UNIT_MOVEMENT, UNIT_RANGE
)


class UnitManager:

    def __init__(self, max_units=MAX_UNITS):
        self.max_units = max_units
        self.positions = np.full((max_units, 2), -1, dtype=np.int32)
        self.owner = np.full(max_units, -1, dtype=np.int32)
        self.unit_type = np.zeros(max_units, dtype=np.int32)
        self.hp = np.zeros(max_units, dtype=np.float32)
        self.max_hp = np.zeros(max_units, dtype=np.float32)
        self.attack_power = np.zeros(max_units, dtype=np.float32)
        self.movement_budget = np.zeros(max_units, dtype=np.int32)
        self.movement_remaining = np.zeros(max_units, dtype=np.int32)
        self.attack_range = np.zeros(max_units, dtype=np.int32)
        self.alive = np.zeros(max_units, dtype=np.bool_)
        self.has_attacked = np.zeros(max_units, dtype=np.bool_)
        self.next_id = 0

    def spawn(self, owner, unit_type, y, x):
        uid = -1
        if self.next_id < self.max_units:
            uid = self.next_id
            self.next_id += 1
        else:
            dead_slots = np.where(~self.alive)[0]
            if len(dead_slots) == 0:
                return -1
            uid = int(dead_slots[0])

        self.positions[uid] = [y, x]
        self.owner[uid] = owner
        self.unit_type[uid] = unit_type
        self.hp[uid] = UNIT_MAX_HP[unit_type]
        self.max_hp[uid] = UNIT_MAX_HP[unit_type]
        self.attack_power[uid] = UNIT_ATTACK_POWER[unit_type]
        self.movement_budget[uid] = UNIT_MOVEMENT[unit_type]
        self.movement_remaining[uid] = UNIT_MOVEMENT[unit_type]
        self.attack_range[uid] = UNIT_RANGE[unit_type]
        self.alive[uid] = True
        self.has_attacked[uid] = False
        return uid

    def kill(self, uid):
        self.alive[uid] = False
        self.positions[uid] = [-1, -1]
        self.owner[uid] = -1
        self.hp[uid] = 0.0

    def build_unit_grid(self, h, w):
        grid = np.full((h, w), -1, dtype=np.int32)
        alive_ids = np.where(self.alive)[0]
        if len(alive_ids) > 0:
            ys = self.positions[alive_ids, 0]
            xs = self.positions[alive_ids, 1]
            valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
            v_ids = alive_ids[valid]
            grid[self.positions[v_ids, 0], self.positions[v_ids, 1]] = v_ids
        return grid

    def get_player_units(self, player_id):
        return np.where((self.owner == player_id) & self.alive)[0]

    def reset_turn(self):
        alive = self.alive
        self.movement_remaining[alive] = self.movement_budget[alive]
        self.has_attacked[alive] = False

    def count_alive(self):
        return int(np.sum(self.alive))

    def count_player_alive(self, player_id):
        return int(np.sum((self.owner == player_id) & self.alive))
