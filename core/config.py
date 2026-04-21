import numpy as np
from enum import IntEnum


class TerrainType(IntEnum):
    PLAINS = 0
    FOREST = 1
    MOUNTAIN = 2
    WATER = 3
    CITY = 4


class UnitType(IntEnum):
    INFANTRY = 0
    CAVALRY = 1
    ARTILLERY = 2
    HEAVY = 3
    KNIGHT = 4


class ActionType(IntEnum):
    NO_OP = 0
    MOVE_N = 1
    MOVE_NE = 2
    MOVE_E = 3
    MOVE_SE = 4
    MOVE_S = 5
    MOVE_SW = 6
    MOVE_W = 7
    MOVE_NW = 8
    ATTACK_N = 9
    ATTACK_NE = 10
    ATTACK_E = 11
    ATTACK_SE = 12
    ATTACK_S = 13
    ATTACK_SW = 14
    ATTACK_W = 15
    ATTACK_NW = 16
    INVEST = 17
    SPAWN_INFANTRY = 18
    SPAWN_CAVALRY = 19
    SPAWN_ARTILLERY = 20
    SPAWN_HEAVY = 21
    SPAWN_KNIGHT = 22
    IPD_COOPERATE = 23
    IPD_DEFECT = 24
    STAG_ATTACK = 25
    STAG_RETREAT = 26
    RESEARCH_SAILING = 27
    RESEARCH_CONVERSION = 28
    RESEARCH_FORTIFICATION = 29
    RESEARCH_NAVIGATION = 30


class TechType(IntEnum):
    SAILING = 0
    CONVERSION = 1
    FORTIFICATION = 2
    NAVIGATION = 3


DIRECTION_DELTAS = np.array([
    [0, 0],
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
    [1, 0],
    [1, -1],
    [0, -1],
    [-1, -1],
], dtype=np.int32)

GRID_WIDTH = 24
GRID_HEIGHT = 24
NUM_PLAYERS = 4
MAX_UNITS = 128
MAX_RESOURCE_CACHES = 8
MAX_TURNS = 500

NUM_TERRAIN_TYPES = 5
NUM_UNIT_TYPES = 5
NUM_TECHS = 4
NUM_ACTION_TYPES = 31
NUM_OBS_CHANNELS = 7

GAMMA = 0.99

TERRAIN_MOVE_COST = np.array([1, 2, 2, 1, 1], dtype=np.int32)
TERRAIN_DEF_MOD = np.array([1.0, 1.5, 1.5, 0.8, 2.0], dtype=np.float32)
TERRAIN_VISION = np.array([1, 1, 2, 1, 2], dtype=np.int32)
TERRAIN_LOS_BLOCK = np.array([False, True, False, False, False])

UNIT_MAX_HP = np.array([15.0, 10.0, 5.0, 30.0, 12.0], dtype=np.float32)
UNIT_ATTACK_POWER = np.array([8.0, 8.0, 12.0, 10.0, 12.0], dtype=np.float32)
UNIT_MOVEMENT = np.array([2, 4, 1, 1, 3], dtype=np.int32)
UNIT_RANGE = np.array([1, 1, 3, 1, 1], dtype=np.int32)
UNIT_COST = np.array([5.0, 8.0, 10.0, 15.0, 12.0], dtype=np.float32)

UNIT_MOVE_AFTER_ATTACK = np.array([False, True, False, False, False])
UNIT_PERSIST_ON_KILL = np.array([False, False, False, False, True])
UNIT_NO_RETALIATION = np.array([False, False, True, False, False])
UNIT_MOUNTAIN_IMPASSABLE = np.array([False, False, True, True, False])

TECH_COST = np.array([10.0, 15.0, 8.0, 12.0], dtype=np.float32)

BASE_CITY_OUTPUT = 3.0
INFRASTRUCTURE_OUTPUT_BONUS = 1.5
INFRASTRUCTURE_COST_BASE = 5.0
POPULATION_PER_INFRASTRUCTURE = 2
CITY_TERRITORY_RADIUS = 2
STARTING_CURRENCY = 10.0

IPD_PAYOFF_A = np.array([[3.0, -5.0], [5.0, -1.0]], dtype=np.float32)
IPD_PAYOFF_B = np.array([[3.0, 5.0], [-5.0, -1.0]], dtype=np.float32)

STAG_PAYOFF_A = np.array([[10.0, -5.0], [3.0, 3.0]], dtype=np.float32)
STAG_PAYOFF_B = np.array([[10.0, 3.0], [-5.0, 3.0]], dtype=np.float32)

REWARD_DESTROY_UNIT_BASE = 0.2
REWARD_DESTROY_UNIT_SCALE = np.array([0.2, 0.3, 0.4, 0.5, 0.4], dtype=np.float32)
REWARD_CAPTURE_CITY = 1.0
REWARD_UNIT_SURVIVES = 0.1
REWARD_STAG_HUNT_COOP = 2.0
REWARD_WIN = 10.0
REWARD_LOSS = -10.0
REWARD_ECON_GROWTH = 0.05
