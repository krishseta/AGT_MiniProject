import numpy as np
from .config import (
    NUM_PLAYERS, GRID_HEIGHT, GRID_WIDTH, NUM_TECHS,
    BASE_CITY_OUTPUT, INFRASTRUCTURE_OUTPUT_BONUS,
    INFRASTRUCTURE_COST_BASE, POPULATION_PER_INFRASTRUCTURE,
    TECH_COST
)


class EconomyManager:

    def __init__(self, num_players=NUM_PLAYERS, h=GRID_HEIGHT, w=GRID_WIDTH):
        self.num_players = num_players
        self.currency = np.zeros(num_players, dtype=np.float32)
        self.tech_unlocked = np.zeros((num_players, NUM_TECHS), dtype=np.bool_)
        self.infrastructure = np.zeros((h, w), dtype=np.int32)
        self.city_owner = np.full((h, w), -1, dtype=np.int32)
        self.population = np.zeros(num_players, dtype=np.int32)

    def initialize_city(self, y, x, owner):
        self.city_owner[y, x] = owner
        self.infrastructure[y, x] = 1
        self.population[owner] += POPULATION_PER_INFRASTRUCTURE

    def compute_income(self, terrain_grid):
        income = np.zeros(self.num_players, dtype=np.float32)
        city_mask = self.city_owner >= 0
        if not np.any(city_mask):
            return income
        owners = self.city_owner[city_mask]
        outputs = (BASE_CITY_OUTPUT +
                   self.infrastructure[city_mask].astype(np.float32) *
                   INFRASTRUCTURE_OUTPUT_BONUS)
        np.add.at(income, owners, outputs)
        return income

    def invest_in_city(self, player, y, x):
        if self.city_owner[y, x] != player:
            return False
        cost = INFRASTRUCTURE_COST_BASE * (self.infrastructure[y, x] + 1)
        if self.currency[player] >= cost:
            self.currency[player] -= cost
            self.infrastructure[y, x] += 1
            self.population[player] += POPULATION_PER_INFRASTRUCTURE
            return True
        return False

    def research_tech(self, player, tech_id):
        if tech_id < 0 or tech_id >= NUM_TECHS:
            return False
        if self.tech_unlocked[player, tech_id]:
            return False
        if self.currency[player] >= TECH_COST[tech_id]:
            self.currency[player] -= TECH_COST[tech_id]
            self.tech_unlocked[player, tech_id] = True
            return True
        return False

    def tick(self, terrain_grid):
        income = self.compute_income(terrain_grid)
        self.currency += income
        return income

    def capture_city(self, y, x, new_owner):
        old_owner = int(self.city_owner[y, x])
        infra = int(self.infrastructure[y, x])
        if old_owner >= 0 and old_owner != new_owner:
            pop_loss = POPULATION_PER_INFRASTRUCTURE * infra
            self.population[old_owner] = max(0, self.population[old_owner] - pop_loss)
        self.city_owner[y, x] = new_owner
        self.population[new_owner] += POPULATION_PER_INFRASTRUCTURE * infra

    def player_city_count(self, player):
        return int(np.sum(self.city_owner == player))
