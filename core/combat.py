import numpy as np
from .config import TERRAIN_DEF_MOD, UNIT_NO_RETALIATION


def compute_damage(attacker_attack, attacker_hp, attacker_max_hp, terrain_def_mod):
    attack_force = attacker_attack * (attacker_hp / attacker_max_hp)
    damage = attack_force / terrain_def_mod
    return float(damage)


def resolve_attack(units, attacker_id, defender_id, terrain_grid):
    if not units.alive[attacker_id] or not units.alive[defender_id]:
        return 0.0, 0.0, False

    def_y, def_x = int(units.positions[defender_id, 0]), int(units.positions[defender_id, 1])
    atk_y, atk_x = int(units.positions[attacker_id, 0]), int(units.positions[attacker_id, 1])

    def_terrain = terrain_grid[def_y, def_x]
    atk_terrain = terrain_grid[atk_y, atk_x]

    damage_to_defender = compute_damage(
        units.attack_power[attacker_id],
        units.hp[attacker_id],
        units.max_hp[attacker_id],
        TERRAIN_DEF_MOD[def_terrain]
    )

    units.hp[defender_id] -= damage_to_defender
    defender_killed = units.hp[defender_id] <= 0

    if defender_killed:
        units.kill(defender_id)
        return damage_to_defender, 0.0, True

    retaliation_damage = 0.0
    atk_type = int(units.unit_type[attacker_id])

    if not UNIT_NO_RETALIATION[atk_type]:
        dist = abs(atk_y - def_y) + abs(atk_x - def_x)
        defender_range = int(units.attack_range[defender_id])
        if dist <= defender_range:
            retaliation_damage = compute_damage(
                units.attack_power[defender_id],
                units.hp[defender_id],
                units.max_hp[defender_id],
                TERRAIN_DEF_MOD[atk_terrain]
            )
            units.hp[attacker_id] -= retaliation_damage
            if units.hp[attacker_id] <= 0:
                units.kill(attacker_id)

    return damage_to_defender, retaliation_damage, False
