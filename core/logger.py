import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def serialize_state(engine, events):
    alive_ids = np.where(engine.units.alive)[0]

    units_data = []
    for uid in alive_ids:
        units_data.append({
            "id": int(uid),
            "y": int(engine.units.positions[uid, 0]),
            "x": int(engine.units.positions[uid, 1]),
            "owner": int(engine.units.owner[uid]),
            "type": int(engine.units.unit_type[uid]),
            "hp": float(engine.units.hp[uid]),
            "max_hp": float(engine.units.max_hp[uid]),
        })

    return {
        "turn": int(engine.turn),
        "terrain": engine.terrain.tolist(),
        "units": units_data,
        "economy": {
            "currency": engine.economy.currency.tolist(),
            "tech_unlocked": engine.economy.tech_unlocked.tolist(),
            "city_owner": engine.economy.city_owner.tolist(),
            "infrastructure": engine.economy.infrastructure.tolist(),
            "population": engine.economy.population.tolist(),
        },
        "player_alive": engine.player_alive.tolist(),
        "resource_caches": engine.resource_caches.tolist(),
        "events": events,
    }


def save_turn_log(log, filepath):
    with open(filepath, 'w') as f:
        json.dump(log, f, cls=NumpyEncoder, indent=2)
