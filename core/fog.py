import numpy as np
from .config import TERRAIN_VISION, TerrainType


def compute_visibility(player_id, units, economy, terrain_grid, h, w):
    visibility = np.zeros((h, w), dtype=np.bool_)

    yy = np.arange(h).reshape(1, -1, 1)
    xx = np.arange(w).reshape(1, 1, -1)

    player_units = (units.owner == player_id) & units.alive
    if np.any(player_units):
        u_ids = np.where(player_units)[0]
        u_y = units.positions[u_ids, 0]
        u_x = units.positions[u_ids, 1]
        terrain_at_units = terrain_grid[u_y, u_x]
        u_vision = TERRAIN_VISION[terrain_at_units]

        src_y = u_y.reshape(-1, 1, 1)
        src_x = u_x.reshape(-1, 1, 1)
        vr = u_vision.reshape(-1, 1, 1)

        dist = np.abs(yy - src_y) + np.abs(xx - src_x)
        visible = dist <= vr
        visibility |= np.any(visible, axis=0)

    player_cities = economy.city_owner == player_id
    if np.any(player_cities):
        city_pos = np.argwhere(player_cities)
        c_y = city_pos[:, 0].reshape(-1, 1, 1)
        c_x = city_pos[:, 1].reshape(-1, 1, 1)
        c_vision = int(TERRAIN_VISION[TerrainType.CITY])

        dist = np.abs(yy - c_y) + np.abs(xx - c_x)
        visible = dist <= c_vision
        visibility |= np.any(visible, axis=0)

    return visibility
