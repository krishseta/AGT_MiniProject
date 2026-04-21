import numpy as np
from .config import TerrainType, MAX_RESOURCE_CACHES


def _bilinear_upsample(coarse, target_h, target_w):
    ch, cw = coarse.shape
    yy, xx = np.meshgrid(
        np.linspace(0, ch - 1, target_h),
        np.linspace(0, cw - 1, target_w),
        indexing='ij'
    )
    y0 = np.floor(yy).astype(np.int32)
    x0 = np.floor(xx).astype(np.int32)
    y1 = np.minimum(y0 + 1, ch - 1)
    x1 = np.minimum(x0 + 1, cw - 1)
    fy = (yy - y0.astype(np.float32))
    fx = (xx - x0.astype(np.float32))
    return (coarse[y0, x0] * (1 - fy) * (1 - fx) +
            coarse[y1, x0] * fy * (1 - fx) +
            coarse[y0, x1] * (1 - fy) * fx +
            coarse[y1, x1] * fy * fx)


def _generate_noise(rng, h, w, octaves=3):
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    for octave in range(octaves):
        scale = 2 ** (octave + 2)
        ch = max(2, h // scale)
        cw = max(2, w // scale)
        coarse = rng.random((ch, cw), dtype=np.float32)
        layer = _bilinear_upsample(coarse, h, w)
        noise += layer * amplitude
        amplitude *= 0.5
    lo, hi = noise.min(), noise.max()
    noise = (noise - lo) / (hi - lo + 1e-8)
    return noise


def _place_diametrical_starts(rng, h, w, num_players, terrain):
    margin = max(2, min(h, w) // 8)
    starts = []
    corners = [
        [margin, margin],
        [h - 1 - margin, w - 1 - margin],
        [margin, w - 1 - margin],
        [h - 1 - margin, margin]
    ]
    for i in range(min(num_players, len(corners))):
        starts.append(corners[i])
    return np.array(starts, dtype=np.int32)


def generate_terrain(h, w, seed, num_players, num_caches=MAX_RESOURCE_CACHES):
    rng = np.random.default_rng(seed)

    height_map = _generate_noise(rng, h, w)
    moisture_map = _generate_noise(rng, h, w)

    terrain = np.full((h, w), TerrainType.PLAINS, dtype=np.int32)
    terrain[height_map > 0.75] = TerrainType.MOUNTAIN
    terrain[(height_map > 0.55) & (height_map <= 0.75) & (moisture_map > 0.5)] = TerrainType.FOREST
    terrain[height_map < 0.25] = TerrainType.WATER

    starts = _place_diametrical_starts(rng, h, w, num_players, terrain)

    starts_y = starts[:, 0].reshape(-1, 1, 1)
    starts_x = starts[:, 1].reshape(-1, 1, 1)
    yy = np.arange(h).reshape(1, -1, 1)
    xx = np.arange(w).reshape(1, 1, -1)
    near_any_start = np.any(
        (np.abs(yy - starts_y) <= 1) & (np.abs(xx - starts_x) <= 1),
        axis=0
    )
    obstacle_near = near_any_start & (
        (terrain == TerrainType.WATER) | (terrain == TerrainType.MOUNTAIN)
    )
    terrain[obstacle_near] = TerrainType.PLAINS

    terrain[starts[:, 0], starts[:, 1]] = TerrainType.CITY

    all_dists = np.abs(yy - starts_y) + np.abs(xx - starts_x)
    min_dist_to_start = np.min(all_dists, axis=0)
    cache_valid = (terrain == TerrainType.PLAINS) & (min_dist_to_start > 5)

    cache_positions = np.full((num_caches, 2), -1, dtype=np.int32)
    # Ensure massive 2x2 stag hunt cache in the exact center
    if num_caches >= 4:
        ch, cw = h // 2, w // 2
        cache_positions[0] = [ch - 1, cw - 1]
        cache_positions[1] = [ch - 1, cw]
        cache_positions[2] = [ch, cw - 1]
        cache_positions[3] = [ch, cw]
        
    valid_for_cache = np.argwhere(cache_valid)
    if len(valid_for_cache) > 0 and num_caches > 4:
        n = min(num_caches - 4, len(valid_for_cache))
        indices = rng.choice(len(valid_for_cache), size=n, replace=False)
        cache_positions[4:4+n] = valid_for_cache[indices]

    return terrain, starts, cache_positions
