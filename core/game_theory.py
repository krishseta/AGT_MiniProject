import numpy as np
from .config import (
    IPD_PAYOFF_A, IPD_PAYOFF_B,
    STAG_PAYOFF_A, STAG_PAYOFF_B,
    CITY_TERRITORY_RADIUS
)


def detect_border_contacts(ownership_grid, num_players):
    contact = np.zeros((num_players, num_players), dtype=np.bool_)

    left = ownership_grid[:, :-1]
    right = ownership_grid[:, 1:]
    hmask = (left >= 0) & (right >= 0) & (left != right)
    if np.any(hmask):
        p1 = left[hmask]
        p2 = right[hmask]
        contact[p1, p2] = True
        contact[p2, p1] = True

    top = ownership_grid[:-1, :]
    bottom = ownership_grid[1:, :]
    vmask = (top >= 0) & (bottom >= 0) & (top != bottom)
    if np.any(vmask):
        p1 = top[vmask]
        p2 = bottom[vmask]
        contact[p1, p2] = True
        contact[p2, p1] = True

    return contact


def resolve_ipd(choice_a, choice_b):
    reward_a = float(IPD_PAYOFF_A[choice_a, choice_b])
    reward_b = float(IPD_PAYOFF_B[choice_a, choice_b])
    return reward_a, reward_b


def resolve_stag_hunt(choice_a, choice_b):
    reward_a = float(STAG_PAYOFF_A[choice_a, choice_b])
    reward_b = float(STAG_PAYOFF_B[choice_a, choice_b])
    return reward_a, reward_b


def compute_ownership_grid(economy, h, w):
    ownership = np.full((h, w), -1, dtype=np.int32)

    city_mask = economy.city_owner >= 0
    if not np.any(city_mask):
        return ownership

    city_positions = np.argwhere(city_mask)
    city_owners = economy.city_owner[city_mask]
    infra = economy.infrastructure[city_mask]

    sort_idx = np.argsort(infra)
    city_positions = city_positions[sort_idx]
    city_owners = city_owners[sort_idx]
    infra = infra[sort_idx]

    cy = city_positions[:, 0].reshape(-1, 1, 1)
    cx = city_positions[:, 1].reshape(-1, 1, 1)
    yy = np.arange(h).reshape(1, -1, 1)
    xx = np.arange(w).reshape(1, 1, -1)

    dists = np.abs(yy - cy) + np.abs(xx - cx)
    radii = (CITY_TERRITORY_RADIUS + infra).reshape(-1, 1, 1)

    in_range = dists <= radii
    any_claim = np.any(in_range, axis=0)

    dists_masked = np.where(in_range, dists, np.iinfo(np.int32).max)
    nearest_city_idx = np.argmin(dists_masked, axis=0)

    ownership = np.where(any_claim, city_owners[nearest_city_idx], -1)

    return ownership
