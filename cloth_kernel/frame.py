import numpy as np

SCALE_EPSILON = 1e-6


def frame_team_mask(world):
    scale = world.team["component_world_scale"]
    scale_alive = np.abs(scale).min(axis=1) >= SCALE_EPSILON
    return world.team["enabled"] & world.team["valid"] & scale_alive


def frame_team_index(world):
    return np.flatnonzero(frame_team_mask(world))


def has_frame_teams(world):
    return bool(frame_team_mask(world).any())
