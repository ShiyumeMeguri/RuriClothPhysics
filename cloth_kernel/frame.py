import numpy as np

from . import defs


def component_scale_is_alive(component_scale):
    return np.asarray(component_scale).min(axis=-1) >= defs.COMPONENT_SCALE_EPSILON


def frame_team_mask(world):
    scale_alive = component_scale_is_alive(world.team["component_world_scale"])
    return world.team["enabled"] & world.team["valid"] & scale_alive


def frame_team_index(world):
    return np.flatnonzero(frame_team_mask(world))


def has_frame_teams(world):
    return bool(frame_team_mask(world).any())
