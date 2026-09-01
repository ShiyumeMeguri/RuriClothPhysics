import math

import numpy as np

from . import armature
from ..cloth_kernel import host_math

RADIAL_MODE = 'SPHERE_RADIAL'
SPHERE_MODES = {'SPHERE_DIRECTION', 'SPHERE_RADIAL'}
BOX_MODE = 'BOX_DIRECTION'

DEFAULT_DISPLAY_TYPE = 'SINGLE_ARROW'

DISPLAY_TYPE = {
    'GLOBAL_DIRECTION': 'SINGLE_ARROW',
    'SPHERE_DIRECTION': 'SPHERE',
    'SPHERE_RADIAL': 'SPHERE',
    BOX_MODE: 'CUBE',
}

ZONE_DISPLAY_REASON = (
    "the region a zone covers is drawn by the empty itself, because an empty already knows "
    "how to be a cube of its display size and a sphere of its display size and those are "
    "exactly the box and the sphere the solver tests against; the overlay draws only what "
    "the empty cannot say, which is where the wind blows inside that region and the six way "
    "burst a radial zone has instead of a direction; drawing the region a second time put "
    "two wireframes on the same coordinates, and neutralising the empty to stop them "
    "fighting left its crossed axes lying through the arrows instead, so the fix for the "
    "overlap was to draw less rather than to draw something else")

GLOBAL_ARROW_REASON = (
    "a global wind has no region and its direction is the empty's own forward axis whenever "
    "the two deflection angles are left at zero, which is what the built in single arrow "
    "already points along, so the overlay adds nothing to it; the arrow stops agreeing once "
    "those angles are set, and that is a known limit of letting the empty speak for itself "
    "rather than a case the overlay silently gets wrong somewhere else")


def sync_display(obj, wind):
    obj.empty_display_type = DISPLAY_TYPE.get(wind.mode, DEFAULT_DISPLAY_TYPE)


def zone_matrix(obj, depsgraph):
    source = obj.evaluated_get(depsgraph) if depsgraph is not None else obj
    return armature.read_matrix(source.matrix_world)


def zone_display_size(obj, depsgraph):
    source = obj.evaluated_get(depsgraph) if depsgraph is not None else obj
    return float(source.empty_display_size)


def local_extent(size, wind):
    if wind.mode == BOX_MODE:
        return np.full(3, size * 2.0, dtype=np.float32)
    return np.full(3, size, dtype=np.float32)


def local_box_half_extent(size, wind):
    return local_extent(size, wind) * 0.5


def local_sphere_radius(size, wind):
    return float(local_extent(size, wind)[0])


def local_reach(size, wind, direction, margin=0.0):
    magnitude = np.abs(np.asarray(direction, dtype=np.float64))
    margin = float(margin)
    if wind.mode == BOX_MODE:
        half = np.maximum(local_box_half_extent(size, wind).astype(np.float64) - margin, 0.0)
        limits = np.divide(half, magnitude, out=np.full(3, np.inf, dtype=np.float64),
                           where=magnitude > 1e-12)
        return float(np.min(limits))
    if wind.mode in SPHERE_MODES:
        radius = local_sphere_radius(size, wind)
        return float(np.sqrt(max(radius * radius - margin * margin, 0.0)))
    return float(size)


def zone_volume(size, wind, world_scale):
    if wind.mode == BOX_MODE:
        extent = size * 2.0 * np.asarray(world_scale, dtype=np.float64)
        return float(extent[0] * extent[1] * extent[2])
    if wind.mode in SPHERE_MODES:
        radius = size * float(world_scale[0])
        return (4.0 / 3.0) * radius * radius * radius * math.pi
    return float("inf")


def local_direction(wind):
    deflection = host_math.euler_yx(np.float32(wind.direction_angle_x),
                                    np.float32(wind.direction_angle_y))
    return host_math.quat_to_tangent(deflection[None])[0].astype(np.float32)


def world_direction(matrix_world, wind):
    rotation = armature.matrix_to_quat(matrix_world).astype(np.float32)
    direction = host_math.quat_rotate(rotation[None], local_direction(wind)[None])[0]
    length = float(np.linalg.norm(direction))
    if length > 1e-30:
        return (direction / length).astype(np.float32)
    return direction


