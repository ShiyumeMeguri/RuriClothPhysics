import math

import numpy as np

from . import armature
from ..cloth_kernel import defs
from ..cloth_kernel import math as pm

RADIAL_MODE = 'SPHERE_RADIAL'
SPHERE_MODES = {'SPHERE_DIRECTION', 'SPHERE_RADIAL'}
BOX_MODE = 'BOX_DIRECTION'

DISPLAY_TYPE = {
    'GLOBAL_DIRECTION': 'SINGLE_ARROW',
    'SPHERE_DIRECTION': 'SPHERE',
    'SPHERE_RADIAL': 'SPHERE',
    BOX_MODE: 'CUBE',
}

TURBULENCE_Y_RATIO = 0.5


def sync_display(obj, wind):
    obj.empty_display_type = DISPLAY_TYPE.get(wind.mode, 'SINGLE_ARROW')


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


def zone_volume(size, wind, world_scale):
    if wind.mode == BOX_MODE:
        extent = size * 2.0 * np.asarray(world_scale, dtype=np.float64)
        return float(extent[0] * extent[1] * extent[2])
    if wind.mode in SPHERE_MODES:
        radius = size * float(world_scale[0])
        return (4.0 / 3.0) * radius * radius * radius * math.pi
    return float("inf")


def local_direction(wind):
    deflection = pm.euler_yx(np.float32(wind.direction_angle_x),
                             np.float32(wind.direction_angle_y))
    return pm.quat_to_tangent(deflection[None])[0].astype(np.float32)


def world_direction(matrix_world, wind):
    rotation = armature.matrix_to_quat(matrix_world).astype(np.float32)
    direction = pm.quat_rotate(rotation[None], local_direction(wind)[None])[0]
    length = float(np.linalg.norm(direction))
    if length > 1e-30:
        return (direction / length).astype(np.float32)
    return direction


def turbulence_spread(matrix_world, wind, offsets):
    base = world_direction(matrix_world, wind)
    dirq = pm.axis_quaternion(base[None].astype(np.float32))
    limit_x = math.radians(defs.WIND_TURBULENCE_ANGLE) * float(wind.turbulence)
    limit_y = limit_x * TURBULENCE_Y_RATIO
    out = []
    for unit_x, unit_y in offsets:
        rq = pm.euler_yx(np.float32(unit_x * limit_x), np.float32(unit_y * limit_y))
        combined = pm.quat_mul(dirq, rq[None])
        out.append(pm.quat_to_tangent(combined)[0].astype(np.float64))
    return base.astype(np.float64), out
