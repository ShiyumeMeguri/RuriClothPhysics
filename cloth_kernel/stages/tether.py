import numpy as np

from .. import defs
from .. import math as pm


def run(world, ctx):
    entries = ctx.entry_index("tether", world.tether)
    if len(entries) == 0:
        return
    pa = world.particles
    tt = world.team
    index = world.tether["particle"][entries]
    team = pa["team"][index]

    compression_limit = 1.0 - tt["tether_compression"][team]
    stretch_limit = 1.0 + defs.TETHER_STRETCH_LIMIT

    root = pa["vertex_root"][index]
    next_pos = pa["next_positions"][index]
    root_pos = pa["next_positions"][root]
    v = root_pos - next_pos
    distance = pm.length(v)
    calc_pos = pa["step_basic_positions"][index]
    calc_root = pa["step_basic_positions"][root]
    calc_distance = pm.length(calc_pos - calc_root)
    valid = (distance >= defs.EPSILON) & (calc_distance != 0.0)
    ratio = distance / np.where(calc_distance != 0.0, calc_distance, 1.0)

    compress = valid & (ratio < compression_limit)
    stretch = valid & (ratio > stretch_limit)

    dist = np.where(compress, distance - compression_limit * calc_distance,
                    np.where(stretch, distance - stretch_limit * calc_distance, 0.0))
    t_c = pm.saturate((compression_limit - ratio) / defs.TETHER_STIFFNESS_WIDTH)
    t_s = pm.saturate((ratio - stretch_limit) / defs.TETHER_STIFFNESS_WIDTH)
    stiffness = np.where(compress, defs.TETHER_COMPRESSION_STIFFNESS * t_c,
                         np.where(stretch, defs.TETHER_STRETCH_STIFFNESS * t_s, 0.0))
    attenuation = np.where(compress, defs.TETHER_COMPRESSION_VELOCITY_ATTENUATION,
                           defs.TETHER_STRETCH_VELOCITY_ATTENUATION)

    active = compress | stretch
    add = (v / np.where(distance > 1e-30, distance, 1.0)[:, None]) * (dist * stiffness)[:, None]
    add = np.where(active[:, None], add, 0.0)
    pa["next_positions"][index] += add
    pa["velocity_positions"][index] += add * attenuation[:, None]
