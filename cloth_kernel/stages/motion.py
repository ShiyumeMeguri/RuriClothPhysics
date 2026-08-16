import numpy as np

from .. import defs
from .. import math as pm


def run(world, ctx):
    entries = ctx.entry_index("motion", world.motion)
    if len(entries) == 0:
        return
    pa = world.particles
    tt = world.team
    index = world.motion["particle"][entries]
    team = pa["team"][index]

    enabled = tt["motion_use_max_distance"][team] | tt["motion_use_backstop"][team]
    if not np.any(enabled):
        return
    index = index[enabled]
    team = team[enabled]

    stiffness = tt["motion_stiffness"][team]
    backstop_radius = tt["motion_backstop_radius"][team]

    next_pos = pa["next_positions"][index]
    base_pos = pa["base_positions"][index]
    base_rot = pa["base_rotations"][index]
    depth = pa["depth"][index]
    original = next_pos.copy()

    radius = np.maximum(pm.evaluate_team_lut(tt["radius_lut"], team, depth), 0.0001)
    cfr = radius * 1.0
    depth2 = depth * depth

    direction = pm.quat_rotate(base_rot, tt["normal_axis_vector"][team])

    use_max = tt["motion_use_max_distance"][team]
    if np.any(use_max):
        max_distance = pm.evaluate_team_lut(tt["motion_max_distance_lut"], team, depth2)
        v = pm.clamp_vector(next_pos - base_pos, max_distance)
        next_pos = np.where(use_max[:, None], base_pos + v, next_pos)

    use_backstop = tt["motion_use_backstop"][team] & (backstop_radius > 0.0)
    if np.any(use_backstop):
        backstop_distance = pm.evaluate_team_lut(tt["motion_backstop_lut"], team, depth2)
        center = base_pos - direction * (backstop_distance + backstop_radius)[:, None]
        v = next_pos - center
        l = pm.length(v)
        near = (l > defs.EPSILON) & (l < backstop_radius + cfr)
        inside = use_backstop & near & (l < backstop_radius)
        n = v / np.where(l > 1e-30, l, 1.0)[:, None]
        pushed = center + n * backstop_radius[:, None]
        next_pos = np.where(inside[:, None], pushed, next_pos)

    next_pos = pm.lerp(original, next_pos, stiffness[:, None])
    pa["next_positions"][index] = next_pos
    add = next_pos - original
    pa["velocity_positions"][index] += add * 0.95
