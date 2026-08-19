import numpy as np

from ...cloth_kernel import defs
from ...cloth_kernel import math as pm


def run(world, ctx):
    entries = ctx.entry_index("distance", world.distance)
    if len(entries) == 0:
        return
    pa = world.particles
    tt = world.team
    da = world.distance
    p = da["particle"][entries]
    t = da["target"][entries]
    rest = da["rest"][entries]
    team = pa["team"][p]

    is_spring = tt["is_spring"][team]
    fix_mass = np.where(is_spring, defs.BONE_SPRING_FIX_MASS, defs.BONE_CLOTH_FIX_MASS)
    anime_ratio = tt["animation_pose_ratio"][team]
    scale = tt["init_scale"][team][:, 0] * tt["scale_ratio"][team]

    next_p = pa["next_positions"][p]
    next_t = pa["next_positions"][t]
    depth_p = pa["depth"][p]
    friction_p = pa["friction"][p]
    fixed_p = ~pa["attr_move"][p]
    fixed_t = ~pa["attr_move"][t]

    inv_mass_p = pm.calc_inverse_mass_fixed(friction_p, depth_p, fixed_p, fix_mass)
    inv_mass_t = pm.calc_inverse_mass_fixed(pa["friction"][t], pa["depth"][t], fixed_t, fix_mass)

    stiffness = pm.evaluate_team_lut_clamp01(tt["distance_lut"], team, depth_p)
    stiffness = stiffness * ctx.power[1]
    final_stiffness = pm.saturate(np.where(rest >= 0.0, stiffness,
                                           stiffness * defs.DISTANCE_HORIZONTAL_STIFFNESS))

    v = next_t - next_p
    zero_rest = rest == 0.0
    distance = pm.length(v)
    ok = zero_rest | (distance >= defs.EPSILON)

    base_p = pa["base_positions"][p]
    base_t = pa["base_positions"][t]
    rest_length = pm.lerp(np.abs(rest) * scale, pm.length(base_p - base_t), anime_ratio)
    n = v / np.where(distance > 1e-30, distance, 1.0)[:, None]
    corr = ((distance - rest_length) * final_stiffness)[:, None] * n \
        / (inv_mass_p + inv_mass_t)[:, None]
    corr = corr * inv_mass_p[:, None]
    corr = np.where(zero_rest[:, None], v * 0.5, corr)
    corr = np.where(ok[:, None], corr, 0.0)

    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return
    particle_run = p[idx]
    correction_run = corr[idx].astype(np.float64)
    run_starts = np.flatnonzero(np.concatenate(([True], particle_run[1:] != particle_run[:-1])))
    run_counts = np.diff(np.concatenate((run_starts, [len(particle_run)])))
    touched = particle_run[run_starts]
    mean = (np.add.reduceat(correction_run, run_starts, axis=0) / run_counts[:, None]).astype(np.float32)
    pa["next_positions"][touched] += mean
    pa["velocity_positions"][touched] += mean * defs.DISTANCE_VELOCITY_ATTENUATION
