import numpy as np

from .. import defs
from .. import math as pm


def run(world, ctx):
    entries = ctx.entry_index("bending", world.bending)
    if len(entries) == 0:
        return
    pa = world.particles
    tt = world.team
    ba = world.bending
    team = ba["team"][entries]
    stiffness_param = tt["bending_stiffness"][team]
    powered = stiffness_param >= 1e-6
    if not np.any(powered):
        return
    entries = entries[powered]
    team = team[powered]
    stiffness = np.clip(stiffness_param[powered] * ctx.power[1], 0.0, 1.0)

    pairs = ba["pair"][entries]
    rest = ba["rest"][entries].copy()
    sign = ba["sign"][entries]
    p0, p1, p2, p3 = pairs[:, 0], pairs[:, 1], pairs[:, 2], pairs[:, 3]
    next0 = pa["next_positions"][p0]
    next1 = pa["next_positions"][p1]
    next2 = pa["next_positions"][p2]
    next3 = pa["next_positions"][p3]

    def _inv_mass(i):
        fix = ~pa["attr_move"][i]
        base = pm.calc_inverse_mass(pa["friction"][i], pa["depth"][i])
        return np.where(fix, 0.01, base)

    inv0, inv1, inv2, inv3 = _inv_mass(p0), _inv_mass(p1), _inv_mass(p2), _inv_mass(p3)

    scale_ratio = tt["scale_ratio"][team]
    negative_sign = tt["negative_scale_sign"][team]
    is_volume = sign == defs.VOLUME_SIGN

    add0 = np.zeros_like(next0)
    add1 = np.zeros_like(next1)
    add2 = np.zeros_like(next2)
    add3 = np.zeros_like(next3)
    result = np.zeros(len(entries), dtype=bool)

    if np.any(is_volume):
        idx = np.flatnonzero(is_volume)
        volume_rest = rest[idx] * scale_ratio[idx] * negative_sign[idx]
        n0, n1_, n2_, n3_ = next0[idx], next1[idx], next2[idx], next3[idx]
        volume = (1.0 / 6.0) * pm.dot(np.cross(n1_ - n0, n2_ - n0), n3_ - n0) * defs.VOLUME_SCALE
        grad0 = np.cross(n1_ - n2_, n3_ - n2_)
        grad1 = np.cross(n2_ - n0, n3_ - n0)
        grad2 = np.cross(n0 - n1_, n3_ - n1_)
        grad3 = np.cross(n1_ - n0, n2_ - n0)
        lam = (inv0[idx] * pm.length(grad0) ** 2 + inv1[idx] * pm.length(grad1) ** 2
               + inv2[idx] * pm.length(grad2) ** 2 + inv3[idx] * pm.length(grad3) ** 2)
        lam = lam * defs.VOLUME_SCALE
        ok = np.abs(lam) >= 1e-6
        lam = np.where(ok, stiffness[idx] * (volume_rest - volume) / np.where(ok, lam, 1.0), 0.0)
        add0[idx] = lam[:, None] * inv0[idx][:, None] * grad0
        add1[idx] = lam[:, None] * inv1[idx][:, None] * grad1
        add2[idx] = lam[:, None] * inv2[idx][:, None] * grad2
        add3[idx] = lam[:, None] * inv3[idx][:, None] * grad3
        result[idx] = ok

    if np.any(~is_volume):
        idx = np.flatnonzero(~is_volume)
        rest_angle = rest[idx] * sign[idx].astype(np.float32) * negative_sign[idx]
        n0, n1_, n2_, n3_ = next0[idx], next1[idx], next2[idx], next3[idx]

        e = n3_ - n2_
        elen = pm.length(e)
        ok = elen >= 1e-8
        inv_elen = 1.0 / np.where(elen > 1e-30, elen, 1.0)
        nn1 = np.cross(n2_ - n0, n3_ - n0)
        nn2 = np.cross(n3_ - n1_, n2_ - n1_)
        sq1 = pm.length(nn1) ** 2
        sq2 = pm.length(nn2) ** 2
        ok = ok & (sq1 != 0.0) & (sq2 != 0.0)
        nn1 = nn1 / np.where(sq1 > 1e-30, sq1, 1.0)[:, None]
        nn2 = nn2 / np.where(sq2 > 1e-30, sq2, 1.0)[:, None]

        d0 = nn1 * elen[:, None]
        d1 = nn2 * elen[:, None]
        d2 = pm.dot(n0 - n3_, e)[:, None] * inv_elen[:, None] * nn1 \
            + pm.dot(n1_ - n3_, e)[:, None] * inv_elen[:, None] * nn2
        d3 = pm.dot(n2_ - n0, e)[:, None] * inv_elen[:, None] * nn1 \
            + pm.dot(n2_ - n1_, e)[:, None] * inv_elen[:, None] * nn2

        un1 = pm.normalize(nn1)
        un2 = pm.normalize(nn2)
        dot = pm.clamp1(pm.dot(un1, un2))
        phi = np.arccos(dot)

        lam = (inv0[idx] * pm.length(d0) ** 2 + inv1[idx] * pm.length(d1) ** 2
               + inv2[idx] * pm.length(d2) ** 2 + inv3[idx] * pm.length(d3) ** 2)
        ok = ok & (lam != 0.0)

        dir_sign = np.sign(pm.dot(np.cross(un1, un2), e))
        phi = phi * dir_sign

        lam = np.where(ok, (rest_angle - phi) / np.where(lam != 0, lam, 1.0) * stiffness[idx], 0.0)
        add0[idx] = -inv0[idx][:, None] * lam[:, None] * d0
        add1[idx] = -inv1[idx][:, None] * lam[:, None] * d1
        add2[idx] = -inv2[idx][:, None] * lam[:, None] * d2
        add3[idx] = -inv3[idx][:, None] * lam[:, None] * d3
        result[idx] = ok

    idx = np.flatnonzero(result)
    if len(idx) == 0:
        return
    corner_targets = pairs[idx].reshape(-1)
    corner_values = np.stack((add0[idx], add1[idx], add2[idx], add3[idx]), axis=1).reshape(-1, 3).astype(np.float64)
    order = np.argsort(corner_targets, kind="stable")
    sorted_targets = corner_targets[order]
    sorted_values = corner_values[order]
    run_starts = np.flatnonzero(np.concatenate(([True], sorted_targets[1:] != sorted_targets[:-1])))
    run_counts = np.diff(np.concatenate((run_starts, [len(sorted_targets)])))
    touched = sorted_targets[run_starts]
    mean = (np.add.reduceat(sorted_values, run_starts, axis=0) / run_counts[:, None]).astype(np.float32)
    movable = pa["attr_move"][touched]
    apply = touched[movable]
    if len(apply):
        pa["next_positions"][apply] += mean[movable]
