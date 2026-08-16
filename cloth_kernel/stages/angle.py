import numpy as np

from .. import defs
from .. import math as pm


class PreparedPass:
    __slots__ = ("v", "p", "use_limit", "use_restoration", "any_limit", "any_restoration",
                 "c_inv_mass", "p_inv_mass", "p_move", "limit_stiffness", "max_angle",
                 "restoration_stiffness", "restoration_attenuation")


def run(world, ctx):
    pa = world.particles
    tt = world.team
    need = ctx.step_team_mask & (tt["angle_use_limit"] | tt["angle_use_restoration"])
    if not np.any(need):
        return

    buffered = ctx.entry_index("angle_buffered", world.angle_buffered)
    entries = world.baseline_entries
    if len(entries):
        active_entries = entries[need[pa["team"][entries]]]
        pa["albuf_rotation"][active_entries] = pa["step_basic_rotations"][active_entries]

    if len(buffered):
        v = world.angle_buffered["particle"][buffered]
        vt = pa["team"][v]
        active = need[vt]
        v = v[active]
        vt = pa["team"][v]
        if len(v):
            p = pa["vertex_parent"][v]
            npos = pa["next_positions"][v]
            pnpos = pa["next_positions"][p]
            bpos = pa["step_basic_positions"][v]
            pbpos = pa["step_basic_positions"][p]
            brot = pa["step_basic_rotations"][v]
            pbrot = pa["step_basic_rotations"][p]

            use_limit = tt["angle_use_limit"][vt]
            if np.any(use_limit):
                vlen = pm.length(npos - pnpos)
                bv = bpos - pbpos
                bvlen = pm.length(bv)
                degenerate = (vlen < defs.EPSILON) | (bvlen < defs.EPSILON)
                direction = bv / np.where(bvlen > 1e-30, bvlen, 1.0)[:, None]
                ipq = pm.quat_inverse(pbrot)
                lpos = pm.quat_rotate(ipq, direction)
                lrot = pm.quat_mul(ipq, brot)
                identity = np.broadcast_to(np.array([0, 0, 0, 1], dtype=np.float32), lrot.shape)
                length_value = np.where(degenerate, 0.0, vlen)
                lpos_value = np.where(degenerate[:, None], 0.0, lpos)
                lrot_value = np.where(degenerate[:, None], identity, lrot)
                mask = use_limit
                pa["albuf_length"][v] = np.where(mask, length_value, pa["albuf_length"][v])
                pa["albuf_local_pos"][v] = np.where(mask[:, None], lpos_value,
                                                    pa["albuf_local_pos"][v])
                pa["albuf_local_rot"][v] = np.where(mask[:, None], lrot_value,
                                                    pa["albuf_local_rot"][v])

            use_restoration = tt["angle_use_restoration"][vt]
            pa["albuf_restore"][v] = np.where(use_restoration[:, None], bpos - pbpos,
                                              pa["albuf_restore"][v])

    prepared = []
    for vertices_all, parents_all in world.angle_passes:
        bucket_active = need[pa["team"][vertices_all]]
        if not np.any(bucket_active):
            continue
        v = vertices_all[bucket_active]
        p = parents_all[bucket_active]
        vt = pa["team"][v]
        cdepth = pa["depth"][v]
        use_limit = tt["angle_use_limit"][vt]
        use_restoration = tt["angle_use_restoration"][vt]
        gravity_falloff = pm.lerp(1.0 - tt["angle_restoration_gravity_falloff"][vt], 1.0,
                                  tt["gravity_dot"][vt])
        restoration_stiffness = pm.evaluate_team_lut_clamp01(tt["angle_restoration_lut"], vt, cdepth)
        restoration_stiffness = pm.saturate(restoration_stiffness * ctx.power[3])
        entry = PreparedPass()
        entry.v = v
        entry.p = p
        entry.use_limit = use_limit
        entry.use_restoration = use_restoration
        entry.any_limit = bool(np.any(use_limit))
        entry.any_restoration = bool(np.any(use_restoration))
        entry.c_inv_mass = 1.0 / (1.0 + pa["friction"][v] * defs.FRICTION_MASS)
        entry.p_inv_mass = 1.0 / (1.0 + pa["friction"][p] * defs.FRICTION_MASS)
        entry.p_move = pa["attr_move"][p]
        entry.limit_stiffness = tt["angle_limit_stiffness"][vt]
        entry.max_angle = np.radians(pm.evaluate_team_lut(tt["angle_limit_lut"], vt, cdepth))
        entry.restoration_stiffness = restoration_stiffness * gravity_falloff
        entry.restoration_attenuation = tt["angle_restoration_attenuation"][vt]
        prepared.append(entry)

    iterations = defs.ANGLE_LIMIT_ITERATION
    limit_rot_ratio = defs.ANGLE_LIMIT_ROTATION_RATIO
    attenuation = defs.ANGLE_LIMIT_ATTENUATION
    for k in range(iterations):
        iteration_ratio = k / (iterations - 1)
        restoration_rot_ratio = np.float32(0.1 + (0.5 - 0.1) * iteration_ratio)

        for entry in prepared:
            v = entry.v
            p = entry.p
            c_inv_mass = entry.c_inv_mass
            p_inv_mass = entry.p_inv_mass
            p_move = entry.p_move

            if entry.any_limit:
                use_limit = entry.use_limit
                limit_stiffness = entry.limit_stiffness
                max_angle = entry.max_angle
                prot = pa["albuf_rotation"][p]
                lpos = pa["albuf_local_pos"][v]
                lrot = pa["albuf_local_rot"][v]

                cpos = pa["next_positions"][v]
                ppos = pa["next_positions"][p]
                v_vec = cpos - ppos
                vlen = pm.length(v_vec)
                skip1 = vlen < defs.EPSILON

                tv = pm.quat_rotate(prot, lpos)
                tvlen = pm.length(tv)
                snap = use_limit & ~skip1 & (tvlen < defs.EPSILON)
                if np.any(snap):
                    idx = np.flatnonzero(snap)
                    add = ppos[idx] - cpos[idx]
                    pa["next_positions"][v[idx]] = ppos[idx]
                    pa["velocity_positions"][v[idx]] += add
                    pa["albuf_rotation"][v[idx]] = pm.quat_mul(prot[idx], lrot[idx])
                    cpos = pa["next_positions"][v]

                work = use_limit & ~skip1 & ~snap
                unit_v = v_vec / np.where(vlen > 1e-30, vlen, 1.0)[:, None]
                unit_tv = tv / np.where(tvlen > 1e-30, tvlen, 1.0)[:, None]
                blen = pa["albuf_length"][v]
                vlen2 = pm.lerp(vlen, blen, np.float32(0.5))
                work = work & (blen >= defs.EPSILON) & (vlen2 >= defs.EPSILON)
                v_scaled = unit_v * vlen2[:, None]

                angle = pm.angle_between(v_scaled, unit_tv)
                over = angle > max_angle
                recovery = pm.lerp(angle, max_angle, limit_stiffness)
                clamped, _ = pm.clamp_angle_vector(v_scaled, unit_tv, recovery)
                rv = np.where((over & work)[:, None], clamped, v_scaled)

                rot_pos = ppos + v_scaled * limit_rot_ratio
                pf = rot_pos - rv * limit_rot_ratio
                cf = rot_pos + rv * (1.0 - limit_rot_ratio)
                padd = (pf - ppos) * p_inv_mass[:, None]
                cadd = (cf - cpos) * c_inv_mass[:, None]
                padd = np.where(work[:, None], padd, 0.0)
                cadd = np.where(work[:, None], cadd, 0.0)

                cpos = cpos + cadd
                pa["next_positions"][v] = cpos
                pa["velocity_positions"][v] += cadd * attenuation
                p_write = work & p_move
                if np.any(p_write):
                    idx = np.flatnonzero(p_write)
                    pa["next_positions"][p[idx]] += padd[idx]
                    pa["velocity_positions"][p[idx]] += padd[idx] * attenuation
                    ppos = pa["next_positions"][p]

                v3 = cpos - ppos
                vlen3 = pm.length(v3)
                fix_ok = work & (vlen3 >= defs.EPSILON)
                unit_v3 = v3 / np.where(vlen3 > 1e-30, vlen3, 1.0)[:, None]
                nrot = pm.quat_mul(prot, lrot)
                q = pm.from_to_rotation(unit_tv, unit_v3, 1.0, pre_normalized=True)
                nrot = pm.quat_mul(q, nrot)
                pa["albuf_rotation"][v] = np.where(fix_ok[:, None], nrot, pa["albuf_rotation"][v])

            if entry.any_restoration:
                use_restoration = entry.use_restoration
                restoration_attenuation = entry.restoration_attenuation
                stiffness = entry.restoration_stiffness
                cpos = pa["next_positions"][v]
                ppos = pa["next_positions"][p]
                tv = pa["albuf_restore"][v]
                tvlen = pm.length(tv)
                snap = use_restoration & (tvlen < defs.EPSILON)
                if np.any(snap):
                    idx = np.flatnonzero(snap)
                    add = ppos[idx] - cpos[idx]
                    pa["next_positions"][v[idx]] = ppos[idx]
                    pa["velocity_positions"][v[idx]] += add
                    cpos = pa["next_positions"][v]

                v_vec = cpos - ppos
                vlen = pm.length(v_vec)
                work = use_restoration & ~snap & (vlen >= defs.EPSILON)

                unit_v = v_vec / np.where(vlen > 1e-30, vlen, 1.0)[:, None]
                unit_tv = tv / np.where(tvlen > 1e-30, tvlen, 1.0)[:, None]
                q = pm.from_to_rotation(unit_v, unit_tv, stiffness, pre_normalized=True)
                rv = pm.quat_rotate(q, v_vec)

                rot_pos = ppos + v_vec * restoration_rot_ratio
                pf = rot_pos - rv * restoration_rot_ratio
                cf = rot_pos + rv * (1.0 - restoration_rot_ratio)
                padd = (pf - ppos) * p_inv_mass[:, None]
                cadd = (cf - cpos) * c_inv_mass[:, None]
                padd = np.where(work[:, None], padd, 0.0)
                cadd = np.where(work[:, None], cadd, 0.0)

                pa["next_positions"][v] = cpos + cadd
                pa["velocity_positions"][v] += cadd * restoration_attenuation[:, None]
                p_write = work & p_move
                if np.any(p_write):
                    idx = np.flatnonzero(p_write)
                    pa["next_positions"][p[idx]] += padd[idx]
                    pa["velocity_positions"][p[idx]] += padd[idx] * restoration_attenuation[idx][:, None]
