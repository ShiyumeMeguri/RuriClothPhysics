import numpy as np

from .. import defs
from .. import math as pm


def run(world, ctx):
    _display(world, ctx)
    _postline(world, ctx)
    _post_triangles(world, ctx)
    _output(world, ctx)


def _display(world, ctx):
    fp = ctx.frame_particle_index
    if len(fp) == 0:
        return
    pa = world.particles
    tt = world.team
    pt = pa["team"][fp]
    sdt = np.float32(ctx.simulation_delta_time)

    anim_position_snapshot = pa["positions"][fp].copy()
    anim_rotation_snapshot = pa["rotations"][fp].copy()

    move = ctx.frame_entry_index("update_move", world.update_move)
    if len(move):
        index = world.update_move["particle"][move]
        team = pa["team"][index]
        dpos = pa["old_positions"][index]
        velocity = pa["real_velocities"][index] * sdt
        fpos = dpos + velocity
        interval = (tt["now_update_time"][team] + sdt) - tt["old_time"][team]
        t = np.where(interval > 0.0,
                     (tt["time"][team] - tt["old_time"][team])
                     / np.where(interval > 0, interval, 1.0),
                     0.0).astype(np.float32)
        fpos = pm.lerp(pa["display_positions"][index], fpos, t[:, None])

        roots = pa["vertex_root"][index]
        has_root = roots >= 0
        if np.any(has_root):
            idx = np.flatnonzero(has_root)
            root_pos = pa["positions"][roots[idx]]
            original_dist = pm.length(root_pos - pa["positions"][index[idx]])
            clamp_dist = original_dist * defs.MAX_DISTANCE_RATIO_FUTURE_PREDICTION
            v = fpos[idx] - root_pos
            v = pm.clamp_vector(v, clamp_dist)
            fpos[idx] = root_pos + v

        pa["display_positions"][index] = fpos
        blend = tt["blend_weight"][team]
        pa["positions"][index] = pm.lerp(pa["positions"][index], fpos, blend[:, None])

    fixed = ctx.frame_entry_index("update_fixed", world.update_fixed)
    if len(fixed):
        index = world.update_fixed["particle"][fixed]
        pa["display_positions"][index] = pa["positions"][index]

    running = tt["running"][pt]
    if np.any(running):
        r = fp[running]
        keep = np.flatnonzero(running)
        pa["old_anim_positions"][r] = anim_position_snapshot[keep]
        pa["old_anim_rotations"][r] = anim_rotation_snapshot[keep]

    negative = tt["is_negative_scale"][pt]
    if np.any(negative):
        g = fp[negative]
        gt = pa["team"][g]
        neg = tt["negative_scale_direction"][gt]
        normals = pm.quat_to_normal(pa["rotations"][g]) * neg[:, 1:2]
        tangents = pm.quat_to_tangent(pa["rotations"][g]) * neg[:, 2:3]
        pa["rotations"][g] = pm.look_rotation(tangents, normals)

    pa["temp_base_positions"][fp] = pa["positions"][fp]
    pa["temp_base_rotations"][fp] = pa["rotations"][fp]


def _postline(world, ctx):
    pa = world.particles
    tt = world.team
    frame_mask = ctx.frame_team_mask

    for entry_vertex_all, child_owner_all, child_vertex_all in world.postline_levels:
        if len(entry_vertex_all) == 0:
            continue
        entry_active = frame_mask[pa["team"][entry_vertex_all]]
        if not np.any(entry_active):
            continue
        remap = np.full(len(entry_vertex_all), -1, dtype=np.int64)
        remap[entry_active] = np.arange(int(entry_active.sum()))
        entry_vertex = entry_vertex_all[entry_active]
        child_keep = entry_active[child_owner_all]
        child_owner = remap[child_owner_all[child_keep]]
        child_vertex = child_vertex_all[child_keep]

        et = pa["team"][entry_vertex]
        average_rate = tt["rotational_interpolation"][et]
        root_interpolation = tt["root_rotation"][et]
        blend_weight = tt["blend_weight"][et]

        rot = pa["rotations"][entry_vertex]
        pos = pa["positions"][entry_vertex]
        base_pos = pa["temp_base_positions"][entry_vertex]
        base_rot = pa["temp_base_rotations"][entry_vertex]
        base_inv = pm.quat_inverse(base_rot)
        entry_invalid = pa["attr_invalid"][entry_vertex]

        if len(child_vertex):
            o = child_owner
            c = child_vertex
            ct = pa["team"][c]
            anime_ratio = tt["animation_pose_ratio"][ct]
            neg_dir = tt["negative_scale_direction"][ct]
            neg_quat = tt["negative_scale_quaternion"][ct]
            c_base_pos = pa["temp_base_positions"][c]
            c_base_rot = pa["temp_base_rotations"][c]
            c_base_local_pos = pm.quat_rotate(base_inv[o], c_base_pos - base_pos[o])
            c_base_local_rot = pm.quat_mul(base_inv[o], c_base_rot)

            lpos = pm.lerp(pa["vertex_local_positions"][c] * neg_dir,
                           c_base_local_pos, anime_ratio[:, None])
            lrot = pm.quat_slerp(pa["vertex_local_rotations"][c] * neg_quat,
                                 c_base_local_rot, anime_ratio)

            is_c0 = pa["attr_zero_distance"][c]
            tv = pm.quat_rotate(rot[o], lpos)
            tv = np.where(is_c0[:, None], 0.0, tv)

            c_move = pa["attr_move"][c]
            v = pa["positions"][c] - pos[o]
            contribution = np.where(c_move[:, None], v, tv)

            owner_valid = ~entry_invalid[o]
            ctv_sum = np.zeros((len(entry_vertex), 3), dtype=np.float64)
            cv_sum = np.zeros((len(entry_vertex), 3), dtype=np.float64)
            gated_owner = o[owner_valid]
            if len(gated_owner):
                run_starts = np.flatnonzero(np.concatenate(([True], gated_owner[1:] != gated_owner[:-1])))
                touched_owner = gated_owner[run_starts]
                ctv_sum[touched_owner] = np.add.reduceat(tv[owner_valid].astype(np.float64), run_starts, axis=0)
                cv_sum[touched_owner] = np.add.reduceat(contribution[owner_valid].astype(np.float64), run_starts, axis=0)

            write = c_move & owner_valid
            if np.any(write):
                idx = np.flatnonzero(write)
                crot = pm.quat_mul(rot[o[idx]], lrot[idx])
                need_fix = ~is_c0[idx]
                if np.any(need_fix):
                    fix_idx = idx[need_fix]
                    q = pm.from_to_rotation(tv[fix_idx].astype(np.float32),
                                            v[fix_idx].astype(np.float32))
                    fixed_rot = pm.quat_mul(q, pm.quat_mul(rot[o[fix_idx]], lrot[fix_idx]))
                    crot[need_fix] = fixed_rot
                pa["rotations"][c[idx]] = crot

            has_children = np.zeros(len(entry_vertex), dtype=bool)
            has_children[o] = True
            t_ratio = np.where(pa["attr_move"][entry_vertex], average_rate, root_interpolation)
            ctv32 = ctv_sum.astype(np.float32)
            cv32 = cv_sum.astype(np.float32)
            zero = (pm.length(ctv32) < 1e-8) | (pm.length(cv32) < 1e-8)
            apply = has_children & ~entry_invalid & ~zero
            cq = pm.from_to_rotation(ctv32, cv32, t_ratio)
            new_rot = pm.quat_mul(cq, rot)
            rot = np.where(apply[:, None], new_rot, rot)

        rot = pm.quat_slerp(base_rot, rot, blend_weight)
        pa["rotations"][entry_vertex] = rot


def _post_triangles(world, ctx):
    pa = world.particles
    tt = world.team
    va = world.v2t
    ta = world.triangles
    frame_mask = ctx.frame_team_mask

    tri_index = np.flatnonzero(frame_mask[ta["team"]])
    if len(tri_index) == 0:
        return
    triangles = ta["triangle"][tri_index]
    tri_team = ta["team"][tri_index]
    p0 = pa["positions"][triangles[:, 0]]
    p1 = pa["positions"][triangles[:, 1]]
    p2 = pa["positions"][triangles[:, 2]]
    cross = pm.cross(p1 - p0, p2 - p0)
    lengths = pm.length(cross)
    tri_normals64 = (cross / np.where(lengths > defs.EPSILON, lengths, 1.0)[:, None]).astype(np.float64)
    tri_normals64 = tri_normals64 * tt["negative_scale_triangle_sign"][tri_team][:, 0:1]

    uv0 = pa["uv"][triangles[:, 0]]
    uv1 = pa["uv"][triangles[:, 1]]
    uv2 = pa["uv"][triangles[:, 2]]
    tri_tangents64 = _triangle_tangent_runtime(p0.astype(np.float64), p1.astype(np.float64),
                                               p2.astype(np.float64),
                                               uv0.astype(np.float64), uv1.astype(np.float64),
                                               uv2.astype(np.float64))
    tri_tangents64 = tri_tangents64 * tt["negative_scale_triangle_sign"][tri_team][:, 1:2]

    entry_index = np.flatnonzero(frame_mask[va["team"]])
    if len(entry_index) == 0:
        return
    owner = va["owner"][entry_index]
    triangle = va["triangle"][entry_index]
    flip_normal = va["flip_normal"][entry_index].astype(np.float64)
    flip_tangent = va["flip_tangent"][entry_index].astype(np.float64)

    triangle_position = np.searchsorted(tri_index, triangle)
    normal_rows = tri_normals64[triangle_position] * flip_normal[:, None]
    tangent_rows = tri_tangents64[triangle_position] * flip_tangent[:, None]
    run_starts = np.flatnonzero(np.concatenate(([True], owner[1:] != owner[:-1])))
    owners = owner[run_starts]
    nor = np.add.reduceat(normal_rows, run_starts, axis=0)
    tan = np.add.reduceat(tangent_rows, run_starts, axis=0)
    ln = np.linalg.norm(nor, axis=1)
    lt = np.linalg.norm(tan, axis=1)
    ok = (ln > 1e-6) & (lt > 1e-6)
    nor = nor / np.where(ln > 1e-30, ln, 1.0)[:, None]
    tan = tan / np.where(lt > 1e-30, lt, 1.0)[:, None]
    d = np.sum(nor * tan, axis=1)
    ok &= (d != 1.0) & (d != -1.0)
    binormal = pm.cross(nor, tan)
    bl = np.linalg.norm(binormal, axis=1)
    binormal = binormal / np.where(bl > 1e-30, bl, 1.0)[:, None]
    rot = pm.look_rotation(binormal.astype(np.float32), nor.astype(np.float32))
    ot = pa["team"][owners]
    adjust = pa["normal_adjustment_rotations"][owners] * tt["negative_scale_quaternion"][ot]
    new_rot = pm.quat_mul(rot, adjust)
    apply = owners[ok]
    if len(apply):
        pa["rotations"][apply] = new_rot[ok]


def _triangle_tangent_runtime(p0, p1, p2, uv0, uv1, uv2):
    dist_ba = p1 - p0
    dist_ca = p2 - p0
    t_ba = uv1 - uv0
    t_ca = uv2 - uv0
    area = t_ba[:, 0] * t_ca[:, 1] - t_ba[:, 1] * t_ca[:, 0]
    area = np.where(area == 0.0, 1.0, area)
    delta = -1.0 / area
    tan = (dist_ba * t_ca[:, 1:2] + dist_ca * -t_ba[:, 1:2]) * delta[:, None]
    l = np.linalg.norm(tan, axis=1)
    return np.where((l > 1e-30)[:, None], tan / np.where(l > 1e-30, l, 1.0)[:, None], tan)


def _output(world, ctx):
    fp = ctx.frame_particle_index
    if len(fp) == 0:
        return
    pa = world.particles
    tt = world.team
    pt = pa["team"][fp]
    v2t = pa["vertex_to_transform_rotations"][fp] * tt["negative_scale_quaternion"][pt]
    pa["out_rotations"][fp] = pm.quat_mul(pa["rotations"][fp], v2t)
