import numpy as np

from ...cloth_kernel import defs
from ...cloth_kernel import math as pm

POINT_HISTORY = ("old_frame_positions", "now_positions", "old_positions",
                 "old_frame_tips", "now_tips", "old_tips")
ROTATION_HISTORY = ("old_frame_rotations", "now_rotations", "old_rotations")


def frame_pre(world, ctx):
    ca = world.colliders
    tt = world.team
    ct = ca["team"]
    frame_columns = np.flatnonzero(ctx.frame_team_mask[ct])
    if len(frame_columns) == 0:
        return
    enabled_now = ca["enabled"][frame_columns]
    rising_edge_all = enabled_now & ~ca["enabled_prev"][frame_columns]
    ca["active"][frame_columns] = enabled_now
    ca["enabled_prev"][frame_columns] = enabled_now
    keep = np.flatnonzero(enabled_now)
    if len(keep) == 0:
        return
    index = frame_columns[keep]
    team = ct[index]
    rising_edge = rising_edge_all[keep]

    ca["frame_positions"][index] = ca["input_positions"][index]
    ca["frame_rotations"][index] = ca["input_rotations"][index]
    ca["frame_tips"][index] = ca["input_tips"][index]
    ca["frame_radii"][index] = ca["input_radii"][index]

    reset = tt["reset_pending"][team] | rising_edge
    if np.any(reset):
        r = index[reset]
        for source, targets in (
                ("frame_positions", ("old_frame_positions", "now_positions", "old_positions")),
                ("frame_rotations", ("old_frame_rotations", "now_rotations", "old_rotations")),
                ("frame_tips", ("old_frame_tips", "now_tips", "old_tips"))):
            value = ca[source][r]
            for name in targets:
                ca[name][r] = value

    live = ~reset
    negative = live & tt["negative_scale_teleport"][team]
    if np.any(negative):
        g = index[negative]
        gt = ct[g]
        matrix = tt["negative_scale_matrix"][gt]
        flip = tt["negative_scale_change"][gt]
        for name in POINT_HISTORY:
            ca[name][g] = pm.transform_points(ca[name][g], matrix)
        for name in ROTATION_HISTORY:
            ca[name][g] = pm.transform_rotations(ca[name][g], matrix, flip)

    shifted = live & tt["inertia_shift"][team]
    if np.any(shifted):
        g = index[shifted]
        gt = ct[g]
        center_pos = tt["old_component_world_position"][gt]
        shift_vector = tt["frame_component_shift_vector"][gt]
        shift_rotation = tt["frame_component_shift_rotation"][gt]
        for name in POINT_HISTORY:
            local = ca[name][g] - center_pos
            ca[name][g] = pm.quat_rotate(shift_rotation, local) + center_pos + shift_vector
        for name in ROTATION_HISTORY:
            ca[name][g] = pm.quat_mul(shift_rotation, ca[name][g])


def start_step(world, ctx):
    ca = world.colliders
    tt = world.team
    ct = ca["team"]
    mask = ctx.step_team_mask[ct] & ca["active"]
    index = np.flatnonzero(mask)
    if len(index) == 0:
        return
    team = ct[index]

    t = tt["frame_interpolation"][team]
    pos = pm.lerp(ca["old_frame_positions"][index], ca["frame_positions"][index], t[:, None])
    rot = pm.quat_slerp(ca["old_frame_rotations"][index], ca["frame_rotations"][index], t)
    tip = pm.lerp(ca["old_frame_tips"][index], ca["frame_tips"][index], t[:, None])
    ca["now_positions"][index] = pos
    ca["now_rotations"][index] = rot
    ca["now_tips"][index] = tip

    move_ratio = tt["step_move_inertia_ratio"][team][:, None]
    old_pos = pm.lerp(ca["old_positions"][index], pos, move_ratio)
    old_rot = pm.quat_slerp(ca["old_rotations"][index], rot, tt["step_rotation_inertia_ratio"][team])
    old_tip = pm.lerp(ca["old_tips"][index], tip, move_ratio)
    ca["old_positions"][index] = old_pos
    ca["old_rotations"][index] = old_rot
    ca["old_tips"][index] = old_tip

    ca["work_rot"][index] = rot
    ca["work_inv_old_rot"][index] = pm.quat_inverse(old_rot)

    kinds = ca["kind"][index]
    radii = ca["frame_radii"][index]

    sphere = kinds == defs.COLLIDER_SPHERE
    if np.any(sphere):
        g = index[sphere]
        radius = radii[sphere, 0]
        ca["work_radius"][g, 0] = radius
        ca["work_radius"][g, 1] = radius
        ca["work_old_pos"][g, 0] = old_pos[sphere]
        ca["work_next_pos"][g, 0] = pos[sphere]
        low = np.minimum(old_pos[sphere], pos[sphere]) - radius[:, None]
        high = np.maximum(old_pos[sphere], pos[sphere]) + radius[:, None]
        ca["work_aabb_min"][g] = low
        ca["work_aabb_max"][g] = high

    capsule = kinds == defs.COLLIDER_CAPSULE
    if np.any(capsule):
        g = index[capsule]
        start_radius = radii[capsule, 0]
        end_radius = radii[capsule, 1]
        s_old = old_pos[capsule]
        e_old = old_tip[capsule]
        s_now = pos[capsule]
        e_now = tip[capsule]

        ca["work_radius"][g, 0] = start_radius
        ca["work_radius"][g, 1] = end_radius
        ca["work_old_pos"][g, 0] = s_old
        ca["work_old_pos"][g, 1] = e_old
        ca["work_next_pos"][g, 0] = s_now
        ca["work_next_pos"][g, 1] = e_now

        low = np.minimum(np.minimum(s_old, s_now) - start_radius[:, None],
                         np.minimum(e_old, e_now) - end_radius[:, None])
        high = np.maximum(np.maximum(s_old, s_now) + start_radius[:, None],
                          np.maximum(e_old, e_now) + end_radius[:, None])
        ca["work_aabb_min"][g] = low
        ca["work_aabb_max"][g] = high

    plane = kinds == defs.COLLIDER_PLANE
    if np.any(plane):
        g = index[plane]
        up = np.zeros((len(g), 3), dtype=np.float32)
        up[:, 2] = 1.0
        normal = pm.quat_rotate(rot[plane], up)
        ca["work_old_pos"][g, 0] = normal
        ca["work_next_pos"][g, 0] = pos[plane]
        ca["work_aabb_min"][g] = -np.inf
        ca["work_aabb_max"][g] = np.inf


def solve(world, ctx):
    _solve_point(world, ctx)
    _solve_edge(world, ctx)


def _solve_point(world, ctx):
    pa = world.particles
    ca = world.colliders
    tt = world.team
    ppa = world.point_pairs
    pair_team = ppa["team"]
    active = ctx.step_team_mask[pair_team] \
        & (tt["collision_mode"][pair_team] == defs.COLLISION_POINT) \
        & ca["active"][ppa["collider"]]
    pair_index = np.flatnonzero(active)
    if len(pair_index) == 0:
        return
    p = ppa["particle"][pair_index]
    c = ppa["collider"][pair_index]
    team = pair_team[pair_index]

    depth = pa["depth"][p]
    radius = np.maximum(pm.evaluate_team_lut(tt["radius_lut"], team, depth), 0.0001) \
        * tt["scale_ratio"][team]
    cfr = radius * 1.0
    next_pos = pa["next_positions"][p]
    original = next_pos
    is_spring = tt["is_spring"][team]
    max_length = np.maximum(pm.evaluate_team_lut(tt["limit_distance_lut"], team, depth), 0.0001) \
        * tt["scale_ratio"][team]
    base_pos = pa["base_positions"][p]

    particle_min = next_pos - (radius + cfr)[:, None]
    particle_max = next_pos + (radius + cfr)[:, None]

    dist = np.full(len(pair_index), np.inf, dtype=np.float32)
    out_pos = next_pos.copy()
    normal = np.zeros((len(pair_index), 3), dtype=np.float32)

    kinds = ca["kind"][c]

    sphere = np.flatnonzero(kinds == defs.COLLIDER_SPHERE)
    if len(sphere):
        overlap = _pair_overlap(particle_min[sphere], particle_max[sphere], ca, c[sphere])
        cold = ca["work_old_pos"][c[sphere], 0]
        cnow = ca["work_next_pos"][c[sphere], 0]
        cradius = ca["work_radius"][c[sphere], 0]
        v = next_pos[sphere] - cold
        n = pm.normalize(v)
        plane_pos = cnow + n * (cradius + radius[sphere])[:, None]
        d, out = pm.intersect_point_plane_dist(plane_pos, n, next_pos[sphere])
        spring_pair = is_spring[sphere]
        if np.any(spring_pair):
            clamped = pm.clamp_distance(base_pos[sphere], out, max_length[sphere])
            l = pm.length(clamped - base_pos[sphere])
            t = pm.saturate(l / radius[sphere]) * 0.85
            blended = pm.lerp(clamped, original[sphere], t[:, None])
            out = np.where(spring_pair[:, None], blended, out)
            d = np.where(spring_pair, d * 3.0, d)
        dist[sphere] = np.where(overlap, d, np.inf)
        out_pos[sphere] = np.where(overlap[:, None], out, next_pos[sphere])
        normal[sphere] = np.where(overlap[:, None], n, 0.0)

    capsule = np.flatnonzero(kinds == defs.COLLIDER_CAPSULE)
    if len(capsule):
        overlap = _pair_overlap(particle_min[capsule], particle_max[capsule], ca, c[capsule])
        cc = c[capsule]
        s_old = ca["work_old_pos"][cc, 0]
        e_old = ca["work_old_pos"][cc, 1]
        s_now = ca["work_next_pos"][cc, 0]
        e_now = ca["work_next_pos"][cc, 1]
        sr = ca["work_radius"][cc, 0]
        er = ca["work_radius"][cc, 1]
        inv_old_rot = ca["work_inv_old_rot"][cc]
        rot = ca["work_rot"][cc]

        point = next_pos[capsule]
        t = pm.closest_pt_point_segment_ratio(point, s_old, e_old)
        r = sr + (er - sr) * t
        d0 = s_old + (e_old - s_old) * t[:, None]
        v = point - d0
        lv = pm.quat_rotate(inv_old_rot, v)
        d1 = s_now + (e_now - s_now) * t[:, None]
        v2 = pm.quat_rotate(rot, lv)
        n = pm.normalize(v2)
        plane_pos = d1 + n * (r + radius[capsule])[:, None]
        d, out = pm.intersect_point_plane_dist(plane_pos, n, point)
        dist[capsule] = np.where(overlap, d, np.inf)
        out_pos[capsule] = np.where(overlap[:, None], out, point)
        normal[capsule] = np.where(overlap[:, None], n, 0.0)

    plane = np.flatnonzero(kinds == defs.COLLIDER_PLANE)
    if len(plane):
        cc = c[plane]
        n = ca["work_old_pos"][cc, 0]
        cpos = ca["work_next_pos"][cc, 0]
        plane_pos = cpos + n * radius[plane][:, None]
        d, out = pm.intersect_point_plane_dist(plane_pos, n, next_pos[plane])
        dist[plane] = d
        out_pos[plane] = out
        normal[plane] = n

    push = dist <= 0.0
    near = dist <= cfr

    run_starts = np.flatnonzero(np.concatenate(([True], p[1:] != p[:-1])))
    touched = p[run_starts]

    push_indicator = push.astype(np.float64)
    near_indicator = near.astype(np.float64)
    push_position_rows = (out_pos - next_pos).astype(np.float64) * push_indicator[:, None]
    push_normal_rows = normal.astype(np.float64) * push_indicator[:, None]
    near_normal_rows = normal.astype(np.float64) * near_indicator[:, None]
    dist_for_min = np.where(near, dist, np.inf)

    count = np.add.reduceat(push_indicator, run_starts)
    position_sum = np.add.reduceat(push_position_rows, run_starts, axis=0)
    normal_sum = np.add.reduceat(push_normal_rows, run_starts, axis=0)
    near_normal_total = np.add.reduceat(near_normal_rows, run_starts, axis=0)
    near_count = np.add.reduceat(near_indicator, run_starts)
    min_dist = np.minimum.reduceat(dist_for_min, run_starts)

    unique_cfr = cfr[run_starts]
    segment_is_spring = is_spring[run_starts]

    has_push = count > 0
    safe_count = np.maximum(count, 1.0)[:, None].astype(np.float32)
    normal_avg = (normal_sum / safe_count).astype(np.float32)
    normal_length = pm.length(normal_avg)
    pos_avg = (position_sum / safe_count).astype(np.float32)
    t = np.minimum(normal_length, 1.0)
    valid_push = has_push & (normal_length >= defs.EPSILON)
    delta = np.where(valid_push[:, None], pos_avg * t[:, None], 0.0)
    pa["next_positions"][touched] += delta

    near_normal_sum = near_normal_total.astype(np.float32)
    has_near = (near_count > 0) & (unique_cfr > 0.0) \
        & (pm.length(near_normal_sum) ** 2 > 1e-6)
    friction = 1.0 - pm.saturate(np.where(np.isfinite(min_dist), min_dist, 0.0)
                                 / np.where(unique_cfr > 0, unique_cfr, 1.0))
    old_friction = pa["friction"][touched]
    pa["friction"][touched] = np.where(has_near, np.maximum(friction, old_friction), old_friction)
    near_out = np.where(has_near[:, None], pm.normalize(near_normal_sum), near_normal_sum)
    pa["collision_normals"][touched] = near_out

    if np.any(segment_is_spring):
        spring_delta = np.where((has_push & segment_is_spring)[:, None], pos_avg, 0.0)
        pa["velocity_positions"][touched] += spring_delta


def _pair_overlap(box_min, box_max, ca, colliders):
    c_min = ca["work_aabb_min"][colliders]
    c_max = ca["work_aabb_max"][colliders]
    return ((box_min <= c_max) & (box_max >= c_min)).all(axis=1)


def _solve_edge(world, ctx):
    pa = world.particles
    ca = world.colliders
    tt = world.team
    epa = world.edge_pairs
    pair_team = epa["team"]
    active = ctx.step_team_mask[pair_team] \
        & (tt["collision_mode"][pair_team] == defs.COLLISION_EDGE) \
        & ca["active"][epa["collider"]]
    pair_index = np.flatnonzero(active)
    if len(pair_index) == 0:
        return
    edge_entry = epa["edge"][pair_index]
    c = epa["collider"][pair_index]
    team = pair_team[pair_index]

    edges = world.collision_edges["edge"][edge_entry]
    e0 = edges[:, 0]
    e1 = edges[:, 1]
    pos0 = pa["next_positions"][e0]
    pos1 = pa["next_positions"][e1]
    depth0 = pa["depth"][e0]
    depth1 = pa["depth"][e1]
    r0 = pm.evaluate_team_lut(tt["radius_lut"], team, depth0) * tt["scale_ratio"][team]
    r1 = pm.evaluate_team_lut(tt["radius_lut"], team, depth1) * tt["scale_ratio"][team]
    cfr = (r0 + r1) * 0.5

    edge_min = np.minimum(pos0 - r0[:, None], pos1 - r1[:, None]) - cfr[:, None]
    edge_max = np.maximum(pos0 + r0[:, None], pos1 + r1[:, None]) + cfr[:, None]

    dist = np.full(len(pair_index), np.inf, dtype=np.float32)
    corr0 = np.zeros((len(pair_index), 3), dtype=np.float32)
    corr1 = np.zeros((len(pair_index), 3), dtype=np.float32)
    normal = np.zeros((len(pair_index), 3), dtype=np.float32)

    kinds = ca["kind"][c]

    sphere = np.flatnonzero(kinds == defs.COLLIDER_SPHERE)
    if len(sphere):
        d, c0, c1, nrm = _edge_sphere(pos0[sphere], pos1[sphere], r0[sphere], r1[sphere],
                                      cfr[sphere], edge_min[sphere], edge_max[sphere],
                                      ca, c[sphere])
        dist[sphere] = d
        corr0[sphere] = c0
        corr1[sphere] = c1
        normal[sphere] = nrm

    capsule = np.flatnonzero(kinds == defs.COLLIDER_CAPSULE)
    if len(capsule):
        d, c0, c1, nrm = _edge_capsule(pos0[capsule], pos1[capsule], r0[capsule], r1[capsule],
                                       cfr[capsule], edge_min[capsule], edge_max[capsule],
                                       ca, c[capsule])
        dist[capsule] = d
        corr0[capsule] = c0
        corr1[capsule] = c1
        normal[capsule] = nrm

    plane = np.flatnonzero(kinds == defs.COLLIDER_PLANE)
    if len(plane):
        cc = c[plane]
        n = ca["work_old_pos"][cc, 0]
        cpos = ca["work_next_pos"][cc, 0]
        dist0, out0 = pm.intersect_point_plane_dist(cpos + n * r0[plane][:, None], n, pos0[plane])
        dist1, out1 = pm.intersect_point_plane_dist(cpos + n * r1[plane][:, None], n, pos1[plane])
        dist[plane] = np.minimum(dist0, dist1)
        corr0[plane] = out0 - pos0[plane]
        corr1[plane] = out1 - pos1[plane]
        normal[plane] = n

    push = dist <= 0.0
    near = dist <= cfr

    edge_run_starts = np.flatnonzero(np.concatenate(([True], edge_entry[1:] != edge_entry[:-1])))
    ue0 = e0[edge_run_starts]
    ue1 = e1[edge_run_starts]
    ucfr = cfr[edge_run_starts]

    push_indicator = push.astype(np.float64)
    near_indicator = near.astype(np.float64)
    edge_pos0_rows = corr0.astype(np.float64) * push_indicator[:, None]
    edge_pos1_rows = corr1.astype(np.float64) * push_indicator[:, None]
    edge_normal_rows = normal.astype(np.float64) * push_indicator[:, None]
    edge_near_normal_rows = normal.astype(np.float64) * near_indicator[:, None]
    dist_for_min = np.where(near, dist, np.inf)

    count = np.add.reduceat(push_indicator, edge_run_starts)
    edge_pos0_sum = np.add.reduceat(edge_pos0_rows, edge_run_starts, axis=0)
    edge_pos1_sum = np.add.reduceat(edge_pos1_rows, edge_run_starts, axis=0)
    edge_normal_sum = np.add.reduceat(edge_normal_rows, edge_run_starts, axis=0)
    edge_near_normal_sum = np.add.reduceat(edge_near_normal_rows, edge_run_starts, axis=0)
    edge_near_count = np.add.reduceat(near_indicator, edge_run_starts)
    edge_min_dist = np.minimum.reduceat(dist_for_min, edge_run_starts)

    has_push = count > 0
    safe_count = np.maximum(count, 1.0)[:, None].astype(np.float32)
    normal_avg = (edge_normal_sum / safe_count).astype(np.float32)
    normal_length = pm.length(normal_avg)
    t = np.minimum(normal_length, 1.0)
    valid = has_push & (normal_length > defs.EPSILON)
    scale = np.where(valid, t, 0.0)[:, None] / safe_count
    delta0 = edge_pos0_sum.astype(np.float32) * scale
    delta1 = edge_pos1_sum.astype(np.float32) * scale

    near_normal_sum = edge_near_normal_sum.astype(np.float32)
    has_near = (edge_near_count > 0) & (ucfr > 0.0) \
        & (pm.length(near_normal_sum) ** 2 > 1e-6)
    friction = 1.0 - pm.saturate(np.where(np.isfinite(edge_min_dist), edge_min_dist, 0.0)
                                 / np.where(ucfr > 0, ucfr, 1.0))
    near_out = np.where(has_near[:, None], pm.normalize(near_normal_sum), 0.0)

    move0 = pa["attr_move"][ue0]
    move1 = pa["attr_move"][ue1]
    mask0 = has_near & move0
    mask1 = has_near & move1

    endpoint_targets = np.concatenate((ue0, ue1))
    endpoint_delta = np.concatenate((delta0, delta1)).astype(np.float64)
    endpoint_count_rows = np.concatenate((valid, valid)).astype(np.float64)
    endpoint_friction_rows = np.concatenate((np.where(mask0, friction, 0.0),
                                             np.where(mask1, friction, 0.0)))
    endpoint_normal_rows = np.concatenate((np.where(mask0[:, None], near_out, 0.0),
                                           np.where(mask1[:, None], near_out, 0.0))).astype(np.float64)

    order = np.argsort(endpoint_targets, kind="stable")
    sorted_endpoints = endpoint_targets[order]
    endpoint_run_starts = np.flatnonzero(np.concatenate(([True], sorted_endpoints[1:] != sorted_endpoints[:-1])))
    endpoints = sorted_endpoints[endpoint_run_starts]
    endpoint_count = np.add.reduceat(endpoint_count_rows[order], endpoint_run_starts)
    endpoint_position_sum = np.add.reduceat(endpoint_delta[order], endpoint_run_starts, axis=0)
    endpoint_friction = np.maximum.reduceat(endpoint_friction_rows[order], endpoint_run_starts).astype(np.float32)
    endpoint_normal_sum = np.add.reduceat(endpoint_normal_rows[order], endpoint_run_starts, axis=0)

    apply = endpoint_count > 0
    if np.any(apply):
        pa["next_positions"][endpoints[apply]] += (endpoint_position_sum[apply]
                                                   / endpoint_count[apply][:, None]).astype(np.float32)
    better = endpoint_friction > pa["friction"][endpoints]
    if np.any(better):
        pa["friction"][endpoints[better]] = endpoint_friction[better]
    normal_valid = pm.length(endpoint_normal_sum) ** 2 > 0.0
    if np.any(normal_valid):
        nv = endpoints[normal_valid]
        pa["collision_normals"][nv] = pm.normalize(endpoint_normal_sum[normal_valid]).astype(np.float32)


def _edge_aabb_overlap(edge_min, edge_max, ca, colliders):
    c_min = ca["work_aabb_min"][colliders]
    c_max = ca["work_aabb_max"][colliders]
    return ((edge_min <= c_max) & (edge_max >= c_min)).all(axis=1)


def _edge_sphere(pos0, pos1, r0, r1, cfr, edge_min, edge_max, ca, colliders):
    overlap = _edge_aabb_overlap(edge_min, edge_max, ca, colliders)
    cold = ca["work_old_pos"][colliders, 0]
    cnow = ca["work_next_pos"][colliders, 0]
    cradius = ca["work_radius"][colliders, 0]

    s = pm.closest_pt_point_segment_ratio(cold, pos0, pos1)
    closest = pos0 + (pos1 - pos0) * s[:, None]
    v = closest - cold
    clen = pm.length(v)
    degenerate = clen < 1e-9
    n = v / np.where(clen > 1e-30, clen, 1.0)[:, None]

    db = cnow - cold
    l1 = pm.dot(n, db)
    l = clen - l1
    r_edge = r0 + (r1 - r0) * s
    thickness = r_edge + cradius
    miss = l > (thickness + cfr)

    v2 = closest - cnow
    l2 = pm.dot(n, v2)
    no_contact = l2 > thickness
    near_dist = l2 - thickness

    c = thickness - l2
    b0 = 1.0 - s
    b1 = s
    denominator = b0 * b0 + b1 * b1
    scale = c / np.where(denominator > 0.0, denominator, 1.0)
    corr0 = n * (b0 * scale)[:, None]
    corr1 = n * (b1 * scale)[:, None]

    dist = np.where(no_contact, near_dist, -c)
    valid = overlap & ~degenerate & ~miss
    dist = np.where(valid, dist, np.inf)
    invalid = ~valid | no_contact
    corr0 = np.where(invalid[:, None], 0.0, corr0)
    corr1 = np.where(invalid[:, None], 0.0, corr1)
    n = np.where(valid[:, None], n, 0.0)
    return dist, corr0, corr1, n


def _edge_capsule(pos0, pos1, r0, r1, cfr, edge_min, edge_max, ca, colliders):
    overlap = _edge_aabb_overlap(edge_min, edge_max, ca, colliders)
    s_old = ca["work_old_pos"][colliders, 0]
    e_old = ca["work_old_pos"][colliders, 1]
    s_now = ca["work_next_pos"][colliders, 0]
    e_now = ca["work_next_pos"][colliders, 1]
    sr = ca["work_radius"][colliders, 0]
    er = ca["work_radius"][colliders, 1]

    s, t, c_a, c_b = pm.closest_pt_segment_segment(pos0, pos1, s_old, e_old)
    v = c_a - c_b
    clen = pm.length(v)
    degenerate = clen < 1e-9
    n = v / np.where(clen > 1e-30, clen, 1.0)[:, None]

    radius_differs = sr != er
    if np.any(radius_differs):
        so2 = s_old + n * sr[:, None]
        eo2 = e_old + n * er[:, None]
        s2, t2, _, _ = pm.closest_pt_segment_segment(pos0, pos1, so2, eo2)
        s = np.where(radius_differs, s2, s)
        t = np.where(radius_differs, t2, t)
        c_a2 = pos0 + (pos1 - pos0) * s[:, None]
        c_b2 = s_old + (e_old - s_old) * t[:, None]
        v2 = c_a2 - c_b2
        clen2 = pm.length(v2)
        n2 = v2 / np.where(clen2 > 1e-30, clen2, 1.0)[:, None]
        c_a = np.where(radius_differs[:, None], c_a2, c_a)
        c_b = np.where(radius_differs[:, None], c_b2, c_b)
        clen = np.where(radius_differs, clen2, clen)
        n = np.where(radius_differs[:, None], n2, n)
        degenerate = degenerate | (clen < 1e-9)

    d_b0 = s_now - s_old
    d_b1 = e_now - e_old
    db = d_b0 + (d_b1 - d_b0) * t[:, None]
    l1 = pm.dot(n, db)
    l = clen - l1
    r_edge = r0 + (r1 - r0) * s
    r_capsule = sr + (er - sr) * t
    thickness = r_edge + r_capsule
    miss = l > (thickness + cfr)

    d = s_now + (e_now - s_now) * t[:, None]
    v2 = c_a - d
    l2 = pm.dot(n, v2)
    no_contact = l2 > thickness
    near_dist = l2 - thickness

    c = thickness - l2
    b0 = 1.0 - s
    b1 = s
    denominator = b0 * b0 + b1 * b1
    scale = c / np.where(denominator > 0.0, denominator, 1.0)
    corr0 = n * (b0 * scale)[:, None]
    corr1 = n * (b1 * scale)[:, None]

    dist = np.where(no_contact, near_dist, -c)
    valid = overlap & ~degenerate & ~miss
    dist = np.where(valid, dist, np.inf)
    invalid = ~valid | no_contact
    corr0 = np.where(invalid[:, None], 0.0, corr0)
    corr1 = np.where(invalid[:, None], 0.0, corr1)
    n = np.where(valid[:, None], n, 0.0)
    return dist, corr0, corr1, n


def end_step(world, ctx):
    ca = world.colliders
    mask = ctx.step_team_mask[ca["team"]] & ca["active"]
    index = np.flatnonzero(mask)
    if len(index) == 0:
        return
    ca["old_positions"][index] = ca["now_positions"][index]
    ca["old_rotations"][index] = ca["now_rotations"][index]
    ca["old_tips"][index] = ca["now_tips"][index]


def frame_post(world, ctx):
    ca = world.colliders
    tt = world.team
    ct = ca["team"]
    mask = ctx.frame_team_mask[ct] & tt["running"][ct] & ca["active"]
    index = np.flatnonzero(mask)
    if len(index) == 0:
        return
    ca["old_frame_positions"][index] = ca["frame_positions"][index]
    ca["old_frame_rotations"][index] = ca["frame_rotations"][index]
    ca["old_frame_tips"][index] = ca["frame_tips"][index]
