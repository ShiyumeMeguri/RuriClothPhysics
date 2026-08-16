import numpy as np

from .. import defs
from .. import math as pm

KIND_CHUNK_FIELDS = {
    defs.KIND_POINT: ("sp_start", "sp_count", "use_point"),
    defs.KIND_EDGE: ("se_start", "se_count", "use_edge"),
    defs.KIND_TRIANGLE: ("st_start", "st_count", "use_triangle"),
}
KIND_AXES = {defs.KIND_POINT: 1, defs.KIND_EDGE: 2, defs.KIND_TRIANGLE: 3}


def frame_begin(world, ctx):
    tt = world.team
    ctx.self_contact_tasks = []
    ctx.self_intersect_tasks = []
    ctx.self_intersect_pairs = []
    frame_teams = ctx.frame_team_index
    tt["use_point"][frame_teams] = False
    tt["use_edge"][frame_teams] = False
    tt["use_triangle"][frame_teams] = False

    for slot in frame_teams:
        row = tt[slot]
        self_mode = int(row["self_mode"])
        sync_mode = int(row["sync_mode"])
        partner = int(row["sync_target"])
        if partner <= 0 or not tt["valid"][partner] or not tt["enabled"][partner] \
                or partner == slot:
            partner = 0

        has_edge = int(row["se_count"]) > 0
        has_triangle = int(row["st_count"]) > 0

        if self_mode == defs.SELF_MODE_FULL_MESH:
            if has_edge:
                tt["use_edge"][slot] = True
                ctx.self_contact_tasks.append(('EE', slot, slot))
            if has_triangle:
                tt["use_point"][slot] = True
                tt["use_triangle"][slot] = True
                ctx.self_contact_tasks.append(('PT', slot, slot))
            if has_edge and has_triangle:
                ctx.self_intersect_tasks.append((slot, slot))

        if sync_mode == defs.SELF_MODE_FULL_MESH and partner > 0:
            partner_edge = int(tt["se_count"][partner]) > 0
            partner_triangle = int(tt["st_count"][partner]) > 0
            if has_edge and partner_edge:
                tt["use_edge"][slot] = True
                tt["use_edge"][partner] = True
                ctx.self_contact_tasks.append(('EE', slot, partner))
            if has_triangle:
                tt["use_triangle"][slot] = True
                tt["use_point"][partner] = True
                ctx.self_contact_tasks.append(('TP', slot, partner))
            if partner_triangle:
                tt["use_point"][slot] = True
                tt["use_triangle"][partner] = True
                ctx.self_contact_tasks.append(('PT', slot, partner))
            if has_edge and partner_triangle:
                ctx.self_intersect_tasks.append((slot, partner))
            if has_triangle and partner_edge:
                ctx.self_intersect_tasks.append((partner, slot))

    ctx.self_use_intersect = len(ctx.self_intersect_tasks) > 0
    if ctx.self_use_intersect:
        ctx.self_intersect_pairs = _detect_intersect(world, ctx)


def step(world, ctx, first_step):
    if not ctx.self_contact_tasks:
        return
    if first_step:
        for kind in (defs.KIND_POINT, defs.KIND_EDGE, defs.KIND_TRIANGLE):
            _update_primitives(world, ctx, kind)
        for kind in (defs.KIND_POINT, defs.KIND_EDGE, defs.KIND_TRIANGLE):
            _update_grid(world, ctx, kind)
        _detect_contacts(world, ctx)
    else:
        _update_contacts(world, ctx, first=False)
    _solve_contacts(world, ctx)


def frame_end(world, ctx):
    if not ctx.self_use_intersect:
        return
    pa = world.particles
    fp = ctx.frame_particle_index
    pa["intersect_flag"][fp] = False
    for edges, triangles in ctx.self_intersect_pairs:
        if len(edges) == 0:
            continue
        p = pa["next_positions"][edges[:, 0]]
        q = pa["next_positions"][edges[:, 1]]
        a = pa["next_positions"][triangles[:, 0]]
        b = pa["next_positions"][triangles[:, 1]]
        c = pa["next_positions"][triangles[:, 2]]

        qp = p - q
        ac = c - a
        ab = b - a
        n = np.cross(ab, ac)
        d = pm.dot(qp, n)
        ok = np.abs(d) >= defs.EPSILON

        flip = d < 0.0
        p2 = np.where(flip[:, None], q, p)
        qp2 = np.where(flip[:, None], -qp, qp)
        d2 = np.abs(d)

        ap = p2 - a
        t = pm.dot(ap, n)
        ok &= (t >= 0.0) & (t <= d2)
        e = np.cross(qp2, ap)
        v = pm.dot(ac, e)
        ok &= (v >= 0.0) & (v <= d2)
        w = -pm.dot(ab, e)
        ok &= (w >= 0.0) & ((v + w) <= d2)

        hit = np.flatnonzero(ok)
        if len(hit):
            pa["intersect_flag"][edges[hit, 0]] = True
            pa["intersect_flag"][edges[hit, 1]] = True


def _update_primitives(world, ctx, kind):
    tt = world.team
    pa = world.particles
    arena = world.prim_arena(kind)
    start_name, count_name, use_name = KIND_CHUNK_FIELDS[kind]
    axes = KIND_AXES[kind]
    team = arena["team"]
    use = ctx.step_team_mask[team] & tt[use_name][team]
    arena["use"][:] = use
    index = np.flatnonzero(use)
    if len(index) == 0:
        _accumulate_max_size(world, ctx, kind, None)
        return
    it = team[index]

    particles = arena["particles"][index].copy()
    particles[particles < 0] = 0
    gathered = particles[:, :axes]
    next_pos = pa["next_positions"][gathered]
    old_pos = pa["old_positions"][gathered]
    friction = pa["friction"][gathered]
    fix = arena["fix"][index][:, :axes]
    inv_mass = pm.calc_self_collision_inverse_mass(friction, fix,
                                                   tt["self_cloth_mass"][it][:, None])
    full_inv = arena["inv_mass"][index]
    full_inv[:, :axes] = inv_mass
    arena["inv_mass"][index] = full_inv

    thickness = pm.evaluate_team_lut(tt["self_thickness_lut"], it, arena["prim_depth"][index]) \
        * tt["scale_ratio"][it]
    arena["thickness"][index] = thickness

    low = np.minimum(next_pos, old_pos).min(axis=1)
    high = np.maximum(next_pos, old_pos).max(axis=1)
    size = (high - low).max(axis=1)
    arena["aabb_min"][index] = low - thickness[:, None]
    arena["aabb_max"][index] = high + thickness[:, None]

    if ctx.self_use_intersect:
        flags = pa["intersect_flag"][gathered]
        full = np.zeros((len(index), 3), dtype=bool)
        full[:, :axes] = flags
        arena["intersect"][index] = full
    else:
        arena["intersect"][index] = False

    valid = ~arena["ignore"][index]
    _accumulate_max_size(world, ctx, kind, (it[valid], size[valid]))


def _accumulate_max_size(world, ctx, kind, data):
    tt = world.team
    if kind == defs.KIND_POINT:
        ctx.self_max_size = np.zeros(len(tt), dtype=np.float32)
    if data is not None:
        team, size = data
        np.maximum.at(ctx.self_max_size, team, size)
    if kind == defs.KIND_TRIANGLE:
        i = ctx.step_team_index
        tt["self_max_primitive_size"][i] = ctx.self_max_size[i]
        tt["self_grid_size"][i] = ctx.self_max_size[i] * defs.SELF_COLLISION_UNIFORM_GRID_SCALE


class _Grid:
    __slots__ = ("valid", "order", "sorted_key", "segment_start", "segment_count",
                 "segment_key", "segment_team", "team_bounds")

    def __init__(self):
        self.valid = False


def _update_grid(world, ctx, kind):
    tt = world.team
    arena = world.prim_arena(kind)
    team = arena["team"]
    grid_size = tt["self_grid_size"][team]
    center = (arena["aabb_min"] + arena["aabb_max"]) * 0.5
    usable = arena["use"] & ~arena["ignore"] & (grid_size > defs.EPSILON)
    cells = np.floor(center / np.where(grid_size > 0, grid_size, 1.0)[:, None]).astype(np.int64)
    key = np.where(usable, pm.pack_grid_cells(cells), defs.GRID_KEY_IGNORE)
    arena["cell_key"][:] = key

    order = np.lexsort((key, team))
    sorted_team = team[order]
    sorted_key = key[order]
    if len(order):
        boundary = np.flatnonzero((np.diff(sorted_team) != 0) | (np.diff(sorted_key) != 0)) + 1
        segment_start = np.concatenate([[0], boundary]).astype(np.int64)
        segment_end = np.concatenate([boundary, [len(order)]]).astype(np.int64)
    else:
        segment_start = np.zeros(0, dtype=np.int64)
        segment_end = np.zeros(0, dtype=np.int64)
    grid = _Grid()
    grid.valid = True
    grid.order = order
    grid.sorted_key = sorted_key
    grid.segment_start = segment_start
    grid.segment_count = segment_end - segment_start
    grid.segment_key = sorted_key[segment_start] if len(segment_start) else np.zeros(0, np.int64)
    grid.segment_team = sorted_team[segment_start] if len(segment_start) else np.zeros(0, np.int32)
    grid.team_bounds = np.searchsorted(grid.segment_team, np.arange(len(tt) + 1))
    world.grids[kind] = grid


def _broad_phase(world, my_arena, my_index, target_team, target_kind, same_object_same_kind,
                 connection_check):
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    grid = world.grids.get(target_kind)
    if grid is None or not getattr(grid, "valid", False):
        return empty
    tt = world.team
    grid_size = float(tt["self_grid_size"][target_team])
    max_size = float(tt["self_max_primitive_size"][target_team])
    if grid_size <= defs.EPSILON:
        return empty
    if len(my_index) == 0:
        return empty

    half = max_size * 0.5
    low = np.floor((my_arena["aabb_min"][my_index] - half) / grid_size).astype(np.int64)
    high = np.floor((my_arena["aabb_max"][my_index] + half) / grid_size).astype(np.int64)
    dims = high - low + 1
    counts = dims.prod(axis=1)
    owner, within = pm.ragged_expand(counts)
    if len(owner) == 0:
        return empty
    dz = dims[owner, 2]
    dy = dims[owner, 1]
    z = within % dz
    rem = within // dz
    y = rem % dy
    x = rem // dy
    cells = low[owner] + np.stack([x, y, z], axis=1)
    qkey = pm.pack_grid_cells(cells)

    seg_lo = int(grid.team_bounds[target_team])
    seg_hi = int(grid.team_bounds[target_team + 1])
    seg_key = grid.segment_key[seg_lo:seg_hi]
    if len(seg_key) == 0:
        return empty
    pos = np.searchsorted(seg_key, qkey)
    hit = (pos < len(seg_key)) & (seg_key[np.minimum(pos, len(seg_key) - 1)] == qkey)
    hit_index = np.flatnonzero(hit)
    if len(hit_index) == 0:
        return empty
    seg = pos[hit_index] + seg_lo
    starts = grid.segment_start[seg]
    counts2 = grid.segment_count[seg]
    owner2, within2 = pm.ragged_expand(counts2)
    my_out = my_index[owner[hit_index][owner2]]
    target_out = grid.order[starts[owner2] + within2]

    if same_object_same_kind:
        keep = my_out < target_out
        my_out = my_out[keep]
        target_out = target_out[keep]
    return my_out, target_out


def _filter_pairs(my_arena, target_arena, my_index, target_index, connection_check):
    if len(my_index) == 0:
        return my_index, target_index
    overlap = ((my_arena["aabb_min"][my_index] <= target_arena["aabb_max"][target_index])
               & (my_arena["aabb_max"][my_index] >= target_arena["aabb_min"][target_index])).all(axis=1)
    keep = overlap
    keep &= ~target_arena["ignore"][target_index]
    keep &= ~(my_arena["all_fix"][my_index] & target_arena["all_fix"][target_index])
    if connection_check:
        my_particles = my_arena["particles"][my_index]
        target_particles = target_arena["particles"][target_index]
        valid = (my_particles[:, :, None] >= 0) & (target_particles[:, None, :] >= 0)
        shared = ((my_particles[:, :, None] == target_particles[:, None, :]) & valid).any(axis=(1, 2))
        keep &= ~shared
    return my_index[keep], target_index[keep]


def _team_prims(world, slot, kind):
    tt = world.team
    start_name, count_name, use_name = KIND_CHUNK_FIELDS[kind]
    start = int(tt[start_name][slot])
    count = int(tt[count_name][slot])
    arena = world.prim_arena(kind)
    index = np.arange(start, start + count, dtype=np.int64)
    valid = arena["use"][index] & ~arena["ignore"][index]
    return index[valid]


def _detect_contacts(world, ctx):
    ee_parts = []
    pt_parts = []
    for kind, my_slot, target_slot in ctx.self_contact_tasks:
        same_object = my_slot == target_slot
        if kind == 'EE':
            my_arena = world.self_edges
            target_arena = world.self_edges
            my_index = _team_prims(world, my_slot, defs.KIND_EDGE)
            my_out, target_out = _broad_phase(world, my_arena, my_index, target_slot,
                                              defs.KIND_EDGE, same_object, same_object)
            my_out, target_out = _filter_pairs(my_arena, target_arena, my_out, target_out,
                                               same_object)
            if len(my_out) == 0:
                continue
            thickness = my_arena["thickness"][my_out] + target_arena["thickness"][target_out]
            ee_parts.append({"my": my_out, "target": target_out, "thickness": thickness})
        elif kind == 'PT':
            my_arena = world.self_points
            target_arena = world.self_triangles
            my_index = _team_prims(world, my_slot, defs.KIND_POINT)
            my_out, target_out = _broad_phase(world, my_arena, my_index, target_slot,
                                              defs.KIND_TRIANGLE, False, same_object)
            my_out, target_out = _filter_pairs(my_arena, target_arena, my_out, target_out,
                                               same_object)
            if len(my_out) == 0:
                continue
            thickness = my_arena["thickness"][my_out] + target_arena["thickness"][target_out]
            pt_parts.append({"my": my_out, "target": target_out, "thickness": thickness})
        else:
            my_arena = world.self_triangles
            target_arena = world.self_points
            my_index = _team_prims(world, my_slot, defs.KIND_TRIANGLE)
            my_out, target_out = _broad_phase(world, my_arena, my_index, target_slot,
                                              defs.KIND_POINT, False, same_object)
            my_out, target_out = _filter_pairs(my_arena, target_arena, my_out, target_out,
                                               same_object)
            if len(my_out) == 0:
                continue
            thickness = world.self_points["thickness"][target_out] \
                + world.self_triangles["thickness"][my_out]
            pt_parts.append({"my": target_out, "target": my_out, "thickness": thickness})

    contacts = {"EE": None, "PT": None}
    if ee_parts:
        table = {
            "my": np.concatenate([p["my"] for p in ee_parts]),
            "target": np.concatenate([p["target"] for p in ee_parts]),
            "thickness": np.concatenate([p["thickness"] for p in ee_parts]).astype(np.float32),
        }
        count = len(table["my"])
        table["enable"] = np.zeros(count, dtype=bool)
        table["s"] = np.zeros(count, dtype=np.float32)
        table["t"] = np.zeros(count, dtype=np.float32)
        table["n"] = np.zeros((count, 3), dtype=np.float32)
        contacts["EE"] = table
    if pt_parts:
        table = {
            "my": np.concatenate([p["my"] for p in pt_parts]),
            "target": np.concatenate([p["target"] for p in pt_parts]),
            "thickness": np.concatenate([p["thickness"] for p in pt_parts]).astype(np.float32),
        }
        count = len(table["my"])
        table["enable"] = np.zeros(count, dtype=bool)
        table["sign"] = np.zeros(count, dtype=np.float32)
        contacts["PT"] = table
    world.contacts = contacts
    _update_contacts(world, ctx, first=True)
    for name in ("EE", "PT"):
        table = world.contacts[name]
        if table is None:
            continue
        keep = np.flatnonzero(table["enable"])
        if len(keep) == 0:
            world.contacts[name] = None
            continue
        for field in list(table.keys()):
            table[field] = table[field][keep]

    world.contacts["SCATTER"] = _build_scatter_plan(world)


def _update_contacts(world, ctx, first):
    pa = world.particles
    ee = world.contacts["EE"]
    if ee is not None and len(ee["my"]):
        scr = ee["thickness"] * defs.SELF_COLLISION_SCR
        e0 = world.self_edges["particles"][ee["my"]][:, :2]
        e1 = world.self_edges["particles"][ee["target"]][:, :2]
        next_a0 = pa["next_positions"][e0[:, 0]]
        next_a1 = pa["next_positions"][e0[:, 1]]
        next_b0 = pa["next_positions"][e1[:, 0]]
        next_b1 = pa["next_positions"][e1[:, 1]]
        old_a0 = pa["old_positions"][e0[:, 0]]
        old_a1 = pa["old_positions"][e0[:, 1]]
        old_b0 = pa["old_positions"][e1[:, 0]]
        old_b1 = pa["old_positions"][e1[:, 1]]

        s, t, c_a, c_b = pm.closest_pt_segment_segment(old_a0, old_a1, old_b0, old_b1)
        clen = pm.length(c_a - c_b)
        ok = clen >= 1e-9
        n = (c_a - c_b) / np.where(clen > 1e-30, clen, 1.0)[:, None]

        da = pm.lerp(next_a0 - old_a0, next_a1 - old_a1, s[:, None])
        db = pm.lerp(next_b0 - old_b0, next_b1 - old_b1, t[:, None])
        l = clen + pm.dot(n, da) - pm.dot(n, db)
        ok &= l <= (ee["thickness"] + scr)

        ee["enable"] = ok
        ee["s"] = s.astype(np.float32)
        ee["t"] = t.astype(np.float32)
        ee["n"] = n.astype(np.float32)

    pt = world.contacts["PT"]
    if pt is not None and len(pt["my"]):
        scr = pt["thickness"] * defs.SELF_COLLISION_SCR
        p = world.self_points["particles"][pt["my"]][:, 0]
        tri = world.self_triangles["particles"][pt["target"]]
        next_a = pa["next_positions"][p]
        old_a = pa["old_positions"][p]
        next_b0 = pa["next_positions"][tri[:, 0]]
        next_b1 = pa["next_positions"][tri[:, 1]]
        next_b2 = pa["next_positions"][tri[:, 2]]
        old_b0 = pa["old_positions"][tri[:, 0]]
        old_b1 = pa["old_positions"][tri[:, 1]]
        old_b2 = pa["old_positions"][tri[:, 2]]

        d_a = next_a - old_a
        d_b0 = next_b0 - old_b0
        d_b1 = next_b1 - old_b1
        d_b2 = next_b2 - old_b2

        cp, uvw = pm.closest_pt_point_triangle(old_a, old_b0, old_b1, old_b2)
        dt = d_b0 * uvw[:, 0:1] + d_b1 * uvw[:, 1:2] + d_b2 * uvw[:, 2:3]
        cv = cp - old_a
        cvlen = pm.length(cv)
        ok = cvlen > defs.EPSILON
        n = cv / np.where(cvlen > 1e-30, cvlen, 1.0)[:, None]
        l = cvlen - pm.dot(n, d_a) + pm.dot(n, dt)
        ok &= l < (pt["thickness"] + scr)

        if first:
            otn = pm.triangle_normal(old_b0, old_b1, old_b2)
            n2 = pm.normalize(old_a - cp)
            d = pm.dot(otn, n2)
            angle_ok = np.abs(d) >= defs.SELF_COLLISION_POINT_TRIANGLE_ANGLE_COS
            ok &= angle_ok
            pt["sign"] = np.sign(d).astype(np.float32)
        pt["enable"] = ok


class _ScatterPlan:
    __slots__ = ("gather", "run_starts", "touched",
                 "edge_my", "edge_target", "inv_my", "inv_target",
                 "point_particle", "triangle_particles", "inv_point", "inv_triangle")

    def __init__(self):
        self.gather = None
        self.run_starts = None
        self.touched = None
        self.edge_my = None
        self.edge_target = None
        self.inv_my = None
        self.inv_target = None
        self.point_particle = None
        self.triangle_particles = None
        self.inv_point = None
        self.inv_triangle = None


def _build_scatter_plan(world):
    ee = world.contacts["EE"]
    pt = world.contacts["PT"]
    if ee is None and pt is None:
        return None
    plan = _ScatterPlan()
    target_blocks = []
    blocked_blocks = []
    if ee is not None:
        edge_my = world.self_edges["particles"][ee["my"]][:, :2]
        edge_target = world.self_edges["particles"][ee["target"]][:, :2]
        plan.edge_my = edge_my
        plan.edge_target = edge_target
        plan.inv_my = world.self_edges["inv_mass"][ee["my"]][:, :2]
        plan.inv_target = world.self_edges["inv_mass"][ee["target"]][:, :2]
        blocked_my = world.self_edges["fix"][ee["my"]][:, :2] \
            | world.self_edges["intersect"][ee["my"]][:, :2]
        blocked_target = world.self_edges["fix"][ee["target"]][:, :2] \
            | world.self_edges["intersect"][ee["target"]][:, :2]
        target_blocks.extend((edge_my[:, 0], edge_my[:, 1], edge_target[:, 0], edge_target[:, 1]))
        blocked_blocks.extend((blocked_my[:, 0], blocked_my[:, 1],
                               blocked_target[:, 0], blocked_target[:, 1]))
    if pt is not None:
        point_particle = world.self_points["particles"][pt["my"]][:, 0]
        triangle_particles = world.self_triangles["particles"][pt["target"]]
        plan.point_particle = point_particle
        plan.triangle_particles = triangle_particles
        plan.inv_point = world.self_points["inv_mass"][pt["my"]][:, 0]
        plan.inv_triangle = world.self_triangles["inv_mass"][pt["target"]]
        blocked_point = world.self_points["fix"][pt["my"]][:, 0] \
            | world.self_points["intersect"][pt["my"]][:, 0]
        blocked_triangle = world.self_triangles["fix"][pt["target"]] \
            | world.self_triangles["intersect"][pt["target"]]
        target_blocks.extend((point_particle, triangle_particles[:, 0],
                              triangle_particles[:, 1], triangle_particles[:, 2]))
        blocked_blocks.extend((blocked_point, blocked_triangle[:, 0],
                               blocked_triangle[:, 1], blocked_triangle[:, 2]))
    targets = np.concatenate(target_blocks)
    blocked = np.concatenate(blocked_blocks)
    canonical_index = np.flatnonzero(~blocked)
    if len(canonical_index) == 0:
        return None
    order = np.argsort(targets[canonical_index], kind="stable")
    gather = canonical_index[order]
    sorted_targets = targets[gather]
    run_starts = np.flatnonzero(np.concatenate(([True], sorted_targets[1:] != sorted_targets[:-1])))
    plan.gather = gather
    plan.run_starts = run_starts
    plan.touched = sorted_targets[run_starts]
    return plan


def _solve_contacts(world, ctx):
    plan = world.contacts.get("SCATTER")
    if plan is None:
        return
    pa = world.particles
    ee = world.contacts["EE"]
    pt = world.contacts["PT"]

    for _ in range(defs.SELF_COLLISION_SOLVER_ITERATION):
        value_blocks = []
        active_blocks = []
        if ee is not None:
            e0 = plan.edge_my
            e1 = plan.edge_target
            s = ee["s"]
            t = ee["t"]
            n = ee["n"]
            thickness = ee["thickness"]
            a = pm.lerp(pa["next_positions"][e0[:, 0]], pa["next_positions"][e0[:, 1]], s[:, None])
            b = pm.lerp(pa["next_positions"][e1[:, 0]], pa["next_positions"][e1[:, 1]], t[:, None])
            l = pm.dot(n, a - b)
            c = thickness - l
            b0 = 1.0 - s
            b1 = s
            b2 = 1.0 - t
            b3 = t
            im = plan.inv_my
            im2 = plan.inv_target
            denominator = (im[:, 0] * b0 * b0 + im[:, 1] * b1 * b1
                           + im2[:, 0] * b2 * b2 + im2[:, 1] * b3 * b3)
            ok_ee = ee["enable"] & (l <= thickness) & (denominator != 0.0)
            safe_denominator = np.where(denominator != 0.0, denominator, 1.0)
            scale = np.where(ok_ee, c / safe_denominator, 0.0)
            corr_a0 = n * (scale * im[:, 0] * b0)[:, None]
            corr_a1 = n * (scale * im[:, 1] * b1)[:, None]
            corr_b0 = -n * (scale * im2[:, 0] * b2)[:, None]
            corr_b1 = -n * (scale * im2[:, 1] * b3)[:, None]
            value_blocks.extend((corr_a0, corr_a1, corr_b0, corr_b1))
            active_blocks.extend((ok_ee, ok_ee, ok_ee, ok_ee))
        if pt is not None:
            point_particle = plan.point_particle
            tri = plan.triangle_particles
            sign = pt["sign"]
            thickness = pt["thickness"]
            next_p = pa["next_positions"][point_particle]
            t0 = pa["next_positions"][tri[:, 0]]
            t1 = pa["next_positions"][tri[:, 1]]
            t2 = pa["next_positions"][tri[:, 2]]
            tn = pm.triangle_normal(t0, t1, t2)
            n = tn * sign[:, None]
            dist = pm.dot(n, next_p - t0)
            _, uvw = pm.closest_pt_point_triangle(next_p, t0, t1, t2)
            c = dist - thickness
            im_p = plan.inv_point
            im_t = plan.inv_triangle
            denominator = im_p + (im_t[:, 0] * uvw[:, 0] ** 2 + im_t[:, 1] * uvw[:, 1] ** 2
                                  + im_t[:, 2] * uvw[:, 2] ** 2)
            ok_pt = pt["enable"] & (dist < thickness) & (denominator != 0.0)
            safe_denominator = np.where(denominator != 0.0, denominator, 1.0)
            scale = np.where(ok_pt, c / safe_denominator, 0.0)
            corr_p = -n * (scale * im_p)[:, None]
            corr_t0 = n * (scale * im_t[:, 0] * uvw[:, 0])[:, None]
            corr_t1 = n * (scale * im_t[:, 1] * uvw[:, 1])[:, None]
            corr_t2 = n * (scale * im_t[:, 2] * uvw[:, 2])[:, None]
            value_blocks.extend((corr_p, corr_t0, corr_t1, corr_t2))
            active_blocks.extend((ok_pt, ok_pt, ok_pt, ok_pt))

        canonical_values = np.concatenate(value_blocks).astype(np.float64)
        canonical_active = np.concatenate(active_blocks)
        gathered_active = canonical_active[plan.gather].astype(np.float64)
        rows = canonical_values[plan.gather] * gathered_active[:, None]
        counts = np.add.reduceat(gathered_active, plan.run_starts)
        sums = np.add.reduceat(rows, plan.run_starts, axis=0)
        apply = counts > 0
        if np.any(apply):
            pa["next_positions"][plan.touched[apply]] += (sums[apply]
                                                          / counts[apply][:, None]).astype(np.float32)


def _detect_intersect(world, ctx):
    tt = world.team
    pairs = []
    for edge_slot, tri_slot in ctx.self_intersect_tasks:
        grid = world.grids.get(defs.KIND_TRIANGLE)
        if grid is None or not getattr(grid, "valid", False):
            continue
        if tt["self_grid_size"][tri_slot] <= defs.EPSILON \
                or tt["self_max_primitive_size"][tri_slot] <= defs.EPSILON:
            continue
        edge_arena = world.self_edges
        start = int(tt["se_start"][edge_slot])
        count = int(tt["se_count"][edge_slot])
        my_index = np.arange(start, start + count, dtype=np.int64)
        my_index = my_index[~edge_arena["ignore"][my_index]]
        my_out, target_out = _broad_phase(world, edge_arena, my_index, tri_slot,
                                          defs.KIND_TRIANGLE, False,
                                          edge_slot == tri_slot)
        if len(my_out) == 0:
            continue
        local = my_out - start
        sliced = (local % defs.SELF_COLLISION_INTERSECT_DIV) == ctx.intersect_frame_index
        my_out = my_out[sliced]
        target_out = target_out[sliced]
        my_out, target_out = _filter_pairs(edge_arena, world.self_triangles, my_out, target_out,
                                           edge_slot == tri_slot)
        if len(my_out) == 0:
            continue
        pairs.append((edge_arena["particles"][my_out][:, :2],
                      world.self_triangles["particles"][target_out]))
    return pairs
