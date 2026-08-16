import math

import numpy as np

from . import defs
from . import math as pm

ATTR_FIXED = defs.ATTR_FIXED
ATTR_MOVE = defs.ATTR_MOVE

_cache = {}


def _is_move(attr):
    return (attr & ATTR_MOVE) != 0


def _is_dont_move(attr):
    return (attr & ATTR_MOVE) == 0


def _is_invalid(attr):
    return (attr & (ATTR_MOVE | ATTR_FIXED)) == 0


class TopologySnapshot:
    def __init__(self):
        self.token = None
        self.bone_names = []
        self.parent_index = None
        self.external_parent = []
        self.rest_world = None
        self.rest_relative = None
        self.matrix_world = None
        self.is_spring = False
        self.connection_mode = 'LINE'
        self.root_names = []
        self.overrides = []
        self.collision_bones = []
        self.skinning = []
        self.normal_alignment_mode = 'NONE'
        self.normal_alignment_center_local = None
        self.gravity_direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.wind_seed = 1


class SetupTables:
    def __init__(self):
        self.valid = False
        self.error = ""
        self.is_spring = False
        self.wind_seed = 1
        self.bone_names = []
        self.index_map = {}

        self.kin_parent = None
        self.kin_rest_relative = None
        self.kin_external_parent = []
        self.kin_levels = []

        self.attributes = None
        self.local_positions = None
        self.local_normals = None
        self.local_tangents = None
        self.uv = None

        self.lines = None
        self.triangles = None
        self.edges = None

        self.vertex_parent = None
        self.vertex_root = None
        self.vertex_depth = None
        self.vertex_local_positions = None
        self.vertex_local_rotations = None
        self.vertex_child_start = None
        self.vertex_child_count = None
        self.vertex_child_data = None
        self.vertex_bind_pose_rotations = None
        self.vertex_to_transform_rotations = None
        self.normal_adjustment_rotations = None

        self.baseline_flags = None
        self.baseline_start = None
        self.baseline_count = None
        self.baseline_data = None

        self.fk_levels = []
        self.angle_passes = []
        self.postline_levels = []
        self.angle_buffered = None

        self.v2t_owner = None
        self.v2t_triangle = None
        self.v2t_flip_normal = None
        self.v2t_flip_tangent = None

        self.center_fixed_list = None
        self.local_center_position = np.zeros(3, dtype=np.float32)
        self.init_local_gravity_direction = np.array([0, 0, -1], dtype=np.float32)

        self.init_matrix_world = np.eye(4)
        self.init_world_to_local = np.eye(4)
        self.init_rotation = np.array([0, 0, 0, 1], dtype=np.float64)
        self.init_scale = np.ones(3)

        self.distance_particle = None
        self.distance_target = None
        self.distance_rest = None

        self.bending_pairs = None
        self.bending_rest = None
        self.bending_sign = None

        self.transform_names = []
        self.transform_bind_pose = None
        self.skin_indices = None
        self.skin_weights = None

        self.tether_index = None
        self.motion_index = None
        self.update_move_index = None
        self.update_fixed_index = None
        self.spring_index = None
        self.collision_process_index = None
        self.collision_edge_index = None

        self.average_vertex_distance = 0.01
        self.max_vertex_distance = 0.01


def _matrix_to_quat(m):
    r = np.array(m, dtype=np.float64)[:3, :3]
    for i in range(3):
        l = np.linalg.norm(r[:, i])
        if l > 1e-30:
            r[:, i] /= l
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.array([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s,
                         (r[1, 0] - r[0, 1]) / s, 0.25 * s], dtype=np.float64)
    if r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
        s = np.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 0.0)) * 2.0
        return np.array([0.25 * s, (r[0, 1] + r[1, 0]) / s,
                         (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s], dtype=np.float64)
    if r[1, 1] >= r[2, 2]:
        s = np.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 0.0)) * 2.0
        return np.array([(r[0, 1] + r[1, 0]) / s, 0.25 * s,
                         (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s], dtype=np.float64)
    s = np.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 0.0)) * 2.0
    return np.array([(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s,
                     0.25 * s, (r[1, 0] - r[0, 1]) / s], dtype=np.float64)


def _matrix_scale(m):
    r = np.asarray(m, dtype=np.float64)
    return np.array([np.linalg.norm(r[:3, 0]), np.linalg.norm(r[:3, 1]),
                     np.linalg.norm(r[:3, 2])], dtype=np.float64)


def _build_line_connection(parent_index):
    lines = []
    for i, p in enumerate(parent_index):
        if p >= 0:
            lines.append((min(p, i), max(p, i)))
    return lines, []


def _build_mesh_connection(connection_mode, local_positions, parent_index, children_of,
                           root_indices_in_setup):
    n = len(parent_index)
    automatic = connection_mode == 'AUTOMATIC_MESH'
    loop_connection = connection_mode == 'SEQUENTIAL_LOOP_MESH'
    sequential = connection_mode in {'SEQUENTIAL_LOOP_MESH', 'SEQUENTIAL_NON_LOOP_MESH'}

    root_list = list(root_indices_in_setup)

    if automatic and len(root_list) > 1:
        temp = list(root_list)
        ordered = [temp[0]]
        last_dist = 0.0
        while len(temp) > 0:
            current = ordered[-1]
            if current in temp:
                temp.remove(current)
            pos = local_positions[current]
            min_dist = float("inf")
            min_root = -1
            for candidate in temp:
                d = float(np.linalg.norm(local_positions[candidate] - pos))
                if d < min_dist:
                    min_dist = d
                    min_root = candidate
            if min_root >= 0:
                if last_dist == 0.0 or min_dist < last_dist * 1.5:
                    ordered.append(min_root)
                    last_dist = min_dist if last_dist == 0.0 else (last_dist + min_dist) * 0.5
                else:
                    ordered.reverse()
                    last_dist = 0.0
        root_list = ordered

        if len(root_list) >= 3:
            d = float(np.linalg.norm(local_positions[root_list[0]] - local_positions[root_list[-1]]))
            if d < last_dist * 1.5:
                loop_connection = True

    root_count = len(root_list)

    vertex_level = np.full(n, -1, dtype=np.int32)
    vertex_root_line = np.full(n, -1, dtype=np.int32)
    link_lists = [[] for _ in range(n)]
    main_edge_set = set()

    for line_index, root in enumerate(root_list):
        stack = [(root, 0)]
        while stack:
            v, lv = stack.pop()
            vertex_level[v] = lv
            vertex_root_line[v] = line_index
            p = parent_index[v]
            if p >= 0:
                link_lists[v].append(p)
                main_edge_set.add((min(v, p), max(v, p)))
            for c in children_of[v]:
                stack.append((c, lv + 1))
                link_lists[v].append(c)
                main_edge_set.add((min(v, c), max(v, c)))

    max_level = int(vertex_level.max()) if n > 0 else -1
    level_members = [[] for _ in range(max_level + 1)]
    for v in range(n):
        if vertex_level[v] >= 0:
            level_members[vertex_level[v]].append(v)

    first_last_pair = (0, root_count - 1) if root_count > 1 else None

    for v in range(n):
        lv = vertex_level[v]
        if lv < 0:
            continue
        members = level_members[lv]
        pos = local_positions[v]
        my_line = vertex_root_line[v]
        link = link_lists[v]

        def _line_skip(other):
            other_line = vertex_root_line[other]
            pair = (min(my_line, other_line), max(my_line, other_line))
            is_first_last = first_last_pair is not None and pair == first_last_pair
            if not loop_connection and is_first_last:
                return True
            if sequential and not (loop_connection and is_first_last):
                if abs(my_line - other_line) > 1:
                    return True
            return False

        first_dist = float("inf")
        first_index = -1
        for other in members:
            if other == v or _line_skip(other):
                continue
            d = float(np.linalg.norm(local_positions[other] - pos))
            if d < first_dist:
                first_dist = d
                first_index = other
        if first_index >= 0:
            link.append(first_index)
            limit = float("inf") if sequential else first_dist * 1.5
            for other in members:
                if other == v or other == first_index or _line_skip(other):
                    continue
                d = float(np.linalg.norm(local_positions[other] - pos))
                if d <= limit:
                    link.append(other)

    edge_set = set()
    triangle_edge_set = set()
    triangle_set = set()
    max_angle = math.radians(defs.PROXY_MESH_BONE_TRIANGLE_ANGLE)
    for v in range(n):
        link = link_lists[v]
        if len(link) == 0:
            continue
        if len(link) == 1:
            edge_set.add((min(v, link[0]), max(v, link[0])))
            continue
        for other in link:
            edge_set.add((min(v, other), max(v, other)))

        my_line = vertex_root_line[v]
        pos = local_positions[v]
        for i in range(len(link) - 1):
            v1 = link[i]
            d1 = local_positions[v1] - pos
            for j in range(i + 1, len(link)):
                v2 = link[j]
                d2 = local_positions[v2] - pos
                if float(np.dot(d1, d1)) < 1e-6 or float(np.dot(d2, d2)) < 1e-6:
                    continue
                cosine = float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
                angle = math.acos(max(-1.0, min(1.0, cosine)))
                if angle >= max_angle:
                    continue
                line1 = vertex_root_line[v1]
                line2 = vertex_root_line[v2]
                if line1 != my_line and line2 != my_line and line1 != line2:
                    continue
                main_count = 0
                main_count += 1 if (min(v, v1), max(v, v1)) in main_edge_set else 0
                main_count += 1 if (min(v, v2), max(v, v2)) in main_edge_set else 0
                main_count += 1 if (min(v1, v2), max(v1, v2)) in main_edge_set else 0
                if main_count == 0:
                    continue
                tri = tuple(sorted((v, v1, v2)))
                if tri not in triangle_set:
                    triangle_set.add(tri)
                    triangle_edge_set.add((min(v, v1), max(v, v1)))
                    triangle_edge_set.add((min(v, v2), max(v, v2)))

    lines = sorted(edge_set - triangle_edge_set)
    triangles = sorted(triangle_set)
    return lines, triangles


def _remaining_vertex(tri, edge):
    for v in tri:
        if v != edge[0] and v != edge[1]:
            return v
    return -1


def _triangle_pair_angle(p0, p1, p2, p3):
    va = p1 - p0
    vb = p2 - p0
    vc = p3 - p0
    n0 = np.cross(va, vb)
    n1 = np.cross(vc, va)
    l0 = np.linalg.norm(n0)
    l1 = np.linalg.norm(n1)
    if l0 * l1 < 1e-30:
        return 0.0
    cosine = max(-1.0, min(1.0, float(np.dot(n0, n1)) / (l0 * l1)))
    return math.acos(cosine)


def _remove_duplicate_triangles(triangles, local_positions):
    if len(triangles) < 2:
        return triangles
    edge_to_triangles = {}
    for i, tri in enumerate(triangles):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (min(a, b), max(a, b))
            edge_to_triangles.setdefault(edge, []).append(i)

    use_quad = set()
    remove = set()
    max_angle = defs.PROXY_MESH_BONE_TRIANGLE_ANGLE / 6.0
    for edge, tri_list in edge_to_triangles.items():
        if len(tri_list) < 2:
            continue
        px = local_positions[edge[0]]
        py = local_positions[edge[1]]
        for i in range(len(tri_list) - 1):
            tri1 = triangles[tri_list[i]]
            z = _remaining_vertex(tri1, edge)
            pz = local_positions[z]
            for j in range(i + 1, len(tri_list)):
                tri2 = triangles[tri_list[j]]
                w = _remaining_vertex(tri2, edge)
                pw = local_positions[w]

                angle = _triangle_pair_angle(px, py, pz, pw)
                if abs(math.degrees(angle)) > max_angle:
                    continue
                s, t, _, _ = pm.closest_pt_segment_segment(
                    px[None], py[None], pz[None], pw[None])
                s = float(s[0])
                t = float(t[0])
                if s == 0.0 or s == 1.0 or t == 0.0 or t == 1.0:
                    continue
                quad = tuple(sorted((edge[0], edge[1], z, w)))
                if quad in use_quad:
                    remove.add(tri1)
                    remove.add(tri2)
                else:
                    use_quad.add(quad)

    return [tri for tri in triangles if tri not in remove]


def _optimize_triangle_direction(triangles, triangle_normals, local_positions, edge_to_triangles):
    t_count = len(triangles)
    if t_count == 0:
        return
    used = set()
    same_angle = defs.SAME_SURFACE_ANGLE
    start = 0
    while start < t_count:
        if start in used:
            start += 1
            continue
        used.add(start)
        queue = [start]
        layer = []
        open_count = 0
        close_count = 0
        while queue:
            t_index = queue.pop(0)
            n = triangle_normals[t_index]
            tri = triangles[t_index]
            layer.append(t_index)
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge = (min(a, b), max(a, b))
                for t_index2 in edge_to_triangles.get(edge, ()):
                    if t_index2 in used:
                        continue
                    tri2 = triangles[t_index2]
                    sv1 = _remaining_vertex(tri, edge)
                    sv2 = _remaining_vertex(tri2, edge)
                    va = local_positions[edge[1]] - local_positions[edge[0]]
                    vb = local_positions[sv1] - local_positions[edge[0]]
                    vc = local_positions[sv2] - local_positions[edge[0]]
                    n0 = np.cross(va, vb)
                    n1 = np.cross(vc, va)
                    l0 = np.linalg.norm(n0)
                    l1 = np.linalg.norm(n1)
                    if l0 * l1 < 1e-30:
                        angle = 0.0
                    else:
                        cosine = max(-1.0, min(1.0, float(np.dot(n0, n1)) / (l0 * l1)))
                        angle = math.degrees(math.acos(cosine))
                    if angle > same_angle:
                        continue
                    n2 = triangle_normals[t_index2]
                    if float(np.dot(n, n2)) < 0.0:
                        triangles[t_index2] = (tri2[0], tri2[2], tri2[1])
                        triangle_normals[t_index2] = -n2
                    sv0 = _remaining_vertex(triangles[t_index2], edge)
                    v = local_positions[sv0] - local_positions[edge[0]]
                    lv = np.linalg.norm(v)
                    if lv > 1e-30:
                        v = v / lv
                    if float(np.dot(n, v)) <= 0.0:
                        open_count += 1
                    else:
                        close_count += 1
                    used.add(t_index2)
                    queue.append(t_index2)
        if close_count > open_count:
            for t_index in layer:
                tri = triangles[t_index]
                triangles[t_index] = (tri[0], tri[2], tri[1])
                triangle_normals[t_index] = -triangle_normals[t_index]
        start += 1


def _triangle_tangent(p0, p1, p2, uv0, uv1, uv2):
    dist_ba = p1 - p0
    dist_ca = p2 - p0
    t_ba = uv1 - uv0
    t_ca = uv2 - uv0
    area = t_ba[0] * t_ca[1] - t_ba[1] * t_ca[0]
    if area == 0.0:
        area = 1.0
    delta = 1.0 / area
    tan = np.array([
        dist_ba[0] * t_ca[1] + dist_ca[0] * -t_ba[1],
        dist_ba[1] * t_ca[1] + dist_ca[1] * -t_ba[1],
        dist_ba[2] * t_ca[1] + dist_ca[2] * -t_ba[1],
    ]) * delta
    tan = -tan
    l = np.linalg.norm(tan)
    return tan / l if l > 1e-30 else np.zeros(3)


def build_setup(snapshot):
    cached = _cache.get(snapshot.token)
    if cached is not None:
        return cached
    setup = _build(snapshot)
    if snapshot.token is not None:
        _cache[snapshot.token] = setup
    return setup


def _build(snapshot):
    setup = SetupTables()
    setup.is_spring = snapshot.is_spring
    setup.wind_seed = snapshot.wind_seed

    bone_names = snapshot.bone_names
    if not bone_names:
        setup.error = "未指定有效的根骨骼"
        return setup

    setup.bone_names = list(bone_names)
    n = len(bone_names)
    index_map = {name: i for i, name in enumerate(bone_names)}
    setup.index_map = index_map

    parent_index = np.asarray(snapshot.parent_index, dtype=np.int32)
    setup.kin_parent = parent_index
    setup.kin_rest_relative = np.asarray(snapshot.rest_relative, dtype=np.float64)
    setup.kin_external_parent = list(snapshot.external_parent)

    depth = np.zeros(n, dtype=np.int32)
    for i in range(n):
        p = parent_index[i]
        depth[i] = depth[p] + 1 if p >= 0 else 0
    max_depth = int(depth.max()) if n else 0
    setup.kin_levels = [np.flatnonzero(depth == d).astype(np.int32) for d in range(max_depth + 1)]

    children_of = [[] for _ in range(n)]
    for i in range(n):
        p = parent_index[i]
        if p >= 0:
            children_of[p].append(i)

    matrix_world = np.asarray(snapshot.matrix_world, dtype=np.float64)
    setup.init_matrix_world = matrix_world
    setup.init_world_to_local = np.linalg.inv(matrix_world)
    setup.init_rotation = _matrix_to_quat(matrix_world)
    setup.init_scale = _matrix_scale(matrix_world)

    rest_world = np.asarray(snapshot.rest_world, dtype=np.float64)
    world_positions = rest_world[:, :3, 3]
    world_rotations = np.empty((n, 4))
    for i in range(n):
        world_rotations[i] = _matrix_to_quat(rest_world[i])
    world_rotations32 = world_rotations.astype(np.float32)

    w2l = setup.init_world_to_local
    local_positions = (np.einsum('ij,nj->ni', w2l[:3, :3], world_positions) + w2l[:3, 3]).astype(np.float32)
    world_normals = pm.quat_rotate(world_rotations32, np.broadcast_to(pm.VEC_UP, (n, 3)).astype(np.float32))
    world_tangents = pm.quat_rotate(world_rotations32, np.broadcast_to(pm.VEC_FORWARD, (n, 3)).astype(np.float32))
    local_normals = pm.normalize(np.einsum('ij,nj->ni', w2l[:3, :3].astype(np.float32), world_normals))
    local_tangents = pm.normalize(np.einsum('ij,nj->ni', w2l[:3, :3].astype(np.float32), world_tangents))

    setup.local_positions = local_positions
    setup.local_normals = local_normals
    setup.local_tangents = local_tangents

    transform_names = list(bone_names)
    bind_pose = np.empty((n, 4, 4))
    for i in range(n):
        bind_pose[i] = np.linalg.inv(rest_world[i]) @ matrix_world

    skin_indices = np.zeros((n, 4), dtype=np.int32)
    skin_weights = np.zeros((n, 4), dtype=np.float32)
    skin_indices[:, 0] = np.arange(n, dtype=np.int32)
    skin_weights[:, 0] = 1.0

    import_attr = np.zeros(n, dtype=np.uint8)
    if setup.is_spring:
        import_attr[:] = defs.ATTR_DISABLE_COLLISION
        for name in snapshot.collision_bones:
            i = index_map.get(name, -1)
            if i >= 0:
                import_attr[i] = 0

    root_names = [name for name in snapshot.root_names if name in index_map]
    selection_attr = np.full(n, ATTR_MOVE, dtype=np.uint8)
    for name in root_names:
        selection_attr[index_map[name]] = ATTR_FIXED
    for bone, attribute, disable_collision, exclude_motion in snapshot.overrides:
        i = index_map.get(bone, -1)
        if i < 0:
            continue
        if attribute == 'FIXED':
            value = ATTR_FIXED
        elif attribute == 'MOVE':
            value = ATTR_MOVE
        else:
            value = 0
        if disable_collision:
            value |= defs.ATTR_DISABLE_COLLISION
        if exclude_motion:
            value |= defs.ATTR_INVALID_MOTION
        selection_attr[i] = value

    attributes = import_attr | selection_attr
    setup.attributes = attributes

    if setup.is_spring or snapshot.connection_mode == 'LINE':
        lines, triangles = _build_line_connection(parent_index)
    else:
        root_indices = [index_map[name] for name in root_names]
        lines, triangles = _build_mesh_connection(snapshot.connection_mode, local_positions,
                                                  parent_index, children_of, root_indices)
        triangles = _remove_duplicate_triangles(triangles, local_positions)

    vertex_to_vertex = [set() for _ in range(n)]
    edge_set = set()
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            vertex_to_vertex[a].add(b)
            vertex_to_vertex[b].add(a)
            edge_set.add((min(a, b), max(a, b)))
    for a, b in lines:
        vertex_to_vertex[a].add(b)
        vertex_to_vertex[b].add(a)
        edge_set.add((min(a, b), max(a, b)))

    edges = sorted(edge_set)
    setup.lines = np.array(lines, dtype=np.int32).reshape(-1, 2)
    setup.edges = np.array(edges, dtype=np.int32).reshape(-1, 2)

    if edges:
        edge_array = np.array(edges, dtype=np.int64)
        edge_lengths = np.linalg.norm(
            local_positions[edge_array[:, 0]] - local_positions[edge_array[:, 1]], axis=1)
        setup.average_vertex_distance = float(edge_lengths.mean())
        setup.max_vertex_distance = float(edge_lengths.max())

    triangles = [tuple(t) for t in triangles]
    uv = np.zeros((n, 2), dtype=np.float32)
    vertex_to_triangles = [[] for _ in range(n)]
    if triangles:
        edge_to_triangles = {}
        for t_index, tri in enumerate(triangles):
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge_to_triangles.setdefault((min(a, b), max(a, b)), []).append(t_index)

        tri_normals = [np.asarray(pm.triangle_normal(
            local_positions[t[0]][None], local_positions[t[1]][None], local_positions[t[2]][None]))[0]
            for t in triangles]
        _optimize_triangle_direction(triangles, tri_normals, local_positions, edge_to_triangles)

        aabb_center = (local_positions.min(axis=0) + local_positions.max(axis=0)) * 0.5
        lv = local_positions - aabb_center
        lv = pm.normalize(lv, fallback=np.broadcast_to(pm.VEC_UP, lv.shape))
        u = np.arctan2(lv[:, 0], lv[:, 2])
        u = np.clip((u + np.pi) / (2.0 * np.pi), 0.0, 1.0)
        v = np.clip((1.0 - lv[:, 1]) / 2.0, 0.0, 1.0)
        uv[:, 0] = v
        uv[:, 1] = u

        tri_tangents = [
            _triangle_tangent(local_positions[t[0]], local_positions[t[1]], local_positions[t[2]],
                              uv[t[0]], uv[t[1]], uv[t[2]])
            for t in triangles]

        for t_index, tri in enumerate(triangles):
            for v_index in tri:
                if len(vertex_to_triangles[v_index]) < 7:
                    vertex_to_triangles[v_index].append(t_index)

        for v_index in range(n):
            t_set = vertex_to_triangles[v_index]
            if not t_set:
                continue
            attributes[v_index] |= defs.ATTR_TRIANGLE
            final_normal = np.zeros(3)
            final_tangent = np.zeros(3)
            for t_index in t_set:
                final_normal = final_normal + tri_normals[t_index]
                final_tangent = final_tangent + tri_tangents[t_index]

            if np.linalg.norm(final_normal) < 0.5:
                best = -1.0
                pick = np.zeros(3)
                for t1 in t_set:
                    candidate = np.zeros(3)
                    n1 = tri_normals[t1]
                    for t2 in t_set:
                        if t2 == t1:
                            continue
                        n2 = tri_normals[t2]
                        candidate = candidate + (n2 if float(np.dot(n1, n2)) >= 0.0 else -n2)
                    d = float(np.dot(candidate, candidate))
                    if d > best:
                        best = d
                        pick = n1
                final_normal = pick
            else:
                final_normal = final_normal / np.linalg.norm(final_normal)

            if np.linalg.norm(final_tangent) < 0.5:
                best = -1.0
                pick = np.zeros(3)
                for t1 in t_set:
                    candidate = np.zeros(3)
                    tt1 = tri_tangents[t1]
                    for t2 in t_set:
                        if t2 == t1:
                            continue
                        tt2 = tri_tangents[t2]
                        candidate = candidate + (tt2 if float(np.dot(tt1, tt2)) >= 0.0 else -tt2)
                    d = float(np.dot(candidate, candidate))
                    if d > best:
                        best = d
                        pick = tt1
                final_tangent = pick
            else:
                final_tangent = final_tangent / np.linalg.norm(final_tangent)

            entries = []
            nor_sum = np.zeros(3)
            tan_sum = np.zeros(3)
            for t_index in t_set:
                flip = 0
                if float(np.dot(final_normal, tri_normals[t_index])) < 0.0:
                    flip |= 0x1
                if float(np.dot(final_tangent, tri_tangents[t_index])) < 0.0:
                    flip |= 0x2
                entries.append((t_index, flip))
                nor_sum = nor_sum + tri_normals[t_index] * (-1.0 if flip & 0x1 else 1.0)
                tan_sum = tan_sum + tri_tangents[t_index] * (-1.0 if flip & 0x2 else 1.0)
            vertex_to_triangles[v_index] = entries

            l = np.linalg.norm(nor_sum)
            if l > 1e-30:
                nor = nor_sum / l
                binormal = np.cross(nor, tan_sum)
                bl = np.linalg.norm(binormal)
                if bl > 1e-30:
                    local_normals[v_index] = nor.astype(np.float32)
                    local_tangents[v_index] = (binormal / bl).astype(np.float32)

    setup.triangles = np.array([list(t) for t in triangles], dtype=np.int32).reshape(-1, 3)
    setup.uv = uv

    v2t_owner = []
    v2t_triangle = []
    v2t_flip_normal = []
    v2t_flip_tangent = []
    for v_index in range(n):
        for t_index, flip in vertex_to_triangles[v_index]:
            v2t_owner.append(v_index)
            v2t_triangle.append(t_index)
            v2t_flip_normal.append(-1.0 if flip & 0x1 else 1.0)
            v2t_flip_tangent.append(-1.0 if flip & 0x2 else 1.0)
    setup.v2t_owner = np.array(v2t_owner, dtype=np.int32)
    setup.v2t_triangle = np.array(v2t_triangle, dtype=np.int32)
    setup.v2t_flip_normal = np.array(v2t_flip_normal, dtype=np.float32)
    setup.v2t_flip_tangent = np.array(v2t_flip_tangent, dtype=np.float32)

    fixed_list = []
    center_sum = np.zeros(3)
    for i in range(n):
        if _is_move(attributes[i]):
            continue
        connections = vertex_to_vertex[i]
        if connections:
            if not any(_is_move(attributes[t]) for t in connections):
                continue
        fixed_list.append(i)
        center_sum = center_sum + local_positions[i]
    setup.center_fixed_list = np.array(fixed_list, dtype=np.int32)
    if fixed_list:
        setup.local_center_position = (center_sum / len(fixed_list)).astype(np.float32)

    baseline_flags = []
    baseline_start = []
    baseline_count = []
    baseline_data = []
    for name in root_names:
        root_stack = [index_map[name]]
        while root_stack:
            anchor = root_stack.pop()
            if _is_move(attributes[anchor]):
                continue
            has_move_child = any(_is_move(attributes[c]) for c in children_of[anchor])
            if not has_move_child:
                for c in children_of[anchor]:
                    if _is_dont_move(attributes[c]):
                        root_stack.append(c)
                continue
            stack = [anchor]
            start = len(baseline_data)
            count = 0
            include_line = False
            while stack:
                v = stack.pop()
                baseline_data.append(v)
                count += 1
                if not (attributes[v] & defs.ATTR_TRIANGLE):
                    include_line = True
                for c in children_of[v]:
                    if _is_dont_move(attributes[c]):
                        continue
                    stack.append(c)
            baseline_flags.append(1 if include_line else 0)
            baseline_start.append(start)
            baseline_count.append(count)

    setup.baseline_flags = np.array(baseline_flags, dtype=np.uint8)
    setup.baseline_start = np.array(baseline_start, dtype=np.int32)
    setup.baseline_count = np.array(baseline_count, dtype=np.int32)
    setup.baseline_data = np.array(baseline_data, dtype=np.int32)

    setup.vertex_parent = parent_index.astype(np.int32)
    child_start = np.zeros(n, dtype=np.int32)
    child_count = np.zeros(n, dtype=np.int32)
    child_data = []
    for i in range(n):
        child_start[i] = len(child_data)
        child_count[i] = len(children_of[i])
        child_data.extend(children_of[i])
    setup.vertex_child_start = child_start
    setup.vertex_child_count = child_count
    setup.vertex_child_data = np.array(child_data, dtype=np.int32)

    normal_adjustment = pm.quat_identity((n,))
    if not setup.is_spring and snapshot.normal_alignment_mode != 'NONE':
        if snapshot.normal_alignment_mode == 'BOUNDING_BOX_CENTER':
            center = (local_positions.min(axis=0) + local_positions.max(axis=0)) * 0.5
        else:
            center = np.zeros(3, dtype=np.float32)
            if snapshot.normal_alignment_center_local is not None:
                center = np.asarray(snapshot.normal_alignment_center_local, dtype=np.float32)
        for i in range(n):
            v = local_positions[i] - center
            if np.linalg.norm(v) < defs.EPSILON:
                continue
            nor = v / np.linalg.norm(v)
            ln = local_normals[i].astype(np.float64)
            lt = local_tangents[i].astype(np.float64)
            old_rot = pm.to_rotation(ln[None].astype(np.float32), lt[None].astype(np.float32))[0]
            d = float(np.dot(nor, lt))
            if d < 0.99:
                bn = np.cross(nor, lt)
                bn = bn / np.linalg.norm(bn)
                lt = np.cross(bn, nor)
                lt = lt / np.linalg.norm(lt)
            else:
                lbn = np.cross(ln, lt)
                lbn = lbn / np.linalg.norm(lbn)
                lt = np.cross(lbn, nor)
                lt = lt / np.linalg.norm(lt)
            local_normals[i] = nor.astype(np.float32)
            local_tangents[i] = lt.astype(np.float32)
            new_rot = pm.to_rotation(nor[None].astype(np.float32), lt[None].astype(np.float32))[0]
            normal_adjustment[i] = pm.quat_mul(pm.quat_inverse(old_rot[None]), new_rot[None])[0]
    setup.normal_adjustment_rotations = normal_adjustment.astype(np.float32)

    init_inverse_rotation = pm.quat_inverse(setup.init_rotation.astype(np.float32)[None])[0]
    vertex_rotations = pm.to_rotation(local_normals, local_tangents)
    transform_local_rotations = pm.quat_mul(
        np.broadcast_to(init_inverse_rotation, (n, 4)).astype(np.float32), world_rotations32)
    setup.vertex_to_transform_rotations = pm.quat_mul(pm.quat_inverse(vertex_rotations),
                                                      transform_local_rotations).astype(np.float32)
    setup.vertex_bind_pose_rotations = pm.quat_inverse(vertex_rotations).astype(np.float32)

    vertex_local_positions = np.zeros((n, 3), dtype=np.float32)
    vertex_local_rotations = pm.quat_identity((n,))
    for v in setup.baseline_data:
        p = parent_index[v]
        if p < 0:
            continue
        p_rot = vertex_rotations[p]
        inv_p = pm.quat_inverse(p_rot[None])[0]
        lpos = pm.quat_rotate(inv_p[None], (local_positions[v] - local_positions[p])[None])[0]
        lrot = pm.quat_mul(inv_p[None], vertex_rotations[v][None])[0]
        vertex_local_positions[v] = lpos
        vertex_local_rotations[v] = lrot
        if np.linalg.norm(lpos) < 1e-8:
            attributes[v] |= defs.ATTR_ZERO_DISTANCE
    setup.vertex_local_positions = vertex_local_positions
    setup.vertex_local_rotations = vertex_local_rotations.astype(np.float32)

    vertex_root = np.full(n, -1, dtype=np.int32)
    root_length = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if not _is_move(attributes[i]):
            continue
        c = i
        p = parent_index[c]
        total = 0.0
        root_of = -1
        while p >= 0:
            total += float(np.linalg.norm(local_positions[c] - local_positions[p]))
            root_of = p
            if not _is_move(attributes[p]):
                break
            c = p
            p = parent_index[c]
        vertex_root[i] = root_of
        root_length[i] = total
    max_length = float(root_length.max()) if n > 0 else 0.0
    vertex_depth = np.zeros(n, dtype=np.float32)
    if max_length > defs.EPSILON:
        vertex_depth = np.clip(root_length / max_length, 0.0, 1.0).astype(np.float32)
    setup.vertex_root = vertex_root
    setup.vertex_depth = vertex_depth

    gravity_direction = np.array(snapshot.gravity_direction, dtype=np.float32)
    if np.linalg.norm(gravity_direction) > defs.EPSILON:
        gravity_direction = gravity_direction / np.linalg.norm(gravity_direction)
    else:
        gravity_direction = np.array([0, 0, -1], dtype=np.float32)
    if len(fixed_list) > 0:
        nor_sum = np.zeros(3, dtype=np.float32)
        tan_sum = np.zeros(3, dtype=np.float32)
        for i in fixed_list:
            q = pm.quat_mul(world_rotations32[i][None], setup.vertex_bind_pose_rotations[i][None])[0]
            nor_sum = nor_sum + pm.quat_to_normal(q[None])[0]
            tan_sum = tan_sum + pm.quat_to_tangent(q[None])[0]
        if np.linalg.norm(nor_sum) > 1e-30 and np.linalg.norm(tan_sum) > 1e-30:
            rot = pm.to_rotation((nor_sum / np.linalg.norm(nor_sum))[None].astype(np.float32),
                                 (tan_sum / np.linalg.norm(tan_sum))[None].astype(np.float32))[0]
            setup.init_local_gravity_direction = pm.quat_rotate(
                pm.quat_inverse(rot[None]), gravity_direction[None])[0].astype(np.float32)
    else:
        setup.init_local_gravity_direction = gravity_direction

    _build_distance_data(setup, vertex_to_vertex, parent_index, attributes, local_positions,
                         triangles)
    _build_bending_data(setup, triangles, attributes, local_positions, matrix_world)

    if (not setup.is_spring) and snapshot.skinning:
        extra = []
        extra_local = []
        base = len(transform_names)
        for name, set_index, rest in snapshot.skinning:
            world = np.asarray(rest, dtype=np.float64)
            if set_index >= 0:
                table_index = set_index
            else:
                transform_names.append(name)
                bind_pose = np.concatenate([bind_pose, (np.linalg.inv(world) @ matrix_world)[None]], axis=0)
                table_index = base
                base += 1
            extra.append((name, table_index))
            extra_local.append((w2l[:3, :3] @ world[:3, 3] + w2l[:3, 3]).astype(np.float32))
        if extra:
            _apply_custom_skinning(setup, attributes, local_positions, extra,
                                   np.array(extra_local, dtype=np.float32),
                                   skin_indices, skin_weights)

    setup.transform_names = transform_names
    setup.transform_bind_pose = bind_pose.astype(np.float32)
    setup.skin_indices = skin_indices
    setup.skin_weights = skin_weights

    _build_pass_buckets(setup, attributes, parent_index)
    _build_static_index_sets(setup, attributes)

    setup.valid = True
    return setup


def _build_distance_data(setup, vertex_to_vertex, parent_index, attributes, local_positions,
                         triangles):
    n = len(attributes)
    connect_set = set()
    vertical = [[] for _ in range(n)]
    horizontal = [[] for _ in range(n)]

    for i in range(n):
        attr = attributes[i]
        p = parent_index[i]
        for t in sorted(vertex_to_vertex[i]):
            t_attr = attributes[t]
            if _is_dont_move(attr) and _is_dont_move(t_attr):
                continue
            if _is_invalid(attr) or _is_invalid(t_attr):
                continue
            if t == p or parent_index[t] == i:
                vertical[i].append(t)
            else:
                horizontal[i].append(t)
            connect_set.add((min(i, t), max(i, t)))

    if len(triangles) > 0:
        edge_to_triangles = {}
        for t_index, tri in enumerate(triangles):
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge_to_triangles.setdefault((min(a, b), max(a, b)), []).append(t_index)
        for edge, tri_list in edge_to_triangles.items():
            if len(tri_list) < 2:
                continue
            p1 = local_positions[edge[0]]
            p2 = local_positions[edge[1]]
            edge_length = float(np.linalg.norm(p1 - p2))
            if edge_length < defs.EPSILON:
                continue
            for i in range(len(tri_list) - 1):
                e3 = _remaining_vertex(triangles[tri_list[i]], edge)
                p3 = local_positions[e3]
                n1 = pm.triangle_normal(p1[None], p2[None], p3[None])[0]
                attr3 = attributes[e3]
                for j in range(i + 1, len(tri_list)):
                    e4 = _remaining_vertex(triangles[tri_list[j]], edge)
                    p4 = local_positions[e4]
                    attr4 = attributes[e4]
                    n2 = pm.triangle_normal(p1[None], p2[None], p4[None])[0]
                    if _is_dont_move(attr3) and _is_dont_move(attr4):
                        continue
                    if abs(float(np.dot(n1, n2))) < 0.9396926:
                        continue
                    diagonal = float(np.linalg.norm(p3 - p4))
                    if abs(diagonal / edge_length - 1.0) <= 0.3:
                        key = (min(e3, e4), max(e3, e4))
                        if key not in connect_set:
                            connect_set.add(key)
                            horizontal[e3].append(e4)
                            horizontal[e4].append(e3)

    particle = []
    target = []
    rest = []
    is_spring = setup.is_spring
    for i in range(n):
        attr = attributes[i]
        if _is_invalid(attr):
            continue
        if _is_dont_move(attr) and not is_spring:
            continue
        pos = local_positions[i]
        for kind, bucket in ((0, vertical[i]), (1, horizontal[i])):
            for t in bucket:
                d = float(np.linalg.norm(pos - local_positions[t]))
                if d < 1e-8:
                    d = 0.0
                elif kind == 1:
                    d = -d
                particle.append(i)
                target.append(t)
                rest.append(d)

    setup.distance_particle = np.array(particle, dtype=np.int32)
    setup.distance_target = np.array(target, dtype=np.int32)
    setup.distance_rest = np.array(rest, dtype=np.float32)


def _build_bending_data(setup, triangles, attributes, local_positions, matrix_world):
    pairs = []
    rest_values = []
    signs = []
    if len(triangles) > 0:
        edge_to_triangles = {}
        for t_index, tri in enumerate(triangles):
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge_to_triangles.setdefault((min(a, b), max(a, b)), []).append(t_index)

        volume_set = set()
        world_positions = (np.einsum('ij,nj->ni', matrix_world[:3, :3], local_positions.astype(np.float64))
                           + matrix_world[:3, 3])
        for edge, tri_list in sorted(edge_to_triangles.items()):
            if len(tri_list) < 2:
                continue
            for i in range(len(tri_list) - 1):
                tri0 = triangles[tri_list[i]]
                for j in range(i + 1, len(tri_list)):
                    tri1 = triangles[tri_list[j]]
                    d0 = _remaining_vertex(tri0, edge)
                    d1 = _remaining_vertex(tri1, edge)
                    vtx = (d0, d1, edge[0], edge[1])
                    a0, a1, a2, a3 = (attributes[v] for v in vtx)
                    if _is_dont_move(a0) and _is_dont_move(a1) and _is_dont_move(a2) and _is_dont_move(a3):
                        continue
                    if _is_invalid(a0) or _is_invalid(a1) or _is_invalid(a2) or _is_invalid(a3):
                        continue

                    p0, p1, p2, p3 = (local_positions[v].astype(np.float64) for v in vtx)
                    n1 = np.cross(p2 - p0, p3 - p0)
                    n2 = np.cross(p3 - p1, p2 - p1)
                    l1 = np.linalg.norm(n1)
                    l2 = np.linalg.norm(n2)
                    if l1 < 1e-30 or l2 < 1e-30:
                        continue
                    n1 = n1 / l1
                    n2 = n2 / l2
                    d = max(-1.0, min(1.0, float(np.dot(n1, n2))))
                    rest_angle = math.acos(d)
                    e = p3 - p2
                    direction = float(np.dot(np.cross(n1, n2), e))
                    sign = -1 if direction < 0 else 1
                    degrees = abs(math.degrees(rest_angle))

                    if degrees < defs.TRIANGLE_BENDING_MAX_ANGLE:
                        pairs.append(vtx)
                        rest_values.append(rest_angle)
                        signs.append(sign)

                    if defs.VOLUME_MIN_ANGLE <= degrees <= 179.0:
                        key = tuple(sorted(vtx))
                        if key not in volume_set:
                            volume_set.add(key)
                            w0, w1, w2, w3 = (world_positions[v] for v in vtx)
                            volume = (1.0 / 6.0) * float(np.dot(np.cross(w1 - w0, w2 - w0), w3 - w0))
                            pairs.append(vtx)
                            rest_values.append(volume * defs.VOLUME_SCALE)
                            signs.append(defs.VOLUME_SIGN)

    setup.bending_pairs = np.array(pairs, dtype=np.int32).reshape(-1, 4)
    setup.bending_rest = np.array(rest_values, dtype=np.float32)
    setup.bending_sign = np.array(signs, dtype=np.int8)


def _apply_custom_skinning(setup, attributes, local_positions, extra, extra_local,
                           skin_indices, skin_weights):
    distance_reduction = 0.6
    distance_pow = 2.0
    for v in range(len(attributes)):
        if _is_dont_move(attributes[v]):
            continue
        pos = local_positions[v]
        costs = []
        for k in range(len(extra)):
            d = float(np.linalg.norm(pos - extra_local[k]))
            costs.append((d, extra[k][1]))
        costs.sort(key=lambda item: item[0])
        costs = costs[:4]
        min_dist = costs[0][0] * distance_reduction
        adjusted = [(max(c - min_dist, 0.0) ** distance_pow, t) for c, t in costs]
        if adjusted[0][0] < defs.EPSILON:
            weights = [(1.0, adjusted[0][1])]
        else:
            total = sum(1.0 / c for c, _ in adjusted)
            weights = [((1.0 / c) / total, t) for c, t in adjusted]
            weights = [(w, t) for w, t in weights if w >= 0.001]
            s = sum(w for w, _ in weights)
            weights = [(w / s, t) for w, t in weights]
        skin_indices[v] = 0
        skin_weights[v] = 0.0
        for slot, (w, t) in enumerate(weights[:4]):
            skin_indices[v, slot] = t
            skin_weights[v, slot] = w


def _build_pass_buckets(setup, attributes, parent_index):
    data = setup.baseline_data
    depth = {}
    order = []
    for b in range(len(setup.baseline_start)):
        start = setup.baseline_start[b]
        count = setup.baseline_count[b]
        for k in range(count):
            v = int(data[start + k])
            p = int(parent_index[v])
            if k == 0 or p < 0 or p not in depth:
                depth[v] = 0
            else:
                depth[v] = depth[p] + 1
            order.append(v)

    max_depth = max(depth.values()) if depth else -1

    fk_levels = [[] for _ in range(max_depth + 1)]
    for v in order:
        fk_levels[depth[v]].append(v)
    setup.fk_levels = []
    for level in fk_levels:
        if not level:
            continue
        level = np.array(level, dtype=np.int32)
        parents = parent_index[level]
        has_parent = (parents >= 0) & _is_move(attributes[level])
        yes = level[has_parent]
        setup.fk_levels.append((yes, parent_index[yes], level[~has_parent]))

    sibling_counter = {}
    passes = {}
    for v in order:
        p = int(parent_index[v])
        if p < 0 or _is_dont_move(attributes[v]):
            continue
        lv = depth[v]
        rank = sibling_counter.get((lv, p), 0)
        sibling_counter[(lv, p)] = rank + 1
        passes.setdefault((lv, rank), []).append(v)
    setup.angle_passes = []
    for key in sorted(passes.keys()):
        vertices = np.array(passes[key], dtype=np.int32)
        setup.angle_passes.append((key, (vertices, parent_index[vertices])))

    postline_levels = []
    entries_by_level = [[] for _ in range(max_depth + 1)]
    for b in range(len(setup.baseline_start)):
        if not (setup.baseline_flags[b] & 1):
            continue
        start = setup.baseline_start[b]
        count = setup.baseline_count[b]
        for k in range(count):
            v = int(data[start + k])
            entries_by_level[depth[v]].append(v)
    for level in entries_by_level:
        if not level:
            continue
        entry_vertex = np.array(level, dtype=np.int32)
        child_owner = []
        child_vertex = []
        for slot, v in enumerate(level):
            start = setup.vertex_child_start[v]
            count = setup.vertex_child_count[v]
            for c in setup.vertex_child_data[start:start + count]:
                child_owner.append(slot)
                child_vertex.append(int(c))
        postline_levels.append((entry_vertex,
                                np.array(child_owner, dtype=np.int32),
                                np.array(child_vertex, dtype=np.int32)))
    setup.postline_levels = postline_levels

    setup.angle_buffered = setup.baseline_data[parent_index[setup.baseline_data] >= 0] \
        if len(setup.baseline_data) else np.zeros(0, dtype=np.int32)


def _build_static_index_sets(setup, attributes):
    move = _is_move(attributes)
    invalid = _is_invalid(attributes)
    fixed = (attributes & ATTR_FIXED) != 0
    disable = (attributes & defs.ATTR_DISABLE_COLLISION) != 0
    motion_ok = (attributes & defs.ATTR_INVALID_MOTION) == 0

    setup.tether_index = np.flatnonzero(move & (setup.vertex_root >= 0)).astype(np.int32)
    setup.motion_index = np.flatnonzero(move & motion_ok).astype(np.int32)

    update_move = move | (setup.is_spring & ~invalid)
    setup.update_move_index = np.flatnonzero(update_move).astype(np.int32)
    setup.update_fixed_index = np.flatnonzero(~update_move).astype(np.int32)
    setup.spring_index = np.flatnonzero(fixed).astype(np.int32) if setup.is_spring \
        else np.zeros(0, dtype=np.int32)

    process = ~invalid & ~disable
    if not setup.is_spring:
        process = process & move
    setup.collision_process_index = np.flatnonzero(process).astype(np.int32)

    if len(setup.edges):
        e0 = setup.edges[:, 0]
        e1 = setup.edges[:, 1]
        setup.collision_edge_index = np.flatnonzero(move[e0] | move[e1]).astype(np.int32)
    else:
        setup.collision_edge_index = np.zeros(0, dtype=np.int32)
