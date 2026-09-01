import heapq

import numpy as np

from . import defs
from . import host_math

F4 = np.float32
F8 = np.float64
I4 = np.int32
I8 = np.int64
B1 = np.bool_
U1 = np.uint8

COMPONENT_WORLD_BASIS_REASON = (
    "component_world_scale is the length of each component basis axis and is never negative, "
    "and component_world_reflected is the sign of the determinant of that basis; together "
    "they are the whole of what the world matrix says about the basis, because the axis a "
    "mirror sits on is not recoverable from the matrix, see "
    "compile.COMPONENT_BASIS_DECOMPOSITION_REASON; the two are split so that the frame mask "
    "can read a collapsed axis off the length without a mirrored component ever reading as "
    "collapsed, which is what happened while one field carried both and a mirrored component "
    "turned a quarter turn read as zero length on two axes and stopped simulating; every "
    "snapshot of the scale carries its own bit next to it for the same reason")

TEAM_DTYPE = np.dtype([
    ("valid", B1), ("enabled", B1), ("is_spring", B1), ("running", B1),
    ("reset_pending", B1), ("time_reset_pending", B1), ("keep_teleport_pending", B1),
    ("inertia_shift", B1), ("negative_scale_teleport", B1),
    ("component_world_reflected", B1), ("old_component_world_reflected", B1),
    ("old_frame_world_reflected", B1),
    ("culling_invisible", B1), ("has_anchor", B1), ("had_anchor", B1), ("step_active", B1),
    ("sync_target", I4), ("sync_top", I4), ("wind_seed", I4),
    ("p_start", I4), ("p_count", I4), ("t_start", I4), ("t_count", I4),
    ("c_start", I4), ("c_count", I4),
    ("cv_start", I4), ("cv_count", I4), ("cf_start", I4), ("cf_count", I4),
    ("sp_start", I4), ("sp_count", I4), ("se_start", I4), ("se_count", I4),
    ("st_start", I4), ("st_count", I4),
    ("use_point", B1), ("use_edge", B1), ("use_triangle", B1),
    ("self_contact_slots", I4),
    ("self_grid_size", F4), ("self_max_primitive_size", F4),
    ("time", F4), ("old_time", F4), ("now_update_time", F4), ("old_update_time", F4),
    ("frame_update_time", F4), ("frame_old_time", F4), ("frame_delta_time", F4),
    ("now_time_scale", F4), ("update_count", I4), ("skip_count", I4),
    ("frame_interpolation", F4), ("velocity_weight", F4), ("blend_weight", F4),
    ("distance_weight", F4), ("gravity_ratio", F4), ("gravity_dot", F4), ("scale_ratio", F4),
    ("negative_scale_matrix", F8, (4, 4)),
    ("force_mode", np.int8), ("impact_force", F4, (3,)),
    ("anchor_position", F4, (3,)), ("anchor_rotation", F4, (4,)),
    ("old_anchor_position", F4, (3,)), ("old_anchor_rotation", F4, (4,)),
    ("anchor_component_local_position", F4, (3,)),
    ("component_world_position", F4, (3,)), ("component_world_rotation", F4, (4,)),
    ("component_world_scale", F4, (3,)),
    ("old_component_world_position", F4, (3,)), ("old_component_world_rotation", F4, (4,)),
    ("old_component_world_scale", F4, (3,)),
    ("frame_component_shift_vector", F4, (3,)), ("frame_component_shift_rotation", F4, (4,)),
    ("frame_moving_speed", F4), ("frame_moving_direction", F4, (3,)),
    ("frame_world_position", F4, (3,)), ("frame_world_rotation", F4, (4,)),
    ("frame_world_scale", F4, (3,)),
    ("old_frame_world_position", F4, (3,)), ("old_frame_world_rotation", F4, (4,)),
    ("old_frame_world_scale", F4, (3,)),
    ("now_world_position", F4, (3,)), ("now_world_rotation", F4, (4,)),
    ("old_world_position", F4, (3,)), ("old_world_rotation", F4, (4,)),
    ("step_move_inertia_ratio", F4), ("step_rotation_inertia_ratio", F4),
    ("step_vector", F4, (3,)), ("step_rotation", F4, (4,)),
    ("inertia_vector", F4, (3,)), ("inertia_rotation", F4, (4,)),
    ("angular_velocity", F4), ("rotation_axis", F4, (3,)),
    ("init_local_gravity_direction", F4, (3,)), ("smoothing_velocity", F4, (3,)),
    ("init_scale", F4, (3,)),
    ("gravity", F4), ("gravity_direction", F4, (3,)), ("gravity_falloff", F4),
    ("stablization_time", F4), ("blend_weight_param", F4),
    ("damping_lut", F4, (16,)), ("radius_lut", F4, (16,)),
    ("normal_axis_vector", F4, (3,)),
    ("rotational_interpolation", F4), ("root_rotation", F4), ("animation_pose_ratio", F4),
    ("time_scale", F4), ("tether_compression", F4),
    ("distance_lut", F4, (16,)), ("bending_stiffness", F4),
    ("angle_use_restoration", B1), ("angle_restoration_lut", F4, (16,)),
    ("angle_restoration_attenuation", F4), ("angle_restoration_gravity_falloff", F4),
    ("angle_use_limit", B1), ("angle_limit_lut", F4, (16,)), ("angle_limit_stiffness", F4),
    ("motion_use_max_distance", B1), ("motion_max_distance_lut", F4, (16,)),
    ("motion_use_backstop", B1), ("motion_backstop_radius", F4),
    ("motion_backstop_lut", F4, (16,)), ("motion_stiffness", F4),
    ("collision_mode", np.int8), ("dynamic_friction", F4), ("static_friction", F4),
    ("limit_distance_lut", F4, (16,)),
    ("self_mode", np.int8), ("sync_mode", np.int8),
    ("self_thickness_lut", F4, (16,)), ("self_cloth_mass", F4),
    ("anchor_inertia", F4), ("world_inertia", F4), ("movement_inertia_smoothing", F4),
    ("movement_speed_limit", F4), ("rotation_speed_limit", F4),
    ("local_inertia", F4), ("local_movement_speed_limit", F4), ("local_rotation_speed_limit", F4),
    ("depth_inertia", F4), ("centrifugal_acceleration", F4), ("particle_speed_limit", F4),
    ("teleport_mode", np.int8), ("teleport_distance", F4), ("teleport_rotation", F4),
    ("wind_influence", F4), ("wind_frequency", F4), ("wind_turbulence", F4),
    ("wind_blend", F4), ("wind_synchronization", F4), ("wind_depth_weight", F4),
    ("wind_moving", F4),
    ("spring_power", F4), ("spring_limit_distance", F4),
    ("spring_normal_limit_ratio", F4), ("spring_noise", F4),
    ("wind_count", np.int8), ("wind_zone_id", I4, (defs.WIND_ZONE_SLOTS,)),
    ("wind_time", F4, (defs.WIND_ZONE_SLOTS,)), ("wind_main", F4, (defs.WIND_ZONE_SLOTS,)),
    ("moving_wind_time", F4), ("moving_wind_main", F4),
    ("moving_wind_direction", F4, (3,)), ("moving_wind_dirq", F4, (4,)),
])

STRUCTURE_TEAM_COLUMNS = ("valid", "is_spring", "p_start", "p_count", "t_start", "t_count",
                          "c_start", "c_count", "cv_start", "cv_count", "cf_start",
                          "cf_count", "sp_start", "sp_count",
                          "se_start", "se_count", "st_start", "st_count",
                          "self_contact_slots")

COLLIDER_MESH_TOPOLOGY_REASON = (
    "the corners of a mesh collider are a structure and its vertices are a frame input, so "
    "the signature that decides whether the device state is rebuilt reads the corners and "
    "not the positions; reading the positions is what made a body that bends rebuild the "
    "whole state on every frame, and not reading the corners is what would let a surface "
    "retriangulated into the same number of faces keep the corners the binding was taken "
    "with, which the block starts and counts cannot see because a released block is handed "
    "straight back to the next request of the same size; this question is now answered by "
    "the world's structure_revision integer, bumped where the topology is written rather "
    "than recomputed from the bytes every frame, and the completeness of that bump set is "
    "proven by revision_audit.py in the criteria repository")

TEAM_ROW_IDENTITY_REASON = (
    "a team row carries the same sentence as the collider block above: the row a chain gives "
    "back is handed straight out to the next chain, and if that chain has the same shape then "
    "valid, the starts and the counts all read the same, so the signature cannot see that the "
    "team behind the row is a different team and the device keeps every field the previous "
    "one left there; the world counts how many times a row has changed hands and the count "
    "goes into the signature, which is what makes a registry clear or a chain rebuilt in "
    "place reload rather than resume; measured on twenty five chains cleared and registered "
    "again three times over, which reloaded three times while the row assignment still "
    "alternated and stopped reloading at all once it was made repeatable, and the three runs "
    "of the same twenty four frames then ended up to 1.57e-01 m apart, the difference "
    "starting in the angle limit buffer the reset does not write; the count of hands a row "
    "has changed and every column above now reach the engine through the world's "
    "structure_revision integer, bumped where the row is written rather than joined from the "
    "bytes every frame, and revision_audit.py in the criteria repository proves that bump "
    "set is complete")

PARTICLE_FIELDS = {
    "team": (I4, ()),
    "positions": (F4, (3,)), "rotations": (F4, (4,)),
    "next_positions": (F4, (3,)), "old_positions": (F4, (3,)), "old_rotations": (F4, (4,)),
    "base_positions": (F4, (3,)), "base_rotations": (F4, (4,)),
    "old_anim_positions": (F4, (3,)), "old_anim_rotations": (F4, (4,)),
    "velocity_positions": (F4, (3,)), "display_positions": (F4, (3,)),
    "velocities": (F4, (3,)), "real_velocities": (F4, (3,)),
    "friction": (F4, ()), "static_friction": (F4, ()),
    "collision_normals": (F4, (3,)),
    "step_basic_positions": (F4, (3,)), "step_basic_rotations": (F4, (4,)),
    "temp_base_positions": (F4, (3,)), "temp_base_rotations": (F4, (4,)),
    "depth": (F4, ()),
    "attr_move": (B1, ()), "attr_fixed": (B1, ()), "attr_invalid": (B1, ()),
    "attr_disable_collision": (B1, ()), "attr_motion": (B1, ()), "attr_zero_distance": (B1, ()),
    "vertex_parent": (I4, ()), "vertex_root": (I4, ()), "vertex_root_local": (I4, ()),
    "local_positions": (F4, (3,)), "local_normals": (F4, (3,)), "local_tangents": (F4, (3,)),
    "skin_indices": (I4, (4,)), "skin_weights": (F4, (4,)),
    "aim_child": (I4, ()), "aim_rest_reach": (F4, ()), "publish_transform": (I4, ()),
    "bone_row": (I4, ()), "publish_position": (B1, ()),
    "vertex_local_positions": (F4, (3,)), "vertex_local_rotations": (F4, (4,)),
    "vertex_bind_pose_rotations": (F4, (4,)),
    "vertex_to_transform_rotations": (F4, (4,)),
    "normal_adjustment_rotations": (F4, (4,)),
    "uv": (F4, (2,)),
    "out_rotations": (F4, (4,)),
    "intersect_flag": (B1, ()),
    "particle_radius": (F4, ()),
    "albuf_length": (F4, ()), "albuf_local_pos": (F4, (3,)), "albuf_local_rot": (F4, (4,)),
    "albuf_restore": (F4, (3,)), "albuf_rotation": (F4, (4,)),
    "wind_count": (I4, ()),
    "wind_zone_id": (I4, (defs.WIND_ZONE_SLOTS,)),
    "wind_main": (F4, (defs.WIND_ZONE_SLOTS,)),
    "wind_dirq": (F4, (defs.WIND_ZONE_SLOTS, 4)),
    "wind_zone_turbulence": (F4, (defs.WIND_ZONE_SLOTS,)),
    "wind_phase_slot": (I4, (defs.WIND_ZONE_SLOTS,)),
}

TRANSFORM_FIELDS = {
    "team": (I4, ()),
    "bind_pose": (F4, (4, 4)),
    "world": (F4, (4, 4)),
    "solved": (F4, (4, 4)),
}

COLLIDER_FIELDS = {
    "team": (I4, ()),
    "kind": (I4, ()),
    "enabled": (B1, ()),
    "enabled_prev": (B1, ()),
    "active": (B1, ()),
    "input_positions": (F4, (3,)), "input_rotations": (F4, (4,)),
    "input_tips": (F4, (3,)), "input_radii": (F4, (2,)),
    "frame_positions": (F4, (3,)), "frame_rotations": (F4, (4,)),
    "frame_tips": (F4, (3,)), "frame_radii": (F4, (2,)),
    "old_frame_positions": (F4, (3,)), "old_frame_rotations": (F4, (4,)),
    "old_frame_tips": (F4, (3,)),
    "now_positions": (F4, (3,)), "now_rotations": (F4, (4,)), "now_tips": (F4, (3,)),
    "old_positions": (F4, (3,)), "old_rotations": (F4, (4,)), "old_tips": (F4, (3,)),
    "work_radius": (F4, (2,)), "work_old_pos": (F4, (2, 3)), "work_next_pos": (F4, (2, 3)),
    "work_rot": (F4, (4,)), "work_inv_old_rot": (F4, (4,)), "work_inv_rot": (F4, (4,)),
    "work_aabb_min": (F4, (3,)), "work_aabb_max": (F4, (3,)),
    "mesh_vertex_start": (I4, ()), "mesh_vertex_count": (I4, ()),
    "mesh_face_start": (I4, ()), "mesh_face_count": (I4, ()),
    "mesh_local_bound_min": (F4, (3,)), "mesh_local_bound_max": (F4, (3,)),
}

COLLIDER_VERTEX_FIELDS = {
    "team": (I4, ()), "collider": (I4, ()), "local_position": (F4, (3,)),
    "fan_face": (I4, ()), "fan_corner": (I4, ()), "pseudo_normal": (F4, (3,)),
}

COLLIDER_FACE_FIELDS = {
    "team": (I4, ()), "collider": (I4, ()), "vertex": (I4, (3,)),
    "fan_next_face": (I4, (3,)), "fan_next_corner": (I4, (3,)),
    "edge_ring_face": (I4, (3,)), "edge_ring_corner": (I4, (3,)),
    "aabb_min": (F4, (3,)), "aabb_max": (F4, (3,)), "normal": (F4, (3,)),
    "edge_normal": (F4, (3, 3)),
}

COLLIDER_MESH_HALF_EDGE_REASON = (
    "a triangle names its three directed edges by the corner they leave, so a face row and "
    "a corner slot together name one directed edge; each of those is linked into two "
    "cycles, the ring of the half edges that walk the same undirected edge and the fan of "
    "the half edges that leave the same corner, and those two answer the two questions the "
    "sign of a distance asks, what meets on the edge the foot landed on and what meets at "
    "the corner it landed on; the two are separate links because they only agree where "
    "every edge carries exactly two faces, and a game mesh is the case where they do not: "
    "along a rim an edge carries one face and its ring is the half edge alone, while the "
    "fan of a corner on that rim is an arc that has to be closed into a cycle across the "
    "opening for the walk to reach the whole of it; each link is kept as a face row and a "
    "corner slot in two columns rather than packed into one number, because unpacking it "
    "would put an integer division and a remainder in the walk to save four bytes a face")

COLLIDER_MESH_VERTEX_FAN_LIMIT = defs.COLLIDER_MESH_VERTEX_FAN_LIMIT

COLLIDER_MESH_EDGE_RING_LIMIT = defs.COLLIDER_MESH_EDGE_RING_LIMIT

COLLIDER_MESH_VERTEX_FAN_REASON = (
    "the angle weighted normal of a corner is a sum over the faces that meet there, and the "
    "device walks that fan one face at a time through the fan link, so the walk "
    "needs a bound it can be compiled against; a corner of a surface a person models is "
    "reached by a handful of faces and this bound is far above anything a body mesh "
    "carries, and a surface that passes it is refused where the triangles arrive rather "
    "than silently walked part way round, because a fan walked part way round gives a "
    "normal that is wrong by whatever the rest of the fan would have added and nothing "
    "downstream can tell that from a normal that is right")

COLLIDER_MESH_EDGE_RING_REASON = defs.COLLIDER_MESH_EDGE_RING_REASON

COLLIDER_MESH_PSEUDO_NORMAL_REASON = (
    "the sign of the distance a mesh collider reports comes from the pseudo normal of the "
    "feature of the closest triangle the foot landed on, and that construction, from "
    "Baerentzen and Aanaes, is one rule and not three: the pseudo normal of a feature is "
    "the sum of the normals of the faces that actually meet on it, each weighted by the "
    "angle that face subtends there; a face interior is met by one face, an edge of a "
    "closed surface by two, an edge along a rim by one, an edge where a surface branches "
    "by however many are there, and a corner by its whole fan; closure is the case of that "
    "rule where every edge has two faces and nothing more, so widening the collider to a "
    "game mesh is not a second construction beside the first, it is the same sum over "
    "whatever is really there, and there is no column anywhere saying which kind of "
    "surface a collider is because no kernel has to ask")

COLLIDER_MESH_WINDING_REASON = (
    "closure and consistent winding are two different properties and only the second one "
    "is required: a sheet with a rim reports a signed distance perfectly well, its outward "
    "side being the side its triangles are wound to face, while two neighbouring triangles "
    "wound against each other put the outward side of one against the inward side of the "
    "other and the field flips sign across the edge between them, which pulls the cloth "
    "into the body; a winding that disagrees is repaired here rather than refused, because "
    "which of the two triangles is the wrong way round is a question with an answer, and "
    "the answer is found by walking the surface and flipping whatever disagrees with what "
    "it is reached from; the whole of a connected piece is then wound the way most of it "
    "already was, so a surface that already agreed with itself is left exactly as it "
    "arrived and the artist's own statement of which side is out is what survives, on a "
    "closed piece exactly as on an open one; only a piece that cannot be made to agree at "
    "all, a surface with no consistent outward side, is refused")

COLLIDER_MESH_REPORTED_EDGES = 8


class ColliderMeshRefused(ValueError):
    pass


COLLIDER_MESH_VALIDATION_REASON = (
    "the triangles of every mesh collider a binding carries are read and checked before "
    "the world releases a single row, because a binding that turns out to be unusable "
    "must leave the world exactly as it was; the arenas that hold colliders, collider "
    "vertices, collider faces and the collision pair lists are released and reallocated "
    "together as one step, so a refusal raised part way through that step would leave the "
    "team owning rows it no longer holds and mesh rows declaring triangles they were never "
    "given, and every later frame would report that inconsistency instead of the refusal "
    "that caused it")


def validated_collider_meshes(binding, team_label):
    meshes = binding.meshes
    collider_count = binding.count
    assert len(meshes) == collider_count, \
        "%s binds %d colliders and hands in %d mesh entries, one entry per collider, " \
        "empty for the kinds that carry no triangles" \
        % (team_label, collider_count, len(meshes))
    blocks = []
    vertex_total = 0
    face_total = 0
    for index in range(collider_count):
        geometry = meshes[index]
        is_mesh = int(binding.kinds[index]) == defs.COLLIDER_MESH
        assert is_mesh == (geometry is not None), \
            "%s\n%s collider %d declares the kind %d and %s triangles, a mesh collider " \
            "carries triangles and no other kind does" \
            % (COLLIDER_MESH_PSEUDO_NORMAL_REASON, team_label, index,
               int(binding.kinds[index]),
               "carries" if geometry is not None else "carries no")
        if geometry is None:
            blocks.append(None)
            continue
        label = "%s of %s" % (binding.names[index], team_label)
        block = assert_collider_mesh_is_orientable(label, geometry[0], geometry[1])
        blocks.append(block)
        vertex_total += int(block[0].shape[0])
        face_total += int(block[1].shape[0])
    return blocks, vertex_total, face_total


def assert_collider_mesh_is_orientable(label, vertices, faces):
    vertices = np.ascontiguousarray(vertices, dtype=F4)
    faces = np.ascontiguousarray(faces, dtype=I4)
    assert vertices.ndim == 2 and vertices.shape[1] == 3, \
        "the mesh collider %s hands in vertices of shape %r and a vertex is three floats" \
        % (label, vertices.shape)
    assert faces.ndim == 2 and faces.shape[1] == 3, \
        "the mesh collider %s hands in faces of shape %r and a face is three vertex " \
        "indices" % (label, faces.shape)
    vertex_count = int(vertices.shape[0])
    face_count = int(faces.shape[0])
    if face_count == 0 or vertex_count == 0:
        raise ColliderMeshRefused(
            "%s\nthe mesh collider %s holds %d vertices and %d triangles, and a surface "
            "with no triangles has no side to be on"
            % (COLLIDER_MESH_PSEUDO_NORMAL_REASON, label, vertex_count, face_count))
    out_of_range = faces[(faces < 0) | (faces >= vertex_count)]
    if out_of_range.shape[0]:
        raise ColliderMeshRefused(
            "%s\nthe mesh collider %s names the vertex indices %r and it holds %d vertices"
            % (COLLIDER_MESH_PSEUDO_NORMAL_REASON, label,
               sorted(set(int(value) for value in out_of_range))[:COLLIDER_MESH_REPORTED_EDGES],
               vertex_count))
    degenerate = faces[(faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2])
                       | (faces[:, 2] == faces[:, 0])]
    if degenerate.shape[0]:
        raise ColliderMeshRefused(
            "%s\nthe mesh collider %s holds %d triangles that name the same vertex twice, "
            "the first of them is %r"
            % (COLLIDER_MESH_PSEUDO_NORMAL_REASON, label, int(degenerate.shape[0]),
               [int(value) for value in degenerate[0]]))
    faces = _wind_collider_mesh_together(label, faces, vertex_count)
    directed = _directed_half_edges(faces)
    grouping = _undirected_edge_grouping(directed, vertex_count)
    ring = _edge_ring_links(label, grouping, face_count, vertex_count)
    twin = _half_edge_twins(directed, grouping)
    fan = _vertex_fan_links(label, directed[:, 0], twin, face_count, vertex_count)
    seeds = _vertex_fan_seeds(label, directed[:, 0], face_count, vertex_count)
    return vertices, faces, ring, fan, seeds


def _directed_half_edges(faces):
    return np.concatenate([faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]], axis=0)


def _half_edge_rows(indices, face_count):
    return (np.asarray(indices) % face_count).astype(I4), \
        (np.asarray(indices) // face_count).astype(I4)


def _undirected_edge_grouping(directed, vertex_count):
    low = np.minimum(directed[:, 0], directed[:, 1]).astype(np.int64)
    high = np.maximum(directed[:, 0], directed[:, 1]).astype(np.int64)
    key = low * vertex_count + high
    forward = (directed[:, 0] < directed[:, 1])
    order = np.lexsort((np.arange(key.shape[0]), ~forward, key))
    ordered = key[order]
    starts = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1])))
    widths = np.diff(np.concatenate((starts, [ordered.shape[0]])))
    return {"key": key, "forward": forward, "order": order,
            "starts": starts, "widths": widths}


def _wind_collider_mesh_together(label, faces, vertex_count):
    directed = _directed_half_edges(faces)
    grouping = _undirected_edge_grouping(directed, vertex_count)
    order = grouping["order"]
    starts = grouping["starts"]
    widths = grouping["widths"]
    face_count = int(faces.shape[0])
    paired = np.flatnonzero(widths == 2)
    first = order[starts[paired]]
    second = order[starts[paired] + 1]
    disagreeing = grouping["forward"][first] == grouping["forward"][second]
    parity = _face_winding_parity(label, first, second, disagreeing, face_count,
                                  directed, vertex_count)
    if not parity.any():
        return faces
    wound = faces.copy()
    wound[parity] = faces[parity][:, (0, 2, 1)]
    return np.ascontiguousarray(wound, dtype=I4)


def _face_winding_parity(label, first, second, disagreeing, face_count, directed,
                         vertex_count):
    parent = list(range(face_count))
    offset = bytearray(face_count)

    def find(node):
        root = node
        shift = 0
        while parent[root] != root:
            shift ^= offset[root]
            root = parent[root]
        answer = shift
        walker = node
        while parent[walker] != root:
            ahead = parent[walker]
            ahead_shift = offset[walker]
            parent[walker] = root
            offset[walker] = shift
            shift ^= ahead_shift
            walker = ahead
        return root, answer

    left = (first % face_count).tolist()
    right = (second % face_count).tolist()
    need = disagreeing.astype(np.int64).tolist()
    for index in range(len(left)):
        left_root, left_shift = find(left[index])
        right_root, right_shift = find(right[index])
        if left_root == right_root:
            if (left_shift ^ right_shift) != need[index]:
                edge = int(first[index])
                raise ColliderMeshRefused(
                    "%s\nthe mesh collider %s carries a piece with no consistent outward "
                    "side at all: walking it around a loop that closes through the edge "
                    "%r arrives back at a triangle demanding to be wound both ways"
                    % (COLLIDER_MESH_WINDING_REASON, label,
                       _named_edges([int(directed[edge, 0]) * vertex_count
                                     + int(directed[edge, 1])], vertex_count)))
            continue
        parent[right_root] = left_root
        offset[right_root] = left_shift ^ right_shift ^ need[index]
    roots = np.zeros(face_count, dtype=np.int64)
    shifts = np.zeros(face_count, dtype=bool)
    for face in range(face_count):
        root, shift = find(face)
        roots[face] = root
        shifts[face] = bool(shift)
    return _minority_winding(shifts, roots)


def _minority_winding(parity, roots):
    _roots, labels = np.unique(roots, return_inverse=True)
    labels = labels.reshape(-1)
    turned = np.bincount(labels[parity], minlength=int(_roots.shape[0]))
    held = np.bincount(labels, minlength=int(_roots.shape[0]))
    return np.where(turned[labels] * 2 > held[labels], ~parity, parity)


COLLIDER_MESH_INWARD_REASON = (
    "which side of a mesh collider is the outside is the side its triangles are wound to "
    "face, so a closed piece wound the other way round is not a broken collider, it is a "
    "container: the distance it reports is negative outside it and positive within, and the "
    "cloth that meets it is held inside it instead of out of it, which is exactly what a "
    "room, a box or any other body a character stands inside is for; it is therefore taken "
    "and never refused, and the one thing owed is a word to the person who built it, because "
    "no reading downstream can tell a container that was meant from a body that was turned "
    "inside out by accident, and nothing downstream fails in a way anything can see: the "
    "cloth simply walks into the body and stays there while every count and every gate reads "
    "exactly as it does on a body that is the right way round; the question is asked of each "
    "closed piece on its own, because one collider can carry several and a piece with a rim "
    "encloses nothing for the question to be about")


def _face_pieces(grouping, face_count):
    order = grouping["order"]
    starts = grouping["starts"]
    widths = grouping["widths"]
    parent = list(range(face_count))

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    face_of_slot = (order % face_count).tolist()
    for index in range(int(starts.shape[0])):
        start = int(starts[index])
        held = find(face_of_slot[start])
        for step in range(1, int(widths[index])):
            other = find(face_of_slot[start + step])
            if other != held:
                parent[other] = held
    return np.array([find(face) for face in range(face_count)], dtype=I8)


def collider_mesh_inward_pieces(vertices, faces):
    vertices = np.ascontiguousarray(vertices, dtype=F4)
    faces = np.ascontiguousarray(faces, dtype=I4)
    face_count = int(faces.shape[0])
    vertex_count = int(vertices.shape[0])
    directed = _directed_half_edges(faces)
    grouping = _undirected_edge_grouping(directed, vertex_count)
    order = grouping["order"]
    starts = grouping["starts"]
    widths = grouping["widths"]
    forward = grouping["forward"]
    pieces = _face_pieces(grouping, face_count)
    opposed = np.zeros(int(starts.shape[0]), dtype=bool)
    paired = np.flatnonzero(widths == 2)
    heads = starts[paired]
    opposed[paired] = forward[order[heads]] != forward[order[heads + 1]]
    rims = np.flatnonzero(~opposed)
    open_pieces = set(int(value) for value
                      in pieces[order[starts[rims]] % face_count].tolist())
    corners = vertices[faces].astype(F8)
    six_volume = np.einsum("ij,ij->i", corners[:, 0],
                           np.cross(corners[:, 1], corners[:, 2]))
    roots, labels = np.unique(pieces, return_inverse=True)
    labels = labels.reshape(-1)
    totals = np.bincount(labels, weights=six_volume, minlength=int(roots.shape[0]))
    closed = np.array([int(root) not in open_pieces for root in roots.tolist()], dtype=bool)
    return int(np.count_nonzero(closed & (totals < 0.0))), int(np.count_nonzero(closed))


def collider_mesh_inward_notice(label, vertices, faces):
    inward, closed = collider_mesh_inward_pieces(vertices, faces)
    if inward == 0:
        return None
    return ("%s\nthe mesh collider %s carries %d closed piece of %d wound with the outward "
            "side facing in, so cloth that reaches it is drawn inside and held there rather "
            "than pushed out; leave it as it is if that container is what was meant, and "
            "flip the normals of that piece if it is not"
            % (COLLIDER_MESH_INWARD_REASON, label, inward, closed))


def _edge_ring_links(label, grouping, face_count, vertex_count):
    order = grouping["order"]
    starts = grouping["starts"]
    widths = grouping["widths"]
    widest = int(widths.max()) if widths.shape[0] else 0
    if widest > COLLIDER_MESH_EDGE_RING_LIMIT:
        raise ColliderMeshRefused(
            "%s\nthe mesh collider %s has an edge where %d faces meet and the walk is "
            "compiled against a bound of %d"
            % (COLLIDER_MESH_EDGE_RING_REASON, label, widest,
               COLLIDER_MESH_EDGE_RING_LIMIT))
    slots = np.arange(order.shape[0], dtype=np.int64)
    start_of = np.repeat(starts, widths)
    width_of = np.repeat(widths, widths)
    ahead = start_of + (slots - start_of + 1) % width_of
    ring = np.zeros(order.shape[0], dtype=np.int64)
    ring[order] = order[ahead]
    return _corner_columns(ring, face_count)


def _half_edge_twins(directed, grouping):
    order = grouping["order"]
    starts = grouping["starts"]
    widths = grouping["widths"]
    forward = grouping["forward"]
    slots = np.arange(order.shape[0], dtype=np.int64)
    start_of = np.repeat(starts, widths)
    width_of = np.repeat(widths, widths)
    inside = slots - start_of
    ahead_of = np.add.reduceat(forward[order].astype(np.int64), starts) \
        if starts.shape[0] else np.zeros(0, dtype=np.int64)
    ahead = np.repeat(ahead_of, widths)
    matched = np.minimum(ahead, width_of - ahead)
    leads = inside < ahead
    partner = np.where(leads, inside + ahead, inside - ahead)
    holds = np.where(leads, inside < matched, inside - ahead < matched)
    slot = np.where(holds, start_of + partner, start_of)
    twin = np.full(order.shape[0], -1, dtype=np.int64)
    twin[order] = np.where(holds, order[slot], -1)
    return twin


def _vertex_fan_links(label, origins, twin, face_count, vertex_count):
    total = int(twin.shape[0])
    ahead = np.full(total, -1, dtype=np.int64)
    linked = twin >= 0
    partner_face = twin[linked] % face_count
    partner_corner = twin[linked] // face_count
    ahead[linked] = ((partner_corner + 1) % 3) * face_count + partner_face
    behind = np.full(total, -1, dtype=np.int64)
    behind[ahead[linked]] = np.flatnonzero(linked)
    order = np.argsort(origins.astype(np.int64), kind="stable")
    boundaries = np.flatnonzero(np.concatenate(
        ([True], origins[order][1:] != origins[order][:-1]))) if total else np.zeros(0, np.int64)
    widths = np.diff(np.concatenate((boundaries, [total]))) if total else np.zeros(0, np.int64)
    seen = np.zeros(total, dtype=bool)
    for start, width in zip(boundaries, widths):
        members = order[start:start + width]
        heads = []
        tails = []
        for member in members:
            if behind[member] >= 0:
                continue
            walker = int(member)
            heads.append(walker)
            seen[walker] = True
            while ahead[walker] >= 0:
                walker = int(ahead[walker])
                seen[walker] = True
            tails.append(walker)
        for member in members:
            if seen[member]:
                continue
            walker = int(member)
            tail = int(behind[walker])
            ahead[tail] = -1
            behind[walker] = -1
            heads.append(walker)
            tails.append(tail)
            seen[walker] = True
            while ahead[walker] >= 0:
                walker = int(ahead[walker])
                seen[walker] = True
        for index, tail in enumerate(tails):
            follower = heads[(index + 1) % len(heads)]
            ahead[tail] = follower
            behind[follower] = tail
    assert not np.any(ahead < 0), \
        "%s\nthe mesh collider %s left %d half edges without a fan successor, so the " \
        "corner walk they belong to would never close" \
        % (COLLIDER_MESH_HALF_EDGE_REASON, label, int(np.count_nonzero(ahead < 0)))
    return _corner_columns(ahead, face_count)


def _corner_columns(links, face_count):
    link_face, link_corner = _half_edge_rows(links, face_count)
    columns_face = np.zeros((face_count, 3), dtype=I4)
    columns_corner = np.zeros((face_count, 3), dtype=I4)
    for slot in range(3):
        span = slice(slot * face_count, (slot + 1) * face_count)
        columns_face[:, slot] = link_face[span]
        columns_corner[:, slot] = link_corner[span]
    return (np.ascontiguousarray(columns_face, dtype=I4),
            np.ascontiguousarray(columns_corner, dtype=I4))


def _vertex_fan_seeds(label, origins, face_count, vertex_count):
    reach = np.bincount(origins.astype(np.int64), minlength=vertex_count)
    widest = int(reach.max()) if reach.shape[0] else 0
    if widest > COLLIDER_MESH_VERTEX_FAN_LIMIT:
        raise ColliderMeshRefused(
            "%s\nthe mesh collider %s has a corner where %d faces meet and the walk is "
            "compiled against a bound of %d"
            % (COLLIDER_MESH_VERTEX_FAN_REASON, label, widest,
               COLLIDER_MESH_VERTEX_FAN_LIMIT))
    seed_face = np.full(vertex_count, -1, dtype=I4)
    seed_corner = np.zeros(vertex_count, dtype=I4)
    owner_face, owner_corner = _half_edge_rows(np.arange(origins.shape[0]), face_count)
    rows = origins.astype(np.int64)
    seed_face[rows] = owner_face
    seed_corner[rows] = owner_corner
    return (np.ascontiguousarray(seed_face, dtype=I4),
            np.ascontiguousarray(seed_corner, dtype=I4))


RESERVED_TEAM_ROW_REASON = (
    "team row zero is never handed out and is therefore permanently invalid, which is what "
    "lets a released arena row be told apart from a live one: the arena zeroes every column "
    "of a row it releases, so a released collider face row carries team zero, and the "
    "family that writes the face bounds drops such a row by asking whether its team is "
    "live; the live extent of the face slab is the widest block any team holds and a "
    "released block inside it is a leaf of the grouped hierarchy whether anybody owns it or "
    "not, so if row zero ever became a live team those leaves would be given real bounds "
    "around a triangle standing on one vertex of somebody else's body and the narrow phase "
    "would answer with it; measured on a world that releases a wide body and lets a narrow "
    "one take part of the freed block, 308 rows inside the live extent belong to nobody and "
    "none of them names a live team")

LOWEST_FREE_TEAM_ROW_REASON = (
    "a team row is the identity every column that names a team is written in, so it is handed "
    "out as a function of the order teams register in and of nothing else: the free list gives "
    "up its lowest row rather than the last one returned to it, which makes two rebuilds of "
    "one scene hand every team the same row and keeps the occupied rows packed against the "
    "bottom of the table; while it was a stack the rows were returned in registration order "
    "and taken off the end, so twenty five chains of one asset took their rows in one order on "
    "one rebuild and in the reverse order on the next, with a period of two, and every column "
    "naming a team changed with them although nothing about the simulation had; the table is "
    "grown by doubling and a whole doubled block falls free at once, so the stack also left "
    "those twenty five chains sitting on rows up to thirty one with holes underneath, and the "
    "lowest free row gives them one to twenty five with none")

TEAM_ROW_GENERATION_REASON = (
    "the count of how many times a row has changed hands, which is what tells a reader that "
    "the team behind a row is not the team that was behind it before: a row that is given "
    "back and handed straight out again to a chain of the same shape carries the same starts "
    "and the same counts, so nothing in the table itself says the occupant changed, and any "
    "reader that decides from the table alone whether its copy of the world is still current "
    "will keep a copy of the previous occupant; see TEAM_ROW_IDENTITY_REASON for the "
    "reader this was measured on, and the same sentence in "
    "COLLIDER_MESH_TOPOLOGY_REASON for the collider block it was already known for")


COLLIDER_MESH_LOCAL_BOUND_REASON = (
    "the local bound of a mesh collider is what the reach it declares is measured from, "
    "and that reach is both the widest the narrow phase will ever search and the box the "
    "push out is allowed to look for a way out inside; the bound is therefore a function "
    "of the vertices and moves whenever they do, so it is written wherever they are "
    "written and never only where the binding is taken, because a body that grew would "
    "otherwise be searched inside the box it used to fill and the exit projection would "
    "stop short of a surface that is now further out")


def store_collider_mesh_bound(colliders, collider_row, vertices):
    colliders["mesh_local_bound_min"][collider_row] = vertices.min(axis=0)
    colliders["mesh_local_bound_max"][collider_row] = vertices.max(axis=0)


def _named_edges(keys, vertex_count):
    listed = sorted(set(int(value) for value in keys))[:COLLIDER_MESH_REPORTED_EDGES]
    return [(int(value // vertex_count), int(value % vertex_count)) for value in listed]


DISTANCE_FIELDS = {"team": (I4, ()), "particle": (I4, ()), "target": (I4, ()), "rest": (F4, ())}
BENDING_FIELDS = {"team": (I4, ()), "pair": (I4, (4,)), "rest": (F4, ()), "sign": (np.int8, ())}
INDEX_FIELDS = {"team": (I4, ()), "particle": (I4, ())}
EDGE_FIELDS = {"team": (I4, ()), "edge": (I4, (2,))}
TRIANGLE_FIELDS = {"team": (I4, ()), "triangle": (I4, (3,))}
V2T_FIELDS = {"team": (I4, ()), "owner": (I4, ()), "triangle": (I4, ()),
              "flip_normal": (F4, ()), "flip_tangent": (F4, ())}
PAIR_POINT_FIELDS = {"team": (I4, ()), "particle": (I4, ()), "collider": (I4, ())}
PAIR_EDGE_FIELDS = {"team": (I4, ()), "edge": (I4, ()), "collider": (I4, ())}
PRIMITIVE_FIELDS = {
    "team": (I4, ()),
    "particles": (I4, (3,)),
    "fix": (B1, (3,)), "all_fix": (B1, ()), "ignore": (B1, ()),
    "prim_depth": (F4, ()),
    "inv_mass": (F4, (3,)), "thickness": (F4, ()),
    "aabb_min": (F4, (3,)), "aabb_max": (F4, (3,)),
    "intersect": (B1, (3,)),
    "use": (B1, ()),
    "cell_key": (I8, ()),
}

PRIMITIVE_PACKED_FIELDS = ("fix", "intersect")

PRIMITIVE_DEVICE_FIELDS = {
    field_name: ((U1, ()) if field_name in PRIMITIVE_PACKED_FIELDS else specification)
    for field_name, specification in PRIMITIVE_FIELDS.items()
}

assert set(PRIMITIVE_DEVICE_FIELDS) == set(PRIMITIVE_FIELDS)
assert all(PRIMITIVE_FIELDS[field_name] == (B1, (3,))
           for field_name in PRIMITIVE_PACKED_FIELDS)


class ChunkArena:
    def __init__(self, fields, capacity=64):
        self.spec = fields
        self.capacity = 0
        self.arrays = {}
        self.free = []
        self._grow(capacity)

    def _grow(self, capacity):
        old = self.capacity
        for name, (dtype, shape) in self.spec.items():
            array = np.zeros((capacity,) + shape, dtype=dtype)
            if name in self.arrays:
                array[:old] = self.arrays[name]
            self.arrays[name] = array
        self.capacity = capacity
        self._release(old, capacity - old)

    def _release(self, start, count):
        if count <= 0:
            return
        merged = (start, count)
        result = []
        for s, l in sorted(self.free + [merged]):
            if result and result[-1][0] + result[-1][1] == s:
                result[-1] = (result[-1][0], result[-1][1] + l)
            else:
                result.append((s, l))
        self.free = result

    def alloc(self, count):
        if count == 0:
            return 0
        for k, (s, l) in enumerate(self.free):
            if l >= count:
                if l == count:
                    self.free.pop(k)
                else:
                    self.free[k] = (s + count, l - count)
                return s
        self._grow(max(self.capacity * 3 // 2 + 1, self.capacity + count))
        return self.alloc(count)

    def release(self, start, count):
        if count == 0:
            return
        for name, (dtype, shape) in self.spec.items():
            self.arrays[name][start:start + count] = np.zeros((), dtype=dtype)
        self._release(start, count)

    def __getitem__(self, name):
        return self.arrays[name]


class World:
    def __init__(self):
        self.team = np.zeros(1, dtype=TEAM_DTYPE)
        self.team_free = []
        self.team_row_generation = 0
        self.particles = ChunkArena(PARTICLE_FIELDS, 256)
        self.transforms = ChunkArena(TRANSFORM_FIELDS, 256)
        self.colliders = ChunkArena(COLLIDER_FIELDS, 32)
        self.collider_vertices = ChunkArena(COLLIDER_VERTEX_FIELDS, 32)
        self.collider_faces = ChunkArena(COLLIDER_FACE_FIELDS, 32)
        self.distance = ChunkArena(DISTANCE_FIELDS, 256)
        self.bending = ChunkArena(BENDING_FIELDS, 64)
        self.tether = ChunkArena(INDEX_FIELDS, 256)
        self.motion = ChunkArena(INDEX_FIELDS, 256)
        self.update_move = ChunkArena(INDEX_FIELDS, 256)
        self.update_fixed = ChunkArena(INDEX_FIELDS, 64)
        self.spring = ChunkArena(INDEX_FIELDS, 64)
        self.collision_process = ChunkArena(INDEX_FIELDS, 256)
        self.center_fixed = ChunkArena(INDEX_FIELDS, 64)
        self.angle_buffered = ChunkArena(INDEX_FIELDS, 256)
        self.edges = ChunkArena(EDGE_FIELDS, 256)
        self.collision_edges = ChunkArena(EDGE_FIELDS, 64)
        self.triangles = ChunkArena(TRIANGLE_FIELDS, 64)
        self.v2t = ChunkArena(V2T_FIELDS, 64)
        self.point_pairs = ChunkArena(PAIR_POINT_FIELDS, 256)
        self.edge_pairs = ChunkArena(PAIR_EDGE_FIELDS, 64)
        self.self_points = ChunkArena(PRIMITIVE_FIELDS, 256)
        self.self_edges = ChunkArena(PRIMITIVE_FIELDS, 256)
        self.self_triangles = ChunkArena(PRIMITIVE_FIELDS, 64)
        self.grids = {}
        self.contact_links = {}
        self.contacts = {"EE": None, "PT": None}
        self.buckets_dirty = True
        self.registered_setups = {}
        self.fk_levels = []
        self.angle_passes = []
        self.postline_levels = []
        self.baseline_entries = np.zeros(0, dtype=I4)
        self.entries = {}
        self.config_revision = 0
        self.structure_revision = 0
        self.contact_link_revision = 0
        self.display_planes = ()
        self.stale_planes = frozenset()

    EXTERNAL_PUBLISHER_REASON = (
        "a chain whose root hangs off a bone another config drives reads that bone out of a "
        "transform row of its own, and the row is filled by whoever owns the bone: the host "
        "puts the animation pose there every frame, and if a registered team drives that "
        "bone its solver overwrites the row with the pose it just produced, so the chain "
        "hangs off where the bone actually went instead of where the animation left it; the "
        "wiring is redone after every registration rather than once, because the team that "
        "owns the bone may be registered after the team that needs it and a rule that only "
        "works in one registration order is a rule that silently stops working")

    def note_configuration_written(self):
        self.config_revision += 1

    def note_structure_written(self):
        self.structure_revision += 1

    def note_contact_links_written(self):
        self.contact_link_revision += 1

    def _resolve_external_publishers(self):
        owner = {}
        for slot, setup in self.registered_setups.items():
            start = int(self.team[slot]["p_start"])
            for local, name in enumerate(setup.bone_names):
                owner.setdefault(name, start + local)
        publish = self.particles["publish_transform"]
        publish[:] = -1
        for slot, setup in self.registered_setups.items():
            transform_start = int(self.team[slot]["t_start"])
            for name, row in setup.external_transform_rows.items():
                source = owner.get(name)
                if source is not None:
                    publish[source] = transform_start + row
        self.note_configuration_written()

    def set_display_planes(self, planes):
        self.display_planes = tuple(sorted({(str(storage_name), str(field_name))
                                            for storage_name, field_name in planes}))

    def set_stale_planes(self, planes):
        self.stale_planes = frozenset(planes)

    def _alloc_team(self):
        if self.team_free:
            slot = heapq.heappop(self.team_free)
        else:
            slot = len(self.team)
            grown = np.zeros(max(slot * 2, 2), dtype=TEAM_DTYPE)
            grown[:slot] = self.team
            self.team = grown
            for row in range(slot + 1, len(grown)):
                heapq.heappush(self.team_free, row)
        assert slot > 0, RESERVED_TEAM_ROW_REASON
        assert not self.team_free or slot < self.team_free[0], LOWEST_FREE_TEAM_ROW_REASON
        self.team_row_generation += 1
        self.note_structure_written()
        return slot

    def prim_arena(self, kind):
        if kind == defs.KIND_POINT:
            return self.self_points
        if kind == defs.KIND_EDGE:
            return self.self_edges
        return self.self_triangles

    def register_team(self, setup, params, binding):
        validated = validated_collider_meshes(binding, "the team being registered")
        slot = self._alloc_team()
        self.entries[slot] = {"setup": setup, "collider_binding": None,
                              "point_pair": (0, 0), "edge_pair": (0, 0), "chunks": []}
        tt = self.team
        tt[slot] = np.zeros((), dtype=TEAM_DTYPE)
        row = tt[slot]
        row["valid"] = True
        row["is_spring"] = setup.spring_active
        row["reset_pending"] = True
        row["time_reset_pending"] = True
        row["sync_target"] = 0
        row["sync_top"] = 0
        row["self_contact_slots"] = defs.SELF_CONTACT_SLOTS_PER_PRIMITIVE
        row["negative_scale_matrix"] = np.eye(4)
        row["component_world_rotation"] = (0, 0, 0, 1)
        row["old_component_world_rotation"] = (0, 0, 0, 1)
        row["component_world_scale"] = 1.0
        row["old_component_world_scale"] = 1.0
        row["anchor_rotation"] = (0, 0, 0, 1)
        row["old_anchor_rotation"] = (0, 0, 0, 1)
        row["frame_component_shift_rotation"] = (0, 0, 0, 1)
        row["frame_world_rotation"] = (0, 0, 0, 1)
        row["old_frame_world_rotation"] = (0, 0, 0, 1)
        row["frame_world_scale"] = 1.0
        row["old_frame_world_scale"] = 1.0
        row["now_world_rotation"] = (0, 0, 0, 1)
        row["old_world_rotation"] = (0, 0, 0, 1)
        row["step_rotation"] = (0, 0, 0, 1)
        row["inertia_rotation"] = (0, 0, 0, 1)
        row["velocity_weight"] = 1.0
        row["blend_weight"] = 1.0
        row["distance_weight"] = 1.0
        row["gravity_ratio"] = 1.0
        row["gravity_dot"] = 1.0
        row["scale_ratio"] = 1.0
        row["now_time_scale"] = 1.0
        row["frame_interpolation"] = 1.0
        row["init_local_gravity_direction"] = setup.init_local_gravity_direction
        row["init_scale"] = setup.init_scale.astype(F4)
        row["moving_wind_time"] = -defs.WIND_MAX_TIME
        row["moving_wind_dirq"] = (0, 0, 0, 1)
        row["wind_seed"] = setup.wind_seed

        n = len(setup.bone_names)
        ps = self.particles.alloc(n)
        row["p_start"] = ps
        row["p_count"] = n
        pa = self.particles
        s = slice(ps, ps + n)
        pa["team"][s] = slot
        attrs = setup.attributes
        pa["attr_move"][s] = (attrs & defs.ATTR_MOVE) != 0
        pa["attr_fixed"][s] = (attrs & defs.ATTR_FIXED) != 0
        pa["attr_invalid"][s] = ((attrs & defs.ATTR_MOVE) == 0) & ((attrs & defs.ATTR_FIXED) == 0)
        pa["attr_disable_collision"][s] = (attrs & defs.ATTR_DISABLE_COLLISION) != 0
        pa["attr_motion"][s] = (attrs & defs.ATTR_INVALID_MOTION) == 0
        pa["attr_zero_distance"][s] = (attrs & defs.ATTR_ZERO_DISTANCE) != 0
        pa["depth"][s] = setup.vertex_depth
        pa["vertex_parent"][s] = np.where(setup.vertex_parent >= 0, setup.vertex_parent + ps, -1)
        pa["vertex_root"][s] = np.where(setup.vertex_root >= 0, setup.vertex_root + ps, -1)
        pa["vertex_root_local"][s] = setup.vertex_root
        pa["local_positions"][s] = setup.local_positions
        pa["local_normals"][s] = setup.local_normals
        pa["local_tangents"][s] = setup.local_tangents
        pa["uv"][s] = setup.uv
        pa["vertex_local_positions"][s] = setup.vertex_local_positions
        pa["vertex_local_rotations"][s] = setup.vertex_local_rotations
        pa["vertex_bind_pose_rotations"][s] = setup.vertex_bind_pose_rotations
        pa["vertex_to_transform_rotations"][s] = setup.vertex_to_transform_rotations
        pa["normal_adjustment_rotations"][s] = setup.normal_adjustment_rotations
        pa["rotations"][s] = host_math.quat_identity((n,))
        pa["old_rotations"][s] = host_math.quat_identity((n,))
        pa["base_rotations"][s] = host_math.quat_identity((n,))
        pa["old_anim_rotations"][s] = host_math.quat_identity((n,))
        pa["step_basic_rotations"][s] = host_math.quat_identity((n,))
        pa["temp_base_rotations"][s] = host_math.quat_identity((n,))
        pa["albuf_local_rot"][s] = host_math.quat_identity((n,))
        pa["albuf_rotation"][s] = host_math.quat_identity((n,))
        pa["out_rotations"][s] = host_math.quat_identity((n,))

        tc = len(setup.transform_names)
        ts = self.transforms.alloc(tc)
        row["t_start"] = ts
        row["t_count"] = tc
        self.transforms["team"][ts:ts + tc] = slot
        self.transforms["bind_pose"][ts:ts + tc] = setup.transform_bind_pose
        pa["skin_indices"][s] = setup.skin_indices + ts
        pa["skin_weights"][s] = setup.skin_weights
        pa["aim_child"][s] = np.where(setup.aim_child >= 0, setup.aim_child + ps, -1)
        pa["aim_rest_reach"][s] = setup.aim_rest_reach
        pa["publish_transform"][s] = setup.publish_transform
        pa["bone_row"][s] = np.arange(n, dtype=I4) + ts
        moves = (setup.attributes & defs.ATTR_MOVE) != 0
        alive = moves | ((setup.attributes & defs.ATTR_FIXED) != 0)
        pa["publish_position"][s] = (moves | setup.spring_active) & alive
        self.registered_setups[slot] = setup
        self._resolve_external_publishers()

        self._alloc_entries(self.distance, slot, len(setup.distance_particle), {
            "particle": setup.distance_particle + ps,
            "target": setup.distance_target + ps,
            "rest": setup.distance_rest,
        })
        self._alloc_entries(self.bending, slot, len(setup.bending_pairs), {
            "pair": setup.bending_pairs + ps,
            "rest": setup.bending_rest,
            "sign": setup.bending_sign,
        })
        self._alloc_entries(self.tether, slot, len(setup.tether_index), {"particle": setup.tether_index + ps})
        self._alloc_entries(self.motion, slot, len(setup.motion_index), {"particle": setup.motion_index + ps})
        self._alloc_entries(self.update_move, slot, len(setup.update_move_index), {"particle": setup.update_move_index + ps})
        self._alloc_entries(self.update_fixed, slot, len(setup.update_fixed_index), {"particle": setup.update_fixed_index + ps})
        self._alloc_entries(self.spring, slot, len(setup.spring_index), {"particle": setup.spring_index + ps})
        self._alloc_entries(self.collision_process, slot, len(setup.collision_process_index), {"particle": setup.collision_process_index + ps})
        self._alloc_entries(self.center_fixed, slot, len(setup.center_fixed_list), {"particle": setup.center_fixed_list + ps})
        self._alloc_entries(self.angle_buffered, slot, len(setup.angle_buffered), {"particle": setup.angle_buffered + ps})
        self._alloc_entries(self.edges, slot, len(setup.edges), {"edge": setup.edges + ps})
        ce_start = self._alloc_entries(self.collision_edges, slot, len(setup.collision_edge_index),
                                       {"edge": setup.edges[setup.collision_edge_index] + ps} if len(setup.collision_edge_index) else {})
        tri_start = self._alloc_entries(self.triangles, slot, len(setup.triangles), {"triangle": setup.triangles + ps})
        self._alloc_entries(self.v2t, slot, len(setup.v2t_owner), {
            "owner": setup.v2t_owner + ps,
            "triangle": setup.v2t_triangle + tri_start,
            "flip_normal": setup.v2t_flip_normal,
            "flip_tangent": setup.v2t_flip_tangent,
        })

        sp = self.self_points.alloc(n)
        row["sp_start"] = sp
        row["sp_count"] = n
        self._init_primitives(self.self_points, sp, slot,
                              np.stack([np.arange(n, dtype=I4), np.full(n, -1, I4), np.full(n, -1, I4)], axis=1),
                              setup, ps, 1)
        ne = len(setup.edges)
        se = self.self_edges.alloc(ne)
        row["se_start"] = se
        row["se_count"] = ne
        if ne:
            self._init_primitives(self.self_edges, se, slot,
                                  np.concatenate([setup.edges.astype(I4), np.full((ne, 1), -1, I4)], axis=1),
                                  setup, ps, 2)
        nt = len(setup.triangles)
        st = self.self_triangles.alloc(nt)
        row["st_start"] = st
        row["st_count"] = nt
        if nt:
            self._init_primitives(self.self_triangles, st, slot, setup.triangles.astype(I4), setup, ps, 3)

        self.update_params(slot, params)
        self._apply_colliders(slot, binding, validated)
        self.buckets_dirty = True
        for grid in self.grids.values():
            grid.valid = False
        self.contacts = {"EE": None, "PT": None}
        self.note_structure_written()
        return slot

    def _alloc_entries(self, arena, slot, count, data):
        start = arena.alloc(count)
        if count:
            arena["team"][start:start + count] = slot
            for name, value in data.items():
                arena[name][start:start + count] = value
        self.entries[slot]["chunks"].append((arena, start, count))
        return start

    def _init_primitives(self, arena, start, slot, particles_local, setup, ps, axis_count):
        count = len(particles_local)
        s = slice(start, start + count)
        arena["team"][s] = slot
        particles = np.where(particles_local >= 0, particles_local + ps, -1)
        arena["particles"][s] = particles
        move = (setup.attributes & defs.ATTR_MOVE) != 0
        invalid = ((setup.attributes & defs.ATTR_MOVE) == 0) & ((setup.attributes & defs.ATTR_FIXED) == 0)
        gathered = particles_local.copy()
        gathered[gathered < 0] = 0
        fix = ~move[gathered]
        ignore = invalid[gathered]
        for axis in range(axis_count, 3):
            fix[:, axis] = True
            ignore[:, axis] = False
        arena["fix"][s] = fix
        arena["all_fix"][s] = fix[:, :axis_count].all(axis=1)
        arena["ignore"][s] = ignore[:, :axis_count].any(axis=1)
        arena["prim_depth"][s] = setup.vertex_depth[gathered][:, :axis_count].mean(axis=1) if count else 0.0
        arena["cell_key"][s] = defs.GRID_KEY_IGNORE

    def unregister_team(self, slot):
        entry = self.entries.pop(slot, None)
        if entry is None:
            return
        self.registered_setups.pop(slot, None)
        row = self.team[slot]
        self.particles.release(int(row["p_start"]), int(row["p_count"]))
        self.transforms.release(int(row["t_start"]), int(row["t_count"]))
        self.colliders.release(int(row["c_start"]), int(row["c_count"]))
        self.collider_vertices.release(int(row["cv_start"]), int(row["cv_count"]))
        self.collider_faces.release(int(row["cf_start"]), int(row["cf_count"]))
        self.self_points.release(int(row["sp_start"]), int(row["sp_count"]))
        self.self_edges.release(int(row["se_start"]), int(row["se_count"]))
        self.self_triangles.release(int(row["st_start"]), int(row["st_count"]))
        for arena, start, count in entry.get("chunks", []):
            arena.release(start, count)
        start, count = entry.get("point_pair", (0, 0))
        self.point_pairs.release(start, count)
        start, count = entry.get("edge_pair", (0, 0))
        self.edge_pairs.release(start, count)
        self.team[slot] = np.zeros((), dtype=TEAM_DTYPE)
        self.note_configuration_written()
        heapq.heappush(self.team_free, slot)
        self.team_row_generation += 1
        self.note_structure_written()
        if self.contact_links.pop(slot, None) is not None:
            self.note_contact_links_written()
        self._resolve_external_publishers()
        self.buckets_dirty = True
        for grid in self.grids.values():
            grid.valid = False
        self.contacts = {"EE": None, "PT": None}

    def update_params(self, slot, params):
        row = self.team[slot]
        for name, value in params.items():
            row[name] = value
        self.note_configuration_written()
        if set(params) & set(STRUCTURE_TEAM_COLUMNS):
            self.note_structure_written()

    def update_colliders(self, slot, binding):
        self._apply_colliders(
            slot, binding, validated_collider_meshes(binding, "team %d" % slot))

    def _apply_colliders(self, slot, binding, validated):
        entry = self.entries[slot]
        row = self.team[slot]
        self.colliders.release(int(row["c_start"]), int(row["c_count"]))
        self.collider_vertices.release(int(row["cv_start"]), int(row["cv_count"]))
        self.collider_faces.release(int(row["cf_start"]), int(row["cf_count"]))
        start, count = entry["point_pair"]
        self.point_pairs.release(start, count)
        start, count = entry["edge_pair"]
        self.edge_pairs.release(start, count)
        entry["collider_binding"] = binding
        c = binding.count
        cs = self.colliders.alloc(c)
        row["c_start"] = cs
        row["c_count"] = c
        if c:
            ca = self.colliders
            s = slice(cs, cs + c)
            ca["team"][s] = slot
            ca["kind"][s] = binding.kinds
            ca["frame_rotations"][s] = host_math.quat_identity((c,))
            ca["old_frame_rotations"][s] = host_math.quat_identity((c,))
            ca["now_rotations"][s] = host_math.quat_identity((c,))
            ca["old_rotations"][s] = host_math.quat_identity((c,))
            ca["input_rotations"][s] = host_math.quat_identity((c,))
            ca["work_rot"][s] = host_math.quat_identity((c,))
            ca["work_inv_old_rot"][s] = host_math.quat_identity((c,))
            ca["work_inv_rot"][s] = host_math.quat_identity((c,))
        self._store_collider_meshes(slot, cs, validated)
        setup = entry["setup"]
        ps = int(row["p_start"])
        np_proc = len(setup.collision_process_index)
        pair_count = np_proc * c
        pp = self.point_pairs.alloc(pair_count)
        entry["point_pair"] = (pp, pair_count)
        if pair_count:
            self.point_pairs["team"][pp:pp + pair_count] = slot
            self.point_pairs["particle"][pp:pp + pair_count] = np.repeat(setup.collision_process_index + ps, c)
            self.point_pairs["collider"][pp:pp + pair_count] = np.tile(np.arange(cs, cs + c, dtype=I4), np_proc)
        ne = len(setup.collision_edge_index)
        epair_count = ne * c
        ep = self.edge_pairs.alloc(epair_count)
        entry["edge_pair"] = (ep, epair_count)
        if epair_count:
            edge_chunk_start = None
            for arena, start, count in entry.get("chunks", []):
                if arena is self.collision_edges:
                    edge_chunk_start = start
                    break
            self.edge_pairs["team"][ep:ep + epair_count] = slot
            self.edge_pairs["edge"][ep:ep + epair_count] = np.repeat(
                np.arange(edge_chunk_start, edge_chunk_start + ne, dtype=I4), c)
            self.edge_pairs["collider"][ep:ep + epair_count] = np.tile(np.arange(cs, cs + c, dtype=I4), ne)
        self.note_structure_written()

    def _store_collider_meshes(self, slot, collider_start, validated):
        row = self.team[slot]
        blocks, vertex_total, face_total = validated
        vertex_start = self.collider_vertices.alloc(vertex_total)
        face_start = self.collider_faces.alloc(face_total)
        row["cv_start"] = vertex_start
        row["cv_count"] = vertex_total
        row["cf_start"] = face_start
        row["cf_count"] = face_total
        vertex_cursor = vertex_start
        face_cursor = face_start
        for index, block in enumerate(blocks):
            collider_row = collider_start + index
            if block is None:
                continue
            vertices, faces, ring, fan, seeds = block
            ring_face, ring_corner = ring
            fan_next_face, fan_next_corner = fan
            seed_face, seed_corner = seeds
            vertex_count = int(vertices.shape[0])
            face_count = int(faces.shape[0])
            vertex_span = slice(vertex_cursor, vertex_cursor + vertex_count)
            face_span = slice(face_cursor, face_cursor + face_count)
            self.collider_vertices["team"][vertex_span] = slot
            self.collider_vertices["collider"][vertex_span] = collider_row
            self.collider_vertices["local_position"][vertex_span] = vertices
            self.collider_vertices["fan_face"][vertex_span] = \
                np.where(seed_face < 0, seed_face, seed_face + face_cursor)
            self.collider_vertices["fan_corner"][vertex_span] = seed_corner
            self.collider_faces["team"][face_span] = slot
            self.collider_faces["collider"][face_span] = collider_row
            self.collider_faces["vertex"][face_span] = faces + vertex_cursor
            self.collider_faces["fan_next_face"][face_span] = fan_next_face + face_cursor
            self.collider_faces["fan_next_corner"][face_span] = fan_next_corner
            self.collider_faces["edge_ring_face"][face_span] = ring_face + face_cursor
            self.collider_faces["edge_ring_corner"][face_span] = ring_corner
            corners = vertices[faces]
            self.collider_faces["aabb_min"][face_span] = corners.min(axis=1)
            self.collider_faces["aabb_max"][face_span] = corners.max(axis=1)
            store_collider_mesh_bound(self.colliders, collider_row, vertices)
            self.colliders["mesh_vertex_start"][collider_row] = vertex_cursor
            self.colliders["mesh_vertex_count"][collider_row] = vertex_count
            self.colliders["mesh_face_start"][collider_row] = face_cursor
            self.colliders["mesh_face_count"][collider_row] = face_count
            vertex_cursor += vertex_count
            face_cursor += face_count

    def ensure_buckets(self):
        if not self.buckets_dirty:
            return
        self.buckets_dirty = False
        fk = {}
        angle = {}
        postline = {}
        baseline_entries = []
        for slot, entry in self.entries.items():
            setup = entry["setup"]
            ps = int(self.team[slot]["p_start"])
            if len(setup.baseline_data):
                baseline_entries.append(setup.baseline_data + ps)
            for depth, (yes, yes_parent, no) in enumerate(setup.fk_levels):
                bucket = fk.setdefault(depth, [[], [], []])
                bucket[0].append(yes + ps)
                bucket[1].append(yes_parent + ps)
                bucket[2].append(no + ps)
            for key, (vertices, parents) in setup.angle_passes:
                bucket = angle.setdefault(key, [[], []])
                bucket[0].append(vertices + ps)
                bucket[1].append(parents + ps)
            for depth, (entry_vertex, child_owner, child_vertex) in enumerate(setup.postline_levels):
                bucket = postline.setdefault(depth, [[], [], [], 0])
                offset = bucket[3]
                bucket[0].append(entry_vertex + ps)
                bucket[1].append(child_owner + offset)
                bucket[2].append(child_vertex + ps)
                bucket[3] = offset + len(entry_vertex)
        self.fk_levels = []
        for depth in sorted(fk.keys()):
            yes, yes_parent, no = fk[depth]
            self.fk_levels.append((
                np.concatenate(yes) if yes else np.zeros(0, I4),
                np.concatenate(yes_parent) if yes_parent else np.zeros(0, I4),
                np.concatenate(no) if no else np.zeros(0, I4),
            ))
        self.angle_passes = []
        for key in sorted(angle.keys()):
            vertices, parents = angle[key]
            self.angle_passes.append((np.concatenate(vertices), np.concatenate(parents)))
        self.postline_levels = []
        for depth in sorted(postline.keys()):
            entry_vertex, child_owner, child_vertex, _ = postline[depth]
            self.postline_levels.append((
                np.concatenate(entry_vertex) if entry_vertex else np.zeros(0, I4),
                np.concatenate(child_owner) if child_owner else np.zeros(0, I4),
                np.concatenate(child_vertex) if child_vertex else np.zeros(0, I4),
            ))
        self.baseline_entries = np.concatenate(baseline_entries).astype(I4) \
            if baseline_entries else np.zeros(0, dtype=I4)

    def request_reset(self, slot, mode):
        row = self.team[slot]
        if mode == 'FULL':
            row["reset_pending"] = True
            row["time_reset_pending"] = True
        else:
            row["keep_teleport_pending"] = True

    def add_force(self, slot, mode, force):
        row = self.team[slot]
        row["force_mode"] = mode
        row["impact_force"] = force

    def particle_slice(self, slot):
        row = self.team[slot]
        return slice(int(row["p_start"]), int(row["p_start"]) + int(row["p_count"]))

    def collider_slice(self, slot):
        row = self.team[slot]
        return slice(int(row["c_start"]), int(row["c_start"]) + int(row["c_count"]))

    def transform_slice(self, slot):
        row = self.team[slot]
        return slice(int(row["t_start"]), int(row["t_start"]) + int(row["t_count"]))
