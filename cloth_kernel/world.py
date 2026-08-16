import numpy as np

from . import defs
from . import math as pm

F4 = np.float32
F8 = np.float64
I4 = np.int32
I8 = np.int64
B1 = np.bool_

TEAM_DTYPE = np.dtype([
    ("valid", B1), ("enabled", B1), ("is_spring", B1), ("running", B1),
    ("reset_pending", B1), ("time_reset_pending", B1), ("keep_teleport_pending", B1),
    ("inertia_shift", B1), ("negative_scale_teleport", B1), ("is_negative_scale", B1),
    ("culling_invisible", B1), ("has_anchor", B1), ("had_anchor", B1), ("step_active", B1),
    ("sync_target", I4), ("sync_top", I4), ("wind_seed", I4),
    ("p_start", I4), ("p_count", I4), ("t_start", I4), ("t_count", I4),
    ("c_start", I4), ("c_count", I4),
    ("sp_start", I4), ("sp_count", I4), ("se_start", I4), ("se_count", I4),
    ("st_start", I4), ("st_count", I4),
    ("use_point", B1), ("use_edge", B1), ("use_triangle", B1),
    ("self_grid_size", F4), ("self_max_primitive_size", F4),
    ("time", F4), ("old_time", F4), ("now_update_time", F4), ("old_update_time", F4),
    ("frame_update_time", F4), ("frame_old_time", F4), ("frame_delta_time", F4),
    ("now_time_scale", F4), ("update_count", I4), ("skip_count", I4),
    ("frame_interpolation", F4), ("velocity_weight", F4), ("blend_weight", F4),
    ("distance_weight", F4), ("gravity_ratio", F4), ("gravity_dot", F4), ("scale_ratio", F4),
    ("negative_scale_sign", F4), ("negative_scale_direction", F4, (3,)),
    ("negative_scale_change", F4, (3,)), ("negative_scale_quaternion", F4, (4,)),
    ("negative_scale_triangle_sign", F4, (2,)), ("negative_scale_matrix", F8, (4, 4)),
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
    ("wind_direction", F4, (defs.WIND_ZONE_SLOTS, 3)),
    ("wind_zone_turbulence", F4, (defs.WIND_ZONE_SLOTS,)),
    ("wind_dirq", F4, (defs.WIND_ZONE_SLOTS, 4)),
    ("moving_wind_time", F4), ("moving_wind_main", F4),
    ("moving_wind_direction", F4, (3,)), ("moving_wind_dirq", F4, (4,)),
])

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
}

TRANSFORM_FIELDS = {
    "team": (I4, ()),
    "bind_pose": (F4, (4, 4)),
    "world": (F4, (4, 4)),
}

COLLIDER_FIELDS = {
    "team": (I4, ()),
    "kind": (I4, ()),
    "center": (F4, (3,)), "size": (F4, (3,)), "axis": (F4, (3,)), "aligned": (B1, ()),
    "enabled": (B1, ()),
    "enabled_prev": (B1, ()),
    "active": (B1, ()),
    "input_positions": (F4, (3,)), "input_rotations": (F4, (4,)), "input_scales": (F4, (3,)),
    "frame_positions": (F4, (3,)), "frame_rotations": (F4, (4,)), "frame_scales": (F4, (3,)),
    "old_frame_positions": (F4, (3,)), "old_frame_rotations": (F4, (4,)),
    "now_positions": (F4, (3,)), "now_rotations": (F4, (4,)),
    "old_positions": (F4, (3,)), "old_rotations": (F4, (4,)),
    "work_radius": (F4, (2,)), "work_old_pos": (F4, (2, 3)), "work_next_pos": (F4, (2, 3)),
    "work_rot": (F4, (4,)), "work_inv_old_rot": (F4, (4,)),
    "work_aabb_min": (F4, (3,)), "work_aabb_max": (F4, (3,)),
}

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


class GridState:
    def __init__(self):
        self.valid = False
        self.order = None
        self.sorted_key = None
        self.sorted_team = None
        self.segment_start = None
        self.segment_count = None
        self.segment_key = None
        self.segment_team = None


class World:
    def __init__(self):
        self.team = np.zeros(1, dtype=TEAM_DTYPE)
        self.team_free = []
        self.particles = ChunkArena(PARTICLE_FIELDS, 256)
        self.transforms = ChunkArena(TRANSFORM_FIELDS, 256)
        self.colliders = ChunkArena(COLLIDER_FIELDS, 32)
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
        self.grids = {defs.KIND_POINT: GridState(), defs.KIND_EDGE: GridState(),
                      defs.KIND_TRIANGLE: GridState()}
        self.contacts = {"EE": None, "PT": None}
        self.buckets_dirty = True
        self.fk_levels = []
        self.angle_passes = []
        self.postline_levels = []
        self.baseline_entries = np.zeros(0, dtype=I4)
        self.entries = {}

    def _alloc_team(self):
        if self.team_free:
            return self.team_free.pop()
        slot = len(self.team)
        grown = np.zeros(max(slot * 2, 2), dtype=TEAM_DTYPE)
        grown[:slot] = self.team
        self.team = grown
        for k in range(slot + 1, len(grown)):
            self.team_free.append(k)
        return slot

    def prim_arena(self, kind):
        if kind == defs.KIND_POINT:
            return self.self_points
        if kind == defs.KIND_EDGE:
            return self.self_edges
        return self.self_triangles

    def register_team(self, setup, params, binding):
        slot = self._alloc_team()
        self.entries[slot] = {"setup": setup, "collider_binding": None,
                              "point_pair": (0, 0), "edge_pair": (0, 0), "chunks": []}
        tt = self.team
        tt[slot] = np.zeros((), dtype=TEAM_DTYPE)
        row = tt[slot]
        row["valid"] = True
        row["is_spring"] = setup.is_spring
        row["reset_pending"] = True
        row["time_reset_pending"] = True
        row["sync_target"] = 0
        row["sync_top"] = 0
        row["negative_scale_sign"] = 1.0
        row["negative_scale_direction"] = 1.0
        row["negative_scale_change"] = 1.0
        row["negative_scale_quaternion"] = 1.0
        row["negative_scale_triangle_sign"] = 1.0
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
        pa["rotations"][s] = pm.quat_identity((n,))
        pa["old_rotations"][s] = pm.quat_identity((n,))
        pa["base_rotations"][s] = pm.quat_identity((n,))
        pa["old_anim_rotations"][s] = pm.quat_identity((n,))
        pa["step_basic_rotations"][s] = pm.quat_identity((n,))
        pa["temp_base_rotations"][s] = pm.quat_identity((n,))
        pa["albuf_local_rot"][s] = pm.quat_identity((n,))
        pa["albuf_rotation"][s] = pm.quat_identity((n,))
        pa["out_rotations"][s] = pm.quat_identity((n,))

        tc = len(setup.transform_names)
        ts = self.transforms.alloc(tc)
        row["t_start"] = ts
        row["t_count"] = tc
        self.transforms["team"][ts:ts + tc] = slot
        self.transforms["bind_pose"][ts:ts + tc] = setup.transform_bind_pose
        pa["skin_indices"][s] = setup.skin_indices + ts
        pa["skin_weights"][s] = setup.skin_weights

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
        self.update_colliders(slot, binding)
        self.buckets_dirty = True
        for grid in self.grids.values():
            grid.valid = False
        self.contacts = {"EE": None, "PT": None}
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
        row = self.team[slot]
        self.particles.release(int(row["p_start"]), int(row["p_count"]))
        self.transforms.release(int(row["t_start"]), int(row["t_count"]))
        self.colliders.release(int(row["c_start"]), int(row["c_count"]))
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
        self.team_free.append(slot)
        self.buckets_dirty = True
        for grid in self.grids.values():
            grid.valid = False
        self.contacts = {"EE": None, "PT": None}

    def update_params(self, slot, params):
        row = self.team[slot]
        for name, value in params.items():
            row[name] = value

    def update_colliders(self, slot, binding):
        entry = self.entries[slot]
        row = self.team[slot]
        self.colliders.release(int(row["c_start"]), int(row["c_count"]))
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
            ca["center"][s] = binding.centers
            ca["size"][s] = binding.sizes
            ca["axis"][s] = binding.axes
            ca["aligned"][s] = binding.aligned
            ca["frame_rotations"][s] = pm.quat_identity((c,))
            ca["old_frame_rotations"][s] = pm.quat_identity((c,))
            ca["now_rotations"][s] = pm.quat_identity((c,))
            ca["old_rotations"][s] = pm.quat_identity((c,))
            ca["input_rotations"][s] = pm.quat_identity((c,))
            ca["work_rot"][s] = pm.quat_identity((c,))
            ca["work_inv_old_rot"][s] = pm.quat_identity((c,))
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
