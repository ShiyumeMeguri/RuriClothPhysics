import numpy as np
from numba import cuda, int32

from cloth_kernel import defs as _defs

from . import device
from . import kernels
from . import staging
from .program import build_program

_MAX_COOP_BLOCKS = 544
_THREADS = 128
_SELF_PAIRS_PER_THREAD = 64

_ZONE_MODE = {"GLOBAL_DIRECTION": 0, "BOX_DIRECTION": 1,
              "SPHERE_DIRECTION": 2, "SPHERE_RADIAL": 3}
_ZONE_ARG_ORDER = ("zone_id", "mode", "is_addition", "main", "turbulence",
                   "world_position", "world_direction", "world_to_local",
                   "size", "zone_volume", "attenuation_lut")

_SELF_TASK_EE = 0
_SELF_TASK_PT = 1
_SELF_PAIR_GUARD = 30_000_000

_STRUCT_TEAM_COLUMNS = ("valid", "is_spring", "p_start", "p_count", "t_start", "t_count",
                        "c_start", "c_count", "sp_start", "sp_count",
                        "se_start", "se_count", "st_start", "st_count")

_INPUT_TEAM_FIELDS = ("enabled", "component_world_position", "component_world_rotation",
                      "component_world_scale", "culling_invisible", "distance_weight",
                      "sync_target", "has_anchor", "anchor_position", "anchor_rotation")
_CONSUMABLE_TEAM_FIELDS = ("reset_pending", "time_reset_pending", "keep_teleport_pending",
                           "force_mode", "impact_force")
_INPUT_UPLOAD_FIELDS = _INPUT_TEAM_FIELDS + _CONSUMABLE_TEAM_FIELDS
_CONFIG_TEAM_FIELDS = (
    "gravity", "gravity_direction", "gravity_falloff", "stablization_time",
    "blend_weight_param", "damping_lut", "radius_lut", "normal_axis_vector",
    "rotational_interpolation", "root_rotation", "animation_pose_ratio", "time_scale",
    "tether_compression", "distance_lut", "bending_stiffness", "angle_use_restoration",
    "angle_restoration_lut", "angle_restoration_attenuation", "angle_restoration_gravity_falloff",
    "angle_use_limit", "angle_limit_lut", "angle_limit_stiffness", "motion_use_max_distance",
    "motion_max_distance_lut", "motion_use_backstop", "motion_backstop_radius",
    "motion_backstop_lut", "motion_stiffness", "collision_mode", "dynamic_friction",
    "static_friction", "limit_distance_lut", "self_mode", "sync_mode", "self_thickness_lut",
    "self_cloth_mass", "anchor_inertia", "world_inertia", "movement_inertia_smoothing",
    "movement_speed_limit", "rotation_speed_limit", "local_inertia", "local_movement_speed_limit",
    "local_rotation_speed_limit", "depth_inertia", "centrifugal_acceleration",
    "particle_speed_limit", "teleport_mode", "teleport_distance", "teleport_rotation",
    "wind_influence", "wind_frequency", "wind_turbulence", "wind_blend", "wind_synchronization",
    "wind_depth_weight", "wind_moving", "spring_power", "spring_limit_distance",
    "spring_normal_limit_ratio", "spring_noise")
_COLLIDER_INPUT_FIELDS = ("input_positions", "input_rotations", "input_scales", "enabled")
_SLIM_OUTPUT_PARTICLE_FIELDS = ("positions", "out_rotations")

assert not (set(_INPUT_UPLOAD_FIELDS) & set(_CONFIG_TEAM_FIELDS))
assert len(set(_INPUT_UPLOAD_FIELDS)) == len(_INPUT_UPLOAD_FIELDS)


def _arena_subset_dtype(spec, fields):
    items = []
    for name in fields:
        dtype, shape = spec[name]
        items.append((name, dtype, shape) if shape else (name, dtype))
    return np.dtype(items)


class GpuEngine:
    def __init__(self, world):
        self.world = None
        self.signature = None
        self.program = None
        self.team = None
        self.particles = None
        self.colliders = None
        self.transforms = None
        self.input_staging = None
        self.feedback_staging = None
        self.config_staging = None
        self.collider_input_staging = None
        self.particle_out_staging = None
        self._config_shadow = None
        self._zone_shadow = None
        self.self_points = None
        self.self_edges = None
        self.self_triangles = None
        self.self_state = {}
        self._self_empty_uploaded = True
        self._self_max_pairs = 0
        self._self_upload_shadow = None
        self._self_upload_totals = (0, 0)
        self._coop_max_blocks = None
        self.load(world)

    @staticmethod
    def _structure_signature(world):
        team = world.team
        parts = [len(team).to_bytes(8, "little")]
        for name in _STRUCT_TEAM_COLUMNS:
            parts.append(np.ascontiguousarray(team[name]).tobytes())
        colliders = world.colliders.arrays
        for name in ("team", "kind", "center", "size", "axis", "aligned"):
            parts.append(colliders[name].tobytes())
        point_pairs = world.point_pairs.arrays
        for name in ("team", "particle", "collider"):
            parts.append(point_pairs[name].tobytes())
        edge_pairs = world.edge_pairs.arrays
        for name in ("team", "edge", "collider"):
            parts.append(edge_pairs[name].tobytes())
        return b"".join(parts)

    _SCRATCH_SPECS = (
        ("dcorr", ("p", 3), np.float32),
        ("dcorr_fixed", ("p", 3), np.int32),
        ("dcount", ("p",), np.int32),
        ("col_friction_fixed", ("p",), np.int32),
        ("col_normal_fixed", ("p", 3), np.int32),
        ("sync_snapshot", ("t", 22), np.float32),
        ("tri_normal_f64", ("tri", 3), np.float64),
        ("tri_tangent_f64", ("tri", 3), np.float64),
    )

    @staticmethod
    def _dump_primitive(arena, count):
        a = arena.arrays

        def pack(field):
            m = a[field][:count].astype(np.uint8)
            if count == 0:
                return np.zeros(0, np.uint8)
            return np.ascontiguousarray(m[:, 0] | (m[:, 1] << 1) | (m[:, 2] << 2))

        u8 = lambda name: np.ascontiguousarray(a[name][:count].astype(np.uint8))
        raw = lambda name: np.ascontiguousarray(a[name][:count])
        return {"team": raw("team"), "particles": raw("particles"), "fix": pack("fix"),
                "all_fix": u8("all_fix"), "ignore": u8("ignore"), "prim_depth": raw("prim_depth"),
                "inv_mass": raw("inv_mass"), "thickness": raw("thickness"),
                "aabb_min": raw("aabb_min"), "aabb_max": raw("aabb_max"),
                "intersect": pack("intersect"), "use": u8("use")}

    def _self_state_ordered(self):
        p = self.program
        nt = max(p.num_teams, 1)
        ce, cp, ci = p.self_cap_ee, p.self_cap_pt, p.self_cap_ip
        mct, mit = p.self_max_contact_tasks, p.self_max_intersect_tasks
        i32 = lambda n: np.zeros(int(n), np.int32)
        f32 = lambda n: np.zeros(int(n), np.float32)
        u8 = lambda n: np.zeros(int(n), np.uint8)
        return [
            ("ee_my", i32(ce)), ("ee_target", i32(ce)), ("ee_thickness", f32(ce)),
            ("ee_s", f32(ce)), ("ee_t", f32(ce)), ("ee_n", np.zeros((int(ce), 3), np.float32)),
            ("ee_enable", u8(ce)),
            ("pt_my", i32(cp)), ("pt_target", i32(cp)), ("pt_thickness", f32(cp)),
            ("pt_sign", f32(cp)), ("pt_enable", u8(cp)),
            ("scl_counts", i32(8)),
            ("ct_kind", i32(mct)), ("ct_my_team", i32(mct)), ("ct_my_start", i32(mct)),
            ("ct_my_count", i32(mct)), ("ct_tgt_team", i32(mct)), ("ct_tgt_start", i32(mct)),
            ("ct_tgt_count", i32(mct)), ("ct_same", u8(mct)), ("ct_pair_off", i32(mct + 1)),
            ("it_edge_team", i32(mit)), ("it_edge_start", i32(mit)), ("it_edge_count", i32(mit)),
            ("it_tri_team", i32(mit)), ("it_tri_start", i32(mit)), ("it_tri_count", i32(mit)),
            ("it_same", u8(mit)), ("it_pair_off", i32(mit + 1)),
            ("ip_edge", i32(ci)), ("ip_tri", i32(ci)),
            ("scl_max_fixed", i32(nt)),
        ]

    def load(self, world):
        signature = self._structure_signature(world)
        if self.world is world and self.signature == signature:
            return
        self.world = world
        self.signature = signature
        self.program = build_program(world)
        num_teams = self.program.num_teams
        num_particles = self.program.num_particles
        num_colliders = self.program.num_colliders
        num_transforms = self.program.num_transforms

        team_host = device.dump_struct(world.team, num_teams)
        particle_host = device.dump_arena(world.particles, num_particles)
        collider_host = device.dump_arena(world.colliders, num_colliders)
        transform_host = device.dump_arena(world.transforms, num_transforms)
        self.team = device.FieldSet(team_host, num_teams, allocate=False)
        self.particles = device.FieldSet(particle_host, num_particles, allocate=False)
        self.colliders = device.FieldSet(collider_host, num_colliders, allocate=False)
        self.transforms = device.FieldSet(transform_host, num_transforms, allocate=False)
        self.static = {}

        ordered = []
        targets = []
        for name in kernels.TEAM_KERNEL_FIELDS:
            ordered.append(team_host[name]); targets.append((self.team, name))
        for name in kernels.PARTICLE_KERNEL_FIELDS:
            ordered.append(particle_host[name]); targets.append((self.particles, name))
        for name in kernels.TRANSFORM_KERNEL_FIELDS:
            ordered.append(transform_host[name]); targets.append((self.transforms, name))
        for name in kernels.COLLIDER_KERNEL_FIELDS:
            ordered.append(collider_host[name]); targets.append((self.colliders, name))
        for kernel_name, attr, field in kernels.STATIC_KERNEL_FIELDS:
            ordered.append(getattr(self.program, attr)[field]); targets.append((self.static, kernel_name))
        for off_name, ord_name, attr in kernels.STATIC_CSR_FIELDS:
            csr = getattr(self.program, attr)
            ordered.append(csr.offsets); targets.append((self.static, off_name))
            ordered.append(csr.order); targets.append((self.static, ord_name))
        for name in kernels.STATIC_DIRECT_FIELDS:
            ordered.append(getattr(self.program, name)); targets.append((self.static, name))
        self.scratch = {}
        dims = {"p": max(num_particles, 1), "t": max(num_teams, 1),
                "tri": max(self.program.num_triangle_entries, 1)}
        for key, shape, dtype in self._SCRATCH_SPECS:
            resolved = tuple(dims[s] if isinstance(s, str) else s for s in shape)
            ordered.append(np.zeros(resolved, dtype)); targets.append((self.scratch, key))

        sp_host = self._dump_primitive(world.self_points, self.program.num_self_points)
        se_host = self._dump_primitive(world.self_edges, self.program.num_self_edges)
        st_host = self._dump_primitive(world.self_triangles, self.program.num_self_triangles)
        self.self_points = device.FieldSet(sp_host, self.program.num_self_points, allocate=False)
        self.self_edges = device.FieldSet(se_host, self.program.num_self_edges, allocate=False)
        self.self_triangles = device.FieldSet(st_host, self.program.num_self_triangles, allocate=False)
        for fieldset, host in ((self.self_points, sp_host), (self.self_edges, se_host),
                               (self.self_triangles, st_host)):
            for name in kernels.PRIMITIVE_KERNEL_FIELDS:
                ordered.append(host[name]); targets.append((fieldset, name))
        for name in kernels.SELF_TEAM_KERNEL_FIELDS:
            ordered.append(team_host[name]); targets.append((self.team, name))
        for name in kernels.SELF_PARTICLE_KERNEL_FIELDS:
            ordered.append(particle_host[name]); targets.append((self.particles, name))
        self.self_state = {}
        for key, array in self._self_state_ordered():
            ordered.append(array); targets.append((self.self_state, key))

        self._assert_layout(ordered)
        self.blobs, self.offs, self.lens, views = device.build_blobs(ordered, kernels.RESIDENT_BLOB_GROUPS)
        for (container, key), view in zip(targets, views):
            if isinstance(container, device.FieldSet):
                container.set_view(key, view)
            else:
                container[key] = view
        for fieldset, host in ((self.team, team_host), (self.particles, particle_host),
                               (self.colliders, collider_host), (self.transforms, transform_host)):
            for name in fieldset.host_dtypes:
                if name not in fieldset.device:
                    fieldset.set_view(name, cuda.to_device(device._device_friendly(host[name])))

        self.input_staging = staging.StructStaging(world.team.dtype, num_teams, fields=_INPUT_UPLOAD_FIELDS)
        self.feedback_staging = staging.StructStaging(world.team.dtype, num_teams, fields=_CONSUMABLE_TEAM_FIELDS)
        self.config_staging = staging.StructStaging(world.team.dtype, num_teams, fields=_CONFIG_TEAM_FIELDS)
        particle_dtype = _arena_subset_dtype(world.particles.spec, _SLIM_OUTPUT_PARTICLE_FIELDS)
        self.particle_out_staging = staging.StructStaging(particle_dtype, num_particles,
                                                          fields=_SLIM_OUTPUT_PARTICLE_FIELDS)
        if num_colliders > 0:
            collider_dtype = _arena_subset_dtype(world.colliders.spec, _COLLIDER_INPUT_FIELDS)
            self.collider_input_staging = staging.StructStaging(
                collider_dtype, num_colliders, fields=_COLLIDER_INPUT_FIELDS)
        else:
            self.collider_input_staging = None

        self.n_zones = 0
        self.zone_blobs, self.zone_offs, self.zone_lens = self._zone_blobs([])
        self._zone_shadow = None
        self.config_staging._repack_in(world.team)
        self._config_shadow = self.config_staging._host.tobytes()
        self.stream = cuda.stream()
        self._scal_f_host = cuda.pinned_array(kernels.SCAL_F_LEN, dtype=np.float32)
        self._scal_i_host = cuda.pinned_array(kernels.SCAL_I_LEN, dtype=np.int32)
        self._scal_f_host[:] = 0
        self._scal_i_host[:] = 0
        self.scal_f = cuda.device_array(kernels.SCAL_F_LEN, dtype=np.float32)
        self.scal_i = cuda.device_array(kernels.SCAL_I_LEN, dtype=np.int32)
        world_field = world.transforms.arrays["world"]
        self._world_pinned = cuda.pinned_array(
            (max(self.program.num_transforms, 1),) + world_field.shape[1:], dtype=world_field.dtype)
        self._self_empty_uploaded = True
        self._self_task_shadow = None
        self._self_task_cache = None
        self._self_max_pairs = 0
        self._self_upload_shadow = None
        self._self_upload_totals = (0, 0)

    def _zone_host(self, zones):
        n = max(len(zones), 1)
        host = {
            "zone_id": np.zeros(n, np.int32),
            "mode": np.zeros(n, np.int32),
            "is_addition": np.zeros(n, np.uint8),
            "main": np.zeros(n, np.float32),
            "turbulence": np.zeros(n, np.float32),
            "world_position": np.zeros((n, 3), np.float32),
            "world_direction": np.zeros((n, 3), np.float32),
            "world_to_local": np.zeros((n, 4, 4), np.float64),
            "size": np.zeros((n, 3), np.float32),
            "zone_volume": np.zeros(n, np.float32),
            "attenuation_lut": np.zeros((n, 16), np.float32),
        }
        for k, zone in enumerate(zones):
            host["zone_id"][k] = zone.zone_id
            host["mode"][k] = _ZONE_MODE.get(zone.mode, 0)
            host["is_addition"][k] = 1 if zone.is_addition else 0
            host["main"][k] = zone.main
            host["turbulence"][k] = zone.turbulence
            host["world_position"][k] = zone.world_position
            host["world_direction"][k] = zone.world_direction
            host["world_to_local"][k] = zone.world_to_local
            host["size"][k] = zone.size
            host["zone_volume"][k] = np.inf if zone.mode == "GLOBAL_DIRECTION" \
                else float(zone.zone_volume)
            if zone.attenuation_lut is not None:
                host["attenuation_lut"][k] = zone.attenuation_lut
        return host

    def _zone_blobs(self, zones):
        host = self._zone_host(zones)
        ordered = [host[name] for name in _ZONE_ARG_ORDER]
        blobs, offs, lens, _views = device.build_blobs(ordered, kernels.ZONE_BLOB_GROUPS)
        return blobs, offs, lens

    @staticmethod
    def _assert_layout(ordered):
        layout = kernels.RESIDENT_BLOB_LAYOUT
        assert len(ordered) == len(layout), "resident slot count %d != layout %d" % (len(ordered), len(layout))
        for slot, array in enumerate(ordered):
            param, group, per_row = layout[slot]
            array = np.ascontiguousarray(array)
            dtype = np.uint8 if array.dtype == np.bool_ else array.dtype
            got = device.group_name(device._DTYPE_FAMILY[np.dtype(dtype)], array.shape[1:])
            assert got == group, "slot %d (%s): group %s != %s" % (slot, param, got, group)
            assert tuple(int(x) for x in array.shape[1:]) == tuple(per_row), \
                "slot %d (%s): per_row %s != %s" % (slot, param, tuple(array.shape[1:]), tuple(per_row))

    def upload_zones(self, zones):
        host = self._zone_host(zones)
        fingerprint = b"".join(host[name].tobytes() for name in _ZONE_ARG_ORDER)
        if len(zones) == self.n_zones and fingerprint == self._zone_shadow:
            return
        self.n_zones = len(zones)
        self._zone_shadow = fingerprint
        ordered = [host[name] for name in _ZONE_ARG_ORDER]
        self.zone_blobs, self.zone_offs, self.zone_lens = device.build_blobs(
            ordered, kernels.ZONE_BLOB_GROUPS)[:3]

    def _cooperative_max_blocks(self):
        if self._coop_max_blocks is None:
            try:
                sig = kernels.frame_kernel.signatures[0]
                defn = kernels.frame_kernel.overloads[sig]
                cap = int(defn.max_cooperative_grid_blocks(_THREADS))
                self._coop_max_blocks = max(1, min(cap, _MAX_COOP_BLOCKS))
            except Exception:
                return None
        return self._coop_max_blocks

    def _blocks(self):
        base = (max(self.program.num_particles, self.program.num_teams, 1) + _THREADS - 1) // _THREADS
        if self._self_max_pairs > 0:
            cap = self._cooperative_max_blocks()
            if cap is not None:
                per = _SELF_PAIRS_PER_THREAD * _THREADS
                pair_blocks = (self._self_max_pairs + per - 1) // per
                base = min(max(base, pair_blocks), cap)
        return base

    def download_team(self, world, names=None):
        names = names or list(self.team.device.keys())
        flat = {name: self.team.download(name) for name in names}
        device.scatter_struct(world.team, flat, self.program.num_teams, names)

    def download_particles(self, world, names=None):
        names = names or list(self.particles.device.keys())
        flat = {name: self.particles.download(name) for name in names}
        device.scatter_arena(world.particles, flat, self.program.num_particles, names)

    def download_colliders(self, world, names=None):
        names = names or list(self.colliders.device.keys())
        flat = {name: self.colliders.download(name) for name in names}
        device.scatter_arena(world.colliders, flat, self.program.num_colliders, names)

    def _frame_scalars(self, frame_globals):
        power = _defs.simulation_power(frame_globals.simulation_frequency)
        return (np.float32(frame_globals.frame_delta_time),
                np.float32(1.0 / frame_globals.simulation_frequency),
                np.int32(frame_globals.max_simulation_count),
                np.float32(frame_globals.global_time_scale),
                np.float32(power[0]), np.float32(power[1]),
                np.float32(power[2]), np.float32(power[3]))

    def _upload_scalars(self, sub_end, frame_globals, stream):
        fdt, sim_dt, msc, gts, pw0, pw1, pw2, pw3 = self._frame_scalars(frame_globals)
        f = self._scal_f_host
        f[kernels.SCAL_FRAME_DT] = fdt
        f[kernels.SCAL_SIM_DT] = sim_dt
        f[kernels.SCAL_TIME_SCALE] = gts
        f[kernels.SCAL_POWER0] = pw0
        f[kernels.SCAL_POWER1] = pw1
        f[kernels.SCAL_POWER2] = pw2
        f[kernels.SCAL_POWER3] = pw3
        i = self._scal_i_host
        i[kernels.SCAL_MAX_SIM] = msc
        i[kernels.SCAL_N_ZONES] = self.n_zones
        i[kernels.SCAL_SUB_END] = sub_end
        self.scal_f.copy_to_device(f, stream=stream)
        self.scal_i.copy_to_device(i, stream=stream)

    def launch(self, sub_end, frame_globals, stream=0):
        self._upload_scalars(sub_end, frame_globals, stream)
        blocks = self._blocks()
        kernels.frame_kernel[blocks, _THREADS, stream](
            self.scal_f, self.scal_i,
            *[self.blobs[group] for group in kernels.RESIDENT_BLOB_GROUPS],
            self.offs, self.lens,
            *[self.zone_blobs[group] for group in kernels.ZONE_BLOB_GROUPS],
            self.zone_offs, self.zone_lens)

    @staticmethod
    def _sub_end(frame_globals):
        return min(kernels.MAX_SIM_COUNT, int(frame_globals.max_simulation_count))

    def step_frame(self, world, frame_globals):
        self.load(world)
        stream = self.stream
        tblocks, tthreads = self._team_bridge_grid()
        self.input_staging.upload_async(world.team, self.team, tblocks, tthreads, stream)
        nt = self.program.num_transforms
        if nt > 0:
            self._world_pinned[:nt] = world.transforms.arrays["world"][:nt]
            self.transforms.device["world"].copy_to_device(self._world_pinned[:nt], stream=stream)
        if self.collider_input_staging is not None:
            cblocks, cthreads = self._collider_bridge_grid()
            self.collider_input_staging.upload_async(
                world.colliders.arrays, self.colliders, cblocks, cthreads, stream)
        self._maybe_upload_config(world, tblocks, tthreads, stream)
        self.upload_zones(frame_globals.zones)
        self._self_frame_prepare(world, frame_globals.frame_index, stream)
        self.launch(self._sub_end(frame_globals), frame_globals, stream=stream)
        pblocks, pthreads = self._particle_bridge_grid()
        self.particle_out_staging.download_issue(self.particles, pblocks, pthreads, stream)
        self.feedback_staging.download_issue(self.team, tblocks, tthreads, stream)
        stream.synchronize()
        self.particle_out_staging.download_finish(world.particles.arrays)
        self.feedback_staging.download_finish(world.team)

    def _team_bridge_grid(self):
        blocks = (max(self.program.num_teams, 1) + _THREADS - 1) // _THREADS
        return blocks, _THREADS

    def _collider_bridge_grid(self):
        blocks = (max(self.program.num_colliders, 1) + _THREADS - 1) // _THREADS
        return blocks, _THREADS

    def _particle_bridge_grid(self):
        blocks = (max(self.program.num_particles, 1) + _THREADS - 1) // _THREADS
        return blocks, _THREADS

    def _maybe_upload_config(self, world, blocks, threads, stream=0):
        self.config_staging._repack_in(world.team)
        fingerprint = self.config_staging._host.tobytes()
        if fingerprint == self._config_shadow:
            return
        self._config_shadow = fingerprint
        self.config_staging.stage.copy_to_device(self.config_staging._host, stream=stream)
        soa = [self.team.device[name] for name in self.config_staging.field_order]
        self.config_staging._explode[blocks, threads, stream](self.config_staging.stage, *soa)

    def _self_task_fingerprint(self, tt, nt):
        cws = tt["component_world_scale"][:nt]
        scale_alive = (np.abs(cws).min(axis=1) >= 1e-6)
        return b"".join((
            int(nt).to_bytes(8, "little"),
            tt["self_mode"][:nt].tobytes(), tt["sync_mode"][:nt].tobytes(),
            tt["sync_target"][:nt].tobytes(), tt["enabled"][:nt].tobytes(),
            tt["valid"][:nt].tobytes(), scale_alive.tobytes(),
            tt["sp_start"][:nt].tobytes(), tt["sp_count"][:nt].tobytes(),
            tt["se_start"][:nt].tobytes(), tt["se_count"][:nt].tobytes(),
            tt["st_start"][:nt].tobytes(), tt["st_count"][:nt].tobytes()))

    def _build_self_tasks(self, tt, nt):
        cws = tt["component_world_scale"][:nt]
        scale_alive = np.abs(cws).min(axis=1) >= 1e-6
        frame_mask = tt["enabled"][:nt] & tt["valid"][:nt] & scale_alive
        frame_teams = np.flatnonzero(frame_mask)
        use_point = np.zeros(nt, np.uint8)
        use_edge = np.zeros(nt, np.uint8)
        use_triangle = np.zeros(nt, np.uint8)
        contact = []
        intersect = []
        full = _defs.SELF_MODE_FULL_MESH
        for slot in frame_teams:
            row = tt[slot]
            self_mode = int(row["self_mode"])
            sync_mode = int(row["sync_mode"])
            partner = int(row["sync_target"])
            if partner <= 0 or not tt["valid"][partner] or not tt["enabled"][partner] or partner == slot:
                partner = 0
            se_c = int(row["se_count"]); se_s = int(row["se_start"])
            st_c = int(row["st_count"]); st_s = int(row["st_start"])
            sp_c = int(row["sp_count"]); sp_s = int(row["sp_start"])
            has_edge = se_c > 0
            has_tri = st_c > 0
            if self_mode == full:
                if has_edge:
                    use_edge[slot] = 1
                    contact.append((_SELF_TASK_EE, slot, se_s, se_c, slot, se_s, se_c, 1))
                if has_tri:
                    use_point[slot] = 1
                    use_triangle[slot] = 1
                    contact.append((_SELF_TASK_PT, slot, sp_s, sp_c, slot, st_s, st_c, 1))
                if has_edge and has_tri:
                    intersect.append((slot, se_s, se_c, slot, st_s, st_c, 1))
            if sync_mode == full and partner > 0:
                prow = tt[partner]
                p_se = int(prow["se_count"]); p_se_s = int(prow["se_start"])
                p_st = int(prow["st_count"]); p_st_s = int(prow["st_start"])
                p_sp = int(prow["sp_count"]); p_sp_s = int(prow["sp_start"])
                p_edge = p_se > 0
                p_tri = p_st > 0
                if has_edge and p_edge:
                    use_edge[slot] = 1
                    use_edge[partner] = 1
                    contact.append((_SELF_TASK_EE, slot, se_s, se_c, partner, p_se_s, p_se, 0))
                if has_tri:
                    use_triangle[slot] = 1
                    use_point[partner] = 1
                    contact.append((_SELF_TASK_PT, partner, p_sp_s, p_sp, slot, st_s, st_c, 0))
                if p_tri:
                    use_point[slot] = 1
                    use_triangle[partner] = 1
                    contact.append((_SELF_TASK_PT, slot, sp_s, sp_c, partner, p_st_s, p_st, 0))
                if has_edge and p_tri:
                    intersect.append((slot, se_s, se_c, partner, p_st_s, p_st, 0))
                if has_tri and p_edge:
                    intersect.append((partner, p_se_s, p_se, slot, st_s, st_c, 0))
        return contact, intersect, use_point, use_edge, use_triangle

    def _self_frame_prepare(self, world, frame_index, stream=0):
        tt = world.team
        nt = self.program.num_teams
        fingerprint = self._self_task_fingerprint(tt, nt)
        if fingerprint != self._self_task_shadow or self._self_task_cache is None:
            self._self_task_cache = self._build_self_tasks(tt, nt)
            self._self_task_shadow = fingerprint
        contact, intersect, use_point, use_edge, use_triangle = self._self_task_cache
        if not contact and not intersect and self._self_empty_uploaded:
            self._self_max_pairs = 0
            return
        self._self_empty_uploaded = (not contact) and (not intersect)
        if fingerprint != self._self_upload_shadow:
            total_ct = self._fill_task_table(contact, self.program.self_max_contact_tasks,
                                             ("ct_kind", "ct_my_team", "ct_my_start", "ct_my_count",
                                              "ct_tgt_team", "ct_tgt_start", "ct_tgt_count", "ct_same"),
                                             "ct_pair_off", stream)
            total_it = self._fill_task_table(intersect, self.program.self_max_intersect_tasks,
                                             ("it_edge_team", "it_edge_start", "it_edge_count",
                                              "it_tri_team", "it_tri_start", "it_tri_count", "it_same"),
                                             "it_pair_off", stream)
            self.team.device["use_point"].copy_to_device(use_point, stream=stream)
            self.team.device["use_edge"].copy_to_device(use_edge, stream=stream)
            self.team.device["use_triangle"].copy_to_device(use_triangle, stream=stream)
            self._self_upload_totals = (total_ct, total_it)
            self._self_upload_shadow = fingerprint
        total_ct, total_it = self._self_upload_totals
        self._self_max_pairs = int(max(total_ct, total_it))
        counts = np.zeros(8, np.int32)
        counts[kernels.SCL_ERROR] = 1 if (total_ct > _SELF_PAIR_GUARD or total_it > _SELF_PAIR_GUARD) else 0
        counts[kernels.SCL_USE_INTERSECT] = 1 if intersect else 0
        counts[kernels.SCL_FRAME_INDEX] = int(frame_index) % int(_defs.SELF_COLLISION_INTERSECT_DIV)
        self.self_state["scl_counts"].copy_to_device(counts, stream=stream)

    def _fill_task_table(self, tasks, capacity, column_keys, pair_off_key, stream):
        columns = [np.zeros(capacity, np.uint8 if key.endswith("same") else np.int32)
                   for key in column_keys]
        pair_off = np.zeros(capacity + 1, np.int32)
        running = 0
        for k, task in enumerate(tasks):
            for c, value in enumerate(task[:len(column_keys)]):
                columns[c][k] = value
            pair_off[k] = running
            my_count = task[3] if len(column_keys) == 8 else task[2]
            tgt_count = task[6] if len(column_keys) == 8 else task[5]
            running += int(my_count) * int(tgt_count)
        pair_off[len(tasks):] = running
        for key, column in zip(column_keys, columns):
            self.self_state[key].copy_to_device(column, stream=stream)
        self.self_state[pair_off_key].copy_to_device(pair_off, stream=stream)
        return running
