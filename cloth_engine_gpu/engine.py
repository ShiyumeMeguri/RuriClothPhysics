"""GpuEngine: resident device state + per-frame cooperative launch.

``load(world)`` uploads the static Program and the world's mutable arenas once and
keeps them resident (cached by ``id(world)`` + registration generation). ``step_frame``
uploads this frame's host-mutated *input* fields, runs the whole pipeline in one
cooperative launch, and reads the *output* fields back into the world numpy arrays so
``io.team_output`` / the gate are backend-agnostic. ``step_frame_captured`` drives the
same kernel in segments for the gate's per-substep assertions.

During bring-up the engine also exposes ``upload_all`` / ``download_all`` /
``launch``: the dev-harness sets the device to an oracle-produced "before phase X"
state, launches just phase X, and compares the readback to the oracle "after X" state.
"""

import numpy as np
from numba import cuda, int32

from cloth_kernel import defs as _defs

from . import device
from . import kernels
from . import staging
from .program import build_program

_MAX_COOP_BLOCKS = 544
_THREADS = 128

# Stable int mapping for WindZoneInput.mode strings (device-side zone arrays).
_ZONE_MODE = {"GLOBAL_DIRECTION": 0, "BOX_DIRECTION": 1,
              "SPHERE_DIRECTION": 2, "SPHERE_RADIAL": 3}
# Ordered zone SoA field names appended to the launch after the scratch args.
_ZONE_ARG_ORDER = ("zone_id", "mode", "is_addition", "main", "turbulence",
                   "world_position", "world_direction", "world_to_local",
                   "size", "zone_volume", "attenuation_lut")

# Phase-bit groups matching the megakernel body layout: [frame-pre][for _k in
# range(sub_begin,sub_end): substep][frame-post]. A full frame is one launch of
# ALL_PHASES over (0, MAX_SIM_COUNT); the captured path fires the same kernel as
# frame-pre-only -> per-substep -> frame-post-only segments (the phase gates mask the
# WORK while every grid.sync barrier still runs, and a kernel boundary is a strictly
# stronger memory fence than grid.sync, so segmented == single bit-for-bit).
FRAME_PRE_PHASES = int(kernels.PHASE_SYNC | kernels.PHASE_ADVANCE | kernels.PHASE_BASE_POSE
                       | kernels.PHASE_CENTER | kernels.PHASE_PARTICLES_PRE
                       | kernels.PHASE_COLLIDER_PRE | kernels.PHASE_SELF_BEGIN)
STEP_PHASES = int(kernels.PHASE_TEAM_STEP | kernels.PHASE_COLLIDER_START
                  | kernels.PHASE_PARTICLES_STEP | kernels.PHASE_BASELINE | kernels.PHASE_TETHER
                  | kernels.PHASE_DISTANCE_A | kernels.PHASE_ANGLE | kernels.PHASE_BENDING
                  | kernels.PHASE_COLLIDER_SOLVE | kernels.PHASE_DISTANCE_B | kernels.PHASE_MOTION
                  | kernels.PHASE_STEP_POST | kernels.PHASE_COLLIDER_END | kernels.PHASE_SELF_STEP)
FRAME_POST_PHASES = int(kernels.PHASE_DISPLAY | kernels.PHASE_COLLIDER_POST
                        | kernels.PHASE_TEAM_POST | kernels.PHASE_SELF_END)

# G3a self-collision: contact-task kind codes (device ct_kind) and the per-frame pair-count FAIL
# guard (D1: a scene whose total n^2 candidate pairs exceeds this sets the error flag rather than
# silently truncating). EE / PT are the two normalized contact kinds (TP is stored flipped to PT).
_SELF_TASK_EE = 0
_SELF_TASK_PT = 1
_SELF_PAIR_GUARD = 30_000_000

# team structural columns (chunk pointers + valid) -- register/unregister move these;
# collider/pair CONTENT is folded in separately so a same-count collider rebind (which
# can reuse the freed [start,count) block, leaving these columns untouched) is still caught.
_STRUCT_TEAM_COLUMNS = ("valid", "p_start", "p_count", "t_start", "t_count",
                        "c_start", "c_count", "sp_start", "sp_count",
                        "se_start", "se_count", "st_start", "st_count")

# ---- slim per-frame IO field partition (G2e-5) ------------------------------------------
# Host write-surface proof (cloth_kernel.io + cloth_kernel.world + blender_host.runtime):
# the resident engine keeps ALL simulation state on the device and re-uploads, each frame,
# ONLY the fields the host actually mutates. The partition below is the sole authority for
# what moves per frame.
#
# PER-FRAME INPUT -- io.set_team_frame_input writes these every frame; runtime.run_frame
# additionally writes sync_target directly after the team loop. Uploaded every frame.
_INPUT_TEAM_FIELDS = ("enabled", "component_world_position", "component_world_rotation",
                      "component_world_scale", "culling_invisible", "distance_weight",
                      "sync_target", "has_anchor", "anchor_position", "anchor_rotation")
# CONSUMABLE (edge-triggered) INPUT -- host sets via world.request_reset / world.add_force,
# the kernel clears them (team_time.frame_post clears the reset flags for every processed
# team; force_mode / impact_force are cleared where running). Folded into the per-frame
# upload AND read back each frame so the CPU mirrors the resident GPU for exactly these
# host+kernel-shared fields. That makes the upload idempotent and clobber-free for teams the
# host did not touch, and a repeated request_reset on an already-set-but-consumed team is
# always re-detected -- with the KERNEL as the single source of truth for the post-frame
# value (no fragile CPU duplication of its clear conditions, which depend on GPU-only
# ``running`` state). This 5-field feedback is the low-cost field-level mechanism the brief
# mandates for reset/force ("漏传即错"); it is NOT the 167-field table round-trip.
_CONSUMABLE_TEAM_FIELDS = ("reset_pending", "time_reset_pending", "keep_teleport_pending",
                           "force_mode", "impact_force")
_INPUT_UPLOAD_FIELDS = _INPUT_TEAM_FIELDS + _CONSUMABLE_TEAM_FIELDS
# LOW-FREQUENCY CONFIG -- written ONLY by world.update_params (host config edit). This list
# MUST equal blender_host.runtime._build_params keys (its only call site). None of these is a
# kernel accumulator: the eight fields T0 resolve_sync copies to sync children are re-derived
# from the (correctly resident) parent every frame, so a byte fingerprint + upload-on-change
# is exact and never clobbers resident state.
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
# collider host inputs (io.set_team_collider_input) + the production output particles that
# io.team_output returns (runtime.run_frame reads only positions + out_rotations).
_COLLIDER_INPUT_FIELDS = ("input_positions", "input_rotations", "input_scales", "enabled")
_SLIM_OUTPUT_PARTICLE_FIELDS = ("positions", "out_rotations")

# The three team input partitions must be disjoint (no field both static-config and
# per-frame); a typo'd field name raises KeyError when StructStaging indexes the dtype.
assert not (set(_INPUT_UPLOAD_FIELDS) & set(_CONFIG_TEAM_FIELDS))
assert len(set(_INPUT_UPLOAD_FIELDS)) == len(_INPUT_UPLOAD_FIELDS)


def _arena_subset_dtype(spec, fields):
    """Build a struct dtype for a subset of a ChunkArena's fields (spec = arena.spec:
    {name: (dtype, shape)}), so the narrow collider/particle bridges can stage them."""
    items = []
    for name in fields:
        dtype, shape = spec[name]
        items.append((name, dtype, shape) if shape else (name, dtype))
    return np.dtype(items)


class GpuEngine:
    def __init__(self, world, io_mode="slim"):
        # io_mode "slim": per-frame upload = host-mutated inputs only, readback = production
        #   output particles + the 5-field consumable feedback (production / default).
        # io_mode "full": whole-team blob round-trip every frame (G2e-4 behaviour) -- the
        #   optional full-state path for the four-mode gate / bit-level equivalence checks.
        self.io_mode = io_mode
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
        self.load(world)

    # ---- lifecycle ----------------------------------------------------------
    @staticmethod
    def _structure_signature(world):
        """Cheap structural fingerprint of the resident Program layout. Reload when it
        changes. Covers every mutation that invalidates the device Program:

        * register_team / unregister_team -> team count and the chunk-pointer columns move
          (and any team grow lengthens the column bytes);
        * update_colliders -> the collider static table (kind/center/size/axis/aligned) and
          the point/edge-pair mapping are rebuilt. A same-count rebind can reuse the freed
          [start,count) block, leaving the team columns byte-identical, so the collider and
          pair CONTENT is hashed in directly rather than trusting the chunk pointers.

        Pure per-frame state (positions, time, world matrices, params via update_params /
        request_reset / add_force) is deliberately excluded, so a running scenario never
        triggers a spurious reload. Byte equality is exact -- no hash-collision blind spot."""
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

    # scratch device buffers (runtime-only accumulators): (key, shape-factory, dtype). Folded into the
    # resident blobs like every other field; segments are cleared by their owning phase before use.
    _SCRATCH_SPECS = (
        ("dcorr", ("p", 3), np.float32),
        ("dcorr_fixed", ("p", 3), np.int32),
        ("dcount", ("p",), np.int32),
        ("col_friction_fixed", ("p",), np.int32),
        ("col_normal_fixed", ("p", 3), np.int32),
        # T0 resolve_sync gather snapshot: 22 scalars/team (7 time + 8 param + 3 cwp + 4 cwr) so the
        # child write reads a pre-gather copy of its sync_top row (mutual-sync race safe).
        ("sync_snapshot", ("t", 22), np.float32),
        # F2 display._post_triangles per-triangle stash (authorised f64), indexed by global triangle row.
        ("tri_normal_f64", ("tri", 3), np.float64),
        ("tri_tangent_f64", ("tri", 3), np.float64),
    )

    @staticmethod
    def _dump_primitive(arena, count):
        """Host dump of a self-collision primitive arena (PRIMITIVE_KERNEL_FIELDS). cell_key is
        dropped (n^2 broad-phase, no grid); the (3,) bool ``fix`` / ``intersect`` masks are packed
        into a single uint8 (bit0/1/2) so no new (u8,(3,)) blob group is introduced (signature stays
        39). aabb / inv_mass / thickness / intersect / use are kernel-written (start as dumped)."""
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
        """The self_state device buffers in RESIDENT_BLOB_LAYOUT tail order (== SELF_STATE_KERNEL_FIELDS).
        Capacities come from the Program (worst-case, structural). All start zeroed; the contact/
        intersect task tables and scl_counts are (re)written host-side each frame before launch."""
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
        # FieldSets record host dtypes only (for bool-restoring readback); their device arrays are
        # blob views injected below, so per-field upload / readback and the staging bridges keep
        # working while the megakernel takes one blob per dtype family instead of ~290 field arrays.
        self.team = device.FieldSet(team_host, num_teams, allocate=False)
        self.particles = device.FieldSet(particle_host, num_particles, allocate=False)
        self.colliders = device.FieldSet(collider_host, num_colliders, allocate=False)
        self.transforms = device.FieldSet(transform_host, num_transforms, allocate=False)
        self.static = {}

        # Ordered resident host arrays == RESIDENT_BLOB_LAYOUT order (== the old signature order): the
        # single ordered source the blobs / offs / lens / kernel view-reconstruction all agree on.
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

        # G3a self-collision resident state, appended AFTER scratch so every slot takes a fresh tail
        # index (no existing preamble slice / layout row moves). Order == RESIDENT_BLOB_LAYOUT tail:
        # 3 primitive arenas, the self team fields, intersect_flag, then the self_state buffers.
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
        # Struct/arena fields the kernel does NOT consume (present in the host dtype but absent from the
        # *_KERNEL_FIELDS surface -- e.g. team self_mode / self_thickness_lut / step_active) still need
        # resident device arrays for the whole-table round-trip (io_mode="full" / config_staging /
        # upload_all / download_*). They are never kernel arguments, so keep them as standalone
        # allocations rather than blob slots (adding them to the blobs would just bloat offs/lens).
        for fieldset, host in ((self.team, team_host), (self.particles, particle_host),
                               (self.colliders, collider_host), (self.transforms, transform_host)):
            for name in fieldset.host_dtypes:
                if name not in fieldset.device:
                    fieldset.set_view(name, cuda.to_device(device._device_friendly(host[name])))

        # Whole-team round-trip goes as one blob transfer + a struct<->SoA bridge kernel (not 167
        # per-field transfers); the narrow per-frame bridges carry far fewer marshalled args. All
        # bridges write/read the blob views the FieldSets now hold. Built from world dtypes (single
        # source of truth); a bad field name raises here.
        self.team_staging = staging.StructStaging(world.team.dtype, num_teams)
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

        # Wind zones are variable-length and re-uploaded each frame as their own per-dtype blob set
        # (same aggregation, separate lifecycle); start empty so zone_count=0 launches carry safe args.
        self.n_zones = 0
        self.zone_blobs, self.zone_offs, self.zone_lens = self._zone_blobs([])
        # Reset the zone fingerprint so the next upload_zones re-establishes the zone blobs (which
        # this reload just reset to empty) rather than skipping on a stale match.
        self._zone_shadow = None
        # A (re)load just uploaded the current config to the device; take its fingerprint so
        # slim frames only re-upload config on a genuine host update_params edit.
        self.config_staging._repack_in(world.team)
        self._config_shadow = self.config_staging._host.tobytes()
        # One CUDA stream + a pinned mirror of the per-frame transform-world matrices, so the slim
        # production path queues upload -> cooperative launch -> readback on a single stream and pays
        # one synchronize instead of ~9 blocking pageable transfers (each ~0.1 ms of WDDM overhead).
        self.stream = cuda.stream()
        world_field = world.transforms.arrays["world"]
        self._world_pinned = cuda.pinned_array(
            (max(self.program.num_transforms, 1),) + world_field.shape[1:], dtype=world_field.dtype)
        # A fresh load zeroes every self_state blob, so the device already reflects the empty-task
        # state: _self_frame_prepare can skip its (~20 small) uploads until the first self frame.
        self._self_empty_uploaded = True
        # The self-collision task table (use flags + EE/PT/intersect task pairs) is a pure function of
        # the per-team columns it reads (self_mode/sync_mode/sync_target/enabled/valid/scale-alive + the
        # sp/se/st chunk offsets). Building it walks every frame team with slow numpy void-scalar access
        # (~0.15 ms/frame for 24 teams -- and it ran unconditionally, even -noself). Cache the built
        # tables keyed on a byte fingerprint of exactly those inputs, so a steady scene (every -noself
        # frame; every unchanged self frame) reuses them and only a genuine layout/config/input change
        # rebuilds. Reset here so the first frame after any (re)load always recomputes.
        self._self_task_shadow = None
        self._self_task_cache = None

    # ---- wind-zone per-frame upload -----------------------------------------
    def _zone_host(self, zones):
        """Pack list[WindZoneInput] into the 11 host SoA arrays (CPU only)."""
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
            # GLOBAL_DIRECTION uses inf volume so it never loses the min-volume race.
            host["zone_volume"][k] = np.inf if zone.mode == "GLOBAL_DIRECTION" \
                else float(zone.zone_volume)
            if zone.attenuation_lut is not None:
                host["attenuation_lut"][k] = zone.attenuation_lut
        return host

    def _zone_blobs(self, zones):
        """Aggregate the 11 zone SoA arrays into per-(family,shape) zone blobs + offs/lens (same
        mechanism as the resident blobs, its own variable-length lifecycle). Returns (blobs, offs, lens)."""
        host = self._zone_host(zones)
        ordered = [host[name] for name in _ZONE_ARG_ORDER]
        blobs, offs, lens, _views = device.build_blobs(ordered, kernels.ZONE_BLOB_GROUPS)
        return blobs, offs, lens

    @staticmethod
    def _assert_layout(ordered):
        """Drift guard: the positionally-built resident arrays must match the RESIDENT_BLOB_LAYOUT
        registry (blob group + per-row) that the kernel view-reconstruction preamble is generated
        from. A mismatch means a world dtype/shape changed without updating the layout + preamble."""
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
        # Zones are a low-frequency input (a static scene never re-uploads; animated wind re-uploads
        # only when a value changes). Rebuilding the zone blobs every frame was the single biggest
        # per-frame transfer cost (~1.9 ms measured); a byte fingerprint skips it whenever the packed
        # zone data is unchanged. The resident zone blobs are read-only inputs the kernel never
        # mutates, so they stay valid across skipped frames.
        host = self._zone_host(zones)
        fingerprint = b"".join(host[name].tobytes() for name in _ZONE_ARG_ORDER)
        if len(zones) == self.n_zones and fingerprint == self._zone_shadow:
            return
        self.n_zones = len(zones)
        self._zone_shadow = fingerprint
        ordered = [host[name] for name in _ZONE_ARG_ORDER]
        self.zone_blobs, self.zone_offs, self.zone_lens = device.build_blobs(
            ordered, kernels.ZONE_BLOB_GROUPS)[:3]

    def _blocks(self):
        needed = max(self.program.num_particles, self.program.num_teams, 1)
        return min((needed + _THREADS - 1) // _THREADS, _MAX_COOP_BLOCKS)

    # ---- full-state sync (dev-harness isolated phase testing) ---------------
    def upload_all(self, world):
        self.team.upload_many(device.dump_struct(world.team, self.program.num_teams),
                              self.team.device.keys())
        self.particles.upload_many(device.dump_arena(world.particles, self.program.num_particles),
                                   self.particles.device.keys())
        self.colliders.upload_many(device.dump_arena(world.colliders, self.program.num_colliders),
                                   self.colliders.device.keys())
        self.transforms.upload_many(device.dump_arena(world.transforms, self.program.num_transforms),
                                    self.transforms.device.keys())
        self.upload_self_primitives(world)

    def upload_self_primitives(self, world):
        """Re-pack (fix/intersect masks) and upload the three self-collision primitive arenas.
        Used by the dev harness to set the device 'before phase X' primitive state from the world."""
        for fieldset, arena, count in (
                (self.self_points, world.self_points, self.program.num_self_points),
                (self.self_edges, world.self_edges, self.program.num_self_edges),
                (self.self_triangles, world.self_triangles, self.program.num_self_triangles)):
            fieldset.upload_many(self._dump_primitive(arena, count), fieldset.device.keys())

    def download_self_primitive(self, arena_name, names=None):
        """Read back a primitive arena's fields; fix/intersect are unpacked from the u8 bitmask into
        (count, 3) bool arrays so the dev harness can compare against the oracle arena directly."""
        fieldset = getattr(self, arena_name)
        names = names or list(fieldset.device.keys())
        out = {}
        for name in names:
            raw = fieldset.device[name].copy_to_host()
            if name in ("fix", "intersect"):
                out[name] = np.stack([(raw >> bit) & 1 for bit in range(3)], axis=1).astype(bool)
            elif name in ("all_fix", "ignore", "use"):
                out[name] = raw.astype(bool)
            else:
                out[name] = raw
        return out

    def download_self_state(self, names=None):
        names = names or list(self.self_state.keys())
        return {name: self.self_state[name].copy_to_host() for name in names}

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

    # ---- launch -------------------------------------------------------------
    def _frame_scalars(self, frame_globals):
        power = _defs.simulation_power(frame_globals.simulation_frequency)
        return (np.float32(frame_globals.frame_delta_time),
                np.float32(1.0 / frame_globals.simulation_frequency),
                np.int32(frame_globals.max_simulation_count),
                np.float32(frame_globals.global_time_scale),
                np.float32(power[0]), np.float32(power[1]),
                np.float32(power[2]), np.float32(power[3]))

    def launch(self, phase_mask, sub_begin, sub_end, frame_globals, stream=0):
        # G2e-7: the whole resident field set is 17 per-(family,shape) blobs (+ int64 offs/lens), zones
        # are 6 zone blobs (+ offs/lens). The kernel reconstructs every field as an axis-0 slice of its
        # blob at entry, so this is 27 array args + 12 scalars instead of ~290 field arrays -- the
        # ~1.4 ms/launch host marshalling floor is gone. Blobs are splatted in RESIDENT_BLOB_GROUPS /
        # ZONE_BLOB_GROUPS order, which is exactly the signature order the preamble was generated from.
        fdt, sim_dt, msc, gts, pw0, pw1, pw2, pw3 = self._frame_scalars(frame_globals)
        blocks = self._blocks()
        kernels.frame_kernel[blocks, _THREADS, stream](
            int32(phase_mask), int32(sub_begin), int32(sub_end),
            fdt, sim_dt, msc, gts, pw0, pw1, pw2, pw3,
            *[self.blobs[group] for group in kernels.RESIDENT_BLOB_GROUPS],
            self.offs, self.lens,
            int32(self.n_zones),
            *[self.zone_blobs[group] for group in kernels.ZONE_BLOB_GROUPS],
            self.zone_offs, self.zone_lens)

    # ---- production API -----------------------------------------------------
    @staticmethod
    def _sub_end(frame_globals):
        """Substep iterations to run. update_count is capped at max_simulation_count
        (team_time.advance: min(computed, max_simulation_count)), so every substep iteration
        _k >= max_simulation_count is a fully gated no-op -- running MAX_SIM_COUNT of them
        wastes that many barrier passes. Capping at min(MAX_SIM_COUNT, max_simulation_count)
        is bit-identical (the dropped iterations do zero work) and never exceeds the previous
        bound. Zero-semantic-risk launch tune; verified bit-for-bit in dev_harness."""
        return min(kernels.MAX_SIM_COUNT, int(frame_globals.max_simulation_count))

    def step_frame(self, world, frame_globals):
        """One host frame: upload this frame's host-mutated inputs, run the whole
        pipeline in a single cooperative launch, read the outputs back into the world."""
        self.load(world)
        if self.io_mode == "slim":
            self._step_frame_slim(world, frame_globals)
            return
        self._upload_inputs(world)
        self.upload_zones(frame_globals.zones)
        self._self_frame_prepare(world, frame_globals.frame_index)
        self.launch(kernels.ALL_PHASES, 0, self._sub_end(frame_globals), frame_globals)
        self._download_outputs(world)

    def _step_frame_slim(self, world, frame_globals):
        """Production slim path fused on one CUDA stream: queue every per-frame input H2D (from
        pinned mirrors) + the two upload bridges, the cooperative megakernel, and both output
        bridges + D2H, then pay a SINGLE stream synchronize before the CPU repacks the pinned
        outputs. The default stream is in-order, so correctness is identical to the blocking path
        (dev-harness test_slim_equivalence proves it bit-for-bit); the win is collapsing ~9 blocking
        pageable transfers into one fenced async batch. Zones/config are byte-fingerprinted and,
        in steady-state playback, never re-upload; when a value genuinely changes they fire once
        on the same stream before the launch reads them."""
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
        # Zones are byte-fingerprinted: in steady-state playback this is a no-op (never fires), so the
        # whole slim frame stays on one stream. A genuine zone change re-uploads once (synchronously)
        # before the launch is queued, so the launch still reads the current zones.
        self.upload_zones(frame_globals.zones)
        self._self_frame_prepare(world, frame_globals.frame_index, stream)
        self.launch(kernels.ALL_PHASES, 0, self._sub_end(frame_globals), frame_globals, stream=stream)
        pblocks, pthreads = self._particle_bridge_grid()
        self.particle_out_staging.download_issue(self.particles, pblocks, pthreads, stream)
        self.feedback_staging.download_issue(self.team, tblocks, tthreads, stream)
        stream.synchronize()
        self.particle_out_staging.download_finish(world.particles.arrays)
        self.feedback_staging.download_finish(world.team)

    def step_frame_captured(self, world, frame_globals, capture=None):
        """Same frame as step_frame, but driven as segments so the gate can read any
        field between segments (frame-pre, each substep, frame-post). ``capture`` is an
        iterable of ``(fieldset_name, field_name)`` snapshotted at every boundary; the
        returned dict is ``{"frame_pre":..., "substeps":[...per k...], "frame_post":...}``.
        The device end-state is bit-identical to step_frame (same kernel, same phase
        order); the multi-frame loop still lives in the host (never one launch per frame,
        to stay under the 2 s TDR)."""
        self.load(world)
        self._upload_inputs(world)
        self.upload_zones(frame_globals.zones)
        self._self_frame_prepare(world, frame_globals.frame_index)
        captured = self.launch_segmented(frame_globals, capture)
        self._download_outputs(world)
        return captured

    def launch_segmented(self, frame_globals, capture=None):
        captured = {"frame_pre": None, "substeps": [], "frame_post": None}
        self.launch(FRAME_PRE_PHASES, 0, 0, frame_globals)
        if capture is not None:
            captured["frame_pre"] = self._capture_fields(capture)
        for k in range(kernels.MAX_SIM_COUNT):
            self.launch(STEP_PHASES, k, k + 1, frame_globals)
            if capture is not None:
                captured["substeps"].append(self._capture_fields(capture))
        self.launch(FRAME_POST_PHASES, 0, 0, frame_globals)
        if capture is not None:
            captured["frame_post"] = self._capture_fields(capture)
        return captured

    def _capture_fields(self, capture):
        snapshot = {}
        for fieldset_name, field_name in capture:
            snapshot[(fieldset_name, field_name)] = getattr(self, fieldset_name).download(field_name)
        return snapshot

    # Per-frame upload is the zero-missed-field superset: the whole team struct (the host
    # may touch any team field via set_team_frame_input / update_params / request_reset /
    # add_force between frames), the transform world matrices, and the collider host inputs.
    # Particles are never host-mutated post-registration, so they stay resident (uploaded
    # once at load / reload) and are only read back.
    _OUTPUT_PARTICLE_FIELDS = ("positions", "out_rotations", "velocities")

    def _team_bridge_grid(self):
        blocks = (max(self.program.num_teams, 1) + _THREADS - 1) // _THREADS
        return blocks, _THREADS

    def _upload_inputs(self, world):
        blocks, threads = self._team_bridge_grid()
        self.team_staging.upload(world.team, self.team, blocks, threads)
        # Only the host-mutated arena fields, sliced directly (no full-arena dump): the
        # transform world matrices and the collider host inputs. Everything else the arenas
        # hold is either static (uploaded at load) or GPU-resident.
        self.transforms.upload("world", world.transforms.arrays["world"][:self.program.num_transforms])
        collider_arrays = world.colliders.arrays
        for name in ("input_positions", "input_rotations", "input_scales", "enabled"):
            if name in self.colliders.device:
                self.colliders.upload(name, collider_arrays[name][:self.program.num_colliders])

    def _download_outputs(self, world):
        # Whole team read back so the CPU stays a byte-exact mirror of the resident GPU
        # team (next frame's full-team upload must not clobber GPU state with stale CPU
        # values); particles limited to the production output fields io.team_output needs.
        blocks, threads = self._team_bridge_grid()
        self.team_staging.download(world.team, self.team, blocks, threads)
        self.download_particles(world, list(self._OUTPUT_PARTICLE_FIELDS))

    # ---- slim per-frame IO (production default) -----------------------------
    # Resident simulation state stays on the device untouched. Each frame only the fields the
    # host actually mutates go up (per-frame inputs + consumable flags always; low-frequency
    # config on a fingerprint change), and only the production output particles + the 5-field
    # consumable feedback come back. This drops the two 145-arg whole-team bridge kernels --
    # the measured IO floor (their cost is per marshalled argument, independent of team count)
    # -- to narrow bridges of ~15 / 5 / 2 args.
    def _collider_bridge_grid(self):
        blocks = (max(self.program.num_colliders, 1) + _THREADS - 1) // _THREADS
        return blocks, _THREADS

    def _particle_bridge_grid(self):
        blocks = (max(self.program.num_particles, 1) + _THREADS - 1) // _THREADS
        return blocks, _THREADS

    def _upload_inputs_slim(self, world):
        tblocks, tthreads = self._team_bridge_grid()
        self.input_staging.upload(world.team, self.team, tblocks, tthreads)
        self.transforms.upload("world", world.transforms.arrays["world"][:self.program.num_transforms])
        if self.collider_input_staging is not None:
            cblocks, cthreads = self._collider_bridge_grid()
            self.collider_input_staging.upload(world.colliders.arrays, self.colliders, cblocks, cthreads)
        self._maybe_upload_config(world, tblocks, tthreads)

    def _maybe_upload_config(self, world, blocks, threads, stream=0):
        # Byte fingerprint of the config columns; upload only on a genuine host update_params
        # edit. Steady-state playback never touches config, so this is repack + memcmp only.
        self.config_staging._repack_in(world.team)
        fingerprint = self.config_staging._host.tobytes()
        if fingerprint == self._config_shadow:
            return
        self._config_shadow = fingerprint
        self.config_staging.stage.copy_to_device(self.config_staging._host, stream=stream)
        soa = [self.team.device[name] for name in self.config_staging.field_order]
        self.config_staging._explode[blocks, threads, stream](self.config_staging.stage, *soa)

    def _download_outputs_slim(self, world):
        pblocks, pthreads = self._particle_bridge_grid()
        self.particle_out_staging.download(world.particles.arrays, self.particles, pblocks, pthreads)
        tblocks, tthreads = self._team_bridge_grid()
        self.feedback_staging.download(world.team, self.team, tblocks, tthreads)

    # ---- G3a self-collision per-frame host prep + upload ---------------------
    def _self_task_fingerprint(self, tt, nt):
        """Byte fingerprint of exactly the per-team columns _build_self_tasks reads, so the built
        task tables are reused whenever none of them changed. Cheap (a handful of tobytes on nt-length
        columns, ~microseconds) vs the ~0.15 ms per-team build loop it guards. scale-alive is folded in
        as its derived bool (matching frame_mask), so sub-threshold scale wiggle never invalidates."""
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
        """Mirror of cloth_kernel.self_collision.frame_begin's TEAM-LEVEL logic (D2: the use_point/
        edge/triangle flags and the (EE/PT/TP) contact + intersect task pairs are frame-level pure
        team logic -> host). Returns (contact, intersect, use_point, use_edge, use_triangle). TP is
        stored flipped to PT (my=points, target=triangles). Cross-team (sync) tasks carry
        same_object=0 (no connection filter, no EE self-dedup). Pure function of the fingerprinted
        columns -> the caller caches the result across frames."""
        cws = tt["component_world_scale"][:nt]
        scale_alive = np.abs(cws).min(axis=1) >= 1e-6
        frame_mask = tt["enabled"][:nt] & tt["valid"][:nt] & scale_alive
        frame_teams = np.flatnonzero(frame_mask)
        use_point = np.zeros(nt, np.uint8)
        use_edge = np.zeros(nt, np.uint8)
        use_triangle = np.zeros(nt, np.uint8)
        contact = []      # (kind, my_team, my_start, my_count, tgt_team, tgt_start, tgt_count, same)
        intersect = []    # (edge_team, edge_start, edge_count, tri_team, tri_start, tri_count, same)
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
                if has_tri:  # oracle 'TP' (my triangles slot, tgt points partner) -> flipped PT
                    use_triangle[slot] = 1
                    use_point[partner] = 1
                    contact.append((_SELF_TASK_PT, partner, p_sp_s, p_sp, slot, st_s, st_c, 0))
                if p_tri:  # oracle 'PT' (my points slot, tgt triangles partner)
                    use_point[slot] = 1
                    use_triangle[partner] = 1
                    contact.append((_SELF_TASK_PT, slot, sp_s, sp_c, partner, p_st_s, p_st, 0))
                if has_edge and p_tri:
                    intersect.append((slot, se_s, se_c, partner, p_st_s, p_st, 0))
                if has_tri and p_edge:
                    intersect.append((partner, p_se_s, p_se, slot, st_s, st_c, 0))
        return contact, intersect, use_point, use_edge, use_triangle

    def _self_frame_prepare(self, world, frame_index, stream=0):
        """Build (or reuse) this frame's self-collision device task tables + use flags + reset
        counters and upload them. The task tables are fingerprint-cached (see _self_task_fingerprint):
        a steady scene rebuilds them once and every later frame reuses them, so the -noself production
        path costs a fingerprint compare instead of the per-team build loop. Uploads still run each
        non-empty frame (scl_counts' frame_index rotates); the empty-task device state is reused as-is."""
        tt = world.team
        nt = self.program.num_teams
        fingerprint = self._self_task_fingerprint(tt, nt)
        if fingerprint != self._self_task_shadow or self._self_task_cache is None:
            self._self_task_cache = self._build_self_tasks(tt, nt)
            self._self_task_shadow = fingerprint
        contact, intersect, use_point, use_edge, use_triangle = self._self_task_cache
        # -noself fast path: no self-collision tasks AND the device already reflects the empty state
        # (fresh load, or the previous frame reset it) -> skip all uploads; the kernel's grid-uniform
        # total-pair gates then execute zero self-collision work / barriers (P6: self off == near-zero
        # overhead). A self->noself transition still resets once (empty but not-yet-empty).
        if not contact and not intersect and self._self_empty_uploaded:
            return
        self._self_empty_uploaded = (not contact) and (not intersect)
        total_ct = self._fill_task_table(contact, self.program.self_max_contact_tasks,
                                         ("ct_kind", "ct_my_team", "ct_my_start", "ct_my_count",
                                          "ct_tgt_team", "ct_tgt_start", "ct_tgt_count", "ct_same"),
                                         "ct_pair_off", stream)
        total_it = self._fill_task_table(intersect, self.program.self_max_intersect_tasks,
                                         ("it_edge_team", "it_edge_start", "it_edge_count",
                                          "it_tri_team", "it_tri_start", "it_tri_count", "it_same"),
                                         "it_pair_off", stream)
        counts = np.zeros(8, np.int32)
        counts[kernels.SCL_ERROR] = 1 if (total_ct > _SELF_PAIR_GUARD or total_it > _SELF_PAIR_GUARD) else 0
        counts[kernels.SCL_USE_INTERSECT] = 1 if intersect else 0
        counts[kernels.SCL_FRAME_INDEX] = int(frame_index) % int(_defs.SELF_COLLISION_INTERSECT_DIV)
        self.self_state["scl_counts"].copy_to_device(counts, stream=stream)
        self.team.device["use_point"].copy_to_device(use_point, stream=stream)
        self.team.device["use_edge"].copy_to_device(use_edge, stream=stream)
        self.team.device["use_triangle"].copy_to_device(use_triangle, stream=stream)

    def _fill_task_table(self, tasks, capacity, column_keys, pair_off_key, stream):
        """Pack a task list into fixed-capacity device columns + a prefix-sum pair-offset array
        (pair_off[k] = pairs before task k; pair_off[capacity] = total). Inactive tail slots keep
        pair_count 0 so no global pair index ever maps to them. Returns the total pair count."""
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
