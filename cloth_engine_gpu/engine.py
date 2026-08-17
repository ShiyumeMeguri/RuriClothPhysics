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
                       | kernels.PHASE_COLLIDER_PRE)
STEP_PHASES = int(kernels.PHASE_TEAM_STEP | kernels.PHASE_COLLIDER_START
                  | kernels.PHASE_PARTICLES_STEP | kernels.PHASE_BASELINE | kernels.PHASE_TETHER
                  | kernels.PHASE_DISTANCE_A | kernels.PHASE_ANGLE | kernels.PHASE_BENDING
                  | kernels.PHASE_COLLIDER_SOLVE | kernels.PHASE_DISTANCE_B | kernels.PHASE_MOTION
                  | kernels.PHASE_STEP_POST | kernels.PHASE_COLLIDER_END)
FRAME_POST_PHASES = int(kernels.PHASE_DISPLAY | kernels.PHASE_COLLIDER_POST
                        | kernels.PHASE_TEAM_POST)

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

    def load(self, world):
        signature = self._structure_signature(world)
        if self.world is world and self.signature == signature:
            return
        self.world = world
        self.signature = signature
        self.program = build_program(world)
        self.team = device.FieldSet(device.dump_struct(world.team, self.program.num_teams),
                                    self.program.num_teams)
        # Whole-team round-trip goes as one blob transfer + a struct<->SoA bridge kernel,
        # not 167 per-field transfers (measured 20 ms -> ~1 ms; see staging.py).
        self.team_staging = staging.StructStaging(world.team.dtype, self.program.num_teams)
        self.particles = device.FieldSet(device.dump_arena(world.particles, self.program.num_particles),
                                         self.program.num_particles)
        self.colliders = device.FieldSet(device.dump_arena(world.colliders, self.program.num_colliders),
                                         self.program.num_colliders)
        self.transforms = device.FieldSet(device.dump_arena(world.transforms, self.program.num_transforms),
                                          self.program.num_transforms)
        # Narrow per-frame bridges (slim IO): far fewer marshalled args than the full team
        # blob. Built from world dtypes (single source of truth); a bad field name raises here.
        self.input_staging = staging.StructStaging(world.team.dtype, self.program.num_teams,
                                                   fields=_INPUT_UPLOAD_FIELDS)
        self.feedback_staging = staging.StructStaging(world.team.dtype, self.program.num_teams,
                                                      fields=_CONSUMABLE_TEAM_FIELDS)
        self.config_staging = staging.StructStaging(world.team.dtype, self.program.num_teams,
                                                    fields=_CONFIG_TEAM_FIELDS)
        particle_dtype = _arena_subset_dtype(world.particles.spec, _SLIM_OUTPUT_PARTICLE_FIELDS)
        self.particle_out_staging = staging.StructStaging(particle_dtype, self.program.num_particles,
                                                          fields=_SLIM_OUTPUT_PARTICLE_FIELDS)
        if self.program.num_colliders > 0:
            collider_dtype = _arena_subset_dtype(world.colliders.spec, _COLLIDER_INPUT_FIELDS)
            self.collider_input_staging = staging.StructStaging(
                collider_dtype, self.program.num_colliders, fields=_COLLIDER_INPUT_FIELDS)
        else:
            self.collider_input_staging = None
        self.static = {}
        for kernel_name, attr, field in kernels.STATIC_KERNEL_FIELDS:
            self.static[kernel_name] = device.upload_readonly(getattr(self.program, attr)[field])
        for off_name, ord_name, attr in kernels.STATIC_CSR_FIELDS:
            csr = getattr(self.program, attr)
            self.static[off_name] = device.upload_readonly(csr.offsets)
            self.static[ord_name] = device.upload_readonly(csr.order)
        for name in kernels.STATIC_DIRECT_FIELDS:
            self.static[name] = device.upload_readonly(getattr(self.program, name))
        np_particles = max(self.program.num_particles, 1)
        nt = max(self.program.num_teams, 1)
        n_tri = max(self.program.num_triangle_entries, 1)
        self.scratch = {
            "dcorr": cuda.device_array((np_particles, 3), np.float32),
            "dcorr_fixed": cuda.device_array((np_particles, 3), np.int32),
            "dcount": cuda.device_array((np_particles,), np.int32),
            "col_friction_fixed": cuda.device_array((np_particles,), np.int32),
            "col_normal_fixed": cuda.device_array((np_particles, 3), np.int32),
            # T0 resolve_sync gather snapshot: 22 scalars/team (7 time + 8 param +
            # 3 component_world_position + 4 component_world_rotation) so the child
            # write reads a pre-gather copy of its sync_top row (mutual-sync race safe).
            "sync_snapshot": cuda.device_array((nt, 22), np.float32),
            # F2 display._post_triangles per-triangle stash (authorised f64, mirrors the oracle's
            # f64 normal cast + full-f64 tangent), device-resident, indexed by global triangle row.
            "tri_normal_f64": cuda.device_array((n_tri, 3), np.float64),
            "tri_tangent_f64": cuda.device_array((n_tri, 3), np.float64),
        }
        # Wind zones are variable-length and re-uploaded each frame; start empty so
        # zone_count=0 launches (all 17 legacy phase tests) carry safe dummy args.
        self.n_zones = 0
        self.zones_dev = self._make_zone_arrays([])
        # Reset the zone fingerprint so the next upload_zones re-establishes zones_dev (which
        # this reload just reset to empty) rather than skipping on a stale match.
        self._zone_shadow = None
        # A (re)load just uploaded the current config to the device; take its fingerprint so
        # slim frames only re-upload config on a genuine host update_params edit.
        self.config_staging._repack_in(world.team)
        self._config_shadow = self.config_staging._host.tobytes()

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

    def _make_zone_arrays(self, zones):
        host = self._zone_host(zones)
        return {name: device.upload_readonly(host[name]) for name in _ZONE_ARG_ORDER}

    def upload_zones(self, zones):
        # Zones are a low-frequency input (a static scene never re-uploads; animated wind
        # re-uploads only when a value changes). Re-allocating 11 device arrays every frame
        # via cuda.to_device is the single biggest per-frame transfer cost (~1.9 ms measured);
        # a byte fingerprint skips it whenever the packed zone data is unchanged. The resident
        # zones_dev arrays are read-only inputs the kernel never mutates, so they stay valid.
        host = self._zone_host(zones)
        fingerprint = b"".join(host[name].tobytes() for name in _ZONE_ARG_ORDER)
        if len(zones) == self.n_zones and fingerprint == self._zone_shadow:
            return
        self.n_zones = len(zones)
        self._zone_shadow = fingerprint
        self.zones_dev = {name: device.upload_readonly(host[name]) for name in _ZONE_ARG_ORDER}

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

    def launch(self, phase_mask, sub_begin, sub_end, frame_globals):
        fdt, sim_dt, msc, gts, pw0, pw1, pw2, pw3 = self._frame_scalars(frame_globals)
        team_args = [self.team.get(name) for name in kernels.TEAM_KERNEL_FIELDS]
        particle_args = [self.particles.get(name) for name in kernels.PARTICLE_KERNEL_FIELDS]
        transform_args = [self.transforms.get(name) for name in kernels.TRANSFORM_KERNEL_FIELDS]
        collider_args = [self.colliders.get(name) for name in kernels.COLLIDER_KERNEL_FIELDS]
        static_args = [self.static[name] for name, _, _ in kernels.STATIC_KERNEL_FIELDS]
        csr_args = []
        for off_name, ord_name, _ in kernels.STATIC_CSR_FIELDS:
            csr_args.append(self.static[off_name])
            csr_args.append(self.static[ord_name])
        direct_args = [self.static[name] for name in kernels.STATIC_DIRECT_FIELDS]
        scratch_args = [self.scratch["dcorr"], self.scratch["dcorr_fixed"], self.scratch["dcount"],
                        self.scratch["col_friction_fixed"], self.scratch["col_normal_fixed"],
                        self.scratch["sync_snapshot"], self.scratch["tri_normal_f64"],
                        self.scratch["tri_tangent_f64"]]
        zone_args = [self.zones_dev[name] for name in _ZONE_ARG_ORDER]
        blocks = self._blocks()
        kernels.frame_kernel[blocks, _THREADS](
            int32(phase_mask), int32(sub_begin), int32(sub_end),
            fdt, sim_dt, msc, gts, pw0, pw1, pw2, pw3,
            *team_args, *particle_args, *transform_args, *collider_args, *static_args,
            *csr_args, *direct_args, *scratch_args, int32(self.n_zones), *zone_args)

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
            self._upload_inputs_slim(world)
        else:
            self._upload_inputs(world)
        self.upload_zones(frame_globals.zones)
        self.launch(kernels.ALL_PHASES, 0, self._sub_end(frame_globals), frame_globals)
        if self.io_mode == "slim":
            self._download_outputs_slim(world)
        else:
            self._download_outputs(world)

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

    def _maybe_upload_config(self, world, blocks, threads):
        # Byte fingerprint of the config columns; upload only on a genuine host update_params
        # edit. Steady-state playback never touches config, so this is repack + memcmp only.
        self.config_staging._repack_in(world.team)
        fingerprint = self.config_staging._host.tobytes()
        if fingerprint == self._config_shadow:
            return
        self._config_shadow = fingerprint
        self.config_staging.stage.copy_to_device(self.config_staging._host)
        soa = [self.team.device[name] for name in self.config_staging.field_order]
        self.config_staging._explode[blocks, threads](self.config_staging.stage, *soa)

    def _download_outputs_slim(self, world):
        pblocks, pthreads = self._particle_bridge_grid()
        self.particle_out_staging.download(world.particles.arrays, self.particles, pblocks, pthreads)
        tblocks, tthreads = self._team_bridge_grid()
        self.feedback_staging.download(world.team, self.team, tblocks, tthreads)
