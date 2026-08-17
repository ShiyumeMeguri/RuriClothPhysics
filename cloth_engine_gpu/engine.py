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


class GpuEngine:
    def __init__(self, world):
        self.world = None
        self.signature = None
        self.program = None
        self.team = None
        self.particles = None
        self.colliders = None
        self.transforms = None
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

    # ---- wind-zone per-frame upload -----------------------------------------
    def _make_zone_arrays(self, zones):
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
        return {name: device.upload_readonly(host[name]) for name in _ZONE_ARG_ORDER}

    def upload_zones(self, zones):
        self.n_zones = len(zones)
        self.zones_dev = self._make_zone_arrays(zones)

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
    def step_frame(self, world, frame_globals):
        """One host frame: upload this frame's host-mutated inputs, run the whole
        pipeline in a single cooperative launch, read the outputs back into the world."""
        self.load(world)
        self._upload_inputs(world)
        self.upload_zones(frame_globals.zones)
        self.launch(kernels.ALL_PHASES, 0, kernels.MAX_SIM_COUNT, frame_globals)
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
