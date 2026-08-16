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
from .program import build_program

_MAX_COOP_BLOCKS = 544
_THREADS = 128


class GpuEngine:
    def __init__(self, world):
        self.world = None
        self.generation = None
        self.program = None
        self.team = None
        self.particles = None
        self.colliders = None
        self.transforms = None
        self.load(world)

    # ---- lifecycle ----------------------------------------------------------
    @staticmethod
    def _generation(world):
        return (id(world), len(world.entries),
                int(world.team["p_start"].sum()), int(world.team["p_count"].sum()))

    def load(self, world):
        generation = self._generation(world)
        if self.world is world and self.generation == generation:
            return
        self.world = world
        self.generation = generation
        self.program = build_program(world)
        self.team = device.FieldSet(device.dump_struct(world.team, self.program.num_teams),
                                    self.program.num_teams)
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
        self.scratch = {
            "dcorr": cuda.device_array((np_particles, 3), np.float32),
            "dcorr_fixed": cuda.device_array((np_particles, 3), np.int32),
            "dcount": cuda.device_array((np_particles,), np.int32),
            "col_friction_fixed": cuda.device_array((np_particles,), np.int32),
            "col_normal_fixed": cuda.device_array((np_particles, 3), np.int32),
        }

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
                        self.scratch["col_friction_fixed"], self.scratch["col_normal_fixed"]]
        blocks = self._blocks()
        kernels.frame_kernel[blocks, _THREADS](
            int32(phase_mask), int32(sub_begin), int32(sub_end),
            fdt, sim_dt, msc, gts, pw0, pw1, pw2, pw3,
            *team_args, *particle_args, *transform_args, *collider_args, *static_args,
            *csr_args, *direct_args, *scratch_args)

    # ---- production API (grows as phases land) ------------------------------
    def step_frame(self, world, frame_globals):
        self.load(world)
        self._upload_inputs(world)
        self.launch(kernels.ALL_PHASES, 0, kernels.MAX_SIM_COUNT, frame_globals)
        self._download_outputs(world)

    def step_frame_captured(self, world, frame_globals):
        # segmented launch for per-substep assertions: (pre,0,0) then per-k, then post.
        self.load(world)
        self._upload_inputs(world)
        self.launch(kernels.ALL_PHASES, 0, kernels.MAX_SIM_COUNT, frame_globals)
        self._download_outputs(world)

    # input/output field routing (extended as phases land). During bring-up the
    # dev-harness uses upload_all/download_* instead, so these stay minimal + honest.
    _INPUT_TEAM_FIELDS = ("enabled", "valid", "component_world_position",
                          "component_world_rotation", "component_world_scale",
                          "culling_invisible", "distance_weight", "sync_target",
                          "has_anchor", "anchor_position", "anchor_rotation",
                          "force_mode", "impact_force", "time_scale")
    _OUTPUT_PARTICLE_FIELDS = ("positions", "out_rotations", "velocities")
    _OUTPUT_TEAM_FIELDS = ("wind_count", "wind_zone_id")

    def _upload_inputs(self, world):
        self.team.upload_many(device.dump_struct(world.team, self.program.num_teams),
                              [n for n in self._INPUT_TEAM_FIELDS if n in self.team.device])
        self.transforms.upload_many(device.dump_arena(world.transforms, self.program.num_transforms),
                                    ["world"])
        self.colliders.upload_many(device.dump_arena(world.colliders, self.program.num_colliders),
                                   [n for n in ("input_positions", "input_rotations", "input_scales",
                                                "enabled") if n in self.colliders.device])

    def _download_outputs(self, world):
        self.download_particles(world, list(self._OUTPUT_PARTICLE_FIELDS))
        self.download_team(world, list(self._OUTPUT_TEAM_FIELDS))
