import numpy as np

from ..cloth_kernel import defs
from ..cloth_kernel import frame
from ..cloth_kernel import io
from .stages import angle
from .stages import baseline
from .stages import bending
from .stages import center
from .stages import collider
from .stages import display
from .stages import distance
from .stages import motion
from .stages import particles
from .stages import self_collision
from .stages import team_time
from .stages import tether


class Context:
    def __init__(self, world, frame_globals):
        self.world = world
        self.zones = frame_globals.zones
        self.frame_delta_time = frame_globals.frame_delta_time
        self.simulation_delta_time = 1.0 / frame_globals.simulation_frequency
        self.max_simulation_count = frame_globals.max_simulation_count
        self.global_time_scale = frame_globals.global_time_scale
        self.power = np.array(defs.simulation_power(frame_globals.simulation_frequency),
                              dtype=np.float32)
        self.intersect_frame_index = frame_globals.frame_index % defs.SELF_COLLISION_INTERSECT_DIV
        self.frame_team_mask = frame.frame_team_mask(world)
        self.frame_team_index = np.flatnonzero(self.frame_team_mask)
        self.frame_particle_index = np.flatnonzero(
            self.frame_team_mask[world.particles["team"]])
        self.step_team_mask = np.zeros(len(world.team), dtype=bool)
        self.step_team_index = np.zeros(0, dtype=np.int64)
        self.step_particle_index = np.zeros(0, dtype=np.int64)
        self._frame_entry_cache = {}
        self._step_entry_cache = {}
        self.self_contact_tasks = []
        self.self_intersect_tasks = []
        self.self_intersect_pairs = []
        self.self_use_intersect = False
        self.self_max_size = None

    def begin_step(self, step_index):
        tt = self.world.team
        self.step_team_mask = self.frame_team_mask & (tt["update_count"] > step_index)
        self.step_team_index = np.flatnonzero(self.step_team_mask)
        self.step_particle_index = np.flatnonzero(
            self.step_team_mask[self.world.particles["team"]])
        self._step_entry_cache = {}

    def entry_index(self, name, arena):
        cached = self._step_entry_cache.get(name)
        if cached is None:
            cached = np.flatnonzero(self.step_team_mask[arena["team"]])
            self._step_entry_cache[name] = cached
        return cached

    def frame_entry_index(self, name, arena):
        cached = self._frame_entry_cache.get(name)
        if cached is None:
            cached = np.flatnonzero(self.frame_team_mask[arena["team"]])
            self._frame_entry_cache[name] = cached
        return cached


STEP_STAGES = (
    team_time.step_update,
    collider.start_step,
    particles.step_update,
    baseline.run,
    tether.run,
    distance.run,
    angle.run,
    bending.run,
    collider.solve,
    distance.run,
    motion.run,
)


def release(world=None):
    pass


def flush(world=None):
    pass


def run_frame(world, frame_globals):
    world.ensure_buckets()
    ctx = Context(world, frame_globals)
    if len(ctx.frame_team_index) == 0:
        io.end_frame(world)
        return ctx

    team_time.resolve_sync(world)
    team_time.advance(world, ctx)
    particles.compute_base_pose(world, ctx)
    center.run(world, ctx)
    particles.frame_pre(world, ctx)
    collider.frame_pre(world, ctx)
    self_collision.frame_begin(world, ctx)

    max_update = int(world.team["update_count"][ctx.frame_team_index].max()) \
        if len(ctx.frame_team_index) else 0
    for step_index in range(max_update):
        ctx.begin_step(step_index)
        if len(ctx.step_team_index) == 0:
            continue
        for stage in STEP_STAGES:
            stage(world, ctx)
        self_collision.step(world, ctx, first_step=step_index == 0)
        particles.step_post(world, ctx)
        collider.end_step(world, ctx)

    self_collision.frame_end(world, ctx)
    display.run(world, ctx)
    collider.frame_post(world, ctx)
    team_time.frame_post(world, ctx)
    io.end_frame(world)
    return ctx
