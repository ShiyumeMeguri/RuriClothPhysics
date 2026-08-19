from cloth_kernel import frame as _frame
from cloth_kernel import io

from .engine import GpuEngine

_engines = {}


def _engine_for(world):
    engine = _engines.get(id(world))
    if engine is None or engine.world is not world:
        engine = GpuEngine(world)
        _engines[id(world)] = engine
    return engine


def run_frame(world, frame_globals):
    world.ensure_buckets()
    if not _frame.has_frame_teams(world):
        io.end_frame(world)
        return
    engine = _engine_for(world)
    engine.step_frame(world, frame_globals)
    io.end_frame(world)


def release(world=None):
    for key, engine in list(_engines.items()):
        target = engine.world
        if world is not None and target is not world:
            continue
        if target is not None:
            if engine.program.num_teams:
                engine.download_team(target)
            if engine.program.num_particles:
                engine.download_particles(target)
            if engine.program.num_colliders:
                engine.download_colliders(target)
        del _engines[key]
