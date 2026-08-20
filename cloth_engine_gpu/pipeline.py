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


def _download(engine, target):
    if engine.program is None or engine.signature != GpuEngine._structure_signature(target):
        return
    if engine.program.num_teams:
        engine.download_state(target)


def flush(world):
    for engine in _engines.values():
        if engine.world is world:
            _download(engine, world)


def release(world=None):
    for key, engine in list(_engines.items()):
        target = engine.world
        if world is not None and target is not world:
            continue
        if target is not None:
            _download(engine, target)
        del _engines[key]
