from ..cloth_kernel import frame as _frame
from ..cloth_kernel import io as _io

from .engine import ClothEngine

_engines = {}


def _engine_for(world, target_name):
    engine = _engines.get(id(world))
    if engine is not None and engine.world is world \
            and engine.target_name == target_name:
        return engine
    if engine is not None:
        release(engine.world)
    engine = ClothEngine(world, target_name)
    _engines[id(world)] = engine
    return engine


def run_frame(world, frame_globals, target_name):
    world.ensure_buckets()
    if not _frame.has_frame_teams(world):
        _io.end_frame(world)
        return
    engine = _engine_for(world, target_name)
    engine.step_frame(world, frame_globals)
    _io.end_frame(world)


def _can_download(engine, target):
    if engine.program is None or engine.structure_revision != int(target.structure_revision):
        return False
    return bool(engine.program.num_teams)


def _download(engine, target):
    if _can_download(engine, target):
        engine.download_state(target)


def flush(world):
    for engine in _engines.values():
        if engine.world is world:
            _download(engine, world)


def download_display(world):
    for engine in _engines.values():
        if engine.world is world and _can_download(engine, world):
            engine.download_display(world)


def release(world=None):
    for key, engine in list(_engines.items()):
        target = engine.world
        if world is not None and target is not world:
            continue
        if target is not None:
            _download(engine, target)
        del _engines[key]
