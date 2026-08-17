"""GPU pipeline entry point -- same shape as ``cloth_kernel.pipeline.run_frame``.

Production (``blender_host/runtime.py``) calls ``run_frame(world, frame_globals)`` and
then reads results with ``cloth_kernel.io.team_output`` (particle positions +
out_rotations); the return value is ignored. We mirror the oracle exactly at that
boundary: build the same ``Context``, short-circuit the empty frame through
``io.end_frame``, otherwise drive the resident GPU engine for one frame and read the
outputs back into the world numpy arrays, then run ``io.end_frame`` (clear ``enabled``)
so the world is left in the identical post-frame state the oracle leaves.

The engine is cached per ``World`` (by ``id``) and kept resident across frames; its own
structural fingerprint reloads the device Program when the world is re-registered or a
collider binding changes. The multi-frame loop stays in the host caller (one cooperative
launch per frame), never one launch spanning many frames -- that would blow the ~2 s TDR.
"""

from cloth_kernel import io
from cloth_kernel.pipeline import Context

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
    ctx = Context(world, frame_globals)
    if len(ctx.frame_team_index) == 0:
        io.end_frame(world)
        return ctx
    engine = _engine_for(world)
    engine.step_frame(world, frame_globals)
    io.end_frame(world)
    return ctx
