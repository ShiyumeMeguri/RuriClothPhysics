"""GPU cloth engine (numba.cuda) - resident-state cooperative megakernel backend.

This package mirrors the CPU oracle ``cloth_kernel`` at the pipeline boundary:
``pipeline.run_frame(world, frame_globals)`` has the same signature and writes the
same world numpy arrays, so ``io.team_output`` and the gate are backend-agnostic.

Scene construction, IO, compile and the data model are single-sourced from the
oracle by re-export (see ``defs.py``, ``compile.py``, ``io.py``, ``world.py`` and
``stages/self_collision.py``). Only the solver (``pipeline.py`` -> ``engine.py`` ->
device kernels) is re-implemented on the GPU.

Self-collision is NOT implemented here (G3 scope); the substep contact slot and the
frame self-collision phases are wired as no-ops so that self_mode=NONE scenes match
the oracle exactly.

Submodules are imported directly by consumers (the gate imports ``pipeline``); the
package ``__init__`` stays import-light so foundational modules (``dmath``,
``program``) can be exercised without pulling in the CUDA engine.
"""
