"""GPU cloth engine (numba.cuda) - resident-state cooperative megakernel backend.

Sibling of ``cloth_engine_cpu``: both consume the same ``cloth_kernel`` core (data model,
scene compiler, IO, frame gate) and expose the same ``pipeline.run_frame(world,
frame_globals)`` boundary, writing the same world numpy arrays -- so ``io.team_output``
and the gate are backend-agnostic. This package imports nothing from the CPU engine.

Only the solver lives here (``pipeline.py`` -> ``engine.py`` -> device kernels). Scene
construction, IO and the data model are used directly from ``cloth_kernel``; there are no
re-export shims in this package.

Self-collision IS implemented (G3a): ``frame_begin`` intersect broad-phase, per-substep
primitive update + contact detect/refresh, the 4-iteration contact solve, and the
``frame_end`` narrow phase all run on device. It differs from the CPU engine in broad-phase
strategy only -- flat n^2 candidate pairs with an exact AABB-overlap predicate, where the
CPU engine uses a uniform grid.

Submodules are imported directly by consumers (the gate imports ``pipeline``); the package
``__init__`` stays import-light so foundational modules (``dmath``, ``program``) can be
exercised without pulling in the CUDA engine.
"""
