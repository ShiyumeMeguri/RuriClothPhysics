"""CPU (numpy) cloth solver -- the reference implementation of the frame pipeline.

Sibling of ``cloth_engine_gpu``: both consume the same ``cloth_kernel`` core (data
model, scene compiler, IO, frame gate) and expose the same
``pipeline.run_frame(world, frame_globals)`` boundary, so a caller can swap backends
without touching scene construction or output reading.

Production drives the GPU backend (see ``blender_host.runtime``). This package is the
cross-source reference the GPU is checked against -- it is deliberately NOT imported
by the runtime, so its numpy stages never load in a normal Blender session.
"""
