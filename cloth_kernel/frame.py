"""Frame-level team gate -- the single definition of "which teams run this frame".

Both solvers must agree exactly on this predicate: a team participates when it is
``enabled`` (the host set frame input for it), ``valid`` (registered, not a freed slot),
and its component world scale is non-degenerate on every axis -- a zero axis makes the
TRS matrix singular, so every downstream world<->local transform would blow up.

It lives in the core rather than in either engine because a divergence here is silent:
the two backends would simply simulate different sets of teams and every downstream
comparison would be meaningless.
"""

import numpy as np

# A scale axis below this counts as collapsed. Matches the device-side check in
# cloth_engine_gpu.kernels.team_frame_mask.
SCALE_EPSILON = 1e-6


def frame_team_mask(world):
    """Boolean mask over team slots: which teams simulate this frame."""
    scale = world.team["component_world_scale"]
    scale_alive = np.abs(scale).min(axis=1) >= SCALE_EPSILON
    return world.team["enabled"] & world.team["valid"] & scale_alive


def frame_team_index(world):
    """Slot indices of the teams that simulate this frame."""
    return np.flatnonzero(frame_team_mask(world))


def has_frame_teams(world):
    """True when at least one team simulates this frame (empty-frame short circuit)."""
    return bool(frame_team_mask(world).any())
