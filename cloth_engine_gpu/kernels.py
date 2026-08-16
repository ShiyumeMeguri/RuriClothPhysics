"""Cooperative frame megakernel.

One kernel runs [frame-pre phases][substep loop][frame-post phases] with grid.sync()
walls at every cross-thread dependency (probe confirmed cooperative launch on sm_75,
544 max blocks @128). A ``phase_mask`` gates the *work* of each phase (the grid.sync
barriers are always executed by every thread, so masking work is safe); production
passes ``ALL_PHASES`` with ``(sub_begin, sub_end) = (0, MAX_SIM_COUNT)`` in a single
launch, while the dev-harness enables one phase at a time to parity-check it in
isolation against oracle intermediate state.

Grid discipline: block=128, grid=min(needed, 544), every phase is a grid-stride loop.
Guard = team enabled & valid & scale_alive (frame phases); substep phases additionally
require ``update_count > k``. Per-team frame-level phases may use internal f64 (mirrors
oracle trs / matrix inverse); per-particle / per-pair phases stay strict-f32.
"""

from numba import cuda, float32, int32
from numba.cuda import cg

from . import dmath

# phase bits (frame-pre / substep / frame-post)
PHASE_ADVANCE = int32(1 << 0)        # T1 team_time.advance
PHASE_TEAM_POST = int32(1 << 1)      # F4 team_time.frame_post

ALL_PHASES = int32(-1)

MAX_SIM_COUNT = 5


@cuda.jit(device=True)
def team_frame_mask(enabled, valid, cws, i):
    """frame_team_mask[i] = enabled & valid & (min|component_world_scale| >= 1e-6)."""
    if enabled[i] == 0 or valid[i] == 0:
        return False
    ax = abs(cws[i, 0])
    ay = abs(cws[i, 1])
    az = abs(cws[i, 2])
    lo = ax
    if ay < lo:
        lo = ay
    if az < lo:
        lo = az
    return lo >= float32(1e-6)


@cuda.jit(device=True)
def do_advance(i, fdt, sim_dt, max_sim_count, global_time_scale,
               time_reset, time, old_time, now_update, old_update, frame_update, frame_old,
               frame_dt, time_scale, now_time_scale, update_count, skip_count, running):
    reset = time_reset[i] != 0
    t_time = float32(0.0) if reset else time[i]
    t_now_update = float32(0.0) if reset else now_update[i]
    t_old_update = float32(0.0) if reset else old_update[i]
    t_frame_update = float32(0.0) if reset else frame_update[i]
    t_frame_old = float32(0.0) if reset else frame_old[i]

    frame_dt[i] = fdt
    ts = time_scale[i] * global_time_scale
    now_time_scale[i] = ts
    add_time = fdt * ts
    new_time = t_time + add_time
    interval = new_time - t_now_update
    uc = int32(interval / sim_dt)
    clamped = uc if uc < max_sim_count else max_sim_count
    skip = uc - clamped
    if skip > 0:
        new_time = new_time - sim_dt * float32(skip)
    guard = (clamped > 0) and (add_time == float32(0.0))
    if guard:
        clamped = int32(0)
        skip = int32(0)
        new_now_update = new_time - sim_dt + float32(0.0001)
    else:
        new_now_update = t_now_update
    updated = clamped > 0

    old_time[i] = t_time
    time[i] = new_time
    now_update[i] = new_now_update
    if updated:
        frame_old[i] = t_frame_update
        frame_update[i] = new_time
        old_update[i] = new_now_update
    else:
        frame_old[i] = t_frame_old
        frame_update[i] = t_frame_update
        old_update[i] = t_old_update
    update_count[i] = clamped
    skip_count[i] = skip
    running[i] = int32(1) if updated else int32(0)


@cuda.jit(cache=True)
def frame_kernel(phase_mask, sub_begin, sub_end,
                 fdt, sim_dt, max_sim_count, global_time_scale,
                 t_enabled, t_valid, t_cws, t_time_reset,
                 t_time, t_old_time, t_now_update, t_old_update, t_frame_update, t_frame_old,
                 t_frame_dt, t_time_scale, t_now_time_scale, t_update_count, t_skip_count,
                 t_running):
    grid = cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    num_teams = t_enabled.shape[0]

    # ----- FRAME-PRE -----
    if phase_mask & PHASE_ADVANCE:
        i = tid
        while i < num_teams:
            if team_frame_mask(t_enabled, t_valid, t_cws, i):
                do_advance(i, fdt, sim_dt, max_sim_count, global_time_scale,
                           t_time_reset, t_time, t_old_time, t_now_update, t_old_update,
                           t_frame_update, t_frame_old, t_frame_dt, t_time_scale,
                           t_now_time_scale, t_update_count, t_skip_count, t_running)
            i += stride
    grid.sync()

    # ----- SUBSTEP LOOP (phases added in dependency order as ported) -----
    for _k in range(sub_begin, sub_end):
        grid.sync()

    # ----- FRAME-POST (F4 team_time.frame_post added next) -----
    grid.sync()


# ordered team-field names the frame_kernel consumes, in signature order after the
# four scalars; the engine builds the launch arg list from this single source of truth.
TEAM_KERNEL_FIELDS = (
    "enabled", "valid", "component_world_scale", "time_reset_pending",
    "time", "old_time", "now_update_time", "old_update_time", "frame_update_time",
    "frame_old_time", "frame_delta_time", "time_scale", "now_time_scale",
    "update_count", "skip_count", "running",
)
