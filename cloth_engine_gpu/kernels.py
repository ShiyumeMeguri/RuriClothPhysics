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
PHASE_BASE_POSE = int32(1 << 1)      # P0/P1 particles.compute_base_pose (skinning)
PHASE_TEAM_POST = int32(1 << 2)      # F4 team_time.frame_post

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


@cuda.jit(device=True)
def _skin_row(world, bind, t, r, c):
    # (world[t] @ bind[t])[r, c] in f32 (matches oracle einsum('tij,tjk->tik') rounding)
    return (world[t, r, 0] * bind[t, 0, c] + world[t, r, 1] * bind[t, 1, c]
            + world[t, r, 2] * bind[t, 2, c] + world[t, r, 3] * bind[t, 3, c])


@cuda.jit(device=True)
def do_base_pose(p, p_team, local_positions, local_normals, local_tangents,
                 skin_indices, skin_weights, positions, rotations, world, bind):
    wp = cuda.local.array(3, float32)
    wn = cuda.local.array(3, float32)
    wt = cuda.local.array(3, float32)
    for r in range(3):
        wp[r] = float32(0.0)
        wn[r] = float32(0.0)
        wt[r] = float32(0.0)
    lx = local_positions[p, 0]
    ly = local_positions[p, 1]
    lz = local_positions[p, 2]
    lnx = local_normals[p, 0]
    lny = local_normals[p, 1]
    lnz = local_normals[p, 2]
    ltx = local_tangents[p, 0]
    lty = local_tangents[p, 1]
    ltz = local_tangents[p, 2]
    for j in range(4):
        w = skin_weights[p, j]
        t = skin_indices[p, j]
        for r in range(3):
            s0 = _skin_row(world, bind, t, r, 0)
            s1 = _skin_row(world, bind, t, r, 1)
            s2 = _skin_row(world, bind, t, r, 2)
            s3 = _skin_row(world, bind, t, r, 3)
            wp[r] += w * (s0 * lx + s1 * ly + s2 * lz + s3)
            wn[r] += w * (s0 * lnx + s1 * lny + s2 * lnz)
            wt[r] += w * (s0 * ltx + s1 * lty + s2 * ltz)
    positions[p, 0] = wp[0]
    positions[p, 1] = wp[1]
    positions[p, 2] = wp[2]
    nx, ny, nz = dmath.normalize3_fb(wn[0], wn[1], wn[2], float32(0.0), float32(1.0), float32(0.0))
    tx, ty, tz = dmath.normalize3_fb(wt[0], wt[1], wt[2], float32(0.0), float32(0.0), float32(1.0))
    qx, qy, qz, qw = dmath.to_rotation(nx, ny, nz, tx, ty, tz)
    rotations[p, 0] = qx
    rotations[p, 1] = qy
    rotations[p, 2] = qz
    rotations[p, 3] = qw


@cuda.jit(cache=True)
def frame_kernel(phase_mask, sub_begin, sub_end,
                 fdt, sim_dt, max_sim_count, global_time_scale,
                 t_enabled, t_valid, t_cws, t_time_reset,
                 t_time, t_old_time, t_now_update, t_old_update, t_frame_update, t_frame_old,
                 t_frame_dt, t_time_scale, t_now_time_scale, t_update_count, t_skip_count,
                 t_running,
                 p_team, p_local_positions, p_local_normals, p_local_tangents,
                 p_skin_indices, p_skin_weights, p_positions, p_rotations,
                 x_world, x_bind):
    grid = cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    num_teams = t_enabled.shape[0]

    num_particles = p_team.shape[0]

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

    if phase_mask & PHASE_BASE_POSE:
        p = tid
        while p < num_particles:
            if team_frame_mask(t_enabled, t_valid, t_cws, p_team[p]):
                do_base_pose(p, p_team, p_local_positions, p_local_normals, p_local_tangents,
                             p_skin_indices, p_skin_weights, p_positions, p_rotations,
                             x_world, x_bind)
            p += stride
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

PARTICLE_KERNEL_FIELDS = (
    "team", "local_positions", "local_normals", "local_tangents",
    "skin_indices", "skin_weights", "positions", "rotations",
)

TRANSFORM_KERNEL_FIELDS = ("world", "bind_pose")
