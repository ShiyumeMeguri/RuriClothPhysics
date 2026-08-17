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

import math

from numba import cuda, float32, float64, int8, int32
from numba.cuda import cg, libdevice

from . import dmath

# phase bits (frame-pre / substep / frame-post)
PHASE_ADVANCE = int32(1 << 0)        # T1 team_time.advance
PHASE_BASE_POSE = int32(1 << 1)      # P0/P1 particles.compute_base_pose (skinning)
PHASE_TETHER = int32(1 << 2)         # S5 tether.run (substep)
PHASE_TEAM_POST = int32(1 << 3)      # F4 team_time.frame_post
PHASE_PARTICLES_STEP = int32(1 << 4)  # S3 particles.step_update (substep)
PHASE_STEP_POST = int32(1 << 5)      # S13 particles.step_post (substep)
PHASE_DISTANCE_A = int32(1 << 6)     # S6 distance.run (first substep occurrence)
PHASE_DISTANCE_B = int32(1 << 7)     # S10 distance.run (second substep occurrence)
PHASE_MOTION = int32(1 << 8)         # S11 motion.run (substep)
PHASE_BENDING = int32(1 << 9)        # S8 bending.run (substep)
PHASE_BASELINE = int32(1 << 10)      # S4 baseline.run FK (substep)
PHASE_TEAM_STEP = int32(1 << 11)     # S1 team_time.step_update (substep, per-team)
PHASE_COLLIDER_PRE = int32(1 << 12)  # K0 collider.frame_pre
PHASE_COLLIDER_START = int32(1 << 13)  # S2 collider.start_step (substep)
PHASE_COLLIDER_SOLVE = int32(1 << 14)  # S9 collider.solve point+edge (substep)
PHASE_COLLIDER_END = int32(1 << 15)  # S14 collider.end_step (substep)
PHASE_COLLIDER_POST = int32(1 << 16)  # F3 collider.frame_post
PHASE_PARTICLES_PRE = int32(1 << 17)  # P2 particles.frame_pre
PHASE_SYNC = int32(1 << 18)          # T0 team_time.resolve_sync (frame-pre, per-team)
PHASE_CENTER = int32(1 << 19)        # C0 center.run + select_team_wind (frame-pre, per-team)
PHASE_ANGLE = int32(1 << 20)         # S7 angle.run (substep; limit + restoration passes)
PHASE_DISPLAY = int32(1 << 21)       # F2 display.run (frame-post; _display/_postline/_post_triangles/_output)

ALL_PHASES = int32(-1)

MAX_SIM_COUNT = 5

# wind-zone mode enum (mirror engine._ZONE_MODE) + slot count
WIND_ZONE_SLOTS = 4
ZONE_GLOBAL = int32(0)
ZONE_BOX = int32(1)
ZONE_SPHERE_DIR = int32(2)
ZONE_SPHERE_RADIAL = int32(3)
TELEPORT_RESET = int32(1)

# defs constants mirrored (device-side literals stay f32-wrapped)
TETHER_STRETCH_LIMIT = float32(0.03)
TETHER_STIFFNESS_WIDTH = float32(0.3)
TETHER_VELOCITY_ATTENUATION = float32(0.7)
EPSILON = float32(1e-8)

WIND_BASE_SPEED = float32(7.5)
WIND_TURBULENCE_ANGLE = float32(45.0)
WIND_MAX_TIME = float32(10000.0)
DEG2RAD = float32(math.pi / 180.0)
RAD2DEG = float32(180.0 / math.pi)

FORCE_VELOCITY_ADD = int32(1)
FORCE_VELOCITY_ADD_WITHOUT_DEPTH = int32(2)
FORCE_VELOCITY_CHANGE = int32(3)
FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH = int32(4)

BONE_SPRING_FIX_MASS = float32(10.0)
BONE_CLOTH_FIX_MASS = float32(50.0)
DISTANCE_HORIZONTAL_STIFFNESS = float32(0.5)
DISTANCE_VELOCITY_ATTENUATION = float32(0.3)

VOLUME_SIGN = int32(100)
VOLUME_SCALE = float32(1000.0)
BENDING_FIX_INV_MASS = float32(0.01)
ONE_SIXTH = float32(1.0 / 6.0)
TO_FIXED = float32(1e6)

# S7 angle.run (defs.ANGLE_LIMIT_*): 3 iterations, per-iteration restoration ratio 0.1/0.3/0.5,
# constant limit rotation ratio 0.4 and velocity attenuation 0.9.
ANGLE_ITERATION = 3
ANGLE_LIMIT_ROT_RATIO = float32(0.4)
ANGLE_LIMIT_ATTENUATION = float32(0.9)

FRICTION_MASS = float32(3.0)
COLLIDER_SPHERE = int32(0)
COLLIDER_CAPSULE = int32(1)
COLLIDER_PLANE = int32(2)
COLLISION_POINT = int32(1)
COLLISION_EDGE = int32(2)
INF = float32(math.inf)
MAX_DISTANCE_RATIO_FUTURE_PREDICTION = float32(1.3)


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


@cuda.jit(device=True)
def do_tether(e, tether_particle, p_team, next_positions, velocity_positions,
              step_basic_positions, vertex_root, t_tether_compression):
    idx = tether_particle[e]
    team = p_team[idx]
    root = vertex_root[idx]
    compression_limit = float32(1.0) - t_tether_compression[team]
    stretch_limit = float32(1.0) + TETHER_STRETCH_LIMIT

    vx = next_positions[root, 0] - next_positions[idx, 0]
    vy = next_positions[root, 1] - next_positions[idx, 1]
    vz = next_positions[root, 2] - next_positions[idx, 2]
    distance = dmath.length3(vx, vy, vz)
    cvx = step_basic_positions[idx, 0] - step_basic_positions[root, 0]
    cvy = step_basic_positions[idx, 1] - step_basic_positions[root, 1]
    cvz = step_basic_positions[idx, 2] - step_basic_positions[root, 2]
    calc_distance = dmath.length3(cvx, cvy, cvz)

    valid = (distance >= EPSILON) and (calc_distance != float32(0.0))
    ratio = distance / (calc_distance if calc_distance != float32(0.0) else float32(1.0))
    compress = valid and (ratio < compression_limit)
    stretch = valid and (ratio > stretch_limit)
    if not (compress or stretch):
        return
    if compress:
        dist = distance - compression_limit * calc_distance
        stiffness = dmath.saturate((compression_limit - ratio) / TETHER_STIFFNESS_WIDTH)
    else:
        dist = distance - stretch_limit * calc_distance
        stiffness = dmath.saturate((ratio - stretch_limit) / TETHER_STIFFNESS_WIDTH)
    inv = float32(1.0) / (distance if distance > float32(1e-30) else float32(1.0))
    scale = dist * stiffness * inv
    ax = vx * scale
    ay = vy * scale
    az = vz * scale
    next_positions[idx, 0] += ax
    next_positions[idx, 1] += ay
    next_positions[idx, 2] += az
    velocity_positions[idx, 0] += ax * TETHER_VELOCITY_ATTENUATION
    velocity_positions[idx, 1] += ay * TETHER_VELOCITY_ATTENUATION
    velocity_positions[idx, 2] += az * TETHER_VELOCITY_ATTENUATION


@cuda.jit(device=True)
def do_wind_blend(wind_main, time, dqx, dqy, dqz, dqw, zone_turbulence,
                  blend, turbulence_param, wind_position):
    """One wind slot's force direction*strength; mirrors wind._wind_force_blend."""
    active = wind_main >= float32(0.01)
    main_ratio = wind_main / WIND_BASE_SPEED

    sin_pos = wind_position + time * float32(10.0)
    sin_wave = libdevice.sinf(sin_pos)

    noise_pos = wind_position + time * float32(2.3132)
    noise_wave = dmath.cnoise2(noise_pos, noise_pos) * float32(2.3)

    wave_x = sin_wave + (noise_wave - sin_wave) * blend
    wave_y = wave_x

    turbulence = zone_turbulence * turbulence_param

    angle_x = (wave_x * WIND_TURBULENCE_ANGLE) * DEG2RAD
    angle_y = (wave_y * WIND_TURBULENCE_ANGLE) * DEG2RAD
    angle_y = angle_y * (float32(0.1) + (float32(0.5) - float32(0.1)) * blend)
    angle_x = angle_x * turbulence
    angle_y = angle_y * turbulence

    rqx, rqy, rqz, rqw = dmath.euler_yx(angle_x, angle_y)
    cqx, cqy, cqz, cqw = dmath.quat_mul(dqx, dqy, dqz, dqw, rqx, rqy, rqz, rqw)
    wdx, wdy, wdz = dmath.quat_to_tangent(cqx, cqy, cqz, cqw)

    main_scale = dmath.saturate(float32(1.0) - main_ratio)
    main_wave = (wave_x + float32(1.0)) * float32(0.5)
    main_wave = main_wave * main_scale * turbulence
    strength = wind_main - wind_main * main_wave
    if not active:
        strength = float32(0.0)
    return (wdx * strength, wdy * strength, wdz * strength)


@cuda.jit(device=True)
def do_distance_gather(p, p_team, next_positions, base_positions, depth, friction, attr_move,
                       t_is_spring, t_animation_pose_ratio, t_init_scale, t_scale_ratio,
                       t_distance_lut, power1, csr_offsets, csr_order,
                       distance_target, distance_rest, sc_dcorr):
    """Jacobi gather for one distance particle: mean correction over its distance entries
    (f64 accumulation mirrors oracle np.add.reduceat / run_counts, then cast to f32)."""
    mt = p_team[p]
    fix_mass = BONE_SPRING_FIX_MASS if t_is_spring[mt] != 0 else BONE_CLOTH_FIX_MASS
    anime_ratio = t_animation_pose_ratio[mt]
    scale = t_init_scale[mt, 0] * t_scale_ratio[mt]
    depth_p = depth[p]
    fixed_p = attr_move[p] == 0
    inv_mass_p = dmath.calc_inverse_mass_fixed(friction[p], depth_p, fixed_p, fix_mass)
    stiffness = dmath.evaluate_team_lut_clamp01(t_distance_lut, mt, depth_p) * power1
    npx = next_positions[p, 0]
    npy = next_positions[p, 1]
    npz = next_positions[p, 2]
    bpx = base_positions[p, 0]
    bpy = base_positions[p, 1]
    bpz = base_positions[p, 2]

    sumx = float64(0.0)
    sumy = float64(0.0)
    sumz = float64(0.0)
    count_ok = 0
    start = csr_offsets[p]
    stop = csr_offsets[p + 1]
    for k in range(start, stop):
        r = csr_order[k]
        tgt = distance_target[r]
        rest = distance_rest[r]
        if rest >= float32(0.0):
            final_stiffness = dmath.saturate(stiffness)
        else:
            final_stiffness = dmath.saturate(stiffness * DISTANCE_HORIZONTAL_STIFFNESS)
        fixed_t = attr_move[tgt] == 0
        inv_mass_t = dmath.calc_inverse_mass_fixed(friction[tgt], depth[tgt], fixed_t, fix_mass)
        vx = next_positions[tgt, 0] - npx
        vy = next_positions[tgt, 1] - npy
        vz = next_positions[tgt, 2] - npz
        zero_rest = rest == float32(0.0)
        distance = dmath.length3(vx, vy, vz)
        ok = zero_rest or (distance >= EPSILON)
        base_len = dmath.length3(bpx - base_positions[tgt, 0],
                                 bpy - base_positions[tgt, 1],
                                 bpz - base_positions[tgt, 2])
        rest_length = dmath.lerp(libdevice.fabsf(rest) * scale, base_len, anime_ratio)
        safe_d = distance if distance > float32(1e-30) else float32(1.0)
        nx = vx / safe_d
        ny = vy / safe_d
        nz = vz / safe_d
        a = (distance - rest_length) * final_stiffness
        denom = inv_mass_p + inv_mass_t
        cxr = a * nx / denom * inv_mass_p
        cyr = a * ny / denom * inv_mass_p
        czr = a * nz / denom * inv_mass_p
        if zero_rest:
            cxr = vx * float32(0.5)
            cyr = vy * float32(0.5)
            czr = vz * float32(0.5)
        if ok:
            count_ok += 1
            sumx += float64(cxr)
            sumy += float64(cyr)
            sumz += float64(czr)
    if count_ok > 0:
        sc_dcorr[p, 0] = float32(sumx / count_ok)
        sc_dcorr[p, 1] = float32(sumy / count_ok)
        sc_dcorr[p, 2] = float32(sumz / count_ok)
    else:
        sc_dcorr[p, 0] = float32(0.0)
        sc_dcorr[p, 1] = float32(0.0)
        sc_dcorr[p, 2] = float32(0.0)


@cuda.jit(device=True)
def do_step_update(i, sim_dt,
                   t_now_update, t_time, t_frame_old, t_frame_interp,
                   t_now_wp, t_now_wr, t_old_wp, t_old_wr,
                   t_ofwp, t_ofwr, t_ofws, t_fwp, t_fwr, t_fws,
                   t_step_vector, t_step_rotation, t_step_mir, t_step_rir,
                   t_local_inertia, t_lmsl, t_lrsl,
                   t_inertia_vector, t_inertia_rotation,
                   t_angular_velocity, t_rotation_axis,
                   t_init_scale, t_scale_ratio,
                   t_gravity_direction, t_gravity_dot, t_ilgd, t_neg_dir,
                   t_gravity, t_gravity_falloff, t_gravity_ratio,
                   t_velocity_weight, t_stab_time, t_blend_weight, t_bwp, t_distance_weight,
                   t_wind_moving, t_frame_moving_speed, t_moving_wind_main,
                   t_frame_moving_dir, t_moving_wind_dir, t_moving_wind_dirq,
                   t_wind_main, t_wind_frequency, t_wind_count, t_wind_time,
                   t_moving_wind_time):
    nu = t_now_update[i] + sim_dt
    t_now_update[i] = nu
    span = t_time[i] - t_frame_old[i]
    if span > float32(0.0):
        interp = dmath.saturate((nu - t_frame_old[i]) / span)
    else:
        interp = float32(1.0)
    t_frame_interp[i] = interp

    owpx = t_now_wp[i, 0]
    owpy = t_now_wp[i, 1]
    owpz = t_now_wp[i, 2]
    owrx = t_now_wr[i, 0]
    owry = t_now_wr[i, 1]
    owrz = t_now_wr[i, 2]
    owrw = t_now_wr[i, 3]
    t_old_wp[i, 0] = owpx
    t_old_wp[i, 1] = owpy
    t_old_wp[i, 2] = owpz
    t_old_wr[i, 0] = owrx
    t_old_wr[i, 1] = owry
    t_old_wr[i, 2] = owrz
    t_old_wr[i, 3] = owrw

    t = interp
    nwpx = dmath.lerp(t_ofwp[i, 0], t_fwp[i, 0], t)
    nwpy = dmath.lerp(t_ofwp[i, 1], t_fwp[i, 1], t)
    nwpz = dmath.lerp(t_ofwp[i, 2], t_fwp[i, 2], t)
    nwrx, nwry, nwrz, nwrw = dmath.quat_slerp(
        t_ofwr[i, 0], t_ofwr[i, 1], t_ofwr[i, 2], t_ofwr[i, 3],
        t_fwr[i, 0], t_fwr[i, 1], t_fwr[i, 2], t_fwr[i, 3], t)
    wsx = dmath.lerp(t_ofws[i, 0], t_fws[i, 0], t)
    wsy = dmath.lerp(t_ofws[i, 1], t_fws[i, 1], t)
    wsz = dmath.lerp(t_ofws[i, 2], t_fws[i, 2], t)
    t_now_wp[i, 0] = nwpx
    t_now_wp[i, 1] = nwpy
    t_now_wp[i, 2] = nwpz
    t_now_wr[i, 0] = nwrx
    t_now_wr[i, 1] = nwry
    t_now_wr[i, 2] = nwrz
    t_now_wr[i, 3] = nwrw

    svx = nwpx - owpx
    svy = nwpy - owpy
    svz = nwpz - owpz
    iowrx, iowry, iowrz, iowrw = dmath.quat_inverse(owrx, owry, owrz, owrw)
    srx, sry, srz, srw = dmath.quat_mul(nwrx, nwry, nwrz, nwrw,
                                        iowrx, iowry, iowrz, iowrw)
    step_angle = dmath.quat_angle(owrx, owry, owrz, owrw, nwrx, nwry, nwrz, nwrw)
    t_step_vector[i, 0] = svx
    t_step_vector[i, 1] = svy
    t_step_vector[i, 2] = svz
    t_step_rotation[i, 0] = srx
    t_step_rotation[i, 1] = sry
    t_step_rotation[i, 2] = srz
    t_step_rotation[i, 3] = srw

    li = t_local_inertia[i]
    lmi = float32(1.0) - li
    lri = float32(1.0) - li
    lvx = svx * (float32(1.0) - lmi)
    lvy = svy * (float32(1.0) - lmi)
    lvz = svz * (float32(1.0) - lmi)
    local_speed = dmath.length3(lvx, lvy, lvz) / sim_dt
    limit = t_lmsl[i]
    if (local_speed > limit) and (limit >= float32(0.0)):
        denom = local_speed if local_speed > float32(0.0) else float32(1.0)
        ratio = limit / denom
        lmi = float32(1.0) + (lmi - float32(1.0)) * ratio
    local_angle = step_angle * (float32(1.0) - lri)
    local_angle_speed = (local_angle / sim_dt) * RAD2DEG
    limit = t_lrsl[i]
    if (local_angle_speed > limit) and (limit >= float32(0.0)):
        denom = local_angle_speed if local_angle_speed > float32(0.0) else float32(1.0)
        ratio = limit / denom
        lri = float32(1.0) + (lri - float32(1.0)) * ratio
    t_step_mir[i] = lmi
    t_step_rir[i] = lri

    t_inertia_vector[i, 0] = svx * lmi
    t_inertia_vector[i, 1] = svy * lmi
    t_inertia_vector[i, 2] = svz * lmi
    irx, iry, irz, irw = dmath.quat_slerp(float32(0.0), float32(0.0), float32(0.0), float32(1.0),
                                          srx, sry, srz, srw, lri)
    t_inertia_rotation[i, 0] = irx
    t_inertia_rotation[i, 1] = iry
    t_inertia_rotation[i, 2] = irz
    t_inertia_rotation[i, 3] = irw

    angular_velocity = step_angle / sim_dt
    t_angular_velocity[i] = angular_velocity
    _ang, axx, axy, axz = dmath.quat_to_angle_axis(srx, sry, srz, srw)
    if angular_velocity > EPSILON:
        t_rotation_axis[i, 0] = axx
        t_rotation_axis[i, 1] = axy
        t_rotation_axis[i, 2] = axz
    else:
        t_rotation_axis[i, 0] = float32(0.0)
        t_rotation_axis[i, 1] = float32(0.0)
        t_rotation_axis[i, 2] = float32(0.0)

    isl = dmath.length3(t_init_scale[i, 0], t_init_scale[i, 1], t_init_scale[i, 2])
    if isl < float32(1e-30):
        isl = float32(1e-30)
    wsl = dmath.length3(wsx, wsy, wsz)
    sr = wsl / isl
    if sr < float32(1e-6):
        sr = float32(1e-6)
    t_scale_ratio[i] = sr

    gdx = t_gravity_direction[i, 0]
    gdy = t_gravity_direction[i, 1]
    gdz = t_gravity_direction[i, 2]
    gravity_dot = float32(1.0)
    if (gdx * gdx + gdy * gdy + gdz * gdz) > EPSILON:
        ilx = t_ilgd[i, 0]
        ily = t_ilgd[i, 1] * t_neg_dir[i, 1]
        ilz = t_ilgd[i, 2]
        wfx, wfy, wfz = dmath.quat_rotate(nwrx, nwry, nwrz, nwrw, ilx, ily, ilz)
        gdot = wfx * gdx + wfy * gdy + wfz * gdz
        gravity_dot = dmath.saturate(gdot * float32(0.5) + float32(0.5))
    t_gravity_dot[i] = gravity_dot

    gravity_ratio = float32(1.0)
    if (t_gravity[i] > float32(1e-6)) and (t_gravity_falloff[i] > float32(1e-6)):
        low = dmath.saturate(float32(1.0) - t_gravity_falloff[i])
        gravity_ratio = low + (float32(1.0) - low) * dmath.saturate(float32(1.0) - gravity_dot)
    t_gravity_ratio[i] = gravity_ratio

    vw = t_velocity_weight[i]
    if vw < float32(1.0):
        stab = t_stab_time[i]
        if stab > float32(1e-6):
            add = sim_dt / stab
        else:
            add = float32(1.0)
        vw = vw + add
        if vw > float32(1.0):
            vw = float32(1.0)
        t_velocity_weight[i] = vw
    t_blend_weight[i] = dmath.saturate(vw * t_bwp[i] * t_distance_weight[i])

    moving_active = t_wind_moving[i] > float32(0.01)
    if moving_active:
        denom = sr if sr > float32(0.0) else float32(1.0)
        mwm = (t_frame_moving_speed[i] * t_wind_moving[i]) / denom
    else:
        mwm = float32(0.0)
    t_moving_wind_main[i] = mwm
    if moving_active:
        mdx = -t_frame_moving_dir[i, 0]
        mdy = -t_frame_moving_dir[i, 1]
        mdz = -t_frame_moving_dir[i, 2]
        t_moving_wind_dir[i, 0] = mdx
        t_moving_wind_dir[i, 1] = mdy
        t_moving_wind_dir[i, 2] = mdz
        mqx, mqy, mqz, mqw = dmath.axis_quaternion(mdx, mdy, mdz)
        t_moving_wind_dirq[i, 0] = mqx
        t_moving_wind_dirq[i, 1] = mqy
        t_moving_wind_dirq[i, 2] = mqz
        t_moving_wind_dirq[i, 3] = mqw

    wf = t_wind_frequency[i]
    wc = t_wind_count[i]
    for s in range(4):
        main_ratio = t_wind_main[i, s] / WIND_BASE_SPEED
        frequency = (float32(0.2) + main_ratio * float32(0.5)) * wf
        if frequency > float32(1.5):
            frequency = float32(1.5)
        frequency = frequency * sim_dt
        if s < wc:
            nt = t_wind_time[i, s] + frequency
            if nt > WIND_MAX_TIME:
                nt = nt - WIND_MAX_TIME * float32(2.0)
            t_wind_time[i, s] = nt
    move_ratio = mwm / WIND_BASE_SPEED
    mf = (float32(0.2) + move_ratio * float32(0.5)) * wf
    if mf > float32(1.5):
        mf = float32(1.5)
    mf = mf * sim_dt
    if moving_active:
        mt2 = t_moving_wind_time[i] + mf
        if mt2 > WIND_MAX_TIME:
            mt2 = mt2 - WIND_MAX_TIME * float32(2.0)
        t_moving_wind_time[i] = mt2


@cuda.jit(device=True)
def _neg_transform_pose(arr_pos, arr_rot, ci, m, flip1, flip2):
    px, py, pz = dmath.transform_point(m, arr_pos[ci, 0], arr_pos[ci, 1], arr_pos[ci, 2])
    arr_pos[ci, 0] = px
    arr_pos[ci, 1] = py
    arr_pos[ci, 2] = pz
    qx, qy, qz, qw = dmath.transform_rotation(m, arr_rot[ci, 0], arr_rot[ci, 1],
                                              arr_rot[ci, 2], arr_rot[ci, 3], flip1, flip2)
    arr_rot[ci, 0] = qx
    arr_rot[ci, 1] = qy
    arr_rot[ci, 2] = qz
    arr_rot[ci, 3] = qw


@cuda.jit(device=True)
def _shift_pose(arr_pos, arr_rot, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw):
    lx = arr_pos[ci, 0] - cpx
    ly = arr_pos[ci, 1] - cpy
    lz = arr_pos[ci, 2] - cpz
    rx, ry, rz = dmath.quat_rotate(srx, sry, srz, srw, lx, ly, lz)
    arr_pos[ci, 0] = rx + cpx + svx
    arr_pos[ci, 1] = ry + cpy + svy
    arr_pos[ci, 2] = rz + cpz + svz
    qx, qy, qz, qw = dmath.quat_mul(srx, sry, srz, srw, arr_rot[ci, 0], arr_rot[ci, 1],
                                    arr_rot[ci, 2], arr_rot[ci, 3])
    arr_rot[ci, 0] = qx
    arr_rot[ci, 1] = qy
    arr_rot[ci, 2] = qz
    arr_rot[ci, 3] = qw


@cuda.jit(device=True)
def _shift_point(arr, p, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw):
    rx, ry, rz = dmath.quat_rotate(srx, sry, srz, srw, arr[p, 0] - cpx, arr[p, 1] - cpy, arr[p, 2] - cpz)
    arr[p, 0] = rx + cpx + svx
    arr[p, 1] = ry + cpy + svy
    arr[p, 2] = rz + cpz + svz


@cuda.jit(device=True)
def _premul_quat(arr, p, srx, sry, srz, srw):
    qx, qy, qz, qw = dmath.quat_mul(srx, sry, srz, srw, arr[p, 0], arr[p, 1], arr[p, 2], arr[p, 3])
    arr[p, 0] = qx
    arr[p, 1] = qy
    arr[p, 2] = qz
    arr[p, 3] = qw


@cuda.jit(device=True)
def _rotate_vec(arr, p, srx, sry, srz, srw):
    vx, vy, vz = dmath.quat_rotate(srx, sry, srz, srw, arr[p, 0], arr[p, 1], arr[p, 2])
    arr[p, 0] = vx
    arr[p, 1] = vy
    arr[p, 2] = vz


@cuda.jit(device=True)
def do_collider_frame_pre(ci, c_team, c_enabled, c_enabled_prev, c_active,
                          c_input_scales, c_input_positions, c_input_rotations, c_center,
                          c_frame_pos, c_frame_rot, c_frame_scl,
                          c_old_frame_pos, c_old_frame_rot, c_now_pos, c_now_rot,
                          c_old_pos, c_old_rot,
                          t_reset_pending, t_neg_teleport, t_neg_matrix, t_neg_change,
                          t_inertia_shift, t_shift_vec, t_shift_rot, t_old_cwp):
    enabled_now = c_enabled[ci] != 0
    rising = enabled_now and (c_enabled_prev[ci] == 0)
    sx = c_input_scales[ci, 0]
    sy = c_input_scales[ci, 1]
    sz = c_input_scales[ci, 2]
    scale_invalid = (abs(sx) < float32(1e-6)) or (abs(sy) < float32(1e-6)) or (abs(sz) < float32(1e-6))
    c_active[ci] = int32(1) if (enabled_now and not scale_invalid) else int32(0)
    c_enabled_prev[ci] = int32(1) if enabled_now else int32(0)
    if not enabled_now:
        return
    team = c_team[ci]
    qx = c_input_rotations[ci, 0]
    qy = c_input_rotations[ci, 1]
    qz = c_input_rotations[ci, 2]
    qw = c_input_rotations[ci, 3]
    cx = sx if abs(sx) >= float32(1e-6) else float32(1e-6)
    cy = sy if abs(sy) >= float32(1e-6) else float32(1e-6)
    cz = sz if abs(sz) >= float32(1e-6) else float32(1e-6)
    sgx = dmath.fsign(cx)
    sgy = dmath.fsign(cy)
    sgz = dmath.fsign(cz)
    rrx, rry, rrz = dmath.quat_rotate(qx, qy, qz, qw, c_center[ci, 0] * sgx,
                                      c_center[ci, 1] * sgy, c_center[ci, 2] * sgz)
    c_frame_pos[ci, 0] = c_input_positions[ci, 0] + rrx * cx * sgx
    c_frame_pos[ci, 1] = c_input_positions[ci, 1] + rry * cy * sgy
    c_frame_pos[ci, 2] = c_input_positions[ci, 2] + rrz * cz * sgz
    c_frame_rot[ci, 0] = qx
    c_frame_rot[ci, 1] = qy
    c_frame_rot[ci, 2] = qz
    c_frame_rot[ci, 3] = qw
    c_frame_scl[ci, 0] = cx
    c_frame_scl[ci, 1] = cy
    c_frame_scl[ci, 2] = cz
    reset = (t_reset_pending[team] != 0) or rising or scale_invalid
    if reset:
        for j in range(3):
            fp = c_frame_pos[ci, j]
            c_old_frame_pos[ci, j] = fp
            c_now_pos[ci, j] = fp
            c_old_pos[ci, j] = fp
        for j in range(4):
            fr = c_frame_rot[ci, j]
            c_old_frame_rot[ci, j] = fr
            c_now_rot[ci, j] = fr
            c_old_rot[ci, j] = fr
        return
    if t_neg_teleport[team] != 0:
        m = t_neg_matrix[team]
        f1 = t_neg_change[team, 1]
        f2 = t_neg_change[team, 2]
        _neg_transform_pose(c_old_frame_pos, c_old_frame_rot, ci, m, f1, f2)
        _neg_transform_pose(c_now_pos, c_now_rot, ci, m, f1, f2)
        _neg_transform_pose(c_old_pos, c_old_rot, ci, m, f1, f2)
    if t_inertia_shift[team] != 0:
        cpx = t_old_cwp[team, 0]
        cpy = t_old_cwp[team, 1]
        cpz = t_old_cwp[team, 2]
        svx = t_shift_vec[team, 0]
        svy = t_shift_vec[team, 1]
        svz = t_shift_vec[team, 2]
        srx = t_shift_rot[team, 0]
        sry = t_shift_rot[team, 1]
        srz = t_shift_rot[team, 2]
        srw = t_shift_rot[team, 3]
        _shift_pose(c_old_frame_pos, c_old_frame_rot, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_pose(c_now_pos, c_now_rot, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_pose(c_old_pos, c_old_rot, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)


@cuda.jit(device=True)
def do_collider_start_step(ci, c_team, c_kind, c_size, c_axis, c_aligned,
                           c_frame_pos, c_frame_rot, c_frame_scl,
                           c_old_frame_pos, c_old_frame_rot, c_now_pos, c_now_rot,
                           c_old_pos, c_old_rot, c_work_rot, c_work_inv_old_rot,
                           c_work_radius, c_work_old_pos, c_work_next_pos,
                           c_work_aabb_min, c_work_aabb_max,
                           t_frame_interp, t_step_mir, t_step_rir):
    team = c_team[ci]
    t = t_frame_interp[team]
    posx = dmath.lerp(c_old_frame_pos[ci, 0], c_frame_pos[ci, 0], t)
    posy = dmath.lerp(c_old_frame_pos[ci, 1], c_frame_pos[ci, 1], t)
    posz = dmath.lerp(c_old_frame_pos[ci, 2], c_frame_pos[ci, 2], t)
    rotx, roty, rotz, rotw = dmath.quat_slerp(
        c_old_frame_rot[ci, 0], c_old_frame_rot[ci, 1], c_old_frame_rot[ci, 2], c_old_frame_rot[ci, 3],
        c_frame_rot[ci, 0], c_frame_rot[ci, 1], c_frame_rot[ci, 2], c_frame_rot[ci, 3], t)
    c_now_pos[ci, 0] = posx
    c_now_pos[ci, 1] = posy
    c_now_pos[ci, 2] = posz
    c_now_rot[ci, 0] = rotx
    c_now_rot[ci, 1] = roty
    c_now_rot[ci, 2] = rotz
    c_now_rot[ci, 3] = rotw
    mir = t_step_mir[team]
    rir = t_step_rir[team]
    opx = dmath.lerp(c_old_pos[ci, 0], posx, mir)
    opy = dmath.lerp(c_old_pos[ci, 1], posy, mir)
    opz = dmath.lerp(c_old_pos[ci, 2], posz, mir)
    orx, ory, orz, orw = dmath.quat_slerp(
        c_old_rot[ci, 0], c_old_rot[ci, 1], c_old_rot[ci, 2], c_old_rot[ci, 3],
        rotx, roty, rotz, rotw, rir)
    c_old_pos[ci, 0] = opx
    c_old_pos[ci, 1] = opy
    c_old_pos[ci, 2] = opz
    c_old_rot[ci, 0] = orx
    c_old_rot[ci, 1] = ory
    c_old_rot[ci, 2] = orz
    c_old_rot[ci, 3] = orw
    c_work_rot[ci, 0] = rotx
    c_work_rot[ci, 1] = roty
    c_work_rot[ci, 2] = rotz
    c_work_rot[ci, 3] = rotw
    iox, ioy, ioz, iow = dmath.quat_inverse(orx, ory, orz, orw)
    c_work_inv_old_rot[ci, 0] = iox
    c_work_inv_old_rot[ci, 1] = ioy
    c_work_inv_old_rot[ci, 2] = ioz
    c_work_inv_old_rot[ci, 3] = iow
    kind = c_kind[ci]
    if kind == COLLIDER_SPHERE:
        radius = c_size[ci, 0] * abs(c_frame_scl[ci, 0])
        c_work_radius[ci, 0] = radius
        c_work_radius[ci, 1] = radius
        c_work_old_pos[ci, 0, 0] = opx
        c_work_old_pos[ci, 0, 1] = opy
        c_work_old_pos[ci, 0, 2] = opz
        c_work_next_pos[ci, 0, 0] = posx
        c_work_next_pos[ci, 0, 1] = posy
        c_work_next_pos[ci, 0, 2] = posz
        c_work_aabb_min[ci, 0] = dmath.fmin2(opx, posx) - radius
        c_work_aabb_min[ci, 1] = dmath.fmin2(opy, posy) - radius
        c_work_aabb_min[ci, 2] = dmath.fmin2(opz, posz) - radius
        c_work_aabb_max[ci, 0] = dmath.fmax2(opx, posx) + radius
        c_work_aabb_max[ci, 1] = dmath.fmax2(opy, posy) + radius
        c_work_aabb_max[ci, 2] = dmath.fmax2(opz, posz) + radius
    elif kind == COLLIDER_CAPSULE:
        axx = c_axis[ci, 0]
        axy = c_axis[ci, 1]
        axz = c_axis[ci, 2]
        scl0 = c_frame_scl[ci, 0] * axx + c_frame_scl[ci, 1] * axy + c_frame_scl[ci, 2] * axz
        sgn = dmath.fsign(scl0)
        dirx = axx * sgn
        diry = axy * sgn
        dirz = axz * sgn
        scl = abs(scl0)
        start_radius = c_size[ci, 0] * scl
        end_radius = c_size[ci, 1] * scl
        cap_length = c_size[ci, 2] * scl
        if c_aligned[ci] != 0:
            start_len = cap_length * float32(0.5)
            end_len = cap_length * float32(0.5)
        else:
            start_len = float32(0.0)
            end_len = cap_length - start_radius
        start_len = start_len - start_radius
        if start_len < float32(0.0):
            start_len = float32(0.0)
        end_len = end_len - end_radius
        if end_len < float32(0.0):
            end_len = float32(0.0)
        dox, doy, doz = dmath.quat_rotate(orx, ory, orz, orw, dirx, diry, dirz)
        dnx, dny, dnz = dmath.quat_rotate(rotx, roty, rotz, rotw, dirx, diry, dirz)
        sox = opx + dox * start_len
        soy = opy + doy * start_len
        soz = opz + doz * start_len
        eox = opx - dox * end_len
        eoy = opy - doy * end_len
        eoz = opz - doz * end_len
        snx = posx + dnx * start_len
        sny = posy + dny * start_len
        snz = posz + dnz * start_len
        enx = posx - dnx * end_len
        eny = posy - dny * end_len
        enz = posz - dnz * end_len
        c_work_radius[ci, 0] = start_radius
        c_work_radius[ci, 1] = end_radius
        c_work_old_pos[ci, 0, 0] = sox
        c_work_old_pos[ci, 0, 1] = soy
        c_work_old_pos[ci, 0, 2] = soz
        c_work_old_pos[ci, 1, 0] = eox
        c_work_old_pos[ci, 1, 1] = eoy
        c_work_old_pos[ci, 1, 2] = eoz
        c_work_next_pos[ci, 0, 0] = snx
        c_work_next_pos[ci, 0, 1] = sny
        c_work_next_pos[ci, 0, 2] = snz
        c_work_next_pos[ci, 1, 0] = enx
        c_work_next_pos[ci, 1, 1] = eny
        c_work_next_pos[ci, 1, 2] = enz
        c_work_aabb_min[ci, 0] = dmath.fmin2(dmath.fmin2(sox, snx) - start_radius, dmath.fmin2(eox, enx) - end_radius)
        c_work_aabb_min[ci, 1] = dmath.fmin2(dmath.fmin2(soy, sny) - start_radius, dmath.fmin2(eoy, eny) - end_radius)
        c_work_aabb_min[ci, 2] = dmath.fmin2(dmath.fmin2(soz, snz) - start_radius, dmath.fmin2(eoz, enz) - end_radius)
        c_work_aabb_max[ci, 0] = dmath.fmax2(dmath.fmax2(sox, snx) + start_radius, dmath.fmax2(eox, enx) + end_radius)
        c_work_aabb_max[ci, 1] = dmath.fmax2(dmath.fmax2(soy, sny) + start_radius, dmath.fmax2(eoy, eny) + end_radius)
        c_work_aabb_max[ci, 2] = dmath.fmax2(dmath.fmax2(soz, snz) + start_radius, dmath.fmax2(eoz, enz) + end_radius)
    else:
        sign_y = dmath.fsign(c_frame_scl[ci, 2] + float32(1e-30))
        nx, ny, nz = dmath.quat_rotate(rotx, roty, rotz, rotw, float32(0.0), float32(0.0), float32(1.0))
        c_work_old_pos[ci, 0, 0] = nx * sign_y
        c_work_old_pos[ci, 0, 1] = ny * sign_y
        c_work_old_pos[ci, 0, 2] = nz * sign_y
        c_work_next_pos[ci, 0, 0] = posx
        c_work_next_pos[ci, 0, 1] = posy
        c_work_next_pos[ci, 0, 2] = posz
        c_work_aabb_min[ci, 0] = -INF
        c_work_aabb_min[ci, 1] = -INF
        c_work_aabb_min[ci, 2] = -INF
        c_work_aabb_max[ci, 0] = INF
        c_work_aabb_max[ci, 1] = INF
        c_work_aabb_max[ci, 2] = INF


@cuda.jit(device=True)
def do_collider_end_step(ci, c_now_pos, c_now_rot, c_old_pos, c_old_rot):
    for j in range(3):
        c_old_pos[ci, j] = c_now_pos[ci, j]
    for j in range(4):
        c_old_rot[ci, j] = c_now_rot[ci, j]


@cuda.jit(device=True)
def do_collider_frame_post(ci, c_frame_pos, c_frame_rot, c_old_frame_pos, c_old_frame_rot):
    for j in range(3):
        c_old_frame_pos[ci, j] = c_frame_pos[ci, j]
    for j in range(4):
        c_old_frame_rot[ci, j] = c_frame_rot[ci, j]


@cuda.jit(device=True)
def do_solve_point(p, p_team, p_next_positions, p_base_positions, p_depth, p_friction,
                   p_collision_normals, p_velocity_positions,
                   t_collision_mode, t_radius_lut, t_scale_ratio, t_is_spring,
                   t_limit_distance_lut,
                   c_kind, c_active, c_work_old_pos, c_work_next_pos, c_work_radius,
                   c_work_inv_old_rot, c_work_rot, c_work_aabb_min, c_work_aabb_max,
                   csr_off, csr_ord, st_pp_collider):
    team = p_team[p]
    if t_collision_mode[team] != COLLISION_POINT:
        return
    start = csr_off[p]
    stop = csr_off[p + 1]
    if start == stop:
        return
    depth = p_depth[p]
    radius = dmath.evaluate_team_lut(t_radius_lut, team, depth)
    if radius < float32(0.0001):
        radius = float32(0.0001)
    radius = radius * t_scale_ratio[team]
    cfr = radius
    npx = p_next_positions[p, 0]
    npy = p_next_positions[p, 1]
    npz = p_next_positions[p, 2]
    is_spring = t_is_spring[team] != 0
    max_length = dmath.evaluate_team_lut(t_limit_distance_lut, team, depth)
    if max_length < float32(0.0001):
        max_length = float32(0.0001)
    max_length = max_length * t_scale_ratio[team]
    bpx = p_base_positions[p, 0]
    bpy = p_base_positions[p, 1]
    bpz = p_base_positions[p, 2]
    box = radius + cfr
    pmnx = npx - box
    pmny = npy - box
    pmnz = npz - box
    pmxx = npx + box
    pmxy = npy + box
    pmxz = npz + box
    count = int32(0)
    near_count = int32(0)
    psumx = float64(0.0)
    psumy = float64(0.0)
    psumz = float64(0.0)
    nsumx = float64(0.0)
    nsumy = float64(0.0)
    nsumz = float64(0.0)
    nnx = float64(0.0)
    nny = float64(0.0)
    nnz = float64(0.0)
    min_dist = INF
    any_active = False
    for k in range(start, stop):
        c = st_pp_collider[csr_ord[k]]
        if c_active[c] == 0:
            continue
        any_active = True
        kind = c_kind[c]
        ux = float32(0.0)
        uy = float32(0.0)
        uz = float32(0.0)
        ox = npx
        oy = npy
        oz = npz
        dist = INF
        if kind == COLLIDER_SPHERE:
            overlap = (pmnx <= c_work_aabb_max[c, 0]) and (pmxx >= c_work_aabb_min[c, 0]) \
                and (pmny <= c_work_aabb_max[c, 1]) and (pmxy >= c_work_aabb_min[c, 1]) \
                and (pmnz <= c_work_aabb_max[c, 2]) and (pmxz >= c_work_aabb_min[c, 2])
            cldx = c_work_old_pos[c, 0, 0]
            cldy = c_work_old_pos[c, 0, 1]
            cldz = c_work_old_pos[c, 0, 2]
            cnwx = c_work_next_pos[c, 0, 0]
            cnwy = c_work_next_pos[c, 0, 1]
            cnwz = c_work_next_pos[c, 0, 2]
            cradius = c_work_radius[c, 0]
            nrx, nry, nrz = dmath.normalize3(npx - cldx, npy - cldy, npz - cldz)
            ppx = cnwx + nrx * (cradius + radius)
            ppy = cnwy + nry * (cradius + radius)
            ppz = cnwz + nrz * (cradius + radius)
            d, outx, outy, outz = dmath.intersect_point_plane_dist(ppx, ppy, ppz, nrx, nry, nrz, npx, npy, npz)
            if is_spring:
                clx, cly, clz = dmath.clamp_distance(bpx, bpy, bpz, outx, outy, outz, max_length)
                lspr = dmath.length3(clx - bpx, cly - bpy, clz - bpz)
                tspr = dmath.saturate(lspr / radius) * float32(0.85)
                outx = dmath.lerp(clx, npx, tspr)
                outy = dmath.lerp(cly, npy, tspr)
                outz = dmath.lerp(clz, npz, tspr)
                d = d * float32(3.0)
            if overlap:
                dist = d
                ox = outx
                oy = outy
                oz = outz
                ux = nrx
                uy = nry
                uz = nrz
        elif kind == COLLIDER_CAPSULE:
            overlap = (pmnx <= c_work_aabb_max[c, 0]) and (pmxx >= c_work_aabb_min[c, 0]) \
                and (pmny <= c_work_aabb_max[c, 1]) and (pmxy >= c_work_aabb_min[c, 1]) \
                and (pmnz <= c_work_aabb_max[c, 2]) and (pmxz >= c_work_aabb_min[c, 2])
            sox = c_work_old_pos[c, 0, 0]
            soy = c_work_old_pos[c, 0, 1]
            soz = c_work_old_pos[c, 0, 2]
            eox = c_work_old_pos[c, 1, 0]
            eoy = c_work_old_pos[c, 1, 1]
            eoz = c_work_old_pos[c, 1, 2]
            snx = c_work_next_pos[c, 0, 0]
            sny = c_work_next_pos[c, 0, 1]
            snz = c_work_next_pos[c, 0, 2]
            enx = c_work_next_pos[c, 1, 0]
            eny = c_work_next_pos[c, 1, 1]
            enz = c_work_next_pos[c, 1, 2]
            sr = c_work_radius[c, 0]
            er = c_work_radius[c, 1]
            ts = dmath.closest_pt_point_segment_ratio(npx, npy, npz, sox, soy, soz, eox, eoy, eoz)
            r = sr + (er - sr) * ts
            d0x = sox + (eox - sox) * ts
            d0y = soy + (eoy - soy) * ts
            d0z = soz + (eoz - soz) * ts
            lvx, lvy, lvz = dmath.quat_rotate(c_work_inv_old_rot[c, 0], c_work_inv_old_rot[c, 1],
                                              c_work_inv_old_rot[c, 2], c_work_inv_old_rot[c, 3],
                                              npx - d0x, npy - d0y, npz - d0z)
            d1x = snx + (enx - snx) * ts
            d1y = sny + (eny - sny) * ts
            d1z = snz + (enz - snz) * ts
            v2x, v2y, v2z = dmath.quat_rotate(c_work_rot[c, 0], c_work_rot[c, 1],
                                              c_work_rot[c, 2], c_work_rot[c, 3], lvx, lvy, lvz)
            nrx, nry, nrz = dmath.normalize3(v2x, v2y, v2z)
            ppx = d1x + nrx * (r + radius)
            ppy = d1y + nry * (r + radius)
            ppz = d1z + nrz * (r + radius)
            d, outx, outy, outz = dmath.intersect_point_plane_dist(ppx, ppy, ppz, nrx, nry, nrz, npx, npy, npz)
            if overlap:
                dist = d
                ox = outx
                oy = outy
                oz = outz
                ux = nrx
                uy = nry
                uz = nrz
        else:
            nrx = c_work_old_pos[c, 0, 0]
            nry = c_work_old_pos[c, 0, 1]
            nrz = c_work_old_pos[c, 0, 2]
            cpx = c_work_next_pos[c, 0, 0]
            cpy = c_work_next_pos[c, 0, 1]
            cpz = c_work_next_pos[c, 0, 2]
            ppx = cpx + nrx * radius
            ppy = cpy + nry * radius
            ppz = cpz + nrz * radius
            d, outx, outy, outz = dmath.intersect_point_plane_dist(ppx, ppy, ppz, nrx, nry, nrz, npx, npy, npz)
            dist = d
            ox = outx
            oy = outy
            oz = outz
            ux = nrx
            uy = nry
            uz = nrz
        if dist <= float32(0.0):
            count += int32(1)
            psumx += float64(ox - npx)
            psumy += float64(oy - npy)
            psumz += float64(oz - npz)
            nsumx += float64(ux)
            nsumy += float64(uy)
            nsumz += float64(uz)
        if dist <= cfr:
            near_count += int32(1)
            nnx += float64(ux)
            nny += float64(uy)
            nnz += float64(uz)
            if dist < min_dist:
                min_dist = dist
    if not any_active:
        return
    has_push = count > int32(0)
    sc = float32(count) if count > int32(0) else float32(1.0)
    navx = float32(nsumx) / sc
    navy = float32(nsumy) / sc
    navz = float32(nsumz) / sc
    normal_length = dmath.length3(navx, navy, navz)
    pavx = float32(psumx) / sc
    pavy = float32(psumy) / sc
    pavz = float32(psumz) / sc
    tclamp = normal_length if normal_length < float32(1.0) else float32(1.0)
    if has_push and (normal_length >= EPSILON):
        p_next_positions[p, 0] = npx + pavx * tclamp
        p_next_positions[p, 1] = npy + pavy * tclamp
        p_next_positions[p, 2] = npz + pavz * tclamp
    nsx = float32(nnx)
    nsy = float32(nny)
    nsz = float32(nnz)
    near_len = dmath.length3(nsx, nsy, nsz)
    has_near = (near_count > int32(0)) and (cfr > float32(0.0)) and (near_len * near_len > float32(1e-6))
    md = min_dist if (min_dist < INF) else float32(0.0)
    denom_cfr = cfr if cfr > float32(0.0) else float32(1.0)
    friction_val = float32(1.0) - dmath.saturate(md / denom_cfr)
    if has_near and (friction_val > p_friction[p]):
        p_friction[p] = friction_val
    if has_near:
        onx, ony, onz = dmath.normalize3(nsx, nsy, nsz)
        p_collision_normals[p, 0] = onx
        p_collision_normals[p, 1] = ony
        p_collision_normals[p, 2] = onz
    else:
        p_collision_normals[p, 0] = nsx
        p_collision_normals[p, 1] = nsy
        p_collision_normals[p, 2] = nsz
    if is_spring and has_push:
        p_velocity_positions[p, 0] += pavx
        p_velocity_positions[p, 1] += pavy
        p_velocity_positions[p, 2] += pavz


@cuda.jit(device=True)
def _edge_sphere(p0x, p0y, p0z, p1x, p1y, p1z, r0, r1, cfr, overlap,
                 cldx, cldy, cldz, cnwx, cnwy, cnwz, cradius):
    s = dmath.closest_pt_point_segment_ratio(cldx, cldy, cldz, p0x, p0y, p0z, p1x, p1y, p1z)
    clx = p0x + (p1x - p0x) * s
    cly = p0y + (p1y - p0y) * s
    clz = p0z + (p1z - p0z) * s
    vx = clx - cldx
    vy = cly - cldy
    vz = clz - cldz
    clen = dmath.length3(vx, vy, vz)
    degenerate = clen < float32(1e-9)
    safe = clen if clen > float32(1e-30) else float32(1.0)
    nx = vx / safe
    ny = vy / safe
    nz = vz / safe
    l1 = nx * (cnwx - cldx) + ny * (cnwy - cldy) + nz * (cnwz - cldz)
    l = clen - l1
    r_edge = r0 + (r1 - r0) * s
    thickness = r_edge + cradius
    miss = l > (thickness + cfr)
    l2 = nx * (clx - cnwx) + ny * (cly - cnwy) + nz * (clz - cnwz)
    no_contact = l2 > thickness
    near_dist = l2 - thickness
    cval = thickness - l2
    b0 = float32(1.0) - s
    b1 = s
    denom = b0 * b0 + b1 * b1
    scale = cval / (denom if denom > float32(0.0) else float32(1.0))
    c0x = nx * (b0 * scale)
    c0y = ny * (b0 * scale)
    c0z = nz * (b0 * scale)
    c1x = nx * (b1 * scale)
    c1y = ny * (b1 * scale)
    c1z = nz * (b1 * scale)
    dist = near_dist if no_contact else -cval
    valid = overlap and (not degenerate) and (not miss)
    if not valid:
        dist = INF
    if (not valid) or no_contact:
        c0x = float32(0.0)
        c0y = float32(0.0)
        c0z = float32(0.0)
        c1x = float32(0.0)
        c1y = float32(0.0)
        c1z = float32(0.0)
    if not valid:
        nx = float32(0.0)
        ny = float32(0.0)
        nz = float32(0.0)
    return (dist, c0x, c0y, c0z, c1x, c1y, c1z, nx, ny, nz)


@cuda.jit(device=True)
def _edge_capsule(p0x, p0y, p0z, p1x, p1y, p1z, r0, r1, cfr, overlap,
                  sox, soy, soz, eox, eoy, eoz, snx, sny, snz, enx, eny, enz, sr, er):
    s, t, cax, cay, caz, cbx, cby, cbz = dmath.closest_pt_segment_segment(
        p0x, p0y, p0z, p1x, p1y, p1z, sox, soy, soz, eox, eoy, eoz)
    vx = cax - cbx
    vy = cay - cby
    vz = caz - cbz
    clen = dmath.length3(vx, vy, vz)
    degenerate = clen < float32(1e-9)
    safe = clen if clen > float32(1e-30) else float32(1.0)
    nx = vx / safe
    ny = vy / safe
    nz = vz / safe
    if sr != er:
        so2x = sox + nx * sr
        so2y = soy + ny * sr
        so2z = soz + nz * sr
        eo2x = eox + nx * er
        eo2y = eoy + ny * er
        eo2z = eoz + nz * er
        s2, t2, _a, _b, _cc, _d, _e, _f = dmath.closest_pt_segment_segment(
            p0x, p0y, p0z, p1x, p1y, p1z, so2x, so2y, so2z, eo2x, eo2y, eo2z)
        s = s2
        t = t2
        cax = p0x + (p1x - p0x) * s
        cay = p0y + (p1y - p0y) * s
        caz = p0z + (p1z - p0z) * s
        cbx = sox + (eox - sox) * t
        cby = soy + (eoy - soy) * t
        cbz = soz + (eoz - soz) * t
        vx = cax - cbx
        vy = cay - cby
        vz = caz - cbz
        clen2 = dmath.length3(vx, vy, vz)
        safe2 = clen2 if clen2 > float32(1e-30) else float32(1.0)
        nx = vx / safe2
        ny = vy / safe2
        nz = vz / safe2
        clen = clen2
        if clen2 < float32(1e-9):
            degenerate = True
    dbx = (snx - sox) + ((enx - eox) - (snx - sox)) * t
    dby = (sny - soy) + ((eny - eoy) - (sny - soy)) * t
    dbz = (snz - soz) + ((enz - eoz) - (snz - soz)) * t
    l1 = nx * dbx + ny * dby + nz * dbz
    l = clen - l1
    r_edge = r0 + (r1 - r0) * s
    r_capsule = sr + (er - sr) * t
    thickness = r_edge + r_capsule
    miss = l > (thickness + cfr)
    dx = snx + (enx - snx) * t
    dy = sny + (eny - sny) * t
    dz = snz + (enz - snz) * t
    l2 = nx * (cax - dx) + ny * (cay - dy) + nz * (caz - dz)
    no_contact = l2 > thickness
    near_dist = l2 - thickness
    cval = thickness - l2
    b0 = float32(1.0) - s
    b1 = s
    denom = b0 * b0 + b1 * b1
    scale = cval / (denom if denom > float32(0.0) else float32(1.0))
    c0x = nx * (b0 * scale)
    c0y = ny * (b0 * scale)
    c0z = nz * (b0 * scale)
    c1x = nx * (b1 * scale)
    c1y = ny * (b1 * scale)
    c1z = nz * (b1 * scale)
    dist = near_dist if no_contact else -cval
    valid = overlap and (not degenerate) and (not miss)
    if not valid:
        dist = INF
    if (not valid) or no_contact:
        c0x = float32(0.0)
        c0y = float32(0.0)
        c0z = float32(0.0)
        c1x = float32(0.0)
        c1y = float32(0.0)
        c1z = float32(0.0)
    if not valid:
        nx = float32(0.0)
        ny = float32(0.0)
        nz = float32(0.0)
    return (dist, c0x, c0y, c0z, c1x, c1y, c1z, nx, ny, nz)


@cuda.jit(device=True)
def do_solve_edge(ee, p_team, p_next_positions, p_depth, p_attr_move,
                  t_radius_lut, t_scale_ratio,
                  c_kind, c_active, c_work_old_pos, c_work_next_pos, c_work_radius,
                  c_work_aabb_min, c_work_aabb_max,
                  csr_off, csr_ord, st_ep_collider, st_collision_edge,
                  sc_dcorr_fixed, sc_dcount, sc_col_friction_fixed, sc_col_normal_fixed):
    e0 = st_collision_edge[ee, 0]
    e1 = st_collision_edge[ee, 1]
    team = p_team[e0]
    start = csr_off[ee]
    stop = csr_off[ee + 1]
    if start == stop:
        return
    p0x = p_next_positions[e0, 0]
    p0y = p_next_positions[e0, 1]
    p0z = p_next_positions[e0, 2]
    p1x = p_next_positions[e1, 0]
    p1y = p_next_positions[e1, 1]
    p1z = p_next_positions[e1, 2]
    r0 = dmath.evaluate_team_lut(t_radius_lut, team, p_depth[e0]) * t_scale_ratio[team]
    r1 = dmath.evaluate_team_lut(t_radius_lut, team, p_depth[e1]) * t_scale_ratio[team]
    cfr = (r0 + r1) * float32(0.5)
    emnx = dmath.fmin2(p0x - r0, p1x - r1) - cfr
    emny = dmath.fmin2(p0y - r0, p1y - r1) - cfr
    emnz = dmath.fmin2(p0z - r0, p1z - r1) - cfr
    emxx = dmath.fmax2(p0x + r0, p1x + r1) + cfr
    emxy = dmath.fmax2(p0y + r0, p1y + r1) + cfr
    emxz = dmath.fmax2(p0z + r0, p1z + r1) + cfr
    count = int32(0)
    near_count = int32(0)
    c0sx = float64(0.0)
    c0sy = float64(0.0)
    c0sz = float64(0.0)
    c1sx = float64(0.0)
    c1sy = float64(0.0)
    c1sz = float64(0.0)
    nsumx = float64(0.0)
    nsumy = float64(0.0)
    nsumz = float64(0.0)
    nnx = float64(0.0)
    nny = float64(0.0)
    nnz = float64(0.0)
    min_dist = INF
    any_active = False
    for k in range(start, stop):
        c = st_ep_collider[csr_ord[k]]
        if c_active[c] == 0:
            continue
        any_active = True
        overlap = (emnx <= c_work_aabb_max[c, 0]) and (emxx >= c_work_aabb_min[c, 0]) \
            and (emny <= c_work_aabb_max[c, 1]) and (emxy >= c_work_aabb_min[c, 1]) \
            and (emnz <= c_work_aabb_max[c, 2]) and (emxz >= c_work_aabb_min[c, 2])
        kind = c_kind[c]
        if kind == COLLIDER_SPHERE:
            d, a0x, a0y, a0z, a1x, a1y, a1z, ux, uy, uz = _edge_sphere(
                p0x, p0y, p0z, p1x, p1y, p1z, r0, r1, cfr, overlap,
                c_work_old_pos[c, 0, 0], c_work_old_pos[c, 0, 1], c_work_old_pos[c, 0, 2],
                c_work_next_pos[c, 0, 0], c_work_next_pos[c, 0, 1], c_work_next_pos[c, 0, 2],
                c_work_radius[c, 0])
        elif kind == COLLIDER_CAPSULE:
            d, a0x, a0y, a0z, a1x, a1y, a1z, ux, uy, uz = _edge_capsule(
                p0x, p0y, p0z, p1x, p1y, p1z, r0, r1, cfr, overlap,
                c_work_old_pos[c, 0, 0], c_work_old_pos[c, 0, 1], c_work_old_pos[c, 0, 2],
                c_work_old_pos[c, 1, 0], c_work_old_pos[c, 1, 1], c_work_old_pos[c, 1, 2],
                c_work_next_pos[c, 0, 0], c_work_next_pos[c, 0, 1], c_work_next_pos[c, 0, 2],
                c_work_next_pos[c, 1, 0], c_work_next_pos[c, 1, 1], c_work_next_pos[c, 1, 2],
                c_work_radius[c, 0], c_work_radius[c, 1])
        else:
            ux = c_work_old_pos[c, 0, 0]
            uy = c_work_old_pos[c, 0, 1]
            uz = c_work_old_pos[c, 0, 2]
            cpx = c_work_next_pos[c, 0, 0]
            cpy = c_work_next_pos[c, 0, 1]
            cpz = c_work_next_pos[c, 0, 2]
            d0, o0x, o0y, o0z = dmath.intersect_point_plane_dist(cpx + ux * r0, cpy + uy * r0,
                                                                 cpz + uz * r0, ux, uy, uz, p0x, p0y, p0z)
            d1, o1x, o1y, o1z = dmath.intersect_point_plane_dist(cpx + ux * r1, cpy + uy * r1,
                                                                 cpz + uz * r1, ux, uy, uz, p1x, p1y, p1z)
            d = d0 if d0 < d1 else d1
            a0x = o0x - p0x
            a0y = o0y - p0y
            a0z = o0z - p0z
            a1x = o1x - p1x
            a1y = o1y - p1y
            a1z = o1z - p1z
        if d <= float32(0.0):
            count += int32(1)
            c0sx += float64(a0x)
            c0sy += float64(a0y)
            c0sz += float64(a0z)
            c1sx += float64(a1x)
            c1sy += float64(a1y)
            c1sz += float64(a1z)
            nsumx += float64(ux)
            nsumy += float64(uy)
            nsumz += float64(uz)
        if d <= cfr:
            near_count += int32(1)
            nnx += float64(ux)
            nny += float64(uy)
            nnz += float64(uz)
            if d < min_dist:
                min_dist = d
    if not any_active:
        return
    has_push = count > int32(0)
    sc = float32(count) if count > int32(0) else float32(1.0)
    navx = float32(nsumx) / sc
    navy = float32(nsumy) / sc
    navz = float32(nsumz) / sc
    normal_length = dmath.length3(navx, navy, navz)
    tclamp = normal_length if normal_length < float32(1.0) else float32(1.0)
    valid = has_push and (normal_length > EPSILON)
    scale = (tclamp if valid else float32(0.0)) / sc
    d0x = float32(c0sx) * scale
    d0y = float32(c0sy) * scale
    d0z = float32(c0sz) * scale
    d1x = float32(c1sx) * scale
    d1y = float32(c1sy) * scale
    d1z = float32(c1sz) * scale
    nsx = float32(nnx)
    nsy = float32(nny)
    nsz = float32(nnz)
    near_len = dmath.length3(nsx, nsy, nsz)
    has_near = (near_count > int32(0)) and (cfr > float32(0.0)) and (near_len * near_len > float32(1e-6))
    md = min_dist if (min_dist < INF) else float32(0.0)
    denom_cfr = cfr if cfr > float32(0.0) else float32(1.0)
    friction = float32(1.0) - dmath.saturate(md / denom_cfr)
    if has_near:
        noutx, nouty, noutz = dmath.normalize3(nsx, nsy, nsz)
    else:
        noutx = float32(0.0)
        nouty = float32(0.0)
        noutz = float32(0.0)
    move0 = p_attr_move[e0] != 0
    move1 = p_attr_move[e1] != 0
    mask0 = has_near and move0
    mask1 = has_near and move1
    vint = int32(1) if valid else int32(0)
    cuda.atomic.add(sc_dcorr_fixed, (e0, 0), int32(d0x * TO_FIXED))
    cuda.atomic.add(sc_dcorr_fixed, (e0, 1), int32(d0y * TO_FIXED))
    cuda.atomic.add(sc_dcorr_fixed, (e0, 2), int32(d0z * TO_FIXED))
    cuda.atomic.add(sc_dcount, e0, vint)
    cuda.atomic.add(sc_dcorr_fixed, (e1, 0), int32(d1x * TO_FIXED))
    cuda.atomic.add(sc_dcorr_fixed, (e1, 1), int32(d1y * TO_FIXED))
    cuda.atomic.add(sc_dcorr_fixed, (e1, 2), int32(d1z * TO_FIXED))
    cuda.atomic.add(sc_dcount, e1, vint)
    f0 = friction if mask0 else float32(0.0)
    f1 = friction if mask1 else float32(0.0)
    cuda.atomic.max(sc_col_friction_fixed, e0, int32(f0 * TO_FIXED))
    cuda.atomic.max(sc_col_friction_fixed, e1, int32(f1 * TO_FIXED))
    if mask0:
        cuda.atomic.add(sc_col_normal_fixed, (e0, 0), int32(noutx * TO_FIXED))
        cuda.atomic.add(sc_col_normal_fixed, (e0, 1), int32(nouty * TO_FIXED))
        cuda.atomic.add(sc_col_normal_fixed, (e0, 2), int32(noutz * TO_FIXED))
    if mask1:
        cuda.atomic.add(sc_col_normal_fixed, (e1, 0), int32(noutx * TO_FIXED))
        cuda.atomic.add(sc_col_normal_fixed, (e1, 1), int32(nouty * TO_FIXED))
        cuda.atomic.add(sc_col_normal_fixed, (e1, 2), int32(noutz * TO_FIXED))


@cuda.jit(device=True)
def do_particles_frame_pre(p, p_team, p_positions, p_rotations, p_next_positions,
                           p_old_positions, p_old_rotations, p_base_positions, p_base_rotations,
                           p_old_anim_positions, p_old_anim_rotations, p_velocity_positions,
                           p_display_positions, p_velocities, p_real_velocities,
                           p_friction, p_static_friction, p_collision_normals,
                           t_reset_pending, t_neg_teleport, t_neg_matrix,
                           t_inertia_shift, t_shift_vec, t_shift_rot, t_old_cwp):
    team = p_team[p]
    if t_reset_pending[team] != 0:
        for j in range(3):
            pv = p_positions[p, j]
            p_next_positions[p, j] = pv
            p_old_positions[p, j] = pv
            p_base_positions[p, j] = pv
            p_old_anim_positions[p, j] = pv
            p_velocity_positions[p, j] = pv
            p_display_positions[p, j] = pv
            p_velocities[p, j] = float32(0.0)
            p_real_velocities[p, j] = float32(0.0)
            p_collision_normals[p, j] = float32(0.0)
        for j in range(4):
            rv = p_rotations[p, j]
            p_old_rotations[p, j] = rv
            p_base_rotations[p, j] = rv
            p_old_anim_rotations[p, j] = rv
        p_friction[p] = float32(0.0)
        p_static_friction[p] = float32(0.0)
        return
    neg = t_neg_teleport[team] != 0
    shift = t_inertia_shift[team] != 0
    if not (neg or shift):
        return
    if neg:
        m = t_neg_matrix[team]
        _neg_transform_pose(p_old_positions, p_old_rotations, p, m, float32(1.0), float32(1.0))
        _neg_transform_pose(p_old_anim_positions, p_old_anim_rotations, p, m, float32(1.0), float32(1.0))
        dpx, dpy, dpz = dmath.transform_point(m, p_display_positions[p, 0], p_display_positions[p, 1],
                                              p_display_positions[p, 2])
        p_display_positions[p, 0] = dpx
        p_display_positions[p, 1] = dpy
        p_display_positions[p, 2] = dpz
        vx, vy, vz = dmath.transform_vector(m, p_velocities[p, 0], p_velocities[p, 1], p_velocities[p, 2])
        p_velocities[p, 0] = vx
        p_velocities[p, 1] = vy
        p_velocities[p, 2] = vz
        rvx, rvy, rvz = dmath.transform_vector(m, p_real_velocities[p, 0], p_real_velocities[p, 1],
                                               p_real_velocities[p, 2])
        p_real_velocities[p, 0] = rvx
        p_real_velocities[p, 1] = rvy
        p_real_velocities[p, 2] = rvz
    if shift:
        cpx = t_old_cwp[team, 0]
        cpy = t_old_cwp[team, 1]
        cpz = t_old_cwp[team, 2]
        svx = t_shift_vec[team, 0]
        svy = t_shift_vec[team, 1]
        svz = t_shift_vec[team, 2]
        srx = t_shift_rot[team, 0]
        sry = t_shift_rot[team, 1]
        srz = t_shift_rot[team, 2]
        srw = t_shift_rot[team, 3]
        _shift_point(p_old_positions, p, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_point(p_old_anim_positions, p, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_point(p_display_positions, p, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _premul_quat(p_old_rotations, p, srx, sry, srz, srw)
        _premul_quat(p_old_anim_rotations, p, srx, sry, srz, srw)
        _rotate_vec(p_velocities, p, srx, sry, srz, srw)
        _rotate_vec(p_real_velocities, p, srx, sry, srz, srw)


@cuda.jit(device=True)
def do_angle_limit(v, p, vt, c_inv, p_inv, p_move,
                   p_next_positions, p_velocity_positions, p_albuf_rotation,
                   p_albuf_local_pos, p_albuf_local_rot, p_albuf_length, p_depth,
                   t_angle_limit_lut, t_angle_limit_stiffness):
    # mirrors stages/angle.py limit block for one pass entry (v child, p parent). (level,rank)
    # bucketing makes v and p unique in the pass, so the parent write is race-free (no atomics).
    prx = p_albuf_rotation[p, 0]
    pry = p_albuf_rotation[p, 1]
    prz = p_albuf_rotation[p, 2]
    prw = p_albuf_rotation[p, 3]
    lpx = p_albuf_local_pos[v, 0]
    lpy = p_albuf_local_pos[v, 1]
    lpz = p_albuf_local_pos[v, 2]
    lrx = p_albuf_local_rot[v, 0]
    lry = p_albuf_local_rot[v, 1]
    lrz = p_albuf_local_rot[v, 2]
    lrw = p_albuf_local_rot[v, 3]
    cpx = p_next_positions[v, 0]
    cpy = p_next_positions[v, 1]
    cpz = p_next_positions[v, 2]
    ppx = p_next_positions[p, 0]
    ppy = p_next_positions[p, 1]
    ppz = p_next_positions[p, 2]
    vvx = cpx - ppx
    vvy = cpy - ppy
    vvz = cpz - ppz
    vlen = dmath.length3(vvx, vvy, vvz)
    skip1 = vlen < EPSILON
    tvx, tvy, tvz = dmath.quat_rotate(prx, pry, prz, prw, lpx, lpy, lpz)
    tvlen = dmath.length3(tvx, tvy, tvz)
    snap = (not skip1) and (tvlen < EPSILON)
    if snap:
        p_velocity_positions[v, 0] += ppx - cpx
        p_velocity_positions[v, 1] += ppy - cpy
        p_velocity_positions[v, 2] += ppz - cpz
        p_next_positions[v, 0] = ppx
        p_next_positions[v, 1] = ppy
        p_next_positions[v, 2] = ppz
        sqx, sqy, sqz, sqw = dmath.quat_mul(prx, pry, prz, prw, lrx, lry, lrz, lrw)
        p_albuf_rotation[v, 0] = sqx
        p_albuf_rotation[v, 1] = sqy
        p_albuf_rotation[v, 2] = sqz
        p_albuf_rotation[v, 3] = sqw
        cpx = ppx
        cpy = ppy
        cpz = ppz
    work = (not skip1) and (not snap)
    safe_vlen = vlen if vlen > float32(1e-30) else float32(1.0)
    uvx = vvx / safe_vlen
    uvy = vvy / safe_vlen
    uvz = vvz / safe_vlen
    safe_tvlen = tvlen if tvlen > float32(1e-30) else float32(1.0)
    utvx = tvx / safe_tvlen
    utvy = tvy / safe_tvlen
    utvz = tvz / safe_tvlen
    blen = p_albuf_length[v]
    vlen2 = dmath.lerp(vlen, blen, float32(0.5))
    work = work and (blen >= EPSILON) and (vlen2 >= EPSILON)
    vsx = uvx * vlen2
    vsy = uvy * vlen2
    vsz = uvz * vlen2
    ang = dmath.angle_between(vsx, vsy, vsz, utvx, utvy, utvz)
    max_angle = DEG2RAD * dmath.evaluate_team_lut(t_angle_limit_lut, vt, p_depth[v])
    over = ang > max_angle
    recovery = dmath.lerp(ang, max_angle, t_angle_limit_stiffness[vt])
    clx, cly, clz, _cn = dmath.clamp_angle_vector(vsx, vsy, vsz, utvx, utvy, utvz, recovery)
    if over and work:
        rvx = clx
        rvy = cly
        rvz = clz
    else:
        rvx = vsx
        rvy = vsy
        rvz = vsz
    rpx = ppx + vsx * ANGLE_LIMIT_ROT_RATIO
    rpy = ppy + vsy * ANGLE_LIMIT_ROT_RATIO
    rpz = ppz + vsz * ANGLE_LIMIT_ROT_RATIO
    pfx = rpx - rvx * ANGLE_LIMIT_ROT_RATIO
    pfy = rpy - rvy * ANGLE_LIMIT_ROT_RATIO
    pfz = rpz - rvz * ANGLE_LIMIT_ROT_RATIO
    cfx = rpx + rvx * (float32(1.0) - ANGLE_LIMIT_ROT_RATIO)
    cfy = rpy + rvy * (float32(1.0) - ANGLE_LIMIT_ROT_RATIO)
    cfz = rpz + rvz * (float32(1.0) - ANGLE_LIMIT_ROT_RATIO)
    if work:
        paddx = (pfx - ppx) * p_inv
        paddy = (pfy - ppy) * p_inv
        paddz = (pfz - ppz) * p_inv
        caddx = (cfx - cpx) * c_inv
        caddy = (cfy - cpy) * c_inv
        caddz = (cfz - cpz) * c_inv
    else:
        paddx = float32(0.0)
        paddy = float32(0.0)
        paddz = float32(0.0)
        caddx = float32(0.0)
        caddy = float32(0.0)
        caddz = float32(0.0)
    cpx = cpx + caddx
    cpy = cpy + caddy
    cpz = cpz + caddz
    p_next_positions[v, 0] = cpx
    p_next_positions[v, 1] = cpy
    p_next_positions[v, 2] = cpz
    p_velocity_positions[v, 0] += caddx * ANGLE_LIMIT_ATTENUATION
    p_velocity_positions[v, 1] += caddy * ANGLE_LIMIT_ATTENUATION
    p_velocity_positions[v, 2] += caddz * ANGLE_LIMIT_ATTENUATION
    if work and p_move:
        ppx = ppx + paddx
        ppy = ppy + paddy
        ppz = ppz + paddz
        p_next_positions[p, 0] = ppx
        p_next_positions[p, 1] = ppy
        p_next_positions[p, 2] = ppz
        p_velocity_positions[p, 0] += paddx * ANGLE_LIMIT_ATTENUATION
        p_velocity_positions[p, 1] += paddy * ANGLE_LIMIT_ATTENUATION
        p_velocity_positions[p, 2] += paddz * ANGLE_LIMIT_ATTENUATION
    v3x = cpx - ppx
    v3y = cpy - ppy
    v3z = cpz - ppz
    vlen3 = dmath.length3(v3x, v3y, v3z)
    fix_ok = work and (vlen3 >= EPSILON)
    safe_v3 = vlen3 if vlen3 > float32(1e-30) else float32(1.0)
    uv3x = v3x / safe_v3
    uv3y = v3y / safe_v3
    uv3z = v3z / safe_v3
    nrx, nry, nrz, nrw = dmath.quat_mul(prx, pry, prz, prw, lrx, lry, lrz, lrw)
    qx, qy, qz, qw = dmath.from_to_rotation(utvx, utvy, utvz, uv3x, uv3y, uv3z, float32(1.0), True)
    frx, fry, frz, frw = dmath.quat_mul(qx, qy, qz, qw, nrx, nry, nrz, nrw)
    if fix_ok:
        p_albuf_rotation[v, 0] = frx
        p_albuf_rotation[v, 1] = fry
        p_albuf_rotation[v, 2] = frz
        p_albuf_rotation[v, 3] = frw


@cuda.jit(device=True)
def do_angle_restoration(v, p, vt, c_inv, p_inv, p_move, rot_ratio, power3,
                         p_next_positions, p_velocity_positions, p_albuf_restore, p_depth,
                         t_angle_restoration_lut, t_angle_restoration_attenuation,
                         t_angle_restoration_gravity_falloff, t_gravity_dot):
    # mirrors stages/angle.py restoration block for one pass entry (runs after the limit block
    # on the same thread; positions refetched so limit's writes feed restoration).
    stiff = dmath.evaluate_team_lut_clamp01(t_angle_restoration_lut, vt, p_depth[v])
    stiff = dmath.saturate(stiff * power3)
    gfo = dmath.lerp(float32(1.0) - t_angle_restoration_gravity_falloff[vt],
                     float32(1.0), t_gravity_dot[vt])
    stiff = stiff * gfo
    r_attn = t_angle_restoration_attenuation[vt]
    cpx = p_next_positions[v, 0]
    cpy = p_next_positions[v, 1]
    cpz = p_next_positions[v, 2]
    ppx = p_next_positions[p, 0]
    ppy = p_next_positions[p, 1]
    ppz = p_next_positions[p, 2]
    tvx = p_albuf_restore[v, 0]
    tvy = p_albuf_restore[v, 1]
    tvz = p_albuf_restore[v, 2]
    tvlen = dmath.length3(tvx, tvy, tvz)
    snap = tvlen < EPSILON
    if snap:
        p_velocity_positions[v, 0] += ppx - cpx
        p_velocity_positions[v, 1] += ppy - cpy
        p_velocity_positions[v, 2] += ppz - cpz
        p_next_positions[v, 0] = ppx
        p_next_positions[v, 1] = ppy
        p_next_positions[v, 2] = ppz
        cpx = ppx
        cpy = ppy
        cpz = ppz
    vvx = cpx - ppx
    vvy = cpy - ppy
    vvz = cpz - ppz
    vlen = dmath.length3(vvx, vvy, vvz)
    work = (not snap) and (vlen >= EPSILON)
    safe_vlen = vlen if vlen > float32(1e-30) else float32(1.0)
    uvx = vvx / safe_vlen
    uvy = vvy / safe_vlen
    uvz = vvz / safe_vlen
    safe_tvlen = tvlen if tvlen > float32(1e-30) else float32(1.0)
    utvx = tvx / safe_tvlen
    utvy = tvy / safe_tvlen
    utvz = tvz / safe_tvlen
    rqx, rqy, rqz, rqw = dmath.from_to_rotation(uvx, uvy, uvz, utvx, utvy, utvz, stiff, True)
    rvx, rvy, rvz = dmath.quat_rotate(rqx, rqy, rqz, rqw, vvx, vvy, vvz)
    rpx = ppx + vvx * rot_ratio
    rpy = ppy + vvy * rot_ratio
    rpz = ppz + vvz * rot_ratio
    pfx = rpx - rvx * rot_ratio
    pfy = rpy - rvy * rot_ratio
    pfz = rpz - rvz * rot_ratio
    cfx = rpx + rvx * (float32(1.0) - rot_ratio)
    cfy = rpy + rvy * (float32(1.0) - rot_ratio)
    cfz = rpz + rvz * (float32(1.0) - rot_ratio)
    if work:
        paddx = (pfx - ppx) * p_inv
        paddy = (pfy - ppy) * p_inv
        paddz = (pfz - ppz) * p_inv
        caddx = (cfx - cpx) * c_inv
        caddy = (cfy - cpy) * c_inv
        caddz = (cfz - cpz) * c_inv
    else:
        paddx = float32(0.0)
        paddy = float32(0.0)
        paddz = float32(0.0)
        caddx = float32(0.0)
        caddy = float32(0.0)
        caddz = float32(0.0)
    p_next_positions[v, 0] = cpx + caddx
    p_next_positions[v, 1] = cpy + caddy
    p_next_positions[v, 2] = cpz + caddz
    p_velocity_positions[v, 0] += caddx * r_attn
    p_velocity_positions[v, 1] += caddy * r_attn
    p_velocity_positions[v, 2] += caddz * r_attn
    if work and p_move:
        p_next_positions[p, 0] = ppx + paddx
        p_next_positions[p, 1] = ppy + paddy
        p_next_positions[p, 2] = ppz + paddz
        p_velocity_positions[p, 0] += paddx * r_attn
        p_velocity_positions[p, 1] += paddy * r_attn
        p_velocity_positions[p, 2] += paddz * r_attn


@cuda.jit(device=True)
def do_display_particle(p, mt, sim_dt,
                        p_positions, p_rotations, p_old_positions, p_real_velocities,
                        p_display_positions, p_vertex_root, p_old_anim_positions,
                        p_old_anim_rotations, p_temp_base_positions, p_temp_base_rotations,
                        st_update_move_mask,
                        t_now_update, t_old_time, t_time, t_blend_weight, t_running,
                        t_is_negative_scale, t_negative_scale_direction):
    # mirrors stages/display._display for one particle. Register snapshot is taken BEFORE any
    # write so the running-team old_anim capture sees the pre-move / pre-flip pose. move / fixed
    # are the two halves of the update_move|update_fixed partition (per-particle mask), 4/5/6 hit
    # all frame particles. No cross-thread hazard: roots are always non-move (never written here),
    # each thread owns positions[p]/rotations[p]/temp_base[p]/display[p]/old_anim[p].
    snap_px = p_positions[p, 0]
    snap_py = p_positions[p, 1]
    snap_pz = p_positions[p, 2]
    snap_rx = p_rotations[p, 0]
    snap_ry = p_rotations[p, 1]
    snap_rz = p_rotations[p, 2]
    snap_rw = p_rotations[p, 3]
    fx = snap_px
    fy = snap_py
    fz = snap_pz
    if st_update_move_mask[p] != 0:
        sdt = sim_dt
        fposx = p_old_positions[p, 0] + p_real_velocities[p, 0] * sdt
        fposy = p_old_positions[p, 1] + p_real_velocities[p, 1] * sdt
        fposz = p_old_positions[p, 2] + p_real_velocities[p, 2] * sdt
        interval = (t_now_update[mt] + sdt) - t_old_time[mt]
        if interval > float32(0.0):
            tval = (t_time[mt] - t_old_time[mt]) / interval
        else:
            tval = float32(0.0)
        fposx = dmath.lerp(p_display_positions[p, 0], fposx, tval)
        fposy = dmath.lerp(p_display_positions[p, 1], fposy, tval)
        fposz = dmath.lerp(p_display_positions[p, 2], fposz, tval)
        root = p_vertex_root[p]
        if root >= 0:
            rpx = p_positions[root, 0]
            rpy = p_positions[root, 1]
            rpz = p_positions[root, 2]
            original_dist = dmath.length3(rpx - snap_px, rpy - snap_py, rpz - snap_pz)
            clamp_dist = original_dist * MAX_DISTANCE_RATIO_FUTURE_PREDICTION
            vx, vy, vz = dmath.clamp_vector(fposx - rpx, fposy - rpy, fposz - rpz, clamp_dist)
            fposx = rpx + vx
            fposy = rpy + vy
            fposz = rpz + vz
        p_display_positions[p, 0] = fposx
        p_display_positions[p, 1] = fposy
        p_display_positions[p, 2] = fposz
        blend = t_blend_weight[mt]
        fx = dmath.lerp(snap_px, fposx, blend)
        fy = dmath.lerp(snap_py, fposy, blend)
        fz = dmath.lerp(snap_pz, fposz, blend)
        p_positions[p, 0] = fx
        p_positions[p, 1] = fy
        p_positions[p, 2] = fz
    else:
        p_display_positions[p, 0] = snap_px
        p_display_positions[p, 1] = snap_py
        p_display_positions[p, 2] = snap_pz
    if t_running[mt] != 0:
        p_old_anim_positions[p, 0] = snap_px
        p_old_anim_positions[p, 1] = snap_py
        p_old_anim_positions[p, 2] = snap_pz
        p_old_anim_rotations[p, 0] = snap_rx
        p_old_anim_rotations[p, 1] = snap_ry
        p_old_anim_rotations[p, 2] = snap_rz
        p_old_anim_rotations[p, 3] = snap_rw
    qx = snap_rx
    qy = snap_ry
    qz = snap_rz
    qw = snap_rw
    if t_is_negative_scale[mt] != 0:
        ndy = t_negative_scale_direction[mt, 1]
        ndz = t_negative_scale_direction[mt, 2]
        nnx, nny, nnz = dmath.quat_to_normal(snap_rx, snap_ry, snap_rz, snap_rw)
        nnx = nnx * ndy
        nny = nny * ndy
        nnz = nnz * ndy
        ttx, tty, ttz = dmath.quat_to_tangent(snap_rx, snap_ry, snap_rz, snap_rw)
        ttx = ttx * ndz
        tty = tty * ndz
        ttz = ttz * ndz
        qx, qy, qz, qw = dmath.look_rotation(ttx, tty, ttz, nnx, nny, nnz)
        p_rotations[p, 0] = qx
        p_rotations[p, 1] = qy
        p_rotations[p, 2] = qz
        p_rotations[p, 3] = qw
    p_temp_base_positions[p, 0] = fx
    p_temp_base_positions[p, 1] = fy
    p_temp_base_positions[p, 2] = fz
    p_temp_base_rotations[p, 0] = qx
    p_temp_base_rotations[p, 1] = qy
    p_temp_base_rotations[p, 2] = qz
    p_temp_base_rotations[p, 3] = qw


@cuda.jit(device=True)
def do_postline_entry(entry, et, ch_start, ch_end, postline_child_vertices,
                      p_positions, p_rotations, p_temp_base_positions, p_temp_base_rotations,
                      p_vertex_local_positions, p_vertex_local_rotations,
                      p_attr_invalid, p_attr_zero_distance, p_attr_move, p_team,
                      t_rotational_interpolation, t_root_rotation, t_blend_weight,
                      t_animation_pose_ratio, t_negative_scale_direction,
                      t_negative_scale_quaternion):
    # mirrors stages/display._postline for one entry (owner). (level,rank) topology makes each
    # entry's children a contiguous CSR run and disjoint across entries in a level, so the child
    # rotation writes are race-free; children (level L+1) become entries at level L+1 with a
    # grid.sync between levels making this level's writes visible. ctv_sum/cv_sum are f64
    # (mirrors oracle np.add.reduceat over np.float64 rows), cast to f32 for the zero test / cq.
    rx = p_rotations[entry, 0]
    ry = p_rotations[entry, 1]
    rz = p_rotations[entry, 2]
    rw = p_rotations[entry, 3]
    posx = p_positions[entry, 0]
    posy = p_positions[entry, 1]
    posz = p_positions[entry, 2]
    bpx = p_temp_base_positions[entry, 0]
    bpy = p_temp_base_positions[entry, 1]
    bpz = p_temp_base_positions[entry, 2]
    brx = p_temp_base_rotations[entry, 0]
    bry = p_temp_base_rotations[entry, 1]
    brz = p_temp_base_rotations[entry, 2]
    brw = p_temp_base_rotations[entry, 3]
    bix, biy, biz, biw = dmath.quat_inverse(brx, bry, brz, brw)
    owner_valid = p_attr_invalid[entry] == 0
    ctvx = float64(0.0)
    ctvy = float64(0.0)
    ctvz = float64(0.0)
    cvx = float64(0.0)
    cvy = float64(0.0)
    cvz = float64(0.0)
    has_children = ch_end > ch_start
    for k in range(ch_start, ch_end):
        c = postline_child_vertices[k]
        ct = p_team[c]
        anime_ratio = t_animation_pose_ratio[ct]
        ndx = t_negative_scale_direction[ct, 0]
        ndy = t_negative_scale_direction[ct, 1]
        ndz = t_negative_scale_direction[ct, 2]
        nqx = t_negative_scale_quaternion[ct, 0]
        nqy = t_negative_scale_quaternion[ct, 1]
        nqz = t_negative_scale_quaternion[ct, 2]
        nqw = t_negative_scale_quaternion[ct, 3]
        clpx, clpy, clpz = dmath.quat_rotate(
            bix, biy, biz, biw, p_temp_base_positions[c, 0] - bpx,
            p_temp_base_positions[c, 1] - bpy, p_temp_base_positions[c, 2] - bpz)
        clrx, clry, clrz, clrw = dmath.quat_mul(
            bix, biy, biz, biw, p_temp_base_rotations[c, 0], p_temp_base_rotations[c, 1],
            p_temp_base_rotations[c, 2], p_temp_base_rotations[c, 3])
        lposx = dmath.lerp(p_vertex_local_positions[c, 0] * ndx, clpx, anime_ratio)
        lposy = dmath.lerp(p_vertex_local_positions[c, 1] * ndy, clpy, anime_ratio)
        lposz = dmath.lerp(p_vertex_local_positions[c, 2] * ndz, clpz, anime_ratio)
        lrx, lry, lrz, lrw = dmath.quat_slerp(
            p_vertex_local_rotations[c, 0] * nqx, p_vertex_local_rotations[c, 1] * nqy,
            p_vertex_local_rotations[c, 2] * nqz, p_vertex_local_rotations[c, 3] * nqw,
            clrx, clry, clrz, clrw, anime_ratio)
        is_c0 = p_attr_zero_distance[c] != 0
        if is_c0:
            tvx = float32(0.0)
            tvy = float32(0.0)
            tvz = float32(0.0)
        else:
            tvx, tvy, tvz = dmath.quat_rotate(rx, ry, rz, rw, lposx, lposy, lposz)
        c_move = p_attr_move[c] != 0
        vx = p_positions[c, 0] - posx
        vy = p_positions[c, 1] - posy
        vz = p_positions[c, 2] - posz
        if c_move:
            contx = vx
            conty = vy
            contz = vz
        else:
            contx = tvx
            conty = tvy
            contz = tvz
        if owner_valid:
            ctvx += float64(tvx)
            ctvy += float64(tvy)
            ctvz += float64(tvz)
            cvx += float64(contx)
            cvy += float64(conty)
            cvz += float64(contz)
            if c_move:
                crx, cry, crz, crw = dmath.quat_mul(rx, ry, rz, rw, lrx, lry, lrz, lrw)
                if not is_c0:
                    qfx, qfy, qfz, qfw = dmath.from_to_rotation(tvx, tvy, tvz, vx, vy, vz,
                                                                float32(1.0), False)
                    crx, cry, crz, crw = dmath.quat_mul(qfx, qfy, qfz, qfw, crx, cry, crz, crw)
                p_rotations[c, 0] = crx
                p_rotations[c, 1] = cry
                p_rotations[c, 2] = crz
                p_rotations[c, 3] = crw
    if has_children and owner_valid:
        ctv32x = float32(ctvx)
        ctv32y = float32(ctvy)
        ctv32z = float32(ctvz)
        cv32x = float32(cvx)
        cv32y = float32(cvy)
        cv32z = float32(cvz)
        zero = (dmath.length3(ctv32x, ctv32y, ctv32z) < float32(1e-8)) \
            or (dmath.length3(cv32x, cv32y, cv32z) < float32(1e-8))
        if not zero:
            if p_attr_move[entry] != 0:
                t_ratio = t_rotational_interpolation[et]
            else:
                t_ratio = t_root_rotation[et]
            cqx, cqy, cqz, cqw = dmath.from_to_rotation(ctv32x, ctv32y, ctv32z,
                                                        cv32x, cv32y, cv32z, t_ratio, False)
            rx, ry, rz, rw = dmath.quat_mul(cqx, cqy, cqz, cqw, rx, ry, rz, rw)
    rx, ry, rz, rw = dmath.quat_slerp(brx, bry, brz, brw, rx, ry, rz, rw, t_blend_weight[et])
    p_rotations[entry, 0] = rx
    p_rotations[entry, 1] = ry
    p_rotations[entry, 2] = rz
    p_rotations[entry, 3] = rw


@cuda.jit(device=True)
def do_triangle_normal_tangent(tri, tt_team, st_triangle_particles, p_positions, p_uv,
                               t_negative_scale_triangle_sign, tri_normal_f64, tri_tangent_f64):
    # mirrors stages/display._post_triangles per-triangle (dtype asymmetry preserved): the NORMAL
    # is an f32 cross+normalize cast to f64 then * sign[0]; the TANGENT is computed wholly in f64
    # (positions + uv promoted first) via _triangle_tangent_runtime then * sign[1]. Stored by global
    # triangle index so the owner gather (C2) needs no searchsorted.
    i0 = st_triangle_particles[tri, 0]
    i1 = st_triangle_particles[tri, 1]
    i2 = st_triangle_particles[tri, 2]
    p0x = p_positions[i0, 0]
    p0y = p_positions[i0, 1]
    p0z = p_positions[i0, 2]
    p1x = p_positions[i1, 0]
    p1y = p_positions[i1, 1]
    p1z = p_positions[i1, 2]
    p2x = p_positions[i2, 0]
    p2y = p_positions[i2, 1]
    p2z = p_positions[i2, 2]
    cx, cy, cz = dmath.cross3(p1x - p0x, p1y - p0y, p1z - p0z,
                              p2x - p0x, p2y - p0y, p2z - p0z)
    lc = dmath.length3(cx, cy, cz)
    if lc > EPSILON:
        nnx = cx / lc
        nny = cy / lc
        nnz = cz / lc
    else:
        nnx = cx
        nny = cy
        nnz = cz
    ts0 = t_negative_scale_triangle_sign[tt_team, 0]
    tri_normal_f64[tri, 0] = float64(nnx) * float64(ts0)
    tri_normal_f64[tri, 1] = float64(nny) * float64(ts0)
    tri_normal_f64[tri, 2] = float64(nnz) * float64(ts0)
    q0x = float64(p0x)
    q0y = float64(p0y)
    q0z = float64(p0z)
    dbax = float64(p1x) - q0x
    dbay = float64(p1y) - q0y
    dbaz = float64(p1z) - q0z
    dcax = float64(p2x) - q0x
    dcay = float64(p2y) - q0y
    dcaz = float64(p2z) - q0z
    uv0x = float64(p_uv[i0, 0])
    uv0y = float64(p_uv[i0, 1])
    tbax = float64(p_uv[i1, 0]) - uv0x
    tbay = float64(p_uv[i1, 1]) - uv0y
    tcax = float64(p_uv[i2, 0]) - uv0x
    tcay = float64(p_uv[i2, 1]) - uv0y
    area = tbax * tcay - tbay * tcax
    if area == float64(0.0):
        area = float64(1.0)
    delta = float64(-1.0) / area
    tanx = (dbax * tcay + dcax * (-tbay)) * delta
    tany = (dbay * tcay + dcay * (-tbay)) * delta
    tanz = (dbaz * tcay + dcaz * (-tbay)) * delta
    ltan = math.sqrt(tanx * tanx + tany * tany + tanz * tanz)
    if ltan > float64(1e-30):
        tanx = tanx / ltan
        tany = tany / ltan
        tanz = tanz / ltan
    ts1 = t_negative_scale_triangle_sign[tt_team, 1]
    tri_tangent_f64[tri, 0] = tanx * float64(ts1)
    tri_tangent_f64[tri, 1] = tany * float64(ts1)
    tri_tangent_f64[tri, 2] = tanz * float64(ts1)


@cuda.jit(device=True)
def do_v2t_owner(p, mt, seg0, seg1, csr_v2t_order, st_v2t_triangle,
                 st_v2t_flip_normal, st_v2t_flip_tangent, tri_normal_f64, tri_tangent_f64,
                 p_rotations, p_normal_adjustment_rotations, t_negative_scale_quaternion):
    # mirrors stages/display._post_triangles owner reduce (f64 gather over the owner's v2t rows via
    # the owner-keyed CSR), the ok gate, and rotation = look_rotation(binormal, nor) * adjust.
    norx = float64(0.0)
    nory = float64(0.0)
    norz = float64(0.0)
    tanx = float64(0.0)
    tany = float64(0.0)
    tanz = float64(0.0)
    for k in range(seg0, seg1):
        row = csr_v2t_order[k]
        tri = st_v2t_triangle[row]
        fn = float64(st_v2t_flip_normal[row])
        ft = float64(st_v2t_flip_tangent[row])
        norx += tri_normal_f64[tri, 0] * fn
        nory += tri_normal_f64[tri, 1] * fn
        norz += tri_normal_f64[tri, 2] * fn
        tanx += tri_tangent_f64[tri, 0] * ft
        tany += tri_tangent_f64[tri, 1] * ft
        tanz += tri_tangent_f64[tri, 2] * ft
    ln = math.sqrt(norx * norx + nory * nory + norz * norz)
    lt = math.sqrt(tanx * tanx + tany * tany + tanz * tanz)
    ok = (ln > float64(1e-6)) and (lt > float64(1e-6))
    if ln > float64(1e-30):
        nnx = norx / ln
        nny = nory / ln
        nnz = norz / ln
    else:
        nnx = norx
        nny = nory
        nnz = norz
    if lt > float64(1e-30):
        ntx = tanx / lt
        nty = tany / lt
        ntz = tanz / lt
    else:
        ntx = tanx
        nty = tany
        ntz = tanz
    d = nnx * ntx + nny * nty + nnz * ntz
    if d == float64(1.0) or d == float64(-1.0):
        ok = False
    bx = nny * ntz - nnz * nty
    by = nnz * ntx - nnx * ntz
    bz = nnx * nty - nny * ntx
    bl = math.sqrt(bx * bx + by * by + bz * bz)
    if bl > float64(1e-30):
        bx = bx / bl
        by = by / bl
        bz = bz / bl
    rrx, rry, rrz, rrw = dmath.look_rotation(float32(bx), float32(by), float32(bz),
                                             float32(nnx), float32(nny), float32(nnz))
    nax = p_normal_adjustment_rotations[p, 0] * t_negative_scale_quaternion[mt, 0]
    nay = p_normal_adjustment_rotations[p, 1] * t_negative_scale_quaternion[mt, 1]
    naz = p_normal_adjustment_rotations[p, 2] * t_negative_scale_quaternion[mt, 2]
    naw = p_normal_adjustment_rotations[p, 3] * t_negative_scale_quaternion[mt, 3]
    frx, fry, frz, frw = dmath.quat_mul(rrx, rry, rrz, rrw, nax, nay, naz, naw)
    if ok:
        p_rotations[p, 0] = frx
        p_rotations[p, 1] = fry
        p_rotations[p, 2] = frz
        p_rotations[p, 3] = frw


@cuda.jit(device=True)
def do_output_particle(p, mt, p_rotations, p_vertex_to_transform_rotations,
                       t_negative_scale_quaternion, p_out_rotations):
    # mirrors stages/display._output: out = rotation * (vertex_to_transform * negative_scale_quat)
    vqx = p_vertex_to_transform_rotations[p, 0] * t_negative_scale_quaternion[mt, 0]
    vqy = p_vertex_to_transform_rotations[p, 1] * t_negative_scale_quaternion[mt, 1]
    vqz = p_vertex_to_transform_rotations[p, 2] * t_negative_scale_quaternion[mt, 2]
    vqw = p_vertex_to_transform_rotations[p, 3] * t_negative_scale_quaternion[mt, 3]
    ox, oy, oz, ow = dmath.quat_mul(p_rotations[p, 0], p_rotations[p, 1], p_rotations[p, 2],
                                    p_rotations[p, 3], vqx, vqy, vqz, vqw)
    p_out_rotations[p, 0] = ox
    p_out_rotations[p, 1] = oy
    p_out_rotations[p, 2] = oz
    p_out_rotations[p, 3] = ow


@cuda.jit(cache=True)
def frame_kernel(phase_mask, sub_begin, sub_end,
                 fdt, sim_dt, max_sim_count, global_time_scale,
                 power0, power1, power2, power3,
                 blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4,
                 blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2,
                 blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3,
                 blob_f32_v22, blob_f64_v3,
                 offs, lens,
                 n_zones,
                 zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4,
                 zone_f32_v16,
                 zone_offs, zone_lens):
    grid = cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    # Block-local coordinates for the single-block small-serial phases (S4 FK levels, S7 angle
    # passes, F2 postline levels, T0 sync passes): only block 0 walks those loops, using
    # __syncthreads() between iterations instead of a full grid.sync (two orders cheaper on this
    # 7-block grid). tid == threadIdx.x for block 0, so it doubles as the block lane index there.
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x

    # ---- view reconstruction: each field is an axis-0 slice of its (family,per_row) blob
    # (reshape-free so cache=True pickles). RESIDENT_BLOB_LAYOUT[k]=(param,group,per_row);
    # offs[k]=row base, lens[k]=row count into blob_<group>. ----
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_time_reset = blob_u8_s[offs[3]:offs[3] + lens[3]]
    t_time = blob_f32_s[offs[4]:offs[4] + lens[4]]
    t_old_time = blob_f32_s[offs[5]:offs[5] + lens[5]]
    t_now_update = blob_f32_s[offs[6]:offs[6] + lens[6]]
    t_old_update = blob_f32_s[offs[7]:offs[7] + lens[7]]
    t_frame_update = blob_f32_s[offs[8]:offs[8] + lens[8]]
    t_frame_old = blob_f32_s[offs[9]:offs[9] + lens[9]]
    t_frame_dt = blob_f32_s[offs[10]:offs[10] + lens[10]]
    t_time_scale = blob_f32_s[offs[11]:offs[11] + lens[11]]
    t_now_time_scale = blob_f32_s[offs[12]:offs[12] + lens[12]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_skip_count = blob_i32_s[offs[14]:offs[14] + lens[14]]
    t_running = blob_u8_s[offs[15]:offs[15] + lens[15]]
    t_tether_compression = blob_f32_s[offs[16]:offs[16] + lens[16]]
    t_frame_interpolation = blob_f32_s[offs[17]:offs[17] + lens[17]]
    t_depth_inertia = blob_f32_s[offs[18]:offs[18] + lens[18]]
    t_inertia_vector = blob_f32_v3[offs[19]:offs[19] + lens[19]]
    t_step_vector = blob_f32_v3[offs[20]:offs[20] + lens[20]]
    t_inertia_rotation = blob_f32_v4[offs[21]:offs[21] + lens[21]]
    t_step_rotation = blob_f32_v4[offs[22]:offs[22] + lens[22]]
    t_old_world_position = blob_f32_v3[offs[23]:offs[23] + lens[23]]
    t_velocity_weight = blob_f32_s[offs[24]:offs[24] + lens[24]]
    t_damping_lut = blob_f32_v16[offs[25]:offs[25] + lens[25]]
    t_force_mode = blob_i8_s[offs[26]:offs[26] + lens[26]]
    t_gravity_direction = blob_f32_v3[offs[27]:offs[27] + lens[27]]
    t_gravity = blob_f32_s[offs[28]:offs[28] + lens[28]]
    t_gravity_ratio = blob_f32_s[offs[29]:offs[29] + lens[29]]
    t_impact_force = blob_f32_v3[offs[30]:offs[30] + lens[30]]
    t_scale_ratio = blob_f32_s[offs[31]:offs[31] + lens[31]]
    t_normal_axis_vector = blob_f32_v3[offs[32]:offs[32] + lens[32]]
    t_spring_limit_distance = blob_f32_s[offs[33]:offs[33] + lens[33]]
    t_spring_normal_limit_ratio = blob_f32_s[offs[34]:offs[34] + lens[34]]
    t_spring_power = blob_f32_s[offs[35]:offs[35] + lens[35]]
    t_spring_noise = blob_f32_s[offs[36]:offs[36] + lens[36]]
    t_wind_seed = blob_i32_s[offs[37]:offs[37] + lens[37]]
    t_wind_synchronization = blob_f32_s[offs[38]:offs[38] + lens[38]]
    t_wind_blend = blob_f32_s[offs[39]:offs[39] + lens[39]]
    t_wind_turbulence = blob_f32_s[offs[40]:offs[40] + lens[40]]
    t_wind_count = blob_i8_s[offs[41]:offs[41] + lens[41]]
    t_wind_main = blob_f32_v4[offs[42]:offs[42] + lens[42]]
    t_wind_time = blob_f32_v4[offs[43]:offs[43] + lens[43]]
    t_wind_dirq = blob_f32_m4x4[offs[44]:offs[44] + lens[44]]
    t_wind_zone_turbulence = blob_f32_v4[offs[45]:offs[45] + lens[45]]
    t_wind_influence = blob_f32_s[offs[46]:offs[46] + lens[46]]
    t_wind_depth_weight = blob_f32_s[offs[47]:offs[47] + lens[47]]
    t_moving_wind_main = blob_f32_s[offs[48]:offs[48] + lens[48]]
    t_wind_moving = blob_f32_s[offs[49]:offs[49] + lens[49]]
    t_moving_wind_time = blob_f32_s[offs[50]:offs[50] + lens[50]]
    t_moving_wind_dirq = blob_f32_v4[offs[51]:offs[51] + lens[51]]
    t_static_friction = blob_f32_s[offs[52]:offs[52] + lens[52]]
    t_dynamic_friction = blob_f32_s[offs[53]:offs[53] + lens[53]]
    t_particle_speed_limit = blob_f32_s[offs[54]:offs[54] + lens[54]]
    t_angular_velocity = blob_f32_s[offs[55]:offs[55] + lens[55]]
    t_centrifugal_acceleration = blob_f32_s[offs[56]:offs[56] + lens[56]]
    t_rotation_axis = blob_f32_v3[offs[57]:offs[57] + lens[57]]
    t_now_world_position = blob_f32_v3[offs[58]:offs[58] + lens[58]]
    t_is_spring = blob_u8_s[offs[59]:offs[59] + lens[59]]
    t_animation_pose_ratio = blob_f32_s[offs[60]:offs[60] + lens[60]]
    t_init_scale = blob_f32_v3[offs[61]:offs[61] + lens[61]]
    t_distance_lut = blob_f32_v16[offs[62]:offs[62] + lens[62]]
    t_motion_use_max_distance = blob_u8_s[offs[63]:offs[63] + lens[63]]
    t_motion_use_backstop = blob_u8_s[offs[64]:offs[64] + lens[64]]
    t_motion_stiffness = blob_f32_s[offs[65]:offs[65] + lens[65]]
    t_motion_backstop_radius = blob_f32_s[offs[66]:offs[66] + lens[66]]
    t_radius_lut = blob_f32_v16[offs[67]:offs[67] + lens[67]]
    t_motion_max_distance_lut = blob_f32_v16[offs[68]:offs[68] + lens[68]]
    t_motion_backstop_lut = blob_f32_v16[offs[69]:offs[69] + lens[69]]
    t_bending_stiffness = blob_f32_s[offs[70]:offs[70] + lens[70]]
    t_negative_scale_sign = blob_f32_s[offs[71]:offs[71] + lens[71]]
    t_negative_scale_direction = blob_f32_v3[offs[72]:offs[72] + lens[72]]
    t_negative_scale_quaternion = blob_f32_v4[offs[73]:offs[73] + lens[73]]
    t_is_negative_scale = blob_u8_s[offs[74]:offs[74] + lens[74]]
    t_component_world_position = blob_f32_v3[offs[75]:offs[75] + lens[75]]
    t_component_world_rotation = blob_f32_v4[offs[76]:offs[76] + lens[76]]
    t_old_component_world_position = blob_f32_v3[offs[77]:offs[77] + lens[77]]
    t_old_component_world_rotation = blob_f32_v4[offs[78]:offs[78] + lens[78]]
    t_old_component_world_scale = blob_f32_v3[offs[79]:offs[79] + lens[79]]
    t_frame_world_position = blob_f32_v3[offs[80]:offs[80] + lens[80]]
    t_frame_world_rotation = blob_f32_v4[offs[81]:offs[81] + lens[81]]
    t_frame_world_scale = blob_f32_v3[offs[82]:offs[82] + lens[82]]
    t_old_frame_world_position = blob_f32_v3[offs[83]:offs[83] + lens[83]]
    t_old_frame_world_rotation = blob_f32_v4[offs[84]:offs[84] + lens[84]]
    t_old_frame_world_scale = blob_f32_v3[offs[85]:offs[85] + lens[85]]
    t_anchor_position = blob_f32_v3[offs[86]:offs[86] + lens[86]]
    t_anchor_rotation = blob_f32_v4[offs[87]:offs[87] + lens[87]]
    t_old_anchor_position = blob_f32_v3[offs[88]:offs[88] + lens[88]]
    t_old_anchor_rotation = blob_f32_v4[offs[89]:offs[89] + lens[89]]
    t_anchor_component_local_position = blob_f32_v3[offs[90]:offs[90] + lens[90]]
    t_reset_pending = blob_u8_s[offs[91]:offs[91] + lens[91]]
    t_keep_teleport_pending = blob_u8_s[offs[92]:offs[92] + lens[92]]
    t_inertia_shift = blob_u8_s[offs[93]:offs[93] + lens[93]]
    t_negative_scale_teleport = blob_u8_s[offs[94]:offs[94] + lens[94]]
    t_now_world_rotation = blob_f32_v4[offs[95]:offs[95] + lens[95]]
    t_old_world_rotation = blob_f32_v4[offs[96]:offs[96] + lens[96]]
    t_step_move_inertia_ratio = blob_f32_s[offs[97]:offs[97] + lens[97]]
    t_step_rotation_inertia_ratio = blob_f32_s[offs[98]:offs[98] + lens[98]]
    t_local_inertia = blob_f32_s[offs[99]:offs[99] + lens[99]]
    t_local_movement_speed_limit = blob_f32_s[offs[100]:offs[100] + lens[100]]
    t_local_rotation_speed_limit = blob_f32_s[offs[101]:offs[101] + lens[101]]
    t_gravity_dot = blob_f32_s[offs[102]:offs[102] + lens[102]]
    t_init_local_gravity_direction = blob_f32_v3[offs[103]:offs[103] + lens[103]]
    t_gravity_falloff = blob_f32_s[offs[104]:offs[104] + lens[104]]
    t_stablization_time = blob_f32_s[offs[105]:offs[105] + lens[105]]
    t_blend_weight = blob_f32_s[offs[106]:offs[106] + lens[106]]
    t_blend_weight_param = blob_f32_s[offs[107]:offs[107] + lens[107]]
    t_distance_weight = blob_f32_s[offs[108]:offs[108] + lens[108]]
    t_frame_moving_speed = blob_f32_s[offs[109]:offs[109] + lens[109]]
    t_frame_moving_direction = blob_f32_v3[offs[110]:offs[110] + lens[110]]
    t_moving_wind_direction = blob_f32_v3[offs[111]:offs[111] + lens[111]]
    t_wind_frequency = blob_f32_s[offs[112]:offs[112] + lens[112]]
    t_collision_mode = blob_i8_s[offs[113]:offs[113] + lens[113]]
    t_limit_distance_lut = blob_f32_v16[offs[114]:offs[114] + lens[114]]
    t_negative_scale_matrix = blob_f64_m4x4[offs[115]:offs[115] + lens[115]]
    t_negative_scale_change = blob_f32_v3[offs[116]:offs[116] + lens[116]]
    t_frame_component_shift_vector = blob_f32_v3[offs[117]:offs[117] + lens[117]]
    t_frame_component_shift_rotation = blob_f32_v4[offs[118]:offs[118] + lens[118]]
    t_sync_target = blob_i32_s[offs[119]:offs[119] + lens[119]]
    t_sync_top = blob_i32_s[offs[120]:offs[120] + lens[120]]
    t_negative_scale_triangle_sign = blob_f32_v2[offs[121]:offs[121] + lens[121]]
    t_smoothing_velocity = blob_f32_v3[offs[122]:offs[122] + lens[122]]
    t_has_anchor = blob_u8_s[offs[123]:offs[123] + lens[123]]
    t_had_anchor = blob_u8_s[offs[124]:offs[124] + lens[124]]
    t_anchor_inertia = blob_f32_s[offs[125]:offs[125] + lens[125]]
    t_world_inertia = blob_f32_s[offs[126]:offs[126] + lens[126]]
    t_movement_inertia_smoothing = blob_f32_s[offs[127]:offs[127] + lens[127]]
    t_movement_speed_limit = blob_f32_s[offs[128]:offs[128] + lens[128]]
    t_rotation_speed_limit = blob_f32_s[offs[129]:offs[129] + lens[129]]
    t_teleport_mode = blob_i8_s[offs[130]:offs[130] + lens[130]]
    t_teleport_distance = blob_f32_s[offs[131]:offs[131] + lens[131]]
    t_teleport_rotation = blob_f32_s[offs[132]:offs[132] + lens[132]]
    t_culling_invisible = blob_u8_s[offs[133]:offs[133] + lens[133]]
    t_wind_direction = blob_f32_m4x3[offs[134]:offs[134] + lens[134]]
    t_wind_zone_id = blob_i32_v4[offs[135]:offs[135] + lens[135]]
    t_angle_use_limit = blob_u8_s[offs[136]:offs[136] + lens[136]]
    t_angle_use_restoration = blob_u8_s[offs[137]:offs[137] + lens[137]]
    t_angle_limit_lut = blob_f32_v16[offs[138]:offs[138] + lens[138]]
    t_angle_limit_stiffness = blob_f32_s[offs[139]:offs[139] + lens[139]]
    t_angle_restoration_lut = blob_f32_v16[offs[140]:offs[140] + lens[140]]
    t_angle_restoration_attenuation = blob_f32_s[offs[141]:offs[141] + lens[141]]
    t_angle_restoration_gravity_falloff = blob_f32_s[offs[142]:offs[142] + lens[142]]
    t_rotational_interpolation = blob_f32_s[offs[143]:offs[143] + lens[143]]
    t_root_rotation = blob_f32_s[offs[144]:offs[144] + lens[144]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    p_local_positions = blob_f32_v3[offs[146]:offs[146] + lens[146]]
    p_local_normals = blob_f32_v3[offs[147]:offs[147] + lens[147]]
    p_local_tangents = blob_f32_v3[offs[148]:offs[148] + lens[148]]
    p_skin_indices = blob_i32_v4[offs[149]:offs[149] + lens[149]]
    p_skin_weights = blob_f32_v4[offs[150]:offs[150] + lens[150]]
    p_positions = blob_f32_v3[offs[151]:offs[151] + lens[151]]
    p_rotations = blob_f32_v4[offs[152]:offs[152] + lens[152]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_velocity_positions = blob_f32_v3[offs[154]:offs[154] + lens[154]]
    p_step_basic_positions = blob_f32_v3[offs[155]:offs[155] + lens[155]]
    p_vertex_root = blob_i32_s[offs[156]:offs[156] + lens[156]]
    p_old_anim_positions = blob_f32_v3[offs[157]:offs[157] + lens[157]]
    p_old_anim_rotations = blob_f32_v4[offs[158]:offs[158] + lens[158]]
    p_base_positions = blob_f32_v3[offs[159]:offs[159] + lens[159]]
    p_base_rotations = blob_f32_v4[offs[160]:offs[160] + lens[160]]
    p_step_basic_rotations = blob_f32_v4[offs[161]:offs[161] + lens[161]]
    p_depth = blob_f32_s[offs[162]:offs[162] + lens[162]]
    p_velocities = blob_f32_v3[offs[163]:offs[163] + lens[163]]
    p_old_positions = blob_f32_v3[offs[164]:offs[164] + lens[164]]
    p_friction = blob_f32_s[offs[165]:offs[165] + lens[165]]
    p_vertex_root_local = blob_i32_s[offs[166]:offs[166] + lens[166]]
    p_collision_normals = blob_f32_v3[offs[167]:offs[167] + lens[167]]
    p_static_friction = blob_f32_s[offs[168]:offs[168] + lens[168]]
    p_real_velocities = blob_f32_v3[offs[169]:offs[169] + lens[169]]
    p_attr_move = blob_u8_s[offs[170]:offs[170] + lens[170]]
    p_vertex_local_positions = blob_f32_v3[offs[171]:offs[171] + lens[171]]
    p_vertex_local_rotations = blob_f32_v4[offs[172]:offs[172] + lens[172]]
    p_old_rotations = blob_f32_v4[offs[173]:offs[173] + lens[173]]
    p_display_positions = blob_f32_v3[offs[174]:offs[174] + lens[174]]
    p_vertex_bind_pose_rotations = blob_f32_v4[offs[175]:offs[175] + lens[175]]
    p_vertex_parent = blob_i32_s[offs[176]:offs[176] + lens[176]]
    p_albuf_length = blob_f32_s[offs[177]:offs[177] + lens[177]]
    p_albuf_local_pos = blob_f32_v3[offs[178]:offs[178] + lens[178]]
    p_albuf_local_rot = blob_f32_v4[offs[179]:offs[179] + lens[179]]
    p_albuf_restore = blob_f32_v3[offs[180]:offs[180] + lens[180]]
    p_albuf_rotation = blob_f32_v4[offs[181]:offs[181] + lens[181]]
    p_uv = blob_f32_v2[offs[182]:offs[182] + lens[182]]
    p_attr_zero_distance = blob_u8_s[offs[183]:offs[183] + lens[183]]
    p_attr_invalid = blob_u8_s[offs[184]:offs[184] + lens[184]]
    p_temp_base_positions = blob_f32_v3[offs[185]:offs[185] + lens[185]]
    p_temp_base_rotations = blob_f32_v4[offs[186]:offs[186] + lens[186]]
    p_normal_adjustment_rotations = blob_f32_v4[offs[187]:offs[187] + lens[187]]
    p_vertex_to_transform_rotations = blob_f32_v4[offs[188]:offs[188] + lens[188]]
    p_out_rotations = blob_f32_v4[offs[189]:offs[189] + lens[189]]
    x_world = blob_f32_m4x4[offs[190]:offs[190] + lens[190]]
    x_bind = blob_f32_m4x4[offs[191]:offs[191] + lens[191]]
    c_team = blob_i32_s[offs[192]:offs[192] + lens[192]]
    c_kind = blob_i32_s[offs[193]:offs[193] + lens[193]]
    c_center = blob_f32_v3[offs[194]:offs[194] + lens[194]]
    c_size = blob_f32_v3[offs[195]:offs[195] + lens[195]]
    c_axis = blob_f32_v3[offs[196]:offs[196] + lens[196]]
    c_aligned = blob_u8_s[offs[197]:offs[197] + lens[197]]
    c_enabled = blob_u8_s[offs[198]:offs[198] + lens[198]]
    c_enabled_prev = blob_u8_s[offs[199]:offs[199] + lens[199]]
    c_active = blob_u8_s[offs[200]:offs[200] + lens[200]]
    c_input_positions = blob_f32_v3[offs[201]:offs[201] + lens[201]]
    c_input_rotations = blob_f32_v4[offs[202]:offs[202] + lens[202]]
    c_input_scales = blob_f32_v3[offs[203]:offs[203] + lens[203]]
    c_frame_pos = blob_f32_v3[offs[204]:offs[204] + lens[204]]
    c_frame_rot = blob_f32_v4[offs[205]:offs[205] + lens[205]]
    c_frame_scl = blob_f32_v3[offs[206]:offs[206] + lens[206]]
    c_old_frame_pos = blob_f32_v3[offs[207]:offs[207] + lens[207]]
    c_old_frame_rot = blob_f32_v4[offs[208]:offs[208] + lens[208]]
    c_now_pos = blob_f32_v3[offs[209]:offs[209] + lens[209]]
    c_now_rot = blob_f32_v4[offs[210]:offs[210] + lens[210]]
    c_old_pos = blob_f32_v3[offs[211]:offs[211] + lens[211]]
    c_old_rot = blob_f32_v4[offs[212]:offs[212] + lens[212]]
    c_work_radius = blob_f32_v2[offs[213]:offs[213] + lens[213]]
    c_work_old_pos = blob_f32_m2x3[offs[214]:offs[214] + lens[214]]
    c_work_next_pos = blob_f32_m2x3[offs[215]:offs[215] + lens[215]]
    c_work_rot = blob_f32_v4[offs[216]:offs[216] + lens[216]]
    c_work_inv_old_rot = blob_f32_v4[offs[217]:offs[217] + lens[217]]
    c_work_aabb_min = blob_f32_v3[offs[218]:offs[218] + lens[218]]
    c_work_aabb_max = blob_f32_v3[offs[219]:offs[219] + lens[219]]
    st_tether_particle = blob_i32_s[offs[220]:offs[220] + lens[220]]
    st_tether_team = blob_i32_s[offs[221]:offs[221] + lens[221]]
    st_move_particle = blob_i32_s[offs[222]:offs[222] + lens[222]]
    st_move_team = blob_i32_s[offs[223]:offs[223] + lens[223]]
    st_fixed_particle = blob_i32_s[offs[224]:offs[224] + lens[224]]
    st_fixed_team = blob_i32_s[offs[225]:offs[225] + lens[225]]
    st_spring_particle = blob_i32_s[offs[226]:offs[226] + lens[226]]
    st_spring_team = blob_i32_s[offs[227]:offs[227] + lens[227]]
    st_distance_target = blob_i32_s[offs[228]:offs[228] + lens[228]]
    st_distance_rest = blob_f32_s[offs[229]:offs[229] + lens[229]]
    st_motion_particle = blob_i32_s[offs[230]:offs[230] + lens[230]]
    st_motion_team = blob_i32_s[offs[231]:offs[231] + lens[231]]
    st_bending_team = blob_i32_s[offs[232]:offs[232] + lens[232]]
    st_bending_pair = blob_i32_v4[offs[233]:offs[233] + lens[233]]
    st_bending_rest = blob_f32_s[offs[234]:offs[234] + lens[234]]
    st_bending_sign = blob_i8_s[offs[235]:offs[235] + lens[235]]
    st_point_pair_collider = blob_i32_s[offs[236]:offs[236] + lens[236]]
    st_edge_pair_collider = blob_i32_s[offs[237]:offs[237] + lens[237]]
    st_collision_edge = blob_i32_v2[offs[238]:offs[238] + lens[238]]
    st_center_fixed_particle = blob_i32_s[offs[239]:offs[239] + lens[239]]
    st_angle_buffered_particle = blob_i32_s[offs[240]:offs[240] + lens[240]]
    st_triangle_team = blob_i32_s[offs[241]:offs[241] + lens[241]]
    st_triangle_particles = blob_i32_v3[offs[242]:offs[242] + lens[242]]
    st_v2t_triangle = blob_i32_s[offs[243]:offs[243] + lens[243]]
    st_v2t_flip_normal = blob_f32_s[offs[244]:offs[244] + lens[244]]
    st_v2t_flip_tangent = blob_f32_s[offs[245]:offs[245] + lens[245]]
    csr_distance_offsets = blob_i32_s[offs[246]:offs[246] + lens[246]]
    csr_distance_order = blob_i32_s[offs[247]:offs[247] + lens[247]]
    csr_point_pair_offsets = blob_i32_s[offs[248]:offs[248] + lens[248]]
    csr_point_pair_order = blob_i32_s[offs[249]:offs[249] + lens[249]]
    csr_edge_pair_offsets = blob_i32_s[offs[250]:offs[250] + lens[250]]
    csr_edge_pair_order = blob_i32_s[offs[251]:offs[251] + lens[251]]
    csr_center_fixed_offsets = blob_i32_s[offs[252]:offs[252] + lens[252]]
    csr_center_fixed_order = blob_i32_s[offs[253]:offs[253] + lens[253]]
    csr_v2t_offsets = blob_i32_s[offs[254]:offs[254] + lens[254]]
    csr_v2t_order = blob_i32_s[offs[255]:offs[255] + lens[255]]
    fk_yes_offsets = blob_i32_s[offs[256]:offs[256] + lens[256]]
    fk_yes = blob_i32_s[offs[257]:offs[257] + lens[257]]
    fk_yes_parent = blob_i32_s[offs[258]:offs[258] + lens[258]]
    fk_no_offsets = blob_i32_s[offs[259]:offs[259] + lens[259]]
    fk_no = blob_i32_s[offs[260]:offs[260] + lens[260]]
    baseline_entries = blob_i32_s[offs[261]:offs[261] + lens[261]]
    angle_pass_offsets = blob_i32_s[offs[262]:offs[262] + lens[262]]
    angle_pass_vertices = blob_i32_s[offs[263]:offs[263] + lens[263]]
    angle_pass_parents = blob_i32_s[offs[264]:offs[264] + lens[264]]
    postline_entry_offsets = blob_i32_s[offs[265]:offs[265] + lens[265]]
    postline_entry_vertices = blob_i32_s[offs[266]:offs[266] + lens[266]]
    postline_child_offsets = blob_i32_s[offs[267]:offs[267] + lens[267]]
    postline_child_vertices = blob_i32_s[offs[268]:offs[268] + lens[268]]
    st_display_update_move_mask = blob_u8_s[offs[269]:offs[269] + lens[269]]
    sc_dcorr = blob_f32_v3[offs[270]:offs[270] + lens[270]]
    sc_dcorr_fixed = blob_i32_v3[offs[271]:offs[271] + lens[271]]
    sc_dcount = blob_i32_s[offs[272]:offs[272] + lens[272]]
    sc_col_friction_fixed = blob_i32_s[offs[273]:offs[273] + lens[273]]
    sc_col_normal_fixed = blob_i32_v3[offs[274]:offs[274] + lens[274]]
    sc_sync = blob_f32_v22[offs[275]:offs[275] + lens[275]]
    sc_tri_normal_f64 = blob_f64_v3[offs[276]:offs[276] + lens[276]]
    sc_tri_tangent_f64 = blob_f64_v3[offs[277]:offs[277] + lens[277]]
    z_zone_id = zone_i32_s[zone_offs[0]:zone_offs[0] + zone_lens[0]]
    z_mode = zone_i32_s[zone_offs[1]:zone_offs[1] + zone_lens[1]]
    z_is_addition = zone_u8_s[zone_offs[2]:zone_offs[2] + zone_lens[2]]
    z_main = zone_f32_s[zone_offs[3]:zone_offs[3] + zone_lens[3]]
    z_turbulence = zone_f32_s[zone_offs[4]:zone_offs[4] + zone_lens[4]]
    z_world_position = zone_f32_v3[zone_offs[5]:zone_offs[5] + zone_lens[5]]
    z_world_direction = zone_f32_v3[zone_offs[6]:zone_offs[6] + zone_lens[6]]
    z_world_to_local = zone_f64_m4x4[zone_offs[7]:zone_offs[7] + zone_lens[7]]
    z_size = zone_f32_v3[zone_offs[8]:zone_offs[8] + zone_lens[8]]
    z_zone_volume = zone_f32_s[zone_offs[9]:zone_offs[9] + zone_lens[9]]
    z_attenuation_lut = zone_f32_v16[zone_offs[10]:zone_offs[10] + zone_lens[10]]
    num_teams = t_enabled.shape[0]

    num_particles = p_team.shape[0]
    num_colliders = c_team.shape[0]

    # ----- FRAME-PRE -----
    # T0 team_time.resolve_sync (per ENABLED team; gate = enabled only, not the
    # frame mask). Pass a: resolve sync_top by climbing sync_target (<=8 hops).
    if bid == 0 and (phase_mask & PHASE_SYNC) != 0:
        i = tid
        while i < num_teams:
            if t_enabled[i] != 0:
                target = t_sync_target[i]
                if target <= 0 or t_valid[target] == 0 or t_enabled[target] == 0:
                    t_sync_top[i] = int32(0)
                else:
                    top = target
                    for _h in range(8):
                        upper = t_sync_target[top]
                        if upper <= 0 or upper == i or t_valid[upper] == 0 or t_enabled[upper] == 0:
                            break
                        top = upper
                    t_sync_top[i] = top
            i += bdim
    cuda.syncthreads()
    # Pass b: snapshot each child's sync_top row (the gather RHS) into sc_sync so
    # the write pass reads pre-gather values (mutual-sync A<->B swap is race-safe).
    if bid == 0 and (phase_mask & PHASE_SYNC) != 0:
        i = tid
        while i < num_teams:
            if t_enabled[i] != 0 and t_sync_top[i] > 0:
                top = t_sync_top[i]
                sc_sync[i, 0] = t_time[top]
                sc_sync[i, 1] = t_old_time[top]
                sc_sync[i, 2] = t_now_update[top]
                sc_sync[i, 3] = t_old_update[top]
                sc_sync[i, 4] = t_frame_update[top]
                sc_sync[i, 5] = t_frame_old[top]
                sc_sync[i, 6] = t_time_scale[top]
                sc_sync[i, 7] = t_anchor_inertia[top]
                sc_sync[i, 8] = t_world_inertia[top]
                sc_sync[i, 9] = t_movement_inertia_smoothing[top]
                sc_sync[i, 10] = t_movement_speed_limit[top]
                sc_sync[i, 11] = t_rotation_speed_limit[top]
                sc_sync[i, 12] = float32(t_teleport_mode[top])
                sc_sync[i, 13] = t_teleport_distance[top]
                sc_sync[i, 14] = t_teleport_rotation[top]
                sc_sync[i, 15] = t_component_world_position[top, 0]
                sc_sync[i, 16] = t_component_world_position[top, 1]
                sc_sync[i, 17] = t_component_world_position[top, 2]
                sc_sync[i, 18] = t_component_world_rotation[top, 0]
                sc_sync[i, 19] = t_component_world_rotation[top, 1]
                sc_sync[i, 20] = t_component_world_rotation[top, 2]
                sc_sync[i, 21] = t_component_world_rotation[top, 3]
            i += bdim
    cuda.syncthreads()
    # Pass c: write children from the snapshot.
    if bid == 0 and (phase_mask & PHASE_SYNC) != 0:
        i = tid
        while i < num_teams:
            if t_enabled[i] != 0 and t_sync_top[i] > 0:
                t_time[i] = sc_sync[i, 0]
                t_old_time[i] = sc_sync[i, 1]
                t_now_update[i] = sc_sync[i, 2]
                t_old_update[i] = sc_sync[i, 3]
                t_frame_update[i] = sc_sync[i, 4]
                t_frame_old[i] = sc_sync[i, 5]
                t_time_scale[i] = sc_sync[i, 6]
                t_anchor_inertia[i] = sc_sync[i, 7]
                t_world_inertia[i] = sc_sync[i, 8]
                t_movement_inertia_smoothing[i] = sc_sync[i, 9]
                t_movement_speed_limit[i] = sc_sync[i, 10]
                t_rotation_speed_limit[i] = sc_sync[i, 11]
                t_teleport_mode[i] = int8(sc_sync[i, 12])
                t_teleport_distance[i] = sc_sync[i, 13]
                t_teleport_rotation[i] = sc_sync[i, 14]
                t_component_world_position[i, 0] = sc_sync[i, 15]
                t_component_world_position[i, 1] = sc_sync[i, 16]
                t_component_world_position[i, 2] = sc_sync[i, 17]
                t_component_world_rotation[i, 0] = sc_sync[i, 18]
                t_component_world_rotation[i, 1] = sc_sync[i, 19]
                t_component_world_rotation[i, 2] = sc_sync[i, 20]
                t_component_world_rotation[i, 3] = sc_sync[i, 21]
            i += bdim
    grid.sync()

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

    # C0 center.run + select_team_wind (per-team; authorised internal f64 for the
    # trs/analytic-inverse matrix path and the fixed-centre gather; mirrors
    # stages/center.py + stages/wind.select_team_wind, f32 elsewhere).
    if phase_mask & PHASE_CENTER:
        mat_a = cuda.local.array((4, 4), float64)
        mat_b = cuda.local.array((4, 4), float64)
        mat_c = cuda.local.array((4, 4), float64)
        res_zone_id = cuda.local.array(8, int32)
        res_time = cuda.local.array(8, float32)
        res_main = cuda.local.array(8, float32)
        res_dx = cuda.local.array(8, float32)
        res_dy = cuda.local.array(8, float32)
        res_dz = cuda.local.array(8, float32)
        res_turb = cuda.local.array(8, float32)
        old_zid = cuda.local.array(4, int32)
        old_wt = cuda.local.array(4, float32)
        i = tid
        while i < num_teams:
            if team_frame_mask(t_enabled, t_valid, t_cws, i):
                cpx = t_component_world_position[i, 0]
                cpy = t_component_world_position[i, 1]
                cpz = t_component_world_position[i, 2]
                crx = t_component_world_rotation[i, 0]
                cry = t_component_world_rotation[i, 1]
                crz = t_component_world_rotation[i, 2]
                crw = t_component_world_rotation[i, 3]
                csx = t_cws[i, 0]
                csy = t_cws[i, 1]
                csz = t_cws[i, 2]

                # --- negative scale ---
                init_scale_len = float64(dmath.length3(
                    t_init_scale[i, 0], t_init_scale[i, 1], t_init_scale[i, 2]))
                if init_scale_len < float64(1e-30):
                    init_scale_len = float64(1e-30)
                csr_ratio = float64(dmath.length3(csx, csy, csz)) / init_scale_len

                old_dx = t_negative_scale_direction[i, 0]
                old_dy = t_negative_scale_direction[i, 1]
                old_dz = t_negative_scale_direction[i, 2]
                sxv = float32(1.0) if csx == float32(0.0) else csx
                syv = float32(1.0) if csy == float32(0.0) else csy
                szv = float32(1.0) if csz == float32(0.0) else csz
                dir_x = dmath.fsign(sxv)
                dir_y = dmath.fsign(syv)
                dir_z = dmath.fsign(szv)
                t_negative_scale_direction[i, 0] = dir_x
                t_negative_scale_direction[i, 1] = dir_y
                t_negative_scale_direction[i, 2] = dir_z
                t_negative_scale_change[i, 0] = old_dx * dir_x
                t_negative_scale_change[i, 1] = old_dy * dir_y
                t_negative_scale_change[i, 2] = old_dz * dir_z
                is_negative = (csx < float32(0.0)) or (csy < float32(0.0)) or (csz < float32(0.0))
                t_is_negative_scale[i] = int32(1) if is_negative else int32(0)
                t_negative_scale_sign[i] = float32(-1.0) if is_negative else float32(1.0)
                if is_negative:
                    t_negative_scale_quaternion[i, 0] = -dir_x
                    t_negative_scale_quaternion[i, 1] = -dir_y
                    t_negative_scale_quaternion[i, 2] = -dir_z
                    t_negative_scale_quaternion[i, 3] = float32(1.0)
                    ts0 = float32(-1.0) if (csx < float32(0.0) or csz < float32(0.0)) else float32(1.0)
                    ts1 = float32(-1.0) if (csx < float32(0.0)) else float32(1.0)
                    t_negative_scale_triangle_sign[i, 0] = ts0
                    t_negative_scale_triangle_sign[i, 1] = ts1
                else:
                    t_negative_scale_quaternion[i, 0] = float32(1.0)
                    t_negative_scale_quaternion[i, 1] = float32(1.0)
                    t_negative_scale_quaternion[i, 2] = float32(1.0)
                    t_negative_scale_quaternion[i, 3] = float32(1.0)
                    t_negative_scale_triangle_sign[i, 0] = float32(1.0)
                    t_negative_scale_triangle_sign[i, 1] = float32(1.0)
                teleport = (old_dx != dir_x) or (old_dy != dir_y) or (old_dz != dir_z)
                t_negative_scale_teleport[i] = int32(1) if teleport else int32(0)

                # --- teleport: negative_component applied to old_* and smoothing ---
                if teleport:
                    dmath.trs_build_f64(mat_a, cpx, cpy, cpz, crx, cry, crz, crw, csx, csy, csz)
                    ocpx = t_old_component_world_position[i, 0]
                    ocpy = t_old_component_world_position[i, 1]
                    ocpz = t_old_component_world_position[i, 2]
                    ocrx = t_old_component_world_rotation[i, 0]
                    ocry = t_old_component_world_rotation[i, 1]
                    ocrz = t_old_component_world_rotation[i, 2]
                    ocrw = t_old_component_world_rotation[i, 3]
                    ocsx = t_old_component_world_scale[i, 0]
                    ocsy = t_old_component_world_scale[i, 1]
                    ocsz = t_old_component_world_scale[i, 2]
                    dmath.trs_inverse_f64(mat_b, ocpx, ocpy, ocpz, ocrx, ocry, ocrz, ocrw, ocsx, ocsy, ocsz)
                    dmath.mat4_mul_f64(mat_c, mat_a, mat_b)
                    nx, ny, nz = dmath.transform_point(mat_c, ocpx, ocpy, ocpz)
                    t_old_component_world_position[i, 0] = nx
                    t_old_component_world_position[i, 1] = ny
                    t_old_component_world_position[i, 2] = nz
                    t_old_component_world_scale[i, 0] = csx
                    t_old_component_world_scale[i, 1] = csy
                    t_old_component_world_scale[i, 2] = csz
                    oax = t_old_anchor_position[i, 0]
                    oay = t_old_anchor_position[i, 1]
                    oaz = t_old_anchor_position[i, 2]
                    tax, tay, taz = dmath.transform_point(mat_c, oax, oay, oaz)
                    t_old_anchor_position[i, 0] = tax
                    t_old_anchor_position[i, 1] = tay
                    t_old_anchor_position[i, 2] = taz
                    tsvx, tsvy, tsvz = dmath.transform_vector(
                        mat_c, t_smoothing_velocity[i, 0], t_smoothing_velocity[i, 1],
                        t_smoothing_velocity[i, 2])
                    t_smoothing_velocity[i, 0] = tsvx
                    t_smoothing_velocity[i, 1] = tsvy
                    t_smoothing_velocity[i, 2] = tsvz

                ocp_x = t_old_component_world_position[i, 0]
                ocp_y = t_old_component_world_position[i, 1]
                ocp_z = t_old_component_world_position[i, 2]
                ocr_x = t_old_component_world_rotation[i, 0]
                ocr_y = t_old_component_world_rotation[i, 1]
                ocr_z = t_old_component_world_rotation[i, 2]
                ocr_w = t_old_component_world_rotation[i, 3]

                # --- fixed-centre gather (f64 accumulate over center_fixed CSR) ---
                cwpx = cpx
                cwpy = cpy
                cwpz = cpz
                cwrx = crx
                cwry = cry
                cwrz = crz
                cwrw = crw
                nor_sx = float64(0.0)
                nor_sy = float64(0.0)
                nor_sz = float64(0.0)
                tan_sx = float64(0.0)
                tan_sy = float64(0.0)
                tan_sz = float64(0.0)
                pos_sx = float64(0.0)
                pos_sy = float64(0.0)
                pos_sz = float64(0.0)
                fcount = 0
                seg0 = csr_center_fixed_offsets[i]
                seg1 = csr_center_fixed_offsets[i + 1]
                for e in range(seg0, seg1):
                    fp = st_center_fixed_particle[csr_center_fixed_order[e]]
                    rx = p_rotations[fp, 0]
                    ry = p_rotations[fp, 1]
                    rz = p_rotations[fp, 2]
                    rw = p_rotations[fp, 3]
                    if is_negative:
                        nnx, nny, nnz = dmath.quat_to_normal(rx, ry, rz, rw)
                        ttx, tty, ttz = dmath.quat_to_tangent(rx, ry, rz, rw)
                        rx, ry, rz, rw = dmath.to_rotation(-nnx, -nny, -nnz, -ttx, -tty, -ttz)
                    rx, ry, rz, rw = dmath.quat_mul(
                        rx, ry, rz, rw, p_vertex_bind_pose_rotations[fp, 0],
                        p_vertex_bind_pose_rotations[fp, 1], p_vertex_bind_pose_rotations[fp, 2],
                        p_vertex_bind_pose_rotations[fp, 3])
                    norx, nory, norz = dmath.quat_to_normal(rx, ry, rz, rw)
                    tanx, tany, tanz = dmath.quat_to_tangent(rx, ry, rz, rw)
                    nflip = float32(-1.0) if (dir_x < float32(0.0) or dir_z < float32(0.0)) else float32(1.0)
                    tflip = float32(-1.0) if (dir_x < float32(0.0) or dir_y < float32(0.0)) else float32(1.0)
                    nor_sx += float64(norx * nflip)
                    nor_sy += float64(nory * nflip)
                    nor_sz += float64(norz * nflip)
                    tan_sx += float64(tanx * tflip)
                    tan_sy += float64(tany * tflip)
                    tan_sz += float64(tanz * tflip)
                    pos_sx += float64(p_positions[fp, 0])
                    pos_sy += float64(p_positions[fp, 1])
                    pos_sz += float64(p_positions[fp, 2])
                    fcount += 1
                if fcount > 0:
                    nl = math.sqrt(nor_sx * nor_sx + nor_sy * nor_sy + nor_sz * nor_sz)
                    tl = math.sqrt(tan_sx * tan_sx + tan_sy * tan_sy + tan_sz * tan_sz)
                    if nl > float64(1e-30) and tl > float64(1e-30):
                        cwpx = float32(pos_sx / float64(fcount))
                        cwpy = float32(pos_sy / float64(fcount))
                        cwpz = float32(pos_sz / float64(fcount))
                        cwrx, cwry, cwrz, cwrw = dmath.to_rotation(
                            float32(nor_sx / nl), float32(nor_sy / nl), float32(nor_sz / nl),
                            float32(tan_sx / tl), float32(tan_sy / tl), float32(tan_sz / tl))

                # --- teleport negative_scale_matrix (uses centre pose) ---
                if teleport:
                    dmath.trs_build_f64(mat_a, cwpx, cwpy, cwpz, cwrx, cwry, cwrz, cwrw, csx, csy, csz)
                    dmath.trs_inverse_f64(
                        mat_b, t_old_frame_world_position[i, 0], t_old_frame_world_position[i, 1],
                        t_old_frame_world_position[i, 2], t_old_frame_world_rotation[i, 0],
                        t_old_frame_world_rotation[i, 1], t_old_frame_world_rotation[i, 2],
                        t_old_frame_world_rotation[i, 3], t_old_frame_world_scale[i, 0],
                        t_old_frame_world_scale[i, 1], t_old_frame_world_scale[i, 2])
                    dmath.mat4_mul_f64(t_negative_scale_matrix[i], mat_a, mat_b)

                # --- anchor ---
                adv_x = float32(0.0)
                adv_y = float32(0.0)
                adv_z = float32(0.0)
                adr_x = float32(0.0)
                adr_y = float32(0.0)
                adr_z = float32(0.0)
                adr_w = float32(1.0)
                has_anc = t_has_anchor[i] != 0
                anchor_reset = (has_anc != (t_had_anchor[i] != 0)) or (t_reset_pending[i] != 0)
                t_had_anchor[i] = int32(1) if has_anc else int32(0)
                if anchor_reset:
                    iqx, iqy, iqz, iqw = dmath.quat_inverse(
                        t_anchor_rotation[i, 0], t_anchor_rotation[i, 1],
                        t_anchor_rotation[i, 2], t_anchor_rotation[i, 3])
                    alx, aly, alz = dmath.quat_rotate(
                        iqx, iqy, iqz, iqw, cpx - t_anchor_position[i, 0],
                        cpy - t_anchor_position[i, 1], cpz - t_anchor_position[i, 2])
                    t_old_anchor_position[i, 0] = t_anchor_position[i, 0]
                    t_old_anchor_position[i, 1] = t_anchor_position[i, 1]
                    t_old_anchor_position[i, 2] = t_anchor_position[i, 2]
                    t_old_anchor_rotation[i, 0] = t_anchor_rotation[i, 0]
                    t_old_anchor_rotation[i, 1] = t_anchor_rotation[i, 1]
                    t_old_anchor_rotation[i, 2] = t_anchor_rotation[i, 2]
                    t_old_anchor_rotation[i, 3] = t_anchor_rotation[i, 3]
                    t_anchor_component_local_position[i, 0] = alx
                    t_anchor_component_local_position[i, 1] = aly
                    t_anchor_component_local_position[i, 2] = alz
                if has_anc:
                    rlx, rly, rlz = dmath.quat_rotate(
                        t_anchor_rotation[i, 0], t_anchor_rotation[i, 1], t_anchor_rotation[i, 2],
                        t_anchor_rotation[i, 3], t_anchor_component_local_position[i, 0],
                        t_anchor_component_local_position[i, 1], t_anchor_component_local_position[i, 2])
                    dvx = (rlx + t_anchor_position[i, 0]) - ocp_x
                    dvy = (rly + t_anchor_position[i, 1]) - ocp_y
                    dvz = (rlz + t_anchor_position[i, 2]) - ocp_z
                    ioax, ioay, ioaz, ioaw = dmath.quat_inverse(
                        t_old_anchor_rotation[i, 0], t_old_anchor_rotation[i, 1],
                        t_old_anchor_rotation[i, 2], t_old_anchor_rotation[i, 3])
                    drx, dry, drz, drw = dmath.quat_mul(
                        t_anchor_rotation[i, 0], t_anchor_rotation[i, 1], t_anchor_rotation[i, 2],
                        t_anchor_rotation[i, 3], ioax, ioay, ioaz, ioaw)
                    a_ratio = float32(1.0) - t_anchor_inertia[i]
                    adv_x = dvx * a_ratio
                    adv_y = dvy * a_ratio
                    adv_z = dvz * a_ratio
                    adr_x, adr_y, adr_z, adr_w = dmath.quat_slerp(
                        float32(0.0), float32(0.0), float32(0.0), float32(1.0),
                        drx, dry, drz, drw, a_ratio)
                    ocp_x = ocp_x + adv_x
                    ocp_y = ocp_y + adv_y
                    ocp_z = ocp_z + adv_z
                    ocr_x, ocr_y, ocr_z, ocr_w = dmath.quat_mul(
                        adr_x, adr_y, adr_z, adr_w, ocr_x, ocr_y, ocr_z, ocr_w)
                    t_inertia_shift[i] = int32(1)

                # --- frame delta + teleport distance/angle check ---
                fdvx = cpx - ocp_x
                fdvy = cpy - ocp_y
                fdvz = cpz - ocp_z
                fda = dmath.quat_angle(ocr_x, ocr_y, ocr_z, ocr_w, crx, cry, crz, crw)
                if (t_teleport_mode[i] != 0) and (t_reset_pending[i] == 0):
                    far = float64(dmath.length3(fdvx, fdvy, fdvz)) >= \
                        float64(t_teleport_distance[i]) * csr_ratio
                    spun = (fda * RAD2DEG) >= t_teleport_rotation[i]
                    if far or spun:
                        if t_teleport_mode[i] == TELEPORT_RESET:
                            t_reset_pending[i] = int32(1)
                        else:
                            t_keep_teleport_pending[i] = int32(1)

                reset = t_reset_pending[i] != 0
                keep = t_keep_teleport_pending[i] != 0

                # --- smoothing ---
                sdv_x = float32(0.0)
                sdv_y = float32(0.0)
                sdv_z = float32(0.0)
                if (t_movement_inertia_smoothing[i] >= float32(1e-6)) and (not (keep or reset)):
                    running = t_running[i] != 0
                    fdt_i = t_frame_dt[i]
                    if fdt_i > float32(0.0):
                        dvx = fdvx / fdt_i
                        dvy = fdvy / fdt_i
                        dvz = fdvz / fdt_i
                    else:
                        dvx = float32(0.0)
                        dvy = float32(0.0)
                        dvz = float32(0.0)
                    limit = t_movement_speed_limit[i] * float32(csr_ratio)
                    mlim = limit if limit > float32(0.0) else float32(0.0)
                    cvx, cvy, cvz = dmath.clamp_vector(dvx, dvy, dvz, mlim)
                    if limit >= float32(0.0):
                        dvx = cvx
                        dvy = cvy
                        dvz = cvz
                    mis = t_movement_inertia_smoothing[i]
                    om = float32(1.0) - mis
                    avg = dmath.saturate(om * om * om * float32(0.99) + float32(0.01))
                    svx = t_smoothing_velocity[i, 0]
                    svy = t_smoothing_velocity[i, 1]
                    svz = t_smoothing_velocity[i, 2]
                    smx = dmath.lerp(svx, dvx, avg)
                    smy = dmath.lerp(svy, dvy, avg)
                    smz = dmath.lerp(svz, dvz, avg)
                    if running:
                        t_smoothing_velocity[i, 0] = smx
                        t_smoothing_velocity[i, 1] = smy
                        t_smoothing_velocity[i, 2] = smz
                        svx = smx
                        svy = smy
                        svz = smz
                    spx = cpx - svx * fdt_i
                    spy = cpy - svy * fdt_i
                    spz = cpz - svz * fdt_i
                    sdv_x = spx - ocp_x
                    sdv_y = spy - ocp_y
                    sdv_z = spz - ocp_z
                    ocp_x = spx
                    ocp_y = spy
                    ocp_z = spz
                    t_inertia_shift[i] = int32(1)

                # --- frame_world store + reset / neg_only snapshots ---
                t_frame_world_position[i, 0] = cwpx
                t_frame_world_position[i, 1] = cwpy
                t_frame_world_position[i, 2] = cwpz
                t_frame_world_rotation[i, 0] = cwrx
                t_frame_world_rotation[i, 1] = cwry
                t_frame_world_rotation[i, 2] = cwrz
                t_frame_world_rotation[i, 3] = cwrw
                t_frame_world_scale[i, 0] = csx
                t_frame_world_scale[i, 1] = csy
                t_frame_world_scale[i, 2] = csz
                if reset:
                    t_old_component_world_position[i, 0] = cpx
                    t_old_component_world_position[i, 1] = cpy
                    t_old_component_world_position[i, 2] = cpz
                    t_old_component_world_rotation[i, 0] = crx
                    t_old_component_world_rotation[i, 1] = cry
                    t_old_component_world_rotation[i, 2] = crz
                    t_old_component_world_rotation[i, 3] = crw
                    t_old_component_world_scale[i, 0] = csx
                    t_old_component_world_scale[i, 1] = csy
                    t_old_component_world_scale[i, 2] = csz
                    ocp_x = cpx
                    ocp_y = cpy
                    ocp_z = cpz
                    ocr_x = crx
                    ocr_y = cry
                    ocr_z = crz
                    ocr_w = crw
                if reset or (teleport and (not reset)):
                    t_old_frame_world_position[i, 0] = cwpx
                    t_old_frame_world_position[i, 1] = cwpy
                    t_old_frame_world_position[i, 2] = cwpz
                    t_old_frame_world_rotation[i, 0] = cwrx
                    t_old_frame_world_rotation[i, 1] = cwry
                    t_old_frame_world_rotation[i, 2] = cwrz
                    t_old_frame_world_rotation[i, 3] = cwrw
                    t_old_frame_world_scale[i, 0] = csx
                    t_old_frame_world_scale[i, 1] = csy
                    t_old_frame_world_scale[i, 2] = csz
                    t_now_world_position[i, 0] = cwpx
                    t_now_world_position[i, 1] = cwpy
                    t_now_world_position[i, 2] = cwpz
                    t_now_world_rotation[i, 0] = cwrx
                    t_now_world_rotation[i, 1] = cwry
                    t_now_world_rotation[i, 2] = cwrz
                    t_now_world_rotation[i, 3] = cwrw
                    t_old_world_position[i, 0] = cwpx
                    t_old_world_position[i, 1] = cwpy
                    t_old_world_position[i, 2] = cwpz
                    t_old_world_rotation[i, 0] = cwrx
                    t_old_world_rotation[i, 1] = cwry
                    t_old_world_rotation[i, 2] = cwrz
                    t_old_world_rotation[i, 3] = cwrw

                # --- work vars + shift setup ---
                wpx = ocp_x
                wpy = ocp_y
                wpz = ocp_z
                wrx = ocr_x
                wry = ocr_y
                wrz = ocr_z
                wrw = ocr_w
                shv_x = float32(0.0)
                shv_y = float32(0.0)
                shv_z = float32(0.0)
                shr_x = float32(0.0)
                shr_y = float32(0.0)
                shr_z = float32(0.0)
                shr_w = float32(1.0)
                if reset:
                    t_smoothing_velocity[i, 0] = float32(0.0)
                    t_smoothing_velocity[i, 1] = float32(0.0)
                    t_smoothing_velocity[i, 2] = float32(0.0)
                    sdv_x = float32(0.0)
                    sdv_y = float32(0.0)
                    sdv_z = float32(0.0)

                # --- world inertia shift (live teams) ---
                if not reset:
                    shv_x = cpx - ocp_x
                    shv_y = cpy - ocp_y
                    shv_z = cpz - ocp_z
                    iox, ioy, ioz, iow = dmath.quat_inverse(ocr_x, ocr_y, ocr_z, ocr_w)
                    shr_x, shr_y, shr_z, shr_w = dmath.quat_mul(crx, cry, crz, crw, iox, ioy, ioz, iow)
                    msr = float32(0.0)
                    rsr = float32(0.0)
                    keep_now = keep or (t_culling_invisible[i] != 0)
                    if keep_now:
                        movement_shift = float32(1.0)
                    else:
                        movement_shift = float32(1.0) - t_world_inertia[i]
                    rotation_shift = movement_shift
                    if movement_shift > EPSILON:
                        t_inertia_shift[i] = int32(1)
                        msr = movement_shift
                        rsr = rotation_shift
                        wpx = dmath.lerp(wpx, cpx, movement_shift)
                        wpy = dmath.lerp(wpy, cpy, movement_shift)
                        wpz = dmath.lerp(wpz, cpz, movement_shift)
                        wrx, wry, wrz, wrw = dmath.quat_slerp(
                            wrx, wry, wrz, wrw, crx, cry, crz, crw, rotation_shift)
                    movement_limit = float64(t_movement_speed_limit[i]) * csr_ratio
                    rotation_limit = t_rotation_speed_limit[i]
                    dvx = cpx - wpx
                    dvy = cpy - wpy
                    dvz = cpz - wpz
                    dang = dmath.quat_angle(wrx, wry, wrz, wrw, crx, cry, crz, crw)
                    fdt_l = t_frame_dt[i]
                    if fdt_l > float32(0.0):
                        frame_speed = dmath.length3(dvx, dvy, dvz) / fdt_l
                        frame_rot_speed = (dang * RAD2DEG) / fdt_l
                    else:
                        frame_speed = float32(0.0)
                        frame_rot_speed = float32(0.0)
                    over_move = (float64(frame_speed) > movement_limit) and \
                        (t_movement_speed_limit[i] >= float32(0.0))
                    if over_move:
                        t_inertia_shift[i] = int32(1)
                        denom_fs = frame_speed if frame_speed > float32(0.0) else float32(1.0)
                        mlr = dmath.saturate((frame_speed - float32(movement_limit)) / denom_fs)
                    else:
                        mlr = float32(0.0)
                    msr = msr + (float32(1.0) - msr) * mlr
                    if over_move:
                        wpx = dmath.lerp(wpx, cpx, mlr)
                        wpy = dmath.lerp(wpy, cpy, mlr)
                        wpz = dmath.lerp(wpz, cpz, mlr)
                    over_rot = (frame_rot_speed > rotation_limit) and (rotation_limit >= float32(0.0))
                    if over_rot:
                        t_inertia_shift[i] = int32(1)
                        denom_frs = frame_rot_speed if frame_rot_speed > float32(0.0) else float32(1.0)
                        rlr = dmath.saturate((frame_rot_speed - rotation_limit) / denom_frs)
                    else:
                        rlr = float32(0.0)
                    rsr = rsr + (float32(1.0) - rsr) * rlr
                    if over_rot:
                        wrx, wry, wrz, wrw = dmath.quat_slerp(
                            wrx, wry, wrz, wrw, crx, cry, crz, crw, rlr)
                    osr = float64(0.0)
                    skip = t_skip_count[i]
                    scaled_dt = fdt_l * t_now_time_scale[i]
                    if (skip > 0) and (scaled_dt > float32(0.0)):
                        sr = float64(skip) * float64(sim_dt) / float64(scaled_dt)
                        if sr < float64(0.0):
                            sr = float64(0.0)
                        elif sr > float64(1.0):
                            sr = float64(1.0)
                        osr = osr + (float64(1.0) - osr) * sr
                    vw = t_velocity_weight[i]
                    if vw < float32(1.0):
                        osr = osr + (float64(1.0) - osr) * (float64(1.0) - float64(vw))
                    nts = t_now_time_scale[i]
                    if nts < float32(1.0):
                        osr = osr + (float64(1.0) - osr) * (float64(1.0) - float64(nts))
                    msr_final = float64(msr)
                    rsr_final = float64(rsr)
                    if osr > float64(0.0):
                        t_inertia_shift[i] = int32(1)
                        msr_final = msr_final + (float64(1.0) - msr_final) * osr
                        osr_f32 = float32(osr)
                        wpx = dmath.lerp(wpx, cpx, osr_f32)
                        wpy = dmath.lerp(wpy, cpy, osr_f32)
                        wpz = dmath.lerp(wpz, cpz, osr_f32)
                        rsr_final = rsr_final + (float64(1.0) - rsr_final) * osr
                        wrx, wry, wrz, wrw = dmath.quat_slerp(
                            wrx, wry, wrz, wrw, crx, cry, crz, crw, osr_f32)
                    if t_inertia_shift[i] != 0:
                        vecx = float64(shv_x) * msr_final + float64(adv_x) + float64(sdv_x)
                        vecy = float64(shv_y) * msr_final + float64(adv_y) + float64(sdv_y)
                        vecz = float64(shv_z) * msr_final + float64(adv_z) + float64(sdv_z)
                        rqx, rqy, rqz, rqw = dmath.quat_slerp(
                            float32(0.0), float32(0.0), float32(0.0), float32(1.0),
                            shr_x, shr_y, shr_z, shr_w, float32(rsr_final))
                        rqx, rqy, rqz, rqw = dmath.quat_mul(adr_x, adr_y, adr_z, adr_w, rqx, rqy, rqz, rqw)
                        t_frame_component_shift_vector[i, 0] = float32(vecx)
                        t_frame_component_shift_vector[i, 1] = float32(vecy)
                        t_frame_component_shift_vector[i, 2] = float32(vecz)
                        t_frame_component_shift_rotation[i, 0] = rqx
                        t_frame_component_shift_rotation[i, 1] = rqy
                        t_frame_component_shift_rotation[i, 2] = rqz
                        t_frame_component_shift_rotation[i, 3] = rqw
                        oc_x = t_old_component_world_position[i, 0]
                        oc_y = t_old_component_world_position[i, 1]
                        oc_z = t_old_component_world_position[i, 2]
                        rlx1, rly1, rlz1 = dmath.quat_rotate(
                            rqx, rqy, rqz, rqw, t_old_frame_world_position[i, 0] - oc_x,
                            t_old_frame_world_position[i, 1] - oc_y, t_old_frame_world_position[i, 2] - oc_z)
                        t_old_frame_world_position[i, 0] = float32(float64(rlx1 + oc_x) + vecx)
                        t_old_frame_world_position[i, 1] = float32(float64(rly1 + oc_y) + vecy)
                        t_old_frame_world_position[i, 2] = float32(float64(rlz1 + oc_z) + vecz)
                        oqx, oqy, oqz, oqw = dmath.quat_mul(
                            rqx, rqy, rqz, rqw, t_old_frame_world_rotation[i, 0],
                            t_old_frame_world_rotation[i, 1], t_old_frame_world_rotation[i, 2],
                            t_old_frame_world_rotation[i, 3])
                        t_old_frame_world_rotation[i, 0] = oqx
                        t_old_frame_world_rotation[i, 1] = oqy
                        t_old_frame_world_rotation[i, 2] = oqz
                        t_old_frame_world_rotation[i, 3] = oqw
                        rlx2, rly2, rlz2 = dmath.quat_rotate(
                            rqx, rqy, rqz, rqw, t_now_world_position[i, 0] - oc_x,
                            t_now_world_position[i, 1] - oc_y, t_now_world_position[i, 2] - oc_z)
                        t_now_world_position[i, 0] = float32(float64(rlx2 + oc_x) + vecx)
                        t_now_world_position[i, 1] = float32(float64(rly2 + oc_y) + vecy)
                        t_now_world_position[i, 2] = float32(float64(rlz2 + oc_z) + vecz)
                        nqx, nqy, nqz, nqw = dmath.quat_mul(
                            rqx, rqy, rqz, rqw, t_now_world_rotation[i, 0],
                            t_now_world_rotation[i, 1], t_now_world_rotation[i, 2],
                            t_now_world_rotation[i, 3])
                        t_now_world_rotation[i, 0] = nqx
                        t_now_world_rotation[i, 1] = nqy
                        t_now_world_rotation[i, 2] = nqz
                        t_now_world_rotation[i, 3] = nqw
                    else:
                        t_frame_component_shift_vector[i, 0] = shv_x
                        t_frame_component_shift_vector[i, 1] = shv_y
                        t_frame_component_shift_vector[i, 2] = shv_z
                        t_frame_component_shift_rotation[i, 0] = shr_x
                        t_frame_component_shift_rotation[i, 1] = shr_y
                        t_frame_component_shift_rotation[i, 2] = shr_z
                        t_frame_component_shift_rotation[i, 3] = shr_w
                if reset:
                    t_frame_component_shift_vector[i, 0] = float32(0.0)
                    t_frame_component_shift_vector[i, 1] = float32(0.0)
                    t_frame_component_shift_vector[i, 2] = float32(0.0)
                    t_frame_component_shift_rotation[i, 0] = float32(0.0)
                    t_frame_component_shift_rotation[i, 1] = float32(0.0)
                    t_frame_component_shift_rotation[i, 2] = float32(0.0)
                    t_frame_component_shift_rotation[i, 3] = float32(1.0)

                # --- moving speed / direction ---
                mvx = cpx - wpx
                mvy = cpy - wpy
                mvz = cpz - wpz
                mlen = dmath.length3(mvx, mvy, mvz)
                fdt_m = t_frame_dt[i]
                if fdt_m > float32(0.0):
                    speed = mlen / fdt_m
                else:
                    speed = float32(0.0)
                nts_m = t_now_time_scale[i]
                if nts_m > float32(1e-6):
                    speed = speed * (float32(1.0) / nts_m)
                else:
                    speed = float32(0.0)
                t_frame_moving_speed[i] = speed
                if mlen > float32(1e-6):
                    t_frame_moving_direction[i, 0] = mvx / mlen
                    t_frame_moving_direction[i, 1] = mvy / mlen
                    t_frame_moving_direction[i, 2] = mvz / mlen
                else:
                    t_frame_moving_direction[i, 0] = float32(0.0)
                    t_frame_moving_direction[i, 1] = float32(0.0)
                    t_frame_moving_direction[i, 2] = float32(0.0)

                # --- stabilize ---
                if (t_reset_pending[i] != 0) or (t_time_reset[i] != 0):
                    if t_stablization_time[i] > float32(1e-6):
                        wgt = float32(0.0)
                    else:
                        wgt = float32(1.0)
                    t_velocity_weight[i] = wgt
                    t_blend_weight[i] = wgt

                # --- select_team_wind ---
                old_count = t_wind_count[i]
                for oc in range(WIND_ZONE_SLOTS):
                    if oc < old_count:
                        old_zid[oc] = t_wind_zone_id[i, oc]
                        old_wt[oc] = t_wind_time[i, oc]
                count = 0
                if n_zones > 0 and t_wind_influence[i] > EPSILON:
                    cx64 = float64(cwpx)
                    cy64 = float64(cwpy)
                    cz64 = float64(cwpz)
                    min_volume = INF
                    addition_count = 0
                    latest_valid = False
                    latest_id = int32(0)
                    for zi in range(n_zones):
                        is_add = z_is_addition[zi] != 0
                        if is_add and addition_count >= 3:
                            continue
                        mode = z_mode[zi]
                        zvol = z_zone_volume[zi]
                        wm = z_world_to_local[zi]
                        lxx = wm[0, 0] * cx64 + wm[0, 1] * cy64 + wm[0, 2] * cz64 + wm[0, 3]
                        lyy = wm[1, 0] * cx64 + wm[1, 1] * cy64 + wm[1, 2] * cz64 + wm[1, 3]
                        lzz = wm[2, 0] * cx64 + wm[2, 1] * cy64 + wm[2, 2] * cz64 + wm[2, 3]
                        llen = math.sqrt(lxx * lxx + lyy * lyy + lzz * lzz)
                        skip_zone = False
                        if mode == ZONE_BOX:
                            if abs(lxx) * float64(2.0) > float64(z_size[zi, 0]) or \
                                    abs(lyy) * float64(2.0) > float64(z_size[zi, 1]) or \
                                    abs(lzz) * float64(2.0) > float64(z_size[zi, 2]):
                                skip_zone = True
                        elif mode == ZONE_SPHERE_DIR or mode == ZONE_SPHERE_RADIAL:
                            if llen > float64(z_size[zi, 0]):
                                skip_zone = True
                        if skip_zone:
                            continue
                        if (not is_add) and (zvol > min_volume):
                            continue
                        dirx = z_world_direction[zi, 0]
                        diry = z_world_direction[zi, 1]
                        dirz = z_world_direction[zi, 2]
                        zmain = z_main[zi]
                        if mode == ZONE_SPHERE_RADIAL:
                            if llen <= float64(1e-6):
                                continue
                            vx64 = cx64 - float64(z_world_position[zi, 0])
                            vy64 = cy64 - float64(z_world_position[zi, 1])
                            vz64 = cz64 - float64(z_world_position[zi, 2])
                            vlen = math.sqrt(vx64 * vx64 + vy64 * vy64 + vz64 * vz64)
                            dirx = float32(vx64 / vlen)
                            diry = float32(vy64 / vlen)
                            dirz = float32(vz64 / vlen)
                            depth = llen / float64(z_size[zi, 0])
                            if depth < float64(0.0):
                                depth = float64(0.0)
                            elif depth > float64(1.0):
                                depth = float64(1.0)
                            zmain = zmain * dmath.evaluate_team_lut_clamp01(
                                z_attenuation_lut, zi, float32(depth))
                        zid = z_zone_id[zi]
                        t_prev = -WIND_MAX_TIME
                        for oi in range(WIND_ZONE_SLOTS):
                            if oi < old_count and old_zid[oi] == zid:
                                t_prev = old_wt[oi]
                        zturb = z_turbulence[zi]
                        if is_add:
                            res_zone_id[count] = zid
                            res_time[count] = t_prev
                            res_main[count] = zmain
                            res_dx[count] = dirx
                            res_dy[count] = diry
                            res_dz[count] = dirz
                            res_turb[count] = zturb
                            count += 1
                            addition_count += 1
                        else:
                            if latest_valid:
                                w = 0
                                for r in range(count):
                                    if res_zone_id[r] != latest_id:
                                        res_zone_id[w] = res_zone_id[r]
                                        res_time[w] = res_time[r]
                                        res_main[w] = res_main[r]
                                        res_dx[w] = res_dx[r]
                                        res_dy[w] = res_dy[r]
                                        res_dz[w] = res_dz[r]
                                        res_turb[w] = res_turb[r]
                                        w += 1
                                count = w
                            res_zone_id[count] = zid
                            res_time[count] = t_prev
                            res_main[count] = zmain
                            res_dx[count] = dirx
                            res_dy[count] = diry
                            res_dz[count] = dirz
                            res_turb[count] = zturb
                            count += 1
                            min_volume = zvol
                            latest_id = zid
                            latest_valid = True
                final = count if count < WIND_ZONE_SLOTS else WIND_ZONE_SLOTS
                t_wind_count[i] = int8(final)
                for s in range(final):
                    t_wind_zone_id[i, s] = res_zone_id[s]
                    t_wind_time[i, s] = res_time[s]
                    t_wind_main[i, s] = res_main[s]
                    t_wind_direction[i, s, 0] = res_dx[s]
                    t_wind_direction[i, s, 1] = res_dy[s]
                    t_wind_direction[i, s, 2] = res_dz[s]
                    t_wind_zone_turbulence[i, s] = res_turb[s]
                    dqx, dqy, dqz, dqw = dmath.axis_quaternion(res_dx[s], res_dy[s], res_dz[s])
                    t_wind_dirq[i, s, 0] = dqx
                    t_wind_dirq[i, s, 1] = dqy
                    t_wind_dirq[i, s, 2] = dqz
                    t_wind_dirq[i, s, 3] = dqw
            i += stride
    grid.sync()

    # P2 particles.frame_pre (per-particle; reset snapshot / negative-scale / inertia shift)
    if phase_mask & PHASE_PARTICLES_PRE:
        p = tid
        while p < num_particles:
            if team_frame_mask(t_enabled, t_valid, t_cws, p_team[p]):
                do_particles_frame_pre(p, p_team, p_positions, p_rotations, p_next_positions,
                                       p_old_positions, p_old_rotations, p_base_positions,
                                       p_base_rotations, p_old_anim_positions, p_old_anim_rotations,
                                       p_velocity_positions, p_display_positions, p_velocities,
                                       p_real_velocities, p_friction, p_static_friction,
                                       p_collision_normals, t_reset_pending,
                                       t_negative_scale_teleport, t_negative_scale_matrix,
                                       t_inertia_shift, t_frame_component_shift_vector,
                                       t_frame_component_shift_rotation,
                                       t_old_component_world_position)
            p += stride
    grid.sync()

    # K0 collider.frame_pre (per-collider; frame_team_mask gate, enable rising-edge reset)
    if phase_mask & PHASE_COLLIDER_PRE:
        ci = tid
        while ci < num_colliders:
            if team_frame_mask(t_enabled, t_valid, t_cws, c_team[ci]):
                do_collider_frame_pre(ci, c_team, c_enabled, c_enabled_prev, c_active,
                                      c_input_scales, c_input_positions, c_input_rotations, c_center,
                                      c_frame_pos, c_frame_rot, c_frame_scl,
                                      c_old_frame_pos, c_old_frame_rot, c_now_pos, c_now_rot,
                                      c_old_pos, c_old_rot,
                                      t_reset_pending, t_negative_scale_teleport,
                                      t_negative_scale_matrix, t_negative_scale_change,
                                      t_inertia_shift, t_frame_component_shift_vector,
                                      t_frame_component_shift_rotation,
                                      t_old_component_world_position)
            ci += stride
    grid.sync()

    # ----- SUBSTEP LOOP (phases added in dependency order as ported) -----
    n_tether = st_tether_particle.shape[0]
    n_move = st_move_particle.shape[0]
    n_fixed = st_fixed_particle.shape[0]
    n_spring = st_spring_particle.shape[0]
    n_motion = st_motion_particle.shape[0]
    n_bending = st_bending_team.shape[0]
    n_baseline = baseline_entries.shape[0]
    num_fk_levels = fk_yes_offsets.shape[0] - 1
    n_angle_buffered = st_angle_buffered_particle.shape[0]
    num_angle_passes = angle_pass_offsets.shape[0] - 1
    for _k in range(sub_begin, sub_end):
        # --- S1 team_time.step_update (per-team, first substep stage) ---
        if phase_mask & PHASE_TEAM_STEP:
            i = tid
            while i < num_teams:
                if team_frame_mask(t_enabled, t_valid, t_cws, i) and t_update_count[i] > _k:
                    do_step_update(i, sim_dt,
                                   t_now_update, t_time, t_frame_old, t_frame_interpolation,
                                   t_now_world_position, t_now_world_rotation,
                                   t_old_world_position, t_old_world_rotation,
                                   t_old_frame_world_position, t_old_frame_world_rotation,
                                   t_old_frame_world_scale, t_frame_world_position,
                                   t_frame_world_rotation, t_frame_world_scale,
                                   t_step_vector, t_step_rotation,
                                   t_step_move_inertia_ratio, t_step_rotation_inertia_ratio,
                                   t_local_inertia, t_local_movement_speed_limit,
                                   t_local_rotation_speed_limit,
                                   t_inertia_vector, t_inertia_rotation,
                                   t_angular_velocity, t_rotation_axis,
                                   t_init_scale, t_scale_ratio,
                                   t_gravity_direction, t_gravity_dot,
                                   t_init_local_gravity_direction, t_negative_scale_direction,
                                   t_gravity, t_gravity_falloff, t_gravity_ratio,
                                   t_velocity_weight, t_stablization_time, t_blend_weight,
                                   t_blend_weight_param, t_distance_weight,
                                   t_wind_moving, t_frame_moving_speed, t_moving_wind_main,
                                   t_frame_moving_direction, t_moving_wind_direction,
                                   t_moving_wind_dirq,
                                   t_wind_main, t_wind_frequency, t_wind_count, t_wind_time,
                                   t_moving_wind_time)
                i += stride
        grid.sync()

        # --- S2 collider.start_step (per-collider; build sphere/capsule/plane work + AABB) ---
        if phase_mask & PHASE_COLLIDER_START:
            ci = tid
            while ci < num_colliders:
                cm = c_team[ci]
                if team_frame_mask(t_enabled, t_valid, t_cws, cm) and t_update_count[cm] > _k \
                        and c_active[ci] != 0:
                    do_collider_start_step(ci, c_team, c_kind, c_size, c_axis, c_aligned,
                                           c_frame_pos, c_frame_rot, c_frame_scl,
                                           c_old_frame_pos, c_old_frame_rot, c_now_pos, c_now_rot,
                                           c_old_pos, c_old_rot, c_work_rot, c_work_inv_old_rot,
                                           c_work_radius, c_work_old_pos, c_work_next_pos,
                                           c_work_aabb_min, c_work_aabb_max,
                                           t_frame_interpolation, t_step_move_inertia_ratio,
                                           t_step_rotation_inertia_ratio)
                ci += stride
        grid.sync()

        # --- S3 particles.step_update PASS 1: base interpolation (all sp) + move set ---
        if phase_mask & PHASE_PARTICLES_STEP:
            p = tid
            while p < num_particles:
                mt = p_team[p]
                if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
                    t = t_frame_interpolation[mt]
                    bx = dmath.lerp(p_old_anim_positions[p, 0], p_positions[p, 0], t)
                    by = dmath.lerp(p_old_anim_positions[p, 1], p_positions[p, 1], t)
                    bz = dmath.lerp(p_old_anim_positions[p, 2], p_positions[p, 2], t)
                    qx, qy, qz, qw = dmath.quat_slerp(
                        p_old_anim_rotations[p, 0], p_old_anim_rotations[p, 1],
                        p_old_anim_rotations[p, 2], p_old_anim_rotations[p, 3],
                        p_rotations[p, 0], p_rotations[p, 1], p_rotations[p, 2], p_rotations[p, 3], t)
                    p_base_positions[p, 0] = bx
                    p_base_positions[p, 1] = by
                    p_base_positions[p, 2] = bz
                    p_step_basic_positions[p, 0] = bx
                    p_step_basic_positions[p, 1] = by
                    p_step_basic_positions[p, 2] = bz
                    p_base_rotations[p, 0] = qx
                    p_base_rotations[p, 1] = qy
                    p_base_rotations[p, 2] = qz
                    p_base_rotations[p, 3] = qw
                    p_step_basic_rotations[p, 0] = qx
                    p_step_basic_rotations[p, 1] = qy
                    p_step_basic_rotations[p, 2] = qz
                    p_step_basic_rotations[p, 3] = qw
                p += stride

            e = tid
            while e < n_move:
                mt = st_move_team[e]
                if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
                    pmi = st_move_particle[e]
                    depth = p_depth[pmi]
                    ox = p_old_positions[pmi, 0]
                    oy = p_old_positions[pmi, 1]
                    oz = p_old_positions[pmi, 2]

                    inertia_depth = t_depth_inertia[mt] * (float32(1.0) - depth * depth)
                    ivx = dmath.lerp(t_inertia_vector[mt, 0], t_step_vector[mt, 0], inertia_depth)
                    ivy = dmath.lerp(t_inertia_vector[mt, 1], t_step_vector[mt, 1], inertia_depth)
                    ivz = dmath.lerp(t_inertia_vector[mt, 2], t_step_vector[mt, 2], inertia_depth)
                    irx, iry, irz, irw = dmath.quat_slerp(
                        t_inertia_rotation[mt, 0], t_inertia_rotation[mt, 1],
                        t_inertia_rotation[mt, 2], t_inertia_rotation[mt, 3],
                        t_step_rotation[mt, 0], t_step_rotation[mt, 1],
                        t_step_rotation[mt, 2], t_step_rotation[mt, 3], inertia_depth)
                    owx = t_old_world_position[mt, 0]
                    owy = t_old_world_position[mt, 1]
                    owz = t_old_world_position[mt, 2]
                    lx = ox - owx
                    ly = oy - owy
                    lz = oz - owz
                    rlx, rly, rlz = dmath.quat_rotate(irx, iry, irz, irw, lx, ly, lz)
                    lx = rlx + ivx
                    ly = rly + ivy
                    lz = rlz + ivz
                    wx = owx + lx
                    wy = owy + ly
                    wz = owz + lz
                    nextx = wx
                    nexty = wy
                    nextz = wz
                    velposx = ox + (wx - ox)
                    velposy = oy + (wy - oy)
                    velposz = oz + (wz - oz)

                    vx, vy, vz = dmath.quat_rotate(irx, iry, irz, irw,
                                                   p_velocities[pmi, 0], p_velocities[pmi, 1],
                                                   p_velocities[pmi, 2])
                    vw = t_velocity_weight[mt]
                    vx = vx * vw
                    vy = vy * vw
                    vz = vz * vw
                    damping = dmath.evaluate_team_lut_clamp01(t_damping_lut, mt, depth)
                    damp = dmath.saturate(float32(1.0) - damping * power2)
                    vx = vx * damp
                    vy = vy * damp
                    vz = vz * damp

                    fm = t_force_mode[mt]
                    change = (fm == FORCE_VELOCITY_CHANGE) or (fm == FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH)
                    if change:
                        vx = float32(0.0)
                        vy = float32(0.0)
                        vz = float32(0.0)

                    g = t_gravity[mt] * t_gravity_ratio[mt]
                    fx = t_gravity_direction[mt, 0] * g
                    fy = t_gravity_direction[mt, 1] * g
                    fz = t_gravity_direction[mt, 2] * g
                    mass = dmath.calc_mass(depth)
                    with_depth = (fm == FORCE_VELOCITY_ADD) or (fm == FORCE_VELOCITY_CHANGE)
                    without_depth = (fm == FORCE_VELOCITY_ADD_WITHOUT_DEPTH) or (fm == FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH)
                    if with_depth:
                        fx = fx + t_impact_force[mt, 0] / mass
                        fy = fy + t_impact_force[mt, 1] / mass
                        fz = fz + t_impact_force[mt, 2] / mass
                    if without_depth:
                        fx = fx + t_impact_force[mt, 0]
                        fy = fy + t_impact_force[mt, 1]
                        fz = fz + t_impact_force[mt, 2]

                    # wind force (inline; do_wind_blend per slot + moving)
                    root = float32(p_vertex_root_local[pmi])
                    seed = float32(t_wind_seed[mt])
                    sync = t_wind_synchronization[mt]
                    wind_position = (seed + float32(1.0)) * float32(4.19230645) \
                        + root * float32(0.0023963) * (float32(1.0) - sync) * float32(100.0)
                    blend = t_wind_blend[mt]
                    turbulence_param = t_wind_turbulence[mt]
                    wfx = float32(0.0)
                    wfy = float32(0.0)
                    wfz = float32(0.0)
                    wc = t_wind_count[mt]
                    for s in range(4):
                        if s < wc:
                            cx, cy, cz = do_wind_blend(
                                t_wind_main[mt, s], t_wind_time[mt, s],
                                t_wind_dirq[mt, s, 0], t_wind_dirq[mt, s, 1],
                                t_wind_dirq[mt, s, 2], t_wind_dirq[mt, s, 3],
                                t_wind_zone_turbulence[mt, s], blend, turbulence_param, wind_position)
                            wfx = wfx + cx
                            wfy = wfy + cy
                            wfz = wfz + cz
                    moving_on = (t_moving_wind_main[mt] > float32(0.01)) and (t_wind_moving[mt] > float32(0.01))
                    if moving_on:
                        cx, cy, cz = do_wind_blend(
                            t_moving_wind_main[mt], t_moving_wind_time[mt],
                            t_moving_wind_dirq[mt, 0], t_moving_wind_dirq[mt, 1],
                            t_moving_wind_dirq[mt, 2], t_moving_wind_dirq[mt, 3],
                            float32(1.0), blend, turbulence_param, wind_position)
                        wfx = wfx + cx
                        wfy = wfy + cy
                        wfz = wfz + cz
                    influence = t_wind_influence[mt] * (float32(1.0) - p_friction[pmi])
                    depth_scale = depth * depth
                    influence = influence * dmath.lerp(float32(1.0), depth_scale, t_wind_depth_weight[mt])
                    fx = fx + wfx * influence
                    fy = fy + wfy * influence
                    fz = fz + wfz * influence

                    sr = t_scale_ratio[mt]
                    fx = fx * sr
                    fy = fy * sr
                    fz = fz * sr

                    vx = vx + fx * sim_dt
                    vy = vy + fy * sim_dt
                    vz = vz + fz * sim_dt
                    nextx = nextx + vx * sim_dt
                    nexty = nexty + vy * sim_dt
                    nextz = nextz + vz * sim_dt

                    p_velocities[pmi, 0] = vx
                    p_velocities[pmi, 1] = vy
                    p_velocities[pmi, 2] = vz
                    p_next_positions[pmi, 0] = nextx
                    p_next_positions[pmi, 1] = nexty
                    p_next_positions[pmi, 2] = nextz
                    p_velocity_positions[pmi, 0] = velposx
                    p_velocity_positions[pmi, 1] = velposy
                    p_velocity_positions[pmi, 2] = velposz
                e += stride
        grid.sync()

        # --- S3 PASS 2: fixed set (next=base) + spring set (limit/elliptic/noise) ---
        if phase_mask & PHASE_PARTICLES_STEP:
            e = tid
            while e < n_fixed:
                ft = st_fixed_team[e]
                if team_frame_mask(t_enabled, t_valid, t_cws, ft) and t_update_count[ft] > _k:
                    pfi = st_fixed_particle[e]
                    p_next_positions[pfi, 0] = p_base_positions[pfi, 0]
                    p_next_positions[pfi, 1] = p_base_positions[pfi, 1]
                    p_next_positions[pfi, 2] = p_base_positions[pfi, 2]
                    p_velocity_positions[pfi, 0] = p_base_positions[pfi, 0]
                    p_velocity_positions[pfi, 1] = p_base_positions[pfi, 1]
                    p_velocity_positions[pfi, 2] = p_base_positions[pfi, 2]
                e += stride

            e = tid
            while e < n_spring:
                st = st_spring_team[e]
                if team_frame_mask(t_enabled, t_valid, t_cws, st) and t_update_count[st] > _k \
                        and t_spring_power[st] > float32(0.0):
                    psi = st_spring_particle[e]
                    bpx = p_base_positions[psi, 0]
                    bpy = p_base_positions[psi, 1]
                    bpz = p_base_positions[psi, 2]
                    n0 = p_next_positions[psi, 0]
                    n1 = p_next_positions[psi, 1]
                    n2 = p_next_positions[psi, 2]
                    vx = n0 - bpx
                    vy = n1 - bpy
                    vz = n2 - bpz
                    dx, dy, dz = dmath.quat_rotate(
                        p_base_rotations[psi, 0], p_base_rotations[psi, 1],
                        p_base_rotations[psi, 2], p_base_rotations[psi, 3],
                        t_normal_axis_vector[st, 0], t_normal_axis_vector[st, 1],
                        t_normal_axis_vector[st, 2])
                    limit = t_spring_limit_distance[st] * t_scale_ratio[st]
                    clampable = limit > float32(1e-8)
                    l = dmath.length3(vx, vy, vz)
                    over = clampable and (l > limit)
                    if over and (l > float32(1e-30)):
                        scale = limit / l
                        vx = vx * scale
                        vy = vy * scale
                        vz = vz * scale
                    ratio = t_spring_normal_limit_ratio[st]
                    elliptic = clampable and (ratio < float32(1.0))
                    ylen = dmath.dot3(dx, dy, dz, vx, vy, vz)
                    vpx = vx - dx * ylen
                    vpy = vy - dy * ylen
                    vpz = vz - dz * ylen
                    xlen = dmath.length3(vpx, vpy, vpz)
                    safe_limit = limit if limit > float32(1e-30) else float32(1.0)
                    tval = dmath.saturate(xlen / safe_limit)
                    y = libdevice.cosf(libdevice.asinf(dmath.clamp1(tval))) * (limit * ratio)
                    exceed = elliptic and (libdevice.fabsf(ylen) > y)
                    if exceed:
                        adjust = (libdevice.fabsf(ylen) - y) * dmath.fsign(ylen)
                        vx = vx - adjust * dx
                        vy = vy - adjust * dy
                        vz = vz - adjust * dz
                    if not clampable:
                        vx = float32(0.0)
                        vy = float32(0.0)
                        vz = float32(0.0)

                    power = t_spring_power[st]
                    noise_param = t_spring_noise[st]
                    if noise_param > float32(0.0):
                        noise_time = (t_time[st] + float32(psi) * float32(49.6198)) * float32(2.4512) \
                            + (n0 + n1 + n2)
                        noise = libdevice.sinf(noise_time) * (noise_param * float32(0.6))
                        power = power + power * noise
                        if power < float32(0.0):
                            power = float32(0.0)
                    vx = vx - vx * power
                    vy = vy - vy * power
                    vz = vz - vz * power
                    p_next_positions[psi, 0] = bpx + vx
                    p_next_positions[psi, 1] = bpy + vy
                    p_next_positions[psi, 2] = bpz + vz
                e += stride
        grid.sync()

        # --- S4 baseline.run FK: block-0 walks all levels (yes then no per level) with
        # __syncthreads() between them instead of a grid.sync (parent step_basic of level L was
        # written at level < L; the intra-block barrier keeps that read-before-write order). Each
        # vertex's write is unique, so the 7-block-stride -> block-0-stride move is bit-exact; max
        # level = 92 < blockDim. Blocks != 0 skip the work and wait at the single exit wall. ---
        for lvl in range(num_fk_levels):
            if bid == 0 and (phase_mask & PHASE_BASELINE) != 0:
                ys = fk_yes_offsets[lvl]
                ye = fk_yes_offsets[lvl + 1]
                i = ys + tid
                while i < ye:
                    v = fk_yes[i]
                    vt = p_team[v]
                    if team_frame_mask(t_enabled, t_valid, t_cws, vt) and t_update_count[vt] > _k \
                            and t_animation_pose_ratio[vt] <= float32(0.99):
                        par = fk_yes_parent[i]
                        sr = t_scale_ratio[vt]
                        scx = t_init_scale[vt, 0] * sr
                        scy = t_init_scale[vt, 1] * sr
                        scz = t_init_scale[vt, 2] * sr
                        lsx = (p_vertex_local_positions[v, 0] * t_negative_scale_direction[vt, 0]) * scx
                        lsy = (p_vertex_local_positions[v, 1] * t_negative_scale_direction[vt, 1]) * scy
                        lsz = (p_vertex_local_positions[v, 2] * t_negative_scale_direction[vt, 2]) * scz
                        prx = p_step_basic_rotations[par, 0]
                        pry = p_step_basic_rotations[par, 1]
                        prz = p_step_basic_rotations[par, 2]
                        prw = p_step_basic_rotations[par, 3]
                        rx, ry, rz = dmath.quat_rotate(prx, pry, prz, prw, lsx, lsy, lsz)
                        p_step_basic_positions[v, 0] = rx + p_step_basic_positions[par, 0]
                        p_step_basic_positions[v, 1] = ry + p_step_basic_positions[par, 1]
                        p_step_basic_positions[v, 2] = rz + p_step_basic_positions[par, 2]
                        lrx = p_vertex_local_rotations[v, 0] * t_negative_scale_quaternion[vt, 0]
                        lry = p_vertex_local_rotations[v, 1] * t_negative_scale_quaternion[vt, 1]
                        lrz = p_vertex_local_rotations[v, 2] * t_negative_scale_quaternion[vt, 2]
                        lrw = p_vertex_local_rotations[v, 3] * t_negative_scale_quaternion[vt, 3]
                        qx, qy, qz, qw = dmath.quat_mul(prx, pry, prz, prw, lrx, lry, lrz, lrw)
                        p_step_basic_rotations[v, 0] = qx
                        p_step_basic_rotations[v, 1] = qy
                        p_step_basic_rotations[v, 2] = qz
                        p_step_basic_rotations[v, 3] = qw
                    i += bdim
            cuda.syncthreads()
            # oracle applies each level's yes (skin from parent) BEFORE its no (negative-
            # scale root flip); a root can be both a parent here and a no-entry, so the
            # yes reads must complete before the no writes.
            if bid == 0 and (phase_mask & PHASE_BASELINE) != 0:
                ns = fk_no_offsets[lvl]
                ne = fk_no_offsets[lvl + 1]
                i = ns + tid
                while i < ne:
                    v = fk_no[i]
                    vt = p_team[v]
                    if team_frame_mask(t_enabled, t_valid, t_cws, vt) and t_update_count[vt] > _k \
                            and t_animation_pose_ratio[vt] <= float32(0.99) \
                            and t_is_negative_scale[vt] != 0:
                        rox = p_step_basic_rotations[v, 0]
                        roy = p_step_basic_rotations[v, 1]
                        roz = p_step_basic_rotations[v, 2]
                        row = p_step_basic_rotations[v, 3]
                        nx, ny, nz = dmath.quat_to_normal(rox, roy, roz, row)
                        dy = t_negative_scale_direction[vt, 1]
                        dz = t_negative_scale_direction[vt, 2]
                        nnx = nx * dy
                        nny = ny * dy
                        nnz = nz * dy
                        tx, ty, tz = dmath.quat_to_tangent(rox, roy, roz, row)
                        ttx = tx * dz
                        tty = ty * dz
                        ttz = tz * dz
                        lqx, lqy, lqz, lqw = dmath.look_rotation(ttx, tty, ttz, nnx, nny, nnz)
                        p_step_basic_rotations[v, 0] = lqx
                        p_step_basic_rotations[v, 1] = lqy
                        p_step_basic_rotations[v, 2] = lqz
                        p_step_basic_rotations[v, 3] = lqw
                    i += bdim
            cuda.syncthreads()
        grid.sync()
        # baseline apply (animation_pose_ratio blend) stays a multi-block grid-stride pass over
        # all entries; the grid.sync above made block 0's FK writes globally visible first.
        if phase_mask & PHASE_BASELINE:
            i = tid
            while i < n_baseline:
                v = baseline_entries[i]
                vt = p_team[v]
                apr = t_animation_pose_ratio[vt]
                if team_frame_mask(t_enabled, t_valid, t_cws, vt) and t_update_count[vt] > _k \
                        and apr <= float32(0.99) and apr > EPSILON:
                    p_step_basic_positions[v, 0] = dmath.lerp(p_step_basic_positions[v, 0],
                                                             p_base_positions[v, 0], apr)
                    p_step_basic_positions[v, 1] = dmath.lerp(p_step_basic_positions[v, 1],
                                                             p_base_positions[v, 1], apr)
                    p_step_basic_positions[v, 2] = dmath.lerp(p_step_basic_positions[v, 2],
                                                             p_base_positions[v, 2], apr)
                    qx, qy, qz, qw = dmath.quat_slerp(
                        p_step_basic_rotations[v, 0], p_step_basic_rotations[v, 1],
                        p_step_basic_rotations[v, 2], p_step_basic_rotations[v, 3],
                        p_base_rotations[v, 0], p_base_rotations[v, 1],
                        p_base_rotations[v, 2], p_base_rotations[v, 3], apr)
                    p_step_basic_rotations[v, 0] = qx
                    p_step_basic_rotations[v, 1] = qy
                    p_step_basic_rotations[v, 2] = qz
                    p_step_basic_rotations[v, 3] = qw
                i += stride
        grid.sync()

        if phase_mask & PHASE_TETHER:
            e = tid
            while e < n_tether:
                tm = st_tether_team[e]
                if team_frame_mask(t_enabled, t_valid, t_cws, tm) and t_update_count[tm] > _k:
                    do_tether(e, st_tether_particle, p_team, p_next_positions,
                              p_velocity_positions, p_step_basic_positions, p_vertex_root,
                              t_tether_compression)
                e += stride
        grid.sync()

        # --- S6 distance.run (first occurrence): Jacobi gather -> apply ---
        if phase_mask & PHASE_DISTANCE_A:
            p = tid
            while p < num_particles:
                mt = p_team[p]
                if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
                    do_distance_gather(p, p_team, p_next_positions, p_base_positions, p_depth,
                                       p_friction, p_attr_move, t_is_spring, t_animation_pose_ratio,
                                       t_init_scale, t_scale_ratio, t_distance_lut, power1,
                                       csr_distance_offsets, csr_distance_order,
                                       st_distance_target, st_distance_rest, sc_dcorr)
                p += stride
        grid.sync()
        if phase_mask & PHASE_DISTANCE_A:
            p = tid
            while p < num_particles:
                mt = p_team[p]
                if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
                    p_next_positions[p, 0] += sc_dcorr[p, 0]
                    p_next_positions[p, 1] += sc_dcorr[p, 1]
                    p_next_positions[p, 2] += sc_dcorr[p, 2]
                    p_velocity_positions[p, 0] += sc_dcorr[p, 0] * DISTANCE_VELOCITY_ATTENUATION
                    p_velocity_positions[p, 1] += sc_dcorr[p, 1] * DISTANCE_VELOCITY_ATTENUATION
                    p_velocity_positions[p, 2] += sc_dcorr[p, 2] * DISTANCE_VELOCITY_ATTENUATION
                p += stride
        grid.sync()

        # --- S7 angle.run: buffer segment -> 3 iterations x P (level,rank) passes ---
        # Buffer a: for baseline entries whose team needs angle, snapshot step_basic_rotations
        # into albuf_rotation (oracle does ALL baseline entries incl. fixed roots). Buffer b:
        # for angle_buffered vertices (parent always >=0) build the limit local frame and the
        # restoration target vector. a/b touch disjoint fields so they need no barrier between.
        if phase_mask & PHASE_ANGLE:
            i = tid
            while i < n_baseline:
                v = baseline_entries[i]
                vt = p_team[v]
                if team_frame_mask(t_enabled, t_valid, t_cws, vt) and t_update_count[vt] > _k \
                        and (t_angle_use_limit[vt] != 0 or t_angle_use_restoration[vt] != 0):
                    p_albuf_rotation[v, 0] = p_step_basic_rotations[v, 0]
                    p_albuf_rotation[v, 1] = p_step_basic_rotations[v, 1]
                    p_albuf_rotation[v, 2] = p_step_basic_rotations[v, 2]
                    p_albuf_rotation[v, 3] = p_step_basic_rotations[v, 3]
                i += stride
            i = tid
            while i < n_angle_buffered:
                v = st_angle_buffered_particle[i]
                vt = p_team[v]
                if team_frame_mask(t_enabled, t_valid, t_cws, vt) and t_update_count[vt] > _k \
                        and (t_angle_use_limit[vt] != 0 or t_angle_use_restoration[vt] != 0):
                    par = p_vertex_parent[v]
                    bvx = p_step_basic_positions[v, 0] - p_step_basic_positions[par, 0]
                    bvy = p_step_basic_positions[v, 1] - p_step_basic_positions[par, 1]
                    bvz = p_step_basic_positions[v, 2] - p_step_basic_positions[par, 2]
                    if t_angle_use_limit[vt] != 0:
                        dvx = p_next_positions[v, 0] - p_next_positions[par, 0]
                        dvy = p_next_positions[v, 1] - p_next_positions[par, 1]
                        dvz = p_next_positions[v, 2] - p_next_positions[par, 2]
                        avlen = dmath.length3(dvx, dvy, dvz)
                        bvlen = dmath.length3(bvx, bvy, bvz)
                        if avlen < EPSILON or bvlen < EPSILON:
                            p_albuf_length[v] = float32(0.0)
                            p_albuf_local_pos[v, 0] = float32(0.0)
                            p_albuf_local_pos[v, 1] = float32(0.0)
                            p_albuf_local_pos[v, 2] = float32(0.0)
                            p_albuf_local_rot[v, 0] = float32(0.0)
                            p_albuf_local_rot[v, 1] = float32(0.0)
                            p_albuf_local_rot[v, 2] = float32(0.0)
                            p_albuf_local_rot[v, 3] = float32(1.0)
                        else:
                            safe_bv = bvlen if bvlen > float32(1e-30) else float32(1.0)
                            dirx = bvx / safe_bv
                            diry = bvy / safe_bv
                            dirz = bvz / safe_bv
                            ipx, ipy, ipz, ipw = dmath.quat_inverse(
                                p_step_basic_rotations[par, 0], p_step_basic_rotations[par, 1],
                                p_step_basic_rotations[par, 2], p_step_basic_rotations[par, 3])
                            lpx, lpy, lpz = dmath.quat_rotate(ipx, ipy, ipz, ipw, dirx, diry, dirz)
                            lrx, lry, lrz, lrw = dmath.quat_mul(
                                ipx, ipy, ipz, ipw,
                                p_step_basic_rotations[v, 0], p_step_basic_rotations[v, 1],
                                p_step_basic_rotations[v, 2], p_step_basic_rotations[v, 3])
                            p_albuf_length[v] = avlen
                            p_albuf_local_pos[v, 0] = lpx
                            p_albuf_local_pos[v, 1] = lpy
                            p_albuf_local_pos[v, 2] = lpz
                            p_albuf_local_rot[v, 0] = lrx
                            p_albuf_local_rot[v, 1] = lry
                            p_albuf_local_rot[v, 2] = lrz
                            p_albuf_local_rot[v, 3] = lrw
                    if t_angle_use_restoration[vt] != 0:
                        p_albuf_restore[v, 0] = bvx
                        p_albuf_restore[v, 1] = bvy
                        p_albuf_restore[v, 2] = bvz
                i += stride
        grid.sync()
        # Single-block the 3xP (level,rank) passes: only block 0 walks every pass, using
        # __syncthreads() between passes (two orders of magnitude cheaper than grid.sync on this
        # small 7-block grid). Each pass fits one block-wave (max pass entries 92 < blockDim) and
        # the per-entry writes are disjoint (v-set at level L, p-set at level L-1), so moving the
        # work from a 7-block grid-stride to a block-0 stride is order-independent -> bit-exact.
        if bid == 0:
            for _ai in range(ANGLE_ITERATION):
                angle_rot_ratio = float32(0.1) + (float32(0.5) - float32(0.1)) \
                    * (float32(_ai) / float32(2.0))
                for _ap in range(num_angle_passes):
                    if phase_mask & PHASE_ANGLE:
                        aps = angle_pass_offsets[_ap]
                        ape = angle_pass_offsets[_ap + 1]
                        e = aps + tid
                        while e < ape:
                            v = angle_pass_vertices[e]
                            p = angle_pass_parents[e]
                            vt = p_team[v]
                            if team_frame_mask(t_enabled, t_valid, t_cws, vt) \
                                    and t_update_count[vt] > _k:
                                ul = t_angle_use_limit[vt] != 0
                                ur = t_angle_use_restoration[vt] != 0
                                if ul or ur:
                                    c_inv = float32(1.0) / (float32(1.0) + p_friction[v] * FRICTION_MASS)
                                    p_inv = float32(1.0) / (float32(1.0) + p_friction[p] * FRICTION_MASS)
                                    p_mv = p_attr_move[p] != 0
                                    if ul:
                                        do_angle_limit(v, p, vt, c_inv, p_inv, p_mv,
                                                       p_next_positions, p_velocity_positions,
                                                       p_albuf_rotation, p_albuf_local_pos,
                                                       p_albuf_local_rot, p_albuf_length, p_depth,
                                                       t_angle_limit_lut, t_angle_limit_stiffness)
                                    if ur:
                                        do_angle_restoration(v, p, vt, c_inv, p_inv, p_mv,
                                                             angle_rot_ratio, power3,
                                                             p_next_positions, p_velocity_positions,
                                                             p_albuf_restore, p_depth,
                                                             t_angle_restoration_lut,
                                                             t_angle_restoration_attenuation,
                                                             t_angle_restoration_gravity_falloff,
                                                             t_gravity_dot)
                            e += bdim
                    cuda.syncthreads()
        grid.sync()

        # --- S8 bending.run: clear -> per-pair scatter (int32 fixed-point) -> apply ---
        if phase_mask & PHASE_BENDING:
            p = tid
            while p < num_particles:
                sc_dcorr_fixed[p, 0] = int32(0)
                sc_dcorr_fixed[p, 1] = int32(0)
                sc_dcorr_fixed[p, 2] = int32(0)
                sc_dcount[p] = int32(0)
                p += stride
        grid.sync()
        if phase_mask & PHASE_BENDING:
            e = tid
            while e < n_bending:
                team = st_bending_team[e]
                if team_frame_mask(t_enabled, t_valid, t_cws, team) and t_update_count[team] > _k \
                        and t_bending_stiffness[team] >= float32(1e-6):
                    stiffness = dmath.saturate(t_bending_stiffness[team] * power1)
                    pp0 = st_bending_pair[e, 0]
                    pp1 = st_bending_pair[e, 1]
                    pp2 = st_bending_pair[e, 2]
                    pp3 = st_bending_pair[e, 3]
                    rest = st_bending_rest[e]
                    sgn = st_bending_sign[e]
                    a0x = p_next_positions[pp0, 0]
                    a0y = p_next_positions[pp0, 1]
                    a0z = p_next_positions[pp0, 2]
                    a1x = p_next_positions[pp1, 0]
                    a1y = p_next_positions[pp1, 1]
                    a1z = p_next_positions[pp1, 2]
                    a2x = p_next_positions[pp2, 0]
                    a2y = p_next_positions[pp2, 1]
                    a2z = p_next_positions[pp2, 2]
                    a3x = p_next_positions[pp3, 0]
                    a3y = p_next_positions[pp3, 1]
                    a3z = p_next_positions[pp3, 2]
                    if p_attr_move[pp0] == 0:
                        inv0 = BENDING_FIX_INV_MASS
                    else:
                        inv0 = dmath.calc_inverse_mass(p_friction[pp0], p_depth[pp0])
                    if p_attr_move[pp1] == 0:
                        inv1 = BENDING_FIX_INV_MASS
                    else:
                        inv1 = dmath.calc_inverse_mass(p_friction[pp1], p_depth[pp1])
                    if p_attr_move[pp2] == 0:
                        inv2 = BENDING_FIX_INV_MASS
                    else:
                        inv2 = dmath.calc_inverse_mass(p_friction[pp2], p_depth[pp2])
                    if p_attr_move[pp3] == 0:
                        inv3 = BENDING_FIX_INV_MASS
                    else:
                        inv3 = dmath.calc_inverse_mass(p_friction[pp3], p_depth[pp3])
                    scale_ratio = t_scale_ratio[team]
                    negative_sign = t_negative_scale_sign[team]
                    result = False
                    a0dx = float32(0.0)
                    a0dy = float32(0.0)
                    a0dz = float32(0.0)
                    a1dx = float32(0.0)
                    a1dy = float32(0.0)
                    a1dz = float32(0.0)
                    a2dx = float32(0.0)
                    a2dy = float32(0.0)
                    a2dz = float32(0.0)
                    a3dx = float32(0.0)
                    a3dy = float32(0.0)
                    a3dz = float32(0.0)
                    if sgn == VOLUME_SIGN:
                        volume_rest = rest * scale_ratio * negative_sign
                        cx, cy, cz = dmath.cross3(a1x - a0x, a1y - a0y, a1z - a0z,
                                                  a2x - a0x, a2y - a0y, a2z - a0z)
                        volume = ONE_SIXTH * (cx * (a3x - a0x) + cy * (a3y - a0y)
                                              + cz * (a3z - a0z)) * VOLUME_SCALE
                        g0x, g0y, g0z = dmath.cross3(a1x - a2x, a1y - a2y, a1z - a2z,
                                                     a3x - a2x, a3y - a2y, a3z - a2z)
                        g1x, g1y, g1z = dmath.cross3(a2x - a0x, a2y - a0y, a2z - a0z,
                                                     a3x - a0x, a3y - a0y, a3z - a0z)
                        g2x, g2y, g2z = dmath.cross3(a0x - a1x, a0y - a1y, a0z - a1z,
                                                     a3x - a1x, a3y - a1y, a3z - a1z)
                        g3x, g3y, g3z = dmath.cross3(a1x - a0x, a1y - a0y, a1z - a0z,
                                                     a2x - a0x, a2y - a0y, a2z - a0z)
                        lam = (inv0 * (g0x * g0x + g0y * g0y + g0z * g0z)
                               + inv1 * (g1x * g1x + g1y * g1y + g1z * g1z)
                               + inv2 * (g2x * g2x + g2y * g2y + g2z * g2z)
                               + inv3 * (g3x * g3x + g3y * g3y + g3z * g3z))
                        lam = lam * VOLUME_SCALE
                        if libdevice.fabsf(lam) >= float32(1e-6):
                            lam = stiffness * (volume_rest - volume) / lam
                            a0dx = lam * inv0 * g0x
                            a0dy = lam * inv0 * g0y
                            a0dz = lam * inv0 * g0z
                            a1dx = lam * inv1 * g1x
                            a1dy = lam * inv1 * g1y
                            a1dz = lam * inv1 * g1z
                            a2dx = lam * inv2 * g2x
                            a2dy = lam * inv2 * g2y
                            a2dz = lam * inv2 * g2z
                            a3dx = lam * inv3 * g3x
                            a3dy = lam * inv3 * g3y
                            a3dz = lam * inv3 * g3z
                            result = True
                    else:
                        rest_angle = rest * float32(sgn) * negative_sign
                        ex = a3x - a2x
                        ey = a3y - a2y
                        ez = a3z - a2z
                        elen = dmath.length3(ex, ey, ez)
                        ok = elen >= float32(1e-8)
                        safe_elen = elen if elen > float32(1e-30) else float32(1.0)
                        inv_elen = float32(1.0) / safe_elen
                        nn1x, nn1y, nn1z = dmath.cross3(a2x - a0x, a2y - a0y, a2z - a0z,
                                                        a3x - a0x, a3y - a0y, a3z - a0z)
                        nn2x, nn2y, nn2z = dmath.cross3(a3x - a1x, a3y - a1y, a3z - a1z,
                                                        a2x - a1x, a2y - a1y, a2z - a1z)
                        sq1 = nn1x * nn1x + nn1y * nn1y + nn1z * nn1z
                        sq2 = nn2x * nn2x + nn2y * nn2y + nn2z * nn2z
                        ok = ok and (sq1 != float32(0.0)) and (sq2 != float32(0.0))
                        safe_sq1 = sq1 if sq1 > float32(1e-30) else float32(1.0)
                        safe_sq2 = sq2 if sq2 > float32(1e-30) else float32(1.0)
                        nn1x = nn1x / safe_sq1
                        nn1y = nn1y / safe_sq1
                        nn1z = nn1z / safe_sq1
                        nn2x = nn2x / safe_sq2
                        nn2y = nn2y / safe_sq2
                        nn2z = nn2z / safe_sq2
                        d0x = nn1x * elen
                        d0y = nn1y * elen
                        d0z = nn1z * elen
                        d1x = nn2x * elen
                        d1y = nn2y * elen
                        d1z = nn2z * elen
                        dot03 = (a0x - a3x) * ex + (a0y - a3y) * ey + (a0z - a3z) * ez
                        dot13 = (a1x - a3x) * ex + (a1y - a3y) * ey + (a1z - a3z) * ez
                        d2x = dot03 * inv_elen * nn1x + dot13 * inv_elen * nn2x
                        d2y = dot03 * inv_elen * nn1y + dot13 * inv_elen * nn2y
                        d2z = dot03 * inv_elen * nn1z + dot13 * inv_elen * nn2z
                        dot20 = (a2x - a0x) * ex + (a2y - a0y) * ey + (a2z - a0z) * ez
                        dot21 = (a2x - a1x) * ex + (a2y - a1y) * ey + (a2z - a1z) * ez
                        d3x = dot20 * inv_elen * nn1x + dot21 * inv_elen * nn2x
                        d3y = dot20 * inv_elen * nn1y + dot21 * inv_elen * nn2y
                        d3z = dot20 * inv_elen * nn1z + dot21 * inv_elen * nn2z
                        un1x, un1y, un1z = dmath.normalize3(nn1x, nn1y, nn1z)
                        un2x, un2y, un2z = dmath.normalize3(nn2x, nn2y, nn2z)
                        dotu = dmath.clamp1(un1x * un2x + un1y * un2y + un1z * un2z)
                        phi = libdevice.acosf(dotu)
                        lam = (inv0 * (d0x * d0x + d0y * d0y + d0z * d0z)
                               + inv1 * (d1x * d1x + d1y * d1y + d1z * d1z)
                               + inv2 * (d2x * d2x + d2y * d2y + d2z * d2z)
                               + inv3 * (d3x * d3x + d3y * d3y + d3z * d3z))
                        ok = ok and (lam != float32(0.0))
                        crx, cry, crz = dmath.cross3(un1x, un1y, un1z, un2x, un2y, un2z)
                        dir_sign = dmath.fsign(crx * ex + cry * ey + crz * ez)
                        phi = phi * dir_sign
                        if ok:
                            lam = (rest_angle - phi) / lam * stiffness
                            a0dx = -inv0 * lam * d0x
                            a0dy = -inv0 * lam * d0y
                            a0dz = -inv0 * lam * d0z
                            a1dx = -inv1 * lam * d1x
                            a1dy = -inv1 * lam * d1y
                            a1dz = -inv1 * lam * d1z
                            a2dx = -inv2 * lam * d2x
                            a2dy = -inv2 * lam * d2y
                            a2dz = -inv2 * lam * d2z
                            a3dx = -inv3 * lam * d3x
                            a3dy = -inv3 * lam * d3y
                            a3dz = -inv3 * lam * d3z
                            result = True
                    if result:
                        cuda.atomic.add(sc_dcorr_fixed, (pp0, 0), int32(a0dx * TO_FIXED))
                        cuda.atomic.add(sc_dcorr_fixed, (pp0, 1), int32(a0dy * TO_FIXED))
                        cuda.atomic.add(sc_dcorr_fixed, (pp0, 2), int32(a0dz * TO_FIXED))
                        cuda.atomic.add(sc_dcount, pp0, int32(1))
                        cuda.atomic.add(sc_dcorr_fixed, (pp1, 0), int32(a1dx * TO_FIXED))
                        cuda.atomic.add(sc_dcorr_fixed, (pp1, 1), int32(a1dy * TO_FIXED))
                        cuda.atomic.add(sc_dcorr_fixed, (pp1, 2), int32(a1dz * TO_FIXED))
                        cuda.atomic.add(sc_dcount, pp1, int32(1))
                        cuda.atomic.add(sc_dcorr_fixed, (pp2, 0), int32(a2dx * TO_FIXED))
                        cuda.atomic.add(sc_dcorr_fixed, (pp2, 1), int32(a2dy * TO_FIXED))
                        cuda.atomic.add(sc_dcorr_fixed, (pp2, 2), int32(a2dz * TO_FIXED))
                        cuda.atomic.add(sc_dcount, pp2, int32(1))
                        cuda.atomic.add(sc_dcorr_fixed, (pp3, 0), int32(a3dx * TO_FIXED))
                        cuda.atomic.add(sc_dcorr_fixed, (pp3, 1), int32(a3dy * TO_FIXED))
                        cuda.atomic.add(sc_dcorr_fixed, (pp3, 2), int32(a3dz * TO_FIXED))
                        cuda.atomic.add(sc_dcount, pp3, int32(1))
                e += stride
        grid.sync()
        if phase_mask & PHASE_BENDING:
            p = tid
            while p < num_particles:
                mt = p_team[p]
                if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k \
                        and p_attr_move[p] != 0 and sc_dcount[p] > 0:
                    inv_c = float32(1.0) / float32(sc_dcount[p])
                    p_next_positions[p, 0] += float32(sc_dcorr_fixed[p, 0]) / TO_FIXED * inv_c
                    p_next_positions[p, 1] += float32(sc_dcorr_fixed[p, 1]) / TO_FIXED * inv_c
                    p_next_positions[p, 2] += float32(sc_dcorr_fixed[p, 2]) / TO_FIXED * inv_c
                p += stride
        grid.sync()

        # --- S9 collider.solve: point (per-particle gather, in-place) ---
        if phase_mask & PHASE_COLLIDER_SOLVE:
            p = tid
            while p < num_particles:
                mt = p_team[p]
                if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
                    do_solve_point(p, p_team, p_next_positions, p_base_positions, p_depth,
                                   p_friction, p_collision_normals, p_velocity_positions,
                                   t_collision_mode, t_radius_lut, t_scale_ratio, t_is_spring,
                                   t_limit_distance_lut, c_kind, c_active, c_work_old_pos,
                                   c_work_next_pos, c_work_radius, c_work_inv_old_rot, c_work_rot,
                                   c_work_aabb_min, c_work_aabb_max,
                                   csr_point_pair_offsets, csr_point_pair_order,
                                   st_point_pair_collider)
                p += stride
        grid.sync()
        # S9 edge: clear endpoint scatter scratch
        if phase_mask & PHASE_COLLIDER_SOLVE:
            p = tid
            while p < num_particles:
                sc_dcorr_fixed[p, 0] = int32(0)
                sc_dcorr_fixed[p, 1] = int32(0)
                sc_dcorr_fixed[p, 2] = int32(0)
                sc_dcount[p] = int32(0)
                sc_col_friction_fixed[p] = int32(0)
                sc_col_normal_fixed[p, 0] = int32(0)
                sc_col_normal_fixed[p, 1] = int32(0)
                sc_col_normal_fixed[p, 2] = int32(0)
                p += stride
        grid.sync()
        # S9 edge: per-edge compute + fixed-point endpoint scatter (EDGE-mode teams only)
        if phase_mask & PHASE_COLLIDER_SOLVE:
            ee = tid
            while ee < st_collision_edge.shape[0]:
                et = p_team[st_collision_edge[ee, 0]]
                if team_frame_mask(t_enabled, t_valid, t_cws, et) and t_update_count[et] > _k \
                        and t_collision_mode[et] == COLLISION_EDGE:
                    do_solve_edge(ee, p_team, p_next_positions, p_depth, p_attr_move,
                                  t_radius_lut, t_scale_ratio, c_kind, c_active, c_work_old_pos,
                                  c_work_next_pos, c_work_radius, c_work_aabb_min, c_work_aabb_max,
                                  csr_edge_pair_offsets, csr_edge_pair_order, st_edge_pair_collider,
                                  st_collision_edge, sc_dcorr_fixed, sc_dcount,
                                  sc_col_friction_fixed, sc_col_normal_fixed)
                ee += stride
        grid.sync()
        # S9 edge: per-particle apply of endpoint accumulation
        if phase_mask & PHASE_COLLIDER_SOLVE:
            p = tid
            while p < num_particles:
                mt = p_team[p]
                if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k \
                        and t_collision_mode[mt] == COLLISION_EDGE:
                    cnt = sc_dcount[p]
                    if cnt > int32(0):
                        inv_e = float32(1.0) / float32(cnt)
                        p_next_positions[p, 0] += float32(sc_dcorr_fixed[p, 0]) / TO_FIXED * inv_e
                        p_next_positions[p, 1] += float32(sc_dcorr_fixed[p, 1]) / TO_FIXED * inv_e
                        p_next_positions[p, 2] += float32(sc_dcorr_fixed[p, 2]) / TO_FIXED * inv_e
                    ef = float32(sc_col_friction_fixed[p]) / TO_FIXED
                    if ef > p_friction[p]:
                        p_friction[p] = ef
                    enx = float32(sc_col_normal_fixed[p, 0]) / TO_FIXED
                    eny = float32(sc_col_normal_fixed[p, 1]) / TO_FIXED
                    enz = float32(sc_col_normal_fixed[p, 2]) / TO_FIXED
                    if (enx * enx + eny * eny + enz * enz) > float32(0.0):
                        onx, ony, onz = dmath.normalize3(enx, eny, enz)
                        p_collision_normals[p, 0] = onx
                        p_collision_normals[p, 1] = ony
                        p_collision_normals[p, 2] = onz
                p += stride
        grid.sync()

        # --- S10 distance.run (second occurrence): same Jacobi gather -> apply ---
        if phase_mask & PHASE_DISTANCE_B:
            p = tid
            while p < num_particles:
                mt = p_team[p]
                if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
                    do_distance_gather(p, p_team, p_next_positions, p_base_positions, p_depth,
                                       p_friction, p_attr_move, t_is_spring, t_animation_pose_ratio,
                                       t_init_scale, t_scale_ratio, t_distance_lut, power1,
                                       csr_distance_offsets, csr_distance_order,
                                       st_distance_target, st_distance_rest, sc_dcorr)
                p += stride
        grid.sync()
        if phase_mask & PHASE_DISTANCE_B:
            p = tid
            while p < num_particles:
                mt = p_team[p]
                if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
                    p_next_positions[p, 0] += sc_dcorr[p, 0]
                    p_next_positions[p, 1] += sc_dcorr[p, 1]
                    p_next_positions[p, 2] += sc_dcorr[p, 2]
                    p_velocity_positions[p, 0] += sc_dcorr[p, 0] * DISTANCE_VELOCITY_ATTENUATION
                    p_velocity_positions[p, 1] += sc_dcorr[p, 1] * DISTANCE_VELOCITY_ATTENUATION
                    p_velocity_positions[p, 2] += sc_dcorr[p, 2] * DISTANCE_VELOCITY_ATTENUATION
                p += stride
        grid.sync()

        # --- S11 motion.run (per-entry independent) ---
        if phase_mask & PHASE_MOTION:
            e = tid
            while e < n_motion:
                mt = st_motion_team[e]
                use_max = t_motion_use_max_distance[mt] != 0
                use_backstop = t_motion_use_backstop[mt] != 0
                if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k \
                        and (use_max or use_backstop):
                    index = st_motion_particle[e]
                    stiffness = t_motion_stiffness[mt]
                    backstop_radius = t_motion_backstop_radius[mt]
                    o0 = p_next_positions[index, 0]
                    o1 = p_next_positions[index, 1]
                    o2 = p_next_positions[index, 2]
                    n0 = o0
                    n1 = o1
                    n2 = o2
                    b0 = p_base_positions[index, 0]
                    b1 = p_base_positions[index, 1]
                    b2 = p_base_positions[index, 2]
                    depth = p_depth[index]
                    radius = dmath.evaluate_team_lut(t_radius_lut, mt, depth)
                    if radius < float32(0.0001):
                        radius = float32(0.0001)
                    cfr = radius
                    depth2 = depth * depth
                    dirx, diry, dirz = dmath.quat_rotate(
                        p_base_rotations[index, 0], p_base_rotations[index, 1],
                        p_base_rotations[index, 2], p_base_rotations[index, 3],
                        t_normal_axis_vector[mt, 0], t_normal_axis_vector[mt, 1],
                        t_normal_axis_vector[mt, 2])
                    if use_max:
                        max_distance = dmath.evaluate_team_lut(t_motion_max_distance_lut, mt, depth2)
                        cvx, cvy, cvz = dmath.clamp_vector(n0 - b0, n1 - b1, n2 - b2, max_distance)
                        n0 = b0 + cvx
                        n1 = b1 + cvy
                        n2 = b2 + cvz
                    if use_backstop and backstop_radius > float32(0.0):
                        backstop_distance = dmath.evaluate_team_lut(t_motion_backstop_lut, mt, depth2)
                        off = backstop_distance + backstop_radius
                        cx = b0 - dirx * off
                        cy = b1 - diry * off
                        cz = b2 - dirz * off
                        vx = n0 - cx
                        vy = n1 - cy
                        vz = n2 - cz
                        l = dmath.length3(vx, vy, vz)
                        near = (l > EPSILON) and (l < backstop_radius + cfr)
                        if near and (l < backstop_radius):
                            safe_l = l if l > float32(1e-30) else float32(1.0)
                            n0 = cx + vx / safe_l * backstop_radius
                            n1 = cy + vy / safe_l * backstop_radius
                            n2 = cz + vz / safe_l * backstop_radius
                    n0 = dmath.lerp(o0, n0, stiffness)
                    n1 = dmath.lerp(o1, n1, stiffness)
                    n2 = dmath.lerp(o2, n2, stiffness)
                    p_next_positions[index, 0] = n0
                    p_next_positions[index, 1] = n1
                    p_next_positions[index, 2] = n2
                    p_velocity_positions[index, 0] += (n0 - o0) * float32(0.95)
                    p_velocity_positions[index, 1] += (n1 - o1) * float32(0.95)
                    p_velocity_positions[index, 2] += (n2 - o2) * float32(0.95)
                e += stride
        grid.sync()

        # --- S13 particles.step_post PASS 1: friction / limit / centrifugal (move set) ---
        if phase_mask & PHASE_STEP_POST:
            e = tid
            while e < n_move:
                mt = st_move_team[e]
                if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
                    pmi = st_move_particle[e]
                    n0 = p_next_positions[pmi, 0]
                    n1 = p_next_positions[pmi, 1]
                    n2 = p_next_positions[pmi, 2]
                    o0 = p_old_positions[pmi, 0]
                    o1 = p_old_positions[pmi, 1]
                    o2 = p_old_positions[pmi, 2]
                    vo0 = p_velocity_positions[pmi, 0]
                    vo1 = p_velocity_positions[pmi, 1]
                    vo2 = p_velocity_positions[pmi, 2]
                    depth = p_depth[pmi]
                    friction = p_friction[pmi]
                    cn0 = p_collision_normals[pmi, 0]
                    cn1 = p_collision_normals[pmi, 1]
                    cn2 = p_collision_normals[pmi, 2]
                    cn_len2 = cn0 * cn0 + cn1 * cn1 + cn2 * cn2
                    is_collision = (cn_len2 > EPSILON) and (friction > EPSILON)
                    static_param = t_static_friction[mt] * t_scale_ratio[mt]
                    dynamic_param = t_dynamic_friction[mt]

                    # static friction
                    sfp = p_static_friction[pmi]
                    static_on = static_param > float32(0.0)
                    vx = n0 - o0
                    vy = n1 - o1
                    vz = n2 - o2
                    vdotcn = vx * cn0 + vy * cn1 + vz * cn2
                    tgx = vx - vdotcn * cn0
                    tgy = vy - vdotcn * cn1
                    tgz = vz - vdotcn * cn2
                    tangent_velocity = dmath.length3(tgx, tgy, tgz) / sim_dt
                    increase = dmath.saturate(sfp + float32(0.04))
                    dec_amount = (tangent_velocity - static_param) / float32(0.2)
                    if dec_amount < float32(0.05):
                        dec_amount = float32(0.05)
                    decrease = dmath.saturate(sfp - dec_amount)
                    new_static = increase if tangent_velocity < static_param else decrease
                    decayed = dmath.saturate(sfp - float32(0.05))
                    updated_sf = new_static if is_collision else decayed
                    sfp_new = updated_sf if static_on else decayed
                    if static_on and is_collision:
                        rbx = tgx * sfp_new
                        rby = tgy * sfp_new
                        rbz = tgz * sfp_new
                    else:
                        rbx = float32(0.0)
                        rby = float32(0.0)
                        rbz = float32(0.0)
                    n0 = n0 - rbx
                    n1 = n1 - rby
                    n2 = n2 - rbz
                    vo0 = vo0 - rbx
                    vo1 = vo1 - rby
                    vo2 = vo2 - rbz
                    p_static_friction[pmi] = sfp_new

                    # dynamic friction
                    velx = (n0 - vo0) / sim_dt
                    vely = (n1 - vo1) / sim_dt
                    velz = (n2 - vo2) / sim_dt
                    sq_velocity = velx * velx + vely * vely + velz * velz
                    nvx, nvy, nvz = dmath.normalize3(velx, vely, velz)
                    if not (sq_velocity > EPSILON):
                        nvx = float32(0.0)
                        nvy = float32(0.0)
                        nvz = float32(0.0)
                    dynamic_on = dynamic_param > float32(0.0)
                    dd = cn0 * nvx + cn1 * nvy + cn2 * nvz
                    dd = float32(0.5) + float32(0.5) * dd
                    dd = dd * dd
                    dd = float32(1.0) - dd
                    damp = dd * dmath.saturate(friction * dynamic_param)
                    if dynamic_on and is_collision and (sq_velocity >= EPSILON):
                        velx = velx - velx * damp
                        vely = vely - vely * damp
                        velz = velz - velz * damp
                    p_friction[pmi] = friction * float32(0.6)

                    # speed limit
                    speed_limit = t_particle_speed_limit[mt]
                    max_len = speed_limit * t_scale_ratio[mt]
                    if max_len < float32(0.0):
                        max_len = float32(0.0)
                    if speed_limit >= float32(0.0):
                        velx, vely, velz = dmath.clamp_vector(velx, vely, velz, max_len)

                    # centrifugal
                    angular = t_angular_velocity[mt]
                    centrifugal = t_centrifugal_acceleration[mt]
                    if (angular > EPSILON) and (centrifugal > EPSILON):
                        axx = t_rotation_axis[mt, 0]
                        axy = t_rotation_axis[mt, 1]
                        axz = t_rotation_axis[mt, 2]
                        lpx = n0 - t_now_world_position[mt, 0]
                        lpy = n1 - t_now_world_position[mt, 1]
                        lpz = n2 - t_now_world_position[mt, 2]
                        lp_dot = lpx * axx + lpy * axy + lpz * axz
                        v2x = lpx - lp_dot * axx
                        v2y = lpy - lp_dot * axy
                        v2z = lpz - lp_dot * axz
                        rr = dmath.length3(v2x, v2y, v2z)
                        if (rr > EPSILON) and (sq_velocity >= EPSILON):
                            nx2, ny2, nz2 = dmath.normalize3(v2x, v2y, v2z)
                            mm = float32(1.0) + (float32(1.0) - depth)
                            ff = mm * angular * angular * rr
                            ucx, ucy, ucz = dmath.cross3(axx, axy, axz, nx2, ny2, nz2)
                            uux, uuy, uuz = dmath.normalize3(ucx, ucy, ucz)
                            ff = ff * dmath.saturate(nvx * uux + nvy * uuy + nvz * uuz)
                            addc = ff * centrifugal * float32(0.02)
                            velx = velx + nx2 * addc
                            vely = vely + ny2 * addc
                            velz = velz + nz2 * addc

                    vw = t_velocity_weight[mt]
                    p_velocities[pmi, 0] = velx * vw
                    p_velocities[pmi, 1] = vely * vw
                    p_velocities[pmi, 2] = velz * vw
                    p_next_positions[pmi, 0] = n0
                    p_next_positions[pmi, 1] = n1
                    p_next_positions[pmi, 2] = n2
                e += stride
        grid.sync()

        # --- S13 PASS 2: real_velocities + old_positions for all substep particles ---
        if phase_mask & PHASE_STEP_POST:
            p = tid
            while p < num_particles:
                mt = p_team[p]
                if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
                    p_real_velocities[p, 0] = (p_next_positions[p, 0] - p_old_positions[p, 0]) / sim_dt
                    p_real_velocities[p, 1] = (p_next_positions[p, 1] - p_old_positions[p, 1]) / sim_dt
                    p_real_velocities[p, 2] = (p_next_positions[p, 2] - p_old_positions[p, 2]) / sim_dt
                    p_old_positions[p, 0] = p_next_positions[p, 0]
                    p_old_positions[p, 1] = p_next_positions[p, 1]
                    p_old_positions[p, 2] = p_next_positions[p, 2]
                p += stride
        grid.sync()

        # --- S14 collider.end_step (per-collider; old = now) ---
        if phase_mask & PHASE_COLLIDER_END:
            ci = tid
            while ci < num_colliders:
                cm = c_team[ci]
                if team_frame_mask(t_enabled, t_valid, t_cws, cm) and t_update_count[cm] > _k \
                        and c_active[ci] != 0:
                    do_collider_end_step(ci, c_now_pos, c_now_rot, c_old_pos, c_old_rot)
                ci += stride
        grid.sync()

    # ----- FRAME-POST -----
    grid.sync()
    # F2 display.run: segment A _display (per-particle) -> B _postline (per level, per entry) ->
    # C _post_triangles (C1 per-triangle normal/tangent, C2 per-owner v2t reduce) -> D _output.
    # Sits at the frame-post head (the self-collision F1 slot is a G3 no-op), before F3; F3/F4
    # are gated by their own bits so isolated PHASE_DISPLAY launches leave them untouched. Zero
    # atomics: postline children walk the entry CSR serially, v2t rows walk the owner CSR serially,
    # grid.sync between the four segments (and per postline level) orders every cross-thread write.
    num_postline_levels = postline_entry_offsets.shape[0] - 1
    num_triangles = st_triangle_team.shape[0]
    if phase_mask & PHASE_DISPLAY:
        p = tid
        while p < num_particles:
            mt = p_team[p]
            if team_frame_mask(t_enabled, t_valid, t_cws, mt):
                do_display_particle(p, mt, sim_dt, p_positions, p_rotations, p_old_positions,
                                    p_real_velocities, p_display_positions, p_vertex_root,
                                    p_old_anim_positions, p_old_anim_rotations,
                                    p_temp_base_positions, p_temp_base_rotations,
                                    st_display_update_move_mask, t_now_update, t_old_time, t_time,
                                    t_blend_weight, t_running, t_is_negative_scale,
                                    t_negative_scale_direction)
            p += stride
    grid.sync()
    # Block-0 walks the postline levels with __syncthreads(): children sit one level below the
    # entry, so the level barrier keeps child-writes visible to the parent-entry read; each
    # entry's write is unique (owner-grouped CSR), so block-0-stride is bit-exact (max level 28).
    for lvl in range(num_postline_levels):
        if bid == 0 and (phase_mask & PHASE_DISPLAY) != 0:
            pl_start = postline_entry_offsets[lvl]
            pl_end = postline_entry_offsets[lvl + 1]
            i = pl_start + tid
            while i < pl_end:
                entry = postline_entry_vertices[i]
                et = p_team[entry]
                if team_frame_mask(t_enabled, t_valid, t_cws, et):
                    do_postline_entry(entry, et, postline_child_offsets[i],
                                      postline_child_offsets[i + 1], postline_child_vertices,
                                      p_positions, p_rotations, p_temp_base_positions,
                                      p_temp_base_rotations, p_vertex_local_positions,
                                      p_vertex_local_rotations, p_attr_invalid,
                                      p_attr_zero_distance, p_attr_move, p_team,
                                      t_rotational_interpolation, t_root_rotation, t_blend_weight,
                                      t_animation_pose_ratio, t_negative_scale_direction,
                                      t_negative_scale_quaternion)
                i += bdim
        cuda.syncthreads()
    grid.sync()
    if phase_mask & PHASE_DISPLAY:
        tri_idx = tid
        while tri_idx < num_triangles:
            tt_team = st_triangle_team[tri_idx]
            if team_frame_mask(t_enabled, t_valid, t_cws, tt_team):
                do_triangle_normal_tangent(tri_idx, tt_team, st_triangle_particles, p_positions,
                                           p_uv, t_negative_scale_triangle_sign, sc_tri_normal_f64,
                                           sc_tri_tangent_f64)
            tri_idx += stride
    grid.sync()
    if phase_mask & PHASE_DISPLAY:
        p = tid
        while p < num_particles:
            seg0 = csr_v2t_offsets[p]
            seg1 = csr_v2t_offsets[p + 1]
            if seg0 < seg1 and team_frame_mask(t_enabled, t_valid, t_cws, p_team[p]):
                do_v2t_owner(p, p_team[p], seg0, seg1, csr_v2t_order, st_v2t_triangle,
                             st_v2t_flip_normal, st_v2t_flip_tangent, sc_tri_normal_f64,
                             sc_tri_tangent_f64, p_rotations, p_normal_adjustment_rotations,
                             t_negative_scale_quaternion)
            p += stride
    grid.sync()
    if phase_mask & PHASE_DISPLAY:
        p = tid
        while p < num_particles:
            mt = p_team[p]
            if team_frame_mask(t_enabled, t_valid, t_cws, mt):
                do_output_particle(p, mt, p_rotations, p_vertex_to_transform_rotations,
                                   t_negative_scale_quaternion, p_out_rotations)
            p += stride
    grid.sync()
    # F3 collider.frame_post (per-collider; running teams: old_frame = frame)
    if phase_mask & PHASE_COLLIDER_POST:
        ci = tid
        while ci < num_colliders:
            cm = c_team[ci]
            if team_frame_mask(t_enabled, t_valid, t_cws, cm) and t_running[cm] != 0 \
                    and c_active[ci] != 0:
                do_collider_frame_post(ci, c_frame_pos, c_frame_rot, c_old_frame_pos, c_old_frame_rot)
            ci += stride
    grid.sync()
    # F4 team_time.frame_post (per-team bookkeeping, no cross-team dependency)
    if phase_mask & PHASE_TEAM_POST:
        i = tid
        while i < num_teams:
            if team_frame_mask(t_enabled, t_valid, t_cws, i):
                run = t_running[i] != 0
                t_old_component_world_position[i, 0] = t_component_world_position[i, 0]
                t_old_component_world_position[i, 1] = t_component_world_position[i, 1]
                t_old_component_world_position[i, 2] = t_component_world_position[i, 2]
                t_old_component_world_rotation[i, 0] = t_component_world_rotation[i, 0]
                t_old_component_world_rotation[i, 1] = t_component_world_rotation[i, 1]
                t_old_component_world_rotation[i, 2] = t_component_world_rotation[i, 2]
                t_old_component_world_rotation[i, 3] = t_component_world_rotation[i, 3]
                t_old_component_world_scale[i, 0] = t_cws[i, 0]
                t_old_component_world_scale[i, 1] = t_cws[i, 1]
                t_old_component_world_scale[i, 2] = t_cws[i, 2]
                if run:
                    t_old_frame_world_position[i, 0] = t_frame_world_position[i, 0]
                    t_old_frame_world_position[i, 1] = t_frame_world_position[i, 1]
                    t_old_frame_world_position[i, 2] = t_frame_world_position[i, 2]
                    t_old_frame_world_rotation[i, 0] = t_frame_world_rotation[i, 0]
                    t_old_frame_world_rotation[i, 1] = t_frame_world_rotation[i, 1]
                    t_old_frame_world_rotation[i, 2] = t_frame_world_rotation[i, 2]
                    t_old_frame_world_rotation[i, 3] = t_frame_world_rotation[i, 3]
                    t_old_frame_world_scale[i, 0] = t_frame_world_scale[i, 0]
                    t_old_frame_world_scale[i, 1] = t_frame_world_scale[i, 1]
                    t_old_frame_world_scale[i, 2] = t_frame_world_scale[i, 2]
                    t_skip_count[i] = int32(0)
                    t_force_mode[i] = 0
                    t_impact_force[i, 0] = float32(0.0)
                    t_impact_force[i, 1] = float32(0.0)
                    t_impact_force[i, 2] = float32(0.0)
                t_old_anchor_position[i, 0] = t_anchor_position[i, 0]
                t_old_anchor_position[i, 1] = t_anchor_position[i, 1]
                t_old_anchor_position[i, 2] = t_anchor_position[i, 2]
                t_old_anchor_rotation[i, 0] = t_anchor_rotation[i, 0]
                t_old_anchor_rotation[i, 1] = t_anchor_rotation[i, 1]
                t_old_anchor_rotation[i, 2] = t_anchor_rotation[i, 2]
                t_old_anchor_rotation[i, 3] = t_anchor_rotation[i, 3]
                qix, qiy, qiz, qiw = dmath.quat_inverse(
                    t_anchor_rotation[i, 0], t_anchor_rotation[i, 1],
                    t_anchor_rotation[i, 2], t_anchor_rotation[i, 3])
                dpx = t_component_world_position[i, 0] - t_anchor_position[i, 0]
                dpy = t_component_world_position[i, 1] - t_anchor_position[i, 1]
                dpz = t_component_world_position[i, 2] - t_anchor_position[i, 2]
                alx, aly, alz = dmath.quat_rotate(qix, qiy, qiz, qiw, dpx, dpy, dpz)
                t_anchor_component_local_position[i, 0] = alx
                t_anchor_component_local_position[i, 1] = aly
                t_anchor_component_local_position[i, 2] = alz
                t_reset_pending[i] = 0
                t_time_reset[i] = 0
                t_running[i] = 0
                t_keep_teleport_pending[i] = 0
                t_inertia_shift[i] = 0
                t_negative_scale_teleport[i] = 0
                if t_time[i] > float32(7200.0):
                    t_time[i] = t_time[i] - float32(3600.0)
                    t_old_time[i] = t_old_time[i] - float32(3600.0)
                    t_now_update[i] = t_now_update[i] - float32(3600.0)
                    t_old_update[i] = t_old_update[i] - float32(3600.0)
                    t_frame_update[i] = t_frame_update[i] - float32(3600.0)
                    t_frame_old[i] = t_frame_old[i] - float32(3600.0)
            i += stride
    grid.sync()


# ordered team-field names the frame_kernel consumes, in signature order after the
# four scalars; the engine builds the launch arg list from this single source of truth.
TEAM_KERNEL_FIELDS = (
    "enabled", "valid", "component_world_scale", "time_reset_pending",
    "time", "old_time", "now_update_time", "old_update_time", "frame_update_time",
    "frame_old_time", "frame_delta_time", "time_scale", "now_time_scale",
    "update_count", "skip_count", "running", "tether_compression",
    "frame_interpolation", "depth_inertia", "inertia_vector", "step_vector",
    "inertia_rotation", "step_rotation", "old_world_position", "velocity_weight",
    "damping_lut", "force_mode", "gravity_direction", "gravity", "gravity_ratio",
    "impact_force", "scale_ratio", "normal_axis_vector", "spring_limit_distance",
    "spring_normal_limit_ratio", "spring_power", "spring_noise",
    "wind_seed", "wind_synchronization", "wind_blend", "wind_turbulence",
    "wind_count", "wind_main", "wind_time", "wind_dirq", "wind_zone_turbulence",
    "wind_influence", "wind_depth_weight", "moving_wind_main", "wind_moving",
    "moving_wind_time", "moving_wind_dirq",
    "static_friction", "dynamic_friction", "particle_speed_limit",
    "angular_velocity", "centrifugal_acceleration", "rotation_axis",
    "now_world_position",
    "is_spring", "animation_pose_ratio", "init_scale", "distance_lut",
    "motion_use_max_distance", "motion_use_backstop", "motion_stiffness",
    "motion_backstop_radius", "radius_lut", "motion_max_distance_lut",
    "motion_backstop_lut",
    "bending_stiffness", "negative_scale_sign",
    "negative_scale_direction", "negative_scale_quaternion", "is_negative_scale",
    "component_world_position", "component_world_rotation",
    "old_component_world_position", "old_component_world_rotation",
    "old_component_world_scale",
    "frame_world_position", "frame_world_rotation", "frame_world_scale",
    "old_frame_world_position", "old_frame_world_rotation", "old_frame_world_scale",
    "anchor_position", "anchor_rotation",
    "old_anchor_position", "old_anchor_rotation", "anchor_component_local_position",
    "reset_pending", "keep_teleport_pending", "inertia_shift",
    "negative_scale_teleport",
    "now_world_rotation", "old_world_rotation",
    "step_move_inertia_ratio", "step_rotation_inertia_ratio",
    "local_inertia", "local_movement_speed_limit", "local_rotation_speed_limit",
    "gravity_dot", "init_local_gravity_direction", "gravity_falloff",
    "stablization_time", "blend_weight", "blend_weight_param", "distance_weight",
    "frame_moving_speed", "frame_moving_direction", "moving_wind_direction",
    "wind_frequency",
    "collision_mode", "limit_distance_lut", "negative_scale_matrix",
    "negative_scale_change", "frame_component_shift_vector",
    "frame_component_shift_rotation",
    "sync_target", "sync_top", "negative_scale_triangle_sign",
    "smoothing_velocity", "has_anchor", "had_anchor", "anchor_inertia",
    "world_inertia", "movement_inertia_smoothing", "movement_speed_limit",
    "rotation_speed_limit", "teleport_mode", "teleport_distance",
    "teleport_rotation", "culling_invisible", "wind_direction", "wind_zone_id",
    "angle_use_limit", "angle_use_restoration", "angle_limit_lut",
    "angle_limit_stiffness", "angle_restoration_lut",
    "angle_restoration_attenuation", "angle_restoration_gravity_falloff",
    "rotational_interpolation", "root_rotation",
)

PARTICLE_KERNEL_FIELDS = (
    "team", "local_positions", "local_normals", "local_tangents",
    "skin_indices", "skin_weights", "positions", "rotations",
    "next_positions", "velocity_positions", "step_basic_positions", "vertex_root",
    "old_anim_positions", "old_anim_rotations", "base_positions", "base_rotations",
    "step_basic_rotations", "depth", "velocities", "old_positions", "friction",
    "vertex_root_local", "collision_normals", "static_friction", "real_velocities",
    "attr_move", "vertex_local_positions", "vertex_local_rotations",
    "old_rotations", "display_positions", "vertex_bind_pose_rotations",
    "vertex_parent", "albuf_length", "albuf_local_pos", "albuf_local_rot",
    "albuf_restore", "albuf_rotation",
    "uv", "attr_zero_distance", "attr_invalid", "temp_base_positions",
    "temp_base_rotations", "normal_adjustment_rotations",
    "vertex_to_transform_rotations", "out_rotations",
)

TRANSFORM_KERNEL_FIELDS = ("world", "bind_pose")

# collider arena fields the frame_kernel consumes, in signature order after transforms
COLLIDER_KERNEL_FIELDS = (
    "team", "kind", "center", "size", "axis", "aligned", "enabled",
    "enabled_prev", "active", "input_positions", "input_rotations", "input_scales",
    "frame_positions", "frame_rotations", "frame_scales",
    "old_frame_positions", "old_frame_rotations", "now_positions", "now_rotations",
    "old_positions", "old_rotations",
    "work_radius", "work_old_pos", "work_next_pos", "work_rot", "work_inv_old_rot",
    "work_aabb_min", "work_aabb_max",
)

# static Program arrays, uploaded once; engine reads program.<attr>[<field>]
STATIC_KERNEL_FIELDS = (
    ("tether_particle", "tether", "particle"),
    ("tether_team", "tether", "team"),
    ("move_particle", "update_move", "particle"),
    ("move_team", "update_move", "team"),
    ("fixed_particle", "update_fixed", "particle"),
    ("fixed_team", "update_fixed", "team"),
    ("spring_particle", "spring", "particle"),
    ("spring_team", "spring", "team"),
    ("distance_target", "distance", "target"),
    ("distance_rest", "distance", "rest"),
    ("motion_particle", "motion", "particle"),
    ("motion_team", "motion", "team"),
    ("bending_team", "bending", "team"),
    ("bending_pair", "bending", "pair"),
    ("bending_rest", "bending", "rest"),
    ("bending_sign", "bending", "sign"),
    ("point_pair_collider", "point_pairs", "collider"),
    ("edge_pair_collider", "edge_pairs", "collider"),
    ("collision_edge", "collision_edges", "edge"),
    ("center_fixed_particle", "center_fixed", "particle"),
    ("angle_buffered_particle", "angle_buffered", "particle"),
    ("triangle_team", "triangles", "team"),
    ("triangle_particles", "triangles", "triangle"),
    ("v2t_triangle", "v2t", "triangle"),
    ("v2t_flip_normal", "v2t", "flip_normal"),
    ("v2t_flip_tangent", "v2t", "flip_tangent"),
)

# CSR gather tables (offsets + order uploaded from a program CsrTable attribute)
STATIC_CSR_FIELDS = (
    ("distance_csr_offsets", "distance_csr_order", "distance_csr"),
    ("point_pair_csr_offsets", "point_pair_csr_order", "point_pair_csr"),
    ("edge_pair_csr_offsets", "edge_pair_csr_order", "edge_pair_csr"),
    ("center_fixed_csr_offsets", "center_fixed_csr_order", "center_fixed_csr"),
    ("v2t_csr_offsets", "v2t_csr_order", "v2t_csr"),
)

# direct program arrays uploaded verbatim (level tables + flat index sets)
STATIC_DIRECT_FIELDS = (
    "fk_yes_offsets", "fk_yes", "fk_yes_parent", "fk_no_offsets", "fk_no",
    "baseline_entries",
    "angle_pass_offsets", "angle_pass_vertices", "angle_pass_parents",
    "postline_entry_offsets", "postline_entry_vertices",
    "postline_child_offsets", "postline_child_vertices", "display_update_move_mask",
)


# ---- G2e-7 blob aggregation registry (cache-preserving group-by-shape) ---------------------
# The megakernel takes one contiguous device blob per (dtype-family, per-row-shape) group, so
# each field is a plain AXIS-0 SLICE of a shaped blob -- no device-code reshape (numba-cuda links
# reshape_funcs.cu, which cache=True cannot pickle). offs[k]/lens[k] are the row base/count of
# slot k into blob_<group>; the slot index matches the reconstruction preamble atop frame_kernel.
# engine.build_blobs() assembles the group blobs from the same ordered field list and asserts
# group/per_row against this table. Adding a field for G3 = append its *_KERNEL_FIELDS entry + one
# layout row + one preamble slice (+ a new group blob only if its (family,shape) is new); the
# SIGNATURE changes only when a brand-new (family,shape) group appears.
RESIDENT_BLOB_GROUPS = (
    "u8_s",
    "f32_v3",
    "f32_s",
    "i32_s",
    "f32_v4",
    "f32_v16",
    "i8_s",
    "f32_m4x4",
    "f64_m4x4",
    "f32_v2",
    "f32_m4x3",
    "i32_v4",
    "f32_m2x3",
    "i32_v2",
    "i32_v3",
    "f32_v22",
    "f64_v3",
)

ZONE_BLOB_GROUPS = (
    "i32_s",
    "u8_s",
    "f32_s",
    "f32_v3",
    "f64_m4x4",
    "f32_v16",
)

RESIDENT_BLOB_LAYOUT = (
    ('t_enabled', 'u8_s', ()),
    ('t_valid', 'u8_s', ()),
    ('t_cws', 'f32_v3', (3,)),
    ('t_time_reset', 'u8_s', ()),
    ('t_time', 'f32_s', ()),
    ('t_old_time', 'f32_s', ()),
    ('t_now_update', 'f32_s', ()),
    ('t_old_update', 'f32_s', ()),
    ('t_frame_update', 'f32_s', ()),
    ('t_frame_old', 'f32_s', ()),
    ('t_frame_dt', 'f32_s', ()),
    ('t_time_scale', 'f32_s', ()),
    ('t_now_time_scale', 'f32_s', ()),
    ('t_update_count', 'i32_s', ()),
    ('t_skip_count', 'i32_s', ()),
    ('t_running', 'u8_s', ()),
    ('t_tether_compression', 'f32_s', ()),
    ('t_frame_interpolation', 'f32_s', ()),
    ('t_depth_inertia', 'f32_s', ()),
    ('t_inertia_vector', 'f32_v3', (3,)),
    ('t_step_vector', 'f32_v3', (3,)),
    ('t_inertia_rotation', 'f32_v4', (4,)),
    ('t_step_rotation', 'f32_v4', (4,)),
    ('t_old_world_position', 'f32_v3', (3,)),
    ('t_velocity_weight', 'f32_s', ()),
    ('t_damping_lut', 'f32_v16', (16,)),
    ('t_force_mode', 'i8_s', ()),
    ('t_gravity_direction', 'f32_v3', (3,)),
    ('t_gravity', 'f32_s', ()),
    ('t_gravity_ratio', 'f32_s', ()),
    ('t_impact_force', 'f32_v3', (3,)),
    ('t_scale_ratio', 'f32_s', ()),
    ('t_normal_axis_vector', 'f32_v3', (3,)),
    ('t_spring_limit_distance', 'f32_s', ()),
    ('t_spring_normal_limit_ratio', 'f32_s', ()),
    ('t_spring_power', 'f32_s', ()),
    ('t_spring_noise', 'f32_s', ()),
    ('t_wind_seed', 'i32_s', ()),
    ('t_wind_synchronization', 'f32_s', ()),
    ('t_wind_blend', 'f32_s', ()),
    ('t_wind_turbulence', 'f32_s', ()),
    ('t_wind_count', 'i8_s', ()),
    ('t_wind_main', 'f32_v4', (4,)),
    ('t_wind_time', 'f32_v4', (4,)),
    ('t_wind_dirq', 'f32_m4x4', (4, 4)),
    ('t_wind_zone_turbulence', 'f32_v4', (4,)),
    ('t_wind_influence', 'f32_s', ()),
    ('t_wind_depth_weight', 'f32_s', ()),
    ('t_moving_wind_main', 'f32_s', ()),
    ('t_wind_moving', 'f32_s', ()),
    ('t_moving_wind_time', 'f32_s', ()),
    ('t_moving_wind_dirq', 'f32_v4', (4,)),
    ('t_static_friction', 'f32_s', ()),
    ('t_dynamic_friction', 'f32_s', ()),
    ('t_particle_speed_limit', 'f32_s', ()),
    ('t_angular_velocity', 'f32_s', ()),
    ('t_centrifugal_acceleration', 'f32_s', ()),
    ('t_rotation_axis', 'f32_v3', (3,)),
    ('t_now_world_position', 'f32_v3', (3,)),
    ('t_is_spring', 'u8_s', ()),
    ('t_animation_pose_ratio', 'f32_s', ()),
    ('t_init_scale', 'f32_v3', (3,)),
    ('t_distance_lut', 'f32_v16', (16,)),
    ('t_motion_use_max_distance', 'u8_s', ()),
    ('t_motion_use_backstop', 'u8_s', ()),
    ('t_motion_stiffness', 'f32_s', ()),
    ('t_motion_backstop_radius', 'f32_s', ()),
    ('t_radius_lut', 'f32_v16', (16,)),
    ('t_motion_max_distance_lut', 'f32_v16', (16,)),
    ('t_motion_backstop_lut', 'f32_v16', (16,)),
    ('t_bending_stiffness', 'f32_s', ()),
    ('t_negative_scale_sign', 'f32_s', ()),
    ('t_negative_scale_direction', 'f32_v3', (3,)),
    ('t_negative_scale_quaternion', 'f32_v4', (4,)),
    ('t_is_negative_scale', 'u8_s', ()),
    ('t_component_world_position', 'f32_v3', (3,)),
    ('t_component_world_rotation', 'f32_v4', (4,)),
    ('t_old_component_world_position', 'f32_v3', (3,)),
    ('t_old_component_world_rotation', 'f32_v4', (4,)),
    ('t_old_component_world_scale', 'f32_v3', (3,)),
    ('t_frame_world_position', 'f32_v3', (3,)),
    ('t_frame_world_rotation', 'f32_v4', (4,)),
    ('t_frame_world_scale', 'f32_v3', (3,)),
    ('t_old_frame_world_position', 'f32_v3', (3,)),
    ('t_old_frame_world_rotation', 'f32_v4', (4,)),
    ('t_old_frame_world_scale', 'f32_v3', (3,)),
    ('t_anchor_position', 'f32_v3', (3,)),
    ('t_anchor_rotation', 'f32_v4', (4,)),
    ('t_old_anchor_position', 'f32_v3', (3,)),
    ('t_old_anchor_rotation', 'f32_v4', (4,)),
    ('t_anchor_component_local_position', 'f32_v3', (3,)),
    ('t_reset_pending', 'u8_s', ()),
    ('t_keep_teleport_pending', 'u8_s', ()),
    ('t_inertia_shift', 'u8_s', ()),
    ('t_negative_scale_teleport', 'u8_s', ()),
    ('t_now_world_rotation', 'f32_v4', (4,)),
    ('t_old_world_rotation', 'f32_v4', (4,)),
    ('t_step_move_inertia_ratio', 'f32_s', ()),
    ('t_step_rotation_inertia_ratio', 'f32_s', ()),
    ('t_local_inertia', 'f32_s', ()),
    ('t_local_movement_speed_limit', 'f32_s', ()),
    ('t_local_rotation_speed_limit', 'f32_s', ()),
    ('t_gravity_dot', 'f32_s', ()),
    ('t_init_local_gravity_direction', 'f32_v3', (3,)),
    ('t_gravity_falloff', 'f32_s', ()),
    ('t_stablization_time', 'f32_s', ()),
    ('t_blend_weight', 'f32_s', ()),
    ('t_blend_weight_param', 'f32_s', ()),
    ('t_distance_weight', 'f32_s', ()),
    ('t_frame_moving_speed', 'f32_s', ()),
    ('t_frame_moving_direction', 'f32_v3', (3,)),
    ('t_moving_wind_direction', 'f32_v3', (3,)),
    ('t_wind_frequency', 'f32_s', ()),
    ('t_collision_mode', 'i8_s', ()),
    ('t_limit_distance_lut', 'f32_v16', (16,)),
    ('t_negative_scale_matrix', 'f64_m4x4', (4, 4)),
    ('t_negative_scale_change', 'f32_v3', (3,)),
    ('t_frame_component_shift_vector', 'f32_v3', (3,)),
    ('t_frame_component_shift_rotation', 'f32_v4', (4,)),
    ('t_sync_target', 'i32_s', ()),
    ('t_sync_top', 'i32_s', ()),
    ('t_negative_scale_triangle_sign', 'f32_v2', (2,)),
    ('t_smoothing_velocity', 'f32_v3', (3,)),
    ('t_has_anchor', 'u8_s', ()),
    ('t_had_anchor', 'u8_s', ()),
    ('t_anchor_inertia', 'f32_s', ()),
    ('t_world_inertia', 'f32_s', ()),
    ('t_movement_inertia_smoothing', 'f32_s', ()),
    ('t_movement_speed_limit', 'f32_s', ()),
    ('t_rotation_speed_limit', 'f32_s', ()),
    ('t_teleport_mode', 'i8_s', ()),
    ('t_teleport_distance', 'f32_s', ()),
    ('t_teleport_rotation', 'f32_s', ()),
    ('t_culling_invisible', 'u8_s', ()),
    ('t_wind_direction', 'f32_m4x3', (4, 3)),
    ('t_wind_zone_id', 'i32_v4', (4,)),
    ('t_angle_use_limit', 'u8_s', ()),
    ('t_angle_use_restoration', 'u8_s', ()),
    ('t_angle_limit_lut', 'f32_v16', (16,)),
    ('t_angle_limit_stiffness', 'f32_s', ()),
    ('t_angle_restoration_lut', 'f32_v16', (16,)),
    ('t_angle_restoration_attenuation', 'f32_s', ()),
    ('t_angle_restoration_gravity_falloff', 'f32_s', ()),
    ('t_rotational_interpolation', 'f32_s', ()),
    ('t_root_rotation', 'f32_s', ()),
    ('p_team', 'i32_s', ()),
    ('p_local_positions', 'f32_v3', (3,)),
    ('p_local_normals', 'f32_v3', (3,)),
    ('p_local_tangents', 'f32_v3', (3,)),
    ('p_skin_indices', 'i32_v4', (4,)),
    ('p_skin_weights', 'f32_v4', (4,)),
    ('p_positions', 'f32_v3', (3,)),
    ('p_rotations', 'f32_v4', (4,)),
    ('p_next_positions', 'f32_v3', (3,)),
    ('p_velocity_positions', 'f32_v3', (3,)),
    ('p_step_basic_positions', 'f32_v3', (3,)),
    ('p_vertex_root', 'i32_s', ()),
    ('p_old_anim_positions', 'f32_v3', (3,)),
    ('p_old_anim_rotations', 'f32_v4', (4,)),
    ('p_base_positions', 'f32_v3', (3,)),
    ('p_base_rotations', 'f32_v4', (4,)),
    ('p_step_basic_rotations', 'f32_v4', (4,)),
    ('p_depth', 'f32_s', ()),
    ('p_velocities', 'f32_v3', (3,)),
    ('p_old_positions', 'f32_v3', (3,)),
    ('p_friction', 'f32_s', ()),
    ('p_vertex_root_local', 'i32_s', ()),
    ('p_collision_normals', 'f32_v3', (3,)),
    ('p_static_friction', 'f32_s', ()),
    ('p_real_velocities', 'f32_v3', (3,)),
    ('p_attr_move', 'u8_s', ()),
    ('p_vertex_local_positions', 'f32_v3', (3,)),
    ('p_vertex_local_rotations', 'f32_v4', (4,)),
    ('p_old_rotations', 'f32_v4', (4,)),
    ('p_display_positions', 'f32_v3', (3,)),
    ('p_vertex_bind_pose_rotations', 'f32_v4', (4,)),
    ('p_vertex_parent', 'i32_s', ()),
    ('p_albuf_length', 'f32_s', ()),
    ('p_albuf_local_pos', 'f32_v3', (3,)),
    ('p_albuf_local_rot', 'f32_v4', (4,)),
    ('p_albuf_restore', 'f32_v3', (3,)),
    ('p_albuf_rotation', 'f32_v4', (4,)),
    ('p_uv', 'f32_v2', (2,)),
    ('p_attr_zero_distance', 'u8_s', ()),
    ('p_attr_invalid', 'u8_s', ()),
    ('p_temp_base_positions', 'f32_v3', (3,)),
    ('p_temp_base_rotations', 'f32_v4', (4,)),
    ('p_normal_adjustment_rotations', 'f32_v4', (4,)),
    ('p_vertex_to_transform_rotations', 'f32_v4', (4,)),
    ('p_out_rotations', 'f32_v4', (4,)),
    ('x_world', 'f32_m4x4', (4, 4)),
    ('x_bind', 'f32_m4x4', (4, 4)),
    ('c_team', 'i32_s', ()),
    ('c_kind', 'i32_s', ()),
    ('c_center', 'f32_v3', (3,)),
    ('c_size', 'f32_v3', (3,)),
    ('c_axis', 'f32_v3', (3,)),
    ('c_aligned', 'u8_s', ()),
    ('c_enabled', 'u8_s', ()),
    ('c_enabled_prev', 'u8_s', ()),
    ('c_active', 'u8_s', ()),
    ('c_input_positions', 'f32_v3', (3,)),
    ('c_input_rotations', 'f32_v4', (4,)),
    ('c_input_scales', 'f32_v3', (3,)),
    ('c_frame_pos', 'f32_v3', (3,)),
    ('c_frame_rot', 'f32_v4', (4,)),
    ('c_frame_scl', 'f32_v3', (3,)),
    ('c_old_frame_pos', 'f32_v3', (3,)),
    ('c_old_frame_rot', 'f32_v4', (4,)),
    ('c_now_pos', 'f32_v3', (3,)),
    ('c_now_rot', 'f32_v4', (4,)),
    ('c_old_pos', 'f32_v3', (3,)),
    ('c_old_rot', 'f32_v4', (4,)),
    ('c_work_radius', 'f32_v2', (2,)),
    ('c_work_old_pos', 'f32_m2x3', (2, 3)),
    ('c_work_next_pos', 'f32_m2x3', (2, 3)),
    ('c_work_rot', 'f32_v4', (4,)),
    ('c_work_inv_old_rot', 'f32_v4', (4,)),
    ('c_work_aabb_min', 'f32_v3', (3,)),
    ('c_work_aabb_max', 'f32_v3', (3,)),
    ('st_tether_particle', 'i32_s', ()),
    ('st_tether_team', 'i32_s', ()),
    ('st_move_particle', 'i32_s', ()),
    ('st_move_team', 'i32_s', ()),
    ('st_fixed_particle', 'i32_s', ()),
    ('st_fixed_team', 'i32_s', ()),
    ('st_spring_particle', 'i32_s', ()),
    ('st_spring_team', 'i32_s', ()),
    ('st_distance_target', 'i32_s', ()),
    ('st_distance_rest', 'f32_s', ()),
    ('st_motion_particle', 'i32_s', ()),
    ('st_motion_team', 'i32_s', ()),
    ('st_bending_team', 'i32_s', ()),
    ('st_bending_pair', 'i32_v4', (4,)),
    ('st_bending_rest', 'f32_s', ()),
    ('st_bending_sign', 'i8_s', ()),
    ('st_point_pair_collider', 'i32_s', ()),
    ('st_edge_pair_collider', 'i32_s', ()),
    ('st_collision_edge', 'i32_v2', (2,)),
    ('st_center_fixed_particle', 'i32_s', ()),
    ('st_angle_buffered_particle', 'i32_s', ()),
    ('st_triangle_team', 'i32_s', ()),
    ('st_triangle_particles', 'i32_v3', (3,)),
    ('st_v2t_triangle', 'i32_s', ()),
    ('st_v2t_flip_normal', 'f32_s', ()),
    ('st_v2t_flip_tangent', 'f32_s', ()),
    ('csr_distance_offsets', 'i32_s', ()),
    ('csr_distance_order', 'i32_s', ()),
    ('csr_point_pair_offsets', 'i32_s', ()),
    ('csr_point_pair_order', 'i32_s', ()),
    ('csr_edge_pair_offsets', 'i32_s', ()),
    ('csr_edge_pair_order', 'i32_s', ()),
    ('csr_center_fixed_offsets', 'i32_s', ()),
    ('csr_center_fixed_order', 'i32_s', ()),
    ('csr_v2t_offsets', 'i32_s', ()),
    ('csr_v2t_order', 'i32_s', ()),
    ('fk_yes_offsets', 'i32_s', ()),
    ('fk_yes', 'i32_s', ()),
    ('fk_yes_parent', 'i32_s', ()),
    ('fk_no_offsets', 'i32_s', ()),
    ('fk_no', 'i32_s', ()),
    ('baseline_entries', 'i32_s', ()),
    ('angle_pass_offsets', 'i32_s', ()),
    ('angle_pass_vertices', 'i32_s', ()),
    ('angle_pass_parents', 'i32_s', ()),
    ('postline_entry_offsets', 'i32_s', ()),
    ('postline_entry_vertices', 'i32_s', ()),
    ('postline_child_offsets', 'i32_s', ()),
    ('postline_child_vertices', 'i32_s', ()),
    ('st_display_update_move_mask', 'u8_s', ()),
    ('sc_dcorr', 'f32_v3', (3,)),
    ('sc_dcorr_fixed', 'i32_v3', (3,)),
    ('sc_dcount', 'i32_s', ()),
    ('sc_col_friction_fixed', 'i32_s', ()),
    ('sc_col_normal_fixed', 'i32_v3', (3,)),
    ('sc_sync', 'f32_v22', (22,)),
    ('sc_tri_normal_f64', 'f64_v3', (3,)),
    ('sc_tri_tangent_f64', 'f64_v3', (3,)),
)

ZONE_BLOB_LAYOUT = (
    ('z_zone_id', 'i32_s', ()),
    ('z_mode', 'i32_s', ()),
    ('z_is_addition', 'u8_s', ()),
    ('z_main', 'f32_s', ()),
    ('z_turbulence', 'f32_s', ()),
    ('z_world_position', 'f32_v3', (3,)),
    ('z_world_direction', 'f32_v3', (3,)),
    ('z_world_to_local', 'f64_m4x4', (4, 4)),
    ('z_size', 'f32_v3', (3,)),
    ('z_zone_volume', 'f32_s', ()),
    ('z_attenuation_lut', 'f32_v16', (16,)),
)
