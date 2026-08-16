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

from numba import cuda, float32, float64, int32
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

ALL_PHASES = int32(-1)

MAX_SIM_COUNT = 5

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


@cuda.jit(cache=True)
def frame_kernel(phase_mask, sub_begin, sub_end,
                 fdt, sim_dt, max_sim_count, global_time_scale,
                 power0, power1, power2, power3,
                 t_enabled, t_valid, t_cws, t_time_reset,
                 t_time, t_old_time, t_now_update, t_old_update, t_frame_update, t_frame_old,
                 t_frame_dt, t_time_scale, t_now_time_scale, t_update_count, t_skip_count,
                 t_running, t_tether_compression,
                 t_frame_interpolation, t_depth_inertia, t_inertia_vector, t_step_vector,
                 t_inertia_rotation, t_step_rotation, t_old_world_position, t_velocity_weight,
                 t_damping_lut, t_force_mode, t_gravity_direction, t_gravity, t_gravity_ratio,
                 t_impact_force, t_scale_ratio, t_normal_axis_vector, t_spring_limit_distance,
                 t_spring_normal_limit_ratio, t_spring_power, t_spring_noise,
                 t_wind_seed, t_wind_synchronization, t_wind_blend, t_wind_turbulence,
                 t_wind_count, t_wind_main, t_wind_time, t_wind_dirq, t_wind_zone_turbulence,
                 t_wind_influence, t_wind_depth_weight, t_moving_wind_main, t_wind_moving,
                 t_moving_wind_time, t_moving_wind_dirq,
                 t_static_friction, t_dynamic_friction, t_particle_speed_limit,
                 t_angular_velocity, t_centrifugal_acceleration, t_rotation_axis,
                 t_now_world_position,
                 t_is_spring, t_animation_pose_ratio, t_init_scale, t_distance_lut,
                 t_motion_use_max_distance, t_motion_use_backstop, t_motion_stiffness,
                 t_motion_backstop_radius, t_radius_lut, t_motion_max_distance_lut,
                 t_motion_backstop_lut,
                 t_bending_stiffness, t_negative_scale_sign,
                 t_negative_scale_direction, t_negative_scale_quaternion, t_is_negative_scale,
                 t_component_world_position, t_component_world_rotation,
                 t_old_component_world_position, t_old_component_world_rotation,
                 t_old_component_world_scale,
                 t_frame_world_position, t_frame_world_rotation, t_frame_world_scale,
                 t_old_frame_world_position, t_old_frame_world_rotation, t_old_frame_world_scale,
                 t_anchor_position, t_anchor_rotation,
                 t_old_anchor_position, t_old_anchor_rotation, t_anchor_component_local_position,
                 t_reset_pending, t_keep_teleport_pending, t_inertia_shift,
                 t_negative_scale_teleport,
                 t_now_world_rotation, t_old_world_rotation,
                 t_step_move_inertia_ratio, t_step_rotation_inertia_ratio,
                 t_local_inertia, t_local_movement_speed_limit, t_local_rotation_speed_limit,
                 t_gravity_dot, t_init_local_gravity_direction, t_gravity_falloff,
                 t_stablization_time, t_blend_weight, t_blend_weight_param, t_distance_weight,
                 t_frame_moving_speed, t_frame_moving_direction, t_moving_wind_direction,
                 t_wind_frequency,
                 t_collision_mode, t_limit_distance_lut, t_negative_scale_matrix,
                 t_negative_scale_change, t_frame_component_shift_vector,
                 t_frame_component_shift_rotation,
                 p_team, p_local_positions, p_local_normals, p_local_tangents,
                 p_skin_indices, p_skin_weights, p_positions, p_rotations,
                 p_next_positions, p_velocity_positions, p_step_basic_positions, p_vertex_root,
                 p_old_anim_positions, p_old_anim_rotations, p_base_positions, p_base_rotations,
                 p_step_basic_rotations, p_depth, p_velocities, p_old_positions, p_friction,
                 p_vertex_root_local, p_collision_normals, p_static_friction, p_real_velocities,
                 p_attr_move, p_vertex_local_positions, p_vertex_local_rotations,
                 x_world, x_bind,
                 c_team, c_kind, c_center, c_size, c_axis, c_aligned, c_enabled,
                 c_enabled_prev, c_active, c_input_positions, c_input_rotations, c_input_scales,
                 c_frame_pos, c_frame_rot, c_frame_scl, c_old_frame_pos, c_old_frame_rot,
                 c_now_pos, c_now_rot, c_old_pos, c_old_rot,
                 c_work_radius, c_work_old_pos, c_work_next_pos, c_work_rot, c_work_inv_old_rot,
                 c_work_aabb_min, c_work_aabb_max,
                 st_tether_particle, st_tether_team,
                 st_move_particle, st_move_team, st_fixed_particle, st_fixed_team,
                 st_spring_particle, st_spring_team,
                 st_distance_target, st_distance_rest,
                 st_motion_particle, st_motion_team,
                 st_bending_team, st_bending_pair, st_bending_rest, st_bending_sign,
                 st_point_pair_collider, st_edge_pair_collider, st_collision_edge,
                 csr_distance_offsets, csr_distance_order,
                 csr_point_pair_offsets, csr_point_pair_order,
                 csr_edge_pair_offsets, csr_edge_pair_order,
                 fk_yes_offsets, fk_yes, fk_yes_parent, fk_no_offsets, fk_no, baseline_entries,
                 sc_dcorr, sc_dcorr_fixed, sc_dcount, sc_col_friction_fixed, sc_col_normal_fixed):
    grid = cg.this_grid()
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    num_teams = t_enabled.shape[0]

    num_particles = p_team.shape[0]
    num_colliders = c_team.shape[0]

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

        # --- S4 baseline.run FK: per-level grid.sync wall (parent written at level < L) ---
        for lvl in range(num_fk_levels):
            if phase_mask & PHASE_BASELINE:
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
                    i += stride
            grid.sync()
            # oracle applies each level's yes (skin from parent) BEFORE its no (negative-
            # scale root flip); a root can be both a parent here and a no-entry, so the
            # yes reads must complete before the no writes.
            if phase_mask & PHASE_BASELINE:
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
                    i += stride
            grid.sync()
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
)

PARTICLE_KERNEL_FIELDS = (
    "team", "local_positions", "local_normals", "local_tangents",
    "skin_indices", "skin_weights", "positions", "rotations",
    "next_positions", "velocity_positions", "step_basic_positions", "vertex_root",
    "old_anim_positions", "old_anim_rotations", "base_positions", "base_rotations",
    "step_basic_rotations", "depth", "velocities", "old_positions", "friction",
    "vertex_root_local", "collision_normals", "static_friction", "real_velocities",
    "attr_move", "vertex_local_positions", "vertex_local_rotations",
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
)

# CSR gather tables (offsets + order uploaded from a program CsrTable attribute)
STATIC_CSR_FIELDS = (
    ("distance_csr_offsets", "distance_csr_order", "distance_csr"),
    ("point_pair_csr_offsets", "point_pair_csr_order", "point_pair_csr"),
    ("edge_pair_csr_offsets", "edge_pair_csr_order", "edge_pair_csr"),
)

# direct program arrays uploaded verbatim (level tables + flat index sets)
STATIC_DIRECT_FIELDS = (
    "fk_yes_offsets", "fk_yes", "fk_yes_parent", "fk_no_offsets", "fk_no",
    "baseline_entries",
)
