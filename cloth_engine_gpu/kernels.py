import math

from numba import cuda, float32, float64, int8, int32, uint8
from numba.cuda import libdevice

from cloth_kernel import defs as _defs

from . import dmath


SELF_COLLISION_SCR = float32(_defs.SELF_COLLISION_SCR)
SELF_COLLISION_SOLVER_ITERATION = _defs.SELF_COLLISION_SOLVER_ITERATION
SELF_COLLISION_INTERSECT_DIV = int32(_defs.SELF_COLLISION_INTERSECT_DIV)
SELF_COLLISION_UNIFORM_GRID_SCALE = float32(_defs.SELF_COLLISION_UNIFORM_GRID_SCALE)
SELF_COLLISION_FIXED_MASS = float32(_defs.SELF_COLLISION_FIXED_MASS)
SELF_COLLISION_FRICTION_MASS = float32(_defs.SELF_COLLISION_FRICTION_MASS)
SELF_COLLISION_CLOTH_MASS = float32(_defs.SELF_COLLISION_CLOTH_MASS)
SELF_COLLISION_POINT_TRIANGLE_ANGLE_COS = float32(_defs.SELF_COLLISION_POINT_TRIANGLE_ANGLE_COS)
SCL_EE_COUNT = 0
SCL_PT_COUNT = 1
SCL_IP_COUNT = 2
SCL_ERROR = 3
SCL_USE_INTERSECT = 4
SCL_FRAME_INDEX = 5

MAX_SIM_COUNT = _defs.MAX_SIMULATION_COUNT_HIGH

SCAL_FRAME_DT = 0
SCAL_SIM_DT = 1
SCAL_TIME_SCALE = 2
SCAL_POWER0 = 3
SCAL_POWER1 = 4
SCAL_POWER2 = 5
SCAL_POWER3 = 6
SCAL_F_LEN = 8

SCAL_MAX_SIM = 0
SCAL_N_ZONES = 1
SCAL_SUB_END = 2
SCAL_I_LEN = 4

WIND_ZONE_SLOTS = _defs.WIND_ZONE_SLOTS
ZONE_BOX = int32(1)
ZONE_SPHERE_DIR = int32(2)
ZONE_SPHERE_RADIAL = int32(3)
TELEPORT_RESET = int32(_defs.TELEPORT_RESET)

TETHER_STRETCH_LIMIT = float32(_defs.TETHER_STRETCH_LIMIT)
TETHER_STIFFNESS_WIDTH = float32(_defs.TETHER_STIFFNESS_WIDTH)
TETHER_COMPRESSION_STIFFNESS = float32(_defs.TETHER_COMPRESSION_STIFFNESS)
TETHER_STRETCH_STIFFNESS = float32(_defs.TETHER_STRETCH_STIFFNESS)
TETHER_COMPRESSION_VELOCITY_ATTENUATION = float32(_defs.TETHER_COMPRESSION_VELOCITY_ATTENUATION)
TETHER_STRETCH_VELOCITY_ATTENUATION = float32(_defs.TETHER_STRETCH_VELOCITY_ATTENUATION)
EPSILON = float32(_defs.EPSILON)

WIND_BASE_SPEED = float32(_defs.WIND_BASE_SPEED)
WIND_TURBULENCE_ANGLE = float32(_defs.WIND_TURBULENCE_ANGLE)
WIND_MAX_TIME = float32(_defs.WIND_MAX_TIME)
WIND_ZONE_MIN_MAIN = float32(_defs.WIND_ZONE_MIN_MAIN)
WIND_MIN_SPEED = float32(_defs.WIND_MIN_SPEED)
DEG2RAD = float32(math.pi / 180.0)
RAD2DEG = float32(180.0 / math.pi)

FORCE_VELOCITY_ADD = int32(_defs.FORCE_VELOCITY_ADD)
FORCE_VELOCITY_ADD_WITHOUT_DEPTH = int32(_defs.FORCE_VELOCITY_ADD_WITHOUT_DEPTH)
FORCE_VELOCITY_CHANGE = int32(_defs.FORCE_VELOCITY_CHANGE)
FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH = int32(_defs.FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH)

BONE_SPRING_FIX_MASS = float32(_defs.BONE_SPRING_FIX_MASS)
BONE_CLOTH_FIX_MASS = float32(_defs.BONE_CLOTH_FIX_MASS)
DISTANCE_HORIZONTAL_STIFFNESS = float32(_defs.DISTANCE_HORIZONTAL_STIFFNESS)
DISTANCE_VELOCITY_ATTENUATION = float32(_defs.DISTANCE_VELOCITY_ATTENUATION)

VOLUME_SIGN = int32(_defs.VOLUME_SIGN)
VOLUME_SCALE = float32(_defs.VOLUME_SCALE)
BENDING_FIX_INV_MASS = float32(0.01)
ONE_SIXTH = float32(1.0 / 6.0)
TO_FIXED = float32(1e6)

ANGLE_ITERATION = _defs.ANGLE_LIMIT_ITERATION
ANGLE_LIMIT_ROT_RATIO = float32(_defs.ANGLE_LIMIT_ROTATION_RATIO)
ANGLE_LIMIT_ATTENUATION = float32(_defs.ANGLE_LIMIT_ATTENUATION)

FRICTION_MASS = float32(_defs.FRICTION_MASS)
COLLIDER_SPHERE = int32(_defs.COLLIDER_SPHERE)
COLLIDER_CAPSULE = int32(_defs.COLLIDER_CAPSULE)
COLLIDER_PLANE = int32(_defs.COLLIDER_PLANE)
COLLISION_POINT = int32(_defs.COLLISION_POINT)
COLLISION_EDGE = int32(_defs.COLLISION_EDGE)
INF = float32(math.inf)
MAX_DISTANCE_RATIO_FUTURE_PREDICTION = float32(_defs.MAX_DISTANCE_RATIO_FUTURE_PREDICTION)


@cuda.jit(device=True)
def team_frame_mask(enabled, valid, cws, i):
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
        stiffness = dmath.saturate((compression_limit - ratio) / TETHER_STIFFNESS_WIDTH) \
            * TETHER_COMPRESSION_STIFFNESS
        attenuation = TETHER_COMPRESSION_VELOCITY_ATTENUATION
    else:
        dist = distance - stretch_limit * calc_distance
        stiffness = dmath.saturate((ratio - stretch_limit) / TETHER_STIFFNESS_WIDTH) \
            * TETHER_STRETCH_STIFFNESS
        attenuation = TETHER_STRETCH_VELOCITY_ATTENUATION
    inv = float32(1.0) / (distance if distance > float32(1e-30) else float32(1.0))
    scale = dist * stiffness * inv
    ax = vx * scale
    ay = vy * scale
    az = vz * scale
    next_positions[idx, 0] += ax
    next_positions[idx, 1] += ay
    next_positions[idx, 2] += az
    velocity_positions[idx, 0] += ax * attenuation
    velocity_positions[idx, 1] += ay * attenuation
    velocity_positions[idx, 2] += az * attenuation


@cuda.jit(device=True)
def do_wind_blend(wind_main, time, dqx, dqy, dqz, dqw, zone_turbulence,
                  blend, turbulence_param, wind_position):
    active = wind_main >= WIND_MIN_SPEED
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
def _neg_transform_point(arr_pos, ci, m):
    px, py, pz = dmath.transform_point(m, arr_pos[ci, 0], arr_pos[ci, 1], arr_pos[ci, 2])
    arr_pos[ci, 0] = px
    arr_pos[ci, 1] = py
    arr_pos[ci, 2] = pz


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
                          c_input_positions, c_input_rotations, c_input_tips, c_input_radii,
                          c_frame_pos, c_frame_rot, c_frame_tip, c_frame_radius,
                          c_old_frame_pos, c_old_frame_rot, c_old_frame_tip,
                          c_now_pos, c_now_rot, c_now_tip,
                          c_old_pos, c_old_rot, c_old_tip,
                          t_reset_pending, t_neg_teleport, t_neg_matrix, t_neg_change,
                          t_inertia_shift, t_shift_vec, t_shift_rot, t_old_cwp):
    enabled_now = c_enabled[ci] != 0
    rising = enabled_now and (c_enabled_prev[ci] == 0)
    c_active[ci] = int32(1) if enabled_now else int32(0)
    c_enabled_prev[ci] = int32(1) if enabled_now else int32(0)
    if not enabled_now:
        return
    team = c_team[ci]
    for j in range(3):
        c_frame_pos[ci, j] = c_input_positions[ci, j]
        c_frame_tip[ci, j] = c_input_tips[ci, j]
    for j in range(4):
        c_frame_rot[ci, j] = c_input_rotations[ci, j]
    c_frame_radius[ci, 0] = c_input_radii[ci, 0]
    c_frame_radius[ci, 1] = c_input_radii[ci, 1]
    reset = (t_reset_pending[team] != 0) or rising
    if reset:
        for j in range(3):
            fp = c_frame_pos[ci, j]
            c_old_frame_pos[ci, j] = fp
            c_now_pos[ci, j] = fp
            c_old_pos[ci, j] = fp
            ft = c_frame_tip[ci, j]
            c_old_frame_tip[ci, j] = ft
            c_now_tip[ci, j] = ft
            c_old_tip[ci, j] = ft
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
        _neg_transform_point(c_old_frame_tip, ci, m)
        _neg_transform_point(c_now_tip, ci, m)
        _neg_transform_point(c_old_tip, ci, m)
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
        _shift_point(c_old_frame_tip, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_point(c_now_tip, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_point(c_old_tip, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)


@cuda.jit(device=True)
def do_collider_start_step(ci, c_team, c_kind,
                           c_frame_pos, c_frame_rot, c_frame_tip, c_frame_radius,
                           c_old_frame_pos, c_old_frame_rot, c_old_frame_tip,
                           c_now_pos, c_now_rot, c_now_tip,
                           c_old_pos, c_old_rot, c_old_tip,
                           c_work_rot, c_work_inv_old_rot,
                           c_work_radius, c_work_old_pos, c_work_next_pos,
                           c_work_aabb_min, c_work_aabb_max,
                           t_frame_interp, t_step_mir, t_step_rir):
    team = c_team[ci]
    t = t_frame_interp[team]
    posx = dmath.lerp(c_old_frame_pos[ci, 0], c_frame_pos[ci, 0], t)
    posy = dmath.lerp(c_old_frame_pos[ci, 1], c_frame_pos[ci, 1], t)
    posz = dmath.lerp(c_old_frame_pos[ci, 2], c_frame_pos[ci, 2], t)
    tipx = dmath.lerp(c_old_frame_tip[ci, 0], c_frame_tip[ci, 0], t)
    tipy = dmath.lerp(c_old_frame_tip[ci, 1], c_frame_tip[ci, 1], t)
    tipz = dmath.lerp(c_old_frame_tip[ci, 2], c_frame_tip[ci, 2], t)
    rotx, roty, rotz, rotw = dmath.quat_slerp(
        c_old_frame_rot[ci, 0], c_old_frame_rot[ci, 1], c_old_frame_rot[ci, 2], c_old_frame_rot[ci, 3],
        c_frame_rot[ci, 0], c_frame_rot[ci, 1], c_frame_rot[ci, 2], c_frame_rot[ci, 3], t)
    c_now_pos[ci, 0] = posx
    c_now_pos[ci, 1] = posy
    c_now_pos[ci, 2] = posz
    c_now_tip[ci, 0] = tipx
    c_now_tip[ci, 1] = tipy
    c_now_tip[ci, 2] = tipz
    c_now_rot[ci, 0] = rotx
    c_now_rot[ci, 1] = roty
    c_now_rot[ci, 2] = rotz
    c_now_rot[ci, 3] = rotw
    mir = t_step_mir[team]
    rir = t_step_rir[team]
    opx = dmath.lerp(c_old_pos[ci, 0], posx, mir)
    opy = dmath.lerp(c_old_pos[ci, 1], posy, mir)
    opz = dmath.lerp(c_old_pos[ci, 2], posz, mir)
    otx = dmath.lerp(c_old_tip[ci, 0], tipx, mir)
    oty = dmath.lerp(c_old_tip[ci, 1], tipy, mir)
    otz = dmath.lerp(c_old_tip[ci, 2], tipz, mir)
    orx, ory, orz, orw = dmath.quat_slerp(
        c_old_rot[ci, 0], c_old_rot[ci, 1], c_old_rot[ci, 2], c_old_rot[ci, 3],
        rotx, roty, rotz, rotw, rir)
    c_old_pos[ci, 0] = opx
    c_old_pos[ci, 1] = opy
    c_old_pos[ci, 2] = opz
    c_old_tip[ci, 0] = otx
    c_old_tip[ci, 1] = oty
    c_old_tip[ci, 2] = otz
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
        radius = c_frame_radius[ci, 0]
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
        start_radius = c_frame_radius[ci, 0]
        end_radius = c_frame_radius[ci, 1]
        sox = opx
        soy = opy
        soz = opz
        eox = otx
        eoy = oty
        eoz = otz
        snx = posx
        sny = posy
        snz = posz
        enx = tipx
        eny = tipy
        enz = tipz
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
        nx, ny, nz = dmath.quat_rotate(rotx, roty, rotz, rotw, float32(0.0), float32(0.0), float32(1.0))
        c_work_old_pos[ci, 0, 0] = nx
        c_work_old_pos[ci, 0, 1] = ny
        c_work_old_pos[ci, 0, 2] = nz
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
def do_collider_end_step(ci, c_now_pos, c_now_rot, c_now_tip, c_old_pos, c_old_rot, c_old_tip):
    for j in range(3):
        c_old_pos[ci, j] = c_now_pos[ci, j]
        c_old_tip[ci, j] = c_now_tip[ci, j]
    for j in range(4):
        c_old_rot[ci, j] = c_now_rot[ci, j]


@cuda.jit(device=True)
def do_collider_frame_post(ci, c_frame_pos, c_frame_rot, c_frame_tip,
                           c_old_frame_pos, c_old_frame_rot, c_old_frame_tip):
    for j in range(3):
        c_old_frame_pos[ci, j] = c_frame_pos[ci, j]
        c_old_frame_tip[ci, j] = c_frame_tip[ci, j]
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


@cuda.jit(device=True)
def do_self_update_primitive(prim, axes, a_team, a_particles, a_fix, a_ignore, a_prim_depth,
                             a_inv_mass, a_thickness, a_aabb_min, a_aabb_max, a_intersect, a_use,
                             t_use_flag, t_thickness_lut, t_cloth_mass, t_scale_ratio,
                             t_enabled, t_valid, t_cws, t_update_count,
                             p_next, p_old, p_friction, p_iflag, scl_counts, scl_max_fixed, k):
    team = a_team[prim]
    if not (team_frame_mask(t_enabled, t_valid, t_cws, team) and t_update_count[team] > k
            and t_use_flag[team] != 0):
        a_use[prim] = uint8(0)
        return
    a_use[prim] = uint8(1)
    fix_mask = a_fix[prim]
    thickness = dmath.evaluate_team_lut(t_thickness_lut, team, a_prim_depth[prim]) * t_scale_ratio[team]
    a_thickness[prim] = thickness
    cloth_mass = t_cloth_mass[team]
    use_intersect = scl_counts[SCL_USE_INTERSECT] != 0
    imask = int32(0)
    lowx = float32(1e30); lowy = float32(1e30); lowz = float32(1e30)
    highx = float32(-1e30); highy = float32(-1e30); highz = float32(-1e30)
    for slot in range(axes):
        raw = a_particles[prim, slot]
        pp = raw if raw >= 0 else int32(0)
        fixed = ((fix_mask >> slot) & int32(1)) != 0
        a_inv_mass[prim, slot] = dmath.calc_self_collision_inverse_mass(
            p_friction[pp], fixed, cloth_mass)
        nx = p_next[pp, 0]; ny = p_next[pp, 1]; nz = p_next[pp, 2]
        ox = p_old[pp, 0]; oy = p_old[pp, 1]; oz = p_old[pp, 2]
        slx = nx if nx < ox else ox
        shx = nx if nx > ox else ox
        sly = ny if ny < oy else oy
        shy = ny if ny > oy else oy
        slz = nz if nz < oz else oz
        shz = nz if nz > oz else oz
        if slx < lowx:
            lowx = slx
        if shx > highx:
            highx = shx
        if sly < lowy:
            lowy = sly
        if shy > highy:
            highy = shy
        if slz < lowz:
            lowz = slz
        if shz > highz:
            highz = shz
        if use_intersect and p_iflag[pp] != 0:
            imask = imask | (int32(1) << slot)
    a_intersect[prim] = uint8(imask)
    a_aabb_min[prim, 0] = lowx - thickness
    a_aabb_min[prim, 1] = lowy - thickness
    a_aabb_min[prim, 2] = lowz - thickness
    a_aabb_max[prim, 0] = highx + thickness
    a_aabb_max[prim, 1] = highy + thickness
    a_aabb_max[prim, 2] = highz + thickness
    size = highx - lowx
    ey = highy - lowy
    ez = highz - lowz
    if ey > size:
        size = ey
    if ez > size:
        size = ez
    if a_ignore[prim] == 0:
        cuda.atomic.max(scl_max_fixed, team, int32(size * TO_FIXED))


@cuda.jit(device=True)
def self_aabb_overlap(a_min, a_max, i, b_min, b_max, j):
    return (a_min[i, 0] <= b_max[j, 0] and a_max[i, 0] >= b_min[j, 0]
            and a_min[i, 1] <= b_max[j, 1] and a_max[i, 1] >= b_min[j, 1]
            and a_min[i, 2] <= b_max[j, 2] and a_max[i, 2] >= b_min[j, 2])


@cuda.jit(device=True)
def self_connection_shared(a_particles, i, b_particles, j):
    for x in range(3):
        pa = a_particles[i, x]
        if pa >= 0:
            for y in range(3):
                pb = b_particles[j, y]
                if pb >= 0 and pa == pb:
                    return True
    return False


@cuda.jit(device=True)
def self_ee_geometry(my_edge, tgt_edge, thickness, sfe_particles, p_next, p_old):
    scr = thickness * SELF_COLLISION_SCR
    a0 = sfe_particles[my_edge, 0]; a1 = sfe_particles[my_edge, 1]
    b0 = sfe_particles[tgt_edge, 0]; b1 = sfe_particles[tgt_edge, 1]
    s, t, c1x, c1y, c1z, c2x, c2y, c2z = dmath.closest_pt_segment_segment(
        p_old[a0, 0], p_old[a0, 1], p_old[a0, 2], p_old[a1, 0], p_old[a1, 1], p_old[a1, 2],
        p_old[b0, 0], p_old[b0, 1], p_old[b0, 2], p_old[b1, 0], p_old[b1, 1], p_old[b1, 2])
    cdx = c1x - c2x; cdy = c1y - c2y; cdz = c1z - c2z
    clen = dmath.length3(cdx, cdy, cdz)
    ok = clen >= float32(1e-9)
    safe = clen if clen > float32(1e-30) else float32(1.0)
    nx = cdx / safe; ny = cdy / safe; nz = cdz / safe
    dax = dmath.lerp(p_next[a0, 0] - p_old[a0, 0], p_next[a1, 0] - p_old[a1, 0], s)
    day = dmath.lerp(p_next[a0, 1] - p_old[a0, 1], p_next[a1, 1] - p_old[a1, 1], s)
    daz = dmath.lerp(p_next[a0, 2] - p_old[a0, 2], p_next[a1, 2] - p_old[a1, 2], s)
    dbx = dmath.lerp(p_next[b0, 0] - p_old[b0, 0], p_next[b1, 0] - p_old[b1, 0], t)
    dby = dmath.lerp(p_next[b0, 1] - p_old[b0, 1], p_next[b1, 1] - p_old[b1, 1], t)
    dbz = dmath.lerp(p_next[b0, 2] - p_old[b0, 2], p_next[b1, 2] - p_old[b1, 2], t)
    l = clen + (nx * dax + ny * day + nz * daz) - (nx * dbx + ny * dby + nz * dbz)
    ok = ok and (l <= (thickness + scr))
    return (s, t, nx, ny, nz, ok)


@cuda.jit(device=True)
def self_pt_geometry(point_prim, tri_prim, thickness, first, sfp_particles, sft_particles,
                     p_next, p_old):
    scr = thickness * SELF_COLLISION_SCR
    pp = sfp_particles[point_prim, 0]
    t0 = sft_particles[tri_prim, 0]; t1 = sft_particles[tri_prim, 1]; t2 = sft_particles[tri_prim, 2]
    oax = p_old[pp, 0]; oay = p_old[pp, 1]; oaz = p_old[pp, 2]
    ob0x = p_old[t0, 0]; ob0y = p_old[t0, 1]; ob0z = p_old[t0, 2]
    ob1x = p_old[t1, 0]; ob1y = p_old[t1, 1]; ob1z = p_old[t1, 2]
    ob2x = p_old[t2, 0]; ob2y = p_old[t2, 1]; ob2z = p_old[t2, 2]
    dax = p_next[pp, 0] - oax; day = p_next[pp, 1] - oay; daz = p_next[pp, 2] - oaz
    db0x = p_next[t0, 0] - ob0x; db0y = p_next[t0, 1] - ob0y; db0z = p_next[t0, 2] - ob0z
    db1x = p_next[t1, 0] - ob1x; db1y = p_next[t1, 1] - ob1y; db1z = p_next[t1, 2] - ob1z
    db2x = p_next[t2, 0] - ob2x; db2y = p_next[t2, 1] - ob2y; db2z = p_next[t2, 2] - ob2z
    cpx, cpy, cpz, u, v, w = dmath.closest_pt_point_triangle(
        oax, oay, oaz, ob0x, ob0y, ob0z, ob1x, ob1y, ob1z, ob2x, ob2y, ob2z)
    dtx = db0x * u + db1x * v + db2x * w
    dty = db0y * u + db1y * v + db2y * w
    dtz = db0z * u + db1z * v + db2z * w
    cvx = cpx - oax; cvy = cpy - oay; cvz = cpz - oaz
    cvlen = dmath.length3(cvx, cvy, cvz)
    ok = cvlen > EPSILON
    safe = cvlen if cvlen > float32(1e-30) else float32(1.0)
    nx = cvx / safe; ny = cvy / safe; nz = cvz / safe
    l = cvlen - (nx * dax + ny * day + nz * daz) + (nx * dtx + ny * dty + nz * dtz)
    ok = ok and (l < (thickness + scr))
    sign = float32(0.0)
    if first:
        otnx, otny, otnz = dmath.triangle_normal(ob0x, ob0y, ob0z, ob1x, ob1y, ob1z, ob2x, ob2y, ob2z)
        n2x, n2y, n2z = dmath.normalize3(oax - cpx, oay - cpy, oaz - cpz)
        d = otnx * n2x + otny * n2y + otnz * n2z
        ok = ok and (libdevice.fabsf(d) >= SELF_COLLISION_POINT_TRIANGLE_ANGLE_COS)
        sign = dmath.fsign(d)
    return (ok, sign)


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

COLLIDER_KERNEL_FIELDS = (
    "team", "kind", "enabled", "enabled_prev", "active",
    "input_positions", "input_rotations", "input_tips", "input_radii",
    "frame_positions", "frame_rotations", "frame_tips", "frame_radii",
    "old_frame_positions", "old_frame_rotations", "old_frame_tips",
    "now_positions", "now_rotations", "now_tips",
    "old_positions", "old_rotations", "old_tips",
    "work_radius", "work_old_pos", "work_next_pos", "work_rot", "work_inv_old_rot",
    "work_aabb_min", "work_aabb_max",
)

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

STATIC_CSR_FIELDS = (
    ("distance_csr_offsets", "distance_csr_order", "distance_csr"),
    ("point_pair_csr_offsets", "point_pair_csr_order", "point_pair_csr"),
    ("edge_pair_csr_offsets", "edge_pair_csr_order", "edge_pair_csr"),
    ("center_fixed_csr_offsets", "center_fixed_csr_order", "center_fixed_csr"),
    ("v2t_csr_offsets", "v2t_csr_order", "v2t_csr"),
)

STATIC_DIRECT_FIELDS = (
    "fk_yes_offsets", "fk_yes", "fk_yes_parent", "fk_no_offsets", "fk_no",
    "baseline_entries",
    "angle_pass_offsets", "angle_pass_vertices", "angle_pass_parents",
    "postline_entry_offsets", "postline_entry_vertices",
    "postline_child_offsets", "postline_child_vertices", "display_update_move_mask",
)

PRIMITIVE_KERNEL_FIELDS = (
    "team", "particles", "fix", "all_fix", "ignore", "prim_depth",
    "inv_mass", "thickness", "aabb_min", "aabb_max", "intersect", "use",
)
SELF_TEAM_KERNEL_FIELDS = (
    "use_point", "use_edge", "use_triangle", "self_grid_size", "self_max_primitive_size",
    "self_mode", "sync_mode", "self_thickness_lut", "self_cloth_mass",
    "sp_start", "sp_count", "se_start", "se_count", "st_start", "st_count",
)
SELF_PARTICLE_KERNEL_FIELDS = ("intersect_flag",)
SELF_STATE_KERNEL_FIELDS = (
    "ee_my", "ee_target", "ee_thickness", "ee_s", "ee_t", "ee_n", "ee_enable",
    "pt_my", "pt_target", "pt_thickness", "pt_sign", "pt_enable",
    "scl_counts",
    "ct_kind", "ct_my_team", "ct_my_start", "ct_my_count", "ct_tgt_team", "ct_tgt_start",
    "ct_tgt_count", "ct_same", "ct_pair_off",
    "it_edge_team", "it_edge_start", "it_edge_count", "it_tri_team", "it_tri_start",
    "it_tri_count", "it_same", "it_pair_off",
    "ip_edge", "ip_tri",
    "scl_max_fixed",
)


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
    ('c_enabled', 'u8_s', ()),
    ('c_enabled_prev', 'u8_s', ()),
    ('c_active', 'u8_s', ()),
    ('c_input_positions', 'f32_v3', (3,)),
    ('c_input_rotations', 'f32_v4', (4,)),
    ('c_input_tips', 'f32_v3', (3,)),
    ('c_input_radii', 'f32_v2', (2,)),
    ('c_frame_pos', 'f32_v3', (3,)),
    ('c_frame_rot', 'f32_v4', (4,)),
    ('c_frame_tip', 'f32_v3', (3,)),
    ('c_frame_radius', 'f32_v2', (2,)),
    ('c_old_frame_pos', 'f32_v3', (3,)),
    ('c_old_frame_rot', 'f32_v4', (4,)),
    ('c_old_frame_tip', 'f32_v3', (3,)),
    ('c_now_pos', 'f32_v3', (3,)),
    ('c_now_rot', 'f32_v4', (4,)),
    ('c_now_tip', 'f32_v3', (3,)),
    ('c_old_pos', 'f32_v3', (3,)),
    ('c_old_rot', 'f32_v4', (4,)),
    ('c_old_tip', 'f32_v3', (3,)),
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
    ('sfp_team', 'i32_s', ()),
    ('sfp_particles', 'i32_v3', (3,)),
    ('sfp_fix', 'u8_s', ()),
    ('sfp_all_fix', 'u8_s', ()),
    ('sfp_ignore', 'u8_s', ()),
    ('sfp_prim_depth', 'f32_s', ()),
    ('sfp_inv_mass', 'f32_v3', (3,)),
    ('sfp_thickness', 'f32_s', ()),
    ('sfp_aabb_min', 'f32_v3', (3,)),
    ('sfp_aabb_max', 'f32_v3', (3,)),
    ('sfp_intersect', 'u8_s', ()),
    ('sfp_use', 'u8_s', ()),
    ('sfe_team', 'i32_s', ()),
    ('sfe_particles', 'i32_v3', (3,)),
    ('sfe_fix', 'u8_s', ()),
    ('sfe_all_fix', 'u8_s', ()),
    ('sfe_ignore', 'u8_s', ()),
    ('sfe_prim_depth', 'f32_s', ()),
    ('sfe_inv_mass', 'f32_v3', (3,)),
    ('sfe_thickness', 'f32_s', ()),
    ('sfe_aabb_min', 'f32_v3', (3,)),
    ('sfe_aabb_max', 'f32_v3', (3,)),
    ('sfe_intersect', 'u8_s', ()),
    ('sfe_use', 'u8_s', ()),
    ('sft_team', 'i32_s', ()),
    ('sft_particles', 'i32_v3', (3,)),
    ('sft_fix', 'u8_s', ()),
    ('sft_all_fix', 'u8_s', ()),
    ('sft_ignore', 'u8_s', ()),
    ('sft_prim_depth', 'f32_s', ()),
    ('sft_inv_mass', 'f32_v3', (3,)),
    ('sft_thickness', 'f32_s', ()),
    ('sft_aabb_min', 'f32_v3', (3,)),
    ('sft_aabb_max', 'f32_v3', (3,)),
    ('sft_intersect', 'u8_s', ()),
    ('sft_use', 'u8_s', ()),
    ('t_use_point', 'u8_s', ()),
    ('t_use_edge', 'u8_s', ()),
    ('t_use_triangle', 'u8_s', ()),
    ('t_self_grid_size', 'f32_s', ()),
    ('t_self_max_primitive_size', 'f32_s', ()),
    ('t_self_mode', 'i8_s', ()),
    ('t_sync_mode', 'i8_s', ()),
    ('t_self_thickness_lut', 'f32_v16', (16,)),
    ('t_self_cloth_mass', 'f32_s', ()),
    ('t_sp_start', 'i32_s', ()),
    ('t_sp_count', 'i32_s', ()),
    ('t_se_start', 'i32_s', ()),
    ('t_se_count', 'i32_s', ()),
    ('t_st_start', 'i32_s', ()),
    ('t_st_count', 'i32_s', ()),
    ('p_intersect_flag', 'u8_s', ()),
    ('ee_my', 'i32_s', ()),
    ('ee_target', 'i32_s', ()),
    ('ee_thickness', 'f32_s', ()),
    ('ee_s', 'f32_s', ()),
    ('ee_t', 'f32_s', ()),
    ('ee_n', 'f32_v3', (3,)),
    ('ee_enable', 'u8_s', ()),
    ('pt_my', 'i32_s', ()),
    ('pt_target', 'i32_s', ()),
    ('pt_thickness', 'f32_s', ()),
    ('pt_sign', 'f32_s', ()),
    ('pt_enable', 'u8_s', ()),
    ('scl_counts', 'i32_s', ()),
    ('ct_kind', 'i32_s', ()),
    ('ct_my_team', 'i32_s', ()),
    ('ct_my_start', 'i32_s', ()),
    ('ct_my_count', 'i32_s', ()),
    ('ct_tgt_team', 'i32_s', ()),
    ('ct_tgt_start', 'i32_s', ()),
    ('ct_tgt_count', 'i32_s', ()),
    ('ct_same', 'u8_s', ()),
    ('ct_pair_off', 'i32_s', ()),
    ('it_edge_team', 'i32_s', ()),
    ('it_edge_start', 'i32_s', ()),
    ('it_edge_count', 'i32_s', ()),
    ('it_tri_team', 'i32_s', ()),
    ('it_tri_start', 'i32_s', ()),
    ('it_tri_count', 'i32_s', ()),
    ('it_same', 'u8_s', ()),
    ('it_pair_off', 'i32_s', ()),
    ('ip_edge', 'i32_s', ()),
    ('ip_tri', 'i32_s', ()),
    ('scl_max_fixed', 'i32_s', ()),
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
