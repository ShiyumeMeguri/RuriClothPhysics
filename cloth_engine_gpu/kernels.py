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

ALL_PHASES = int32(-1)

MAX_SIM_COUNT = 5

# defs constants mirrored (device-side literals stay f32-wrapped)
TETHER_STRETCH_LIMIT = float32(0.03)
TETHER_STIFFNESS_WIDTH = float32(0.3)
TETHER_VELOCITY_ATTENUATION = float32(0.7)
EPSILON = float32(1e-8)

WIND_BASE_SPEED = float32(7.5)
WIND_TURBULENCE_ANGLE = float32(45.0)
DEG2RAD = float32(math.pi / 180.0)

FORCE_VELOCITY_ADD = int32(1)
FORCE_VELOCITY_ADD_WITHOUT_DEPTH = int32(2)
FORCE_VELOCITY_CHANGE = int32(3)
FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH = int32(4)

BONE_SPRING_FIX_MASS = float32(10.0)
BONE_CLOTH_FIX_MASS = float32(50.0)
DISTANCE_HORIZONTAL_STIFFNESS = float32(0.5)
DISTANCE_VELOCITY_ATTENUATION = float32(0.3)


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
                 p_team, p_local_positions, p_local_normals, p_local_tangents,
                 p_skin_indices, p_skin_weights, p_positions, p_rotations,
                 p_next_positions, p_velocity_positions, p_step_basic_positions, p_vertex_root,
                 p_old_anim_positions, p_old_anim_rotations, p_base_positions, p_base_rotations,
                 p_step_basic_rotations, p_depth, p_velocities, p_old_positions, p_friction,
                 p_vertex_root_local, p_collision_normals, p_static_friction, p_real_velocities,
                 p_attr_move,
                 x_world, x_bind,
                 st_tether_particle, st_tether_team,
                 st_move_particle, st_move_team, st_fixed_particle, st_fixed_team,
                 st_spring_particle, st_spring_team,
                 st_distance_target, st_distance_rest,
                 csr_distance_offsets, csr_distance_order,
                 sc_dcorr):
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
    n_tether = st_tether_particle.shape[0]
    n_move = st_move_particle.shape[0]
    n_fixed = st_fixed_particle.shape[0]
    n_spring = st_spring_particle.shape[0]
    for _k in range(sub_begin, sub_end):
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

    # ----- FRAME-POST (F4 team_time.frame_post added next) -----
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
)

PARTICLE_KERNEL_FIELDS = (
    "team", "local_positions", "local_normals", "local_tangents",
    "skin_indices", "skin_weights", "positions", "rotations",
    "next_positions", "velocity_positions", "step_basic_positions", "vertex_root",
    "old_anim_positions", "old_anim_rotations", "base_positions", "base_rotations",
    "step_basic_rotations", "depth", "velocities", "old_positions", "friction",
    "vertex_root_local", "collision_normals", "static_friction", "real_velocities",
    "attr_move",
)

TRANSFORM_KERNEL_FIELDS = ("world", "bind_pose")

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
)

# CSR gather tables (offsets + order uploaded from a program CsrTable attribute)
STATIC_CSR_FIELDS = (
    ("distance_csr_offsets", "distance_csr_order", "distance_csr"),
)
