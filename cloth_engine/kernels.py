import math

import warp as wp

from ..cloth_kernel import defs as _defs
from . import dmath
from . import policy

wp.set_module_options(policy.MODULE_OPTIONS)

EPSILON = wp.constant(float(_defs.EPSILON))
TETHER_STRETCH_LIMIT = wp.constant(float(_defs.TETHER_STRETCH_LIMIT))
TETHER_STIFFNESS_WIDTH = wp.constant(float(_defs.TETHER_STIFFNESS_WIDTH))
TETHER_COMPRESSION_STIFFNESS = wp.constant(float(_defs.TETHER_COMPRESSION_STIFFNESS))
TETHER_STRETCH_STIFFNESS = wp.constant(float(_defs.TETHER_STRETCH_STIFFNESS))
TETHER_COMPRESSION_VELOCITY_ATTENUATION = wp.constant(
    float(_defs.TETHER_COMPRESSION_VELOCITY_ATTENUATION))
TETHER_STRETCH_VELOCITY_ATTENUATION = wp.constant(
    float(_defs.TETHER_STRETCH_VELOCITY_ATTENUATION))
BONE_SPRING_FIX_MASS = wp.constant(float(_defs.BONE_SPRING_FIX_MASS))
BONE_CLOTH_FIX_MASS = wp.constant(float(_defs.BONE_CLOTH_FIX_MASS))
DISTANCE_HORIZONTAL_STIFFNESS = wp.constant(float(_defs.DISTANCE_HORIZONTAL_STIFFNESS))
WIND_BASE_SPEED = wp.constant(float(_defs.WIND_BASE_SPEED))
WIND_TURBULENCE_ANGLE = wp.constant(float(_defs.WIND_TURBULENCE_ANGLE))
WIND_MIN_SPEED = wp.constant(float(_defs.WIND_MIN_SPEED))
WIND_MAX_TIME = wp.constant(float(_defs.WIND_MAX_TIME))
SELF_COLLISION_SCR = wp.constant(float(_defs.SELF_COLLISION_SCR))
SELF_COLLISION_POINT_TRIANGLE_ANGLE_COS = wp.constant(
    float(_defs.SELF_COLLISION_POINT_TRIANGLE_ANGLE_COS))
SCL_USE_INTERSECT = wp.constant(int(4))
DEG2RAD = wp.constant(float(math.pi / 180.0))
RAD2DEG = wp.constant(float(180.0 / math.pi))
ANGLE_LIMIT_ROT_RATIO = wp.constant(float(_defs.ANGLE_LIMIT_ROTATION_RATIO))
ANGLE_LIMIT_ATTENUATION = wp.constant(float(_defs.ANGLE_LIMIT_ATTENUATION))
COLLIDER_SPHERE = wp.constant(int(_defs.COLLIDER_SPHERE))
COLLIDER_CAPSULE = wp.constant(int(_defs.COLLIDER_CAPSULE))
COLLISION_POINT = wp.constant(int(_defs.COLLISION_POINT))
TO_FIXED = wp.constant(float(1.0e6))
MAX_DISTANCE_RATIO_FUTURE_PREDICTION = wp.constant(
    float(_defs.MAX_DISTANCE_RATIO_FUTURE_PREDICTION))


@wp.func
def do_tether(e: int,
              tether_particle: wp.array(dtype=int),
              p_team: wp.array(dtype=int),
              next_positions: wp.array2d(dtype=float),
              velocity_positions: wp.array2d(dtype=float),
              step_basic_positions: wp.array2d(dtype=float),
              vertex_root: wp.array(dtype=int),
              t_tether_compression: wp.array(dtype=float)):
    idx = tether_particle[e]
    team = p_team[idx]
    root = vertex_root[idx]
    compression_limit = 1.0 - t_tether_compression[team]
    stretch_limit = 1.0 + TETHER_STRETCH_LIMIT

    vx = next_positions[root, 0] - next_positions[idx, 0]
    vy = next_positions[root, 1] - next_positions[idx, 1]
    vz = next_positions[root, 2] - next_positions[idx, 2]
    distance = dmath.length3(vx, vy, vz)
    cvx = step_basic_positions[idx, 0] - step_basic_positions[root, 0]
    cvy = step_basic_positions[idx, 1] - step_basic_positions[root, 1]
    cvz = step_basic_positions[idx, 2] - step_basic_positions[root, 2]
    calc_distance = dmath.length3(cvx, cvy, cvz)

    valid = (distance >= EPSILON) and (calc_distance != 0.0)
    ratio = distance / (calc_distance if calc_distance != 0.0 else 1.0)
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
    inv = 1.0 / (distance if distance > 1.0e-30 else 1.0)
    scale = dist * stiffness * inv
    ax = vx * scale
    ay = vy * scale
    az = vz * scale
    next_positions[idx, 0] = next_positions[idx, 0] + ax
    next_positions[idx, 1] = next_positions[idx, 1] + ay
    next_positions[idx, 2] = next_positions[idx, 2] + az
    velocity_positions[idx, 0] = velocity_positions[idx, 0] + ax * attenuation
    velocity_positions[idx, 1] = velocity_positions[idx, 1] + ay * attenuation
    velocity_positions[idx, 2] = velocity_positions[idx, 2] + az * attenuation


@wp.func
def do_wind_blend(wind_main: float, time: float, dqx: float, dqy: float, dqz: float, dqw: float,
                  zone_turbulence: float,
                  blend: float, turbulence_param: float, wind_position: float):
    active = wind_main >= WIND_MIN_SPEED
    main_ratio = wind_main / WIND_BASE_SPEED

    sin_pos = wind_position + time * 10.0
    sin_wave = wp.sin(sin_pos)

    noise_pos = wind_position + time * 2.3132
    noise_wave = dmath.cnoise2(noise_pos, noise_pos) * 2.3

    wave_x = sin_wave + (noise_wave - sin_wave) * blend
    wave_y = wave_x

    turbulence = zone_turbulence * turbulence_param

    angle_x = (wave_x * WIND_TURBULENCE_ANGLE) * DEG2RAD
    angle_y = (wave_y * WIND_TURBULENCE_ANGLE) * DEG2RAD
    angle_y = angle_y * (0.1 + (0.5 - 0.1) * blend)
    angle_x = angle_x * turbulence
    angle_y = angle_y * turbulence

    rqx, rqy, rqz, rqw = dmath.euler_yx(angle_x, angle_y)
    cqx, cqy, cqz, cqw = dmath.quat_mul(dqx, dqy, dqz, dqw, rqx, rqy, rqz, rqw)
    wdx, wdy, wdz = dmath.quat_to_tangent(cqx, cqy, cqz, cqw)

    main_scale = dmath.saturate(1.0 - main_ratio)
    main_wave = (wave_x + 1.0) * 0.5
    main_wave = main_wave * main_scale * turbulence
    strength = wind_main - wind_main * main_wave
    if not active:
        strength = 0.0
    return (wdx * strength, wdy * strength, wdz * strength)


@wp.func
def _edge_sphere(p0x: float, p0y: float, p0z: float, p1x: float, p1y: float, p1z: float,
                 r0: float, r1: float, cfr: float, overlap: bool,
                 cldx: float, cldy: float, cldz: float,
                 cnwx: float, cnwy: float, cnwz: float, cradius: float):
    s = dmath.closest_pt_point_segment_ratio(cldx, cldy, cldz, p0x, p0y, p0z, p1x, p1y, p1z)
    clx = p0x + (p1x - p0x) * s
    cly = p0y + (p1y - p0y) * s
    clz = p0z + (p1z - p0z) * s
    vx = clx - cldx
    vy = cly - cldy
    vz = clz - cldz
    clen = dmath.length3(vx, vy, vz)
    degenerate = clen < 1.0e-9
    safe = clen if clen > 1.0e-30 else 1.0
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
    b0 = 1.0 - s
    b1 = s
    denom = b0 * b0 + b1 * b1
    scale = cval / (denom if denom > 0.0 else 1.0)
    c0x = nx * (b0 * scale)
    c0y = ny * (b0 * scale)
    c0z = nz * (b0 * scale)
    c1x = nx * (b1 * scale)
    c1y = ny * (b1 * scale)
    c1z = nz * (b1 * scale)
    valid = overlap and (not degenerate) and (not miss)
    if not valid:
        return (wp.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if no_contact:
        return (near_dist, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nx, ny, nz)
    return (dmath.negate(cval), c0x, c0y, c0z, c1x, c1y, c1z, nx, ny, nz)


@wp.func
def _edge_capsule(p0x: float, p0y: float, p0z: float, p1x: float, p1y: float, p1z: float,
                  r0: float, r1: float, cfr: float, overlap: bool,
                  sox: float, soy: float, soz: float, eox: float, eoy: float, eoz: float,
                  snx: float, sny: float, snz: float, enx: float, eny: float, enz: float,
                  sr: float, er: float):
    s, t, cax, cay, caz, cbx, cby, cbz = dmath.closest_pt_segment_segment(
        p0x, p0y, p0z, p1x, p1y, p1z, sox, soy, soz, eox, eoy, eoz)
    vx = cax - cbx
    vy = cay - cby
    vz = caz - cbz
    clen = dmath.length3(vx, vy, vz)
    degenerate = clen < 1.0e-9
    safe = clen if clen > 1.0e-30 else 1.0
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
        safe2 = clen2 if clen2 > 1.0e-30 else 1.0
        nx = vx / safe2
        ny = vy / safe2
        nz = vz / safe2
        clen = clen2
        if clen2 < 1.0e-9:
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
    b0 = 1.0 - s
    b1 = s
    denom = b0 * b0 + b1 * b1
    scale = cval / (denom if denom > 0.0 else 1.0)
    c0x = nx * (b0 * scale)
    c0y = ny * (b0 * scale)
    c0z = nz * (b0 * scale)
    c1x = nx * (b1 * scale)
    c1y = ny * (b1 * scale)
    c1z = nz * (b1 * scale)
    valid = overlap and (not degenerate) and (not miss)
    if not valid:
        return (wp.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if no_contact:
        return (near_dist, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nx, ny, nz)
    return (dmath.negate(cval), c0x, c0y, c0z, c1x, c1y, c1z, nx, ny, nz)


@wp.func
def team_frame_mask(enabled: wp.array(dtype=int), valid: wp.array(dtype=int),
                    cws: wp.array2d(dtype=float), i: int):
    if enabled[i] == 0 or valid[i] == 0:
        return False
    ax = wp.abs(cws[i, 0])
    ay = wp.abs(cws[i, 1])
    az = wp.abs(cws[i, 2])
    lo = ax
    if ay < lo:
        lo = ay
    if az < lo:
        lo = az
    return lo >= 1e-6


@wp.func
def _skin_row(world: wp.array3d(dtype=float), bind: wp.array3d(dtype=float),
              t: int, r: int, c: int):
    return (world[t, r, 0] * bind[t, 0, c] + world[t, r, 1] * bind[t, 1, c]
            + world[t, r, 2] * bind[t, 2, c] + world[t, r, 3] * bind[t, 3, c])


@wp.func
def do_base_pose(p: int, p_team: wp.array(dtype=int),
                 local_positions: wp.array2d(dtype=float),
                 local_normals: wp.array2d(dtype=float),
                 local_tangents: wp.array2d(dtype=float),
                 skin_indices: wp.array2d(dtype=int),
                 skin_weights: wp.array2d(dtype=float),
                 positions: wp.array2d(dtype=float),
                 rotations: wp.array2d(dtype=float),
                 world: wp.array3d(dtype=float),
                 bind: wp.array3d(dtype=float)):
    world_position = wp.vec3()
    world_normal = wp.vec3()
    world_tangent = wp.vec3()
    for r in range(3):
        world_position[r] = 0.0
        world_normal[r] = 0.0
        world_tangent[r] = 0.0
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
            world_position[r] = world_position[r] + w * (s0 * lx + s1 * ly + s2 * lz + s3)
            world_normal[r] = world_normal[r] + w * (s0 * lnx + s1 * lny + s2 * lnz)
            world_tangent[r] = world_tangent[r] + w * (s0 * ltx + s1 * lty + s2 * ltz)
    positions[p, 0] = world_position[0]
    positions[p, 1] = world_position[1]
    positions[p, 2] = world_position[2]
    nx, ny, nz = dmath.normalize3_fb(world_normal[0], world_normal[1], world_normal[2],
                                     0.0, 1.0, 0.0)
    tx, ty, tz = dmath.normalize3_fb(world_tangent[0], world_tangent[1], world_tangent[2],
                                     0.0, 0.0, 1.0)
    qx, qy, qz, qw = dmath.to_rotation(nx, ny, nz, tx, ty, tz)
    rotations[p, 0] = qx
    rotations[p, 1] = qy
    rotations[p, 2] = qz
    rotations[p, 3] = qw


@wp.func
def do_distance_gather(p: int, p_team: wp.array(dtype=int),
                       next_positions: wp.array2d(dtype=float),
                       base_positions: wp.array2d(dtype=float),
                       depth: wp.array(dtype=float),
                       friction: wp.array(dtype=float),
                       attr_move: wp.array(dtype=int),
                       t_is_spring: wp.array(dtype=int),
                       t_animation_pose_ratio: wp.array(dtype=float),
                       t_init_scale: wp.array2d(dtype=float),
                       t_scale_ratio: wp.array(dtype=float),
                       t_distance_lut: wp.array2d(dtype=float),
                       power1: float,
                       csr_offsets: wp.array(dtype=int),
                       csr_order: wp.array(dtype=int),
                       distance_target: wp.array(dtype=int),
                       distance_rest: wp.array(dtype=float),
                       sc_dcorr: wp.array2d(dtype=float)):
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

    sumx = wp.float64(0.0)
    sumy = wp.float64(0.0)
    sumz = wp.float64(0.0)
    count_ok = int(0)
    start = csr_offsets[p]
    stop = csr_offsets[p + 1]
    for k in range(start, stop):
        r = csr_order[k]
        tgt = distance_target[r]
        rest = distance_rest[r]
        if rest >= 0.0:
            final_stiffness = dmath.saturate(stiffness)
        else:
            final_stiffness = dmath.saturate(stiffness * DISTANCE_HORIZONTAL_STIFFNESS)
        fixed_t = attr_move[tgt] == 0
        inv_mass_t = dmath.calc_inverse_mass_fixed(friction[tgt], depth[tgt], fixed_t, fix_mass)
        vx = next_positions[tgt, 0] - npx
        vy = next_positions[tgt, 1] - npy
        vz = next_positions[tgt, 2] - npz
        zero_rest = rest == 0.0
        distance = dmath.length3(vx, vy, vz)
        ok = zero_rest or (distance >= EPSILON)
        base_len = dmath.length3(bpx - base_positions[tgt, 0],
                                 bpy - base_positions[tgt, 1],
                                 bpz - base_positions[tgt, 2])
        rest_length = dmath.lerp(wp.abs(rest) * scale, base_len, anime_ratio)
        safe_d = distance if distance > 1e-30 else 1.0
        nx = vx / safe_d
        ny = vy / safe_d
        nz = vz / safe_d
        a = (distance - rest_length) * final_stiffness
        denom = inv_mass_p + inv_mass_t
        cxr = a * nx / denom * inv_mass_p
        cyr = a * ny / denom * inv_mass_p
        czr = a * nz / denom * inv_mass_p
        if zero_rest:
            cxr = vx * 0.5
            cyr = vy * 0.5
            czr = vz * 0.5
        if ok:
            count_ok += 1
            sumx += wp.float64(cxr)
            sumy += wp.float64(cyr)
            sumz += wp.float64(czr)
    if count_ok > 0:
        sc_dcorr[p, 0] = wp.float32(sumx / wp.float64(count_ok))
        sc_dcorr[p, 1] = wp.float32(sumy / wp.float64(count_ok))
        sc_dcorr[p, 2] = wp.float32(sumz / wp.float64(count_ok))
    else:
        sc_dcorr[p, 0] = 0.0
        sc_dcorr[p, 1] = 0.0
        sc_dcorr[p, 2] = 0.0


@wp.func
def do_step_update(i: int, sim_dt: float,
                   t_now_update: wp.array(dtype=float),
                   t_time: wp.array(dtype=float),
                   t_frame_old: wp.array(dtype=float),
                   t_frame_interp: wp.array(dtype=float),
                   t_now_wp: wp.array2d(dtype=float),
                   t_now_wr: wp.array2d(dtype=float),
                   t_old_wp: wp.array2d(dtype=float),
                   t_old_wr: wp.array2d(dtype=float),
                   t_ofwp: wp.array2d(dtype=float),
                   t_ofwr: wp.array2d(dtype=float),
                   t_ofws: wp.array2d(dtype=float),
                   t_fwp: wp.array2d(dtype=float),
                   t_fwr: wp.array2d(dtype=float),
                   t_fws: wp.array2d(dtype=float),
                   t_step_vector: wp.array2d(dtype=float),
                   t_step_rotation: wp.array2d(dtype=float),
                   t_step_mir: wp.array(dtype=float),
                   t_step_rir: wp.array(dtype=float),
                   t_local_inertia: wp.array(dtype=float),
                   t_lmsl: wp.array(dtype=float),
                   t_lrsl: wp.array(dtype=float),
                   t_inertia_vector: wp.array2d(dtype=float),
                   t_inertia_rotation: wp.array2d(dtype=float),
                   t_angular_velocity: wp.array(dtype=float),
                   t_rotation_axis: wp.array2d(dtype=float),
                   t_init_scale: wp.array2d(dtype=float),
                   t_scale_ratio: wp.array(dtype=float),
                   t_gravity_direction: wp.array2d(dtype=float),
                   t_gravity_dot: wp.array(dtype=float),
                   t_ilgd: wp.array2d(dtype=float),
                   t_neg_dir: wp.array2d(dtype=float),
                   t_gravity: wp.array(dtype=float),
                   t_gravity_falloff: wp.array(dtype=float),
                   t_gravity_ratio: wp.array(dtype=float),
                   t_velocity_weight: wp.array(dtype=float),
                   t_stab_time: wp.array(dtype=float),
                   t_blend_weight: wp.array(dtype=float),
                   t_bwp: wp.array(dtype=float),
                   t_distance_weight: wp.array(dtype=float),
                   t_wind_moving: wp.array(dtype=float),
                   t_frame_moving_speed: wp.array(dtype=float),
                   t_moving_wind_main: wp.array(dtype=float),
                   t_frame_moving_dir: wp.array2d(dtype=float),
                   t_moving_wind_dir: wp.array2d(dtype=float),
                   t_moving_wind_dirq: wp.array2d(dtype=float),
                   t_wind_main: wp.array2d(dtype=float),
                   t_wind_frequency: wp.array(dtype=float),
                   t_wind_count: wp.array(dtype=int),
                   t_wind_time: wp.array2d(dtype=float),
                   t_moving_wind_time: wp.array(dtype=float)):
    nu = t_now_update[i] + sim_dt
    t_now_update[i] = nu
    span = t_time[i] - t_frame_old[i]
    if span > 0.0:
        interp = dmath.saturate((nu - t_frame_old[i]) / span)
    else:
        interp = 1.0
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
    lmi = 1.0 - li
    lri = 1.0 - li
    lvx = svx * (1.0 - lmi)
    lvy = svy * (1.0 - lmi)
    lvz = svz * (1.0 - lmi)
    local_speed = dmath.length3(lvx, lvy, lvz) / sim_dt
    limit = t_lmsl[i]
    if (local_speed > limit) and (limit >= 0.0):
        denom = local_speed if local_speed > 0.0 else 1.0
        ratio = limit / denom
        lmi = 1.0 + (lmi - 1.0) * ratio
    local_angle = step_angle * (1.0 - lri)
    local_angle_speed = (local_angle / sim_dt) * RAD2DEG
    limit = t_lrsl[i]
    if (local_angle_speed > limit) and (limit >= 0.0):
        denom = local_angle_speed if local_angle_speed > 0.0 else 1.0
        ratio = limit / denom
        lri = 1.0 + (lri - 1.0) * ratio
    t_step_mir[i] = lmi
    t_step_rir[i] = lri

    t_inertia_vector[i, 0] = svx * lmi
    t_inertia_vector[i, 1] = svy * lmi
    t_inertia_vector[i, 2] = svz * lmi
    irx, iry, irz, irw = dmath.quat_slerp(0.0, 0.0, 0.0, 1.0,
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
        t_rotation_axis[i, 0] = 0.0
        t_rotation_axis[i, 1] = 0.0
        t_rotation_axis[i, 2] = 0.0

    isl = dmath.length3(t_init_scale[i, 0], t_init_scale[i, 1], t_init_scale[i, 2])
    if isl < 1.0e-30:
        isl = 1.0e-30
    wsl = dmath.length3(wsx, wsy, wsz)
    sr = wsl / isl
    if sr < 1.0e-6:
        sr = 1.0e-6
    t_scale_ratio[i] = sr

    gdx = t_gravity_direction[i, 0]
    gdy = t_gravity_direction[i, 1]
    gdz = t_gravity_direction[i, 2]
    gravity_dot = float(1.0)
    if (gdx * gdx + gdy * gdy + gdz * gdz) > EPSILON:
        ilx = t_ilgd[i, 0]
        ily = t_ilgd[i, 1] * t_neg_dir[i, 1]
        ilz = t_ilgd[i, 2]
        wfx, wfy, wfz = dmath.quat_rotate(nwrx, nwry, nwrz, nwrw, ilx, ily, ilz)
        gdot = wfx * gdx + wfy * gdy + wfz * gdz
        gravity_dot = dmath.saturate(gdot * 0.5 + 0.5)
    t_gravity_dot[i] = gravity_dot

    gravity_ratio = float(1.0)
    if (t_gravity[i] > 1.0e-6) and (t_gravity_falloff[i] > 1.0e-6):
        low = dmath.saturate(1.0 - t_gravity_falloff[i])
        gravity_ratio = low + (1.0 - low) * dmath.saturate(1.0 - gravity_dot)
    t_gravity_ratio[i] = gravity_ratio

    vw = t_velocity_weight[i]
    if vw < 1.0:
        stab = t_stab_time[i]
        if stab > 1.0e-6:
            add = sim_dt / stab
        else:
            add = 1.0
        vw = vw + add
        if vw > 1.0:
            vw = 1.0
        t_velocity_weight[i] = vw
    t_blend_weight[i] = dmath.saturate(vw * t_bwp[i] * t_distance_weight[i])

    moving_active = t_wind_moving[i] > 0.01
    if moving_active:
        denom = sr if sr > 0.0 else 1.0
        mwm = (t_frame_moving_speed[i] * t_wind_moving[i]) / denom
    else:
        mwm = 0.0
    t_moving_wind_main[i] = mwm
    if moving_active:
        mdx = dmath.negate(t_frame_moving_dir[i, 0])
        mdy = dmath.negate(t_frame_moving_dir[i, 1])
        mdz = dmath.negate(t_frame_moving_dir[i, 2])
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
        frequency = (0.2 + main_ratio * 0.5) * wf
        if frequency > 1.5:
            frequency = 1.5
        frequency = frequency * sim_dt
        if s < wc:
            nt = t_wind_time[i, s] + frequency
            if nt > WIND_MAX_TIME:
                nt = nt - WIND_MAX_TIME * 2.0
            t_wind_time[i, s] = nt
    move_ratio = mwm / WIND_BASE_SPEED
    mf = (0.2 + move_ratio * 0.5) * wf
    if mf > 1.5:
        mf = 1.5
    mf = mf * sim_dt
    if moving_active:
        mt2 = t_moving_wind_time[i] + mf
        if mt2 > WIND_MAX_TIME:
            mt2 = mt2 - WIND_MAX_TIME * 2.0
        t_moving_wind_time[i] = mt2


@wp.func
def _neg_transform_pose(arr_pos: wp.array2d(dtype=float), arr_rot: wp.array2d(dtype=float),
                        ci: int, m: wp.mat44d, flip1: float, flip2: float):
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


@wp.func
def _neg_transform_point(arr_pos: wp.array2d(dtype=float), ci: int, m: wp.mat44d):
    px, py, pz = dmath.transform_point(m, arr_pos[ci, 0], arr_pos[ci, 1], arr_pos[ci, 2])
    arr_pos[ci, 0] = px
    arr_pos[ci, 1] = py
    arr_pos[ci, 2] = pz


@wp.func
def _shift_pose(arr_pos: wp.array2d(dtype=float), arr_rot: wp.array2d(dtype=float), ci: int,
                cpx: float, cpy: float, cpz: float, svx: float, svy: float, svz: float,
                srx: float, sry: float, srz: float, srw: float):
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


@wp.func
def _shift_point(arr: wp.array2d(dtype=float), p: int,
                 cpx: float, cpy: float, cpz: float, svx: float, svy: float, svz: float,
                 srx: float, sry: float, srz: float, srw: float):
    rx, ry, rz = dmath.quat_rotate(srx, sry, srz, srw,
                                   arr[p, 0] - cpx, arr[p, 1] - cpy, arr[p, 2] - cpz)
    arr[p, 0] = rx + cpx + svx
    arr[p, 1] = ry + cpy + svy
    arr[p, 2] = rz + cpz + svz


@wp.func
def _premul_quat(arr: wp.array2d(dtype=float), p: int,
                 srx: float, sry: float, srz: float, srw: float):
    qx, qy, qz, qw = dmath.quat_mul(srx, sry, srz, srw,
                                    arr[p, 0], arr[p, 1], arr[p, 2], arr[p, 3])
    arr[p, 0] = qx
    arr[p, 1] = qy
    arr[p, 2] = qz
    arr[p, 3] = qw


@wp.func
def _rotate_vec(arr: wp.array2d(dtype=float), p: int,
                srx: float, sry: float, srz: float, srw: float):
    vx, vy, vz = dmath.quat_rotate(srx, sry, srz, srw, arr[p, 0], arr[p, 1], arr[p, 2])
    arr[p, 0] = vx
    arr[p, 1] = vy
    arr[p, 2] = vz


@wp.func
def do_advance(i: int, fdt: float, sim_dt: float, max_sim_count: int, global_time_scale: float,
               time_reset: wp.array(dtype=int),
               time: wp.array(dtype=float),
               old_time: wp.array(dtype=float),
               now_update: wp.array(dtype=float),
               old_update: wp.array(dtype=float),
               frame_update: wp.array(dtype=float),
               frame_old: wp.array(dtype=float),
               frame_dt: wp.array(dtype=float),
               time_scale: wp.array(dtype=float),
               now_time_scale: wp.array(dtype=float),
               update_count: wp.array(dtype=int),
               skip_count: wp.array(dtype=int),
               running: wp.array(dtype=int)):
    reset = time_reset[i] != 0
    t_time = 0.0 if reset else time[i]
    t_now_update = 0.0 if reset else now_update[i]
    t_old_update = 0.0 if reset else old_update[i]
    t_frame_update = 0.0 if reset else frame_update[i]
    t_frame_old = 0.0 if reset else frame_old[i]

    frame_dt[i] = fdt
    ts = time_scale[i] * global_time_scale
    now_time_scale[i] = ts
    add_time = fdt * ts
    new_time = t_time + add_time
    interval = new_time - t_now_update
    uc = int(interval / sim_dt)
    clamped = uc if uc < max_sim_count else max_sim_count
    skip = uc - clamped
    if skip > 0:
        new_time = new_time - sim_dt * float(skip)
    guard = (clamped > 0) and (add_time == 0.0)
    if guard:
        clamped = int(0)
        skip = int(0)
        new_now_update = new_time - sim_dt + 0.0001
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
    running[i] = 1 if updated else 0


@wp.func
def do_particles_frame_pre(p: int, p_team: wp.array(dtype=int),
                           p_positions: wp.array2d(dtype=float),
                           p_rotations: wp.array2d(dtype=float),
                           p_next_positions: wp.array2d(dtype=float),
                           p_old_positions: wp.array2d(dtype=float),
                           p_old_rotations: wp.array2d(dtype=float),
                           p_base_positions: wp.array2d(dtype=float),
                           p_base_rotations: wp.array2d(dtype=float),
                           p_old_anim_positions: wp.array2d(dtype=float),
                           p_old_anim_rotations: wp.array2d(dtype=float),
                           p_velocity_positions: wp.array2d(dtype=float),
                           p_display_positions: wp.array2d(dtype=float),
                           p_velocities: wp.array2d(dtype=float),
                           p_real_velocities: wp.array2d(dtype=float),
                           p_friction: wp.array(dtype=float),
                           p_static_friction: wp.array(dtype=float),
                           p_collision_normals: wp.array2d(dtype=float),
                           t_reset_pending: wp.array(dtype=int),
                           t_neg_teleport: wp.array(dtype=int),
                           t_neg_matrix: wp.array(dtype=wp.mat44d),
                           t_inertia_shift: wp.array(dtype=int),
                           t_shift_vec: wp.array2d(dtype=float),
                           t_shift_rot: wp.array2d(dtype=float),
                           t_old_cwp: wp.array2d(dtype=float)):
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
            p_velocities[p, j] = 0.0
            p_real_velocities[p, j] = 0.0
            p_collision_normals[p, j] = 0.0
        for j in range(4):
            rv = p_rotations[p, j]
            p_old_rotations[p, j] = rv
            p_base_rotations[p, j] = rv
            p_old_anim_rotations[p, j] = rv
        p_friction[p] = 0.0
        p_static_friction[p] = 0.0
        return
    neg = t_neg_teleport[team] != 0
    shift = t_inertia_shift[team] != 0
    if not (neg or shift):
        return
    if neg:
        m = t_neg_matrix[team]
        _neg_transform_pose(p_old_positions, p_old_rotations, p, m, 1.0, 1.0)
        _neg_transform_pose(p_old_anim_positions, p_old_anim_rotations, p, m, 1.0, 1.0)
        dpx, dpy, dpz = dmath.transform_point(m, p_display_positions[p, 0],
                                              p_display_positions[p, 1],
                                              p_display_positions[p, 2])
        p_display_positions[p, 0] = dpx
        p_display_positions[p, 1] = dpy
        p_display_positions[p, 2] = dpz
        vx, vy, vz = dmath.transform_vector(m, p_velocities[p, 0], p_velocities[p, 1],
                                            p_velocities[p, 2])
        p_velocities[p, 0] = vx
        p_velocities[p, 1] = vy
        p_velocities[p, 2] = vz
        rvx, rvy, rvz = dmath.transform_vector(m, p_real_velocities[p, 0],
                                               p_real_velocities[p, 1],
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


@wp.func
def do_collider_frame_pre(ci: int, c_team: wp.array(dtype=int),
                          c_enabled: wp.array(dtype=int),
                          c_enabled_prev: wp.array(dtype=int),
                          c_active: wp.array(dtype=int),
                          c_input_positions: wp.array2d(dtype=float),
                          c_input_rotations: wp.array2d(dtype=float),
                          c_input_tips: wp.array2d(dtype=float),
                          c_input_radii: wp.array2d(dtype=float),
                          c_frame_pos: wp.array2d(dtype=float),
                          c_frame_rot: wp.array2d(dtype=float),
                          c_frame_tip: wp.array2d(dtype=float),
                          c_frame_radius: wp.array2d(dtype=float),
                          c_old_frame_pos: wp.array2d(dtype=float),
                          c_old_frame_rot: wp.array2d(dtype=float),
                          c_old_frame_tip: wp.array2d(dtype=float),
                          c_now_pos: wp.array2d(dtype=float),
                          c_now_rot: wp.array2d(dtype=float),
                          c_now_tip: wp.array2d(dtype=float),
                          c_old_pos: wp.array2d(dtype=float),
                          c_old_rot: wp.array2d(dtype=float),
                          c_old_tip: wp.array2d(dtype=float),
                          t_reset_pending: wp.array(dtype=int),
                          t_neg_teleport: wp.array(dtype=int),
                          t_neg_matrix: wp.array(dtype=wp.mat44d),
                          t_neg_change: wp.array2d(dtype=float),
                          t_inertia_shift: wp.array(dtype=int),
                          t_shift_vec: wp.array2d(dtype=float),
                          t_shift_rot: wp.array2d(dtype=float),
                          t_old_cwp: wp.array2d(dtype=float)):
    enabled_now = c_enabled[ci] != 0
    rising = enabled_now and (c_enabled_prev[ci] == 0)
    c_active[ci] = 1 if enabled_now else 0
    c_enabled_prev[ci] = 1 if enabled_now else 0
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
        _shift_pose(c_old_frame_pos, c_old_frame_rot, ci, cpx, cpy, cpz, svx, svy, svz,
                    srx, sry, srz, srw)
        _shift_pose(c_now_pos, c_now_rot, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_pose(c_old_pos, c_old_rot, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_point(c_old_frame_tip, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_point(c_now_tip, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_point(c_old_tip, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)


@wp.func
def do_collider_start_step(ci: int, c_team: wp.array(dtype=int), c_kind: wp.array(dtype=int),
                           c_frame_pos: wp.array2d(dtype=float),
                           c_frame_rot: wp.array2d(dtype=float),
                           c_frame_tip: wp.array2d(dtype=float),
                           c_frame_radius: wp.array2d(dtype=float),
                           c_old_frame_pos: wp.array2d(dtype=float),
                           c_old_frame_rot: wp.array2d(dtype=float),
                           c_old_frame_tip: wp.array2d(dtype=float),
                           c_now_pos: wp.array2d(dtype=float),
                           c_now_rot: wp.array2d(dtype=float),
                           c_now_tip: wp.array2d(dtype=float),
                           c_old_pos: wp.array2d(dtype=float),
                           c_old_rot: wp.array2d(dtype=float),
                           c_old_tip: wp.array2d(dtype=float),
                           c_work_rot: wp.array2d(dtype=float),
                           c_work_inv_old_rot: wp.array2d(dtype=float),
                           c_work_radius: wp.array2d(dtype=float),
                           c_work_old_pos: wp.array3d(dtype=float),
                           c_work_next_pos: wp.array3d(dtype=float),
                           c_work_aabb_min: wp.array2d(dtype=float),
                           c_work_aabb_max: wp.array2d(dtype=float),
                           t_frame_interp: wp.array(dtype=float),
                           t_step_mir: wp.array(dtype=float),
                           t_step_rir: wp.array(dtype=float)):
    team = c_team[ci]
    t = t_frame_interp[team]
    posx = dmath.lerp(c_old_frame_pos[ci, 0], c_frame_pos[ci, 0], t)
    posy = dmath.lerp(c_old_frame_pos[ci, 1], c_frame_pos[ci, 1], t)
    posz = dmath.lerp(c_old_frame_pos[ci, 2], c_frame_pos[ci, 2], t)
    tipx = dmath.lerp(c_old_frame_tip[ci, 0], c_frame_tip[ci, 0], t)
    tipy = dmath.lerp(c_old_frame_tip[ci, 1], c_frame_tip[ci, 1], t)
    tipz = dmath.lerp(c_old_frame_tip[ci, 2], c_frame_tip[ci, 2], t)
    rotx, roty, rotz, rotw = dmath.quat_slerp(
        c_old_frame_rot[ci, 0], c_old_frame_rot[ci, 1], c_old_frame_rot[ci, 2],
        c_old_frame_rot[ci, 3],
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
        c_work_aabb_min[ci, 0] = dmath.fmin2(dmath.fmin2(sox, snx) - start_radius,
                                             dmath.fmin2(eox, enx) - end_radius)
        c_work_aabb_min[ci, 1] = dmath.fmin2(dmath.fmin2(soy, sny) - start_radius,
                                             dmath.fmin2(eoy, eny) - end_radius)
        c_work_aabb_min[ci, 2] = dmath.fmin2(dmath.fmin2(soz, snz) - start_radius,
                                             dmath.fmin2(eoz, enz) - end_radius)
        c_work_aabb_max[ci, 0] = dmath.fmax2(dmath.fmax2(sox, snx) + start_radius,
                                             dmath.fmax2(eox, enx) + end_radius)
        c_work_aabb_max[ci, 1] = dmath.fmax2(dmath.fmax2(soy, sny) + start_radius,
                                             dmath.fmax2(eoy, eny) + end_radius)
        c_work_aabb_max[ci, 2] = dmath.fmax2(dmath.fmax2(soz, snz) + start_radius,
                                             dmath.fmax2(eoz, enz) + end_radius)
    else:
        nx, ny, nz = dmath.quat_rotate(rotx, roty, rotz, rotw, 0.0, 0.0, 1.0)
        c_work_old_pos[ci, 0, 0] = nx
        c_work_old_pos[ci, 0, 1] = ny
        c_work_old_pos[ci, 0, 2] = nz
        c_work_next_pos[ci, 0, 0] = posx
        c_work_next_pos[ci, 0, 1] = posy
        c_work_next_pos[ci, 0, 2] = posz
        c_work_aabb_min[ci, 0] = -wp.inf
        c_work_aabb_min[ci, 1] = -wp.inf
        c_work_aabb_min[ci, 2] = -wp.inf
        c_work_aabb_max[ci, 0] = wp.inf
        c_work_aabb_max[ci, 1] = wp.inf
        c_work_aabb_max[ci, 2] = wp.inf


@wp.func
def do_collider_end_step(ci: int, c_now_pos: wp.array2d(dtype=float),
                         c_now_rot: wp.array2d(dtype=float),
                         c_now_tip: wp.array2d(dtype=float),
                         c_old_pos: wp.array2d(dtype=float),
                         c_old_rot: wp.array2d(dtype=float),
                         c_old_tip: wp.array2d(dtype=float)):
    for j in range(3):
        c_old_pos[ci, j] = c_now_pos[ci, j]
        c_old_tip[ci, j] = c_now_tip[ci, j]
    for j in range(4):
        c_old_rot[ci, j] = c_now_rot[ci, j]


@wp.func
def do_collider_frame_post(ci: int, c_frame_pos: wp.array2d(dtype=float),
                           c_frame_rot: wp.array2d(dtype=float),
                           c_frame_tip: wp.array2d(dtype=float),
                           c_old_frame_pos: wp.array2d(dtype=float),
                           c_old_frame_rot: wp.array2d(dtype=float),
                           c_old_frame_tip: wp.array2d(dtype=float)):
    for j in range(3):
        c_old_frame_pos[ci, j] = c_frame_pos[ci, j]
        c_old_frame_tip[ci, j] = c_frame_tip[ci, j]
    for j in range(4):
        c_old_frame_rot[ci, j] = c_frame_rot[ci, j]


@wp.func
def do_solve_point(p: int, p_team: wp.array(dtype=int),
                   p_next_positions: wp.array2d(dtype=float),
                   p_base_positions: wp.array2d(dtype=float),
                   p_depth: wp.array(dtype=float),
                   p_friction: wp.array(dtype=float),
                   p_collision_normals: wp.array2d(dtype=float),
                   p_velocity_positions: wp.array2d(dtype=float),
                   t_collision_mode: wp.array(dtype=int),
                   t_radius_lut: wp.array2d(dtype=float),
                   t_scale_ratio: wp.array(dtype=float),
                   t_is_spring: wp.array(dtype=int),
                   t_limit_distance_lut: wp.array2d(dtype=float),
                   c_kind: wp.array(dtype=int),
                   c_active: wp.array(dtype=int),
                   c_work_old_pos: wp.array3d(dtype=float),
                   c_work_next_pos: wp.array3d(dtype=float),
                   c_work_radius: wp.array2d(dtype=float),
                   c_work_inv_old_rot: wp.array2d(dtype=float),
                   c_work_rot: wp.array2d(dtype=float),
                   c_work_aabb_min: wp.array2d(dtype=float),
                   c_work_aabb_max: wp.array2d(dtype=float),
                   csr_off: wp.array(dtype=int),
                   csr_ord: wp.array(dtype=int),
                   st_pp_collider: wp.array(dtype=int)):
    team = p_team[p]
    if t_collision_mode[team] != COLLISION_POINT:
        return
    start = csr_off[p]
    stop = csr_off[p + 1]
    if start == stop:
        return
    depth = p_depth[p]
    radius = dmath.evaluate_team_lut(t_radius_lut, team, depth)
    if radius < 0.0001:
        radius = 0.0001
    radius = radius * t_scale_ratio[team]
    cfr = radius
    npx = p_next_positions[p, 0]
    npy = p_next_positions[p, 1]
    npz = p_next_positions[p, 2]
    is_spring = t_is_spring[team] != 0
    max_length = dmath.evaluate_team_lut(t_limit_distance_lut, team, depth)
    if max_length < 0.0001:
        max_length = 0.0001
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
    count = int(0)
    near_count = int(0)
    psumx = wp.float64(0.0)
    psumy = wp.float64(0.0)
    psumz = wp.float64(0.0)
    nsumx = wp.float64(0.0)
    nsumy = wp.float64(0.0)
    nsumz = wp.float64(0.0)
    nnx = wp.float64(0.0)
    nny = wp.float64(0.0)
    nnz = wp.float64(0.0)
    min_dist = float(wp.inf)
    any_active = wp.bool(False)
    for k in range(start, stop):
        c = st_pp_collider[csr_ord[k]]
        if c_active[c] == 0:
            continue
        any_active = True
        kind = c_kind[c]
        ux = 0.0
        uy = 0.0
        uz = 0.0
        ox = npx
        oy = npy
        oz = npz
        dist = wp.inf
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
            d, outx, outy, outz = dmath.intersect_point_plane_dist(ppx, ppy, ppz,
                                                                   nrx, nry, nrz,
                                                                   npx, npy, npz)
            if is_spring:
                clx, cly, clz = dmath.clamp_distance(bpx, bpy, bpz, outx, outy, outz,
                                                     max_length)
                lspr = dmath.length3(clx - bpx, cly - bpy, clz - bpz)
                tspr = dmath.saturate(lspr / radius) * 0.85
                outx = dmath.lerp(clx, npx, tspr)
                outy = dmath.lerp(cly, npy, tspr)
                outz = dmath.lerp(clz, npz, tspr)
                d = d * 3.0
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
            ts = dmath.closest_pt_point_segment_ratio(npx, npy, npz, sox, soy, soz,
                                                      eox, eoy, eoz)
            r = sr + (er - sr) * ts
            d0x = sox + (eox - sox) * ts
            d0y = soy + (eoy - soy) * ts
            d0z = soz + (eoz - soz) * ts
            lvx, lvy, lvz = dmath.quat_rotate(c_work_inv_old_rot[c, 0],
                                              c_work_inv_old_rot[c, 1],
                                              c_work_inv_old_rot[c, 2],
                                              c_work_inv_old_rot[c, 3],
                                              npx - d0x, npy - d0y, npz - d0z)
            d1x = snx + (enx - snx) * ts
            d1y = sny + (eny - sny) * ts
            d1z = snz + (enz - snz) * ts
            v2x, v2y, v2z = dmath.quat_rotate(c_work_rot[c, 0], c_work_rot[c, 1],
                                              c_work_rot[c, 2], c_work_rot[c, 3],
                                              lvx, lvy, lvz)
            nrx, nry, nrz = dmath.normalize3(v2x, v2y, v2z)
            ppx = d1x + nrx * (r + radius)
            ppy = d1y + nry * (r + radius)
            ppz = d1z + nrz * (r + radius)
            d, outx, outy, outz = dmath.intersect_point_plane_dist(ppx, ppy, ppz,
                                                                   nrx, nry, nrz,
                                                                   npx, npy, npz)
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
            d, outx, outy, outz = dmath.intersect_point_plane_dist(ppx, ppy, ppz,
                                                                   nrx, nry, nrz,
                                                                   npx, npy, npz)
            dist = d
            ox = outx
            oy = outy
            oz = outz
            ux = nrx
            uy = nry
            uz = nrz
        if dist <= 0.0:
            count += 1
            psumx += wp.float64(ox - npx)
            psumy += wp.float64(oy - npy)
            psumz += wp.float64(oz - npz)
            nsumx += wp.float64(ux)
            nsumy += wp.float64(uy)
            nsumz += wp.float64(uz)
        if dist <= cfr:
            near_count += 1
            nnx += wp.float64(ux)
            nny += wp.float64(uy)
            nnz += wp.float64(uz)
            if dist < min_dist:
                min_dist = dist
    if not any_active:
        return
    has_push = count > 0
    sc = float(count) if count > 0 else 1.0
    navx = wp.float32(nsumx) / sc
    navy = wp.float32(nsumy) / sc
    navz = wp.float32(nsumz) / sc
    normal_length = dmath.length3(navx, navy, navz)
    pavx = wp.float32(psumx) / sc
    pavy = wp.float32(psumy) / sc
    pavz = wp.float32(psumz) / sc
    tclamp = normal_length if normal_length < 1.0 else 1.0
    if has_push and (normal_length >= EPSILON):
        p_next_positions[p, 0] = npx + pavx * tclamp
        p_next_positions[p, 1] = npy + pavy * tclamp
        p_next_positions[p, 2] = npz + pavz * tclamp
    nsx = wp.float32(nnx)
    nsy = wp.float32(nny)
    nsz = wp.float32(nnz)
    near_len = dmath.length3(nsx, nsy, nsz)
    has_near = (near_count > 0) and (cfr > 0.0) and (near_len * near_len > 1e-6)
    md = min_dist if (min_dist < wp.inf) else 0.0
    denom_cfr = cfr if cfr > 0.0 else 1.0
    friction_val = 1.0 - dmath.saturate(md / denom_cfr)
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
        p_velocity_positions[p, 0] = p_velocity_positions[p, 0] + pavx
        p_velocity_positions[p, 1] = p_velocity_positions[p, 1] + pavy
        p_velocity_positions[p, 2] = p_velocity_positions[p, 2] + pavz


@wp.func
def do_solve_edge(ee: int, p_team: wp.array(dtype=int),
                  p_next_positions: wp.array2d(dtype=float),
                  p_depth: wp.array(dtype=float),
                  p_attr_move: wp.array(dtype=int),
                  t_radius_lut: wp.array2d(dtype=float),
                  t_scale_ratio: wp.array(dtype=float),
                  c_kind: wp.array(dtype=int),
                  c_active: wp.array(dtype=int),
                  c_work_old_pos: wp.array3d(dtype=float),
                  c_work_next_pos: wp.array3d(dtype=float),
                  c_work_radius: wp.array2d(dtype=float),
                  c_work_aabb_min: wp.array2d(dtype=float),
                  c_work_aabb_max: wp.array2d(dtype=float),
                  csr_off: wp.array(dtype=int),
                  csr_ord: wp.array(dtype=int),
                  st_ep_collider: wp.array(dtype=int),
                  st_collision_edge: wp.array2d(dtype=int),
                  sc_dcorr_fixed: wp.array2d(dtype=int),
                  sc_dcount: wp.array(dtype=int),
                  sc_col_friction_fixed: wp.array(dtype=int),
                  sc_col_normal_fixed: wp.array2d(dtype=int)):
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
    cfr = (r0 + r1) * 0.5
    emnx = dmath.fmin2(p0x - r0, p1x - r1) - cfr
    emny = dmath.fmin2(p0y - r0, p1y - r1) - cfr
    emnz = dmath.fmin2(p0z - r0, p1z - r1) - cfr
    emxx = dmath.fmax2(p0x + r0, p1x + r1) + cfr
    emxy = dmath.fmax2(p0y + r0, p1y + r1) + cfr
    emxz = dmath.fmax2(p0z + r0, p1z + r1) + cfr
    count = int(0)
    near_count = int(0)
    c0sx = wp.float64(0.0)
    c0sy = wp.float64(0.0)
    c0sz = wp.float64(0.0)
    c1sx = wp.float64(0.0)
    c1sy = wp.float64(0.0)
    c1sz = wp.float64(0.0)
    nsumx = wp.float64(0.0)
    nsumy = wp.float64(0.0)
    nsumz = wp.float64(0.0)
    nnx = wp.float64(0.0)
    nny = wp.float64(0.0)
    nnz = wp.float64(0.0)
    min_dist = float(wp.inf)
    any_active = wp.bool(False)
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
                                                                 cpz + uz * r0, ux, uy, uz,
                                                                 p0x, p0y, p0z)
            d1, o1x, o1y, o1z = dmath.intersect_point_plane_dist(cpx + ux * r1, cpy + uy * r1,
                                                                 cpz + uz * r1, ux, uy, uz,
                                                                 p1x, p1y, p1z)
            d = d0 if d0 < d1 else d1
            a0x = o0x - p0x
            a0y = o0y - p0y
            a0z = o0z - p0z
            a1x = o1x - p1x
            a1y = o1y - p1y
            a1z = o1z - p1z
        if d <= 0.0:
            count += 1
            c0sx += wp.float64(a0x)
            c0sy += wp.float64(a0y)
            c0sz += wp.float64(a0z)
            c1sx += wp.float64(a1x)
            c1sy += wp.float64(a1y)
            c1sz += wp.float64(a1z)
            nsumx += wp.float64(ux)
            nsumy += wp.float64(uy)
            nsumz += wp.float64(uz)
        if d <= cfr:
            near_count += 1
            nnx += wp.float64(ux)
            nny += wp.float64(uy)
            nnz += wp.float64(uz)
            if d < min_dist:
                min_dist = d
    if not any_active:
        return
    has_push = count > 0
    sc = float(count) if count > 0 else 1.0
    navx = wp.float32(nsumx) / sc
    navy = wp.float32(nsumy) / sc
    navz = wp.float32(nsumz) / sc
    normal_length = dmath.length3(navx, navy, navz)
    tclamp = normal_length if normal_length < 1.0 else 1.0
    valid = has_push and (normal_length > EPSILON)
    scale = (tclamp if valid else 0.0) / sc
    d0x = wp.float32(c0sx) * scale
    d0y = wp.float32(c0sy) * scale
    d0z = wp.float32(c0sz) * scale
    d1x = wp.float32(c1sx) * scale
    d1y = wp.float32(c1sy) * scale
    d1z = wp.float32(c1sz) * scale
    nsx = wp.float32(nnx)
    nsy = wp.float32(nny)
    nsz = wp.float32(nnz)
    near_len = dmath.length3(nsx, nsy, nsz)
    has_near = (near_count > 0) and (cfr > 0.0) and (near_len * near_len > 1e-6)
    md = min_dist if (min_dist < wp.inf) else 0.0
    denom_cfr = cfr if cfr > 0.0 else 1.0
    friction = 1.0 - dmath.saturate(md / denom_cfr)
    if has_near:
        noutx, nouty, noutz = dmath.normalize3(nsx, nsy, nsz)
    else:
        noutx = 0.0
        nouty = 0.0
        noutz = 0.0
    move0 = p_attr_move[e0] != 0
    move1 = p_attr_move[e1] != 0
    mask0 = has_near and move0
    mask1 = has_near and move1
    vint = 1 if valid else 0
    wp.atomic_add(sc_dcorr_fixed, e0, 0, int(d0x * TO_FIXED))
    wp.atomic_add(sc_dcorr_fixed, e0, 1, int(d0y * TO_FIXED))
    wp.atomic_add(sc_dcorr_fixed, e0, 2, int(d0z * TO_FIXED))
    wp.atomic_add(sc_dcount, e0, vint)
    wp.atomic_add(sc_dcorr_fixed, e1, 0, int(d1x * TO_FIXED))
    wp.atomic_add(sc_dcorr_fixed, e1, 1, int(d1y * TO_FIXED))
    wp.atomic_add(sc_dcorr_fixed, e1, 2, int(d1z * TO_FIXED))
    wp.atomic_add(sc_dcount, e1, vint)
    f0 = friction if mask0 else 0.0
    f1 = friction if mask1 else 0.0
    wp.atomic_max(sc_col_friction_fixed, e0, int(f0 * TO_FIXED))
    wp.atomic_max(sc_col_friction_fixed, e1, int(f1 * TO_FIXED))
    if mask0:
        wp.atomic_add(sc_col_normal_fixed, e0, 0, int(noutx * TO_FIXED))
        wp.atomic_add(sc_col_normal_fixed, e0, 1, int(nouty * TO_FIXED))
        wp.atomic_add(sc_col_normal_fixed, e0, 2, int(noutz * TO_FIXED))
    if mask1:
        wp.atomic_add(sc_col_normal_fixed, e1, 0, int(noutx * TO_FIXED))
        wp.atomic_add(sc_col_normal_fixed, e1, 1, int(nouty * TO_FIXED))
        wp.atomic_add(sc_col_normal_fixed, e1, 2, int(noutz * TO_FIXED))


@wp.func
def do_self_update_primitive(prim: int, axes: int,
                             a_team: wp.array(dtype=int),
                             a_particles: wp.array2d(dtype=int),
                             a_fix: wp.array(dtype=int),
                             a_ignore: wp.array(dtype=int),
                             a_prim_depth: wp.array(dtype=float),
                             a_inv_mass: wp.array2d(dtype=float),
                             a_thickness: wp.array(dtype=float),
                             a_aabb_min: wp.array2d(dtype=float),
                             a_aabb_max: wp.array2d(dtype=float),
                             a_intersect: wp.array(dtype=int),
                             a_use: wp.array(dtype=int),
                             t_use_flag: wp.array(dtype=int),
                             t_thickness_lut: wp.array2d(dtype=float),
                             t_cloth_mass: wp.array(dtype=float),
                             t_scale_ratio: wp.array(dtype=float),
                             t_enabled: wp.array(dtype=int),
                             t_valid: wp.array(dtype=int),
                             t_cws: wp.array2d(dtype=float),
                             t_update_count: wp.array(dtype=int),
                             p_next: wp.array2d(dtype=float),
                             p_old: wp.array2d(dtype=float),
                             p_friction: wp.array(dtype=float),
                             p_iflag: wp.array(dtype=int),
                             scl_counts: wp.array(dtype=int),
                             scl_max_fixed: wp.array(dtype=int),
                             k: int):
    team = a_team[prim]
    if not (team_frame_mask(t_enabled, t_valid, t_cws, team) and t_update_count[team] > k
            and t_use_flag[team] != 0):
        a_use[prim] = 0
        return
    a_use[prim] = 1
    fix_mask = a_fix[prim]
    thickness = dmath.evaluate_team_lut(t_thickness_lut, team,
                                        a_prim_depth[prim]) * t_scale_ratio[team]
    a_thickness[prim] = thickness
    cloth_mass = t_cloth_mass[team]
    use_intersect = scl_counts[SCL_USE_INTERSECT] != 0
    imask = int(0)
    lowx = float(1.0e30)
    lowy = float(1.0e30)
    lowz = float(1.0e30)
    highx = float(-1.0e30)
    highy = float(-1.0e30)
    highz = float(-1.0e30)
    for slot in range(axes):
        raw = a_particles[prim, slot]
        pp = raw if raw >= 0 else int(0)
        fixed = ((fix_mask >> slot) & int(1)) != 0
        a_inv_mass[prim, slot] = dmath.calc_self_collision_inverse_mass(
            p_friction[pp], fixed, cloth_mass)
        nx = p_next[pp, 0]
        ny = p_next[pp, 1]
        nz = p_next[pp, 2]
        ox = p_old[pp, 0]
        oy = p_old[pp, 1]
        oz = p_old[pp, 2]
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
            imask = imask | (int(1) << slot)
    a_intersect[prim] = imask
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
        wp.atomic_max(scl_max_fixed, team, int(size * TO_FIXED))


@wp.func
def self_aabb_overlap(a_min: wp.array2d(dtype=float), a_max: wp.array2d(dtype=float), i: int,
                      b_min: wp.array2d(dtype=float), b_max: wp.array2d(dtype=float), j: int):
    return (a_min[i, 0] <= b_max[j, 0] and a_max[i, 0] >= b_min[j, 0]
            and a_min[i, 1] <= b_max[j, 1] and a_max[i, 1] >= b_min[j, 1]
            and a_min[i, 2] <= b_max[j, 2] and a_max[i, 2] >= b_min[j, 2])


@wp.func
def self_connection_shared(a_particles: wp.array2d(dtype=int), i: int,
                           b_particles: wp.array2d(dtype=int), j: int):
    for x in range(3):
        pa = a_particles[i, x]
        if pa >= 0:
            for y in range(3):
                pb = b_particles[j, y]
                if pb >= 0 and pa == pb:
                    return True
    return False


@wp.func
def self_ee_geometry(my_edge: int, tgt_edge: int, thickness: float,
                     sfe_particles: wp.array2d(dtype=int),
                     p_next: wp.array2d(dtype=float),
                     p_old: wp.array2d(dtype=float)):
    scr = thickness * SELF_COLLISION_SCR
    a0 = sfe_particles[my_edge, 0]
    a1 = sfe_particles[my_edge, 1]
    b0 = sfe_particles[tgt_edge, 0]
    b1 = sfe_particles[tgt_edge, 1]
    s, t, c1x, c1y, c1z, c2x, c2y, c2z = dmath.closest_pt_segment_segment(
        p_old[a0, 0], p_old[a0, 1], p_old[a0, 2], p_old[a1, 0], p_old[a1, 1], p_old[a1, 2],
        p_old[b0, 0], p_old[b0, 1], p_old[b0, 2], p_old[b1, 0], p_old[b1, 1], p_old[b1, 2])
    cdx = c1x - c2x
    cdy = c1y - c2y
    cdz = c1z - c2z
    clen = dmath.length3(cdx, cdy, cdz)
    ok = clen >= 1.0e-9
    safe = clen if clen > 1.0e-30 else 1.0
    nx = cdx / safe
    ny = cdy / safe
    nz = cdz / safe
    dax = dmath.lerp(p_next[a0, 0] - p_old[a0, 0], p_next[a1, 0] - p_old[a1, 0], s)
    day = dmath.lerp(p_next[a0, 1] - p_old[a0, 1], p_next[a1, 1] - p_old[a1, 1], s)
    daz = dmath.lerp(p_next[a0, 2] - p_old[a0, 2], p_next[a1, 2] - p_old[a1, 2], s)
    dbx = dmath.lerp(p_next[b0, 0] - p_old[b0, 0], p_next[b1, 0] - p_old[b1, 0], t)
    dby = dmath.lerp(p_next[b0, 1] - p_old[b0, 1], p_next[b1, 1] - p_old[b1, 1], t)
    dbz = dmath.lerp(p_next[b0, 2] - p_old[b0, 2], p_next[b1, 2] - p_old[b1, 2], t)
    l = clen + (nx * dax + ny * day + nz * daz) - (nx * dbx + ny * dby + nz * dbz)
    ok = ok and (l <= (thickness + scr))
    return s, t, nx, ny, nz, ok


@wp.func
def self_pt_geometry(point_prim: int, tri_prim: int, thickness: float, first: bool,
                     sfp_particles: wp.array2d(dtype=int),
                     sft_particles: wp.array2d(dtype=int),
                     p_next: wp.array2d(dtype=float),
                     p_old: wp.array2d(dtype=float)):
    scr = thickness * SELF_COLLISION_SCR
    pp = sfp_particles[point_prim, 0]
    t0 = sft_particles[tri_prim, 0]
    t1 = sft_particles[tri_prim, 1]
    t2 = sft_particles[tri_prim, 2]
    oax = p_old[pp, 0]
    oay = p_old[pp, 1]
    oaz = p_old[pp, 2]
    ob0x = p_old[t0, 0]
    ob0y = p_old[t0, 1]
    ob0z = p_old[t0, 2]
    ob1x = p_old[t1, 0]
    ob1y = p_old[t1, 1]
    ob1z = p_old[t1, 2]
    ob2x = p_old[t2, 0]
    ob2y = p_old[t2, 1]
    ob2z = p_old[t2, 2]
    dax = p_next[pp, 0] - oax
    day = p_next[pp, 1] - oay
    daz = p_next[pp, 2] - oaz
    db0x = p_next[t0, 0] - ob0x
    db0y = p_next[t0, 1] - ob0y
    db0z = p_next[t0, 2] - ob0z
    db1x = p_next[t1, 0] - ob1x
    db1y = p_next[t1, 1] - ob1y
    db1z = p_next[t1, 2] - ob1z
    db2x = p_next[t2, 0] - ob2x
    db2y = p_next[t2, 1] - ob2y
    db2z = p_next[t2, 2] - ob2z
    cpx, cpy, cpz, u, v, w = dmath.closest_pt_point_triangle(
        oax, oay, oaz, ob0x, ob0y, ob0z, ob1x, ob1y, ob1z, ob2x, ob2y, ob2z)
    dtx = db0x * u + db1x * v + db2x * w
    dty = db0y * u + db1y * v + db2y * w
    dtz = db0z * u + db1z * v + db2z * w
    cvx = cpx - oax
    cvy = cpy - oay
    cvz = cpz - oaz
    cvlen = dmath.length3(cvx, cvy, cvz)
    ok = cvlen > EPSILON
    safe = cvlen if cvlen > 1.0e-30 else 1.0
    nx = cvx / safe
    ny = cvy / safe
    nz = cvz / safe
    l = cvlen - (nx * dax + ny * day + nz * daz) + (nx * dtx + ny * dty + nz * dtz)
    ok = ok and (l < (thickness + scr))
    sign = float(0.0)
    if first:
        otnx, otny, otnz = dmath.triangle_normal(ob0x, ob0y, ob0z, ob1x, ob1y, ob1z,
                                                 ob2x, ob2y, ob2z)
        n2x, n2y, n2z = dmath.normalize3(oax - cpx, oay - cpy, oaz - cpz)
        d = otnx * n2x + otny * n2y + otnz * n2z
        ok = ok and (wp.abs(d) >= SELF_COLLISION_POINT_TRIANGLE_ANGLE_COS)
        sign = dmath.fsign(d)
    return ok, sign


@wp.func
def do_angle_limit(v: int, p: int, vt: int, c_inv: float, p_inv: float, p_move: bool,
                   p_next_positions: wp.array2d(dtype=float),
                   p_velocity_positions: wp.array2d(dtype=float),
                   p_albuf_rotation: wp.array2d(dtype=float),
                   p_albuf_local_pos: wp.array2d(dtype=float),
                   p_albuf_local_rot: wp.array2d(dtype=float),
                   p_albuf_length: wp.array(dtype=float),
                   p_depth: wp.array(dtype=float),
                   t_angle_limit_lut: wp.array2d(dtype=float),
                   t_angle_limit_stiffness: wp.array(dtype=float)):
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
        p_velocity_positions[v, 0] = p_velocity_positions[v, 0] + (ppx - cpx)
        p_velocity_positions[v, 1] = p_velocity_positions[v, 1] + (ppy - cpy)
        p_velocity_positions[v, 2] = p_velocity_positions[v, 2] + (ppz - cpz)
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
    safe_vlen = vlen if vlen > 1.0e-30 else 1.0
    uvx = vvx / safe_vlen
    uvy = vvy / safe_vlen
    uvz = vvz / safe_vlen
    safe_tvlen = tvlen if tvlen > 1.0e-30 else 1.0
    utvx = tvx / safe_tvlen
    utvy = tvy / safe_tvlen
    utvz = tvz / safe_tvlen
    blen = p_albuf_length[v]
    vlen2 = dmath.lerp(vlen, blen, 0.5)
    work = work and (blen >= EPSILON) and (vlen2 >= EPSILON)
    vsx = uvx * vlen2
    vsy = uvy * vlen2
    vsz = uvz * vlen2
    ang = dmath.angle_between(vsx, vsy, vsz, utvx, utvy, utvz)
    max_angle = DEG2RAD * dmath.evaluate_team_lut(t_angle_limit_lut, vt, p_depth[v])
    over = ang > max_angle
    recovery = dmath.lerp(ang, max_angle, t_angle_limit_stiffness[vt])
    clx, cly, clz = dmath.clamp_angle_vector(vsx, vsy, vsz, utvx, utvy, utvz, recovery)
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
    cfx = rpx + rvx * (1.0 - ANGLE_LIMIT_ROT_RATIO)
    cfy = rpy + rvy * (1.0 - ANGLE_LIMIT_ROT_RATIO)
    cfz = rpz + rvz * (1.0 - ANGLE_LIMIT_ROT_RATIO)
    if work:
        paddx = (pfx - ppx) * p_inv
        paddy = (pfy - ppy) * p_inv
        paddz = (pfz - ppz) * p_inv
        caddx = (cfx - cpx) * c_inv
        caddy = (cfy - cpy) * c_inv
        caddz = (cfz - cpz) * c_inv
    else:
        paddx = 0.0
        paddy = 0.0
        paddz = 0.0
        caddx = 0.0
        caddy = 0.0
        caddz = 0.0
    cpx = cpx + caddx
    cpy = cpy + caddy
    cpz = cpz + caddz
    p_next_positions[v, 0] = cpx
    p_next_positions[v, 1] = cpy
    p_next_positions[v, 2] = cpz
    p_velocity_positions[v, 0] = p_velocity_positions[v, 0] + caddx * ANGLE_LIMIT_ATTENUATION
    p_velocity_positions[v, 1] = p_velocity_positions[v, 1] + caddy * ANGLE_LIMIT_ATTENUATION
    p_velocity_positions[v, 2] = p_velocity_positions[v, 2] + caddz * ANGLE_LIMIT_ATTENUATION
    if work and p_move:
        ppx = ppx + paddx
        ppy = ppy + paddy
        ppz = ppz + paddz
        p_next_positions[p, 0] = ppx
        p_next_positions[p, 1] = ppy
        p_next_positions[p, 2] = ppz
        p_velocity_positions[p, 0] = p_velocity_positions[p, 0] \
            + paddx * ANGLE_LIMIT_ATTENUATION
        p_velocity_positions[p, 1] = p_velocity_positions[p, 1] \
            + paddy * ANGLE_LIMIT_ATTENUATION
        p_velocity_positions[p, 2] = p_velocity_positions[p, 2] \
            + paddz * ANGLE_LIMIT_ATTENUATION
    v3x = cpx - ppx
    v3y = cpy - ppy
    v3z = cpz - ppz
    vlen3 = dmath.length3(v3x, v3y, v3z)
    fix_ok = work and (vlen3 >= EPSILON)
    safe_v3 = vlen3 if vlen3 > 1.0e-30 else 1.0
    uv3x = v3x / safe_v3
    uv3y = v3y / safe_v3
    uv3z = v3z / safe_v3
    nrx, nry, nrz, nrw = dmath.quat_mul(prx, pry, prz, prw, lrx, lry, lrz, lrw)
    qx, qy, qz, qw = dmath.from_to_rotation(utvx, utvy, utvz, uv3x, uv3y, uv3z, 1.0, True)
    frx, fry, frz, frw = dmath.quat_mul(qx, qy, qz, qw, nrx, nry, nrz, nrw)
    if fix_ok:
        p_albuf_rotation[v, 0] = frx
        p_albuf_rotation[v, 1] = fry
        p_albuf_rotation[v, 2] = frz
        p_albuf_rotation[v, 3] = frw


@wp.func
def do_angle_restoration(v: int, p: int, vt: int, c_inv: float, p_inv: float, p_move: bool,
                         rot_ratio: float, power3: float,
                         p_next_positions: wp.array2d(dtype=float),
                         p_velocity_positions: wp.array2d(dtype=float),
                         p_albuf_restore: wp.array2d(dtype=float),
                         p_depth: wp.array(dtype=float),
                         t_angle_restoration_lut: wp.array2d(dtype=float),
                         t_angle_restoration_attenuation: wp.array(dtype=float),
                         t_angle_restoration_gravity_falloff: wp.array(dtype=float),
                         t_gravity_dot: wp.array(dtype=float)):
    stiff = dmath.evaluate_team_lut_clamp01(t_angle_restoration_lut, vt, p_depth[v])
    stiff = dmath.saturate(stiff * power3)
    gfo = dmath.lerp(1.0 - t_angle_restoration_gravity_falloff[vt], 1.0, t_gravity_dot[vt])
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
        p_velocity_positions[v, 0] = p_velocity_positions[v, 0] + (ppx - cpx)
        p_velocity_positions[v, 1] = p_velocity_positions[v, 1] + (ppy - cpy)
        p_velocity_positions[v, 2] = p_velocity_positions[v, 2] + (ppz - cpz)
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
    safe_vlen = vlen if vlen > 1.0e-30 else 1.0
    uvx = vvx / safe_vlen
    uvy = vvy / safe_vlen
    uvz = vvz / safe_vlen
    safe_tvlen = tvlen if tvlen > 1.0e-30 else 1.0
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
    cfx = rpx + rvx * (1.0 - rot_ratio)
    cfy = rpy + rvy * (1.0 - rot_ratio)
    cfz = rpz + rvz * (1.0 - rot_ratio)
    if work:
        paddx = (pfx - ppx) * p_inv
        paddy = (pfy - ppy) * p_inv
        paddz = (pfz - ppz) * p_inv
        caddx = (cfx - cpx) * c_inv
        caddy = (cfy - cpy) * c_inv
        caddz = (cfz - cpz) * c_inv
    else:
        paddx = 0.0
        paddy = 0.0
        paddz = 0.0
        caddx = 0.0
        caddy = 0.0
        caddz = 0.0
    p_next_positions[v, 0] = cpx + caddx
    p_next_positions[v, 1] = cpy + caddy
    p_next_positions[v, 2] = cpz + caddz
    p_velocity_positions[v, 0] = p_velocity_positions[v, 0] + caddx * r_attn
    p_velocity_positions[v, 1] = p_velocity_positions[v, 1] + caddy * r_attn
    p_velocity_positions[v, 2] = p_velocity_positions[v, 2] + caddz * r_attn
    if work and p_move:
        p_next_positions[p, 0] = ppx + paddx
        p_next_positions[p, 1] = ppy + paddy
        p_next_positions[p, 2] = ppz + paddz
        p_velocity_positions[p, 0] = p_velocity_positions[p, 0] + paddx * r_attn
        p_velocity_positions[p, 1] = p_velocity_positions[p, 1] + paddy * r_attn
        p_velocity_positions[p, 2] = p_velocity_positions[p, 2] + paddz * r_attn


@wp.func
def do_display_particle(p: int, mt: int, sim_dt: float,
                        p_positions: wp.array2d(dtype=float),
                        p_rotations: wp.array2d(dtype=float),
                        p_old_positions: wp.array2d(dtype=float),
                        p_real_velocities: wp.array2d(dtype=float),
                        p_display_positions: wp.array2d(dtype=float),
                        p_vertex_root: wp.array(dtype=int),
                        p_old_anim_positions: wp.array2d(dtype=float),
                        p_old_anim_rotations: wp.array2d(dtype=float),
                        p_temp_base_positions: wp.array2d(dtype=float),
                        p_temp_base_rotations: wp.array2d(dtype=float),
                        st_update_move_mask: wp.array(dtype=int),
                        t_now_update: wp.array(dtype=float),
                        t_old_time: wp.array(dtype=float),
                        t_time: wp.array(dtype=float),
                        t_blend_weight: wp.array(dtype=float),
                        t_running: wp.array(dtype=int),
                        t_is_negative_scale: wp.array(dtype=int),
                        t_negative_scale_direction: wp.array2d(dtype=float)):
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
        if interval > 0.0:
            tval = (t_time[mt] - t_old_time[mt]) / interval
        else:
            tval = 0.0
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


@wp.func
def do_postline_entry(entry: int, et: int, ch_start: int, ch_end: int,
                      postline_child_vertices: wp.array(dtype=int),
                      p_positions: wp.array2d(dtype=float),
                      p_rotations: wp.array2d(dtype=float),
                      p_temp_base_positions: wp.array2d(dtype=float),
                      p_temp_base_rotations: wp.array2d(dtype=float),
                      p_vertex_local_positions: wp.array2d(dtype=float),
                      p_vertex_local_rotations: wp.array2d(dtype=float),
                      p_attr_invalid: wp.array(dtype=int),
                      p_attr_zero_distance: wp.array(dtype=int),
                      p_attr_move: wp.array(dtype=int),
                      p_team: wp.array(dtype=int),
                      t_rotational_interpolation: wp.array(dtype=float),
                      t_root_rotation: wp.array(dtype=float),
                      t_blend_weight: wp.array(dtype=float),
                      t_animation_pose_ratio: wp.array(dtype=float),
                      t_negative_scale_direction: wp.array2d(dtype=float),
                      t_negative_scale_quaternion: wp.array2d(dtype=float)):
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
    ctvx = wp.float64(0.0)
    ctvy = wp.float64(0.0)
    ctvz = wp.float64(0.0)
    cvx = wp.float64(0.0)
    cvy = wp.float64(0.0)
    cvz = wp.float64(0.0)
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
            tvx = 0.0
            tvy = 0.0
            tvz = 0.0
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
            ctvx += wp.float64(tvx)
            ctvy += wp.float64(tvy)
            ctvz += wp.float64(tvz)
            cvx += wp.float64(contx)
            cvy += wp.float64(conty)
            cvz += wp.float64(contz)
            if c_move:
                crx, cry, crz, crw = dmath.quat_mul(rx, ry, rz, rw, lrx, lry, lrz, lrw)
                if not is_c0:
                    qfx, qfy, qfz, qfw = dmath.from_to_rotation(tvx, tvy, tvz, vx, vy, vz,
                                                                1.0, False)
                    crx, cry, crz, crw = dmath.quat_mul(qfx, qfy, qfz, qfw, crx, cry, crz, crw)
                p_rotations[c, 0] = crx
                p_rotations[c, 1] = cry
                p_rotations[c, 2] = crz
                p_rotations[c, 3] = crw
    if has_children and owner_valid:
        ctv32x = wp.float32(ctvx)
        ctv32y = wp.float32(ctvy)
        ctv32z = wp.float32(ctvz)
        cv32x = wp.float32(cvx)
        cv32y = wp.float32(cvy)
        cv32z = wp.float32(cvz)
        zero = (dmath.length3(ctv32x, ctv32y, ctv32z) < 1e-8) \
            or (dmath.length3(cv32x, cv32y, cv32z) < 1e-8)
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


@wp.func
def do_triangle_normal_tangent(tri: int, tt_team: int,
                               st_triangle_particles: wp.array2d(dtype=int),
                               p_positions: wp.array2d(dtype=float),
                               p_uv: wp.array2d(dtype=float),
                               t_negative_scale_triangle_sign: wp.array2d(dtype=float),
                               tri_normal_f64: wp.array2d(dtype=wp.float64),
                               tri_tangent_f64: wp.array2d(dtype=wp.float64)):
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
    tri_normal_f64[tri, 0] = wp.float64(nnx) * wp.float64(ts0)
    tri_normal_f64[tri, 1] = wp.float64(nny) * wp.float64(ts0)
    tri_normal_f64[tri, 2] = wp.float64(nnz) * wp.float64(ts0)
    q0x = wp.float64(p0x)
    q0y = wp.float64(p0y)
    q0z = wp.float64(p0z)
    dbax = wp.float64(p1x) - q0x
    dbay = wp.float64(p1y) - q0y
    dbaz = wp.float64(p1z) - q0z
    dcax = wp.float64(p2x) - q0x
    dcay = wp.float64(p2y) - q0y
    dcaz = wp.float64(p2z) - q0z
    uv0x = wp.float64(p_uv[i0, 0])
    uv0y = wp.float64(p_uv[i0, 1])
    tbax = wp.float64(p_uv[i1, 0]) - uv0x
    tbay = wp.float64(p_uv[i1, 1]) - uv0y
    tcax = wp.float64(p_uv[i2, 0]) - uv0x
    tcay = wp.float64(p_uv[i2, 1]) - uv0y
    area = tbax * tcay - tbay * tcax
    if area == wp.float64(0.0):
        area = wp.float64(1.0)
    delta = wp.float64(-1.0) / area
    tanx = (dbax * tcay + dcax * dmath.negate(tbay)) * delta
    tany = (dbay * tcay + dcay * dmath.negate(tbay)) * delta
    tanz = (dbaz * tcay + dcaz * dmath.negate(tbay)) * delta
    ltan = wp.sqrt(tanx * tanx + tany * tany + tanz * tanz)
    if ltan > wp.float64(1e-30):
        tanx = tanx / ltan
        tany = tany / ltan
        tanz = tanz / ltan
    ts1 = t_negative_scale_triangle_sign[tt_team, 1]
    tri_tangent_f64[tri, 0] = tanx * wp.float64(ts1)
    tri_tangent_f64[tri, 1] = tany * wp.float64(ts1)
    tri_tangent_f64[tri, 2] = tanz * wp.float64(ts1)


@wp.func
def do_v2t_owner(p: int, mt: int, seg0: int, seg1: int,
                 csr_v2t_order: wp.array(dtype=int),
                 st_v2t_triangle: wp.array(dtype=int),
                 st_v2t_flip_normal: wp.array(dtype=float),
                 st_v2t_flip_tangent: wp.array(dtype=float),
                 tri_normal_f64: wp.array2d(dtype=wp.float64),
                 tri_tangent_f64: wp.array2d(dtype=wp.float64),
                 p_rotations: wp.array2d(dtype=float),
                 p_normal_adjustment_rotations: wp.array2d(dtype=float),
                 t_negative_scale_quaternion: wp.array2d(dtype=float)):
    norx = wp.float64(0.0)
    nory = wp.float64(0.0)
    norz = wp.float64(0.0)
    tanx = wp.float64(0.0)
    tany = wp.float64(0.0)
    tanz = wp.float64(0.0)
    for k in range(seg0, seg1):
        row = csr_v2t_order[k]
        tri = st_v2t_triangle[row]
        fn = wp.float64(st_v2t_flip_normal[row])
        ft = wp.float64(st_v2t_flip_tangent[row])
        norx += tri_normal_f64[tri, 0] * fn
        nory += tri_normal_f64[tri, 1] * fn
        norz += tri_normal_f64[tri, 2] * fn
        tanx += tri_tangent_f64[tri, 0] * ft
        tany += tri_tangent_f64[tri, 1] * ft
        tanz += tri_tangent_f64[tri, 2] * ft
    ln = wp.sqrt(norx * norx + nory * nory + norz * norz)
    lt = wp.sqrt(tanx * tanx + tany * tany + tanz * tanz)
    ok = (ln > wp.float64(1e-6)) and (lt > wp.float64(1e-6))
    if ln > wp.float64(1e-30):
        nnx = norx / ln
        nny = nory / ln
        nnz = norz / ln
    else:
        nnx = norx
        nny = nory
        nnz = norz
    if lt > wp.float64(1e-30):
        ntx = tanx / lt
        nty = tany / lt
        ntz = tanz / lt
    else:
        ntx = tanx
        nty = tany
        ntz = tanz
    d = nnx * ntx + nny * nty + nnz * ntz
    if d == wp.float64(1.0) or d == wp.float64(-1.0):
        ok = False
    bx = nny * ntz - nnz * nty
    by = nnz * ntx - nnx * ntz
    bz = nnx * nty - nny * ntx
    bl = wp.sqrt(bx * bx + by * by + bz * bz)
    if bl > wp.float64(1e-30):
        bx = bx / bl
        by = by / bl
        bz = bz / bl
    rrx, rry, rrz, rrw = dmath.look_rotation(wp.float32(bx), wp.float32(by), wp.float32(bz),
                                             wp.float32(nnx), wp.float32(nny), wp.float32(nnz))
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


@wp.func
def do_output_particle(p: int, mt: int, p_rotations: wp.array2d(dtype=float),
                       p_vertex_to_transform_rotations: wp.array2d(dtype=float),
                       t_negative_scale_quaternion: wp.array2d(dtype=float),
                       p_out_rotations: wp.array2d(dtype=float)):
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
