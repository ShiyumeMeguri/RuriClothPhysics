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
DEG2RAD = wp.constant(float(math.pi / 180.0))


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
    dist = near_dist if no_contact else -cval
    valid = overlap and (not degenerate) and (not miss)
    if not valid:
        dist = wp.inf
    if (not valid) or no_contact:
        c0x = 0.0
        c0y = 0.0
        c0z = 0.0
        c1x = 0.0
        c1y = 0.0
        c1z = 0.0
    if not valid:
        nx = 0.0
        ny = 0.0
        nz = 0.0
    return (dist, c0x, c0y, c0z, c1x, c1y, c1z, nx, ny, nz)


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
    dist = near_dist if no_contact else -cval
    valid = overlap and (not degenerate) and (not miss)
    if not valid:
        dist = wp.inf
    if (not valid) or no_contact:
        c0x = 0.0
        c0y = 0.0
        c0z = 0.0
        c1x = 0.0
        c1y = 0.0
        c1z = 0.0
    if not valid:
        nx = 0.0
        ny = 0.0
        nz = 0.0
    return (dist, c0x, c0y, c0z, c1x, c1y, c1z, nx, ny, nz)


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
