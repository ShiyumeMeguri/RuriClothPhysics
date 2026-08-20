import warp as wp

from ..cloth_kernel import defs as _defs

wp.set_module_options({"fuse_fp": False})

DEPTH_MASS = wp.constant(float(_defs.DEPTH_MASS))
FRICTION_MASS = wp.constant(float(_defs.FRICTION_MASS))
SELF_COLLISION_FIXED_MASS = wp.constant(float(_defs.SELF_COLLISION_FIXED_MASS))
SELF_COLLISION_FRICTION_MASS = wp.constant(float(_defs.SELF_COLLISION_FRICTION_MASS))
SELF_COLLISION_CLOTH_MASS = wp.constant(float(_defs.SELF_COLLISION_CLOTH_MASS))
PI = wp.constant(3.14159265358979)


@wp.func
def fsign(x: float):
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


@wp.func
def saturate(x: float):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@wp.func
def clamp1(x: float):
    if x < -1.0:
        return -1.0
    if x > 1.0:
        return 1.0
    return x


@wp.func
def lerp(a: float, b: float, t: float):
    return a + (b - a) * t


@wp.func
def fmin2(a: float, b: float):
    if a < b:
        return a
    return b


@wp.func
def fmax2(a: float, b: float):
    if a > b:
        return a
    return b


@wp.func
def dot3(ax: float, ay: float, az: float, bx: float, by: float, bz: float):
    return ax * bx + ay * by + az * bz


@wp.func
def cross3(ax: float, ay: float, az: float, bx: float, by: float, bz: float):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


@wp.func
def length3(x: float, y: float, z: float):
    return wp.sqrt(x * x + y * y + z * z)


@wp.func
def normalize3(x: float, y: float, z: float):
    l = length3(x, y, z)
    safe = l
    if l <= 1.0e-30:
        safe = 1.0
    return x / safe, y / safe, z / safe


@wp.func
def normalize3_fb(x: float, y: float, z: float, fx: float, fy: float, fz: float):
    l = length3(x, y, z)
    if l > 1.0e-30:
        return x / l, y / l, z / l
    return fx, fy, fz


@wp.func
def quat_mul(ax: float, ay: float, az: float, aw: float,
             bx: float, by: float, bz: float, bw: float):
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


@wp.func
def quat_inverse(x: float, y: float, z: float, w: float):
    return -x, -y, -z, w


@wp.func
def quat_normalize(x: float, y: float, z: float, w: float):
    l = wp.sqrt(x * x + y * y + z * z + w * w)
    if l > 1.0e-30:
        return x / l, y / l, z / l, w / l
    return 0.0, 0.0, 0.0, 1.0


@wp.func
def quat_rotate(qx: float, qy: float, qz: float, qw: float,
                vx: float, vy: float, vz: float):
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx))


@wp.func
def quat_slerp(ax: float, ay: float, az: float, aw: float,
               bx: float, by: float, bz: float, bw: float, t: float):
    d = ax * bx + ay * by + az * bz + aw * bw
    cx = bx
    cy = by
    cz = bz
    cw = bw
    if d < 0.0:
        cx = -bx
        cy = -by
        cz = -bz
        cw = -bw
        d = -d
    d = clamp1(d)
    w1 = 1.0 - t
    w2 = t
    if d < 0.9995:
        angle = wp.acos(d)
        sin_angle = wp.sin(angle)
        safe_sin = sin_angle
        if sin_angle <= 1.0e-30:
            safe_sin = 1.0
        w1 = wp.sin(angle * (1.0 - t)) / safe_sin
        w2 = wp.sin(angle * t) / safe_sin
    ox = ax * w1 + cx * w2
    oy = ay * w1 + cy * w2
    oz = az * w1 + cz * w2
    ow = aw * w1 + cw * w2
    return quat_normalize(ox, oy, oz, ow)


@wp.func
def matrix3_to_quat(m00: float, m10: float, m20: float,
                    m01: float, m11: float, m21: float,
                    m02: float, m12: float, m22: float):
    trace = m00 + m11 + m22
    qx = 0.0
    qy = 0.0
    qz = 0.0
    qw = 1.0
    if trace > 0.0:
        s0 = wp.sqrt(fmax2(trace + 1.0, 0.0)) * 2.0
        s0s = s0
        if s0 <= 1.0e-30:
            s0s = 1.0
        qx = (m21 - m12) / s0s
        qy = (m02 - m20) / s0s
        qz = (m10 - m01) / s0s
        qw = 0.25 * s0
    elif m00 >= m11 and m00 >= m22:
        s1 = wp.sqrt(fmax2(1.0 + m00 - m11 - m22, 0.0)) * 2.0
        s1s = s1
        if s1 <= 1.0e-30:
            s1s = 1.0
        qx = 0.25 * s1
        qy = (m01 + m10) / s1s
        qz = (m02 + m20) / s1s
        qw = (m21 - m12) / s1s
    elif m11 >= m22:
        s2 = wp.sqrt(fmax2(1.0 + m11 - m00 - m22, 0.0)) * 2.0
        s2s = s2
        if s2 <= 1.0e-30:
            s2s = 1.0
        qx = (m01 + m10) / s2s
        qy = 0.25 * s2
        qz = (m12 + m21) / s2s
        qw = (m02 - m20) / s2s
    else:
        s3 = wp.sqrt(fmax2(1.0 + m22 - m00 - m11, 0.0)) * 2.0
        s3s = s3
        if s3 <= 1.0e-30:
            s3s = 1.0
        qx = (m02 + m20) / s3s
        qy = (m12 + m21) / s3s
        qz = 0.25 * s3
        qw = (m10 - m01) / s3s
    return quat_normalize(qx, qy, qz, qw)


@wp.func
def look_rotation(fx: float, fy: float, fz: float, ux: float, uy: float, uz: float):
    zx, zy, zz = normalize3_fb(fx, fy, fz, 0.0, 0.0, 1.0)
    ax, ay, az = cross3(ux, uy, uz, zx, zy, zz)
    xx, xy, xz = normalize3_fb(ax, ay, az, 1.0, 0.0, 0.0)
    yx, yy, yz = cross3(zx, zy, zz, xx, xy, xz)
    return matrix3_to_quat(xx, xy, xz, yx, yy, yz, zx, zy, zz)


@wp.func
def to_rotation(nx: float, ny: float, nz: float, tx: float, ty: float, tz: float):
    return look_rotation(tx, ty, tz, nx, ny, nz)


@wp.func
def alt_axis_anti(v1x: float, v1y: float, v1z: float):
    if v1x > v1y and v1x > v1z:
        return -v1z, 0.0, v1x
    return 0.0, v1z, -v1y


@wp.func
def from_to_rotation(fx: float, fy: float, fz: float,
                     tx: float, ty: float, tz: float,
                     t: float, pre_normalized: bool):
    v1x = fx
    v1y = fy
    v1z = fz
    v2x = tx
    v2y = ty
    v2z = tz
    if not pre_normalized:
        v1x, v1y, v1z = normalize3(fx, fy, fz)
        v2x, v2y, v2z = normalize3(tx, ty, tz)

    c = clamp1(v1x * v2x + v1y * v2y + v1z * v2z)
    angle = wp.acos(c)
    axis0 = v1y * v2z - v1z * v2y
    axis1 = v1z * v2x - v1x * v2z
    axis2 = v1x * v2y - v1y * v2x

    anti = wp.abs(1.0 + c) < 1.0e-6
    para = wp.abs(1.0 - c) < 1.0e-6
    if anti:
        axis0, axis1, axis2 = alt_axis_anti(v1x, v1y, v1z)
        angle = 3.14159265358979

    axis_len = wp.sqrt(axis0 * axis0 + axis1 * axis1 + axis2 * axis2)
    naxis0 = 1.0
    naxis1 = 0.0
    naxis2 = 0.0
    if axis_len > 1.0e-30:
        naxis0 = axis0 / axis_len
        naxis1 = axis1 / axis_len
        naxis2 = axis2 / axis_len

    half = (angle * t) * 0.5
    s = wp.sin(half)
    cq = wp.cos(half)
    if para:
        return 0.0, 0.0, 0.0, 1.0
    return naxis0 * s, naxis1 * s, naxis2 * s, cq


@wp.func
def closest_pt_point_segment_ratio(cx: float, cy: float, cz: float,
                                   ax: float, ay: float, az: float,
                                   bx: float, by: float, bz: float):
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    d = abx * abx + aby * aby + abz * abz
    t = 0.0
    if d > 0.0:
        t = ((cx - ax) * abx + (cy - ay) * aby + (cz - az) * abz) / d
    return saturate(t)


@wp.func
def closest_pt_segment_segment(p1x: float, p1y: float, p1z: float,
                               q1x: float, q1y: float, q1z: float,
                               p2x: float, p2y: float, p2z: float,
                               q2x: float, q2y: float, q2z: float):
    d1x = q1x - p1x
    d1y = q1y - p1y
    d1z = q1z - p1z
    d2x = q2x - p2x
    d2y = q2y - p2y
    d2z = q2z - p2z
    rx = p1x - p2x
    ry = p1y - p2y
    rz = p1z - p2z
    a = d1x * d1x + d1y * d1y + d1z * d1z
    e = d2x * d2x + d2y * d2y + d2z * d2z
    f = d2x * rx + d2y * ry + d2z * rz
    c = d1x * rx + d1y * ry + d1z * rz

    both = a <= 1.0e-8 and e <= 1.0e-8
    first = a <= 1.0e-8 and not both
    second = e <= 1.0e-8 and not both and not first

    safe_a = a
    if a <= 1.0e-30:
        safe_a = 1.0
    safe_e = e
    if e <= 1.0e-30:
        safe_e = 1.0

    b = d1x * d2x + d1y * d2y + d1z * d2z
    denom = a * e - b * b
    s = 0.0
    if wp.abs(denom) > 0.0:
        s = saturate((b * f - c * e) / denom)
    t = (b * s + f) / safe_e

    if t < 0.0:
        s = saturate(-c / safe_a)
    elif t > 1.0:
        s = saturate((b - c) / safe_a)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    if first:
        s = 0.0
        t = saturate(f / safe_e)
    if second:
        t = 0.0
        s = saturate(-c / safe_a)
    if both:
        s = 0.0
        t = 0.0

    c1x = p1x + d1x * s
    c1y = p1y + d1y * s
    c1z = p1z + d1z * s
    c2x = p2x + d2x * t
    c2y = p2y + d2y * t
    c2z = p2z + d2z * t
    return s, t, c1x, c1y, c1z, c2x, c2y, c2z


@wp.func
def closest_pt_point_triangle(px: float, py: float, pz: float,
                              ax: float, ay: float, az: float,
                              bx: float, by: float, bz: float,
                              cx: float, cy: float, cz: float):
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    acx = cx - ax
    acy = cy - ay
    acz = cz - az
    apx = px - ax
    apy = py - ay
    apz = pz - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    bpx = px - bx
    bpy = py - by
    bpz = pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    cpx = px - cx
    cpy = py - cy
    cpz = pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    u = 1.0
    v = 0.0
    w = 0.0
    if d1 <= 0.0 and d2 <= 0.0:
        u = 1.0
        v = 0.0
        w = 0.0
    elif d3 >= 0.0 and d4 <= d3:
        u = 0.0
        v = 1.0
        w = 0.0
    elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        den = d1 - d3
        if not (wp.abs(den) > 0.0):
            den = 1.0
        vab = d1 / den
        u = 1.0 - vab
        v = vab
        w = 0.0
    elif d6 >= 0.0 and d5 <= d6:
        u = 0.0
        v = 0.0
        w = 1.0
    elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        den = d2 - d6
        if not (wp.abs(den) > 0.0):
            den = 1.0
        wac = d2 / den
        u = 1.0 - wac
        v = 0.0
        w = wac
    elif va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        g = (d4 - d3) + (d5 - d6)
        if not (wp.abs(g) > 0.0):
            g = 1.0
        wbc = (d4 - d3) / g
        u = 0.0
        v = 1.0 - wbc
        w = wbc
    else:
        h = va + vb + vc
        if not (wp.abs(h) > 0.0):
            h = 1.0
        vf = vb / h
        wf = vc / h
        u = 1.0 - vf - wf
        v = vf
        w = wf
    ptx = ax * u + bx * v + cx * w
    pty = ay * u + by * v + cy * w
    ptz = az * u + bz * v + cz * w
    return ptx, pty, ptz, u, v, w


@wp.func
def quat_to_normal(x: float, y: float, z: float, w: float):
    return quat_rotate(x, y, z, w, 0.0, 1.0, 0.0)


@wp.func
def quat_to_tangent(x: float, y: float, z: float, w: float):
    return quat_rotate(x, y, z, w, 0.0, 0.0, 1.0)


@wp.func
def quat_angle(ax: float, ay: float, az: float, aw: float,
               bx: float, by: float, bz: float, bw: float):
    d = wp.abs(ax * bx + ay * by + az * bz + aw * bw)
    result = 0.0
    if d < 0.9999:
        ang = wp.acos(clamp1(d)) * 2.0
        if ang > PI:
            ang = PI * 2.0 - ang
        result = ang
    return result


@wp.func
def quat_to_angle_axis(qx: float, qy: float, qz: float, qw: float):
    w = clamp1(qw)
    a = 0.0
    if wp.abs(w) < 0.9999:
        a = wp.acos(w)
    angle = 2.0 * a
    s = wp.sin(a)
    if wp.abs(s) > 1.0e-6:
        return angle, qx / s, qy / s, qz / s
    return angle, 0.0, 0.0, 0.0


@wp.func
def transform_point(m: wp.mat44d, px: float, py: float, pz: float):
    dx = wp.float64(px)
    dy = wp.float64(py)
    dz = wp.float64(pz)
    x = m[0, 0] * dx + m[0, 1] * dy + m[0, 2] * dz + m[0, 3]
    y = m[1, 0] * dx + m[1, 1] * dy + m[1, 2] * dz + m[1, 3]
    z = m[2, 0] * dx + m[2, 1] * dy + m[2, 2] * dz + m[2, 3]
    return wp.float32(x), wp.float32(y), wp.float32(z)


@wp.func
def transform_vector(m: wp.mat44d, vx: float, vy: float, vz: float):
    dx = wp.float64(vx)
    dy = wp.float64(vy)
    dz = wp.float64(vz)
    x = m[0, 0] * dx + m[0, 1] * dy + m[0, 2] * dz
    y = m[1, 0] * dx + m[1, 1] * dy + m[1, 2] * dz
    z = m[2, 0] * dx + m[2, 1] * dy + m[2, 2] * dz
    return wp.float32(x), wp.float32(y), wp.float32(z)


@wp.func
def transform_rotation(m: wp.mat44d, qx: float, qy: float, qz: float, qw: float,
                       flip_normal: float, flip_tangent: float):
    nx, ny, nz = quat_to_normal(qx, qy, qz, qw)
    tx, ty, tz = quat_to_tangent(qx, qy, qz, qw)
    wnx, wny, wnz = transform_vector(m, nx, ny, nz)
    wnx = wnx * flip_normal
    wny = wny * flip_normal
    wnz = wnz * flip_normal
    wtx, wty, wtz = transform_vector(m, tx, ty, tz)
    wtx = wtx * flip_tangent
    wty = wty * flip_tangent
    wtz = wtz * flip_tangent
    return look_rotation(wtx, wty, wtz, wnx, wny, wnz)


@wp.func
def clamp_angle_vector(dx: float, dy: float, dz: float,
                       bx: float, by: float, bz: float, max_angle: float):
    v1x, v1y, v1z = normalize3(dx, dy, dz)
    v2x, v2y, v2z = normalize3(bx, by, bz)
    c = clamp1(v1x * v2x + v1y * v2y + v1z * v2z)
    angle = wp.acos(c)
    need = angle > max_angle

    safe_angle = angle
    if angle <= 1.0e-30:
        safe_angle = 1.0
    tval = (angle - max_angle) / safe_angle

    axis0 = v1y * v2z - v1z * v2y
    axis1 = v1z * v2x - v1x * v2z
    axis2 = v1x * v2y - v1y * v2x
    anti = wp.abs(1.0 + c) < 1.0e-6
    rot_angle = angle
    if anti:
        axis0, axis1, axis2 = alt_axis_anti(v1x, v1y, v1z)
        rot_angle = PI
    para = wp.abs(1.0 - c) < 1.0e-6
    need = need and not para

    axis_len = wp.sqrt(axis0 * axis0 + axis1 * axis1 + axis2 * axis2)
    naxis0 = 1.0
    naxis1 = 0.0
    naxis2 = 0.0
    if axis_len > 1.0e-30:
        naxis0 = axis0 / axis_len
        naxis1 = axis1 / axis_len
        naxis2 = axis2 / axis_len

    half = (rot_angle * tval) * 0.5
    s = wp.sin(half)
    cq = wp.cos(half)
    rx, ry, rz = quat_rotate(naxis0 * s, naxis1 * s, naxis2 * s, cq, dx, dy, dz)
    if need:
        return rx, ry, rz
    return dx, dy, dz


@wp.func
def euler_yx(angle_x: float, angle_y: float):
    hx = angle_x * 0.5
    hy = angle_y * 0.5
    sx = wp.sin(hx)
    cx = wp.cos(hx)
    sy = wp.sin(hy)
    cy = wp.cos(hy)
    return cy * sx, sy * cx, -sy * sx, cy * cx


@wp.func
def axis_quaternion(ax: float, ay: float, az: float):
    angle_y = wp.atan2(ax, az)
    flat_len = wp.sqrt(ax * ax + az * az)
    angle_x = wp.atan2(-ay, flat_len)
    return euler_yx(angle_x, angle_y)


@wp.func
def triangle_normal(p0x: float, p0y: float, p0z: float,
                    p1x: float, p1y: float, p1z: float,
                    p2x: float, p2y: float, p2z: float):
    cx, cy, cz = cross3(p1x - p0x, p1y - p0y, p1z - p0z,
                        p2x - p0x, p2y - p0y, p2z - p0z)
    return normalize3_fb(cx, cy, cz, 0.0, 1.0, 0.0)


@wp.func
def clamp_vector(vx: float, vy: float, vz: float, max_length: float):
    l = length3(vx, vy, vz)
    m = fmax2(max_length, 0.0)
    over = l > m and l > 1.0e-9
    scale = 1.0
    if over and l > 1.0e-30:
        scale = max_length / l
    return vx * scale, vy * scale, vz * scale


@wp.func
def clamp_distance(ox: float, oy: float, oz: float,
                   tx: float, ty: float, tz: float, max_length: float):
    vx = tx - ox
    vy = ty - oy
    vz = tz - oz
    l = length3(vx, vy, vz)
    t = 1.0
    if l > max_length:
        t = 0.0
        if l > 1.0e-30:
            t = max_length / l
    return ox + vx * t, oy + vy * t, oz + vz * t


@wp.func
def angle_between(v1x: float, v1y: float, v1z: float,
                  v2x: float, v2y: float, v2z: float):
    l1 = length3(v1x, v1y, v1z)
    l2 = length3(v2x, v2y, v2z)
    denom = 1.0
    if l1 * l2 > 0.0:
        denom = l1 * l2
    c = clamp1((v1x * v2x + v1y * v2y + v1z * v2z) / denom)
    return wp.acos(c)


@wp.func
def intersect_point_plane_dist(px: float, py: float, pz: float,
                               dx: float, dy: float, dz: float,
                               qx: float, qy: float, qz: float):
    vx = qx - px
    vy = qy - py
    vz = qz - pz
    g = vx * dx + vy * dy + vz * dz
    gvx = g * dx
    gvy = g * dy
    gvz = g * dz
    l = length3(gvx, gvy, gvz)
    inside = (dx * vx + dy * vy + dz * vz) < 0.0
    if inside:
        return -l, qx - gvx, qy - gvy, qz - gvz
    return l, qx, qy, qz


@wp.func
def mod289(x: float):
    return x - wp.floor(x * (1.0 / 289.0)) * 289.0


@wp.func
def permute(x: float):
    return mod289((x * 34.0 + 1.0) * x)


@wp.func
def fade(t: float):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@wp.func
def cnoise_corner(ixc: float, iyc: float, fxc: float, fyc: float):
    ic = permute(permute(mod289(ixc)) + mod289(iyc))
    gx = wp.mod(ic * (1.0 / 41.0), 1.0) * 2.0 - 1.0
    gy = wp.abs(gx) - 0.5
    tx = wp.floor(gx + 0.5)
    gx = gx - tx
    norm = 1.79284291400159 - 0.85373472095314 * (gx * gx + gy * gy)
    gx = gx * norm
    gy = gy * norm
    return gx * fxc + gy * fyc


@wp.func
def cnoise2(px: float, py: float):
    pix = wp.floor(px)
    piy = wp.floor(py)
    pfx = px - pix
    pfy = py - piy

    n0 = cnoise_corner(pix, piy, pfx, pfy)
    n1 = cnoise_corner(pix + 1.0, piy, pfx - 1.0, pfy)
    n2 = cnoise_corner(pix, piy + 1.0, pfx, pfy - 1.0)
    n3 = cnoise_corner(pix + 1.0, piy + 1.0, pfx - 1.0, pfy - 1.0)

    fade_x = fade(pfx)
    fade_y = fade(pfy)
    nx0 = n0 + (n1 - n0) * fade_x
    nx1 = n2 + (n3 - n2) * fade_x
    nxy = nx0 + (nx1 - nx0) * fade_y
    return 2.3 * nxy


@wp.func
def calc_mass(depth: float):
    a = 1.0 - depth
    return 1.0 + a * a * DEPTH_MASS


@wp.func
def calc_inverse_mass(friction: float, depth: float):
    a = 1.0 - depth
    mass = 1.0 + friction * FRICTION_MASS + a * a * DEPTH_MASS
    return 1.0 / mass


@wp.func
def calc_inverse_mass_fixed(friction: float, depth: float,
                            fixed_mask: bool, fix_mass: float):
    if fixed_mask:
        return 1.0 / fix_mass
    return calc_inverse_mass(friction, depth)


@wp.func
def calc_self_collision_inverse_mass(friction: float, fixed_mask: bool, cloth_mass: float):
    mass = 1.0 + friction * SELF_COLLISION_FRICTION_MASS
    if fixed_mask:
        mass = SELF_COLLISION_FIXED_MASS
    mass = mass + cloth_mass * SELF_COLLISION_CLOTH_MASS
    return 1.0 / mass


@wp.func
def evaluate_team_lut(luts: wp.array2d(dtype=float), team: int, time: float):
    t = saturate(time)
    f = t * 15.0
    index = int(f)
    frac = f - float(index)
    index2 = index + 1
    if index2 > 15:
        index2 = 15
    a = luts[team, index]
    b = luts[team, index2]
    return a + (b - a) * frac


@wp.func
def evaluate_team_lut_clamp01(luts: wp.array2d(dtype=float), team: int, time: float):
    return saturate(evaluate_team_lut(luts, team, time))


@wp.func
def quat_to_matrix3_f32(x: float, y: float, z: float, w: float):
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)
    return m00, m01, m02, m10, m11, m12, m20, m21, m22


@wp.func
def trs_build_f64(px: float, py: float, pz: float,
                  qx: float, qy: float, qz: float, qw: float,
                  sx: float, sy: float, sz: float):
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = quat_to_matrix3_f32(qx, qy, qz, qw)
    s0 = wp.float64(sx)
    s1 = wp.float64(sy)
    s2 = wp.float64(sz)
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    return wp.mat44d(
        wp.float64(m00) * s0, wp.float64(m01) * s1, wp.float64(m02) * s2, wp.float64(px),
        wp.float64(m10) * s0, wp.float64(m11) * s1, wp.float64(m12) * s2, wp.float64(py),
        wp.float64(m20) * s0, wp.float64(m21) * s1, wp.float64(m22) * s2, wp.float64(pz),
        zero, zero, zero, one)


@wp.func
def trs_inverse_f64(px: float, py: float, pz: float,
                    qx: float, qy: float, qz: float, qw: float,
                    sx: float, sy: float, sz: float):
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = quat_to_matrix3_f32(qx, qy, qz, qw)
    inv0 = wp.float64(1.0) / wp.float64(sx)
    inv1 = wp.float64(1.0) / wp.float64(sy)
    inv2 = wp.float64(1.0) / wp.float64(sz)
    r00 = wp.float64(m00)
    r01 = wp.float64(m01)
    r02 = wp.float64(m02)
    r10 = wp.float64(m10)
    r11 = wp.float64(m11)
    r12 = wp.float64(m12)
    r20 = wp.float64(m20)
    r21 = wp.float64(m21)
    r22 = wp.float64(m22)
    tx = wp.float64(px)
    ty = wp.float64(py)
    tz = wp.float64(pz)
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    return wp.mat44d(
        inv0 * r00, inv0 * r10, inv0 * r20, -inv0 * (r00 * tx + r10 * ty + r20 * tz),
        inv1 * r01, inv1 * r11, inv1 * r21, -inv1 * (r01 * tx + r11 * ty + r21 * tz),
        inv2 * r02, inv2 * r12, inv2 * r22, -inv2 * (r02 * tx + r12 * ty + r22 * tz),
        zero, zero, zero, one)


@wp.func
def mat4_mul_f64(a: wp.mat44d, b: wp.mat44d):
    return a * b
