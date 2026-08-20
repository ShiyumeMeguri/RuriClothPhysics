import warp as wp

wp.set_module_options({"fuse_fp": False})


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
