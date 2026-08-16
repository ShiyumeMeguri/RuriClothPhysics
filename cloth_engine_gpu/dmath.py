"""Device math for the GPU cloth engine.

Every function mirrors the same-named routine in ``cloth_kernel/math.py`` with
identical semantics and branch structure, ported to per-thread scalar / tuple form
for numba.cuda. Vectors are ``(x, y, z)`` tuples, quaternions ``(x, y, z, w)``.

f32 discipline (red line): every floating literal is wrapped in ``float32(...)`` so
sm_75 never silently promotes an f32 expression to f64 (1/32 throughput + drift).
Transcendentals go through ``libdevice`` f32 variants (sinf/cosf/...), matching the
oracle's numpy float32 transcendentals to within a few ULP.
"""

from numba import cuda, float32, int32
from numba.cuda import libdevice


# ---------------------------------------------------------------------------
# scalar helpers
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def fsign(x):
    # numpy np.sign: 0 -> 0, positive -> 1, negative -> -1
    if x > float32(0.0):
        return float32(1.0)
    if x < float32(0.0):
        return float32(-1.0)
    return float32(0.0)


@cuda.jit(device=True)
def saturate(x):
    if x < float32(0.0):
        return float32(0.0)
    if x > float32(1.0):
        return float32(1.0)
    return x


@cuda.jit(device=True)
def clamp1(x):
    if x < float32(-1.0):
        return float32(-1.0)
    if x > float32(1.0):
        return float32(1.0)
    return x


@cuda.jit(device=True)
def lerp(a, b, t):
    return a + (b - a) * t


@cuda.jit(device=True)
def fmin2(a, b):
    return a if a < b else b


@cuda.jit(device=True)
def fmax2(a, b):
    return a if a > b else b


# ---------------------------------------------------------------------------
# vec3
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def dot3(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


@cuda.jit(device=True)
def cross3(ax, ay, az, bx, by, bz):
    return (ay * bz - az * by,
            az * bx - ax * bz,
            ax * by - ay * bx)


@cuda.jit(device=True)
def length3(x, y, z):
    return libdevice.sqrtf(x * x + y * y + z * z)


@cuda.jit(device=True)
def normalize3(x, y, z):
    # math.normalize without fallback: v / max(len, 1) -> unchanged when len<=1e-30
    l = length3(x, y, z)
    safe = l if l > float32(1e-30) else float32(1.0)
    return (x / safe, y / safe, z / safe)


@cuda.jit(device=True)
def normalize3_fb(x, y, z, fx, fy, fz):
    l = length3(x, y, z)
    if l > float32(1e-30):
        return (x / l, y / l, z / l)
    return (fx, fy, fz)


# ---------------------------------------------------------------------------
# quaternion (x, y, z, w)
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def quat_mul(ax, ay, az, aw, bx, by, bz, bw):
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


@cuda.jit(device=True)
def quat_inverse(x, y, z, w):
    return (-x, -y, -z, w)


@cuda.jit(device=True)
def quat_normalize(x, y, z, w):
    l = libdevice.sqrtf(x * x + y * y + z * z + w * w)
    if l > float32(1e-30):
        return (x / l, y / l, z / l, w / l)
    return (float32(0.0), float32(0.0), float32(0.0), float32(1.0))


@cuda.jit(device=True)
def quat_rotate(qx, qy, qz, qw, vx, vy, vz):
    tx = float32(2.0) * (qy * vz - qz * vy)
    ty = float32(2.0) * (qz * vx - qx * vz)
    tz = float32(2.0) * (qx * vy - qy * vx)
    return (vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx))


@cuda.jit(device=True)
def quat_to_normal(x, y, z, w):
    # rotate VEC_UP (0,1,0)
    return quat_rotate(x, y, z, w, float32(0.0), float32(1.0), float32(0.0))


@cuda.jit(device=True)
def quat_to_tangent(x, y, z, w):
    # rotate VEC_FORWARD (0,0,1)
    return quat_rotate(x, y, z, w, float32(0.0), float32(0.0), float32(1.0))


@cuda.jit(device=True)
def quat_slerp(ax, ay, az, aw, bx, by, bz, bw, t):
    d = ax * bx + ay * by + az * bz + aw * bw
    if d < float32(0.0):
        bx = -bx
        by = -by
        bz = -bz
        bw = -bw
        d = -d
    d = clamp1(d)
    if d < float32(0.9995):
        angle = libdevice.acosf(d)
        sin_angle = libdevice.sinf(angle)
        safe_sin = sin_angle if sin_angle > float32(1e-30) else float32(1.0)
        w1 = libdevice.sinf(angle * (float32(1.0) - t)) / safe_sin
        w2 = libdevice.sinf(angle * t) / safe_sin
    else:
        w1 = float32(1.0) - t
        w2 = t
    ox = ax * w1 + bx * w2
    oy = ay * w1 + by * w2
    oz = az * w1 + bz * w2
    ow = aw * w1 + bw * w2
    return quat_normalize(ox, oy, oz, ow)


@cuda.jit(device=True)
def quat_angle(ax, ay, az, aw, bx, by, bz, bw):
    d = libdevice.fabsf(ax * bx + ay * by + az * bz + aw * bw)
    if d < float32(0.9999):
        ang = libdevice.acosf(clamp1(d)) * float32(2.0)
        if ang > float32(3.14159265358979):
            ang = float32(3.14159265358979) * float32(2.0) - ang
        return ang
    return float32(0.0)


@cuda.jit(device=True)
def quat_to_angle_axis(qx, qy, qz, qw):
    w = clamp1(qw)
    if libdevice.fabsf(w) < float32(0.9999):
        a = libdevice.acosf(w)
    else:
        a = float32(0.0)
    angle = float32(2.0) * a
    s = libdevice.sinf(a)
    if libdevice.fabsf(s) > float32(1e-6):
        return (angle, qx / s, qy / s, qz / s)
    return (angle, float32(0.0), float32(0.0), float32(0.0))


@cuda.jit(device=True)
def matrix3_to_quat(m00, m10, m20, m01, m11, m21, m02, m12, m22):
    trace = m00 + m11 + m22
    if trace > float32(0.0):
        s0 = libdevice.sqrtf(fmax2(trace + float32(1.0), float32(0.0))) * float32(2.0)
        s0s = s0 if s0 > float32(1e-30) else float32(1.0)
        qx = (m21 - m12) / s0s
        qy = (m02 - m20) / s0s
        qz = (m10 - m01) / s0s
        qw = float32(0.25) * s0
    elif m00 >= m11 and m00 >= m22:
        s1 = libdevice.sqrtf(fmax2(float32(1.0) + m00 - m11 - m22, float32(0.0))) * float32(2.0)
        s1s = s1 if s1 > float32(1e-30) else float32(1.0)
        qx = float32(0.25) * s1
        qy = (m01 + m10) / s1s
        qz = (m02 + m20) / s1s
        qw = (m21 - m12) / s1s
    elif m11 >= m22:
        s2 = libdevice.sqrtf(fmax2(float32(1.0) + m11 - m00 - m22, float32(0.0))) * float32(2.0)
        s2s = s2 if s2 > float32(1e-30) else float32(1.0)
        qx = (m01 + m10) / s2s
        qy = float32(0.25) * s2
        qz = (m12 + m21) / s2s
        qw = (m02 - m20) / s2s
    else:
        s3 = libdevice.sqrtf(fmax2(float32(1.0) + m22 - m00 - m11, float32(0.0))) * float32(2.0)
        s3s = s3 if s3 > float32(1e-30) else float32(1.0)
        qx = (m02 + m20) / s3s
        qy = (m12 + m21) / s3s
        qz = float32(0.25) * s3
        qw = (m10 - m01) / s3s
    return quat_normalize(qx, qy, qz, qw)


@cuda.jit(device=True)
def look_rotation(fx, fy, fz, ux, uy, uz):
    # z = normalize(forward, fallback FORWARD)
    zx, zy, zz = normalize3_fb(fx, fy, fz, float32(0.0), float32(0.0), float32(1.0))
    xx, xy, xz = cross3(ux, uy, uz, zx, zy, zz)
    xx, xy, xz = normalize3_fb(xx, xy, xz, float32(1.0), float32(0.0), float32(0.0))
    yx, yy, yz = cross3(zx, zy, zz, xx, xy, xz)
    return matrix3_to_quat(xx, xy, xz, yx, yy, yz, zx, zy, zz)


@cuda.jit(device=True)
def to_rotation(nx, ny, nz, tx, ty, tz):
    # to_rotation(normal, tangent) = look_rotation(tangent, normal)
    return look_rotation(tx, ty, tz, nx, ny, nz)


@cuda.jit(device=True)
def transform_point(m, px, py, pz):
    # (m[:3,:3] @ p + m[:3,3]); m is a f64 (>=3x4) matrix view (mirrors math.transform_points f64 path)
    x = m[0, 0] * px + m[0, 1] * py + m[0, 2] * pz + m[0, 3]
    y = m[1, 0] * px + m[1, 1] * py + m[1, 2] * pz + m[1, 3]
    z = m[2, 0] * px + m[2, 1] * py + m[2, 2] * pz + m[2, 3]
    return (float32(x), float32(y), float32(z))


@cuda.jit(device=True)
def transform_vector(m, vx, vy, vz):
    x = m[0, 0] * vx + m[0, 1] * vy + m[0, 2] * vz
    y = m[1, 0] * vx + m[1, 1] * vy + m[1, 2] * vz
    z = m[2, 0] * vx + m[2, 1] * vy + m[2, 2] * vz
    return (float32(x), float32(y), float32(z))


@cuda.jit(device=True)
def transform_rotation(m, qx, qy, qz, qw, flip_normal, flip_tangent):
    # mirrors math.transform_rotations(rot, matrix, flip): rotate normal/tangent, flip, look_rotation
    nx, ny, nz = quat_to_normal(qx, qy, qz, qw)
    tx, ty, tz = quat_to_tangent(qx, qy, qz, qw)
    nx, ny, nz = transform_vector(m, nx, ny, nz)
    nx = nx * flip_normal
    ny = ny * flip_normal
    nz = nz * flip_normal
    tx, ty, tz = transform_vector(m, tx, ty, tz)
    tx = tx * flip_tangent
    ty = ty * flip_tangent
    tz = tz * flip_tangent
    return look_rotation(tx, ty, tz, nx, ny, nz)


@cuda.jit(device=True)
def _alt_axis_anti(v1x, v1y, v1z):
    # anti-parallel alternate axis, mirrors math.from_to_rotation
    if v1x > v1y and v1x > v1z:
        # alt_up = cross-ish (0,0,1)-based branch: (-v1z, 0, v1x)
        return (-v1z, float32(0.0), v1x)
    return (float32(0.0), v1z, -v1y)


@cuda.jit(device=True)
def from_to_rotation(fx, fy, fz, tx, ty, tz, t, pre_normalized):
    if pre_normalized:
        v1x, v1y, v1z = fx, fy, fz
        v2x, v2y, v2z = tx, ty, tz
    else:
        v1x, v1y, v1z = normalize3(fx, fy, fz)
        v2x, v2y, v2z = normalize3(tx, ty, tz)

    c = clamp1(v1x * v2x + v1y * v2y + v1z * v2z)
    angle = libdevice.acosf(c)
    axis0 = v1y * v2z - v1z * v2y
    axis1 = v1z * v2x - v1x * v2z
    axis2 = v1x * v2y - v1y * v2x

    anti = libdevice.fabsf(float32(1.0) + c) < float32(1e-6)
    para = libdevice.fabsf(float32(1.0) - c) < float32(1e-6)
    if anti:
        axis0, axis1, axis2 = _alt_axis_anti(v1x, v1y, v1z)
        angle = float32(3.14159265358979)

    axis_len = libdevice.sqrtf(axis0 * axis0 + axis1 * axis1 + axis2 * axis2)
    if axis_len > float32(1e-30):
        naxis0 = axis0 / axis_len
        naxis1 = axis1 / axis_len
        naxis2 = axis2 / axis_len
    else:
        naxis0 = float32(1.0)
        naxis1 = float32(0.0)
        naxis2 = float32(0.0)

    half = (angle * t) * float32(0.5)
    s = libdevice.sinf(half)
    cq = libdevice.cosf(half)
    if para:
        return (float32(0.0), float32(0.0), float32(0.0), float32(1.0))
    return (naxis0 * s, naxis1 * s, naxis2 * s, cq)


@cuda.jit(device=True)
def clamp_angle_vector(dx, dy, dz, bx, by, bz, max_angle):
    # returns (ox, oy, oz, need)
    v1x, v1y, v1z = normalize3(dx, dy, dz)
    v2x, v2y, v2z = normalize3(bx, by, bz)
    c = clamp1(v1x * v2x + v1y * v2y + v1z * v2z)
    angle = libdevice.acosf(c)
    need = angle > max_angle

    safe_angle = angle if angle > float32(1e-30) else float32(1.0)
    tval = (angle - max_angle) / safe_angle

    axis0 = v1y * v2z - v1z * v2y
    axis1 = v1z * v2x - v1x * v2z
    axis2 = v1x * v2y - v1y * v2x
    anti = libdevice.fabsf(float32(1.0) + c) < float32(1e-6)
    if anti:
        axis0, axis1, axis2 = _alt_axis_anti(v1x, v1y, v1z)
        rot_angle = float32(3.14159265358979)
    else:
        rot_angle = angle
    para = libdevice.fabsf(float32(1.0) - c) < float32(1e-6)
    need = need and (not para)

    axis_len = libdevice.sqrtf(axis0 * axis0 + axis1 * axis1 + axis2 * axis2)
    if axis_len > float32(1e-30):
        naxis0 = axis0 / axis_len
        naxis1 = axis1 / axis_len
        naxis2 = axis2 / axis_len
    else:
        naxis0 = float32(1.0)
        naxis1 = float32(0.0)
        naxis2 = float32(0.0)

    half = (rot_angle * tval) * float32(0.5)
    s = libdevice.sinf(half)
    cq = libdevice.cosf(half)
    rx, ry, rz = quat_rotate(naxis0 * s, naxis1 * s, naxis2 * s, cq, dx, dy, dz)
    if need:
        return (rx, ry, rz, True)
    return (dx, dy, dz, False)


@cuda.jit(device=True)
def euler_yx(angle_x, angle_y):
    hx = angle_x * float32(0.5)
    hy = angle_y * float32(0.5)
    sx = libdevice.sinf(hx)
    cx = libdevice.cosf(hx)
    sy = libdevice.sinf(hy)
    cy = libdevice.cosf(hy)
    return (cy * sx, sy * cx, -sy * sx, cy * cx)


@cuda.jit(device=True)
def axis_quaternion(ax, ay, az):
    angle_y = libdevice.atan2f(ax, az)
    flat_len = libdevice.sqrtf(ax * ax + az * az)
    angle_x = libdevice.atan2f(-ay, flat_len)
    return euler_yx(angle_x, angle_y)


@cuda.jit(device=True)
def triangle_normal(p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z):
    cx, cy, cz = cross3(p1x - p0x, p1y - p0y, p1z - p0z,
                        p2x - p0x, p2y - p0y, p2z - p0z)
    return normalize3_fb(cx, cy, cz, float32(0.0), float32(1.0), float32(0.0))


# ---------------------------------------------------------------------------
# clamp / project helpers
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def clamp_vector(vx, vy, vz, max_length):
    l = length3(vx, vy, vz)
    m = fmax2(max_length, float32(0.0))
    over = (l > m) and (l > float32(1e-9))
    if over and l > float32(1e-30):
        scale = max_length / l
    else:
        scale = float32(1.0)
    return (vx * scale, vy * scale, vz * scale)


@cuda.jit(device=True)
def clamp_distance(ox, oy, oz, tx, ty, tz, max_length):
    vx = tx - ox
    vy = ty - oy
    vz = tz - oz
    l = length3(vx, vy, vz)
    if l > max_length:
        t = (max_length / l) if l > float32(1e-30) else float32(0.0)
    else:
        t = float32(1.0)
    return (ox + vx * t, oy + vy * t, oz + vz * t)


@cuda.jit(device=True)
def project_on_plane(vx, vy, vz, nx, ny, nz):
    d = vx * nx + vy * ny + vz * nz
    return (vx - d * nx, vy - d * ny, vz - d * nz)


@cuda.jit(device=True)
def angle_between(v1x, v1y, v1z, v2x, v2y, v2z):
    l1 = length3(v1x, v1y, v1z)
    l2 = length3(v2x, v2y, v2z)
    denom = l1 * l2 if (l1 * l2) > float32(0.0) else float32(1.0)
    c = clamp1((v1x * v2x + v1y * v2y + v1z * v2z) / denom)
    return libdevice.acosf(c)


@cuda.jit(device=True)
def intersect_point_plane_dist(px, py, pz, dx, dy, dz, qx, qy, qz):
    # plane_pos p, plane_dir d, pos q -> (dist, ox, oy, oz)
    vx = qx - px
    vy = qy - py
    vz = qz - pz
    g = vx * dx + vy * dy + vz * dz
    gvx = g * dx
    gvy = g * dy
    gvz = g * dz
    l = length3(gvx, gvy, gvz)
    inside = (dx * vx + dy * vy + dz * vz) < float32(0.0)
    if inside:
        return (-l, qx - gvx, qy - gvy, qz - gvz)
    return (l, qx, qy, qz)


@cuda.jit(device=True)
def closest_pt_point_segment_ratio(cx, cy, cz, ax, ay, az, bx, by, bz):
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    d = abx * abx + aby * aby + abz * abz
    if d > float32(0.0):
        t = ((cx - ax) * abx + (cy - ay) * aby + (cz - az) * abz) / d
    else:
        t = float32(0.0)
    return saturate(t)


@cuda.jit(device=True)
def closest_pt_segment_segment(p1x, p1y, p1z, q1x, q1y, q1z,
                               p2x, p2y, p2z, q2x, q2y, q2z):
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

    both = (a <= float32(1e-8)) and (e <= float32(1e-8))
    first = (a <= float32(1e-8)) and (not both)
    second = (e <= float32(1e-8)) and (not both) and (not first)

    safe_a = a if a > float32(1e-30) else float32(1.0)
    safe_e = e if e > float32(1e-30) else float32(1.0)

    b = d1x * d2x + d1y * d2y + d1z * d2z
    denom = a * e - b * b
    if libdevice.fabsf(denom) > float32(0.0):
        s = saturate((b * f - c * e) / denom)
    else:
        s = float32(0.0)
    t = (b * s + f) / safe_e

    if t < float32(0.0):
        s = saturate(-c / safe_a)
    elif t > float32(1.0):
        s = saturate((b - c) / safe_a)
    if t < float32(0.0):
        t = float32(0.0)
    elif t > float32(1.0):
        t = float32(1.0)

    if first:
        s = float32(0.0)
        t = saturate(f / safe_e)
    if second:
        t = float32(0.0)
        s = saturate(-c / safe_a)
    if both:
        s = float32(0.0)
        t = float32(0.0)

    c1x = p1x + d1x * s
    c1y = p1y + d1y * s
    c1z = p1z + d1z * s
    c2x = p2x + d2x * t
    c2y = p2y + d2y * t
    c2z = p2z + d2z * t
    return (s, t, c1x, c1y, c1z, c2x, c2y, c2z)


# ---------------------------------------------------------------------------
# classic perlin noise (cnoise2) - matches math.cnoise2
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def _mod289(x):
    return x - libdevice.floorf(x * (float32(1.0) / float32(289.0))) * float32(289.0)


@cuda.jit(device=True)
def _permute(x):
    return _mod289((x * float32(34.0) + float32(1.0)) * x)


@cuda.jit(device=True)
def _fade(t):
    return t * t * t * (t * (t * float32(6.0) - float32(15.0)) + float32(10.0))


@cuda.jit(device=True)
def _cnoise_corner(ixc, iyc, fxc, fyc):
    ic = _permute(_permute(_mod289(ixc)) + _mod289(iyc))
    gx = libdevice.fmodf(ic * (float32(1.0) / float32(41.0)), float32(1.0)) * float32(2.0) - float32(1.0)
    gy = libdevice.fabsf(gx) - float32(0.5)
    tx = libdevice.floorf(gx + float32(0.5))
    gx = gx - tx
    norm = float32(1.79284291400159) - float32(0.85373472095314) * (gx * gx + gy * gy)
    gx = gx * norm
    gy = gy * norm
    return gx * fxc + gy * fyc


@cuda.jit(device=True)
def cnoise2(px, py):
    pix = libdevice.floorf(px)
    piy = libdevice.floorf(py)
    pfx = px - pix
    pfy = py - piy

    n0 = _cnoise_corner(pix, piy, pfx, pfy)
    n1 = _cnoise_corner(pix + float32(1.0), piy, pfx - float32(1.0), pfy)
    n2 = _cnoise_corner(pix, piy + float32(1.0), pfx, pfy - float32(1.0))
    n3 = _cnoise_corner(pix + float32(1.0), piy + float32(1.0), pfx - float32(1.0), pfy - float32(1.0))

    fade_x = _fade(pfx)
    fade_y = _fade(pfy)
    nx0 = n0 + (n1 - n0) * fade_x
    nx1 = n2 + (n3 - n2) * fade_x
    nxy = nx0 + (nx1 - nx0) * fade_y
    return float32(2.3) * nxy


# ---------------------------------------------------------------------------
# mass / LUT
# ---------------------------------------------------------------------------

DEPTH_MASS = float32(5.0)
FRICTION_MASS = float32(3.0)


@cuda.jit(device=True)
def calc_mass(depth):
    a = float32(1.0) - depth
    return float32(1.0) + a * a * DEPTH_MASS


@cuda.jit(device=True)
def calc_inverse_mass(friction, depth):
    a = float32(1.0) - depth
    mass = float32(1.0) + friction * FRICTION_MASS + a * a * DEPTH_MASS
    return float32(1.0) / mass


@cuda.jit(device=True)
def calc_inverse_mass_fixed(friction, depth, fixed_mask, fix_mass):
    if fixed_mask:
        return float32(1.0) / fix_mass
    return calc_inverse_mass(friction, depth)


@cuda.jit(device=True)
def evaluate_lut(lut, base, time):
    # lut is a flat device array; row starts at `base`, 16 samples
    t = saturate(time)
    f = t * float32(15.0)
    index = int32(f)
    frac = f - float32(index)
    index2 = index + 1
    if index2 > 15:
        index2 = 15
    a = lut[base + index]
    b = lut[base + index2]
    return a + (b - a) * frac


@cuda.jit(device=True)
def evaluate_lut_clamp01(lut, base, time):
    return saturate(evaluate_lut(lut, base, time))


@cuda.jit(device=True)
def evaluate_team_lut(luts, team, time):
    # luts is a (num_teams, 16) device array; mirrors math.evaluate_team_lut
    t = saturate(time)
    f = t * float32(15.0)
    index = int32(f)
    frac = f - float32(index)
    index2 = index + 1
    if index2 > 15:
        index2 = 15
    a = luts[team, index]
    b = luts[team, index2]
    return a + (b - a) * frac


@cuda.jit(device=True)
def evaluate_team_lut_clamp01(luts, team, time):
    return saturate(evaluate_team_lut(luts, team, time))
