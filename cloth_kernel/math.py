import numpy as np

from . import defs

VEC_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
VEC_FORWARD = np.array([0.0, 0.0, 1.0], dtype=np.float32)
VEC_RIGHT = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def dot(a, b):
    return np.sum(a * b, axis=-1)


def length(v):
    return np.sqrt(np.sum(v * v, axis=-1))


def normalize(v, fallback=None):
    l = length(v)
    safe = np.where(l > 1e-30, l, 1.0)
    out = v / safe[..., None]
    if fallback is not None:
        out = np.where((l > 1e-30)[..., None], out, fallback)
    return out


def saturate(x):
    return np.clip(x, 0.0, 1.0)


def clamp1(x):
    return np.clip(x, -1.0, 1.0)


def lerp(a, b, t):
    return a + (b - a) * t


def project(v, n):
    return dot(v, n)[..., None] * n


def project_on_plane(v, n):
    return v - dot(v, n)[..., None] * n


def angle_between(v1, v2):
    l1 = length(v1)
    l2 = length(v2)
    denominator = np.where(l1 * l2 > 0.0, l1 * l2, 1.0)
    cosine = clamp1(dot(v1, v2) / denominator)
    return np.arccos(cosine)


def clamp_vector(v, max_length):
    l = length(v)
    over = l > np.maximum(max_length, 0.0)
    over &= l > 1e-9
    scale = np.where(over, np.where(l > 1e-30, np.asarray(max_length, dtype=v.dtype) / np.where(l > 1e-30, l, 1.0), 1.0), 1.0)
    return v * scale[..., None]


def clamp_distance(origin, target, max_length):
    v = target - origin
    l = length(v)
    over = l > max_length
    t = np.where(over, np.where(l > 1e-30, max_length / np.where(l > 1e-30, l, 1.0), 0.0), 1.0)
    return origin + v * t[..., None]


def quat_identity(shape_prefix=()):
    q = np.zeros(shape_prefix + (4,), dtype=np.float32)
    q[..., 3] = 1.0
    return q


def quat_mul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_conjugate(q):
    out = np.array(q, copy=True)
    out[..., :3] = -out[..., :3]
    return out


def quat_inverse(q):
    return quat_conjugate(q)


def quat_normalize(q):
    l = np.sqrt(np.sum(q * q, axis=-1))
    safe = np.where(l > 1e-30, l, 1.0)
    out = q / safe[..., None]
    identity = quat_identity(q.shape[:-1])
    return np.where((l > 1e-30)[..., None], out, identity)


def quat_rotate(q, v):
    qv = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def quat_from_axis_angle(axis, angle):
    half = angle * 0.5
    s = np.sin(half)
    return np.concatenate([axis * s[..., None], np.cos(half)[..., None]], axis=-1)


def quat_slerp(a, b, t):
    d = np.sum(a * b, axis=-1)
    sign = np.where(d < 0.0, -1.0, 1.0)
    b2 = b * sign[..., None]
    d = np.abs(d)
    d = np.clip(d, -1.0, 1.0)

    use_slerp = d < 0.9995
    angle = np.arccos(np.where(use_slerp, d, 0.0))
    sin_angle = np.sin(angle)
    safe_sin = np.where(sin_angle > 1e-30, sin_angle, 1.0)
    t = np.asarray(t, dtype=a.dtype)
    w1 = np.where(use_slerp, np.sin(angle * (1.0 - t)) / safe_sin, 1.0 - t)
    w2 = np.where(use_slerp, np.sin(angle * t) / safe_sin, t)
    out = a * w1[..., None] + b2 * w2[..., None]
    return quat_normalize(out)


def quat_angle(a, b):
    d = np.abs(np.sum(a * b, axis=-1))
    small = d < 0.9999
    ang = np.arccos(clamp1(np.where(small, d, 0.0))) * 2.0
    ang = np.where(ang > np.pi, np.pi * 2.0 - ang, ang)
    return np.where(small, ang, 0.0)


def quat_to_angle_axis(q):
    w = np.clip(q[..., 3], -1.0, 1.0)
    a = np.where(np.abs(w) < 0.9999, np.arccos(w), 0.0)
    angle = 2.0 * a
    s = np.sin(a)
    safe = np.where(np.abs(s) > 1e-6, s, 1.0)
    axis = q[..., :3] / safe[..., None]
    axis = np.where((np.abs(s) > 1e-6)[..., None], axis, 0.0)
    return angle, axis


def from_to_rotation(v_from, v_to, t=1.0, pre_normalized=False):
    if pre_normalized:
        v1 = v_from
        v2 = v_to
    else:
        v1 = normalize(v_from)
        v2 = normalize(v_to)

    c = clamp1(dot(v1, v2))
    angle = np.arccos(c)
    axis = np.cross(v1, v2)

    anti = np.abs(1.0 + c) < 1e-6
    para = np.abs(1.0 - c) < 1e-6

    pick_y = (v1[..., 0] > v1[..., 1]) & (v1[..., 0] > v1[..., 2])
    alt_axis = np.where(pick_y[..., None],
                        np.cross(v1, np.broadcast_to(VEC_UP, v1.shape)),
                        np.cross(v1, np.broadcast_to(VEC_RIGHT, v1.shape)))
    axis = np.where(anti[..., None], alt_axis, axis)
    angle = np.where(anti, np.pi, angle)

    axis = normalize(axis, fallback=np.broadcast_to(VEC_RIGHT, axis.shape))
    q = quat_from_axis_angle(axis, angle * t)
    identity = quat_identity(q.shape[:-1])
    return np.where(para[..., None], identity, q)


def clamp_angle_vector(direction, base_direction, max_angle):
    v1 = normalize(direction)
    v2 = normalize(base_direction)
    c = clamp1(dot(v1, v2))
    angle = np.arccos(c)
    need = angle > max_angle

    safe_angle = np.where(angle > 1e-30, angle, 1.0)
    t = (angle - max_angle) / safe_angle

    axis = np.cross(v1, v2)
    anti = np.abs(1.0 + c) < 1e-6
    pick_y = (v1[..., 0] > v1[..., 1]) & (v1[..., 0] > v1[..., 2])
    alt_axis = np.where(pick_y[..., None],
                        np.cross(v1, np.broadcast_to(VEC_UP, v1.shape)),
                        np.cross(v1, np.broadcast_to(VEC_RIGHT, v1.shape)))
    axis = np.where(anti[..., None], alt_axis, axis)
    rot_angle = np.where(anti, np.pi, angle)
    para = np.abs(1.0 - c) < 1e-6
    need = need & ~para

    axis = normalize(axis, fallback=np.broadcast_to(VEC_RIGHT, axis.shape))
    q = quat_from_axis_angle(axis, rot_angle * t)
    rotated = quat_rotate(q, direction)
    out = np.where(need[..., None], rotated, direction)
    return out, need


def matrix3_to_quat(x_axis, y_axis, z_axis):
    m00, m10, m20 = x_axis[..., 0], x_axis[..., 1], x_axis[..., 2]
    m01, m11, m21 = y_axis[..., 0], y_axis[..., 1], y_axis[..., 2]
    m02, m12, m22 = z_axis[..., 0], z_axis[..., 1], z_axis[..., 2]

    trace = m00 + m11 + m22

    s0 = np.sqrt(np.maximum(trace + 1.0, 0.0)) * 2.0
    s0s = np.where(s0 > 1e-30, s0, 1.0)
    q0 = np.stack([(m21 - m12) / s0s, (m02 - m20) / s0s, (m10 - m01) / s0s, 0.25 * s0], axis=-1)

    s1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0)) * 2.0
    s1s = np.where(s1 > 1e-30, s1, 1.0)
    q1 = np.stack([0.25 * s1, (m01 + m10) / s1s, (m02 + m20) / s1s, (m21 - m12) / s1s], axis=-1)

    s2 = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 0.0)) * 2.0
    s2s = np.where(s2 > 1e-30, s2, 1.0)
    q2 = np.stack([(m01 + m10) / s2s, 0.25 * s2, (m12 + m21) / s2s, (m02 - m20) / s2s], axis=-1)

    s3 = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 0.0)) * 2.0
    s3s = np.where(s3 > 1e-30, s3, 1.0)
    q3 = np.stack([(m02 + m20) / s3s, (m12 + m21) / s3s, 0.25 * s3, (m10 - m01) / s3s], axis=-1)

    cond0 = trace > 0.0
    cond1 = (~cond0) & (m00 >= m11) & (m00 >= m22)
    cond2 = (~cond0) & (~cond1) & (m11 >= m22)

    q = np.where(cond0[..., None], q0, np.where(cond1[..., None], q1, np.where(cond2[..., None], q2, q3)))
    return quat_normalize(q)


def look_rotation(forward, up):
    z = normalize(forward, fallback=np.broadcast_to(VEC_FORWARD, forward.shape))
    x = np.cross(up, z)
    x = normalize(x, fallback=np.broadcast_to(VEC_RIGHT, x.shape))
    y = np.cross(z, x)
    return matrix3_to_quat(x, y, z)


def to_rotation(normal, tangent):
    return look_rotation(tangent, normal)


def quat_to_normal(q):
    return quat_rotate(q, np.broadcast_to(VEC_UP, q[..., :3].shape))


def quat_to_tangent(q):
    return quat_rotate(q, np.broadcast_to(VEC_FORWARD, q[..., :3].shape))


def quat_to_matrix3(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=-1)
    row1 = np.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=-1)
    row2 = np.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def trs(position, quat, scale):
    r = quat_to_matrix3(quat).astype(np.float64) * np.asarray(scale, dtype=np.float64)[..., None, :]
    m = np.zeros(position.shape[:-1] + (4, 4), dtype=np.float64)
    m[..., :3, :3] = r
    m[..., :3, 3] = position
    m[..., 3, 3] = 1.0
    return m


def transform_points(points, matrix):
    return (np.einsum('...ij,...j->...i', matrix[..., :3, :3], points.astype(np.float64))
            + matrix[..., :3, 3]).astype(np.float32)


def transform_vectors(vectors, matrix):
    return np.einsum('...ij,...j->...i', matrix[..., :3, :3], vectors.astype(np.float64)).astype(np.float32)


def transform_rotations(rotations, matrix, flip):
    normals = quat_to_normal(rotations)
    tangents = quat_to_tangent(rotations)
    normals = transform_vectors(normals, matrix) * flip[..., 1:2]
    tangents = transform_vectors(tangents, matrix) * flip[..., 2:3]
    return look_rotation(tangents, normals)


def shift_points(points, center, shift_vector, shift_rotation):
    local = points - center
    local = quat_rotate(shift_rotation, local)
    return local + center + shift_vector


def euler_yx(angle_x, angle_y):
    hx = angle_x * 0.5
    hy = angle_y * 0.5
    sx, cx = np.sin(hx), np.cos(hx)
    sy, cy = np.sin(hy), np.cos(hy)
    return np.stack([
        cy * sx,
        sy * cx,
        -sy * sx,
        cy * cx,
    ], axis=-1)


def axis_to_euler(axis):
    angle_y = np.arctan2(axis[..., 0], axis[..., 2])
    flat = axis.copy()
    flat[..., 1] = 0.0
    angle_x = np.arctan2(-axis[..., 1], length(flat))
    return angle_x, angle_y


def axis_quaternion(direction):
    angle_x, angle_y = axis_to_euler(direction)
    return euler_yx(angle_x, angle_y)


def triangle_normal(p0, p1, p2):
    c = np.cross(p1 - p0, p2 - p0)
    return normalize(c, fallback=np.broadcast_to(VEC_UP, c.shape))


def triangle_tangent(p0, p1, p2, uv0, uv1, uv2):
    dist_ba = p1 - p0
    dist_ca = p2 - p0
    t_ba = uv1 - uv0
    t_ca = uv2 - uv0
    area = t_ba[..., 0] * t_ca[..., 1] - t_ba[..., 1] * t_ca[..., 0]
    area = np.where(area == 0.0, 1.0, area)
    delta = -1.0 / area
    tan = (dist_ba * t_ca[..., 1:2] + dist_ca * -t_ba[..., 1:2]) * delta[..., None]
    l = length(tan)
    return np.where((l > 1e-30)[..., None], tan / np.where(l > 1e-30, l, 1.0)[..., None], tan)


def intersect_point_plane_dist(plane_pos, plane_dir, pos):
    v = pos - plane_pos
    gv = dot(v, plane_dir)[..., None] * plane_dir
    l = length(gv)
    inside = dot(plane_dir, v) < 0.0
    out_pos = np.where(inside[..., None], pos - gv, pos)
    dist = np.where(inside, -l, l)
    return dist, out_pos


def closest_pt_point_segment_ratio(c, a, b):
    ab = b - a
    d = dot(ab, ab)
    safe = np.where(d > 0.0, d, 1.0)
    t = dot(c - a, ab) / safe
    t = np.where(d > 0.0, t, 0.0)
    return saturate(t)


def closest_pt_segment_segment(p1, q1, p2, q2):
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = dot(d1, d1)
    e = dot(d2, d2)
    f = dot(d2, r)
    c = dot(d1, r)

    both = (a <= 1e-8) & (e <= 1e-8)
    first = (a <= 1e-8) & ~both
    second = (e <= 1e-8) & ~both & ~first

    safe_a = np.where(a > 1e-30, a, 1.0)
    safe_e = np.where(e > 1e-30, e, 1.0)

    b = dot(d1, d2)
    denom = a * e - b * b
    safe_denom = np.where(np.abs(denom) > 0.0, denom, 1.0)
    s = np.where(np.abs(denom) > 0.0, saturate((b * f - c * e) / safe_denom), 0.0)
    t = (b * s + f) / safe_e

    t_low = t < 0.0
    t_high = t > 1.0
    s = np.where(t_low, saturate(-c / safe_a), s)
    s = np.where(t_high, saturate((b - c) / safe_a), s)
    t = np.clip(t, 0.0, 1.0)

    s = np.where(first, 0.0, s)
    t = np.where(first, saturate(f / safe_e), t)
    t = np.where(second, 0.0, t)
    s = np.where(second, saturate(-c / safe_a), s)
    s = np.where(both, 0.0, s)
    t = np.where(both, 0.0, t)

    c1 = p1 + d1 * s[..., None]
    c2 = p2 + d2 * t[..., None]
    return s, t, c1, c2


def closest_pt_point_triangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = dot(ab, ap)
    d2 = dot(ac, ap)

    bp = p - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)

    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    in_a = (d1 <= 0.0) & (d2 <= 0.0)
    in_b = ~in_a & (d3 >= 0.0) & (d4 <= d3)
    denom_ab = np.where(np.abs(d1 - d3) > 0.0, d1 - d3, 1.0)
    v_ab = d1 / denom_ab
    in_ab = ~in_a & ~in_b & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    in_c = ~in_a & ~in_b & ~in_ab & (d6 >= 0.0) & (d5 <= d6)
    denom_ac = np.where(np.abs(d2 - d6) > 0.0, d2 - d6, 1.0)
    w_ac = d2 / denom_ac
    in_ac = ~in_a & ~in_b & ~in_ab & ~in_c & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    g = (d4 - d3) + (d5 - d6)
    safe_g = np.where(np.abs(g) > 0.0, g, 1.0)
    w_bc = (d4 - d3) / safe_g
    in_bc = ~in_a & ~in_b & ~in_ab & ~in_c & ~in_ac & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    in_face = ~in_a & ~in_b & ~in_ab & ~in_c & ~in_ac & ~in_bc
    h = va + vb + vc
    safe_h = np.where(np.abs(h) > 0.0, h, 1.0)
    v_face = vb / safe_h
    w_face = vc / safe_h

    u = np.where(in_a, 1.0,
        np.where(in_b, 0.0,
        np.where(in_ab, 1.0 - v_ab,
        np.where(in_c, 0.0,
        np.where(in_ac, 1.0 - w_ac,
        np.where(in_bc, 0.0, 1.0 - v_face - w_face))))))
    v = np.where(in_a, 0.0,
        np.where(in_b, 1.0,
        np.where(in_ab, v_ab,
        np.where(in_c, 0.0,
        np.where(in_ac, 0.0,
        np.where(in_bc, 1.0 - w_bc, v_face))))))
    w = np.where(in_a, 0.0,
        np.where(in_b, 0.0,
        np.where(in_ab, 0.0,
        np.where(in_c, 1.0,
        np.where(in_ac, w_ac,
        np.where(in_bc, w_bc, w_face))))))

    point = a * u[..., None] + b * v[..., None] + c * w[..., None]
    uvw = np.stack([u, v, w], axis=-1)
    return point, uvw


def _mod289(x):
    return x - np.floor(x * (1.0 / 289.0)) * 289.0


def _permute(x):
    return _mod289((x * 34.0 + 1.0) * x)


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def cnoise2(px, py):
    pix = np.floor(px)
    piy = np.floor(py)
    pfx = px - pix
    pfy = py - piy

    ix = np.stack([pix, pix + 1.0, pix, pix + 1.0], axis=-1)
    iy = np.stack([piy, piy, piy + 1.0, piy + 1.0], axis=-1)
    fx = np.stack([pfx, pfx - 1.0, pfx, pfx - 1.0], axis=-1)
    fy = np.stack([pfy, pfy, pfy - 1.0, pfy - 1.0], axis=-1)

    ix = _mod289(ix)
    iy = _mod289(iy)

    i = _permute(_permute(ix) + iy)

    gx = (i * (1.0 / 41.0)) % 1.0 * 2.0 - 1.0
    gy = np.abs(gx) - 0.5
    tx = np.floor(gx + 0.5)
    gx = gx - tx

    norm = 1.79284291400159 - 0.85373472095314 * (gx * gx + gy * gy)
    gx = gx * norm
    gy = gy * norm

    n = gx * fx + gy * fy

    fade_x = _fade(pfx)
    fade_y = _fade(pfy)

    nx0 = n[..., 0] + (n[..., 1] - n[..., 0]) * fade_x
    nx1 = n[..., 2] + (n[..., 3] - n[..., 2]) * fade_x
    nxy = nx0 + (nx1 - nx0) * fade_y
    return 2.3 * nxy


def calc_mass(depth):
    a = 1.0 - depth
    return 1.0 + a * a * defs.DEPTH_MASS


def calc_inverse_mass(friction, depth):
    a = 1.0 - depth
    mass = 1.0 + friction * defs.FRICTION_MASS + a * a * defs.DEPTH_MASS
    return 1.0 / mass


def calc_inverse_mass_fixed(friction, depth, fixed_mask, fix_mass):
    inv = calc_inverse_mass(friction, depth)
    return np.where(fixed_mask, 1.0 / fix_mass, inv)


def calc_self_collision_inverse_mass(friction, fixed_mask, cloth_mass):
    mass = np.where(fixed_mask, defs.SELF_COLLISION_FIXED_MASS, 1.0 + friction * defs.SELF_COLLISION_FRICTION_MASS)
    mass = mass + cloth_mass * defs.SELF_COLLISION_CLOTH_MASS
    return 1.0 / mass


def evaluate_curve_lut(lut, time):
    t = saturate(time)
    index = (t * 15.0).astype(np.int32)
    frac = t * 15.0 - index
    index2 = np.minimum(index + 1, 15)
    return lut[index] + (lut[index2] - lut[index]) * frac


def evaluate_curve_lut_clamp01(lut, time):
    return saturate(evaluate_curve_lut(lut, time))


def evaluate_team_lut(luts, team_index, time):
    t = saturate(time)
    index = (t * 15.0).astype(np.int32)
    frac = t * 15.0 - index
    index2 = np.minimum(index + 1, 15)
    a = luts[team_index, index]
    b = luts[team_index, index2]
    return a + (b - a) * frac


def evaluate_team_lut_clamp01(luts, team_index, time):
    return saturate(evaluate_team_lut(luts, team_index, time))


def ragged_expand(counts):
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    ends = np.cumsum(counts)
    starts = ends - counts
    owner = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    within = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    return owner, within


def pack_grid_cells(cells):
    b = defs.GRID_KEY_BIAS
    c = np.clip(cells, -b + 1, b - 1).astype(np.int64)
    return ((c[..., 0] + b) << 42) | ((c[..., 1] + b) << 21) | (c[..., 2] + b)
