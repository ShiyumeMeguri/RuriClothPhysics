import math

import numpy as np

_circle_cache = {}


def unit_circle(segments):
    circle = _circle_cache.get(segments)
    if circle is None:
        angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
        circle = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        _circle_cache[segments] = circle
    return circle


def orthonormal_basis(axis):
    axis = np.asarray(axis, dtype=np.float64)
    length = float(np.linalg.norm(axis))
    axis = axis / length if length > 1e-12 else np.array([0.0, 1.0, 0.0])
    reference = np.array([0.0, 0.0, 1.0]) if abs(float(axis[2])) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    side = np.cross(axis, reference)
    length = float(np.linalg.norm(side))
    side = side / length if length > 1e-12 else np.array([1.0, 0.0, 0.0])
    return axis, side, np.cross(axis, side)


def _loop(points):
    return points, np.roll(points, -1, axis=0)


def circle(center, axis_a, axis_b, radius, segments=24):
    ring = unit_circle(segments)
    center = np.asarray(center, dtype=np.float32)
    axis_a = np.asarray(axis_a, dtype=np.float32)
    axis_b = np.asarray(axis_b, dtype=np.float32)
    points = center[None, :] + axis_a[None, :] * (ring[:, 0:1] * radius) \
        + axis_b[None, :] * (ring[:, 1:2] * radius)
    return _loop(points.astype(np.float32))


def sphere(center, radius, segments=24):
    starts, ends = [], []
    for axis_a, axis_b in (((1, 0, 0), (0, 1, 0)), ((0, 1, 0), (0, 0, 1)), ((0, 0, 1), (1, 0, 0))):
        first, second = circle(center, axis_a, axis_b, radius, segments)
        starts.append(first)
        ends.append(second)
    return np.concatenate(starts), np.concatenate(ends)


def capsule(first, second, first_radius, second_radius, segments=24):
    starts, ends = [], []
    for center, radius in ((first, first_radius), (second, second_radius)):
        a, b = sphere(center, radius, segments)
        starts.append(a)
        ends.append(b)
    axis, side_a, side_b = orthonormal_basis(np.asarray(second) - np.asarray(first))
    rails_start, rails_end = [], []
    for direction in (side_a, -side_a, side_b, -side_b):
        rails_start.append(np.asarray(first) + direction * first_radius)
        rails_end.append(np.asarray(second) + direction * second_radius)
    starts.append(np.array(rails_start, dtype=np.float32))
    ends.append(np.array(rails_end, dtype=np.float32))
    return np.concatenate(starts), np.concatenate(ends)


def plane(origin, axis, size=0.25):
    axis, side_a, side_b = orthonormal_basis(axis)
    origin = np.asarray(origin, dtype=np.float64)
    corners = np.array([origin + side_a * size + side_b * size,
                        origin - side_a * size + side_b * size,
                        origin - side_a * size - side_b * size,
                        origin + side_a * size - side_b * size], dtype=np.float32)
    starts, ends = _loop(corners)
    normal_start = origin.astype(np.float32)[None]
    normal_end = (origin + axis * size * 0.5).astype(np.float32)[None]
    return (np.concatenate([starts, normal_start]), np.concatenate([ends, normal_end]))


def cross(center, size=0.02):
    center = np.asarray(center, dtype=np.float32)
    axes = np.eye(3, dtype=np.float32) * size
    return center[None, :] - axes, center[None, :] + axes


def octahedron(head, tail, width=None):
    head = np.asarray(head, dtype=np.float64)
    tail = np.asarray(tail, dtype=np.float64)
    axis = tail - head
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    axis, side_a, side_b = orthonormal_basis(axis)
    if width is None:
        width = length * 0.1
    waist = head + axis * (length * 0.1)
    ring = np.array([waist + side_a * width, waist + side_b * width,
                     waist - side_a * width, waist - side_b * width])
    starts = np.concatenate([ring, np.repeat(head[None], 4, axis=0),
                             np.repeat(tail[None], 4, axis=0)])
    ends = np.concatenate([np.roll(ring, -1, axis=0), ring, ring])
    return starts.astype(np.float32), ends.astype(np.float32)


def spheres(centers, radii, segments=12):
    centers = np.asarray(centers, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32)
    if centers.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    ring = unit_circle(segments)
    starts, ends = [], []
    for first, second in ((0, 1), (1, 2), (2, 0)):
        offsets = np.zeros((centers.shape[0], segments, 3), dtype=np.float32)
        offsets[:, :, first] = ring[None, :, 0] * radii[:, None]
        offsets[:, :, second] = ring[None, :, 1] * radii[:, None]
        points = centers[:, None, :] + offsets
        starts.append(points.reshape(-1, 3))
        ends.append(np.roll(points, -1, axis=1).reshape(-1, 3))
    return np.concatenate(starts), np.concatenate(ends)


def swept_rails(first, second, first_radius, second_radius):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    axis = second - first
    length = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = np.divide(axis, np.where(length > 1e-12, length, 1.0))
    reference = np.tile(np.array([0.0, 0.0, 1.0]), (first.shape[0], 1))
    swap = np.abs(axis[:, 2]) > 0.9
    reference[swap] = np.array([1.0, 0.0, 0.0])
    side_a = np.cross(axis, reference)
    norm = np.linalg.norm(side_a, axis=1, keepdims=True)
    side_a = np.divide(side_a, np.where(norm > 1e-12, norm, 1.0))
    side_b = np.cross(axis, side_a)
    starts, ends = [], []
    for direction in (side_a, -side_a, side_b, -side_b):
        starts.append(first + direction * np.asarray(first_radius)[:, None])
        ends.append(second + direction * np.asarray(second_radius)[:, None])
    return (np.concatenate(starts).astype(np.float32),
            np.concatenate(ends).astype(np.float32))


def segments(pairs):
    if not len(pairs):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    pairs = np.asarray(pairs, dtype=np.float32)
    return pairs[:, 0], pairs[:, 1]
