"""Wireframe geometry builders.

Pure numpy in, numpy out: no bpy, no gpu, no knowledge of colliders or bones. Every builder returns
(starts, ends) float32 arrays of matched length, which is the only vertex form the draw kernel
consumes. Keeping this file free of Blender is what makes the shapes checkable without a viewport.
"""

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
    """Wire hull of the solver's swept segment: a ball at each end plus four tangent rails."""
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
    """Blender's own bone silhouette: a diamond one tenth along the head->tail axis.

    Bones are drawn as their own shape rather than a bare segment so that a highlighted chain reads
    at a glance against the armature underneath it, which a coincident line never can.
    """
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


def segments(pairs):
    if not len(pairs):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    pairs = np.asarray(pairs, dtype=np.float32)
    return pairs[:, 0], pairs[:, 1]
