import numpy as np

from ..cloth_kernel import defs

AXIS_VECTORS = {
    'X': np.array([1.0, 0.0, 0.0]),
    'Y': np.array([0.0, 1.0, 0.0]),
    'Z': np.array([0.0, 0.0, 1.0]),
}

KIND_VALUES = {'SPHERE': defs.COLLIDER_SPHERE, 'CAPSULE': defs.COLLIDER_CAPSULE,
               'PLANE': defs.COLLIDER_PLANE}


def bone_matrix(obj, bone_name):
    if bone_name:
        pose_bone = obj.pose.bones.get(bone_name)
        if pose_bone is not None:
            return np.array(obj.matrix_world @ pose_bone.matrix, dtype=np.float64)
    return np.array(obj.matrix_world, dtype=np.float64)


def solve(item, matrix):
    """World-space geometry of one collider, matching the solver's own construction.

    Returns (kind, first, second, radius_first, radius_second) where for a capsule the
    two points are the segment ends carrying their respective radii.
    """
    rotation = matrix[:3, :3]
    scale = np.linalg.norm(rotation, axis=0)
    basis = rotation / np.where(scale > 1e-12, scale, 1.0)
    origin = rotation @ np.asarray(item.center, dtype=np.float64) + matrix[:3, 3]

    kind = KIND_VALUES[item.shape]
    if kind == defs.COLLIDER_SPHERE:
        radius = item.radius * abs(scale[0])
        return kind, origin, origin, radius, radius

    axis = AXIS_VECTORS[item.direction] * (-1.0 if item.reverse_direction else 1.0)
    projected = float(scale @ axis)
    direction = basis @ (axis * np.sign(projected) if projected != 0.0 else axis)
    magnitude = abs(projected) if projected != 0.0 else 1.0

    if kind == defs.COLLIDER_PLANE:
        return kind, origin, origin + direction, 0.0, 0.0

    start_radius = item.start_radius * magnitude
    end_radius = (item.end_radius if item.radius_separation else item.start_radius) * magnitude
    length = item.length * magnitude

    if item.aligned_on_center:
        start_len = length * 0.5
        end_len = length * 0.5
    else:
        start_len = 0.0
        end_len = length - start_radius
    start_len = max(start_len - start_radius, 0.0)
    end_len = max(end_len - end_radius, 0.0)

    return (kind, origin + direction * start_len, origin - direction * end_len,
            start_radius, end_radius)
