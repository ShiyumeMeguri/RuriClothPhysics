import numpy as np

from . import compat
from ..cloth_kernel import compile as kc

_BONE_PATH_TOKEN = 'pose.bones["'


def collect_animated_bone_names(obj):
    names = set()

    def _scan(fcurves):
        for fcurve in fcurves:
            if fcurve.mute:
                continue
            data_path = fcurve.data_path
            start = data_path.find(_BONE_PATH_TOKEN)
            if start < 0:
                continue
            start += len(_BONE_PATH_TOKEN)
            end = data_path.find('"]', start)
            if end < 0:
                continue
            names.add(data_path[start:end])

    _scan(compat.iter_object_fcurves(obj))
    _scan(compat.iter_nla_fcurves(obj))
    return names


def read_matrix(mathutils_matrix):
    return np.array(mathutils_matrix, dtype=np.float64)


def matrix_to_quat(m):
    return kc._matrix_to_quat(m)


def matrix_scale(m):
    return kc._matrix_scale(m)


def matrix_scale_batch(matrices):
    return np.linalg.norm(matrices[:, :3, :3], axis=1)


def quat_to_matrix3(q):
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def quat_to_matrix3_batch(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.empty((len(q), 3, 3), dtype=np.float64)
    m[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    m[:, 0, 1] = 2.0 * (xy - wz)
    m[:, 0, 2] = 2.0 * (xz + wy)
    m[:, 1, 0] = 2.0 * (xy + wz)
    m[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    m[:, 1, 2] = 2.0 * (yz - wx)
    m[:, 2, 0] = 2.0 * (xz - wy)
    m[:, 2, 1] = 2.0 * (yz + wx)
    m[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return m


def evaluate_live_bone_world(obj, bone_name):
    pose_bone = obj.pose.bones.get(bone_name)
    if pose_bone is None:
        return None
    return read_matrix(obj.matrix_world) @ read_matrix(pose_bone.matrix)


def build_snapshot(obj, config):
    snapshot = kc.TopologySnapshot()
    snapshot.is_spring = config.cloth_type == 'BONE_SPRING'
    snapshot.connection_mode = config.connection_mode

    bones = obj.data.bones
    root_names = []
    seen = set()
    for item in config.root_bones:
        name = item.bone
        if name and name in bones and name not in seen:
            seen.add(name)
            root_names.append(name)
    snapshot.root_names = root_names
    if not root_names:
        return snapshot

    visited = set()
    ordered = []

    def _visit(bone):
        if bone.name in visited:
            return
        visited.add(bone.name)
        ordered.append(bone.name)
        for child in bone.children:
            _visit(child)

    for name in root_names:
        bone = bones.get(name)
        if bone is not None:
            _visit(bone)
    if not ordered:
        return snapshot

    snapshot.bone_names = ordered
    n = len(ordered)
    index_map = {name: i for i, name in enumerate(ordered)}

    matrix_world = read_matrix(obj.matrix_world)
    snapshot.matrix_world = matrix_world

    parent_index = np.full(n, -1, dtype=np.int32)
    external_parent = [None] * n
    rest_world = np.empty((n, 4, 4), dtype=np.float64)
    rest_relative = np.empty((n, 4, 4), dtype=np.float64)
    for i, name in enumerate(ordered):
        bone = bones[name]
        rest = read_matrix(bone.matrix_local)
        rest_world[i] = matrix_world @ rest
        parent = bone.parent
        if parent is not None:
            parent_rest = read_matrix(parent.matrix_local)
            rest_relative[i] = np.linalg.inv(parent_rest) @ rest
            if parent.name in index_map:
                parent_index[i] = index_map[parent.name]
            else:
                external_parent[i] = parent.name
        else:
            rest_relative[i] = rest
    snapshot.parent_index = parent_index
    snapshot.external_parent = external_parent
    snapshot.rest_world = rest_world
    snapshot.rest_relative = rest_relative

    snapshot.overrides = [(o.bone, o.attribute, o.disable_collision, o.exclude_motion)
                          for o in config.attribute_overrides]
    snapshot.collision_bones = [item.bone for item in config.collider_collision.collision_bones]

    skinning = []
    if (not snapshot.is_spring) and config.custom_skinning_enable:
        seen_extra = set()
        for item in config.skinning_bones:
            name = item.bone
            if not name or name not in bones or name in seen_extra:
                continue
            seen_extra.add(name)
            world = matrix_world @ read_matrix(bones[name].matrix_local)
            skinning.append((name, index_map.get(name, -1), world))
    snapshot.skinning = skinning

    snapshot.normal_alignment_mode = config.normal_alignment_mode
    if config.normal_alignment_mode == 'TRANSFORM':
        center = np.zeros(3, dtype=np.float32)
        reference = config.normal_alignment_object
        if reference is not None:
            world = read_matrix(reference.matrix_world)
            if config.normal_alignment_bone and reference.type == 'ARMATURE':
                bone_world = evaluate_live_bone_world(reference, config.normal_alignment_bone)
                if bone_world is not None:
                    world = bone_world
            w2l = np.linalg.inv(matrix_world)
            center = (w2l[:3, :3] @ world[:3, 3] + w2l[:3, 3]).astype(np.float32)
        snapshot.normal_alignment_center_local = center

    snapshot.gravity_direction = np.array(config.gravity_direction, dtype=np.float32)
    return snapshot


class KinematicsHost:
    def __init__(self, setup):
        self.bone_names = setup.bone_names
        self.parent_index = setup.kin_parent
        self.rest_relative = setup.kin_rest_relative
        self.rest_relative_inverse = np.linalg.inv(setup.kin_rest_relative)
        self.external_parent = setup.kin_external_parent
        self.levels = setup.kin_levels

    def capture_missing_basis(self, obj, animated_names, storage):
        pose_bones = obj.pose.bones
        for name in self.bone_names:
            if name in animated_names or name in storage:
                continue
            pose_bone = pose_bones.get(name)
            if pose_bone is not None:
                storage[name] = read_matrix(pose_bone.matrix_basis)

    def gather_basis(self, obj, animated_names, storage):
        pose_bones = obj.pose.bones
        n = len(self.bone_names)
        basis = np.empty((n, 4, 4), dtype=np.float64)
        identity = np.eye(4, dtype=np.float64)
        for i, name in enumerate(self.bone_names):
            if name in animated_names:
                pose_bone = pose_bones.get(name)
                basis[i] = read_matrix(pose_bone.matrix_basis) if pose_bone is not None else identity
            else:
                captured = storage.get(name)
                if captured is None:
                    pose_bone = pose_bones.get(name)
                    captured = read_matrix(pose_bone.matrix_basis) if pose_bone is not None else identity
                    storage[name] = captured
                basis[i] = captured
        return basis

    def compute_pose(self, obj, basis):
        pose_bones = obj.pose.bones
        n = len(self.bone_names)
        local = np.einsum('nij,njk->nik', self.rest_relative, basis)
        pose = np.empty((n, 4, 4), dtype=np.float64)
        for level in self.levels:
            parents = self.parent_index[level]
            internal = parents >= 0
            if np.any(internal):
                v = level[internal]
                pose[v] = np.einsum('nij,njk->nik', pose[parents[internal]], local[v])
            roots = level[~internal]
            for v in roots:
                external = self.external_parent[v]
                if external is not None:
                    parent_bone = pose_bones.get(external)
                    if parent_bone is not None:
                        pose[v] = read_matrix(parent_bone.matrix) @ local[v]
                    else:
                        pose[v] = local[v]
                else:
                    pose[v] = local[v]
        return pose

    def world_from_pose(self, obj_matrix_world, pose):
        return np.einsum('ij,njk->nik', obj_matrix_world, pose)

    def write_back(self, obj, world_positions, world_rotations, write_mask, position_mask,
                   anim_world, obj_matrix_world_inverse):
        pose_bones = obj.pose.bones
        n = len(self.bone_names)

        scale = np.linalg.norm(anim_world[:, :3, :3], axis=1)
        rotation = quat_to_matrix3_batch(world_rotations.astype(np.float64))
        final_world = np.zeros((n, 4, 4), dtype=np.float64)
        final_world[:, :3, :3] = rotation * scale[:, None, :]
        final_world[:, :3, 3] = np.where(position_mask[:, None],
                                         world_positions, anim_world[:, :3, 3])
        final_world[:, 3, 3] = 1.0
        final_world = np.where(write_mask[:, None, None], final_world, anim_world)

        final_pose = np.einsum('ij,njk->nik', obj_matrix_world_inverse, final_world)

        parent_pose = np.empty((n, 4, 4), dtype=np.float64)
        internal = self.parent_index >= 0
        if np.any(internal):
            parent_pose[internal] = final_pose[self.parent_index[internal]]
        for i in np.flatnonzero(~internal):
            external = self.external_parent[i]
            if external is not None:
                parent_bone = pose_bones.get(external)
                parent_pose[i] = read_matrix(parent_bone.matrix) if parent_bone is not None \
                    else np.eye(4)
            else:
                parent_pose[i] = np.eye(4)

        local = np.einsum('nij,njk->nik', np.linalg.inv(parent_pose), final_pose)
        basis = np.einsum('nij,njk->nik', self.rest_relative_inverse, local)

        for i in np.flatnonzero(write_mask):
            pose_bone = pose_bones.get(self.bone_names[i])
            if pose_bone is not None:
                pose_bone.matrix_basis = basis[i].tolist()
        return basis
