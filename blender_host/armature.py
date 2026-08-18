import numpy as np

from . import chain
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
    root_names, ordered = chain.config_chain(obj, config)
    snapshot.root_names = root_names
    if not root_names or not ordered:
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


def read_pose_matrices(obj, attribute):
    pose_bones = obj.pose.bones
    count = len(pose_bones)
    flat = np.empty(count * 16, dtype=np.float64)
    pose_bones.foreach_get(attribute, flat)
    return flat.reshape(count, 4, 4).transpose(0, 2, 1)


class BatchedKinematics:
    def __init__(self, obj, entries):
        name_to_index = {bone.name: index for index, bone in enumerate(obj.pose.bones)}
        rest_relative = []
        rest_relative_inverse = []
        parent_index = []
        pose_index = []
        external_index = []
        bone_names = []
        write_mask = []
        position_mask = []
        transform_extra = []
        collider_pose_index = []
        self.slices = []
        self.collider_slices = []
        levels_by_depth = {}
        offset = 0
        collider_offset = 0
        for entry in entries:
            kinematics = entry.kinematics
            count = len(kinematics.bone_names)
            self.slices.append((offset, offset + count))
            rest_relative.append(kinematics.rest_relative)
            rest_relative_inverse.append(kinematics.rest_relative_inverse)
            for local in range(count):
                parent = int(kinematics.parent_index[local])
                parent_index.append(parent + offset if parent >= 0 else -1)
            for name in kinematics.bone_names:
                pose_index.append(name_to_index.get(name, -1))
            for name in kinematics.external_parent:
                external_index.append(name_to_index.get(name, -1) if name is not None else -1)
            bone_names.extend(kinematics.bone_names)
            write_mask.append(entry.write_mask)
            position_mask.append(entry.position_mask)
            for depth, level in enumerate(kinematics.levels):
                levels_by_depth.setdefault(depth, []).append(np.asarray(level, dtype=np.int64) + offset)
            extras = entry.setup.transform_names[count:]
            transform_extra.append(np.array([name_to_index.get(name, -1) for name in extras], dtype=np.int64))
            binding = entry.binding
            self.collider_slices.append((collider_offset, collider_offset + binding.count))
            for index in range(binding.count):
                name = binding.bone_names[index]
                collider_pose_index.append(name_to_index.get(name, -1) if name else -1)
            collider_offset += binding.count
            offset += count
        self.count = offset
        self.rest_relative = np.concatenate(rest_relative, axis=0) if rest_relative \
            else np.zeros((0, 4, 4))
        self.rest_relative_inverse = np.concatenate(rest_relative_inverse, axis=0) if rest_relative_inverse \
            else np.zeros((0, 4, 4))
        self.parent_index = np.array(parent_index, dtype=np.int64)
        self.pose_index = np.array(pose_index, dtype=np.int64)
        self.external_index = np.array(external_index, dtype=np.int64)
        self.bone_names = bone_names
        self.write_mask = np.concatenate(write_mask) if write_mask else np.zeros(0, dtype=bool)
        self.position_mask = np.concatenate(position_mask) if position_mask else np.zeros(0, dtype=bool)
        self.transform_extra = transform_extra
        self.collider_pose_index = np.array(collider_pose_index, dtype=np.int64)
        self.root_mask = self.parent_index < 0
        self.pose_safe = np.where(self.pose_index >= 0, self.pose_index, 0)
        self.pose_missing = self.pose_index < 0
        self.level_groups = []
        for depth in sorted(levels_by_depth.keys()):
            level = np.concatenate(levels_by_depth[depth])
            parents = self.parent_index[level]
            internal = parents >= 0
            self.level_groups.append((level[internal], parents[internal]))
        self.frozen_basis = None
        self.captured_mask = None

    def _build_frozen(self, storage):
        self.frozen_basis = np.zeros((self.count, 4, 4), dtype=np.float64)
        self.captured_mask = np.zeros(self.count, dtype=bool)
        for index, name in enumerate(self.bone_names):
            captured = storage.get(name)
            if captured is not None:
                self.frozen_basis[index] = captured
                self.captured_mask[index] = True

    def gather(self, all_basis, animated_names, storage):
        live = all_basis[self.pose_safe]
        if self.pose_missing.any():
            live[self.pose_missing] = np.eye(4)
        animated = np.fromiter((name in animated_names for name in self.bone_names),
                               dtype=bool, count=self.count)
        if self.frozen_basis is None:
            self._build_frozen(storage)
        pending = (~animated) & (~self.captured_mask)
        if pending.any():
            for index in np.flatnonzero(pending):
                name = self.bone_names[index]
                captured = storage.get(name)
                if captured is None:
                    captured = live[index].copy()
                    storage[name] = captured
                self.frozen_basis[index] = captured
                self.captured_mask[index] = True
        return np.where(animated[:, None, None], live, self.frozen_basis)

    def compute_world(self, matrix_world, all_matrix, basis):
        local = np.einsum('nij,njk->nik', self.rest_relative, basis)
        pose = np.empty_like(local)
        if self.root_mask.any():
            roots = np.flatnonzero(self.root_mask)
            external = self.external_index[roots]
            has_external = external >= 0
            attached = roots[has_external]
            if attached.size:
                pose[attached] = np.matmul(all_matrix[external[has_external]], local[attached])
            detached = roots[~has_external]
            if detached.size:
                pose[detached] = local[detached]
        for level, parents in self.level_groups:
            if level.size:
                pose[level] = np.einsum('nij,njk->nik', pose[parents], local[level])
        return np.einsum('ij,njk->nik', matrix_world, pose)

    def write_basis(self, matrix_world_inverse, all_matrix, world_positions, world_rotations, anim_world):
        count = self.count
        scale = np.linalg.norm(anim_world[:, :3, :3], axis=1)
        rotation = quat_to_matrix3_batch(world_rotations.astype(np.float64))
        final_world = np.zeros((count, 4, 4), dtype=np.float64)
        final_world[:, :3, :3] = rotation * scale[:, None, :]
        final_world[:, :3, 3] = np.where(self.position_mask[:, None], world_positions, anim_world[:, :3, 3])
        final_world[:, 3, 3] = 1.0
        final_world = np.where(self.write_mask[:, None, None], final_world, anim_world)
        final_pose = np.einsum('ij,njk->nik', matrix_world_inverse, final_world)
        parent_pose = np.empty((count, 4, 4), dtype=np.float64)
        internal = ~self.root_mask
        parent_pose[internal] = final_pose[self.parent_index[internal]]
        if self.root_mask.any():
            roots = np.flatnonzero(self.root_mask)
            external = self.external_index[roots]
            has_external = external >= 0
            attached = roots[has_external]
            if attached.size:
                parent_pose[attached] = all_matrix[external[has_external]]
            detached = roots[~has_external]
            if detached.size:
                parent_pose[detached] = np.eye(4)
        local = np.einsum('nij,njk->nik', np.linalg.inv(parent_pose), final_pose)
        return np.einsum('nij,njk->nik', self.rest_relative_inverse, local)

    def entry_transform_worlds(self, entry_index, matrix_world, all_matrix, anim_world):
        extras = self.transform_extra[entry_index]
        count = anim_world.shape[0]
        transform_worlds = np.empty((count + extras.shape[0], 4, 4))
        transform_worlds[:count] = anim_world
        if extras.shape[0]:
            safe = np.where(extras >= 0, extras, 0)
            worlds = np.einsum('ij,njk->nik', matrix_world, all_matrix[safe])
            missing = extras < 0
            if missing.any():
                worlds[missing] = matrix_world
            transform_worlds[count:] = worlds
        return transform_worlds
