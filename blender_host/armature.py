import re

import numpy as np

from . import chain
from ..cloth_kernel import compile as kc


def read_matrix(mathutils_matrix):
    return np.array(mathutils_matrix, dtype=np.float64)


def matrix_to_quat(m):
    return kc._matrix_to_quat(m)


def matrix_scale(m):
    return kc._matrix_scale(m)


def decompose_component_basis(m):
    return kc.decompose_component_basis(m)


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
    snapshot.spring_active = snapshot.is_spring and config.spring.use_spring
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
    use_connect = np.zeros(n, dtype=bool)
    rest_world_external = {}
    rest_world = np.empty((n, 4, 4), dtype=np.float64)
    rest_relative = np.empty((n, 4, 4), dtype=np.float64)
    for i, name in enumerate(ordered):
        bone = bones[name]
        use_connect[i] = bone.use_connect
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
                rest_world_external[parent.name] = matrix_world @ parent_rest
        else:
            rest_relative[i] = rest
    snapshot.parent_index = parent_index
    snapshot.external_parent = external_parent
    snapshot.use_connect = use_connect
    snapshot.rest_world_external = rest_world_external
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


IDENTITY_BASIS = np.eye(4).tolist()


def clear_pose_basis(obj, names):
    if not len(names):
        return
    pose_bones = obj.pose.bones
    identity = IDENTITY_BASIS
    for name in names:
        pose_bone = pose_bones.get(name)
        if pose_bone is not None:
            pose_bone.matrix_basis = identity


def read_pose_matrices(obj, attribute):
    pose_bones = obj.pose.bones
    count = len(pose_bones)
    flat = np.empty(count * 16, dtype=np.float64)
    pose_bones.foreach_get(attribute, flat)
    return flat.reshape(count, 4, 4).transpose(0, 2, 1)


POSE_BASIS_DTYPE = np.float32

IDENTITY_MATRIX = np.eye(4, dtype=POSE_BASIS_DTYPE)

ANIMATION_POSE_REST_REASON = (
    "the animation pose the solver solves against is the basis with every driven bone "
    "nothing animates put back to rest, and that is done in this array rather than by "
    "writing rest onto the armature, because the write is a change to a datablock the "
    "dependency graph owns and blender answers it by evaluating the whole rig -- twenty two "
    "milliseconds a frame on this character to recover a matrix already known to be the "
    "identity. Which bones nothing animates is a property of the curves, the strips and the "
    "drivers the armature carries, so it is asked of them; a bone some curve does drive "
    "keeps whatever its basis holds, because for that bone the basis is the animation, and "
    "blender rewrites it from the curve on every frame anyway")

_BONE_DATA_PATH = re.compile(r'pose\.bones\["((?:[^"\\]|\\.)*)"\]')


def _bone_of_data_path(data_path):
    match = _BONE_DATA_PATH.match(data_path)
    if match is None:
        return ""
    return match.group(1).replace('\\"', '"').replace("\\\\", "\\")


def animated_bone_names(obj):
    from . import compat
    names = set()
    for fcurve in compat.iter_object_fcurves(obj):
        name = _bone_of_data_path(fcurve.data_path)
        if name:
            names.add(name)
    for fcurve in compat.iter_nla_fcurves(obj):
        name = _bone_of_data_path(fcurve.data_path)
        if name:
            names.add(name)
    animation = obj.animation_data
    if animation is not None:
        for driver in animation.drivers:
            name = _bone_of_data_path(driver.data_path)
            if name:
                names.add(name)
    return names

POSE_BUFFER_REASON = (
    "the frame reads the whole basis plane once and writes it back once, out of one buffer "
    "it keeps, because the read the solver needs and the write it produces are over the "
    "same array: the bones a config drives are a handful of rows in it and every other row "
    "has to come back unchanged, so the write has to start from what was read. The buffer "
    "is single precision because that is what blender stores, which keeps the untouched "
    "rows bit for bit what they were rather than rounding all of them through a double")


class PoseBuffer:

    def __init__(self):
        self.flat = None
        self.rest_rows = np.zeros(0, dtype=np.int64)
        self.rest_token = None

    def _resize(self, count):
        if self.flat is None or self.flat.shape[0] != count * 16:
            self.flat = np.empty(count * 16, dtype=POSE_BASIS_DTYPE)

    def read(self, obj):
        pose_bones = obj.pose.bones
        count = len(pose_bones)
        self._resize(count)
        pose_bones.foreach_get("matrix_basis", self.flat)
        return self.flat.reshape(count, 4, 4).transpose(0, 2, 1)

    def rest_driven_rows(self, obj, bone_names, pose_index, token):
        if self.rest_token == token:
            return self.rest_rows
        animated = animated_bone_names(obj)
        rows = {int(pose_index[position]) for position, name in enumerate(bone_names)
                if int(pose_index[position]) >= 0 and name not in animated}
        self.rest_rows = np.array(sorted(rows), dtype=np.int64)
        self.rest_token = token
        return self.rest_rows

    def read_animation(self, obj, rest_rows):
        matrices = self.read(obj)
        if rest_rows.size:
            self.flat.reshape(-1, 4, 4)[rest_rows] = IDENTITY_MATRIX
        return matrices

    def matrices(self):
        return self.flat.reshape(-1, 4, 4)


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
        self.slices = []
        levels_by_depth = {}
        offset = 0
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
        self.bone_position = {}
        for position, name in enumerate(bone_names):
            self.bone_position.setdefault(name, position)
        self.write_mask = np.concatenate(write_mask) if write_mask else np.zeros(0, dtype=bool)
        self.position_mask = np.concatenate(position_mask) if position_mask else np.zeros(0, dtype=bool)
        self.transform_extra = transform_extra
        self.root_mask = self.parent_index < 0
        self.pose_safe = np.where(self.pose_index >= 0, self.pose_index, 0)
        self.pose_missing = self.pose_index < 0
        self.level_groups = []
        for depth in sorted(levels_by_depth.keys()):
            level = np.concatenate(levels_by_depth[depth])
            parents = self.parent_index[level]
            internal = parents >= 0
            self.level_groups.append((level[internal], parents[internal]))

    def gather(self, all_basis):
        basis = all_basis[self.pose_safe]
        if self.pose_missing.any():
            basis[self.pose_missing] = np.eye(4)
        return basis

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

    def write_basis(self, matrix_world_inverse, all_matrix, final_world):
        count = self.count
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
