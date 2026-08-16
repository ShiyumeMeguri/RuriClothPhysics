import zlib

import numpy as np

_caches = {}


class FrameCache:
    def __init__(self):
        self.signature = None
        self.indices = None
        self.bone_names = []
        self.frames = {}

    def reset(self, signature, indices, bone_names):
        self.signature = signature
        self.indices = indices
        self.bone_names = [bone_names[i] for i in indices]
        self.frames = {}

    def store(self, frame, input_hash, basis):
        self.frames[frame] = (input_hash, np.array(basis, dtype=np.float32))

    def fetch(self, frame, input_hash):
        entry = self.frames.get(frame)
        if entry is None or entry[0] != input_hash:
            return None
        return entry[1]


def get(uid, config_index):
    return _caches.setdefault((uid, config_index), FrameCache())


def clear_object(uid):
    for key in [key for key in _caches if key[0] == uid]:
        _caches.pop(key, None)


def clear_all():
    _caches.clear()


def frame_count(uid, config_index):
    cache = _caches.get((uid, config_index))
    return len(cache.frames) if cache is not None else 0


def cached_range(uid):
    frames = set()
    for (cache_uid, _), cache in _caches.items():
        if cache_uid == uid:
            frames.update(cache.frames)
    if not frames:
        return None
    return min(frames), max(frames), len(frames)


def external_input_names(setup, binding, config):
    names = set()
    for name in setup.kin_external_parent:
        if name:
            names.add(name)
    for name in binding.bone_names:
        if name:
            names.add(name)
    if config.inertia.anchor_object is None:
        anchor = config.inertia.anchor_bone
        if anchor:
            names.add(anchor)
    if config.normal_alignment_object is None:
        alignment = config.normal_alignment_bone
        if alignment:
            names.add(alignment)
    return sorted(names)


def animated_inputs(setup, animated_names):
    if not animated_names:
        return ()
    return tuple(name for name in setup.bone_names if name in animated_names)


def input_hash(obj, names, animated, scene_salt):
    chunks = [scene_salt, np.asarray(obj.matrix_world, dtype=np.float64).tobytes()]
    pose_bones = obj.pose.bones
    for name in names:
        pose_bone = pose_bones.get(name)
        if pose_bone is None:
            chunks.append(b"\x00")
        else:
            chunks.append(np.asarray(pose_bone.matrix, dtype=np.float64).tobytes())
    chunks.append(repr(animated).encode("utf-8"))
    for name in animated:
        pose_bone = pose_bones.get(name)
        if pose_bone is None:
            chunks.append(b"\x00")
        else:
            chunks.append(np.asarray(pose_bone.matrix_basis, dtype=np.float64).tobytes())
    return zlib.crc32(b"".join(chunks))


def apply(obj, cache, basis):
    pose_bones = obj.pose.bones
    for slot, name in enumerate(cache.bone_names):
        pose_bone = pose_bones.get(name)
        if pose_bone is not None:
            pose_bone.matrix_basis = basis[slot].tolist()
