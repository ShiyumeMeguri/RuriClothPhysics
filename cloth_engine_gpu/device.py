import numpy as np
from numba import cuda

FAMILY_DTYPE = {"f32": np.float32, "f64": np.float64, "i32": np.int32,
                "u8": np.uint8, "i8": np.int8}
_DTYPE_FAMILY = {np.dtype("float32"): "f32", np.dtype("float64"): "f64", np.dtype("int32"): "i32",
                 np.dtype("uint8"): "u8", np.dtype("int8"): "i8", np.dtype("bool"): "u8"}


def _device_friendly(array):
    array = np.ascontiguousarray(array)
    if array.dtype == np.bool_:
        return array.astype(np.uint8)
    return array


def group_name(family, per_row):
    per_row = tuple(int(x) for x in per_row)
    if not per_row:
        code = "s"
    elif len(per_row) == 1:
        code = "v%d" % per_row[0]
    else:
        code = "m" + "x".join(str(x) for x in per_row)
    return "%s_%s" % (family, code)


def build_blobs(host_arrays, groups):
    count = len(host_arrays)
    offs = np.zeros(count, np.int64)
    lens = np.zeros(count, np.int64)
    parts = {group: [] for group in groups}
    running = {group: 0 for group in groups}
    slot_group = []
    for slot, array in enumerate(host_arrays):
        array = np.ascontiguousarray(array)
        if array.dtype == np.bool_:
            array = array.astype(np.uint8)
        group = group_name(_DTYPE_FAMILY[array.dtype], array.shape[1:])
        assert group in parts, "slot %d -> group %s not in %r" % (slot, group, groups)
        rows = int(array.shape[0])
        offs[slot] = running[group]
        lens[slot] = rows
        running[group] += rows
        parts[group].append(array)
        slot_group.append(group)
    blobs = {}
    for group in groups:
        pieces = parts[group]
        assert pieces, "blob group %s has no fields" % group
        blob = np.concatenate(pieces, axis=0)
        if blob.shape[0] == 0:
            blob = np.zeros((1,) + blob.shape[1:], blob.dtype)
        blobs[group] = cuda.to_device(np.ascontiguousarray(blob))
    views = []
    for slot, group in enumerate(slot_group):
        base = int(offs[slot])
        rows = int(lens[slot])
        views.append(blobs[group][base:base + rows])
    return blobs, cuda.to_device(offs), cuda.to_device(lens), views


class FieldSet:

    __slots__ = ("device", "host_dtypes", "count")

    def __init__(self, host_dict, count, allocate=True):
        self.device = {}
        self.host_dtypes = {}
        self.count = count
        for name, array in host_dict.items():
            self.host_dtypes[name] = np.bool_ if array.dtype == np.bool_ else array.dtype
            if allocate:
                self.device[name] = cuda.to_device(_device_friendly(array))

    def set_view(self, name, view):
        self.device[name] = view

    def upload(self, name, host_array):
        self.device[name].copy_to_device(_device_friendly(host_array))

    def upload_many(self, host_dict, names):
        for name in names:
            self.upload(name, host_dict[name])

    def download(self, name):
        raw = self.device[name].copy_to_host()
        if self.host_dtypes[name] == np.bool_:
            return raw.astype(np.bool_)
        return raw

    def get(self, name):
        return self.device[name]


def dump_struct(struct_array, count):
    return {name: np.ascontiguousarray(struct_array[name][:count])
            for name in struct_array.dtype.names}


def dump_arena(arena, count):
    return {name: np.ascontiguousarray(arena.arrays[name][:count]) for name in arena.spec}


def scatter_struct(struct_array, flat, count, names):
    for name in names:
        values = flat[name]
        target = struct_array[name]
        if target.dtype == np.bool_:
            target[:count] = values.astype(np.bool_)
        else:
            target[:count] = values


def scatter_arena(arena, flat, count, names):
    for name in names:
        values = flat[name]
        target = arena.arrays[name]
        if target.dtype == np.bool_:
            target[:count] = values.astype(np.bool_)
        else:
            target[:count] = values


def upload_readonly(array):
    return cuda.to_device(_device_friendly(array))
