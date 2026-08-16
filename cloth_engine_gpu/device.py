"""Device buffer registry: struct-of-arrays mirror of the CPU world on the GPU.

The engine keeps the world's mutable arenas resident on the device as one flat
device array per field (SoA), plus the static Program tables. Each frame only the
*input* fields (host-mutated by ``io.set_team_*`` / ``add_force``) are re-uploaded and
only the *output* fields are read back, so ``io.team_output`` and the gate never see
the backend.

Booleans are stored as ``uint8`` on the device (numba-friendly, 0/1 truthy); f64
fields stay f64 (the coordinator authorises internal f64 for O(24) per-team phases to
mirror the oracle's ``trs`` / matrix-inverse paths). Everything is C-contiguous.
"""

import numpy as np
from numba import cuda


def _device_friendly(array):
    array = np.ascontiguousarray(array)
    if array.dtype == np.bool_:
        return array.astype(np.uint8)
    return array


class FieldSet:
    """A named set of resident device arrays mirroring a host struct/arena.

    ``host_dtypes`` records the ORIGINAL host dtype per field so readback restores
    bool fields from their uint8 device form.
    """

    __slots__ = ("device", "host_dtypes", "count")

    def __init__(self, host_dict, count):
        self.device = {}
        self.host_dtypes = {}
        self.count = count
        for name, array in host_dict.items():
            self.host_dtypes[name] = np.bool_ if array.dtype == np.bool_ else array.dtype
            self.device[name] = cuda.to_device(_device_friendly(array))

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
    """Structured ndarray (world.team) -> {field: contiguous host array of first `count`}."""
    return {name: np.ascontiguousarray(struct_array[name][:count])
            for name in struct_array.dtype.names}


def dump_arena(arena, count):
    """ChunkArena (particles/transforms/colliders) -> {field: contiguous host array}."""
    return {name: np.ascontiguousarray(arena.arrays[name][:count]) for name in arena.spec}


def scatter_struct(struct_array, flat, count, names):
    """Write named device-downloaded fields back into a structured ndarray."""
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
    """Upload a static array once (Program tables); returns the device array."""
    return cuda.to_device(_device_friendly(array))
