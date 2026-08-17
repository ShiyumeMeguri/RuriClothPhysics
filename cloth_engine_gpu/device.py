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

# Blob aggregation (G2e-7): every resident/zone kernel array is stacked into one contiguous device
# blob per (dtype-family, per-row-shape) GROUP, so each field is a plain axis-0 slice of a shaped
# blob and the megakernel takes ~27 blob args instead of ~290 (host-side per-argument marshalling was
# the fixed ~1.4 ms launch floor). Grouping by shape (not just family) is what lets the kernel avoid
# device-code reshape -- numba-cuda implements reshape via a linked reshape_funcs.cu that cache=True
# cannot pickle -- so axis-0 slicing keeps cache=True. bool folds into u8.
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
    """Blob-group key for a (dtype-family, per-row-shape): '<family>_s' scalar, '_v<n>' 1-D row,
    '_m<a>x<b>...' multi-dim row. Fields sharing a group stack along axis 0 into one shaped blob."""
    per_row = tuple(int(x) for x in per_row)
    if not per_row:
        code = "s"
    elif len(per_row) == 1:
        code = "v%d" % per_row[0]
    else:
        code = "m" + "x".join(str(x) for x in per_row)
    return "%s_%s" % (family, code)


def build_blobs(host_arrays, groups):
    """Stack host ndarrays (one per slot, already sliced to its live count) into one contiguous device
    blob per (family, per-row-shape) group.

    Returns ``(blobs, offs_dev, lens_dev, views)``:
      * ``blobs`` : ``{group: device array of shape (total_rows, *per_row)}`` -- one per group in
        ``groups`` (every group must hold >= 1 field);
      * ``offs_dev`` / ``lens_dev`` : int64 device arrays -- the ROW base + ROW count of slot k into
        its group blob (read at runtime by the kernel; the brief mandates runtime offsets);
      * ``views`` : per-slot ``blob[base:base+rows]`` -- an axis-0 slice of shape ``(rows, *per_row)``,
        no reshape. A view IS the field's device array, so the staging bridges / per-field upload /
        readback keep working unchanged; only the megakernel signature collapses.

    Bool arrays fold into the u8 family (their device form); the only cast is bool->uint8."""
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
            # A group whose fields are all zero-row (e.g. collider work buffers when num_colliders==0)
            # would be a zero-size device allocation, which numba cannot slice ("non-empty slice into
            # empty slice"). Pad to one dummy row -- every field still slices to [base:base+0] = 0 rows
            # (never indexed: the owning phase's live-count guard, e.g. c < num_colliders == 0, skips it).
            blob = np.zeros((1,) + blob.shape[1:], blob.dtype)
        blobs[group] = cuda.to_device(np.ascontiguousarray(blob))
    views = []
    for slot, group in enumerate(slot_group):
        base = int(offs[slot])
        rows = int(lens[slot])
        views.append(blobs[group][base:base + rows])
    return blobs, cuda.to_device(offs), cuda.to_device(lens), views


class FieldSet:
    """A named set of resident device arrays mirroring a host struct/arena.

    ``host_dtypes`` records the ORIGINAL host dtype per field so readback restores
    bool fields from their uint8 device form.
    """

    __slots__ = ("device", "host_dtypes", "count")

    def __init__(self, host_dict, count, allocate=True):
        # ``allocate=False`` records only the host dtypes (for bool-restoring readback) and leaves the
        # device arrays to be injected as blob views via ``set_view`` (G2e-7 blob aggregation): the
        # field's device array then aliases a shared per-family blob instead of owning an allocation,
        # so per-field upload/download and the staging bridges keep working while the megakernel takes
        # the blob (not 145 field arrays).
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
