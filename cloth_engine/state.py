import numpy as np
import warp as wp

from ..cloth_kernel import defs as _defs
from ..cloth_kernel import io as _io
from ..cloth_kernel import program as _program
from ..cloth_kernel import world as _world

DEVICE = "cuda:0"

FIELD_ALIGNMENT = 16

WARP_DTYPE_TABLE = {
    ("float32", ()): (wp.float32, ()),
    ("float32", (2,)): (wp.float32, (2,)),
    ("float32", (3,)): (wp.float32, (3,)),
    ("float32", (4,)): (wp.float32, (4,)),
    ("float32", (16,)): (wp.float32, (16,)),
    ("float32", (22,)): (wp.float32, (22,)),
    ("float32", (2, 3)): (wp.float32, (2, 3)),
    ("float32", (4, 3)): (wp.float32, (4, 3)),
    ("float32", (4, 4)): (wp.float32, (4, 4)),
    ("float64", (3,)): (wp.float64, (3,)),
    ("float64", (4, 4)): (wp.mat44d, ()),
    ("int32", ()): (wp.int32, ()),
    ("int32", (2,)): (wp.int32, (2,)),
    ("int32", (3,)): (wp.int32, (3,)),
    ("int32", (4,)): (wp.int32, (4,)),
    ("int8", ()): (wp.int32, ()),
    ("uint8", ()): (wp.int32, ()),
    ("bool", ()): (wp.int32, ()),
    ("bool", (3,)): (wp.int32, (3,)),
    ("int64", ()): (wp.int64, ()),
}

WARP_SCALAR_BYTES = {
    wp.float32: 4,
    wp.float64: 8,
    wp.int32: 4,
    wp.int64: 8,
    wp.mat44d: 128,
}


def warp_dtype_for(numpy_dtype, shape):
    key = (numpy_dtype.name, tuple(int(extent) for extent in shape))
    mapping = WARP_DTYPE_TABLE.get(key)
    if mapping is None:
        raise TypeError("no warp dtype mapping declared for numpy field specification %r" % (key,))
    return mapping


def _fields_from_structured_dtype(structured_dtype):
    fields = {}
    for field_name in structured_dtype.names:
        member_dtype = structured_dtype.fields[field_name][0]
        if member_dtype.subdtype is None:
            fields[field_name] = (member_dtype, ())
        else:
            base_dtype, inner_shape = member_dtype.subdtype
            fields[field_name] = (base_dtype, tuple(int(extent) for extent in inner_shape))
    return fields


def _fields_from_specification_map(specification_map):
    fields = {}
    for field_name, (scalar_type, inner_shape) in specification_map.items():
        fields[field_name] = (np.dtype(scalar_type), tuple(int(extent) for extent in inner_shape))
    return fields


def _normalized_fields(field_source):
    if isinstance(field_source, np.dtype):
        return _fields_from_structured_dtype(field_source)
    return _fields_from_specification_map(field_source)


STORAGE_COUNT_PER_STORAGE = "per_storage"
STORAGE_COUNT_PER_PLANE = "per_plane"
STORAGE_COUNT_FROM_SPECIFICATION = "from_specification"

STORAGE_COUNT_RULES = (STORAGE_COUNT_PER_STORAGE, STORAGE_COUNT_PER_PLANE,
                       STORAGE_COUNT_FROM_SPECIFICATION)

DERIVED_STORAGE_NAME = "derived"

FRAME_SCALAR_STORAGE_NAME = "frame_scalar"

FRAME_SCALAR_FIELD_SOURCE = {
    plane_name: (scalar_type, ())
    for plane_name, scalar_type, _element_count in _defs.FRAME_SCALAR_PLANE_SPECIFICATION}

FRAME_SCALAR_PLANE_COUNTS = {
    plane_name: int(element_count)
    for plane_name, _scalar_type, element_count in _defs.FRAME_SCALAR_PLANE_SPECIFICATION}

STORAGE_SPECIFICATION_TABLE = (
    ("team", _world.TEAM_DTYPE, STORAGE_COUNT_PER_STORAGE, None),
    ("particle", _world.PARTICLE_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("transform", _world.TRANSFORM_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("collider", _world.COLLIDER_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("distance", _world.DISTANCE_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("bending", _world.BENDING_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("tether", _world.INDEX_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("motion", _world.INDEX_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("update_move", _world.INDEX_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("update_fixed", _world.INDEX_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("spring", _world.INDEX_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("collision_process", _world.INDEX_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("center_fixed", _world.INDEX_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("angle_buffered", _world.INDEX_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("edges", _world.EDGE_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("collision_edges", _world.EDGE_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("triangles", _world.TRIANGLE_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("v2t", _world.V2T_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("point_pairs", _world.PAIR_POINT_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("edge_pairs", _world.PAIR_EDGE_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("self_points", _world.PRIMITIVE_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("self_edges", _world.PRIMITIVE_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("self_triangles", _world.PRIMITIVE_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("zone", _io.ZONE_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    (DERIVED_STORAGE_NAME, _program.DERIVED_PLANE_FIELDS, STORAGE_COUNT_PER_PLANE, None),
    (FRAME_SCALAR_STORAGE_NAME, FRAME_SCALAR_FIELD_SOURCE, STORAGE_COUNT_FROM_SPECIFICATION,
     FRAME_SCALAR_PLANE_COUNTS),
)


def _validate_storage_specification():
    seen = set()
    for row in STORAGE_SPECIFICATION_TABLE:
        assert len(row) == 4, \
            "a storage row declares name, field source, count rule and declared counts, " \
            "got %r" % (row,)
        storage_name, field_source, count_rule, declared_counts = row
        assert storage_name not in seen, "storage %s is declared twice" % storage_name
        seen.add(storage_name)
        assert count_rule in STORAGE_COUNT_RULES, \
            "storage %s declares the count rule %r, only %r are defined" \
            % (storage_name, count_rule, STORAGE_COUNT_RULES)
        fields = _normalized_fields(field_source)
        assert fields, "storage %s declares no fields" % storage_name
        if count_rule == STORAGE_COUNT_FROM_SPECIFICATION:
            assert declared_counts is not None and set(declared_counts) == set(fields), \
                "storage %s takes its element counts from its own specification so the " \
                "specification has to carry one count per field" % storage_name
        else:
            assert declared_counts is None, \
                "storage %s takes its element counts from the caller so it must not also " \
                "carry declared counts" % storage_name


_validate_storage_specification()

STORAGE_NAMES = tuple(row[0] for row in STORAGE_SPECIFICATION_TABLE)

STORAGE_FIELDS = {row[0]: _normalized_fields(row[1]) for row in STORAGE_SPECIFICATION_TABLE}

STORAGE_COUNT_RULE = {row[0]: row[2] for row in STORAGE_SPECIFICATION_TABLE}

STORAGE_DECLARED_COUNTS = {row[0]: row[3] for row in STORAGE_SPECIFICATION_TABLE
                           if row[3] is not None}

DOMAIN_NAMES = tuple(storage_name for storage_name in STORAGE_NAMES
                     if STORAGE_COUNT_RULE[storage_name] == STORAGE_COUNT_PER_STORAGE)

PLANE_STORAGE_NAMES = tuple(storage_name for storage_name in STORAGE_NAMES
                            if STORAGE_COUNT_RULE[storage_name] == STORAGE_COUNT_PER_PLANE)

DOMAIN_FIELDS = {storage_name: STORAGE_FIELDS[storage_name] for storage_name in DOMAIN_NAMES}

DERIVED_FIELDS = STORAGE_FIELDS[DERIVED_STORAGE_NAME]

DERIVED_PLANE_NAMES = _program.DERIVED_PLANE_NAMES

FRAME_SCALAR_FIELDS = STORAGE_FIELDS[FRAME_SCALAR_STORAGE_NAME]


def _aligned(value):
    return (value + FIELD_ALIGNMENT - 1) // FIELD_ALIGNMENT * FIELD_ALIGNMENT


def _field_pointer(slab, offset):
    if slab.ptr is None:
        return None
    return slab.ptr + offset


def uniform_element_counts(fields, element_count):
    return {field_name: element_count for field_name in fields}


class SlabStorage:
    def __init__(self, storage_name, fields, element_counts):
        missing = sorted(set(fields) - set(element_counts))
        if missing:
            raise ValueError("storage %r has no element count for fields %r"
                             % (storage_name, missing))
        unknown = sorted(set(element_counts) - set(fields))
        if unknown:
            raise ValueError("storage %r was given element counts for undeclared fields %r"
                             % (storage_name, unknown))
        self.storage_name = storage_name
        self.element_counts = {}
        self.field_order = []
        self.field_indices = {}
        self.byte_offsets = {}
        self.byte_sizes = {}
        self.array_shapes = {}
        self.warp_dtypes = {}
        cursor = 0
        for field_name, (numpy_dtype, inner_shape) in fields.items():
            element_count = int(element_counts[field_name])
            if element_count < 0:
                raise ValueError("storage %r field %r requires a non negative element count, "
                                 "got %d" % (storage_name, field_name, element_count))
            warp_dtype, trailing_shape = warp_dtype_for(numpy_dtype, inner_shape)
            item_size_in_bytes = WARP_SCALAR_BYTES[warp_dtype]
            for extent in trailing_shape:
                item_size_in_bytes *= extent
            self.element_counts[field_name] = element_count
            self.field_indices[field_name] = len(self.field_order)
            self.field_order.append(field_name)
            self.byte_offsets[field_name] = cursor
            self.byte_sizes[field_name] = item_size_in_bytes * element_count
            self.array_shapes[field_name] = (element_count,) + trailing_shape
            self.warp_dtypes[field_name] = warp_dtype
            cursor = _aligned(cursor + item_size_in_bytes * element_count)
        self.total_size_in_bytes = cursor
        self.device_slab = wp.zeros(self.total_size_in_bytes, dtype=wp.uint8, device=DEVICE)
        self.upload_slab = wp.zeros(self.total_size_in_bytes, dtype=wp.uint8, device="cpu",
                                    pinned=True)
        self.download_slab = wp.zeros(self.total_size_in_bytes, dtype=wp.uint8, device="cpu",
                                      pinned=True)
        self.device_arrays = {}
        self.upload_views = {}
        self.download_views = {}
        for field_name in self.field_order:
            offset = self.byte_offsets[field_name]
            shape = self.array_shapes[field_name]
            warp_dtype = self.warp_dtypes[field_name]
            device_array = wp.array(
                ptr=_field_pointer(self.device_slab, offset), dtype=warp_dtype, shape=shape,
                device=DEVICE)
            device_array._backing_slab = self.device_slab
            self.device_arrays[field_name] = device_array
            upload_array = wp.array(
                ptr=_field_pointer(self.upload_slab, offset), dtype=warp_dtype, shape=shape,
                device="cpu", pinned=True)
            download_array = wp.array(
                ptr=_field_pointer(self.download_slab, offset), dtype=warp_dtype, shape=shape,
                device="cpu", pinned=True)
            upload_view = upload_array.numpy()
            download_view = download_array.numpy()
            if upload_view.nbytes != self.byte_sizes[field_name]:
                raise TypeError(
                    "warp dtype mapping for %s.%s occupies %d bytes but the field layout "
                    "reserves %d bytes"
                    % (storage_name, field_name, upload_view.nbytes,
                       self.byte_sizes[field_name]))
            self.upload_views[field_name] = upload_view
            self.download_views[field_name] = download_view
        self.dirty_indices = set()

    def write(self, field_name, values):
        view = self.upload_views[field_name]
        incoming = np.asarray(values)
        if incoming.shape != view.shape:
            raise ValueError("%s.%s expects shape %r, got %r"
                             % (self.storage_name, field_name, view.shape, incoming.shape))
        if incoming.dtype != view.dtype:
            raise TypeError("%s.%s expects dtype %s, got %s"
                            % (self.storage_name, field_name, view.dtype, incoming.dtype))
        view[...] = incoming
        self.dirty_indices.add(self.field_indices[field_name])

    def upload_dirty(self):
        if not self.dirty_indices:
            return 0
        ordered = sorted(self.dirty_indices)
        uploaded = len(ordered)
        run_first = ordered[0]
        run_last = ordered[0]
        for index in ordered[1:]:
            if index != run_last + 1:
                self._upload_run(run_first, run_last)
                run_first = index
            run_last = index
        self._upload_run(run_first, run_last)
        self.dirty_indices.clear()
        return uploaded

    def _upload_run(self, first_index, last_index):
        first_name = self.field_order[first_index]
        last_name = self.field_order[last_index]
        start = self.byte_offsets[first_name]
        end = self.byte_offsets[last_name] + self.byte_sizes[last_name]
        if end == start:
            return
        wp.copy(self.device_slab, self.upload_slab, start, start, end - start)

    def read(self, field_name):
        byte_size = self.byte_sizes[field_name]
        if byte_size > 0:
            start = self.byte_offsets[field_name]
            wp.copy(self.download_slab, self.device_slab, start, start, byte_size)
            wp.synchronize_device(DEVICE)
        return self.download_views[field_name].copy()


class ClothState:
    def __init__(self, element_counts, plane_counts):
        missing = sorted(set(DOMAIN_NAMES) - set(element_counts))
        if missing:
            raise ValueError("element counts missing for domains %r" % (missing,))
        unknown = sorted(set(element_counts) - set(DOMAIN_NAMES))
        if unknown:
            raise ValueError("element counts given for unknown domains %r" % (unknown,))
        declared_planes = set()
        for storage_name in PLANE_STORAGE_NAMES:
            declared_planes |= set(STORAGE_FIELDS[storage_name])
        missing_planes = sorted(declared_planes - set(plane_counts))
        if missing_planes:
            raise ValueError("element counts missing for planes %r" % (missing_planes,))
        unknown_planes = sorted(set(plane_counts) - declared_planes)
        if unknown_planes:
            raise ValueError("element counts given for unknown planes %r" % (unknown_planes,))
        self.storages = {}
        for storage_name in STORAGE_NAMES:
            fields = STORAGE_FIELDS[storage_name]
            count_rule = STORAGE_COUNT_RULE[storage_name]
            if count_rule == STORAGE_COUNT_PER_STORAGE:
                counts = uniform_element_counts(fields, int(element_counts[storage_name]))
            elif count_rule == STORAGE_COUNT_PER_PLANE:
                counts = {plane_name: int(plane_counts[plane_name]) for plane_name in fields}
            else:
                counts = dict(STORAGE_DECLARED_COUNTS[storage_name])
            self.storages[storage_name] = SlabStorage(storage_name, fields, counts)
        self.domain_element_counts = {domain_name: int(element_counts[domain_name])
                                      for domain_name in DOMAIN_NAMES}

    def element_count(self, domain_name):
        return self.domain_element_counts[domain_name]

    def plane_element_count(self, storage_name, field_name):
        return self.storages[storage_name].element_counts[field_name]

    def field_names(self, storage_name):
        return tuple(self.storages[storage_name].field_order)

    def array(self, storage_name, field_name):
        return self.storages[storage_name].device_arrays[field_name]

    def arrays(self, storage_name):
        return self.storages[storage_name].device_arrays

    def warp_dtype(self, storage_name, field_name):
        return self.storages[storage_name].warp_dtypes[field_name]

    def value_specification(self, storage_name, field_name):
        view = self.storages[storage_name].upload_views[field_name]
        return (view.dtype, view.shape)

    def write(self, storage_name, field_name, values):
        self.storages[storage_name].write(field_name, values)

    def flush(self):
        uploaded = 0
        for storage_name in STORAGE_NAMES:
            uploaded += self.storages[storage_name].upload_dirty()
        return uploaded

    def read(self, storage_name, field_name):
        return self.storages[storage_name].read(field_name)

    def structure_key(self):
        entries = []
        for storage_name in STORAGE_NAMES:
            storage = self.storages[storage_name]
            if STORAGE_COUNT_RULE[storage_name] == STORAGE_COUNT_PER_STORAGE:
                entries.append((storage_name, self.domain_element_counts[storage_name]))
                continue
            for plane_name in storage.field_order:
                entries.append((plane_name, storage.element_counts[plane_name]))
        return tuple(entries)

    def total_size_in_bytes(self):
        return sum(self.storages[storage_name].total_size_in_bytes
                   for storage_name in STORAGE_NAMES)
