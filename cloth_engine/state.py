import numpy as np
import warp as wp

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


DOMAIN_SPECIFICATION_TABLE = (
    ("particle", _world.PARTICLE_FIELDS),
    ("transform", _world.TRANSFORM_FIELDS),
    ("collider", _world.COLLIDER_FIELDS),
    ("distance", _world.DISTANCE_FIELDS),
    ("bending", _world.BENDING_FIELDS),
    ("tether", _world.INDEX_FIELDS),
    ("motion", _world.INDEX_FIELDS),
    ("update_move", _world.INDEX_FIELDS),
    ("update_fixed", _world.INDEX_FIELDS),
    ("spring", _world.INDEX_FIELDS),
    ("collision_process", _world.INDEX_FIELDS),
    ("center_fixed", _world.INDEX_FIELDS),
    ("angle_buffered", _world.INDEX_FIELDS),
    ("edges", _world.EDGE_FIELDS),
    ("collision_edges", _world.EDGE_FIELDS),
    ("triangles", _world.TRIANGLE_FIELDS),
    ("v2t", _world.V2T_FIELDS),
    ("point_pairs", _world.PAIR_POINT_FIELDS),
    ("edge_pairs", _world.PAIR_EDGE_FIELDS),
    ("self_points", _world.PRIMITIVE_FIELDS),
    ("self_edges", _world.PRIMITIVE_FIELDS),
    ("self_triangles", _world.PRIMITIVE_FIELDS),
)


def _domain_fields_table():
    table = {"team": _fields_from_structured_dtype(_world.TEAM_DTYPE)}
    for domain_name, specification_map in DOMAIN_SPECIFICATION_TABLE:
        table[domain_name] = _fields_from_specification_map(specification_map)
    return table


DOMAIN_FIELDS = _domain_fields_table()

DOMAIN_NAMES = tuple(DOMAIN_FIELDS.keys())

DERIVED_STORAGE_NAME = "derived"

DERIVED_FIELDS = dict(_program.DERIVED_PLANE_FIELDS)

DERIVED_PLANE_NAMES = _program.DERIVED_PLANE_NAMES

STORAGE_NAMES = DOMAIN_NAMES + (DERIVED_STORAGE_NAME,)

STORAGE_FIELDS = dict(DOMAIN_FIELDS)
STORAGE_FIELDS[DERIVED_STORAGE_NAME] = DERIVED_FIELDS


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
    def __init__(self, element_counts, derived_plane_counts):
        missing = sorted(set(DOMAIN_NAMES) - set(element_counts))
        if missing:
            raise ValueError("element counts missing for domains %r" % (missing,))
        unknown = sorted(set(element_counts) - set(DOMAIN_NAMES))
        if unknown:
            raise ValueError("element counts given for unknown domains %r" % (unknown,))
        self.storages = {}
        for domain_name in DOMAIN_NAMES:
            self.storages[domain_name] = SlabStorage(
                domain_name, DOMAIN_FIELDS[domain_name],
                uniform_element_counts(DOMAIN_FIELDS[domain_name],
                                       int(element_counts[domain_name])))
        self.domain_element_counts = {domain_name: int(element_counts[domain_name])
                                      for domain_name in DOMAIN_NAMES}
        self.storages[DERIVED_STORAGE_NAME] = SlabStorage(
            DERIVED_STORAGE_NAME, DERIVED_FIELDS, dict(derived_plane_counts))

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
        domains = tuple((domain_name, self.domain_element_counts[domain_name])
                        for domain_name in DOMAIN_NAMES)
        planes = tuple((plane_name,
                        self.storages[DERIVED_STORAGE_NAME].element_counts[plane_name])
                       for plane_name in DERIVED_PLANE_NAMES)
        return domains + planes

    def total_size_in_bytes(self):
        return sum(self.storages[storage_name].total_size_in_bytes
                   for storage_name in STORAGE_NAMES)
