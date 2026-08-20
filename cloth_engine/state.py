import numpy as np
import warp as wp

from ..cloth_kernel import world as _world

DEVICE = "cuda:0"

FIELD_ALIGNMENT = 16

WARP_DTYPE_TABLE = {
    ("float32", ()): (wp.float32, ()),
    ("float32", (2,)): (wp.vec2, ()),
    ("float32", (3,)): (wp.vec3, ()),
    ("float32", (4,)): (wp.vec4, ()),
    ("float32", (16,)): (wp.float32, (16,)),
    ("float32", (4, 3)): (wp.float32, (4, 3)),
    ("float32", (4, 4)): (wp.float32, (4, 4)),
    ("float32", (2, 3)): (wp.float32, (2, 3)),
    ("float64", (4, 4)): (wp.mat44d, ()),
    ("int32", ()): (wp.int32, ()),
    ("int32", (2,)): (wp.vec2i, ()),
    ("int32", (3,)): (wp.vec3i, ()),
    ("int32", (4,)): (wp.vec4i, ()),
    ("int8", ()): (wp.int8, ()),
    ("bool", ()): (wp.uint8, ()),
    ("bool", (3,)): (wp.vec3ub, ()),
    ("int64", ()): (wp.int64, ()),
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


def _aligned(value):
    return (value + FIELD_ALIGNMENT - 1) // FIELD_ALIGNMENT * FIELD_ALIGNMENT


def _field_pointer(slab, offset):
    if slab.ptr is None:
        return None
    return slab.ptr + offset


class DomainStorage:
    def __init__(self, domain_name, fields, element_count):
        if element_count < 0:
            raise ValueError("domain %r requires a non negative element count, got %d"
                             % (domain_name, element_count))
        self.domain_name = domain_name
        self.element_count = element_count
        self.field_order = []
        self.field_indices = {}
        self.byte_offsets = {}
        self.byte_sizes = {}
        self.array_shapes = {}
        self.warp_dtypes = {}
        cursor = 0
        for field_name, (numpy_dtype, inner_shape) in fields.items():
            warp_dtype, trailing_shape = warp_dtype_for(numpy_dtype, inner_shape)
            item_size_in_bytes = numpy_dtype.itemsize
            for extent in inner_shape:
                item_size_in_bytes *= extent
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
                    "warp dtype mapping for %s.%s occupies %d bytes but the numpy specification "
                    "occupies %d bytes"
                    % (domain_name, field_name, upload_view.nbytes, self.byte_sizes[field_name]))
            self.upload_views[field_name] = upload_view
            self.download_views[field_name] = download_view
        self.dirty_indices = set()

    def write(self, field_name, values):
        view = self.upload_views[field_name]
        incoming = np.asarray(values)
        if incoming.shape != view.shape:
            raise ValueError("%s.%s expects shape %r, got %r"
                             % (self.domain_name, field_name, view.shape, incoming.shape))
        if incoming.dtype != view.dtype:
            raise TypeError("%s.%s expects dtype %s, got %s"
                            % (self.domain_name, field_name, view.dtype, incoming.dtype))
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
        wp.copy(self.device_slab, self.upload_slab, start, start, end - start)

    def read(self, field_name):
        start = self.byte_offsets[field_name]
        wp.copy(self.download_slab, self.device_slab, start, start, self.byte_sizes[field_name])
        wp.synchronize_device(DEVICE)
        return self.download_views[field_name].copy()


class ClothState:
    def __init__(self, element_counts):
        missing = set(DOMAIN_NAMES) - set(element_counts)
        if missing:
            raise ValueError("element counts missing for domains %r" % (sorted(missing),))
        unknown = set(element_counts) - set(DOMAIN_NAMES)
        if unknown:
            raise ValueError("element counts given for unknown domains %r" % (sorted(unknown),))
        self.storages = {}
        for domain_name in DOMAIN_NAMES:
            self.storages[domain_name] = DomainStorage(
                domain_name, DOMAIN_FIELDS[domain_name], int(element_counts[domain_name]))

    def element_count(self, domain_name):
        return self.storages[domain_name].element_count

    def field_names(self, domain_name):
        return tuple(self.storages[domain_name].field_order)

    def array(self, domain_name, field_name):
        return self.storages[domain_name].device_arrays[field_name]

    def arrays(self, domain_name):
        return self.storages[domain_name].device_arrays

    def warp_dtype(self, domain_name, field_name):
        return self.storages[domain_name].warp_dtypes[field_name]

    def value_specification(self, domain_name, field_name):
        view = self.storages[domain_name].upload_views[field_name]
        return (view.dtype, view.shape)

    def write(self, domain_name, field_name, values):
        self.storages[domain_name].write(field_name, values)

    def flush(self):
        uploaded = 0
        for domain_name in DOMAIN_NAMES:
            uploaded += self.storages[domain_name].upload_dirty()
        return uploaded

    def read(self, domain_name, field_name):
        return self.storages[domain_name].read(field_name)

    def structure_key(self):
        return tuple((domain_name, self.storages[domain_name].element_count)
                     for domain_name in DOMAIN_NAMES)

    def total_size_in_bytes(self):
        return sum(self.storages[domain_name].total_size_in_bytes
                   for domain_name in DOMAIN_NAMES)
