import numpy as np
import warp as wp

from ..cloth_kernel import defs as _defs
from ..cloth_kernel import io as _io
from ..cloth_kernel import program as _program
from ..cloth_kernel import world as _world
from . import target as _target

PINNED_STAGING_REASON = (
    "the staging slabs exist so that a host write reaches the device without the driver "
    "bouncing it through an anonymous copy, and page locked host memory is a CUDA resource "
    "handed out by the CUDA driver; on a host device the storage slab is already host "
    "memory and pinning buys nothing, so asking for it there would make a compile target "
    "whose whole point is running without a GPU depend on the CUDA driver being present; "
    "the flag is therefore read off the device the state is being built on and never "
    "declared a second time")

FIELD_ALIGNMENT = 16

WARP_DTYPE_TABLE = {
    ("float32", ()): (wp.float32, ()),
    ("float32", (2,)): (wp.float32, (2,)),
    ("float32", (3,)): (wp.float32, (3,)),
    ("float32", (4,)): (wp.float32, (4,)),
    ("float32", (8,)): (wp.float32, (8,)),
    ("float32", (11,)): (wp.float32, (11,)),
    ("float32", (16,)): (wp.float32, (16,)),
    ("float32", (22,)): (wp.float32, (22,)),
    ("float32", (_defs.CARRY_LEN,)): (wp.float32, (_defs.CARRY_LEN,)),
    ("float32", (2, 3)): (wp.float32, (2, 3)),
    ("float32", (3, 3)): (wp.float32, (3, 3)),
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
    ("int64", (3,)): (wp.int64, (3,)),
    ("uint64", ()): (wp.uint64, ()),
}

WARP_SCALAR_BYTES = {
    wp.float32: 4,
    wp.float64: 8,
    wp.int32: 4,
    wp.int64: 8,
    wp.uint64: 8,
    wp.mat44d: 128,
}

WARP_VECTOR_ALIAS_TABLE = {
    (wp.float32, (3,)): wp.vec3,
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
    ("collider_vertices", _world.COLLIDER_VERTEX_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("collider_faces", _world.COLLIDER_FACE_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
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
    ("self_points", _world.PRIMITIVE_DEVICE_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("self_edges", _world.PRIMITIVE_DEVICE_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
    ("self_triangles", _world.PRIMITIVE_DEVICE_FIELDS, STORAGE_COUNT_PER_STORAGE, None),
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

SPATIAL_INDEX_LEAF_SIZE_REASON = (
    "warp packs this many bounds into one leaf of the hierarchy, and its own guidance is "
    "that an intersection query wants one while a closest point query wants four or eight, "
    "because the second kind measures every candidate it reaches anyway and is better off "
    "reaching them in batches; every tree here is read through bvh_query_aabb, which is an "
    "intersection query however the answer is used afterwards, and the measurement agrees "
    "with the shape of the call rather than with the use: on sixteen thousand bounds bent "
    "through a turn and a quarter, at the candidate count a narrow phase actually returns, "
    "one leaf reads 0.049 ms against 0.056 at eight on the device target and the gap opens "
    "to 0.352 against 0.927 once a query returns a hundred candidates, which is what the "
    "growing search radius does when a point sits far from the surface; the host target "
    "prefers eight by a tenth, which is the smaller of the two effects, so every tree "
    "stays at one and bvhrefresh.py in the criteria repository holds the whole table")

REFIT_OVER_REBUILD_REASON = (
    "warp can record either a refit or an in place rebuild into a captured frame, on both "
    "compile targets, so the choice between them is a measurement; after bending sixteen "
    "thousand bounds through a turn and a quarter, which is a larger deformation than a "
    "skinned body ever goes through in one binding, the refit costs 0.041 ms against 0.386 "
    "on the device target and 0.044 ms against 3.560 on the host target, and the query "
    "that follows it is the same or quicker at every leaf size and every reach measured; "
    "the topology the build chose survives a coherent deformation, which is the only kind "
    "a pose change produces, so there is one refresh path here and not a choice")

SPATIAL_INDEX_SPECIFICATION = (
    ("self_edges", "aabb_min", "aabb_max", "team", 1),
    ("self_triangles", "aabb_min", "aabb_max", "team", 1),
    ("collider_faces", "aabb_min", "aabb_max", "collider", 1),
)

SPATIAL_INDEX_NAMES = tuple(row[0] for row in SPATIAL_INDEX_SPECIFICATION)

SPATIAL_INDEX_LEAF_SIZE = {row[0]: row[4] for row in SPATIAL_INDEX_SPECIFICATION}

SPATIAL_INDEX_BOUND_PLANES = tuple((row[0], field_name)
                                   for row in SPATIAL_INDEX_SPECIFICATION
                                   for field_name in (row[1], row[2], row[3]))

EMPTY_SPATIAL_INDEX_IDENTIFIER = 0

COLLIDER_MESH_POINT_STORAGE = "collider_vertices"

INDEX_BACKING_STORAGE_NAMES = SPATIAL_INDEX_NAMES + (COLLIDER_MESH_POINT_STORAGE,)

COLLIDER_MESH_INDEX_REASON = (
    "the triangles of every mesh collider sit in one shared face slab and one shared "
    "vertex slab, and one grouped hierarchy over that slab keeps each collider's triangles "
    "in a subtree of its own, which is what makes the query correct and not only quick: "
    "the bounds of a face are in the frame of the collider that owns it, so a traversal "
    "that reached another collider's faces would be comparing boxes in two different "
    "frames; the slabs never move once the state exists, because the state is rebuilt "
    "whenever the structure of the world changes, so the handle a captured frame baked in "
    "is current for as long as that frame is replayable")


def _validate_spatial_index_specification():
    seen = set()
    for row in SPATIAL_INDEX_SPECIFICATION:
        assert len(row) == 5, \
            "a spatial index row declares the storage, the lower bound field, the upper " \
            "bound field, the group field and the leaf size, got %r" % (row,)
        storage_name, lower_field, upper_field, group_field, leaf_size = row
        assert isinstance(leaf_size, int) and leaf_size >= 1, \
            "%s; the spatial index over %s declares the leaf size %r" \
            % (SPATIAL_INDEX_LEAF_SIZE_REASON, storage_name, leaf_size)
        assert storage_name not in seen, "storage %s is indexed twice" % storage_name
        seen.add(storage_name)
        assert storage_name in DOMAIN_NAMES, \
            "the spatial index over %s needs a domain, the state layer declares the domains " \
            "%r" % (storage_name, list(DOMAIN_NAMES))
        fields = STORAGE_FIELDS[storage_name]
        for field_name in (lower_field, upper_field):
            assert fields.get(field_name) == (np.dtype("float32"), (3,)), \
                "the spatial index over %s reads %s.%s as a bound and a bound is three " \
                "float32 components, the storage declares %r" \
                % (storage_name, storage_name, field_name, fields.get(field_name))
        assert fields.get(group_field) == (np.dtype("int32"), ()), \
            "the spatial index over %s groups by %s.%s and a group is one int32, the storage " \
            "declares %r" % (storage_name, storage_name, group_field, fields.get(group_field))


_validate_spatial_index_specification()


def _aligned(value):
    return (value + FIELD_ALIGNMENT - 1) // FIELD_ALIGNMENT * FIELD_ALIGNMENT


def _field_pointer(slab, offset):
    if slab.ptr is None:
        return None
    return slab.ptr + offset


def uniform_element_counts(fields, element_count):
    return {field_name: element_count for field_name in fields}


def adjacent_runs(indices):
    ordered = sorted(indices)
    if not ordered:
        return ()
    runs = []
    run_first = ordered[0]
    run_last = ordered[0]
    for index in ordered[1:]:
        if index != run_last + 1:
            runs.append((run_first, run_last))
            run_first = index
        run_last = index
    runs.append((run_first, run_last))
    return tuple(runs)


class SlabStorage:
    def __init__(self, storage_name, fields, element_counts, device, pinned_staging):
        missing = sorted(set(fields) - set(element_counts))
        if missing:
            raise ValueError("storage %r has no element count for fields %r"
                             % (storage_name, missing))
        unknown = sorted(set(element_counts) - set(fields))
        if unknown:
            raise ValueError("storage %r was given element counts for undeclared fields %r"
                             % (storage_name, unknown))
        assert not pinned_staging or device.is_cuda, PINNED_STAGING_REASON
        self.storage_name = storage_name
        self.device = device
        self.pinned_staging = pinned_staging
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
        self.device_slab = wp.zeros(self.total_size_in_bytes, dtype=wp.uint8, device=device)
        self.upload_slab = wp.zeros(self.total_size_in_bytes, dtype=wp.uint8, device="cpu",
                                    pinned=pinned_staging)
        self.download_slab = wp.zeros(self.total_size_in_bytes, dtype=wp.uint8, device="cpu",
                                      pinned=pinned_staging)
        self.device_arrays = {}
        self.vector_arrays = {}
        self.upload_views = {}
        self.download_views = {}
        for field_name in self.field_order:
            offset = self.byte_offsets[field_name]
            shape = self.array_shapes[field_name]
            warp_dtype = self.warp_dtypes[field_name]
            device_array = wp.array(
                ptr=_field_pointer(self.device_slab, offset), dtype=warp_dtype, shape=shape,
                device=device)
            device_array._backing_slab = self.device_slab
            self.device_arrays[field_name] = device_array
            vector_dtype = WARP_VECTOR_ALIAS_TABLE.get((warp_dtype, tuple(shape[1:])))
            if vector_dtype is not None:
                vector_array = wp.array(
                    ptr=_field_pointer(self.device_slab, offset), dtype=vector_dtype,
                    shape=(shape[0],), device=device)
                vector_array._backing_slab = self.device_slab
                self.vector_arrays[field_name] = vector_array
            upload_array = wp.array(
                ptr=_field_pointer(self.upload_slab, offset), dtype=warp_dtype, shape=shape,
                device="cpu", pinned=pinned_staging)
            download_array = wp.array(
                ptr=_field_pointer(self.download_slab, offset), dtype=warp_dtype, shape=shape,
                device="cpu", pinned=pinned_staging)
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
        uploaded = len(self.dirty_indices)
        for first_index, last_index in adjacent_runs(self.dirty_indices):
            self._copy_run(self.device_slab, self.upload_slab, first_index, last_index)
        self.dirty_indices.clear()
        return uploaded

    def _run_bytes(self, first_index, last_index):
        first_name = self.field_order[first_index]
        last_name = self.field_order[last_index]
        start = self.byte_offsets[first_name]
        end = self.byte_offsets[last_name] + self.byte_sizes[last_name]
        return start, end - start

    def _copy_run(self, destination, source, first_index, last_index):
        start, length = self._run_bytes(first_index, last_index)
        if length == 0:
            return 0
        wp.copy(destination, source, start, start, length)
        return 1

    def download_fields(self, field_names):
        indices = {self.field_indices[field_name] for field_name in field_names}
        copies = 0
        for first_index, last_index in adjacent_runs(indices):
            copies += self._copy_run(self.download_slab, self.device_slab, first_index,
                                     last_index)
        return copies

    def value(self, field_name):
        return self.download_views[field_name].copy()


class ClothState:
    def __init__(self, element_counts, plane_counts, target_name):
        self.target_name = target_name
        self.device = wp.get_device(_target.device_of(target_name))
        self.pinned_staging = bool(self.device.is_cuda)
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
            self.storages[storage_name] = SlabStorage(storage_name, fields, counts,
                                                      self.device, self.pinned_staging)
        self.domain_element_counts = {domain_name: int(element_counts[domain_name])
                                      for domain_name in DOMAIN_NAMES}
        self.spatial_indexes = {}
        self.revision = 0

    def build_spatial_indexes(self):
        self.spatial_indexes = {}
        self.revision += 1
        for row in SPATIAL_INDEX_SPECIFICATION:
            storage_name, lower_field, upper_field, group_field, leaf_size = row
            storage = self.storages[storage_name]
            element_count = storage.element_counts[lower_field]
            if element_count == 0:
                continue
            self.spatial_indexes[storage_name] = wp.Bvh(
                storage.vector_arrays[lower_field], storage.vector_arrays[upper_field],
                groups=storage.device_arrays[group_field], leaf_size=leaf_size)
        return tuple(sorted(self.spatial_indexes))

    def spatial_index_identifier(self, storage_name):
        assert storage_name in SPATIAL_INDEX_NAMES, \
            "there is no spatial index over %r, the state layer indexes %r" \
            % (storage_name, list(SPATIAL_INDEX_NAMES))
        held = self.spatial_indexes.get(storage_name)
        if held is None:
            return EMPTY_SPATIAL_INDEX_IDENTIFIER
        return held.id

    def refit_spatial_index(self, storage_name):
        held = self.spatial_indexes.get(storage_name)
        if held is None:
            return False
        held.refit()
        return True

    def element_count(self, domain_name):
        return self.domain_element_counts[domain_name]

    def resize_domain(self, domain_name, element_count):
        if domain_name not in self.domain_element_counts:
            raise ValueError("there is no domain named %r to resize" % (domain_name,))
        requested = int(element_count)
        if self.domain_element_counts[domain_name] == requested:
            return False
        assert domain_name not in INDEX_BACKING_STORAGE_NAMES, \
            "the domain %r carries a spatial index built on the arrays of its current slab, " \
            "and resizing replaces that slab, so the held index would point at released " \
            "memory and the graph that captured its identifier would keep replaying against " \
            "it" % (domain_name,)
        fields = STORAGE_FIELDS[domain_name]
        self.storages[domain_name] = SlabStorage(
            domain_name, fields, uniform_element_counts(fields, requested),
            self.device, self.pinned_staging)
        self.domain_element_counts[domain_name] = requested
        self.revision += 1
        return True

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

    def all_fields(self):
        return tuple((storage_name, field_name)
                     for storage_name in STORAGE_NAMES
                     for field_name in self.storages[storage_name].field_order)

    def read_batch(self, requests):
        requested = tuple(requests)
        by_storage = {}
        for storage_name, field_name in requested:
            by_storage.setdefault(storage_name, []).append(field_name)
        copies = 0
        for storage_name, field_names in by_storage.items():
            copies += self.storages[storage_name].download_fields(field_names)
        if copies > 0:
            wp.synchronize_device(self.device)
        return {(storage_name, field_name): self.storages[storage_name].value(field_name)
                for storage_name, field_name in requested}

    def read(self, storage_name, field_name):
        return self.read_batch(((storage_name, field_name),))[(storage_name, field_name)]

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
