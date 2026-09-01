import warp as wp

from . import policy
from . import state as _state

wp.set_module_options(policy.MODULE_OPTIONS)

SPATIAL_INDEX_MEMBER_SUFFIX = "_index"


def storage_structure_name(storage_name):
    return "".join(part.capitalize() for part in storage_name.split("_")) + "Storage"


def spatial_index_member_name(index_name):
    return index_name + SPATIAL_INDEX_MEMBER_SUFFIX


def _array_annotation(numpy_dtype, inner_shape):
    warp_dtype, trailing_shape = _state.warp_dtype_for(numpy_dtype, inner_shape)
    return wp.array(dtype=warp_dtype, ndim=1 + len(trailing_shape))


def _declare_structure(structure_name, annotations):
    namespace = {"__annotations__": dict(annotations), "__qualname__": structure_name,
                 "__module__": __name__}
    return wp.struct(type(structure_name, (), namespace))


def _storage_structures():
    structures = {}
    for storage_name in _state.STORAGE_NAMES:
        fields = _state.STORAGE_FIELDS[storage_name]
        annotations = {}
        for field_name in fields:
            numpy_dtype, inner_shape = fields[field_name]
            annotations[field_name] = _array_annotation(numpy_dtype, inner_shape)
        structures[storage_name] = _declare_structure(storage_structure_name(storage_name),
                                                      annotations)
    return structures


STORAGE_STRUCTURES = _storage_structures()


def _cloth_state_structure():
    annotations = {}
    for storage_name in _state.STORAGE_NAMES:
        annotations[storage_name] = STORAGE_STRUCTURES[storage_name]
    for index_name in _state.SPATIAL_INDEX_NAMES:
        annotations[spatial_index_member_name(index_name)] = wp.uint64
    return _declare_structure("ClothState", annotations)


ClothState = _cloth_state_structure()

MEMBER_NAMES = tuple(ClothState.vars)


def _assert_member_names():
    expected = tuple(_state.STORAGE_NAMES) \
        + tuple(spatial_index_member_name(index_name)
                for index_name in _state.SPATIAL_INDEX_NAMES)
    assert MEMBER_NAMES == expected, \
        "the device state carries one member per storage followed by one member per spatial " \
        "index, the state layer declares %r and the structure carries %r" \
        % (list(expected), list(MEMBER_NAMES))
    for storage_name in _state.STORAGE_NAMES:
        declared = tuple(_state.STORAGE_FIELDS[storage_name])
        carried = tuple(STORAGE_STRUCTURES[storage_name].vars)
        assert declared == carried, \
            "the device storage %s carries %r while the state layer declares %r" \
            % (storage_name, list(carried), list(declared))


_assert_member_names()


def build(state):
    instance = ClothState()
    for storage_name in _state.STORAGE_NAMES:
        storage = STORAGE_STRUCTURES[storage_name]()
        arrays = state.arrays(storage_name)
        for field_name in state.field_names(storage_name):
            setattr(storage, field_name, arrays[field_name])
        setattr(instance, storage_name, storage)
    for index_name in _state.SPATIAL_INDEX_NAMES:
        setattr(instance, spatial_index_member_name(index_name),
                wp.uint64(state.spatial_index_identifier(index_name)))
    return instance


class DeviceStateView:

    def __init__(self, state):
        self.state = state
        self.revision = None
        self.instance = None
        self.refresh()

    def refresh(self):
        self.instance = build(self.state)
        self.revision = self.state.revision
        return self.instance

    def current(self):
        if self.revision != self.state.revision:
            return self.refresh()
        return self.instance
