from .kernels import RESIDENT_BLOB_LAYOUT, ZONE_BLOB_LAYOUT

__all__ = []

for _table in (RESIDENT_BLOB_LAYOUT, ZONE_BLOB_LAYOUT):
    for _index, _entry in enumerate(_table):
        _slot = 'S_' + _entry[0]
        assert _slot not in globals(), _slot
        globals()[_slot] = _index
        __all__.append(_slot)
