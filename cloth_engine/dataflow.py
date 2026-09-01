from ..cloth_kernel import fixed_point as _fixed_point
from ..cloth_kernel import program as _program
from . import effects as _effects
from . import plan as _plan
from . import schedule as _schedule
from . import state as _state

FRAME_ORDER = _schedule.SCHEDULED_FAMILY_NAMES

FAMILY_EFFECTS = _effects.family_effects(_schedule.FAMILY_NAMES)

FAMILY_SUMMING_PLANES = _effects.family_summing_planes(_schedule.FAMILY_NAMES)

FAMILY_REDUCING_PLANES = _effects.family_reducing_planes(_schedule.FAMILY_NAMES)

SCRATCH_PLANE_NAMES = frozenset(
    row[0] for row in _program.DERIVED_PLANE_SPECIFICATION
    if row[1] == _program.DERIVED_SOURCE_SCRATCH)

DERIVED_STORAGE_NAME = "derived"

LIVENESS_REASON = (
    "a plane that a family writes and nothing ever reads is either a computation whose "
    "result is thrown away or a signal whose consumer was never written, and the second one "
    "is silent: the device raises a flag, the host never looks, and the user is handed a "
    "wrong answer with no indication that anything went wrong; the reader may be another "
    "family or the host, because a plane that leaves the device has left the frame")

INITIALISATION_REASON = (
    "a scratch plane holds whatever the previous frame left in it, so a family that reads "
    "one before any family or the host has written it this frame is reading stale state, "
    "which makes the frame depend on how many frames ran before it; the tables that the "
    "program fills once at load are a different kind of plane and are excluded by their "
    "declared source kind rather than by a name list")

ACCUMULATION_WINDOW_REASON = (
    "an accumulation window is the run of the schedule between one clear of a fixed point "
    "plane and the next, so it is a property of the kernel source and the frame order and "
    "never a table somebody maintains beside them; the overflow bound multiplies the worst "
    "number of contributions inside one window by the magnitude of one contribution")

HAZARD_REASON = (
    "two families have to keep their relative order when one writes a plane the other "
    "touches, in either direction; families with no such plane in common are independent of "
    "each other, and the longest chain of ordered families is the shortest the frame could "
    "ever be if every independent family ran at the same time")


def _entries(family_name, effect_kinds):
    row = FAMILY_EFFECTS[family_name]
    collected = set()
    for effect_kind in effect_kinds:
        collected |= row[effect_kind]
    return collected


def read_entries(family_name):
    return _entries(family_name, (_effects.EFFECT_READ,))


def written_entries(family_name):
    return _entries(family_name, _effects.MUTATING_EFFECT_KINDS)


def _plane_of(entry):
    return entry[0]


def frame_writers():
    writers = {}
    for family_name in FRAME_ORDER:
        for entry in written_entries(family_name):
            writers.setdefault(entry, []).append(family_name)
    return writers


def frame_readers():
    readers = {}
    for family_name in FRAME_ORDER:
        for entry in read_entries(family_name):
            readers.setdefault(entry, []).append(family_name)
    return readers


FRAME_WRITERS = frame_writers()

FRAME_READERS = frame_readers()


def device_written_planes():
    return frozenset(_plane_of(entry) for entry in FRAME_WRITERS)


DEVICE_WRITTEN_PLANES = device_written_planes()


def dead_writes(host_read_planes):
    held = frozenset(host_read_planes)
    rows = []
    for entry in sorted(FRAME_WRITERS, key=str):
        if entry in FRAME_READERS or _plane_of(entry) in held:
            continue
        rows.append((entry, tuple(sorted(set(FRAME_WRITERS[entry])))))
    return tuple(rows)


def device_writes_that_only_leave_on_release(frame_read_planes, release_read_planes):
    frame_held = frozenset(frame_read_planes)
    release_held = frozenset(release_read_planes)
    rows = []
    for entry in sorted(FRAME_WRITERS, key=str):
        if entry in FRAME_READERS or _plane_of(entry) in frame_held:
            continue
        if _plane_of(entry) not in release_held:
            continue
        rows.append((entry, tuple(sorted(set(FRAME_WRITERS[entry])))))
    return tuple(rows)


def uninitialised_reads(host_written_planes):
    held = frozenset(host_written_planes)
    produced = set()
    rows = []
    for position, family_name in enumerate(FRAME_ORDER):
        own = written_entries(family_name)
        for entry in sorted(read_entries(family_name), key=str):
            storage_name, field_name = _plane_of(entry)
            if storage_name != DERIVED_STORAGE_NAME or field_name not in SCRATCH_PLANE_NAMES:
                continue
            if entry in produced or entry in own or _plane_of(entry) in held:
                continue
            rows.append((position, family_name, entry))
        produced |= own
    return tuple(rows)


def _clearing_families(plane):
    clearing = set()
    for family_name in _schedule.FAMILY_NAMES:
        for entry in FAMILY_EFFECTS[family_name][_effects.EFFECT_CLEAR]:
            if _plane_of(entry) == plane:
                clearing.add(family_name)
    return clearing


def accumulation_windows():
    planes = set()
    for family_name in FRAME_ORDER:
        planes |= FAMILY_SUMMING_PLANES[family_name]
    windows = []
    for plane in sorted(planes):
        clearing = _clearing_families(plane)
        current = None
        for family_name in FRAME_ORDER:
            if family_name in clearing:
                current = {"plane": plane, "cleared_by": family_name, "accumulated_by": []}
                windows.append(current)
            if plane not in FAMILY_SUMMING_PLANES[family_name]:
                continue
            if current is None:
                current = {"plane": plane, "cleared_by": None, "accumulated_by": []}
                windows.append(current)
            if family_name not in current["accumulated_by"]:
                current["accumulated_by"].append(family_name)
    return tuple({"plane": window["plane"], "cleared_by": window["cleared_by"],
                  "accumulated_by": tuple(window["accumulated_by"])} for window in windows)


ACCUMULATION_WINDOWS = accumulation_windows()


def summing_planes():
    planes = set()
    for family_name in FRAME_ORDER:
        planes |= FAMILY_SUMMING_PLANES[family_name]
    return tuple(sorted(planes))


SUMMING_PLANES = summing_planes()


def reducing_planes():
    planes = set()
    for family_name in FRAME_ORDER:
        planes |= FAMILY_REDUCING_PLANES[family_name]
    return tuple(sorted(planes))


REDUCING_PLANES = reducing_planes()


def _assert_summing_planes_are_bounded():
    bounded = set(_fixed_point.ACCUMULATOR_PLANE_NAMES)
    exempt = set(_fixed_point.NON_ACCUMULATING_ATOMIC_PLANES)
    unbounded = []
    for storage_name, field_name in SUMMING_PLANES:
        if field_name in bounded or field_name in exempt:
            continue
        unbounded.append("%s.%s" % (storage_name, field_name))
    assert not unbounded, \
        "these planes take a summing atomic from a scheduled family and no row of the fixed " \
        "point accumulator table bounds them and no exemption names them, so their sum can " \
        "wrap around silently: %r" % (unbounded,)
    idle = sorted(bounded - {field_name for _storage_name, field_name in SUMMING_PLANES})
    assert not idle, \
        "the fixed point accumulator table bounds %r and no scheduled family accumulates " \
        "into them, so the bound is guarding nothing" % (idle,)


_assert_summing_planes_are_bounded()

_fixed_point.assert_contributor_rules(ACCUMULATION_WINDOWS)


def hazard_edges():
    rows = []
    for position, family_name in enumerate(FRAME_ORDER):
        rows.append((position, family_name, read_entries(family_name),
                     written_entries(family_name)))
    edges = set()
    for first_position, _first_name, first_reads, first_writes in rows:
        for second_position, _second_name, second_reads, second_writes in rows:
            if second_position <= first_position:
                continue
            if first_writes & (second_reads | second_writes) \
                    or first_reads & second_writes:
                edges.add((first_position, second_position))
    return tuple(sorted(edges))


def chain_report():
    edges = hazard_edges()
    row_count = len(FRAME_ORDER)
    depth = [0] * row_count
    predecessors = {position: [] for position in range(row_count)}
    for first_position, second_position in edges:
        predecessors[second_position].append(first_position)
    for position in range(row_count):
        for earlier in predecessors[position]:
            depth[position] = max(depth[position], depth[earlier] + 1)
    levels = {}
    for position in range(row_count):
        levels.setdefault(depth[position], []).append(position)
    parent = list(range(row_count))

    def find(position):
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = parent[position]
        return position

    for first_position, second_position in edges:
        first_root = find(first_position)
        second_root = find(second_position)
        if first_root != second_root:
            parent[first_root] = second_root
    components = {}
    for position in range(row_count):
        components.setdefault(find(position), []).append(position)
    return {"rows": row_count, "edges": len(edges),
            "longest_chain": max(depth) + 1 if row_count else 0,
            "components": tuple(tuple(members) for members in
                                sorted(components.values(), key=lambda members: members[0])),
            "levels": tuple((level, tuple(levels[level])) for level in sorted(levels)),
            "depth": tuple(depth)}


SPATIAL_INDEX_TOKEN_STORAGE = "spatial_index"

SPATIAL_INDEX_TOKEN_REASON = (
    "a spatial index is a handle a refit node rebuilds and the queries that walk it read, "
    "and neither the refit nor the query touches a state plane the other does, so the order "
    "between them is invisible to a plane only hazard; the refit is modelled as writing a "
    "token in this private namespace and every family that queries the index as reading it, "
    "which turns build before query into an ordinary write before read edge in the same "
    "graph the planes drive; the namespace is a word no storage is named after, so a token "
    "never collides with a real plane")

ENTRY_LEVEL_REASON = (
    "the level of an entry is the length of the longest chain of hazards that has to finish "
    "before it can run, the same depth chain_report takes over the static family order but "
    "taken over the expanded entry sequence a frame actually records; two entries carry the "
    "same level exactly when no chain of ordered work separates them, so a level is a set of "
    "entries a device may run at once, and the same family recorded twice in one frame lands "
    "on two levels because its whole effect set collides with itself and forces the order")

LEVEL_CONFLICT_REASON = (
    "the device capture records every entry of a level onto a rotating stream with one fork "
    "and one join around the level, which is sound only if no two entries in a level touch a "
    "plane one of them writes; the levelling makes that true by construction, because any "
    "such pair carries a hazard edge and an edge forces the deeper entry a level down, and "
    "this gate proves it pairwise before the capture trusts it")


def _refit_read_set(index_name):
    return frozenset((plane, None) for plane in _state.SPATIAL_INDEX_BOUND_PLANES
                     if plane[0] == index_name)


def _entry_effect_sets(descriptors):
    reads = []
    writes = []
    for kind, name in descriptors:
        if kind == _plan.DESCRIPTOR_LAUNCH:
            entry_reads = set(read_entries(name))
            for index_name in _schedule.QUERIED_SPATIAL_INDEXES[name + "_element"]:
                entry_reads.add(((SPATIAL_INDEX_TOKEN_STORAGE, index_name), None))
            reads.append(frozenset(entry_reads))
            writes.append(frozenset(written_entries(name)))
            continue
        reads.append(_refit_read_set(name))
        writes.append(frozenset({((SPATIAL_INDEX_TOKEN_STORAGE, name), None)}))
    return reads, writes


def _entry_edges_and_depth(descriptors):
    reads, writes = _entry_effect_sets(descriptors)
    count = len(descriptors)
    depth = [0] * count
    edge_count = 0
    for second in range(count):
        second_read = reads[second]
        second_write = writes[second]
        best = 0
        for first in range(second):
            if writes[first] & (second_read | second_write) or reads[first] & second_write:
                edge_count += 1
                candidate = depth[first] + 1
                if candidate > best:
                    best = candidate
        depth[second] = best
    return depth, edge_count


def entry_levels(descriptors):
    depth, _edge_count = _entry_edges_and_depth(descriptors)
    return depth


def entry_level_report(descriptors):
    depth, edge_count = _entry_edges_and_depth(descriptors)
    count = len(depth)
    longest = max(depth) + 1 if count else 0
    widths = [0] * longest
    for value in depth:
        widths[value] += 1
    return {"entries": count, "edges": edge_count, "longest_chain": longest,
            "level_widths": tuple(widths)}


def assert_levels_are_conflict_free(descriptors, levels):
    assert len(levels) == len(descriptors), \
        "%s\n%d levels were assigned for %d entries" \
        % (LEVEL_CONFLICT_REASON, len(levels), len(descriptors))
    reads, writes = _entry_effect_sets(descriptors)
    by_level = {}
    for position, level in enumerate(levels):
        by_level.setdefault(level, []).append(position)
    conflicts = []
    for level, members in by_level.items():
        for outer in range(len(members)):
            first = members[outer]
            for inner in range(outer + 1, len(members)):
                second = members[inner]
                if writes[first] & (reads[second] | writes[second]) \
                        or writes[second] & (reads[first] | writes[first]):
                    conflicts.append((level, first, second))
    assert not conflicts, \
        "%s\nthese entries share a level yet one writes a plane the other touches:\n%s" \
        % (LEVEL_CONFLICT_REASON,
           "\n".join("  level %d entries %d and %d" % row for row in conflicts[:20]))


def effect_table():
    table = {}
    for family_name in _schedule.FAMILY_NAMES:
        row = FAMILY_EFFECTS[family_name]
        table[family_name] = {
            effect_kind: tuple(sorted("%s.%s%s" % (entry[0][0], entry[0][1],
                                                   "" if entry[1] is None else "[%s]" % entry[1])
                                      for entry in row[effect_kind]))
            for effect_kind in _effects.EFFECT_KINDS}
    return table


SPATIAL_INDEX_READER_REASON = (
    "the bounds and the group column a spatial index is built on are read by the index "
    "itself, inside the refit node the frame records, and not by any kernel that names "
    "them; a family that writes them therefore has a reader and is not a dead write, and "
    "the reader is named from the same table the index is declared in rather than from a "
    "list beside it")

SPATIAL_INDEX_READ_PLANES = _state.SPATIAL_INDEX_BOUND_PLANES


def assert_frame_boundary(host_written_planes, host_frame_read_planes,
                          host_release_read_planes):
    dead = dead_writes(tuple(host_frame_read_planes) + tuple(host_release_read_planes)
                       + tuple(SPATIAL_INDEX_READ_PLANES))
    assert not dead, \
        "%s\nthese planes are written by a scheduled family and read by no family and by no " \
        "host download:\n%s" \
        % (LIVENESS_REASON,
           "\n".join("  %s.%s%s written by %s"
                     % (entry[0][0], entry[0][1],
                        "" if entry[1] is None else "[%s]" % entry[1], ", ".join(writers))
                     for entry, writers in dead))
    stale = uninitialised_reads(host_written_planes)
    assert not stale, \
        "%s\nthese scratch planes are read before anything writes them in the frame:\n%s" \
        % (INITIALISATION_REASON,
           "\n".join("  schedule row %d %s reads %s.%s%s"
                     % (position, family_name, entry[0][0], entry[0][1],
                        "" if entry[1] is None else "[%s]" % entry[1])
                     for position, family_name, entry in stale))
