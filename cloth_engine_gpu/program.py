"""Build the static device Program from a CPU ``World``.

The oracle's numpy stages are the CPU projection of a data-parallel job graph
(mirroring the MC2 HLSL decomposition). This module turns a ``World`` into the
static tables that job graph needs, once, at ``engine.load`` time:

* CSR groupings that preserve the oracle's *summation order* per accumulator, so a
  per-particle gather reproduces the oracle's ``np.add.reduceat`` result bit-for-bit
  (distance / collider-point / collider-edge / postline / v2t / center-fixed).
* the level / pass tables (FK levels, angle (level,rank) passes, postline levels,
  baseline entries) harvested straight from ``world.ensure_buckets()`` -- single
  source of truth, no re-derivation.
* the static per-team index sets already materialised in the arenas.

Field flattening (struct-of-arrays -> flat contiguous device arrays) is provided by
``flatten_fields`` / ``scatter_fields``; the kernels declare exactly which fields
they touch and the engine flattens only those (see ``engine.py``).

Nothing here is GPU-specific: it produces numpy arrays. ``device.py`` uploads them.
All index arrays are ``int32`` C-contiguous; offsets are ``int32``.
"""

import numpy as np

I4 = np.int32


class CsrTable:
    """Compressed grouping: rows of ``order`` in ``[offsets[k], offsets[k+1])`` are the
    original row indices whose key == k, in stable (original) order."""

    __slots__ = ("offsets", "order", "num_keys")

    def __init__(self, offsets, order, num_keys):
        self.offsets = offsets
        self.order = order
        self.num_keys = num_keys

    @property
    def num_rows(self):
        return int(self.order.shape[0])


def build_csr(keys, num_keys):
    """Group row indices by integer key, preserving original order within a key.

    Reproduces the oracle's ``run_starts`` / ``reduceat`` grouping: the oracle relies
    on rows for one accumulator being *consecutive* in arena order; a stable sort by
    key yields exactly that ordering (identical for the already-consecutive case and
    well-defined when a scatter needs a global regroup).
    """
    keys = np.ascontiguousarray(keys, dtype=np.int64)
    n = int(keys.shape[0])
    offsets = np.zeros(num_keys + 1, dtype=I4)
    if n == 0:
        return CsrTable(offsets, np.zeros(0, dtype=I4), num_keys)
    counts = np.bincount(keys, minlength=num_keys).astype(np.int64)
    offsets[1:] = np.cumsum(counts).astype(I4)
    order = np.argsort(keys, kind="stable").astype(I4)
    return CsrTable(offsets, order, num_keys)


def flatten_fields(arena_arrays, field_names, count):
    """Flatten selected struct fields of an arena into C-contiguous arrays.

    ``arena_arrays`` maps field name -> ndarray of shape ``(capacity, *shape)``. Only
    the first ``count`` rows (the live prefix the kernels address by absolute index)
    are taken. Returns ``{name: contiguous ndarray}`` with the field dtype preserved
    (bool -> uint8 for device friendliness)."""
    out = {}
    for name in field_names:
        source = arena_arrays[name][:count]
        if source.dtype == np.bool_:
            source = source.astype(np.uint8)
        out[name] = np.ascontiguousarray(source)
    return out


def scatter_fields(arena_arrays, flat, count):
    """Write flat arrays back into the arena struct fields (readback path)."""
    for name, values in flat.items():
        target = arena_arrays[name]
        if target.dtype == np.bool_:
            target[:count] = values.astype(np.bool_)
        else:
            target[:count] = values


class Program:
    """Static tables for one ``World`` instance (built once at engine.load)."""

    def __init__(self):
        self.num_teams = 0
        self.num_particles = 0
        self.num_transforms = 0
        self.num_colliders = 0

        # per-accumulator CSR groupings (summation-order preserving)
        self.distance_csr = None          # key = distance particle (global)
        self.point_pair_csr = None        # key = point-pair particle (global)
        self.edge_pair_csr = None         # key = edge-pair edge entry index
        self.postline_level_csr = []      # per level: CSR keyed by owner slot
        self.v2t_csr = None               # key = v2t owner (global particle)
        self.center_fixed_csr = None      # key = center_fixed team

        # flat static index / topology arrays (arena order)
        self.distance = {}                # particle/target/rest/team
        self.bending = {}                 # pair(4)/rest/sign/team
        self.tether = {}
        self.motion = {}
        self.update_move = {}
        self.update_fixed = {}
        self.spring = {}
        self.collision_process = {}
        self.collision_edges = {}
        self.edges = {}
        self.triangles = {}
        self.v2t = {}
        self.point_pairs = {}
        self.edge_pairs = {}
        self.center_fixed = {}
        self.angle_buffered = {}
        self.baseline_entries = None

        # level / pass tables (from ensure_buckets)
        self.fk_levels = []               # list of (yes, yes_parent, no)
        self.angle_passes = []            # list of (vertices, parents)
        self.postline_levels = []         # list of (entry_vertex, child_owner, child_vertex)


def _arena_dump(arena, fields):
    """Dump only live rows (team != 0). Free/unallocated arena rows carry team slot 0
    (the permanent sentinel team, never registered) and would otherwise pollute the
    per-particle CSR gathers with zero-valued rows aliased to particle 0. Live rows are
    a contiguous prefix in a sequentially-built world; the boolean mask preserves their
    order regardless, and an explicit prefix assertion guards the absolute-index contract."""
    team = arena["team"]
    live = np.flatnonzero(team != 0)
    if live.shape[0]:
        assert live[-1] - live[0] == live.shape[0] - 1 and live[0] == 0, \
            "live arena rows must be a contiguous prefix for absolute indexing"
    return {name: np.ascontiguousarray(arena[name][live]) for name in fields}


def _live_extent(team_rows, start_field, count_field):
    extent = team_rows[start_field] + team_rows[count_field]
    return int(extent.max()) if extent.shape[0] else 0


def build_program(world):
    """Construct the static Program from a fully-registered World."""
    world.ensure_buckets()
    program = Program()
    program.num_teams = int(len(world.team))
    program.num_particles = _live_extent(world.team, "p_start", "p_count")
    program.num_transforms = _live_extent(world.team, "t_start", "t_count")
    program.num_colliders = _live_extent(world.team, "c_start", "c_count")

    program.distance = _arena_dump(world.distance, ("team", "particle", "target", "rest"))
    program.bending = _arena_dump(world.bending, ("team", "pair", "rest", "sign"))
    program.tether = _arena_dump(world.tether, ("team", "particle"))
    program.motion = _arena_dump(world.motion, ("team", "particle"))
    program.update_move = _arena_dump(world.update_move, ("team", "particle"))
    program.update_fixed = _arena_dump(world.update_fixed, ("team", "particle"))
    program.spring = _arena_dump(world.spring, ("team", "particle"))
    program.collision_process = _arena_dump(world.collision_process, ("team", "particle"))
    program.collision_edges = _arena_dump(world.collision_edges, ("team", "edge"))
    program.edges = _arena_dump(world.edges, ("team", "edge"))
    program.triangles = _arena_dump(world.triangles, ("team", "triangle"))
    program.v2t = _arena_dump(world.v2t, ("team", "owner", "triangle", "flip_normal", "flip_tangent"))
    program.point_pairs = _arena_dump(world.point_pairs, ("team", "particle", "collider"))
    program.edge_pairs = _arena_dump(world.edge_pairs, ("team", "edge", "collider"))
    program.center_fixed = _arena_dump(world.center_fixed, ("team", "particle"))
    program.angle_buffered = _arena_dump(world.angle_buffered, ("team", "particle"))
    program.baseline_entries = np.ascontiguousarray(world.baseline_entries)

    n_particles = program.num_particles
    program.distance_csr = build_csr(program.distance["particle"], n_particles)
    program.point_pair_csr = build_csr(program.point_pairs["particle"], n_particles)
    program.edge_pair_csr = _build_edge_pair_csr(program)
    program.v2t_csr = build_csr(program.v2t["owner"], n_particles)
    program.center_fixed_csr = build_csr(program.center_fixed["team"], program.num_teams)

    program.fk_levels = [(np.ascontiguousarray(a, I4), np.ascontiguousarray(b, I4),
                          np.ascontiguousarray(c, I4)) for a, b, c in world.fk_levels]
    program.angle_passes = [(np.ascontiguousarray(v, I4), np.ascontiguousarray(p, I4))
                            for v, p in world.angle_passes]
    program.postline_levels = [(np.ascontiguousarray(ev, I4), np.ascontiguousarray(co, I4),
                                np.ascontiguousarray(cv, I4)) for ev, co, cv in world.postline_levels]
    program.postline_level_csr = [build_csr(co, int(ev.shape[0]))
                                  for ev, co, cv in program.postline_levels]
    return program


def _build_edge_pair_csr(program):
    # collision_edges live rows form a contiguous prefix, so edge-pair 'edge' values are
    # absolute indices in [0, num_edge_entries); group edge-pair rows by their edge entry.
    edge_key = program.edge_pairs["edge"]
    num_edge_entries = int(program.collision_edges["edge"].shape[0])
    return build_csr(edge_key, num_edge_entries)
