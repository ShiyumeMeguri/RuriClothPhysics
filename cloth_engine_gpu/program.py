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
        self.num_triangle_entries = 0

        # G3a self-collision: live primitive-arena extents + worst-case device table capacities
        # (sized from the structural primitive counts, INDEPENDENT of the runtime self_mode/sync
        # config so a config change -- config is not in the reload signature -- never overflows a
        # too-small table). Bounds assume every team could run FULL_MESH self AND sync to the team
        # with the most primitives.
        self.num_self_points = 0
        self.num_self_edges = 0
        self.num_self_triangles = 0
        self.self_cap_ee = 1
        self.self_cap_pt = 1
        self.self_cap_ip = 1
        self.self_max_contact_tasks = 1
        self.self_max_intersect_tasks = 1

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

        # flattened postline (display._postline): a device loop over range(entry_offsets-1) walks
        # levels with a grid.sync wall between them; per-entry children are a global CSR keyed by the
        # flat entry index (child_offsets[i]:child_offsets[i+1] into child_vertices, owner-grouped).
        self.postline_entry_offsets = None
        self.postline_entry_vertices = None
        self.postline_child_offsets = None
        self.postline_child_vertices = None

        # per-particle update_move mask (materialised from the update_move table): the display
        # _display split is move (mask=1) vs fixed (mask=0), which partition all particles.
        self.display_update_move_mask = None

        # flattened angle (level,rank) passes (offsets[P+1] + concatenated vertices/parents),
        # so a device loop over ``range(offsets.shape[0]-1)`` walks passes in sorted (level,rank)
        # order with a grid.sync wall between them (Gauss-Seidel: parent albuf_rotation of a
        # level-L vertex was finalised in an earlier pass at level < L this iteration).
        self.angle_pass_offsets = None
        self.angle_pass_vertices = None
        self.angle_pass_parents = None

        # flattened FK level tables (offsets[num_levels+1] + concatenated values), so a
        # device loop over ``range(offsets.shape[0]-1)`` walks levels with a grid.sync
        # wall between them (parent step_basic of level L was written at level < L).
        self.fk_yes_offsets = None
        self.fk_yes = None
        self.fk_yes_parent = None
        self.fk_no_offsets = None
        self.fk_no = None


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
    program.num_self_points = _live_extent(world.team, "sp_start", "sp_count")
    program.num_self_edges = _live_extent(world.team, "se_start", "se_count")
    program.num_self_triangles = _live_extent(world.team, "st_start", "st_count")
    _compute_self_capacities(program, world)

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
    # display._post_triangles indexes tri_normal/tangent scratch by GLOBAL triangle-arena row; the
    # dumped live rows form a contiguous prefix [0, num) that v2t['triangle'] addresses directly.
    program.num_triangle_entries = int(program.triangles["team"].shape[0])
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
    program.fk_yes_offsets, program.fk_yes = _flatten_levels([lv[0] for lv in program.fk_levels])
    _, program.fk_yes_parent = _flatten_levels([lv[1] for lv in program.fk_levels])
    program.fk_no_offsets, program.fk_no = _flatten_levels([lv[2] for lv in program.fk_levels])
    program.angle_passes = [(np.ascontiguousarray(v, I4), np.ascontiguousarray(p, I4))
                            for v, p in world.angle_passes]
    program.angle_pass_offsets, program.angle_pass_vertices = \
        _flatten_levels([v for v, _p in program.angle_passes])
    _, program.angle_pass_parents = _flatten_levels([p for _v, p in program.angle_passes])
    program.postline_levels = [(np.ascontiguousarray(ev, I4), np.ascontiguousarray(co, I4),
                                np.ascontiguousarray(cv, I4)) for ev, co, cv in world.postline_levels]
    program.postline_level_csr = [build_csr(co, int(ev.shape[0]))
                                  for ev, co, cv in program.postline_levels]
    (program.postline_entry_offsets, program.postline_entry_vertices,
     program.postline_child_offsets, program.postline_child_vertices) = \
        _flatten_postline(program.postline_levels, program.postline_level_csr)
    program.display_update_move_mask = _build_update_move_mask(
        program.update_move["particle"], n_particles)
    return program


def _compute_self_capacities(program, world):
    """Worst-case device table capacities for self-collision (D3). Sized from structural per-team
    primitive counts, assuming every team could run FULL_MESH self collision AND sync to the team
    with the most primitives -- a safe upper bound independent of the runtime self_mode/sync_target
    (which are per-frame/config, not in the reload signature). The exact per-frame candidate-pair
    count (and the >3e7 FAIL guard, D1) is computed host-side each frame from the actual task set."""
    tt = world.team
    nt = program.num_teams
    se = tt["se_count"][:nt].astype(np.int64)
    sp = tt["sp_count"][:nt].astype(np.int64)
    st = tt["st_count"][:nt].astype(np.int64)
    max_se = int(se.max()) if nt else 0
    max_sp = int(sp.max()) if nt else 0
    max_st = int(st.max()) if nt else 0
    # EE: self se^2 + sync se*max_partner_se. PT: self sp*st + sync PT (sp*max_st) + sync TP (st*max_sp).
    # IP (intersect edge-triangle): self se*st + sync se*max_st + sync max_se*st.
    cap_ee = int((se * se).sum() + (se * max_se).sum())
    cap_pt = int((sp * st).sum() + (sp * max_st).sum() + (st * max_sp).sum())
    cap_ip = int((se * st).sum() + (se * max_st).sum() + (st * max_se).sum())
    program.self_cap_ee = max(cap_ee, 1)
    program.self_cap_pt = max(cap_pt, 1)
    program.self_cap_ip = max(cap_ip, 1)
    # task-slot bounds: per team up to 2 self tasks (EE + PT) + 3 sync tasks (EE + TP + PT);
    # intersect up to 1 self + 2 sync per team.
    program.self_max_contact_tasks = max(nt * 5, 1)
    program.self_max_intersect_tasks = max(nt * 3, 1)


def _flatten_postline(levels, level_csr):
    """Flatten per-level postline into (entry_offsets[L+1], entry_vertices, child_offsets[E+1],
    child_vertices). ``entry_offsets`` slices each level's entries out of the concatenated
    ``entry_vertices``; ``child_offsets`` is a GLOBAL per-entry CSR (indexed by flat entry row)
    into ``child_vertices``, which is owner-grouped (csr.order) so each entry's children are the
    contiguous run the oracle iterates."""
    num_levels = len(levels)
    entry_offsets = np.zeros(num_levels + 1, dtype=I4)
    entry_parts = []
    child_parts = []
    child_offset_parts = [np.zeros(1, dtype=I4)]
    base = 0
    for lvl, ((ev, co, cv), csr) in enumerate(zip(levels, level_csr)):
        entries = int(ev.shape[0])
        entry_parts.append(np.ascontiguousarray(ev, I4))
        entry_offsets[lvl + 1] = entry_offsets[lvl] + entries
        order = csr.order
        child_parts.append(cv[order] if order.shape[0] else np.zeros(0, dtype=I4))
        # csr.offsets has length entries+1; global per-entry offsets = per-level offsets + base.
        child_offset_parts.append((csr.offsets[1:entries + 1].astype(I4) + I4(base)))
        base += int(csr.offsets[entries]) if entries else 0
    entry_vertices = np.concatenate(entry_parts).astype(I4) if entry_parts else np.zeros(0, dtype=I4)
    child_vertices = np.concatenate(child_parts).astype(I4) if child_parts else np.zeros(0, dtype=I4)
    child_offsets = np.concatenate(child_offset_parts).astype(I4)
    return (np.ascontiguousarray(entry_offsets), np.ascontiguousarray(entry_vertices),
            np.ascontiguousarray(child_offsets), np.ascontiguousarray(child_vertices))


def _build_update_move_mask(move_particles, num_particles):
    """Per-particle uint8 mask: 1 where the particle is an update_move entry, 0 otherwise
    (update_move / update_fixed partition all particles, so the complement is update_fixed)."""
    mask = np.zeros(max(num_particles, 0), dtype=np.uint8)
    if move_particles.shape[0]:
        mask[move_particles] = 1
    return np.ascontiguousarray(mask)


def _flatten_levels(levels):
    """Concatenate a list of per-level int32 index arrays into (offsets, values), where
    ``offsets[i]:offsets[i+1]`` selects level i's rows in ``values``."""
    offsets = np.zeros(len(levels) + 1, dtype=I4)
    parts = []
    for i, arr in enumerate(levels):
        a = np.ascontiguousarray(arr, I4)
        parts.append(a)
        offsets[i + 1] = offsets[i] + int(a.shape[0])
    values = np.concatenate(parts).astype(I4) if parts else np.zeros(0, dtype=I4)
    return offsets, np.ascontiguousarray(values)


def _build_edge_pair_csr(program):
    # collision_edges live rows form a contiguous prefix, so edge-pair 'edge' values are
    # absolute indices in [0, num_edge_entries); group edge-pair rows by their edge entry.
    edge_key = program.edge_pairs["edge"]
    num_edge_entries = int(program.collision_edges["edge"].shape[0])
    return build_csr(edge_key, num_edge_entries)
