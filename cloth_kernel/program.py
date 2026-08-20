import numpy as np

I4 = np.int32
F4 = np.float32
F8 = np.float64
U1 = np.uint8

DERIVED_SOURCE_CSR_OFFSETS = "csr_offsets"
DERIVED_SOURCE_CSR_ORDER = "csr_order"
DERIVED_SOURCE_ATTRIBUTE = "attribute"
DERIVED_SOURCE_SCRATCH = "scratch"

DERIVED_SOURCE_KINDS = (DERIVED_SOURCE_CSR_OFFSETS, DERIVED_SOURCE_CSR_ORDER,
                        DERIVED_SOURCE_ATTRIBUTE, DERIVED_SOURCE_SCRATCH)

DERIVED_PLANE_SPECIFICATION = (
    ("distance_csr_offsets", DERIVED_SOURCE_CSR_OFFSETS, "distance_csr", I4, ()),
    ("distance_csr_order", DERIVED_SOURCE_CSR_ORDER, "distance_csr", I4, ()),
    ("point_pair_csr_offsets", DERIVED_SOURCE_CSR_OFFSETS, "point_pair_csr", I4, ()),
    ("point_pair_csr_order", DERIVED_SOURCE_CSR_ORDER, "point_pair_csr", I4, ()),
    ("edge_pair_csr_offsets", DERIVED_SOURCE_CSR_OFFSETS, "edge_pair_csr", I4, ()),
    ("edge_pair_csr_order", DERIVED_SOURCE_CSR_ORDER, "edge_pair_csr", I4, ()),
    ("center_fixed_csr_offsets", DERIVED_SOURCE_CSR_OFFSETS, "center_fixed_csr", I4, ()),
    ("center_fixed_csr_order", DERIVED_SOURCE_CSR_ORDER, "center_fixed_csr", I4, ()),
    ("v2t_csr_offsets", DERIVED_SOURCE_CSR_OFFSETS, "v2t_csr", I4, ()),
    ("v2t_csr_order", DERIVED_SOURCE_CSR_ORDER, "v2t_csr", I4, ()),
    ("fk_yes_offsets", DERIVED_SOURCE_ATTRIBUTE, "fk_yes_offsets", I4, ()),
    ("fk_yes", DERIVED_SOURCE_ATTRIBUTE, "fk_yes", I4, ()),
    ("fk_yes_parent", DERIVED_SOURCE_ATTRIBUTE, "fk_yes_parent", I4, ()),
    ("fk_no_offsets", DERIVED_SOURCE_ATTRIBUTE, "fk_no_offsets", I4, ()),
    ("fk_no", DERIVED_SOURCE_ATTRIBUTE, "fk_no", I4, ()),
    ("baseline_entries", DERIVED_SOURCE_ATTRIBUTE, "baseline_entries", I4, ()),
    ("angle_pass_offsets", DERIVED_SOURCE_ATTRIBUTE, "angle_pass_offsets", I4, ()),
    ("angle_pass_vertices", DERIVED_SOURCE_ATTRIBUTE, "angle_pass_vertices", I4, ()),
    ("angle_pass_parents", DERIVED_SOURCE_ATTRIBUTE, "angle_pass_parents", I4, ()),
    ("postline_entry_offsets", DERIVED_SOURCE_ATTRIBUTE, "postline_entry_offsets", I4, ()),
    ("postline_entry_vertices", DERIVED_SOURCE_ATTRIBUTE, "postline_entry_vertices", I4, ()),
    ("postline_child_offsets", DERIVED_SOURCE_ATTRIBUTE, "postline_child_offsets", I4, ()),
    ("postline_child_vertices", DERIVED_SOURCE_ATTRIBUTE, "postline_child_vertices", I4, ()),
    ("display_update_move_mask", DERIVED_SOURCE_ATTRIBUTE, "display_update_move_mask", U1, ()),
    ("distance_correction", DERIVED_SOURCE_SCRATCH, "num_particles", F4, (3,)),
    ("distance_correction_fixed", DERIVED_SOURCE_SCRATCH, "num_particles", I4, (3,)),
    ("distance_count", DERIVED_SOURCE_SCRATCH, "num_particles", I4, ()),
    ("collision_friction_fixed", DERIVED_SOURCE_SCRATCH, "num_particles", I4, ()),
    ("collision_normal_fixed", DERIVED_SOURCE_SCRATCH, "num_particles", I4, (3,)),
    ("synchronization_snapshot", DERIVED_SOURCE_SCRATCH, "num_teams", F4, (22,)),
    ("triangle_normal_double", DERIVED_SOURCE_SCRATCH, "num_triangle_entries", F8, (3,)),
    ("triangle_tangent_double", DERIVED_SOURCE_SCRATCH, "num_triangle_entries", F8, (3,)),
)


def _validate_derived_specification():
    seen = set()
    for row in DERIVED_PLANE_SPECIFICATION:
        assert len(row) == 5, \
            "a derived plane row declares name, source kind, source key, scalar type and " \
            "inner shape, got %r" % (row,)
        plane_name, source_kind, source_key, scalar_type, inner_shape = row
        assert isinstance(plane_name, str) and plane_name, \
            "a derived plane row must name the plane, got %r" % (plane_name,)
        assert plane_name not in seen, \
            "derived plane %s is declared twice" % plane_name
        seen.add(plane_name)
        assert source_kind in DERIVED_SOURCE_KINDS, \
            "derived plane %s declares source kind %r, only %r are defined" \
            % (plane_name, source_kind, DERIVED_SOURCE_KINDS)
        assert isinstance(source_key, str) and source_key, \
            "derived plane %s must name the program attribute it comes from" % plane_name
        assert isinstance(inner_shape, tuple), \
            "derived plane %s must declare its inner shape as a tuple, got %r" \
            % (plane_name, inner_shape)
        np.dtype(scalar_type)


_validate_derived_specification()

DERIVED_PLANE_NAMES = tuple(row[0] for row in DERIVED_PLANE_SPECIFICATION)

DERIVED_PLANE_FIELDS = {row[0]: (np.dtype(row[3]), row[4])
                        for row in DERIVED_PLANE_SPECIFICATION}


class CsrTable:

    __slots__ = ("offsets", "order", "num_keys")

    def __init__(self, offsets, order, num_keys):
        self.offsets = offsets
        self.order = order
        self.num_keys = num_keys

    @property
    def num_rows(self):
        return int(self.order.shape[0])


def build_csr(keys, num_keys):
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
    out = {}
    for name in field_names:
        source = arena_arrays[name][:count]
        if source.dtype == np.bool_:
            source = source.astype(np.uint8)
        out[name] = np.ascontiguousarray(source)
    return out


def scatter_fields(arena_arrays, flat, count):
    for name, values in flat.items():
        target = arena_arrays[name]
        if target.dtype == np.bool_:
            target[:count] = values.astype(np.bool_)
        else:
            target[:count] = values


class Program:

    def __init__(self):
        self.num_teams = 0
        self.num_particles = 0
        self.num_transforms = 0
        self.num_colliders = 0
        self.num_triangle_entries = 0

        self.num_self_points = 0
        self.num_self_edges = 0
        self.num_self_triangles = 0
        self.self_cap_ee = 1
        self.self_cap_pt = 1
        self.self_cap_ip = 1
        self.self_max_contact_tasks = 1
        self.self_max_intersect_tasks = 1

        self.distance_csr = None
        self.point_pair_csr = None
        self.edge_pair_csr = None
        self.postline_level_csr = []
        self.v2t_csr = None
        self.center_fixed_csr = None

        self.distance = {}
        self.bending = {}
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

        self.fk_levels = []
        self.angle_passes = []
        self.postline_levels = []

        self.postline_entry_offsets = None
        self.postline_entry_vertices = None
        self.postline_child_offsets = None
        self.postline_child_vertices = None

        self.display_update_move_mask = None

        self.angle_pass_offsets = None
        self.angle_pass_vertices = None
        self.angle_pass_parents = None

        self.fk_yes_offsets = None
        self.fk_yes = None
        self.fk_yes_parent = None
        self.fk_no_offsets = None
        self.fk_no = None


def _arena_dump(arena, fields):
    live = np.flatnonzero(arena["team"] != 0)
    return {name: np.ascontiguousarray(arena[name][live]) for name in fields}, live


def _compacted_rows(arena, live):
    lookup = np.full(int(arena["team"].shape[0]), -1, dtype=np.int64)
    lookup[live] = np.arange(live.shape[0], dtype=np.int64)
    return lookup


def _retarget(values, lookup, label):
    if values.shape[0] == 0:
        return values
    mapped = lookup[values]
    assert int(mapped.min()) >= 0, "%s points at a released arena row" % label
    return np.ascontiguousarray(mapped, dtype=I4)


def _live_extent(team_rows, start_field, count_field):
    extent = team_rows[start_field] + team_rows[count_field]
    return int(extent.max()) if extent.shape[0] else 0


def build_program(world):
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

    program.distance, _ = _arena_dump(world.distance, ("team", "particle", "target", "rest"))
    program.bending, _ = _arena_dump(world.bending, ("team", "pair", "rest", "sign"))
    program.tether, _ = _arena_dump(world.tether, ("team", "particle"))
    program.motion, _ = _arena_dump(world.motion, ("team", "particle"))
    program.update_move, _ = _arena_dump(world.update_move, ("team", "particle"))
    program.update_fixed, _ = _arena_dump(world.update_fixed, ("team", "particle"))
    program.spring, _ = _arena_dump(world.spring, ("team", "particle"))
    program.collision_process, _ = _arena_dump(world.collision_process, ("team", "particle"))
    program.collision_edges, edge_live = _arena_dump(world.collision_edges, ("team", "edge"))
    program.edges, _ = _arena_dump(world.edges, ("team", "edge"))
    program.triangles, triangle_live = _arena_dump(world.triangles, ("team", "triangle"))
    program.v2t, _ = _arena_dump(world.v2t,
                                 ("team", "owner", "triangle", "flip_normal", "flip_tangent"))
    program.v2t["triangle"] = _retarget(program.v2t["triangle"],
                                        _compacted_rows(world.triangles, triangle_live),
                                        "v2t.triangle")
    program.num_triangle_entries = int(program.triangles["team"].shape[0])
    program.point_pairs, _ = _arena_dump(world.point_pairs, ("team", "particle", "collider"))
    program.edge_pairs, _ = _arena_dump(world.edge_pairs, ("team", "edge", "collider"))
    program.edge_pairs["edge"] = _retarget(program.edge_pairs["edge"],
                                           _compacted_rows(world.collision_edges, edge_live),
                                           "edge_pairs.edge")
    program.center_fixed, _ = _arena_dump(world.center_fixed, ("team", "particle"))
    program.angle_buffered, _ = _arena_dump(world.angle_buffered, ("team", "particle"))
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
    _assert_derived_specification_covers(program)
    return program


def _derived_scratch_plane(program, source_key, scalar_type, inner_shape):
    element_count = getattr(program, source_key)
    assert isinstance(element_count, int), \
        "scratch plane source %s must name an integer program count, got %r" \
        % (source_key, element_count)
    return np.zeros((element_count,) + inner_shape, dtype=scalar_type)


def _derived_plane_values(program, source_kind, source_key, scalar_type, inner_shape):
    if source_kind == DERIVED_SOURCE_SCRATCH:
        return _derived_scratch_plane(program, source_key, scalar_type, inner_shape)
    if source_kind == DERIVED_SOURCE_CSR_OFFSETS:
        return getattr(program, source_key).offsets
    if source_kind == DERIVED_SOURCE_CSR_ORDER:
        return getattr(program, source_key).order
    return getattr(program, source_key)


def derived_planes(program):
    planes = {}
    for plane_name, source_kind, source_key, scalar_type, inner_shape in \
            DERIVED_PLANE_SPECIFICATION:
        values = np.ascontiguousarray(
            _derived_plane_values(program, source_kind, source_key, scalar_type, inner_shape))
        assert values.dtype == np.dtype(scalar_type), \
            "derived plane %s is declared as %s but the program produced %s" \
            % (plane_name, np.dtype(scalar_type), values.dtype)
        assert values.shape[1:] == inner_shape, \
            "derived plane %s is declared with inner shape %r but the program produced %r" \
            % (plane_name, inner_shape, values.shape[1:])
        planes[plane_name] = values
    return planes


def derived_plane_counts(program):
    return {plane_name: int(values.shape[0])
            for plane_name, values in derived_planes(program).items()}


def _assert_derived_specification_covers(program):
    declared_csr = set()
    declared_attributes = set()
    for _plane_name, source_kind, source_key, _scalar_type, _inner_shape in \
            DERIVED_PLANE_SPECIFICATION:
        if source_kind in (DERIVED_SOURCE_CSR_OFFSETS, DERIVED_SOURCE_CSR_ORDER):
            declared_csr.add(source_key)
        elif source_kind == DERIVED_SOURCE_ATTRIBUTE:
            declared_attributes.add(source_key)
    for attribute_name, attribute_value in vars(program).items():
        if isinstance(attribute_value, CsrTable):
            assert attribute_name in declared_csr, \
                "the program builds the compressed row table %s but no derived plane row " \
                "declares it, every derived table needs a plane declaration" % attribute_name
        elif isinstance(attribute_value, np.ndarray):
            assert attribute_name in declared_attributes, \
                "the program builds the array %s but no derived plane row declares it, " \
                "every derived table needs a plane declaration" % attribute_name


def _compute_self_capacities(program, world):
    tt = world.team
    nt = program.num_teams
    se = tt["se_count"][:nt].astype(np.int64)
    sp = tt["sp_count"][:nt].astype(np.int64)
    st = tt["st_count"][:nt].astype(np.int64)
    max_se = int(se.max()) if nt else 0
    max_sp = int(sp.max()) if nt else 0
    max_st = int(st.max()) if nt else 0
    cap_ee = int((se * se).sum() + (se * max_se).sum())
    cap_pt = int((sp * st).sum() + (sp * max_st).sum() + (st * max_sp).sum())
    cap_ip = int((se * st).sum() + (se * max_st).sum() + (st * max_se).sum())
    program.self_cap_ee = max(cap_ee, 1)
    program.self_cap_pt = max(cap_pt, 1)
    program.self_cap_ip = max(cap_ip, 1)
    program.self_max_contact_tasks = max(nt * 5, 1)
    program.self_max_intersect_tasks = max(nt * 3, 1)


def _flatten_postline(levels, level_csr):
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
        child_offset_parts.append((csr.offsets[1:entries + 1].astype(I4) + I4(base)))
        base += int(csr.offsets[entries]) if entries else 0
    entry_vertices = np.concatenate(entry_parts).astype(I4) if entry_parts else np.zeros(0, dtype=I4)
    child_vertices = np.concatenate(child_parts).astype(I4) if child_parts else np.zeros(0, dtype=I4)
    child_offsets = np.concatenate(child_offset_parts).astype(I4)
    return (np.ascontiguousarray(entry_offsets), np.ascontiguousarray(entry_vertices),
            np.ascontiguousarray(child_offsets), np.ascontiguousarray(child_vertices))


def _build_update_move_mask(move_particles, num_particles):
    mask = np.zeros(max(num_particles, 0), dtype=np.uint8)
    if move_particles.shape[0]:
        mask[move_particles] = 1
    return np.ascontiguousarray(mask)


def _flatten_levels(levels):
    offsets = np.zeros(len(levels) + 1, dtype=I4)
    parts = []
    for i, arr in enumerate(levels):
        a = np.ascontiguousarray(arr, I4)
        parts.append(a)
        offsets[i + 1] = offsets[i] + int(a.shape[0])
    values = np.concatenate(parts).astype(I4) if parts else np.zeros(0, dtype=I4)
    return offsets, np.ascontiguousarray(values)


def _build_edge_pair_csr(program):
    edge_key = program.edge_pairs["edge"]
    num_edge_entries = int(program.collision_edges["edge"].shape[0])
    return build_csr(edge_key, num_edge_entries)
