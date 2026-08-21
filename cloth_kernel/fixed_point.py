import numpy as np

from . import defs

SIGNED_32_BIT_LIMIT = 2 ** 31 - 1

MAGNITUDE_UNIT_VECTOR_COMPONENT = 1.0
MAGNITUDE_ONE_PER_CONTRIBUTION = 1.0
MAGNITUDE_ONE_CORRECTION_IN_METRES = 0.25

CONTRIBUTOR_BENDING_PAIRS = "bending_pairs"
CONTRIBUTOR_COLLISION_EDGE_ENDPOINTS = "collision_edge_endpoints"
CONTRIBUTOR_SELF_EDGE_CONTACTS = "self_edge_contacts"
CONTRIBUTOR_SELF_POINT_CONTACTS = "self_point_contacts"

EVERY_CONTRIBUTOR_RULE = (CONTRIBUTOR_BENDING_PAIRS, CONTRIBUTOR_COLLISION_EDGE_ENDPOINTS,
                          CONTRIBUTOR_SELF_EDGE_CONTACTS, CONTRIBUTOR_SELF_POINT_CONTACTS)

WINDOW_BENDING = "bending"
WINDOW_COLLIDER_EDGES = "collider edges"
WINDOW_SELF_COLLISION = "self collision"

ACCUMULATION_WINDOW_SPECIFICATION = (
    {"window": WINDOW_BENDING, "cleared_by": "phase_21", "accumulated_by": ("phase_22",),
     "contributors": (CONTRIBUTOR_BENDING_PAIRS,)},
    {"window": WINDOW_COLLIDER_EDGES, "cleared_by": "phase_25",
     "accumulated_by": ("phase_26",),
     "contributors": (CONTRIBUTOR_COLLISION_EDGE_ENDPOINTS,)},
    {"window": WINDOW_SELF_COLLISION, "cleared_by": "phase_39",
     "accumulated_by": ("phase_40",),
     "contributors": (CONTRIBUTOR_SELF_EDGE_CONTACTS, CONTRIBUTOR_SELF_POINT_CONTACTS)},
)

EVERY_WINDOW = tuple(row["window"] for row in ACCUMULATION_WINDOW_SPECIFICATION)

ACCUMULATOR_SPECIFICATION = (
    {"plane": "distance_correction_fixed",
     "scale": float(defs.TO_FIXED),
     "magnitude": MAGNITUDE_ONE_CORRECTION_IN_METRES,
     "windows": EVERY_WINDOW,
     "magnitude_reason": "a signed 32 bit accumulator at a scale of one part in a million "
                         "holds 2147.48 metres of summed correction, and every contribution "
                         "is the position correction in metres that one constraint asks for "
                         "inside one accumulation window; a quarter of a metre from a single "
                         "constraint is already far outside what this solver produces, so the "
                         "budget sits there and the build refuses any world whose worst "
                         "particle could take more than 8589 same sign contributions inside "
                         "one window"},
    {"plane": "distance_count",
     "scale": 1.0,
     "magnitude": MAGNITUDE_ONE_PER_CONTRIBUTION,
     "windows": EVERY_WINDOW,
     "magnitude_reason": "every contribution raises the count by one, so this magnitude is "
                         "exact rather than budgeted"},
    {"plane": "collision_normal_fixed",
     "scale": float(defs.TO_FIXED),
     "magnitude": MAGNITUDE_UNIT_VECTOR_COMPONENT,
     "windows": (WINDOW_COLLIDER_EDGES,),
     "magnitude_reason": "every contribution is one component of a normalised collision "
                         "normal, so this magnitude is exact rather than budgeted"},
)


NON_ACCUMULATING_ATOMIC_PLANES = {
    "self_counters": "the counter only hands out the next free index and every append is "
                     "clamped against the element count of the plane it appends into, so its "
                     "sum is bounded by the launch width rather than by a magnitude",
}


def _validate_specification():
    seen_windows = set()
    covered_rules = set()
    for row in ACCUMULATION_WINDOW_SPECIFICATION:
        window_name = row["window"]
        assert window_name not in seen_windows, \
            "the accumulation window %s is declared twice" % window_name
        seen_windows.add(window_name)
        assert row["cleared_by"] and row["accumulated_by"], \
            "the accumulation window %s declares the phase that clears it and the phases " \
            "that accumulate into it" % window_name
        for rule_name in row["contributors"]:
            assert rule_name in EVERY_CONTRIBUTOR_RULE, \
                "the accumulation window %s declares the contributor rule %r, only %r are " \
                "defined" % (window_name, rule_name, EVERY_CONTRIBUTOR_RULE)
            covered_rules.add(rule_name)
    assert covered_rules == set(EVERY_CONTRIBUTOR_RULE), \
        "every contributor rule belongs to one accumulation window, the windows cover %r and " \
        "the rules are %r" % (sorted(covered_rules), sorted(EVERY_CONTRIBUTOR_RULE))
    seen = set()
    for row in ACCUMULATOR_SPECIFICATION:
        plane_name = row["plane"]
        assert plane_name not in seen, \
            "the fixed point accumulator %s is declared twice" % plane_name
        seen.add(plane_name)
        assert row["scale"] > 0.0, \
            "the fixed point accumulator %s declares the scale %r" % (plane_name, row["scale"])
        assert row["magnitude"] > 0.0, \
            "the fixed point accumulator %s declares the magnitude %r" \
            % (plane_name, row["magnitude"])
        assert row["windows"], \
            "the fixed point accumulator %s declares no accumulation window" % plane_name
        for window_name in row["windows"]:
            assert window_name in seen_windows, \
                "the fixed point accumulator %s declares the accumulation window %r, only %r " \
                "are defined" % (plane_name, window_name, EVERY_WINDOW)
        assert row["magnitude_reason"], \
            "the fixed point accumulator %s declares a magnitude without a reason" % plane_name


_validate_specification()

ACCUMULATOR_PLANE_NAMES = tuple(row["plane"] for row in ACCUMULATOR_SPECIFICATION)


def _zero_counts(particle_count):
    return np.zeros(max(int(particle_count), 0), dtype=np.int64)


def _incidence(counts, indices, weights):
    for column in range(indices.shape[1]):
        column_indices = indices[:, column]
        live = column_indices >= 0
        if not np.any(live):
            continue
        np.add.at(counts, column_indices[live], weights[live])


def _bending_pairs(program, world, particle_count):
    counts = _zero_counts(particle_count)
    pairs = program.bending.get("pair")
    if pairs is None or pairs.shape[0] == 0:
        return counts
    indices = np.ascontiguousarray(pairs, dtype=np.int64)
    _incidence(counts, indices, np.ones(indices.shape[0], dtype=np.int64))
    return counts


def _collision_edge_endpoints(program, world, particle_count):
    counts = _zero_counts(particle_count)
    edges = program.collision_edges.get("edge")
    if edges is None or edges.shape[0] == 0:
        return counts
    indices = np.ascontiguousarray(edges, dtype=np.int64)
    _incidence(counts, indices, np.ones(indices.shape[0], dtype=np.int64))
    return counts


def _team_counts(world, count_field, team_count):
    values = world.team[count_field][:team_count].astype(np.int64)
    return values, int(values.max()) if team_count else 0


def _primitive_rows(world, arena_name):
    arena = getattr(world, arena_name)
    teams = np.ascontiguousarray(arena["team"], dtype=np.int64)
    live = np.flatnonzero(teams != 0)
    particles = np.ascontiguousarray(arena["particles"][live], dtype=np.int64)
    return teams[live], particles


def _paired_reach(own_counts, largest, teams):
    return own_counts[teams] + largest


def _self_edge_contacts(program, world, particle_count):
    counts = _zero_counts(particle_count)
    team_count = program.num_teams
    if team_count == 0:
        return counts
    edge_counts, largest_edges = _team_counts(world, "se_count", team_count)
    teams, particles = _primitive_rows(world, "self_edges")
    if teams.shape[0] == 0:
        return counts
    reach = _paired_reach(edge_counts, largest_edges, teams)
    _incidence(counts, particles, reach)
    return counts


def _self_point_contacts(program, world, particle_count):
    counts = _zero_counts(particle_count)
    team_count = program.num_teams
    if team_count == 0:
        return counts
    point_counts, largest_points = _team_counts(world, "sp_count", team_count)
    triangle_counts, largest_triangles = _team_counts(world, "st_count", team_count)
    point_teams, point_particles = _primitive_rows(world, "self_points")
    if point_teams.shape[0]:
        _incidence(counts, point_particles,
                   _paired_reach(triangle_counts, largest_triangles, point_teams))
    triangle_teams, triangle_particles = _primitive_rows(world, "self_triangles")
    if triangle_teams.shape[0]:
        _incidence(counts, triangle_particles,
                   _paired_reach(point_counts, largest_points, triangle_teams))
    return counts


CONTRIBUTOR_RULES = {
    CONTRIBUTOR_BENDING_PAIRS: _bending_pairs,
    CONTRIBUTOR_COLLISION_EDGE_ENDPOINTS: _collision_edge_endpoints,
    CONTRIBUTOR_SELF_EDGE_CONTACTS: _self_edge_contacts,
    CONTRIBUTOR_SELF_POINT_CONTACTS: _self_point_contacts,
}

assert set(CONTRIBUTOR_RULES) == set(EVERY_CONTRIBUTOR_RULE), \
    "every declared contributor rule needs an implementation, the table declares %r and the " \
    "implementations cover %r" % (sorted(EVERY_CONTRIBUTOR_RULE), sorted(CONTRIBUTOR_RULES))


def window_contributor_counts(program, world):
    particle_count = program.num_particles
    counts = {}
    for row in ACCUMULATION_WINDOW_SPECIFICATION:
        total = _zero_counts(particle_count)
        for rule_name in row["contributors"]:
            total = total + CONTRIBUTOR_RULES[rule_name](program, world, particle_count)
        counts[row["window"]] = total
    return counts


def accumulator_bounds(program, world):
    per_window = window_contributor_counts(program, world)
    rows = []
    for specification in ACCUMULATOR_SPECIFICATION:
        worst = 0
        element = -1
        worst_window = None
        window_worst = {}
        for window_name in specification["windows"]:
            counts = per_window[window_name]
            reached = int(counts.max()) if counts.size else 0
            window_worst[window_name] = reached
            if reached > worst:
                worst = reached
                element = int(np.argmax(counts))
                worst_window = window_name
        bound = float(worst) * specification["magnitude"] * specification["scale"]
        rows.append({"plane": specification["plane"],
                     "worst_contributors": worst,
                     "worst_element": element,
                     "worst_window": worst_window,
                     "per_window": window_worst,
                     "magnitude": specification["magnitude"],
                     "scale": specification["scale"],
                     "bound": bound,
                     "limit": float(SIGNED_32_BIT_LIMIT),
                     "headroom": (float(SIGNED_32_BIT_LIMIT) - bound)
                     / float(SIGNED_32_BIT_LIMIT)})
    return tuple(rows)


def assert_headroom(program, world):
    rows = accumulator_bounds(program, world)
    for row in rows:
        assert row["bound"] <= float(SIGNED_32_BIT_LIMIT), \
            "the fixed point accumulator %s takes at most %d same sign contributions on one " \
            "element inside the %s window, element %d, and each contribution is at most %g " \
            "before the scale %g, so the worst same sign sum is %.6g against the signed 32 " \
            "bit limit %d; that sum wraps around silently, so this world is refused at build " \
            "time" % (row["plane"], row["worst_contributors"], row["worst_window"],
                      row["worst_element"], row["magnitude"], row["scale"], row["bound"],
                      SIGNED_32_BIT_LIMIT)
    return rows
