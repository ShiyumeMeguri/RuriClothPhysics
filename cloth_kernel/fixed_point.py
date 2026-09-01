import numpy as np

from . import defs

ACCUMULATOR_SCALAR_TYPE = np.int64

ACCUMULATOR_LIMIT = int(np.iinfo(ACCUMULATOR_SCALAR_TYPE).max)

ACCUMULATOR_SCALAR_TYPE_REASON = (
    "the width of the accumulator planes and the limit every bound below is measured against "
    "are the same fact, so the width is declared here and the limit is read off it; the "
    "plane table in program.py is refused at import if it declares any accumulator plane at "
    "another width, which is what stops the two from drifting apart")

GAP_ORDER_SCALE = 1.0e6

GAP_ORDER_LIMIT = 2147483520

GAP_ORDER_REASON = (
    "the self contact selection ranks candidates by a signed integer so the ordering never "
    "depends on a floating point comparison, and this scale is what one metre of gap is cut "
    "into on the way into that key; it is declared here rather than shared with the "
    "correction accumulators because the two answer different questions, the accumulator "
    "scale trades resolution against how large a sum of contributions may grow while this "
    "one trades resolution against how large a single gap may grow before it saturates, and "
    "the key stays a signed 32 bit value because it is compared and never summed; one part "
    "in a million of a metre separates two gaps that differ by a micrometre while the "
    "thickness this gap is measured against is between a millimetre and five centimetres, "
    "and the saturation bound is the largest signed 32 bit value that a float32 represents "
    "exactly, two thousand one hundred and forty seven metres of gap, so the truncation on "
    "the way in is lossless at the bound; a gap that is not a number is pinned to the bound "
    "so it ranks last")


def gap_order_key(gaps):
    values = np.asarray(gaps, dtype=np.float32)
    undefined = np.isnan(values)
    upper = np.float32(GAP_ORDER_LIMIT)
    lower = np.float32(-GAP_ORDER_LIMIT)
    defined = np.where(undefined, np.float32(0.0), values)
    with np.errstate(over="ignore"):
        scaled = defined * np.float32(GAP_ORDER_SCALE)
    keys = np.trunc(np.clip(scaled, lower, upper)).astype(np.int32)
    keys = np.where(undefined, np.int32(GAP_ORDER_LIMIT), keys)
    return np.ascontiguousarray(keys, dtype=np.int32)


MAGNITUDE_UNIT_VECTOR_COMPONENT = 1.0
MAGNITUDE_ONE_PER_CONTRIBUTION = 1.0
MAGNITUDE_ONE_CORRECTION_IN_METRES = 0.25

ACCUMULATION_WINDOW_SOURCE_REASON = (
    "the run of the schedule between one clear of a fixed point plane and the next is "
    "written in the kernel source and in the frame order, so the engine derives it there "
    "and hands it to this layer; the table that used to sit here named the phases of a "
    "reference implementation that no longer exists and had to be kept in step with the "
    "kernel source by hand")

CONTRIBUTOR_RULE_REASON = (
    "how many contributions one family lands on the worst particle is a property of the "
    "world being built, not of the kernel text, so it is the one thing here that is "
    "declared; the family it belongs to is named because a family is the unit the schedule "
    "runs, and a family that accumulates into a bounded plane without a rule is refused "
    "rather than counted as zero")

ACCUMULATOR_SPECIFICATION = (
    {"plane": "distance_correction_fixed",
     "scale": float(defs.TO_FIXED),
     "magnitude": MAGNITUDE_ONE_CORRECTION_IN_METRES,
     "magnitude_reason": "every contribution is the position correction in metres that one "
                         "constraint asks for inside one accumulation window, and a quarter "
                         "of a metre from a single constraint is already far outside what "
                         "this solver produces, so the budget sits there; a signed 64 bit "
                         "accumulator at this scale holds eight billion five hundred and "
                         "eighty nine million metres of summed correction, which is thirty "
                         "four billion same sign contributions of that size on one component "
                         "of one particle inside one window"},
    {"plane": "distance_count",
     "scale": 1.0,
     "magnitude": MAGNITUDE_ONE_PER_CONTRIBUTION,
     "magnitude_reason": "every contribution raises the count by one, so this magnitude is "
                         "exact rather than budgeted"},
    {"plane": "collision_normal_fixed",
     "scale": float(defs.TO_FIXED),
     "magnitude": MAGNITUDE_UNIT_VECTOR_COMPONENT,
     "magnitude_reason": "every contribution is one component of a normalised collision "
                         "normal, so this magnitude is exact rather than budgeted"},
)


SPILL_COUNTER_REASON = (
    "the counter counts candidates that the fixed slot budget of one query primitive could "
    "not keep, one increment per rejected candidate, so its sum over a frame is bounded by "
    "the number of primitive pairs the broad phase visited rather than by a magnitude; no "
    "solver value reads it, it is the diagnostic the coverage gate turns into a refusal")

NON_ACCUMULATING_ATOMIC_PLANES = {
    "self_counters": "the counter only hands out the next free index and every append is "
                     "clamped against the element count of the plane it appends into, so its "
                     "sum is bounded by the launch width rather than by a magnitude",
    "self_contact_overflow": SPILL_COUNTER_REASON,
    "self_intersect_overflow": SPILL_COUNTER_REASON,
}


CONTRIBUTION_STEP_REASON = (
    "one contribution costs this many integer steps and the bound of a world is that count "
    "multiplied by how many contributions the worst element takes, so the product of the "
    "magnitude and the scale has to be a whole number of steps; it is required to be whole "
    "rather than rounded because the bound is then an exact python integer compared against "
    "an exact limit, and a float64 cannot tell the largest signed 64 bit value apart from the "
    "power of two above it, so a floating point bound would accept the first world that "
    "overflows by one step")


def contribution_step(specification):
    step = specification["magnitude"] * specification["scale"]
    return int(step)


def _validate_specification():
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
        assert row["magnitude_reason"], \
            "the fixed point accumulator %s declares a magnitude without a reason" % plane_name
        step = row["magnitude"] * row["scale"]
        assert step == float(int(step)) and int(step) > 0, \
            "%s; the fixed point accumulator %s declares the magnitude %r at the scale %r, " \
            "which is %r steps" \
            % (CONTRIBUTION_STEP_REASON, plane_name, row["magnitude"], row["scale"], step)


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


SELF_CONTACT_REACH_REASON = (
    "one self collision primitive can pair with every primitive of its own team and every "
    "primitive of the largest team it is linked to, and the pair is counted once under each "
    "of the two primitives, so this is a bound on both roles a primitive plays in a contact; "
    "the kept contact slots of one query primitive are not used to tighten it, because the "
    "slot budget only caps the contacts a primitive keeps as the source and a primitive is "
    "hit again for every slot anywhere in the world whose target names it, so the bound "
    "through the slots is the degree multiplied by the slot count plus the whole slot table, "
    "which on a cloth of two thousand one hundred edges at the default forty eight slots is "
    "one hundred and one thousand contributions against the sixteen thousand eight hundred "
    "this rule reports; the loose rule is the tighter of the two at every degree a mesh "
    "actually has, so it stays")


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
    "solve_bending_constraint": _bending_pairs,
    "solve_collider_edge_contacts": _collision_edge_endpoints,
    "accumulate_self_edge_contacts": _self_edge_contacts,
    "accumulate_self_point_contacts": _self_point_contacts,
}

NO_CLEAR_WINDOW_LABEL = "no clear"


def window_label(window):
    return window["cleared_by"] or NO_CLEAR_WINDOW_LABEL


def _windows_of(windows, plane_name):
    return tuple(window for window in windows if window["plane"][1] == plane_name)


def assert_contributor_rules(windows):
    assert windows, \
        "%s; the caller handed this layer no accumulation window at all, so every bound " \
        "below would come out as zero contributions" % ACCUMULATION_WINDOW_SOURCE_REASON
    bounded = set(ACCUMULATOR_PLANE_NAMES)
    missing = []
    reached = set()
    for window in windows:
        if window["plane"][1] not in bounded:
            continue
        for family_name in window["accumulated_by"]:
            if family_name not in CONTRIBUTOR_RULES:
                missing.append("%s accumulates into %s.%s inside the window cleared by %s"
                               % (family_name, window["plane"][0], window["plane"][1],
                                  window_label(window)))
                continue
            reached.add(family_name)
    assert not missing, \
        "%s; these families have no contributor rule: %r" % (CONTRIBUTOR_RULE_REASON, missing)
    idle = sorted(set(CONTRIBUTOR_RULES) - reached)
    assert not idle, \
        "these contributor rules are declared and no family accumulates under them, so they " \
        "count nothing: %r" % (idle,)


def window_contributor_counts(program, world, windows):
    particle_count = program.num_particles
    counts = {}
    for window in windows:
        total = _zero_counts(particle_count)
        for family_name in window["accumulated_by"]:
            rule = CONTRIBUTOR_RULES.get(family_name)
            if rule is None:
                continue
            total = total + rule(program, world, particle_count)
        counts[id(window)] = total
    return counts


def accumulator_bounds(program, world, windows):
    per_window = window_contributor_counts(program, world, windows)
    rows = []
    for specification in ACCUMULATOR_SPECIFICATION:
        worst = 0
        element = -1
        worst_window = None
        window_worst = {}
        for window in _windows_of(windows, specification["plane"]):
            counts = per_window[id(window)]
            reached = int(counts.max()) if counts.size else 0
            window_worst[window_label(window)] = reached
            if reached > worst:
                worst = reached
                element = int(np.argmax(counts))
                worst_window = window_label(window)
        step = contribution_step(specification)
        bound = worst * step
        rows.append({"plane": specification["plane"],
                     "worst_contributors": worst,
                     "worst_element": element,
                     "worst_window": worst_window,
                     "per_window": window_worst,
                     "magnitude": specification["magnitude"],
                     "scale": specification["scale"],
                     "step": step,
                     "bound": bound,
                     "limit": ACCUMULATOR_LIMIT,
                     "headroom": (ACCUMULATOR_LIMIT - bound) / float(ACCUMULATOR_LIMIT)})
    return tuple(rows)


def assert_headroom(program, world, windows):
    rows = accumulator_bounds(program, world, windows)
    for row in rows:
        assert row["bound"] <= ACCUMULATOR_LIMIT, \
            "the fixed point accumulator %s takes at most %d same sign contributions on one " \
            "element inside the window cleared by %s, element %d, and each contribution is " \
            "at most %g before the scale %g, which is %d integer steps, so the worst same " \
            "sign sum is %d against the signed 64 bit limit %d; that sum wraps around " \
            "silently, so this world is refused at build time" \
            % (row["plane"], row["worst_contributors"], row["worst_window"],
               row["worst_element"], row["magnitude"], row["scale"], row["step"],
               row["bound"], ACCUMULATOR_LIMIT)
    return rows
