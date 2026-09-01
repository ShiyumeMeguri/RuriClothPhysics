import numpy as np

from . import frame as _frame

SELF_TASK_KIND_EDGE_EDGE = 0
SELF_TASK_KIND_POINT_TRIANGLE = 1

CONTACT_TASK_COLUMNS = ("kind", "source_team", "source_start", "source_count",
                        "target_team", "target_start", "target_count", "same_team")

INTERSECT_TASK_COLUMNS = ("edge_team", "edge_start", "edge_count",
                          "triangle_team", "triangle_start", "triangle_count", "same_team")

SAME_TEAM = 1
DIFFERENT_TEAM = 0

def frame_team_mask(team, team_count):
    scale_alive = _frame.component_scale_is_alive(team["component_world_scale"][:team_count])
    return team["enabled"][:team_count] & team["valid"][:team_count] & scale_alive


def structural_team_mask(team, team_count):
    return np.ascontiguousarray(team["valid"][:team_count])


def link_fingerprint(contact_links, team_count):
    parts = []
    for source in range(team_count):
        targets = contact_links.get(source, ())
        parts.append(int(source).to_bytes(4, "little"))
        parts.append(len(targets).to_bytes(4, "little"))
        for target in targets:
            parts.append(int(target).to_bytes(4, "little"))
    return b"".join(parts)


def _primitive_span(row, start_field, count_field):
    return int(row[start_field]), int(row[count_field])


def _accepted_targets(source, targets, target_mask):
    accepted = []
    diagonal_seen = False
    for target in targets:
        slot = int(target)
        if slot == source:
            assert not diagonal_seen, \
                "team %d declares the diagonal contact link twice, a team collides with " \
                "itself at most once" % source
            diagonal_seen = True
            accepted.append(slot)
            continue
        if slot <= 0 or slot >= target_mask.shape[0] or not target_mask[slot]:
            continue
        accepted.append(slot)
    return accepted


def _expand_diagonal(team, source, contact, intersect, use_point, use_edge, use_triangle):
    row = team[source]
    edge_start, edge_count = _primitive_span(row, "se_start", "se_count")
    triangle_start, triangle_count = _primitive_span(row, "st_start", "st_count")
    point_start, point_count = _primitive_span(row, "sp_start", "sp_count")
    if edge_count > 0:
        use_edge[source] = 1
        contact.append((SELF_TASK_KIND_EDGE_EDGE, source, edge_start, edge_count,
                        source, edge_start, edge_count, SAME_TEAM))
    if triangle_count > 0:
        use_point[source] = 1
        use_triangle[source] = 1
        contact.append((SELF_TASK_KIND_POINT_TRIANGLE, source, point_start, point_count,
                        source, triangle_start, triangle_count, SAME_TEAM))
    if edge_count > 0 and triangle_count > 0:
        intersect.append((source, edge_start, edge_count, source, triangle_start,
                          triangle_count, SAME_TEAM))


def _expand_pair(team, source, target, contact, intersect, use_point, use_edge, use_triangle):
    row = team[source]
    partner_row = team[target]
    edge_start, edge_count = _primitive_span(row, "se_start", "se_count")
    triangle_start, triangle_count = _primitive_span(row, "st_start", "st_count")
    point_start, point_count = _primitive_span(row, "sp_start", "sp_count")
    partner_edge_start, partner_edge_count = _primitive_span(partner_row, "se_start", "se_count")
    partner_triangle_start, partner_triangle_count = _primitive_span(
        partner_row, "st_start", "st_count")
    partner_point_start, partner_point_count = _primitive_span(
        partner_row, "sp_start", "sp_count")
    if edge_count > 0 and partner_edge_count > 0:
        use_edge[source] = 1
        use_edge[target] = 1
        contact.append((SELF_TASK_KIND_EDGE_EDGE, source, edge_start, edge_count,
                        target, partner_edge_start, partner_edge_count, DIFFERENT_TEAM))
    if triangle_count > 0:
        use_triangle[source] = 1
        use_point[target] = 1
        contact.append((SELF_TASK_KIND_POINT_TRIANGLE, target, partner_point_start,
                        partner_point_count, source, triangle_start, triangle_count,
                        DIFFERENT_TEAM))
    if partner_triangle_count > 0:
        use_point[source] = 1
        use_triangle[target] = 1
        contact.append((SELF_TASK_KIND_POINT_TRIANGLE, source, point_start, point_count,
                        target, partner_triangle_start, partner_triangle_count,
                        DIFFERENT_TEAM))
    if edge_count > 0 and partner_triangle_count > 0:
        intersect.append((source, edge_start, edge_count, target, partner_triangle_start,
                          partner_triangle_count, DIFFERENT_TEAM))
    if triangle_count > 0 and partner_edge_count > 0:
        intersect.append((target, partner_edge_start, partner_edge_count, source,
                          triangle_start, triangle_count, DIFFERENT_TEAM))


def expand_tasks(team, team_count, contact_links, source_mask, target_mask):
    use_point = np.zeros(team_count, np.uint8)
    use_edge = np.zeros(team_count, np.uint8)
    use_triangle = np.zeros(team_count, np.uint8)
    contact = []
    intersect = []
    for source in np.flatnonzero(source_mask):
        slot = int(source)
        for target in _accepted_targets(slot, contact_links.get(slot, ()), target_mask):
            if target == slot:
                _expand_diagonal(team, slot, contact, intersect, use_point, use_edge,
                                 use_triangle)
                continue
            _expand_pair(team, slot, target, contact, intersect, use_point, use_edge,
                         use_triangle)
    return contact, intersect, use_point, use_edge, use_triangle


def linked_team_mask(team, team_count):
    return team["valid"][:team_count] & team["enabled"][:team_count]


def build_tasks(team, team_count, contact_links):
    return expand_tasks(team, team_count, contact_links,
                        frame_team_mask(team, team_count),
                        linked_team_mask(team, team_count))


def maximal_tasks(team, team_count, contact_links):
    structural_mask = structural_team_mask(team, team_count)
    return expand_tasks(team, team_count, contact_links, structural_mask, structural_mask)
