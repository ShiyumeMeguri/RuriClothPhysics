import ast
import io

from . import families as _families
from . import plan as _plan
from . import state as _state

SCHEDULE_LOOP = "loop"
SCHEDULE_IF = "if"
SCHEDULE_ELSE = "else"

SCHEDULE_KINDS = (SCHEDULE_LOOP, SCHEDULE_IF, SCHEDULE_ELSE)

LOOP_COUNT_FROM_FLAG = "flag"
LOOP_COUNT_FROM_OFFSET_PLANES = "offset_planes"

LOOP_COUNT_RULES = (LOOP_COUNT_FROM_FLAG, LOOP_COUNT_FROM_OFFSET_PLANES)

FLAG_SUBSTEP_COUNT = "substep_count"
FLAG_SELF_ITERATION_COUNT = "self_iteration_count"
FLAG_CONTACT_PAIR_COUNT = "contact_pair_count"
FLAG_INTERSECT_PAIR_COUNT = "intersect_pair_count"

FLAG_NAMES = (FLAG_SUBSTEP_COUNT, FLAG_SELF_ITERATION_COUNT, FLAG_CONTACT_PAIR_COUNT,
              FLAG_INTERSECT_PAIR_COUNT)

PREDICATE_CONTACT_PAIRS_PRESENT = "contact_pairs_present"
PREDICATE_INTERSECT_PAIRS_PRESENT = "intersect_pairs_present"
PREDICATE_FIRST_SUBSTEP = "first_substep"

SUBSTEP_SCALAR = "substep"
LEVEL_SCALAR = "level"
ITERATION_SCALAR = "iteration"
SELF_ITERATION_SCALAR = "self_iteration"

KERNEL_SCALAR_NAMES = (SUBSTEP_SCALAR, LEVEL_SCALAR, ITERATION_SCALAR)

LOOP_SCALAR_NAMES = KERNEL_SCALAR_NAMES + (SELF_ITERATION_SCALAR,)

KERNEL_SIGNATURE = ("state",) + KERNEL_SCALAR_NAMES

SPATIAL_INDEX_MEMBER_SUFFIX = "_index"

FAMILY_SPECIFICATION_WIDTH = 4

SUBSTEP_LOOP = (SCHEDULE_LOOP, SUBSTEP_SCALAR, LOOP_COUNT_FROM_FLAG, FLAG_SUBSTEP_COUNT)

SELF_ITERATION_LOOP = (SCHEDULE_LOOP, SELF_ITERATION_SCALAR, LOOP_COUNT_FROM_FLAG,
                       FLAG_SELF_ITERATION_COUNT)

CONTACT_PAIRS_PRESENT = (SCHEDULE_IF, PREDICATE_CONTACT_PAIRS_PRESENT)

INTERSECT_PAIRS_PRESENT = (SCHEDULE_IF, PREDICATE_INTERSECT_PAIRS_PRESENT)

FIRST_SUBSTEP = (SCHEDULE_IF, PREDICATE_FIRST_SUBSTEP)

LATER_SUBSTEP = (SCHEDULE_ELSE, PREDICATE_FIRST_SUBSTEP)


ANGLE_ROOT_GROUP_REASON = (
    "the angle constraint reads and writes only a vertex and its parent, and a vertex and "
    "its parent lie on one chain so they share one root; grouping the pass entries by that "
    "root therefore splits them into sets whose touched vertices are pairwise disjoint, and a "
    "stable sort by root holds each group in the level major pass order it had before, so one "
    "thread walks a whole root group through the three iterations in that order and no two "
    "threads ever touch the same vertex, which is why the reorder needs no barrier and stays "
    "bit identical; this retires the launch storm that gave one launch to each of the eleven "
    "passes of each of the three iterations of each of the three substeps, ninety nine "
    "launches of a hundred and sixty nine wide grid for a family measured at thirty nine "
    "point seven percent of the frame device time, and leaves one launch per substep, three "
    "in all, each as wide as the number of root groups plus one")


STATIC_CHAIN_HOIST_REASON = (
    "the static chain pass reads and writes only the step basic rotation of its own vertex and "
    "reaches no parent and no other entry, so the family carries no cross entry dependency and "
    "borrows its level structure only from the animated side it was scheduled beside; the "
    "compiler splits each chain level into a moving bucket that has a parent and a static "
    "bucket that does not, so the animated pass writes only moving vertices and the static "
    "pass writes only static ones and the two vertex sets never meet, which means the animated "
    "pass can never produce a value the static pass reads; the one dependency that does exist "
    "runs the other way, since an animated vertex at a level reads the step basic rotation of "
    "its parent and that parent can be a static vertex one level up, so the static write has "
    "to land before the animated read, which the original level interleaving already gave and "
    "which hoisting the whole static pass ahead of every animated level only strengthens; with "
    "no entry depending on another the static pass collapses from one launch per chain level of "
    "each substep, thirty launches, to one launch per substep, three in all, each as wide as "
    "the whole static bucket, and the result stays bit identical")


FAMILY_SPECIFICATION_TABLE = (
    ("resolve_team_synchronization_top", _families.resolve_team_synchronization_top,
        (("team", "enabled"),), ()),
    ("snapshot_team_synchronization_clock", _families.snapshot_team_synchronization_clock,
        (("team", "enabled"),), ()),
    ("apply_team_synchronization_clock", _families.apply_team_synchronization_clock,
        (("team", "enabled"),), ()),
    ("advance_team_frame_clock", _families.advance_team_frame_clock,
        (("team", "enabled"),), ()),
    ("skin_particle_pose", _families.skin_particle_pose, (("particle", "team"),), ()),
    ("resolve_team_negative_scale", _families.resolve_team_negative_scale,
        (("team", "enabled"),), ()),
    ("resolve_team_frame_pose", _families.resolve_team_frame_pose,
        (("team", "enabled"),), ()),
    ("advance_team_component_inertia", _families.advance_team_component_inertia,
        (("team", "enabled"),), ()),
    ("resolve_team_world_inertia", _families.resolve_team_world_inertia,
        (("team", "enabled"),), ()),
    ("resolve_particle_wind_zones", _families.resolve_particle_wind_zones,
        (("particle", "team"),), ()),
    ("sample_team_wind_zones", _families.sample_team_wind_zones, (("team", "enabled"),), ()),
    ("reset_particle_frame_state", _families.reset_particle_frame_state,
        (("particle", "team"),), ()),
    ("prepare_collider_frame_pose", _families.prepare_collider_frame_pose,
        (("collider", "team"),), ()),
    ("update_collider_face_primitives", _families.update_collider_face_primitives,
        (("collider_faces", "team"),), ()),
    ("update_collider_vertex_pseudo_normals", _families.update_collider_vertex_pseudo_normals,
        (("collider_vertices", "team"),), ("collider_faces",)),
    ("clear_intersect_pair_counter", _families.clear_intersect_pair_counter,
        (("derived", "self_counters"),), ()),
    ("gather_intersect_pairs", _families.gather_intersect_pairs,
        (("derived", "self_intersect_query_slots"),), ("self_triangles",)),
    ("advance_team_substep_motion", _families.advance_team_substep_motion,
        (("team", "enabled"),), ()),
    ("interpolate_collider_substep_pose", _families.interpolate_collider_substep_pose,
        (("collider", "team"),), ()),
    ("animate_particle_base_pose", _families.animate_particle_base_pose,
        (("particle", "team"),), ()),
    ("integrate_particle_motion", _families.integrate_particle_motion,
        (("update_move", "particle"),), ()),
    ("pin_fixed_particles", _families.pin_fixed_particles, (("update_fixed", "particle"),), ()),
    ("pin_spring_particles", _families.pin_spring_particles, (("spring", "particle"),), ()),
    ("propagate_animated_chain_pose", _families.propagate_animated_chain_pose,
        (("derived", "fk_yes_root_offsets"),), ()),
    ("propagate_static_chain_pose", _families.propagate_static_chain_pose,
        (("derived", "fk_no"),), ()),
    ("blend_baseline_chain_pose", _families.blend_baseline_chain_pose,
        (("derived", "baseline_entries"),), ()),
    ("solve_tether_constraint", _families.solve_tether_constraint,
        (("tether", "particle"),), ()),
    ("gather_distance_correction", _families.gather_distance_correction,
        (("particle", "team"),), ()),
    ("apply_distance_correction", _families.apply_distance_correction,
        (("particle", "team"),), ()),
    ("buffer_baseline_angle_state", _families.buffer_baseline_angle_state,
        (("derived", "baseline_entries"),), ()),
    ("buffer_carried_angle_state", _families.buffer_carried_angle_state,
        (("angle_buffered", "particle"),), ()),
    ("solve_angle_constraint", _families.solve_angle_constraint,
        (("derived", "angle_root_offsets"),), ()),
    ("clear_distance_accumulator", _families.clear_distance_accumulator,
        (("particle", "team"),), ()),
    ("solve_bending_constraint", _families.solve_bending_constraint,
        (("bending", "team"),), ()),
    ("apply_distance_accumulator", _families.apply_distance_accumulator,
        (("particle", "team"),), ()),
    ("measure_collider_point_contacts", _families.measure_collider_point_contacts,
        (("point_pairs", "collider"),), ()),
    ("measure_collider_edge_feet", _families.measure_collider_edge_feet,
        (("edge_pairs", "collider"),), ()),
    ("measure_collider_edge_contacts", _families.measure_collider_edge_contacts,
        (("edge_pairs", "collider"),), ()),
    ("apply_collider_spring_response", _families.apply_collider_spring_response,
        (("point_pairs", "collider"),), ()),
    ("gather_collider_point_contacts", _families.gather_collider_point_contacts,
        (("particle", "team"),), ()),
    ("resolve_collider_point_contacts", _families.resolve_collider_point_contacts,
        (("particle", "team"),), ()),
    ("clear_collision_accumulator", _families.clear_collision_accumulator,
        (("particle", "team"),), ()),
    ("solve_collider_edge_contacts", _families.solve_collider_edge_contacts,
        (("collision_edges", "edge"),), ()),
    ("apply_collision_accumulator", _families.apply_collision_accumulator,
        (("particle", "team"),), ()),
    ("solve_motion_constraint", _families.solve_motion_constraint,
        (("motion", "particle"),), ()),
    ("clear_self_primitive_size", _families.clear_self_primitive_size,
        (("team", "enabled"),), ()),
    ("update_self_point_primitives", _families.update_self_point_primitives,
        (("self_points", "team"),), ()),
    ("update_self_edge_primitives", _families.update_self_edge_primitives,
        (("self_edges", "team"),), ()),
    ("update_self_triangle_primitives", _families.update_self_triangle_primitives,
        (("self_triangles", "team"),), ()),
    ("publish_self_primitive_size", _families.publish_self_primitive_size,
        (("team", "enabled"),), ()),
    ("clear_self_contact_counters", _families.clear_self_contact_counters,
        (("derived", "self_counters"),), ()),
    ("query_self_edge_contacts", _families.query_self_edge_contacts,
        (("derived", "self_contact_query_slots"),), ("self_edges", "self_triangles")),
    ("query_self_point_contacts", _families.query_self_point_contacts,
        (("derived", "self_contact_query_slots"),), ()),
    ("measure_self_edge_contacts", _families.measure_self_edge_contacts,
        (("derived", "self_edge_contact_source"),), ()),
    ("measure_self_point_contacts", _families.measure_self_point_contacts,
        (("derived", "self_point_contact_source"),), ()),
    ("clear_self_contact_accumulator", _families.clear_self_contact_accumulator,
        (("particle", "team"),), ()),
    ("accumulate_self_edge_contacts", _families.accumulate_self_edge_contacts,
        (("derived", "self_edge_contact_source"),), ()),
    ("accumulate_self_point_contacts", _families.accumulate_self_point_contacts,
        (("derived", "self_point_contact_source"),), ()),
    ("apply_self_contact_accumulator", _families.apply_self_contact_accumulator,
        (("particle", "team"),), ()),
    ("apply_particle_friction", _families.apply_particle_friction,
        (("update_move", "particle"),), ()),
    ("commit_particle_velocity", _families.commit_particle_velocity,
        (("particle", "team"),), ()),
    ("commit_collider_substep_pose", _families.commit_collider_substep_pose,
        (("collider", "team"),), ()),
    ("clear_particle_intersect_flag", _families.clear_particle_intersect_flag,
        (("particle", "team"),), ()),
    ("mark_particle_intersect_flag", _families.mark_particle_intersect_flag,
        (("derived", "self_intersect_pair_edge"),), ()),
    ("blend_particle_display_pose", _families.blend_particle_display_pose,
        (("particle", "team"),), ()),
    ("resolve_collider_frame_pose", _families.resolve_collider_frame_pose,
        (("collider", "team"),), ()),
    ("project_particle_out_of_colliders", _families.project_particle_out_of_colliders,
        (("particle", "team"),), ()),
    ("propagate_postline_rotation", _families.propagate_postline_rotation,
        (("derived", "postline_root_offsets"),), ()),
    ("accumulate_triangle_basis", _families.accumulate_triangle_basis,
        (("triangles", "team"),), ()),
    ("orient_particle_from_triangles", _families.orient_particle_from_triangles,
        (("particle", "team"),), ()),
    ("emit_particle_output_rotation", _families.emit_particle_output_rotation,
        (("particle", "team"),), ()),
    ("publish_bone_transform", _families.publish_bone_transform,
        (("particle", "team"),), ()),
    ("commit_collider_frame_pose", _families.commit_collider_frame_pose,
        (("collider", "team"),), ()),
    ("close_team_frame", _families.close_team_frame, (("team", "enabled"),), ()),
)

FRAME_SCHEDULE = (
    ("resolve_team_synchronization_top", ()),
    ("snapshot_team_synchronization_clock", ()),
    ("apply_team_synchronization_clock", ()),
    ("advance_team_frame_clock", ()),
    ("publish_bone_transform", ()),
    ("skin_particle_pose", ()),
    ("resolve_team_negative_scale", ()),
    ("resolve_team_frame_pose", ()),
    ("advance_team_component_inertia", ()),
    ("resolve_team_world_inertia", ()),
    ("resolve_particle_wind_zones", ()),
    ("sample_team_wind_zones", ()),
    ("reset_particle_frame_state", ()),
    ("prepare_collider_frame_pose", ()),
    ("update_collider_face_primitives", ()),
    ("update_collider_vertex_pseudo_normals", ()),
    ("clear_intersect_pair_counter", (INTERSECT_PAIRS_PRESENT,)),
    ("gather_intersect_pairs", (INTERSECT_PAIRS_PRESENT,)),
    ("advance_team_substep_motion", (SUBSTEP_LOOP,)),
    ("publish_bone_transform", (SUBSTEP_LOOP,)),
    ("interpolate_collider_substep_pose", (SUBSTEP_LOOP,)),
    ("animate_particle_base_pose", (SUBSTEP_LOOP,)),
    ("integrate_particle_motion", (SUBSTEP_LOOP,)),
    ("pin_fixed_particles", (SUBSTEP_LOOP,)),
    ("pin_spring_particles", (SUBSTEP_LOOP,)),
    ("propagate_static_chain_pose", (SUBSTEP_LOOP,)),
    ("propagate_animated_chain_pose", (SUBSTEP_LOOP,)),
    ("blend_baseline_chain_pose", (SUBSTEP_LOOP,)),
    ("solve_tether_constraint", (SUBSTEP_LOOP,)),
    ("gather_distance_correction", (SUBSTEP_LOOP,)),
    ("apply_distance_correction", (SUBSTEP_LOOP,)),
    ("buffer_baseline_angle_state", (SUBSTEP_LOOP,)),
    ("buffer_carried_angle_state", (SUBSTEP_LOOP,)),
    ("solve_angle_constraint", (SUBSTEP_LOOP,)),
    ("clear_distance_accumulator", (SUBSTEP_LOOP,)),
    ("solve_bending_constraint", (SUBSTEP_LOOP,)),
    ("apply_distance_accumulator", (SUBSTEP_LOOP,)),
    ("measure_collider_point_contacts", (SUBSTEP_LOOP,)),
    ("measure_collider_edge_feet", (SUBSTEP_LOOP,)),
    ("measure_collider_edge_contacts", (SUBSTEP_LOOP,)),
    ("apply_collider_spring_response", (SUBSTEP_LOOP,)),
    ("gather_collider_point_contacts", (SUBSTEP_LOOP,)),
    ("resolve_collider_point_contacts", (SUBSTEP_LOOP,)),
    ("clear_collision_accumulator", (SUBSTEP_LOOP,)),
    ("solve_collider_edge_contacts", (SUBSTEP_LOOP,)),
    ("apply_collision_accumulator", (SUBSTEP_LOOP,)),
    ("gather_distance_correction", (SUBSTEP_LOOP,)),
    ("apply_distance_correction", (SUBSTEP_LOOP,)),
    ("solve_motion_constraint", (SUBSTEP_LOOP,)),
    ("clear_self_primitive_size", (SUBSTEP_LOOP, CONTACT_PAIRS_PRESENT, FIRST_SUBSTEP)),
    ("update_self_point_primitives", (SUBSTEP_LOOP, CONTACT_PAIRS_PRESENT, FIRST_SUBSTEP)),
    ("update_self_edge_primitives", (SUBSTEP_LOOP, CONTACT_PAIRS_PRESENT, FIRST_SUBSTEP)),
    ("update_self_triangle_primitives", (SUBSTEP_LOOP, CONTACT_PAIRS_PRESENT, FIRST_SUBSTEP)),
    ("publish_self_primitive_size", (SUBSTEP_LOOP, CONTACT_PAIRS_PRESENT, FIRST_SUBSTEP)),
    ("clear_self_contact_counters", (SUBSTEP_LOOP, CONTACT_PAIRS_PRESENT, FIRST_SUBSTEP)),
    ("query_self_edge_contacts", (SUBSTEP_LOOP, CONTACT_PAIRS_PRESENT, FIRST_SUBSTEP)),
    ("query_self_point_contacts", (SUBSTEP_LOOP, CONTACT_PAIRS_PRESENT, FIRST_SUBSTEP)),
    ("measure_self_edge_contacts", (SUBSTEP_LOOP, CONTACT_PAIRS_PRESENT, LATER_SUBSTEP)),
    ("measure_self_point_contacts", (SUBSTEP_LOOP, CONTACT_PAIRS_PRESENT, LATER_SUBSTEP)),
    ("clear_self_contact_accumulator", (SUBSTEP_LOOP, CONTACT_PAIRS_PRESENT)),
    ("accumulate_self_edge_contacts",
        (SUBSTEP_LOOP, CONTACT_PAIRS_PRESENT, SELF_ITERATION_LOOP)),
    ("accumulate_self_point_contacts",
        (SUBSTEP_LOOP, CONTACT_PAIRS_PRESENT, SELF_ITERATION_LOOP)),
    ("apply_self_contact_accumulator",
        (SUBSTEP_LOOP, CONTACT_PAIRS_PRESENT, SELF_ITERATION_LOOP)),
    ("apply_particle_friction", (SUBSTEP_LOOP,)),
    ("commit_particle_velocity", (SUBSTEP_LOOP,)),
    ("commit_collider_substep_pose", (SUBSTEP_LOOP,)),
    ("clear_particle_intersect_flag", (INTERSECT_PAIRS_PRESENT,)),
    ("mark_particle_intersect_flag", (INTERSECT_PAIRS_PRESENT,)),
    ("blend_particle_display_pose", ()),
    ("resolve_collider_frame_pose", ()),
    ("project_particle_out_of_colliders", ()),
    ("propagate_postline_rotation", ()),
    ("accumulate_triangle_basis", ()),
    ("orient_particle_from_triangles", ()),
    ("emit_particle_output_rotation", ()),
    ("publish_bone_transform", ()),
    ("commit_collider_frame_pose", ()),
    ("close_team_frame", ()),
)


def _predicate_contact_pairs_present(flags, scalars):
    return flags[FLAG_CONTACT_PAIR_COUNT] > 0


def _predicate_intersect_pairs_present(flags, scalars):
    return flags[FLAG_INTERSECT_PAIR_COUNT] > 0


def _predicate_first_substep(flags, scalars):
    return scalars[SUBSTEP_SCALAR] == 0


PREDICATES = {
    PREDICATE_CONTACT_PAIRS_PRESENT: _predicate_contact_pairs_present,
    PREDICATE_INTERSECT_PAIRS_PRESENT: _predicate_intersect_pairs_present,
    PREDICATE_FIRST_SUBSTEP: _predicate_first_substep,
}

FAMILY_NAMES = tuple(row[0] for row in FAMILY_SPECIFICATION_TABLE)

FAMILY_KERNEL = {row[0]: row[1] for row in FAMILY_SPECIFICATION_TABLE}

FAMILY_LAUNCH_PLANES = {row[0]: row[2] for row in FAMILY_SPECIFICATION_TABLE}

FAMILY_REFRESHED_INDEXES = {row[0]: row[3] for row in FAMILY_SPECIFICATION_TABLE}

SCHEDULED_FAMILY_NAMES = tuple(row[0] for row in FRAME_SCHEDULE)


def kernel_argument_names(kernel):
    return tuple(argument.label for argument in kernel.adj.args)


def queried_spatial_indexes():
    tree = ast.parse(io.open(_families.__file__, encoding="utf-8").read())
    queried = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        names = set()
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Attribute) or not isinstance(inner.value, ast.Name):
                continue
            if inner.value.id != "state" or not inner.attr.endswith(SPATIAL_INDEX_MEMBER_SUFFIX):
                continue
            names.add(inner.attr[:-len(SPATIAL_INDEX_MEMBER_SUFFIX)])
        queried[node.name] = names
    return queried


QUERIED_SPATIAL_INDEXES = queried_spatial_indexes()


def _validate_family_specification():
    seen = set()
    for row in FAMILY_SPECIFICATION_TABLE:
        assert len(row) == FAMILY_SPECIFICATION_WIDTH, \
            "a family row declares name, kernel, launch planes and refreshed spatial " \
            "indexes, got %r" % (row,)
        family_name, kernel, launch_planes, refreshed = row
        assert family_name not in seen, "family %s is declared twice" % family_name
        seen.add(family_name)
        assert kernel.key == family_name, \
            "family %s runs the kernel %s, a family and its kernel carry one name" \
            % (family_name, kernel.key)
        names = kernel_argument_names(kernel)
        assert names == KERNEL_SIGNATURE, \
            "family %s takes %r while every family kernel takes %r" \
            % (family_name, list(names), list(KERNEL_SIGNATURE))
        assert isinstance(launch_planes, tuple) and launch_planes, \
            "family %s declares its launch width as a non empty tuple of planes, it " \
            "declares %r" % (family_name, (launch_planes,))
        widths = set()
        for plane in launch_planes:
            assert plane not in widths, \
                "family %s names the plane %r twice in its launch width" % (family_name, plane)
            widths.add(plane)
            storage_name, field_name = plane
            assert storage_name in _state.STORAGE_NAMES, \
                "family %s launches over the unknown storage %r" % (family_name, storage_name)
            assert field_name in _state.STORAGE_FIELDS[storage_name], \
                "family %s launches over the unknown field %s.%s" \
                % (family_name, storage_name, field_name)
        indexes = set()
        for index_name in refreshed:
            assert index_name not in indexes, \
                "family %s refreshes the spatial index %s twice" % (family_name, index_name)
            indexes.add(index_name)
            assert index_name in _state.SPATIAL_INDEX_NAMES, \
                "family %s refreshes the spatial index %r which the state layer does not " \
                "build, it builds %r" \
                % (family_name, index_name, list(_state.SPATIAL_INDEX_NAMES))


def _validate_context_frame(family_name, frame, declared_scalars):
    assert isinstance(frame, tuple) and frame, \
        "family %s declares a context frame as a tuple, got %r" % (family_name, (frame,))
    kind = frame[0]
    assert kind in SCHEDULE_KINDS, \
        "family %s declares the context kind %r, only %r are defined" \
        % (family_name, kind, SCHEDULE_KINDS)
    if kind != SCHEDULE_LOOP:
        assert len(frame) == 2 and frame[1] in PREDICATES, \
            "family %s tests %r which no predicate declares, only %r are defined" \
            % (family_name, (frame,), tuple(PREDICATES))
        return
    assert len(frame) == 4, \
        "family %s declares a loop frame as kind, scalar, count rule and count source, got " \
        "%r" % (family_name, (frame,))
    _kind, scalar_name, count_rule, count_source = frame
    assert scalar_name in LOOP_SCALAR_NAMES, \
        "family %s loops over the scalar %r, only %r are bound" \
        % (family_name, scalar_name, list(LOOP_SCALAR_NAMES))
    assert scalar_name not in declared_scalars, \
        "family %s loops over the scalar %s twice in one context" % (family_name, scalar_name)
    declared_scalars.add(scalar_name)
    assert count_rule in LOOP_COUNT_RULES, \
        "family %s counts the loop over %s with the rule %r, only %r are defined" \
        % (family_name, scalar_name, count_rule, LOOP_COUNT_RULES)
    if count_rule == LOOP_COUNT_FROM_FLAG:
        assert count_source in FLAG_NAMES, \
            "family %s counts the loop over %s from the flag %r which the flag table does " \
            "not declare" % (family_name, scalar_name, count_source)
        return
    assert isinstance(count_source, tuple) and count_source, \
        "family %s counts the loop over %s from offset planes so the source is a non empty " \
        "tuple of planes, it declares %r" % (family_name, scalar_name, count_source)
    for storage_name, field_name in count_source:
        assert storage_name in _state.STORAGE_NAMES \
            and field_name in _state.STORAGE_FIELDS[storage_name], \
            "family %s counts the loop over %s from the unknown plane %s.%s" \
            % (family_name, scalar_name, storage_name, field_name)


SPATIAL_INDEX_REFRESH_REASON = (
    "a spatial index is refitted inside the recorded frame, so the refit has to be a node "
    "of the same graph and it has to sit before every launch that reads the tree; the rule "
    "used to be the narrower one that the family scheduled immediately before, in exactly "
    "the same context, is the one that refreshes, which is true of the self collision "
    "search because its bounds move on every substep and each query is preceded by the "
    "family that wrote them; the collider surface is the other shape, its bounds are the "
    "triangles in the collider's own frame and those change once per frame and not once "
    "per substep, so one refit near the top of the frame serves four families that read it "
    "from two different contexts; what has to hold in both shapes is that the refresher "
    "runs earlier in the frame and that it runs whenever the reader does, which is what a "
    "context the reader always sits inside means, and the narrower rule is a special case "
    "of it")


def _context_covers(outer, inner):
    return len(outer) <= len(inner) and tuple(inner[:len(outer)]) == tuple(outer)


def _validate_frame_schedule():
    scheduled = set()
    refreshed_before = []
    for row in FRAME_SCHEDULE:
        assert len(row) == 2, \
            "a schedule row declares the family and its context, got %r" % (row,)
        family_name, context = row
        assert family_name in FAMILY_KERNEL, \
            "the schedule runs %r which no family declares" % (family_name,)
        scheduled.add(family_name)
        assert isinstance(context, tuple), \
            "family %s declares its context as a tuple of frames, got %r" \
            % (family_name, (context,))
        declared_scalars = set()
        for frame in context:
            _validate_context_frame(family_name, frame, declared_scalars)
        queried = QUERIED_SPATIAL_INDEXES[family_name + "_element"]
        for index_name in queried:
            if index_name in FAMILY_REFRESHED_INDEXES[family_name]:
                continue
            assert any(index_name in FAMILY_REFRESHED_INDEXES[earlier_name]
                       and _context_covers(earlier_context, context)
                       for earlier_name, earlier_context in refreshed_before), \
                "%s\nfamily %s queries the spatial index %s and no family scheduled before " \
                "it refreshes that index in a context this one always runs inside" \
                % (SPATIAL_INDEX_REFRESH_REASON, family_name, index_name)
        refreshed_before.append((family_name, context))
    missing = sorted(set(FAMILY_NAMES) - scheduled)
    assert not missing, \
        "these families are declared and never scheduled, a family that never runs is a " \
        "placeholder: %r" % (missing,)


_validate_family_specification()
_validate_frame_schedule()


def schedule_tree(entries, depth):
    out = []
    index = 0
    while index < len(entries):
        row_index, family_name, context = entries[index]
        if len(context) == depth:
            out.append((None, (row_index, family_name), None))
            index += 1
            continue
        head = context[depth]
        end = index
        while (end < len(entries) and len(entries[end][2]) > depth
               and entries[end][2][depth] == head):
            end += 1
        out.append((head, None, schedule_tree(entries[index:end], depth + 1)))
        index = end
    return out


NUMBERED_SCHEDULE = tuple((index, family_name, context) for index, (family_name, context)
                          in enumerate(FRAME_SCHEDULE))

SCHEDULE_TREE = schedule_tree(NUMBERED_SCHEDULE, 0)


def loop_extent(state, flags, frame):
    _kind, scalar_name, count_rule, count_source = frame
    if count_rule == LOOP_COUNT_FROM_FLAG:
        return int(flags[count_source])
    declared = {state.plane_element_count(storage_name, field_name) - 1
                for storage_name, field_name in count_source}
    assert len(declared) == 1, \
        "the loop over %s counts its levels from every one of its offset planes so they all " \
        "hold the same number of levels, the state holds %r" % (scalar_name, sorted(declared))
    extent = declared.pop()
    assert extent >= 0, \
        "the loop over %s reads an offset plane of %d elements, an offset plane holds one " \
        "element more than the number of levels" % (scalar_name, extent + 1)
    return extent


def flatten(tree, state, flags, scalars, out):
    for head, row, subtree in tree:
        if row is not None:
            out.append((row[0], row[1], dict(scalars)))
            continue
        if head[0] == SCHEDULE_LOOP:
            held = dict(scalars)
            for value in range(loop_extent(state, flags, head)):
                held[head[1]] = value
                flatten(subtree, state, flags, held, out)
            continue
        taken = PREDICATES[head[1]](flags, scalars)
        if head[0] == SCHEDULE_ELSE:
            taken = not taken
        if taken:
            flatten(subtree, state, flags, scalars, out)


def frame_families(state, flags):
    out = []
    flatten(SCHEDULE_TREE, state, flags, {name: 0 for name in LOOP_SCALAR_NAMES}, out)
    return tuple(out)


def launch_width(state, family_name):
    return sum(state.plane_element_count(storage_name, field_name)
               for storage_name, field_name in FAMILY_LAUNCH_PLANES[family_name])


def record_frame(plan, state, device_state, flags):
    for _row_index, family_name, scalars in frame_families(state, flags):
        for index_name in FAMILY_REFRESHED_INDEXES[family_name]:
            if state.spatial_index_identifier(index_name) \
                    == _state.EMPTY_SPATIAL_INDEX_IDENTIFIER:
                continue
            plan.record_spatial_index_refit(state, index_name)
        plan.record(FAMILY_KERNEL[family_name],
                    _plan.launch_dimension(launch_width(state, family_name)),
                    [device_state, scalars[SUBSTEP_SCALAR], scalars[LEVEL_SCALAR],
                     scalars[ITERATION_SCALAR]],
                    family_name)
