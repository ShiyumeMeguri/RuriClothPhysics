import numpy as np

from ..cloth_kernel import contact_plan as _contact_plan
from ..cloth_kernel import defs as _defs
from ..cloth_kernel import io as _io
from ..cloth_kernel import program as _program
from ..cloth_kernel import world as _world
from . import dataflow as _dataflow
from . import device_state as _device_state
from . import lint as _lint
from . import plan as _plan
from . import schedule as _schedule
from . import state as _state

ZONE_MODE_VALUES = {"GLOBAL_DIRECTION": 0, "BOX_DIRECTION": 1,
                    "SPHERE_DIRECTION": 2, "SPHERE_RADIAL": 3}

SELF_PAIR_GUARD = 30_000_000

INPUT_TEAM_FIELDS = ("enabled", "component_world_position", "component_world_rotation",
                     "component_world_scale", "component_world_reflected",
                     "culling_invisible", "distance_weight",
                     "sync_target", "has_anchor", "anchor_position", "anchor_rotation")

CONSUMABLE_TEAM_FIELDS = ("reset_pending", "time_reset_pending", "keep_teleport_pending",
                          "force_mode", "impact_force")

INPUT_UPLOAD_TEAM_FIELDS = INPUT_TEAM_FIELDS + CONSUMABLE_TEAM_FIELDS

CONFIG_TEAM_FIELDS = (
    "gravity", "gravity_direction", "gravity_falloff", "stablization_time",
    "blend_weight_param", "damping_lut", "radius_lut", "normal_axis_vector",
    "rotational_interpolation", "root_rotation", "animation_pose_ratio", "time_scale",
    "tether_compression", "distance_lut", "bending_stiffness", "angle_use_restoration",
    "angle_restoration_lut", "angle_restoration_attenuation", "angle_restoration_gravity_falloff",
    "angle_use_limit", "angle_limit_lut", "angle_limit_stiffness", "motion_use_max_distance",
    "motion_max_distance_lut", "motion_use_backstop", "motion_backstop_radius",
    "motion_backstop_lut", "motion_stiffness", "collision_mode", "dynamic_friction",
    "static_friction", "limit_distance_lut", "self_mode", "sync_mode", "self_thickness_lut",
    "self_cloth_mass", "anchor_inertia", "world_inertia", "movement_inertia_smoothing",
    "movement_speed_limit", "rotation_speed_limit", "local_inertia", "local_movement_speed_limit",
    "local_rotation_speed_limit", "depth_inertia", "centrifugal_acceleration",
    "particle_speed_limit", "teleport_mode", "teleport_distance", "teleport_rotation",
    "wind_influence", "wind_frequency", "wind_turbulence", "wind_blend", "wind_synchronization",
    "wind_depth_weight", "wind_moving", "spring_power", "spring_limit_distance",
    "spring_normal_limit_ratio", "spring_noise")

COLLIDER_INPUT_FIELDS = ("input_positions", "input_rotations", "input_tips", "input_radii",
                         "enabled", "mesh_local_bound_min", "mesh_local_bound_max")

COLLIDER_VERTEX_INPUT_FIELDS = ("local_position",)

OUTPUT_PARTICLE_FIELDS = ("positions", "out_rotations")

SELF_TASK_TEAM_FIELDS = ("use_point", "use_edge", "use_triangle")

assert not (set(INPUT_UPLOAD_TEAM_FIELDS) & set(CONFIG_TEAM_FIELDS))
assert len(set(INPUT_UPLOAD_TEAM_FIELDS)) == len(INPUT_UPLOAD_TEAM_FIELDS)

HOST_TEAM_FIELDS = frozenset(INPUT_UPLOAD_TEAM_FIELDS + CONFIG_TEAM_FIELDS)
HOST_COLLIDER_FIELDS = frozenset(COLLIDER_INPUT_FIELDS)

DOMAIN_SOURCE_WORLD_TEAM = "world_team"
DOMAIN_SOURCE_WORLD_ARENA = "world_arena"
DOMAIN_SOURCE_PROGRAM_TABLE = "program_table"
DOMAIN_SOURCE_PRIMITIVE_ARENA = "primitive_arena"
DOMAIN_SOURCE_WIND_ZONES = "wind_zones"

DOMAIN_SOURCE_KINDS = (DOMAIN_SOURCE_WORLD_TEAM, DOMAIN_SOURCE_WORLD_ARENA,
                       DOMAIN_SOURCE_PROGRAM_TABLE, DOMAIN_SOURCE_PRIMITIVE_ARENA,
                       DOMAIN_SOURCE_WIND_ZONES)

DOMAIN_SOURCE_TABLE = (
    ("team", DOMAIN_SOURCE_WORLD_TEAM, "team", "num_teams"),
    ("particle", DOMAIN_SOURCE_WORLD_ARENA, "particles", "num_particles"),
    ("transform", DOMAIN_SOURCE_WORLD_ARENA, "transforms", "num_transforms"),
    ("collider", DOMAIN_SOURCE_WORLD_ARENA, "colliders", "num_colliders"),
    ("collider_vertices", DOMAIN_SOURCE_WORLD_ARENA, "collider_vertices",
     "num_collider_vertices"),
    ("collider_faces", DOMAIN_SOURCE_WORLD_ARENA, "collider_faces", "num_collider_faces"),
    ("distance", DOMAIN_SOURCE_PROGRAM_TABLE, "distance", None),
    ("bending", DOMAIN_SOURCE_PROGRAM_TABLE, "bending", None),
    ("tether", DOMAIN_SOURCE_PROGRAM_TABLE, "tether", None),
    ("motion", DOMAIN_SOURCE_PROGRAM_TABLE, "motion", None),
    ("update_move", DOMAIN_SOURCE_PROGRAM_TABLE, "update_move", None),
    ("update_fixed", DOMAIN_SOURCE_PROGRAM_TABLE, "update_fixed", None),
    ("spring", DOMAIN_SOURCE_PROGRAM_TABLE, "spring", None),
    ("collision_process", DOMAIN_SOURCE_PROGRAM_TABLE, "collision_process", None),
    ("center_fixed", DOMAIN_SOURCE_PROGRAM_TABLE, "center_fixed", None),
    ("angle_buffered", DOMAIN_SOURCE_PROGRAM_TABLE, "angle_buffered", None),
    ("edges", DOMAIN_SOURCE_PROGRAM_TABLE, "edges", None),
    ("collision_edges", DOMAIN_SOURCE_PROGRAM_TABLE, "collision_edges", None),
    ("triangles", DOMAIN_SOURCE_PROGRAM_TABLE, "triangles", None),
    ("v2t", DOMAIN_SOURCE_PROGRAM_TABLE, "v2t", None),
    ("point_pairs", DOMAIN_SOURCE_PROGRAM_TABLE, "point_pairs", None),
    ("edge_pairs", DOMAIN_SOURCE_PROGRAM_TABLE, "edge_pairs", None),
    ("self_points", DOMAIN_SOURCE_PRIMITIVE_ARENA, "self_points", "num_self_points"),
    ("self_edges", DOMAIN_SOURCE_PRIMITIVE_ARENA, "self_edges", "num_self_edges"),
    ("self_triangles", DOMAIN_SOURCE_PRIMITIVE_ARENA, "self_triangles", "num_self_triangles"),
    ("zone", DOMAIN_SOURCE_WIND_ZONES, None, None),
)

ZONE_DOMAIN_NAME = "zone"


def _validate_domain_source_table():
    declared = tuple(row[0] for row in DOMAIN_SOURCE_TABLE)
    assert declared == _state.DOMAIN_NAMES, \
        "the host loop sources the domains %r while the state layer declares %r" \
        % (list(declared), list(_state.DOMAIN_NAMES))
    for row in DOMAIN_SOURCE_TABLE:
        assert len(row) == 4, \
            "a domain source row declares the domain, the source kind, the source key and " \
            "the program count attribute, got %r" % (row,)
        domain_name, source_kind, source_key, count_attribute = row
        assert source_kind in DOMAIN_SOURCE_KINDS, \
            "domain %s declares the source kind %r, only %r are defined" \
            % (domain_name, source_kind, DOMAIN_SOURCE_KINDS)
        if source_kind in (DOMAIN_SOURCE_WORLD_TEAM, DOMAIN_SOURCE_WORLD_ARENA,
                           DOMAIN_SOURCE_PRIMITIVE_ARENA):
            assert count_attribute is not None, \
                "domain %s counts its elements from a program attribute, it names none" \
                % domain_name
            continue
        assert count_attribute is None, \
            "domain %s counts its elements from its own source so it must not also name a " \
            "program count attribute" % domain_name


_validate_domain_source_table()

CONTACT_TASK_UPLOADED_COLUMNS = (
    ("kind", "self_contact_task_kind"),
    ("source_start", "self_contact_task_source_start"),
    ("target_team", "self_contact_task_target_team"),
    ("same_team", "self_contact_task_same_team"),
)

INTERSECT_TASK_UPLOADED_COLUMNS = (
    ("edge_start", "self_intersect_task_edge_start"),
    ("triangle_team", "self_intersect_task_triangle_team"),
    ("same_team", "self_intersect_task_same_team"),
)


def _task_columns(declared, uploaded):
    return tuple((declared.index(column_name), plane_name)
                 for column_name, plane_name in uploaded)


CONTACT_TASK_COLUMNS = _task_columns(_contact_plan.CONTACT_TASK_COLUMNS,
                                     CONTACT_TASK_UPLOADED_COLUMNS)
CONTACT_TASK_QUERY_OFFSET_PLANE = "self_contact_task_query_offsets"
CONTACT_TASK_EDGE_SLOT_OFFSET_PLANE = "self_contact_task_edge_slot_offsets"
CONTACT_TASK_POINT_SLOT_OFFSET_PLANE = "self_contact_task_point_slot_offsets"
CONTACT_TASK_KIND_COLUMN = _contact_plan.CONTACT_TASK_COLUMNS.index("kind")
CONTACT_TASK_SOURCE_TEAM_COLUMN = _contact_plan.CONTACT_TASK_COLUMNS.index("source_team")
CONTACT_TASK_SOURCE_COUNT_COLUMN = _contact_plan.CONTACT_TASK_COLUMNS.index("source_count")

INTERSECT_TASK_COLUMNS = _task_columns(_contact_plan.INTERSECT_TASK_COLUMNS,
                                       INTERSECT_TASK_UPLOADED_COLUMNS)
INTERSECT_TASK_QUERY_OFFSET_PLANE = "self_intersect_task_query_offsets"
INTERSECT_TASK_SLOT_OFFSET_PLANE = "self_intersect_task_slot_offsets"
INTERSECT_TASK_EDGE_TEAM_COLUMN = _contact_plan.INTERSECT_TASK_COLUMNS.index("edge_team")
INTERSECT_TASK_EDGE_COUNT_COLUMN = _contact_plan.INTERSECT_TASK_COLUMNS.index("edge_count")

SELF_COUNTER_PLANE = "self_counters"

SELF_BUDGET_TABLE = (
    ("contact", "self_contact_overflow", "self_contact_demand"),
    ("intersection", "self_intersect_overflow", "self_intersect_demand"),
)

SELF_OVERFLOW_PLANES = tuple(row[1] for row in SELF_BUDGET_TABLE)

SELF_DEMAND_PLANES = tuple(row[2] for row in SELF_BUDGET_TABLE)

SELF_CONTACT_SLOT_COLUMN = "self_contact_slots"

SELF_TASK_OFFSET_PLANES = (CONTACT_TASK_QUERY_OFFSET_PLANE,
                           CONTACT_TASK_EDGE_SLOT_OFFSET_PLANE,
                           CONTACT_TASK_POINT_SLOT_OFFSET_PLANE,
                           INTERSECT_TASK_QUERY_OFFSET_PLANE,
                           INTERSECT_TASK_SLOT_OFFSET_PLANE)

FRAME_OUTPUT_REQUESTS = (
    (("transform", "solved"),)
    + tuple(("particle", field_name) for field_name in OUTPUT_PARTICLE_FIELDS)
    + tuple(("team", field_name) for field_name in CONSUMABLE_TEAM_FIELDS)
    + tuple((_state.DERIVED_STORAGE_NAME, plane_name)
            for plane_name in SELF_OVERFLOW_PLANES + SELF_DEMAND_PLANES
            + (SELF_COUNTER_PLANE,)))

RELEASE_DOWNLOAD_REQUESTS = (
    tuple(("team", field_name) for field_name in _state.STORAGE_FIELDS["team"]
          if field_name not in HOST_TEAM_FIELDS)
    + tuple(("particle", field_name) for field_name in _state.STORAGE_FIELDS["particle"])
    + tuple(("collider", field_name) for field_name in _state.STORAGE_FIELDS["collider"]
            if field_name not in HOST_COLLIDER_FIELDS))

HOST_FRAME_WRITTEN_PLANES = (
    tuple(("team", field_name)
          for field_name in INPUT_UPLOAD_TEAM_FIELDS + CONFIG_TEAM_FIELDS
          + SELF_TASK_TEAM_FIELDS)
    + (("transform", "world"),)
    + tuple(("collider", field_name) for field_name in COLLIDER_INPUT_FIELDS)
    + tuple(("collider_vertices", field_name) for field_name in COLLIDER_VERTEX_INPUT_FIELDS)
    + tuple((ZONE_DOMAIN_NAME, field_name) for field_name in _io.ZONE_FIELDS)
    + tuple((_state.DERIVED_STORAGE_NAME, plane_name)
            for plane_name in SELF_OVERFLOW_PLANES + SELF_DEMAND_PLANES
            + (SELF_COUNTER_PLANE,) + SELF_TASK_OFFSET_PLANES)
    + tuple((_state.DERIVED_STORAGE_NAME, plane_name)
            for _column, plane_name in CONTACT_TASK_UPLOADED_COLUMNS
            + INTERSECT_TASK_UPLOADED_COLUMNS)
    + tuple((_state.FRAME_SCALAR_STORAGE_NAME, plane_name)
            for plane_name in _state.FRAME_SCALAR_FIELDS))

DOWNLOAD_TARGET_TABLE = (
    ("team", lambda world, field_name: world.team[field_name], "num_teams"),
    ("particle", lambda world, field_name: world.particles.arrays[field_name],
     "num_particles"),
    ("collider", lambda world, field_name: world.colliders.arrays[field_name],
     "num_colliders"),
    ("transform", lambda world, field_name: world.transforms.arrays[field_name],
     "num_transforms"),
)

DOWNLOAD_TARGETS = {row[0]: (row[1], row[2]) for row in DOWNLOAD_TARGET_TABLE}

DISPLAY_PLANE_REASON = (
    "what leaves the device every frame is the pose the host writes back plus whatever the "
    "viewport layers said they would read, and the layers say it because they are the only "
    "readers and the only place that knows which of their switches is on; a fixed list here "
    "would either carry planes nobody is looking at, which is a copy and a synchronisation "
    "per frame of playback for nothing, or miss one, which is the defect this replaces: the "
    "overlay read the host mirror of a plane the frame never brought home and drew the value "
    "left there by the last rebuild, or by the registration, which for a position is the "
    "world origin")

DISPLAY_PLANE_HOST_OWNED_REASON = (
    "a plane no scheduled family writes is the host's own and is current by definition, so "
    "asking the frame to download one says the caller believes the device owns it; the "
    "answer would be the same bytes that were uploaded and the request would cost a copy "
    "per frame to confirm it, which is why it is refused rather than served")


def _validate_download_target_table():
    declared = tuple(row[0] for row in DOWNLOAD_TARGET_TABLE)
    assert len(set(declared)) == len(declared), \
        "a download target storage is declared twice, the table declares %r" % (list(declared),)
    for storage_name, _reader, count_attribute in DOWNLOAD_TARGET_TABLE:
        assert storage_name in _state.STORAGE_FIELDS, \
            "the download target %s is not a storage the state layer declares, it declares " \
            "%r" % (storage_name, list(_state.STORAGE_FIELDS))
        assert count_attribute, \
            "the download target %s names no program count attribute" % storage_name


_validate_download_target_table()


CONFIG_REVISION_REASON = (
    "the configuration columns of a team are written by exactly one host operation, so the "
    "question the upload asks every frame, has the configuration changed, is answered where "
    "the change happens rather than recomputed from the bytes; the fingerprint this replaces "
    "read sixty arrays out of the world and joined them on every frame of every playback, "
    "and it was also a second answer to a question the host layer already answers with its "
    "own parameter token, so the two could disagree about the same frame")

SELF_CONTACT_OVERFLOW_REASON = (
    "the self contact search keeps a fixed number of contacts per query primitive, and the "
    "device raised the error slot because a primitive found more candidates than it could "
    "keep; the contacts that were dropped are the ones the solver would have resolved, so "
    "the positions this frame produced are wrong by an amount nothing measures, and a wrong "
    "answer that nobody is told about is worse than a stopped frame; raise the contact slot "
    "count of the teams listed below, or thin the mesh they collide with")


class SelfContactOverflow(RuntimeError):
    pass


def _packed_primitive_bits(arena_arrays, field_name, element_count):
    rows = arena_arrays[field_name][:element_count].astype(np.uint8)
    return np.ascontiguousarray(rows[:, 0] | (rows[:, 1] << 1) | (rows[:, 2] << 2))


class ClothEngine:

    def __init__(self, world, target_name):
        self.target_name = target_name
        self.world = None
        self.structure_revision = None
        self.program = None
        self.state = None
        self.device_state = None
        self.plans = {}
        self.config_revision_shadow = None
        self.zone_shadow = None
        self.zone_count = 0
        self.self_task_token = None
        self.self_task_mask = None
        self.self_task_cache = None
        self.self_upload_token = None
        self.self_upload_mask = None
        self.self_upload_totals = (0, 0)
        self.self_empty_uploaded = True
        self.self_overflow_counts = {}
        self.self_demand_counts = {}
        self.self_counter_values = np.zeros(int(_defs.SCL_LEN), np.int32)
        self.uploaded_field_count = 0
        self.display_plane_shadow = None
        self.display_stale_shadow = self._stale_planes(FRAME_OUTPUT_REQUESTS)
        self.load(world)

    def _domain_element_count(self, source_kind, source_key, count_attribute):
        if source_kind == DOMAIN_SOURCE_WIND_ZONES:
            return max(self.zone_count, 1)
        if source_kind == DOMAIN_SOURCE_PROGRAM_TABLE:
            return int(getattr(self.program, source_key)["team"].shape[0])
        return int(getattr(self.program, count_attribute))

    def _domain_values(self, world, zones, source_kind, source_key, element_count,
                       field_name):
        if source_kind == DOMAIN_SOURCE_WORLD_TEAM:
            return world.team[field_name][:element_count]
        if source_kind == DOMAIN_SOURCE_WORLD_ARENA:
            return getattr(world, source_key).arrays[field_name][:element_count]
        if source_kind == DOMAIN_SOURCE_PROGRAM_TABLE:
            return getattr(self.program, source_key)[field_name]
        if source_kind == DOMAIN_SOURCE_PRIMITIVE_ARENA:
            arena_arrays = getattr(world, source_key).arrays
            if field_name in _world.PRIMITIVE_PACKED_FIELDS:
                return _packed_primitive_bits(arena_arrays, field_name, element_count)
            return arena_arrays[field_name][:element_count]
        return self._zone_values(zones, element_count, field_name)

    @staticmethod
    def _zone_values(zones, element_count, field_name):
        scalar_type, inner_shape = _io.ZONE_FIELDS[field_name]
        values = np.zeros((element_count,) + inner_shape, dtype=scalar_type)
        for index, zone in enumerate(zones):
            if field_name == "mode":
                values[index] = ZONE_MODE_VALUES.get(zone.mode, 0)
                continue
            if field_name == "zone_volume":
                values[index] = np.inf if zone.mode == "GLOBAL_DIRECTION" \
                    else float(zone.zone_volume)
                continue
            if field_name == "attenuation_lut":
                if zone.attenuation_lut is not None:
                    values[index] = zone.attenuation_lut
                continue
            values[index] = getattr(zone, field_name)
        return values

    def _write_field(self, storage_name, field_name, values):
        dtype, shape = self.state.value_specification(storage_name, field_name)
        payload = np.ascontiguousarray(values)
        if payload.dtype != dtype:
            payload = payload.astype(dtype)
        assert payload.shape == shape, \
            "%s.%s expects shape %r, the host produced %r" \
            % (storage_name, field_name, shape, payload.shape)
        self.state.write(storage_name, field_name, payload)

    def _write_domain(self, world, zones, domain_name, source_kind, source_key,
                      count_attribute):
        element_count = self._domain_element_count(source_kind, source_key, count_attribute)
        for field_name in self.state.field_names(domain_name):
            self._write_field(domain_name, field_name,
                              self._domain_values(world, zones, source_kind, source_key,
                                                  element_count, field_name))

    def load(self, world):
        revision = int(world.structure_revision)
        if self.world is world and self.structure_revision == revision:
            return False
        self.world = world
        self.structure_revision = revision
        self.program = _program.build_program(world, _dataflow.ACCUMULATION_WINDOWS)
        self.zone_count = 0
        derived_values = _program.derived_planes(self.program)
        element_counts = {}
        for domain_name, source_kind, source_key, count_attribute in DOMAIN_SOURCE_TABLE:
            element_counts[domain_name] = self._domain_element_count(
                source_kind, source_key, count_attribute)
        plane_counts = {plane_name: int(values.shape[0])
                        for plane_name, values in derived_values.items()}
        self.state = _state.ClothState(element_counts, plane_counts, self.target_name)
        self.device_state = _device_state.DeviceStateView(self.state)
        self.plans = {}
        for domain_name, source_kind, source_key, count_attribute in DOMAIN_SOURCE_TABLE:
            self._write_domain(world, [], domain_name, source_kind, source_key,
                               count_attribute)
        for plane_name, values in derived_values.items():
            self._write_field(_state.DERIVED_STORAGE_NAME, plane_name, values)
        for plane_name in self.state.field_names(_state.FRAME_SCALAR_STORAGE_NAME):
            dtype, shape = self.state.value_specification(
                _state.FRAME_SCALAR_STORAGE_NAME, plane_name)
            self._write_field(_state.FRAME_SCALAR_STORAGE_NAME, plane_name,
                              np.zeros(shape, dtype=dtype))
        self.state.flush()
        self.state.build_spatial_indexes()
        self.config_revision_shadow = world.config_revision
        self.zone_shadow = None
        self.self_task_token = None
        self.self_task_mask = None
        self.self_task_cache = None
        self.self_upload_token = None
        self.self_upload_mask = None
        self.self_upload_totals = (0, 0)
        self.self_empty_uploaded = True
        self.self_overflow_counts = {}
        self.self_demand_counts = {}
        self.self_counter_values = np.zeros(int(_defs.SCL_LEN), np.int32)
        return True

    def _upload_team_inputs(self, world):
        team_count = self.program.num_teams
        for field_name in INPUT_UPLOAD_TEAM_FIELDS:
            self._write_field("team", field_name, world.team[field_name][:team_count])

    def _upload_transform_worlds(self, world):
        transform_count = self.program.num_transforms
        if transform_count <= 0:
            return
        self._write_field("transform", "world",
                          world.transforms.arrays["world"][:transform_count])

    def _upload_collider_inputs(self, world):
        collider_count = self.program.num_colliders
        if collider_count <= 0:
            return
        arena_arrays = world.colliders.arrays
        for field_name in COLLIDER_INPUT_FIELDS:
            self._write_field("collider", field_name, arena_arrays[field_name][:collider_count])

    def _upload_collider_vertices(self, world):
        vertex_count = self.program.num_collider_vertices
        if vertex_count <= 0:
            return
        arena_arrays = world.collider_vertices.arrays
        for field_name in COLLIDER_VERTEX_INPUT_FIELDS:
            self._write_field("collider_vertices", field_name,
                              arena_arrays[field_name][:vertex_count])

    def _upload_config(self, world):
        revision = int(world.config_revision)
        assert revision >= self.config_revision_shadow, \
            "%s\nthe bound world reports configuration revision %d after reporting %d, and a " \
            "revision that goes backwards means the engine is reading a world it never " \
            "loaded" % (CONFIG_REVISION_REASON, revision, self.config_revision_shadow)
        if revision == self.config_revision_shadow:
            return
        self.config_revision_shadow = revision
        team_count = self.program.num_teams
        for field_name in CONFIG_TEAM_FIELDS:
            self._write_field("team", field_name, world.team[field_name][:team_count])

    def _upload_zones(self, zones):
        element_count = max(len(zones), 1)
        field_names = tuple(_io.ZONE_FIELDS)
        values = {field_name: self._zone_values(zones, element_count, field_name)
                  for field_name in field_names}
        fingerprint = b"".join(np.ascontiguousarray(values[field_name]).tobytes()
                               for field_name in field_names)
        if len(zones) == self.zone_count and fingerprint == self.zone_shadow:
            return
        self.zone_count = len(zones)
        self.zone_shadow = fingerprint
        if self.state.resize_domain(ZONE_DOMAIN_NAME, element_count):
            self.plans = {}
        for field_name in field_names:
            self._write_field(ZONE_DOMAIN_NAME, field_name, values[field_name])

    def _write_task_planes(self, tasks, capacity, columns):
        planes = {plane_name: np.zeros(capacity, np.int32) for _column, plane_name in columns}
        for task_index, task in enumerate(tasks):
            for column, plane_name in columns:
                planes[plane_name][task_index] = task[column]
        for plane_name, values in planes.items():
            self._write_field(_state.DERIVED_STORAGE_NAME, plane_name, values)

    def _fill_contact_task_table(self, tasks, slots_per_team):
        capacity = self.program.self_max_contact_tasks
        self._write_task_planes(tasks, capacity, CONTACT_TASK_COLUMNS)
        query_offsets = np.zeros(capacity + 1, np.int32)
        edge_slot_offsets = np.zeros(capacity + 1, np.int32)
        point_slot_offsets = np.zeros(capacity + 1, np.int32)
        queries = 0
        edge_slots = 0
        point_slots = 0
        for task_index, task in enumerate(tasks):
            query_offsets[task_index] = queries
            edge_slot_offsets[task_index] = edge_slots
            point_slot_offsets[task_index] = point_slots
            source_count = int(task[CONTACT_TASK_SOURCE_COUNT_COLUMN])
            kept = int(slots_per_team[int(task[CONTACT_TASK_SOURCE_TEAM_COLUMN])])
            queries += source_count
            if int(task[CONTACT_TASK_KIND_COLUMN]) == _contact_plan.SELF_TASK_KIND_EDGE_EDGE:
                edge_slots += source_count * kept
                continue
            point_slots += source_count * kept
        query_offsets[len(tasks):] = queries
        edge_slot_offsets[len(tasks):] = edge_slots
        point_slot_offsets[len(tasks):] = point_slots
        self._write_field(_state.DERIVED_STORAGE_NAME, CONTACT_TASK_QUERY_OFFSET_PLANE,
                          query_offsets)
        self._write_field(_state.DERIVED_STORAGE_NAME, CONTACT_TASK_EDGE_SLOT_OFFSET_PLANE,
                          edge_slot_offsets)
        self._write_field(_state.DERIVED_STORAGE_NAME, CONTACT_TASK_POINT_SLOT_OFFSET_PLANE,
                          point_slot_offsets)
        return queries

    def _fill_intersect_task_table(self, tasks, slots_per_team):
        capacity = self.program.self_max_intersect_tasks
        self._write_task_planes(tasks, capacity, INTERSECT_TASK_COLUMNS)
        query_offsets = np.zeros(capacity + 1, np.int32)
        slot_offsets = np.zeros(capacity + 1, np.int32)
        queries = 0
        slots = 0
        for task_index, task in enumerate(tasks):
            query_offsets[task_index] = queries
            slot_offsets[task_index] = slots
            edge_count = int(task[INTERSECT_TASK_EDGE_COUNT_COLUMN])
            kept = int(slots_per_team[int(task[INTERSECT_TASK_EDGE_TEAM_COLUMN])])
            queries += edge_count
            slots += edge_count * kept
        query_offsets[len(tasks):] = queries
        slot_offsets[len(tasks):] = slots
        self._write_field(_state.DERIVED_STORAGE_NAME, INTERSECT_TASK_QUERY_OFFSET_PLANE,
                          query_offsets)
        self._write_field(_state.DERIVED_STORAGE_NAME, INTERSECT_TASK_SLOT_OFFSET_PLANE,
                          slot_offsets)
        return queries

    def _clear_self_overflow(self):
        team_count = self.program.num_teams
        for plane_name in SELF_OVERFLOW_PLANES + SELF_DEMAND_PLANES:
            self._write_field(_state.DERIVED_STORAGE_NAME, plane_name,
                              np.zeros(team_count, np.int32))

    def _prepare_self_frame(self, world, frame_index):
        team = world.team
        team_count = self.program.num_teams
        self._clear_self_overflow()
        mask = _contact_plan.frame_team_mask(team, team_count)
        token = (int(world.structure_revision), int(world.contact_link_revision))
        if (self.self_task_cache is None or token != self.self_task_token
                or not np.array_equal(mask, self.self_task_mask)):
            self.self_task_cache = _contact_plan.build_tasks(team, team_count,
                                                             world.contact_links)
            self.self_task_token = token
            self.self_task_mask = mask
        contact, intersect, use_point, use_edge, use_triangle = self.self_task_cache
        if not contact and not intersect and self.self_empty_uploaded:
            return
        self.self_empty_uploaded = (not contact) and (not intersect)
        if (self.self_upload_token is None or token != self.self_upload_token
                or not np.array_equal(mask, self.self_upload_mask)):
            slots_per_team = team[SELF_CONTACT_SLOT_COLUMN][:team_count]
            total_contact = self._fill_contact_task_table(contact, slots_per_team)
            total_intersect = self._fill_intersect_task_table(intersect, slots_per_team)
            for field_name, values in zip(SELF_TASK_TEAM_FIELDS,
                                          (use_point, use_edge, use_triangle)):
                self._write_field("team", field_name, values)
            self.self_upload_totals = (total_contact, total_intersect)
            self.self_upload_token = token
            self.self_upload_mask = mask
        total_contact, total_intersect = self.self_upload_totals
        counters = np.zeros(int(_defs.SCL_LEN), np.int32)
        counters[_defs.SCL_ERROR] = 1 if (total_contact > SELF_PAIR_GUARD
                                          or total_intersect > SELF_PAIR_GUARD) else 0
        counters[_defs.SCL_USE_INTERSECT] = 1 if intersect else 0
        counters[_defs.SCL_FRAME_INDEX] = \
            int(frame_index) % int(_defs.SELF_COLLISION_INTERSECT_DIV)
        self._write_field(_state.DERIVED_STORAGE_NAME, SELF_COUNTER_PLANE, counters)

    @staticmethod
    def _frame_scalars(frame_globals):
        power = _defs.simulation_power(frame_globals.simulation_frequency)
        return (np.float32(frame_globals.frame_delta_time),
                np.float32(1.0 / frame_globals.simulation_frequency),
                np.int32(frame_globals.max_simulation_count),
                np.float32(frame_globals.global_time_scale),
                np.float32(power[0]), np.float32(power[1]),
                np.float32(power[2]), np.float32(power[3]))

    def _upload_frame_scalars(self, substep_count, frame_globals):
        (frame_delta_time, simulation_delta_time, max_simulation_count, global_time_scale,
         power_zero, power_one, power_two, power_three) = self._frame_scalars(frame_globals)
        float_dtype, float_shape = self.state.value_specification(
            _state.FRAME_SCALAR_STORAGE_NAME, "frame_float")
        integer_dtype, integer_shape = self.state.value_specification(
            _state.FRAME_SCALAR_STORAGE_NAME, "frame_int")
        float_plane = np.zeros(float_shape, dtype=float_dtype)
        float_plane[_defs.SCAL_FRAME_DT] = frame_delta_time
        float_plane[_defs.SCAL_SIM_DT] = simulation_delta_time
        float_plane[_defs.SCAL_TIME_SCALE] = global_time_scale
        float_plane[_defs.SCAL_POWER0] = power_zero
        float_plane[_defs.SCAL_POWER1] = power_one
        float_plane[_defs.SCAL_POWER2] = power_two
        float_plane[_defs.SCAL_POWER3] = power_three
        integer_plane = np.zeros(integer_shape, dtype=integer_dtype)
        integer_plane[_defs.SCAL_MAX_SIM] = max_simulation_count
        integer_plane[_defs.SCAL_N_ZONES] = self.zone_count
        integer_plane[_defs.SCAL_SUB_END] = substep_count
        self._write_field(_state.FRAME_SCALAR_STORAGE_NAME, "frame_float", float_plane)
        self._write_field(_state.FRAME_SCALAR_STORAGE_NAME, "frame_int", integer_plane)

    @staticmethod
    def _substep_count(frame_globals):
        return min(int(_defs.MAX_SIMULATION_COUNT_HIGH),
                   int(frame_globals.max_simulation_count))

    def _flags(self, substep_count):
        total_contact, total_intersect = self.self_upload_totals
        return {_schedule.FLAG_SUBSTEP_COUNT: int(substep_count),
                _schedule.FLAG_SELF_ITERATION_COUNT:
                    int(_defs.SELF_COLLISION_SOLVER_ITERATION),
                _schedule.FLAG_CONTACT_PAIR_COUNT: int(total_contact),
                _schedule.FLAG_INTERSECT_PAIR_COUNT: int(total_intersect)}

    def plan_for(self, flags):
        key = tuple(sorted(flags.items()))
        held = self.plans.get(key)
        if held is not None:
            return held
        recorded = _plan.Plan()
        _schedule.record_frame(recorded, self.state, self.device_state.current(), flags)
        descriptors = recorded.descriptors()
        levels = _dataflow.entry_levels(descriptors)
        _dataflow.assert_levels_are_conflict_free(descriptors, levels)
        recorded.assign_levels(levels)
        recorded.capture(self.state)
        self.plans[key] = recorded
        return recorded

    def _scatter_requests(self, world, requests, values):
        for storage_name, field_name in requests:
            target = DOWNLOAD_TARGETS.get(storage_name)
            if target is None:
                continue
            reader, count_attribute = target
            _scatter(reader(world, field_name), int(getattr(self.program, count_attribute)),
                     values[(storage_name, field_name)])

    @staticmethod
    def _stale_planes(fresh):
        return _dataflow.DEVICE_WRITTEN_PLANES - frozenset(fresh)

    def _apply_display_planes(self, world):
        planes = tuple(world.display_planes)
        if planes != self.display_plane_shadow:
            for plane in planes:
                storage_name, field_name = plane
                assert storage_name in DOWNLOAD_TARGETS, \
                    "%s\nthe viewport asked for %s.%s and the frame downloads into %r" \
                    % (DISPLAY_PLANE_REASON, storage_name, field_name, list(DOWNLOAD_TARGETS))
                assert field_name in _state.STORAGE_FIELDS[storage_name], \
                    "%s\nthe viewport asked for %s.%s and that storage declares %r" \
                    % (DISPLAY_PLANE_REASON, storage_name, field_name,
                       list(_state.STORAGE_FIELDS[storage_name]))
                assert plane in _dataflow.DEVICE_WRITTEN_PLANES, \
                    "%s\nthe viewport asked the frame to download %s.%s" \
                    % (DISPLAY_PLANE_HOST_OWNED_REASON, storage_name, field_name)
            self.display_plane_shadow = planes
            self.display_stale_shadow = self._stale_planes(FRAME_OUTPUT_REQUESTS + planes)
        world.set_stale_planes(self.display_stale_shadow)
        return planes

    def download_display(self, world):
        planes = self._apply_display_planes(world)
        if not planes:
            return
        self._scatter_requests(world, planes, self.state.read_batch(planes))

    def _download_frame_outputs(self, world):
        requests = FRAME_OUTPUT_REQUESTS + self._apply_display_planes(world)
        values = self.state.read_batch(requests)
        self._scatter_requests(world, requests, values)
        self.self_overflow_counts = {
            plane_name: values[(_state.DERIVED_STORAGE_NAME, plane_name)]
            for plane_name in SELF_OVERFLOW_PLANES}
        self.self_demand_counts = {
            plane_name: values[(_state.DERIVED_STORAGE_NAME, plane_name)]
            for plane_name in SELF_DEMAND_PLANES}
        self.self_counter_values = values[(_state.DERIVED_STORAGE_NAME, SELF_COUNTER_PLANE)]
        self._refuse_on_self_contact_overflow(world)

    def _refuse_on_self_contact_overflow(self, world):
        if not int(self.self_counter_values[_defs.SCL_ERROR]):
            return
        total_contact, total_intersect = self.self_upload_totals
        lines = ["the frame planned %d contact queries and %d intersection queries against "
                 "the guard of %d" % (total_contact, total_intersect, SELF_PAIR_GUARD)]
        slots = world.team[SELF_CONTACT_SLOT_COLUMN][:self.program.num_teams]
        for search_name, overflow_plane, demand_plane in SELF_BUDGET_TABLE:
            overflow = self.self_overflow_counts[overflow_plane]
            demand = self.self_demand_counts[demand_plane]
            for team_index in range(self.program.num_teams):
                if not int(overflow[team_index]):
                    continue
                lines.append(
                    "team %d keeps %d slots per query primitive, its worst %s query asked "
                    "for %d candidates, and %d candidates were dropped"
                    % (team_index, int(slots[team_index]), search_name,
                       int(demand[team_index]), int(overflow[team_index])))
        raise SelfContactOverflow("%s\n%s" % (SELF_CONTACT_OVERFLOW_REASON, "\n".join(lines)))

    def download_state(self, world):
        requests = tuple(
            (storage_name, field_name) for storage_name, field_name in RELEASE_DOWNLOAD_REQUESTS
            if (storage_name != "particle" or self.program.num_particles)
            and (storage_name != "collider" or self.program.num_colliders))
        self._scatter_requests(world, requests, self.state.read_batch(requests))
        world.set_stale_planes(self._stale_planes(tuple(requests) + FRAME_OUTPUT_REQUESTS))

    def step_frame(self, world, frame_globals):
        self.load(world)
        self._upload_team_inputs(world)
        self._upload_transform_worlds(world)
        self._upload_collider_inputs(world)
        self._upload_collider_vertices(world)
        self._upload_config(world)
        self._upload_zones(frame_globals.zones)
        self._prepare_self_frame(world, frame_globals.frame_index)
        substep_count = self._substep_count(frame_globals)
        self._upload_frame_scalars(substep_count, frame_globals)
        self.uploaded_field_count = self.state.flush()
        self.plan_for(self._flags(substep_count)).launch()
        self._download_frame_outputs(world)


def _scatter(target, element_count, values):
    if target.dtype == np.bool_:
        target[:element_count] = values.astype(np.bool_)
        return
    target[:element_count] = values


_lint.assert_clean()

_dataflow.assert_frame_boundary(HOST_FRAME_WRITTEN_PLANES, FRAME_OUTPUT_REQUESTS,
                                RELEASE_DOWNLOAD_REQUESTS)
