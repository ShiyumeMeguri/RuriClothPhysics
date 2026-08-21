import numpy as np

from ..cloth_kernel import contact_plan as _contact_plan
from ..cloth_kernel import defs as _defs
from ..cloth_kernel import frame as _frame
from ..cloth_kernel import io as _io
from ..cloth_kernel import phase_plan as _phase_plan
from ..cloth_kernel import program as _program
from ..cloth_kernel import world as _world
from . import plan as _plan
from . import state as _state
from . import wiring as _wiring

SUBSTEP_SCALAR_NAME = "k"

assert _wiring.SCALAR_NAMES == (SUBSTEP_SCALAR_NAME,), \
    "the host loop supplies the caller scalars %r while the wiring declares %r" \
    % ((SUBSTEP_SCALAR_NAME,), _wiring.SCALAR_NAMES)

ZONE_MODE_VALUES = {"GLOBAL_DIRECTION": 0, "BOX_DIRECTION": 1,
                    "SPHERE_DIRECTION": 2, "SPHERE_RADIAL": 3}

SELF_PAIR_GUARD = 30_000_000

STRUCTURE_TEAM_COLUMNS = ("valid", "is_spring", "p_start", "p_count", "t_start", "t_count",
                          "c_start", "c_count", "sp_start", "sp_count",
                          "se_start", "se_count", "st_start", "st_count")

INPUT_TEAM_FIELDS = ("enabled", "component_world_position", "component_world_rotation",
                     "component_world_scale", "culling_invisible", "distance_weight",
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
                         "enabled")

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
    ("target_start", "self_contact_task_target_start"),
    ("target_count", "self_contact_task_target_count"),
    ("same_team", "self_contact_task_same_team"),
)

INTERSECT_TASK_UPLOADED_COLUMNS = (
    ("edge_start", "self_intersect_task_edge_start"),
    ("triangle_team", "self_intersect_task_triangle_team"),
    ("triangle_start", "self_intersect_task_triangle_start"),
    ("triangle_count", "self_intersect_task_triangle_count"),
    ("same_team", "self_intersect_task_same_team"),
)


def _task_columns(declared, uploaded):
    return tuple((declared.index(column_name), plane_name)
                 for column_name, plane_name in uploaded)


CONTACT_TASK_COLUMNS = _task_columns(_contact_plan.CONTACT_TASK_COLUMNS,
                                     CONTACT_TASK_UPLOADED_COLUMNS)
CONTACT_TASK_PAIR_OFFSET_PLANE = "self_contact_task_pair_offsets"
CONTACT_TASK_SOURCE_COUNT_COLUMN = _contact_plan.CONTACT_TASK_COLUMNS.index("source_count")
CONTACT_TASK_TARGET_COUNT_COLUMN = _contact_plan.CONTACT_TASK_COLUMNS.index("target_count")

INTERSECT_TASK_COLUMNS = _task_columns(_contact_plan.INTERSECT_TASK_COLUMNS,
                                       INTERSECT_TASK_UPLOADED_COLUMNS)
INTERSECT_TASK_PAIR_OFFSET_PLANE = "self_intersect_task_pair_offsets"
INTERSECT_TASK_SOURCE_COUNT_COLUMN = _contact_plan.INTERSECT_TASK_COLUMNS.index("edge_count")
INTERSECT_TASK_TARGET_COUNT_COLUMN = \
    _contact_plan.INTERSECT_TASK_COLUMNS.index("triangle_count")

SELF_COUNTER_PLANE = "self_counters"


def _packed_primitive_bits(arena_arrays, field_name, element_count):
    rows = arena_arrays[field_name][:element_count].astype(np.uint8)
    return np.ascontiguousarray(rows[:, 0] | (rows[:, 1] << 1) | (rows[:, 2] << 2))


class ClothEngine:

    def __init__(self, world):
        self.world = None
        self.signature = None
        self.program = None
        self.state = None
        self.plans = {}
        self.config_shadow = None
        self.zone_shadow = None
        self.zone_count = 0
        self.self_task_shadow = None
        self.self_task_cache = None
        self.self_upload_shadow = None
        self.self_upload_totals = (0, 0)
        self.self_empty_uploaded = True
        self.uploaded_field_count = 0
        self.load(world)

    @staticmethod
    def _structure_signature(world):
        team = world.team
        parts = [len(team).to_bytes(8, "little")]
        for name in STRUCTURE_TEAM_COLUMNS:
            parts.append(np.ascontiguousarray(team[name]).tobytes())
        colliders = world.colliders.arrays
        for name in ("team", "kind"):
            parts.append(colliders[name].tobytes())
        point_pairs = world.point_pairs.arrays
        for name in ("team", "particle", "collider"):
            parts.append(point_pairs[name].tobytes())
        edge_pairs = world.edge_pairs.arrays
        for name in ("team", "edge", "collider"):
            parts.append(edge_pairs[name].tobytes())
        return b"".join(parts)

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
        signature = self._structure_signature(world)
        if self.world is world and self.signature == signature:
            return False
        self.world = world
        self.signature = signature
        self.program = _program.build_program(world)
        self.zone_count = 0
        derived_values = _program.derived_planes(self.program)
        element_counts = {}
        for domain_name, source_kind, source_key, count_attribute in DOMAIN_SOURCE_TABLE:
            element_counts[domain_name] = self._domain_element_count(
                source_kind, source_key, count_attribute)
        plane_counts = {plane_name: int(values.shape[0])
                        for plane_name, values in derived_values.items()}
        self.state = _state.ClothState(element_counts, plane_counts)
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
        self.config_shadow = self._config_fingerprint(world)
        self.zone_shadow = None
        self.self_task_shadow = None
        self.self_task_cache = None
        self.self_upload_shadow = None
        self.self_upload_totals = (0, 0)
        self.self_empty_uploaded = True
        return True

    def _config_fingerprint(self, world):
        team = world.team
        team_count = self.program.num_teams
        return b"".join(np.ascontiguousarray(team[name][:team_count]).tobytes()
                        for name in CONFIG_TEAM_FIELDS)

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

    def _upload_config(self, world):
        fingerprint = self._config_fingerprint(world)
        if fingerprint == self.config_shadow:
            return
        self.config_shadow = fingerprint
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

    @staticmethod
    def _self_task_fingerprint(team, team_count, contact_links):
        component_scale = team["component_world_scale"][:team_count]
        scale_alive = (np.abs(component_scale).min(axis=1) >= _frame.SCALE_EPSILON)
        return b"".join((
            int(team_count).to_bytes(8, "little"),
            _contact_plan.link_fingerprint(contact_links, team_count),
            team["enabled"][:team_count].tobytes(),
            team["valid"][:team_count].tobytes(), scale_alive.tobytes(),
            team["sp_start"][:team_count].tobytes(), team["sp_count"][:team_count].tobytes(),
            team["se_start"][:team_count].tobytes(), team["se_count"][:team_count].tobytes(),
            team["st_start"][:team_count].tobytes(), team["st_count"][:team_count].tobytes()))

    def _fill_task_table(self, tasks, capacity, columns, pair_offset_plane,
                         source_count_column, target_count_column):
        planes = {plane_name: np.zeros(capacity, np.int32) for _column, plane_name in columns}
        pair_offsets = np.zeros(capacity + 1, np.int32)
        running = 0
        for task_index, task in enumerate(tasks):
            for column, plane_name in columns:
                planes[plane_name][task_index] = task[column]
            pair_offsets[task_index] = running
            running += int(task[source_count_column]) * int(task[target_count_column])
        pair_offsets[len(tasks):] = running
        for plane_name, values in planes.items():
            self._write_field(_state.DERIVED_STORAGE_NAME, plane_name, values)
        self._write_field(_state.DERIVED_STORAGE_NAME, pair_offset_plane, pair_offsets)
        return running

    def _prepare_self_frame(self, world, frame_index):
        team = world.team
        team_count = self.program.num_teams
        fingerprint = self._self_task_fingerprint(team, team_count, world.contact_links)
        if fingerprint != self.self_task_shadow or self.self_task_cache is None:
            self.self_task_cache = _contact_plan.build_tasks(team, team_count,
                                                             world.contact_links)
            self.self_task_shadow = fingerprint
        contact, intersect, use_point, use_edge, use_triangle = self.self_task_cache
        if not contact and not intersect and self.self_empty_uploaded:
            return
        self.self_empty_uploaded = (not contact) and (not intersect)
        if fingerprint != self.self_upload_shadow:
            total_contact = self._fill_task_table(
                contact, self.program.self_max_contact_tasks, CONTACT_TASK_COLUMNS,
                CONTACT_TASK_PAIR_OFFSET_PLANE, CONTACT_TASK_SOURCE_COUNT_COLUMN,
                CONTACT_TASK_TARGET_COUNT_COLUMN)
            total_intersect = self._fill_task_table(
                intersect, self.program.self_max_intersect_tasks, INTERSECT_TASK_COLUMNS,
                INTERSECT_TASK_PAIR_OFFSET_PLANE, INTERSECT_TASK_SOURCE_COUNT_COLUMN,
                INTERSECT_TASK_TARGET_COUNT_COLUMN)
            for field_name, values in zip(SELF_TASK_TEAM_FIELDS,
                                          (use_point, use_edge, use_triangle)):
                self._write_field("team", field_name, values)
            self.self_upload_totals = (total_contact, total_intersect)
            self.self_upload_shadow = fingerprint
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
        return {_phase_plan.FLAG_SUBSTEP_COUNT: int(substep_count),
                _phase_plan.FLAG_SELF_ITERATION_COUNT:
                    int(_defs.SELF_COLLISION_SOLVER_ITERATION),
                _phase_plan.FLAG_TOTAL_CONTACT_PAIRS: int(total_contact),
                _phase_plan.FLAG_TOTAL_INTERSECT_PAIRS: int(total_intersect)}

    def plan_for(self, flags):
        key = tuple(sorted(flags.items()))
        held = self.plans.get(key)
        if held is not None:
            return held
        recorded = _plan.Plan()
        for phase_name, substep_index, _self_iteration_index in _phase_plan.frame_plan(flags):
            _wiring.record_phase(recorded, self.state, phase_name,
                                 {SUBSTEP_SCALAR_NAME: substep_index})
        recorded.capture(self.state)
        self.plans[key] = recorded
        return recorded

    def _download_frame_outputs(self, world):
        requests = tuple(("particle", field_name) for field_name in OUTPUT_PARTICLE_FIELDS) \
            + tuple(("team", field_name) for field_name in CONSUMABLE_TEAM_FIELDS)
        values = self.state.read_batch(requests)
        particle_count = self.program.num_particles
        for field_name in OUTPUT_PARTICLE_FIELDS:
            _scatter(world.particles.arrays[field_name], particle_count,
                     values[("particle", field_name)])
        team_count = self.program.num_teams
        for field_name in CONSUMABLE_TEAM_FIELDS:
            _scatter(world.team[field_name], team_count, values[("team", field_name)])

    def download_state(self, world):
        requests = tuple(("team", field_name) for field_name in self.state.field_names("team")
                         if field_name not in HOST_TEAM_FIELDS)
        if self.program.num_particles:
            requests += tuple(("particle", field_name)
                              for field_name in self.state.field_names("particle"))
        if self.program.num_colliders:
            requests += tuple(("collider", field_name)
                              for field_name in self.state.field_names("collider")
                              if field_name not in HOST_COLLIDER_FIELDS)
        values = self.state.read_batch(requests)
        for storage_name, field_name in requests:
            if storage_name == "team":
                _scatter(world.team[field_name], self.program.num_teams,
                         values[(storage_name, field_name)])
                continue
            if storage_name == "particle":
                _scatter(world.particles.arrays[field_name], self.program.num_particles,
                         values[(storage_name, field_name)])
                continue
            _scatter(world.colliders.arrays[field_name], self.program.num_colliders,
                     values[(storage_name, field_name)])

    def step_frame(self, world, frame_globals):
        self.load(world)
        self._upload_team_inputs(world)
        self._upload_transform_worlds(world)
        self._upload_collider_inputs(world)
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
