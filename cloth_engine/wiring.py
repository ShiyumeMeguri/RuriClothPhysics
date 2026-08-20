from . import phases as _phases
from . import plan as _plan
from . import state as _state

SLOT_BINDINGS = (
    ("c_active", "collider", "active"),
    ("c_enabled", "collider", "enabled"),
    ("c_enabled_prev", "collider", "enabled_prev"),
    ("c_frame_pos", "collider", "frame_positions"),
    ("c_frame_radius", "collider", "frame_radii"),
    ("c_frame_rot", "collider", "frame_rotations"),
    ("c_frame_tip", "collider", "frame_tips"),
    ("c_input_positions", "collider", "input_positions"),
    ("c_input_radii", "collider", "input_radii"),
    ("c_input_rotations", "collider", "input_rotations"),
    ("c_input_tips", "collider", "input_tips"),
    ("c_now_pos", "collider", "now_positions"),
    ("c_now_rot", "collider", "now_rotations"),
    ("c_now_tip", "collider", "now_tips"),
    ("c_old_frame_pos", "collider", "old_frame_positions"),
    ("c_old_frame_rot", "collider", "old_frame_rotations"),
    ("c_old_frame_tip", "collider", "old_frame_tips"),
    ("c_old_pos", "collider", "old_positions"),
    ("c_old_rot", "collider", "old_rotations"),
    ("c_old_tip", "collider", "old_tips"),
    ("c_team", "collider", "team"),
    ("csr_center_fixed_offsets", "derived", "center_fixed_csr_offsets"),
    ("csr_center_fixed_order", "derived", "center_fixed_csr_order"),
    ("csr_distance_offsets", "derived", "distance_csr_offsets"),
    ("csr_distance_order", "derived", "distance_csr_order"),
    ("p_attr_move", "particle", "attr_move"),
    ("p_base_positions", "particle", "base_positions"),
    ("p_base_rotations", "particle", "base_rotations"),
    ("p_collision_normals", "particle", "collision_normals"),
    ("p_depth", "particle", "depth"),
    ("p_display_positions", "particle", "display_positions"),
    ("p_friction", "particle", "friction"),
    ("p_local_normals", "particle", "local_normals"),
    ("p_local_positions", "particle", "local_positions"),
    ("p_local_tangents", "particle", "local_tangents"),
    ("p_next_positions", "particle", "next_positions"),
    ("p_old_anim_positions", "particle", "old_anim_positions"),
    ("p_old_anim_rotations", "particle", "old_anim_rotations"),
    ("p_old_positions", "particle", "old_positions"),
    ("p_old_rotations", "particle", "old_rotations"),
    ("p_positions", "particle", "positions"),
    ("p_real_velocities", "particle", "real_velocities"),
    ("p_rotations", "particle", "rotations"),
    ("p_skin_indices", "particle", "skin_indices"),
    ("p_skin_weights", "particle", "skin_weights"),
    ("p_static_friction", "particle", "static_friction"),
    ("p_step_basic_positions", "particle", "step_basic_positions"),
    ("p_team", "particle", "team"),
    ("p_velocities", "particle", "velocities"),
    ("p_velocity_positions", "particle", "velocity_positions"),
    ("p_vertex_bind_pose_rotations", "particle", "vertex_bind_pose_rotations"),
    ("p_vertex_root", "particle", "vertex_root"),
    ("sc_dcorr", "derived", "distance_correction"),
    ("sc_sync", "derived", "synchronization_snapshot"),
    ("scal_f", "frame_scalar", "frame_float"),
    ("scal_i", "frame_scalar", "frame_int"),
    ("st_center_fixed_particle", "center_fixed", "particle"),
    ("st_distance_rest", "distance", "rest"),
    ("st_distance_target", "distance", "target"),
    ("st_tether_particle", "tether", "particle"),
    ("st_tether_team", "tether", "team"),
    ("t_anchor_component_local_position", "team", "anchor_component_local_position"),
    ("t_anchor_inertia", "team", "anchor_inertia"),
    ("t_anchor_position", "team", "anchor_position"),
    ("t_anchor_rotation", "team", "anchor_rotation"),
    ("t_animation_pose_ratio", "team", "animation_pose_ratio"),
    ("t_blend_weight", "team", "blend_weight"),
    ("t_component_world_position", "team", "component_world_position"),
    ("t_component_world_rotation", "team", "component_world_rotation"),
    ("t_culling_invisible", "team", "culling_invisible"),
    ("t_cws", "team", "component_world_scale"),
    ("t_distance_lut", "team", "distance_lut"),
    ("t_enabled", "team", "enabled"),
    ("t_frame_component_shift_rotation", "team", "frame_component_shift_rotation"),
    ("t_frame_component_shift_vector", "team", "frame_component_shift_vector"),
    ("t_frame_dt", "team", "frame_delta_time"),
    ("t_frame_moving_direction", "team", "frame_moving_direction"),
    ("t_frame_moving_speed", "team", "frame_moving_speed"),
    ("t_frame_old", "team", "frame_old_time"),
    ("t_frame_update", "team", "frame_update_time"),
    ("t_frame_world_position", "team", "frame_world_position"),
    ("t_frame_world_rotation", "team", "frame_world_rotation"),
    ("t_frame_world_scale", "team", "frame_world_scale"),
    ("t_had_anchor", "team", "had_anchor"),
    ("t_has_anchor", "team", "has_anchor"),
    ("t_inertia_shift", "team", "inertia_shift"),
    ("t_init_scale", "team", "init_scale"),
    ("t_is_negative_scale", "team", "is_negative_scale"),
    ("t_is_spring", "team", "is_spring"),
    ("t_keep_teleport_pending", "team", "keep_teleport_pending"),
    ("t_movement_inertia_smoothing", "team", "movement_inertia_smoothing"),
    ("t_movement_speed_limit", "team", "movement_speed_limit"),
    ("t_negative_scale_change", "team", "negative_scale_change"),
    ("t_negative_scale_direction", "team", "negative_scale_direction"),
    ("t_negative_scale_matrix", "team", "negative_scale_matrix"),
    ("t_negative_scale_quaternion", "team", "negative_scale_quaternion"),
    ("t_negative_scale_sign", "team", "negative_scale_sign"),
    ("t_negative_scale_teleport", "team", "negative_scale_teleport"),
    ("t_negative_scale_triangle_sign", "team", "negative_scale_triangle_sign"),
    ("t_now_time_scale", "team", "now_time_scale"),
    ("t_now_update", "team", "now_update_time"),
    ("t_now_world_position", "team", "now_world_position"),
    ("t_now_world_rotation", "team", "now_world_rotation"),
    ("t_old_anchor_position", "team", "old_anchor_position"),
    ("t_old_anchor_rotation", "team", "old_anchor_rotation"),
    ("t_old_component_world_position", "team", "old_component_world_position"),
    ("t_old_component_world_rotation", "team", "old_component_world_rotation"),
    ("t_old_component_world_scale", "team", "old_component_world_scale"),
    ("t_old_frame_world_position", "team", "old_frame_world_position"),
    ("t_old_frame_world_rotation", "team", "old_frame_world_rotation"),
    ("t_old_frame_world_scale", "team", "old_frame_world_scale"),
    ("t_old_time", "team", "old_time"),
    ("t_old_update", "team", "old_update_time"),
    ("t_old_world_position", "team", "old_world_position"),
    ("t_old_world_rotation", "team", "old_world_rotation"),
    ("t_reset_pending", "team", "reset_pending"),
    ("t_rotation_speed_limit", "team", "rotation_speed_limit"),
    ("t_running", "team", "running"),
    ("t_scale_ratio", "team", "scale_ratio"),
    ("t_skip_count", "team", "skip_count"),
    ("t_smoothing_velocity", "team", "smoothing_velocity"),
    ("t_stablization_time", "team", "stablization_time"),
    ("t_sync_target", "team", "sync_target"),
    ("t_sync_top", "team", "sync_top"),
    ("t_teleport_distance", "team", "teleport_distance"),
    ("t_teleport_mode", "team", "teleport_mode"),
    ("t_teleport_rotation", "team", "teleport_rotation"),
    ("t_tether_compression", "team", "tether_compression"),
    ("t_time", "team", "time"),
    ("t_time_reset", "team", "time_reset_pending"),
    ("t_time_scale", "team", "time_scale"),
    ("t_update_count", "team", "update_count"),
    ("t_valid", "team", "valid"),
    ("t_velocity_weight", "team", "velocity_weight"),
    ("t_wind_count", "team", "wind_count"),
    ("t_wind_direction", "team", "wind_direction"),
    ("t_wind_dirq", "team", "wind_dirq"),
    ("t_wind_influence", "team", "wind_influence"),
    ("t_wind_main", "team", "wind_main"),
    ("t_wind_time", "team", "wind_time"),
    ("t_wind_zone_id", "team", "wind_zone_id"),
    ("t_wind_zone_turbulence", "team", "wind_zone_turbulence"),
    ("t_world_inertia", "team", "world_inertia"),
    ("x_bind", "transform", "bind_pose"),
    ("x_world", "transform", "world"),
    ("z_attenuation_lut", "zone", "attenuation_lut"),
    ("z_is_addition", "zone", "is_addition"),
    ("z_main", "zone", "main"),
    ("z_mode", "zone", "mode"),
    ("z_size", "zone", "size"),
    ("z_turbulence", "zone", "turbulence"),
    ("z_world_direction", "zone", "world_direction"),
    ("z_world_position", "zone", "world_position"),
    ("z_world_to_local", "zone", "world_to_local"),
    ("z_zone_id", "zone", "zone_id"),
    ("z_zone_volume", "zone", "zone_volume"),
)

SCALAR_NAMES = ("k",)


def _validate_bindings():
    seen = set()
    for row in SLOT_BINDINGS:
        assert len(row) == 3, \
            "a slot binding row declares slot name, storage name and field name, got %r" % (row,)
        slot_name, storage_name, field_name = row
        assert slot_name not in seen, "slot %s is bound twice" % slot_name
        seen.add(slot_name)
        assert storage_name in _state.STORAGE_NAMES, \
            "slot %s binds to storage %r which the state layer does not declare" \
            % (slot_name, storage_name)
        assert field_name in _state.STORAGE_FIELDS[storage_name], \
            "slot %s binds to %s.%s which the state layer does not declare" \
            % (slot_name, storage_name, field_name)
    collision = seen & set(SCALAR_NAMES)
    assert not collision, \
        "these names are declared both as a bound slot and as a scalar: %s" % sorted(collision)


_validate_bindings()

SLOT_SOURCE = {slot_name: (storage_name, field_name)
               for slot_name, storage_name, field_name in SLOT_BINDINGS}

PHASE_LAUNCH_DOMAIN = {phase_name: launch_domain
                       for phase_name, _kernel, launch_domain, _launch_strategy
                       in _phases.PHASE_TABLE}

PHASE_LAUNCH_STRATEGY = {phase_name: launch_strategy
                         for phase_name, _kernel, _launch_domain, launch_strategy
                         in _phases.PHASE_TABLE}

PHASE_KERNEL = {phase_name: kernel
                for phase_name, kernel, _launch_domain, _launch_strategy
                in _phases.PHASE_TABLE}


def argument_names(kernel):
    return tuple(argument.label for argument in kernel.adj.args)


def _validate_phase_table():
    seen = set()
    for phase_name, kernel, launch_domain, launch_strategy in _phases.PHASE_TABLE:
        assert phase_name not in seen, "phase %s is declared twice" % phase_name
        seen.add(phase_name)
        assert kernel.key == phase_name, \
            "phase row %s carries the kernel %s, a row must carry the kernel of the same name" \
            % (phase_name, kernel.key)
        assert launch_domain in _state.DOMAIN_NAMES, \
            "phase %s launches over %r which is not a state domain" % (phase_name, launch_domain)
        assert launch_strategy in _plan.LAUNCH_STRATEGIES, \
            "phase %s declares the launch strategy %r, only %r are defined" \
            % (phase_name, launch_strategy, _plan.LAUNCH_STRATEGIES)
        for name in argument_names(kernel):
            assert name in SLOT_SOURCE or name in SCALAR_NAMES, \
                "phase %s takes the argument %s which is neither a bound slot nor a declared " \
                "scalar" % (phase_name, name)


_validate_phase_table()


def phase_inputs(state, kernel, scalars):
    inputs = []
    for name in argument_names(kernel):
        source = SLOT_SOURCE.get(name)
        if source is None:
            assert name in scalars, \
                "the scalar %s required by %s was not supplied" % (name, kernel.key)
            inputs.append(scalars[name])
            continue
        inputs.append(state.array(source[0], source[1]))
    return inputs


def record_phase(plan, state, phase_name, scalars):
    kernel = PHASE_KERNEL[phase_name]
    launch_domain = PHASE_LAUNCH_DOMAIN[phase_name]
    launch_strategy = PHASE_LAUNCH_STRATEGY[phase_name]
    plan.record(kernel,
                _plan.launch_dimension(launch_strategy, state.element_count(launch_domain)),
                phase_inputs(state, kernel, scalars))
