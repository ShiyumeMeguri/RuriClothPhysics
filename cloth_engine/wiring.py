from . import phases as _phases
from . import plan as _plan
from . import state as _state

SLOT_BINDINGS = (
    ("baseline_entries", "derived", "baseline_entries"),
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
    ("c_kind", "collider", "kind"),
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
    ("c_work_aabb_max", "collider", "work_aabb_max"),
    ("c_work_aabb_min", "collider", "work_aabb_min"),
    ("c_work_inv_old_rot", "collider", "work_inv_old_rot"),
    ("c_work_next_pos", "collider", "work_next_pos"),
    ("c_work_old_pos", "collider", "work_old_pos"),
    ("c_work_radius", "collider", "work_radius"),
    ("c_work_rot", "collider", "work_rot"),
    ("csr_center_fixed_offsets", "derived", "center_fixed_csr_offsets"),
    ("csr_center_fixed_order", "derived", "center_fixed_csr_order"),
    ("csr_distance_offsets", "derived", "distance_csr_offsets"),
    ("csr_distance_order", "derived", "distance_csr_order"),
    ("fk_no", "derived", "fk_no"),
    ("fk_no_offsets", "derived", "fk_no_offsets"),
    ("fk_yes", "derived", "fk_yes"),
    ("fk_yes_offsets", "derived", "fk_yes_offsets"),
    ("fk_yes_parent", "derived", "fk_yes_parent"),
    ("p_albuf_length", "particle", "albuf_length"),
    ("p_albuf_local_pos", "particle", "albuf_local_pos"),
    ("p_albuf_local_rot", "particle", "albuf_local_rot"),
    ("p_albuf_restore", "particle", "albuf_restore"),
    ("p_albuf_rotation", "particle", "albuf_rotation"),
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
    ("p_step_basic_rotations", "particle", "step_basic_rotations"),
    ("p_team", "particle", "team"),
    ("p_velocities", "particle", "velocities"),
    ("p_velocity_positions", "particle", "velocity_positions"),
    ("p_vertex_bind_pose_rotations", "particle", "vertex_bind_pose_rotations"),
    ("p_vertex_local_positions", "particle", "vertex_local_positions"),
    ("p_vertex_local_rotations", "particle", "vertex_local_rotations"),
    ("p_vertex_parent", "particle", "vertex_parent"),
    ("p_vertex_root", "particle", "vertex_root"),
    ("p_vertex_root_local", "particle", "vertex_root_local"),
    ("sc_dcorr", "derived", "distance_correction"),
    ("sc_sync", "derived", "synchronization_snapshot"),
    ("scal_f", "frame_scalar", "frame_float"),
    ("scal_i", "frame_scalar", "frame_int"),
    ("st_angle_buffered_particle", "angle_buffered", "particle"),
    ("st_center_fixed_particle", "center_fixed", "particle"),
    ("st_distance_rest", "distance", "rest"),
    ("st_distance_target", "distance", "target"),
    ("st_fixed_particle", "update_fixed", "particle"),
    ("st_fixed_team", "update_fixed", "team"),
    ("st_move_particle", "update_move", "particle"),
    ("st_move_team", "update_move", "team"),
    ("st_spring_particle", "spring", "particle"),
    ("st_spring_team", "spring", "team"),
    ("st_tether_particle", "tether", "particle"),
    ("st_tether_team", "tether", "team"),
    ("t_anchor_component_local_position", "team", "anchor_component_local_position"),
    ("t_anchor_inertia", "team", "anchor_inertia"),
    ("t_anchor_position", "team", "anchor_position"),
    ("t_anchor_rotation", "team", "anchor_rotation"),
    ("t_angle_use_limit", "team", "angle_use_limit"),
    ("t_angle_use_restoration", "team", "angle_use_restoration"),
    ("t_angular_velocity", "team", "angular_velocity"),
    ("t_animation_pose_ratio", "team", "animation_pose_ratio"),
    ("t_blend_weight", "team", "blend_weight"),
    ("t_blend_weight_param", "team", "blend_weight_param"),
    ("t_component_world_position", "team", "component_world_position"),
    ("t_component_world_rotation", "team", "component_world_rotation"),
    ("t_culling_invisible", "team", "culling_invisible"),
    ("t_cws", "team", "component_world_scale"),
    ("t_damping_lut", "team", "damping_lut"),
    ("t_depth_inertia", "team", "depth_inertia"),
    ("t_distance_lut", "team", "distance_lut"),
    ("t_distance_weight", "team", "distance_weight"),
    ("t_enabled", "team", "enabled"),
    ("t_force_mode", "team", "force_mode"),
    ("t_frame_component_shift_rotation", "team", "frame_component_shift_rotation"),
    ("t_frame_component_shift_vector", "team", "frame_component_shift_vector"),
    ("t_frame_dt", "team", "frame_delta_time"),
    ("t_frame_interpolation", "team", "frame_interpolation"),
    ("t_frame_moving_direction", "team", "frame_moving_direction"),
    ("t_frame_moving_speed", "team", "frame_moving_speed"),
    ("t_frame_old", "team", "frame_old_time"),
    ("t_frame_update", "team", "frame_update_time"),
    ("t_frame_world_position", "team", "frame_world_position"),
    ("t_frame_world_rotation", "team", "frame_world_rotation"),
    ("t_frame_world_scale", "team", "frame_world_scale"),
    ("t_gravity", "team", "gravity"),
    ("t_gravity_direction", "team", "gravity_direction"),
    ("t_gravity_dot", "team", "gravity_dot"),
    ("t_gravity_falloff", "team", "gravity_falloff"),
    ("t_gravity_ratio", "team", "gravity_ratio"),
    ("t_had_anchor", "team", "had_anchor"),
    ("t_has_anchor", "team", "has_anchor"),
    ("t_impact_force", "team", "impact_force"),
    ("t_inertia_rotation", "team", "inertia_rotation"),
    ("t_inertia_shift", "team", "inertia_shift"),
    ("t_inertia_vector", "team", "inertia_vector"),
    ("t_init_local_gravity_direction", "team", "init_local_gravity_direction"),
    ("t_init_scale", "team", "init_scale"),
    ("t_is_negative_scale", "team", "is_negative_scale"),
    ("t_is_spring", "team", "is_spring"),
    ("t_keep_teleport_pending", "team", "keep_teleport_pending"),
    ("t_local_inertia", "team", "local_inertia"),
    ("t_local_movement_speed_limit", "team", "local_movement_speed_limit"),
    ("t_local_rotation_speed_limit", "team", "local_rotation_speed_limit"),
    ("t_movement_inertia_smoothing", "team", "movement_inertia_smoothing"),
    ("t_movement_speed_limit", "team", "movement_speed_limit"),
    ("t_moving_wind_direction", "team", "moving_wind_direction"),
    ("t_moving_wind_dirq", "team", "moving_wind_dirq"),
    ("t_moving_wind_main", "team", "moving_wind_main"),
    ("t_moving_wind_time", "team", "moving_wind_time"),
    ("t_negative_scale_change", "team", "negative_scale_change"),
    ("t_negative_scale_direction", "team", "negative_scale_direction"),
    ("t_negative_scale_matrix", "team", "negative_scale_matrix"),
    ("t_negative_scale_quaternion", "team", "negative_scale_quaternion"),
    ("t_negative_scale_sign", "team", "negative_scale_sign"),
    ("t_negative_scale_teleport", "team", "negative_scale_teleport"),
    ("t_negative_scale_triangle_sign", "team", "negative_scale_triangle_sign"),
    ("t_normal_axis_vector", "team", "normal_axis_vector"),
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
    ("t_rotation_axis", "team", "rotation_axis"),
    ("t_rotation_speed_limit", "team", "rotation_speed_limit"),
    ("t_running", "team", "running"),
    ("t_scale_ratio", "team", "scale_ratio"),
    ("t_skip_count", "team", "skip_count"),
    ("t_smoothing_velocity", "team", "smoothing_velocity"),
    ("t_spring_limit_distance", "team", "spring_limit_distance"),
    ("t_spring_noise", "team", "spring_noise"),
    ("t_spring_normal_limit_ratio", "team", "spring_normal_limit_ratio"),
    ("t_spring_power", "team", "spring_power"),
    ("t_stablization_time", "team", "stablization_time"),
    ("t_step_move_inertia_ratio", "team", "step_move_inertia_ratio"),
    ("t_step_rotation", "team", "step_rotation"),
    ("t_step_rotation_inertia_ratio", "team", "step_rotation_inertia_ratio"),
    ("t_step_vector", "team", "step_vector"),
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
    ("t_wind_blend", "team", "wind_blend"),
    ("t_wind_count", "team", "wind_count"),
    ("t_wind_depth_weight", "team", "wind_depth_weight"),
    ("t_wind_direction", "team", "wind_direction"),
    ("t_wind_dirq", "team", "wind_dirq"),
    ("t_wind_frequency", "team", "wind_frequency"),
    ("t_wind_influence", "team", "wind_influence"),
    ("t_wind_main", "team", "wind_main"),
    ("t_wind_moving", "team", "wind_moving"),
    ("t_wind_seed", "team", "wind_seed"),
    ("t_wind_synchronization", "team", "wind_synchronization"),
    ("t_wind_time", "team", "wind_time"),
    ("t_wind_turbulence", "team", "wind_turbulence"),
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

LAUNCH_SCALAR_NAMES = (_phases.LEVEL_SCALAR_NAME,)


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

PHASE_NAMES = tuple(phase_name for phase_name, _rule, _passes in _phases.PHASE_TABLE)

PHASE_LAUNCH_RULE = {phase_name: rule
                     for phase_name, rule, _passes in _phases.PHASE_TABLE}

PHASE_PASSES = {phase_name: passes for phase_name, _rule, passes in _phases.PHASE_TABLE}

PHASE_KERNELS = {phase_name: tuple(row[0] for row in passes)
                 for phase_name, passes in PHASE_PASSES.items()}

PHASE_LAUNCH_SLOTS = {phase_name: tuple(row[1] for row in passes)
                      for phase_name, passes in PHASE_PASSES.items()}

PHASE_LEVEL_SLOTS = {phase_name: tuple(row[2] for row in passes)
                     for phase_name, passes in PHASE_PASSES.items()
                     if PHASE_LAUNCH_RULE[phase_name] == _phases.PHASE_LAUNCH_PER_LEVEL}

PASS_WIDTH = {_phases.PHASE_LAUNCH_ONCE: 2, _phases.PHASE_LAUNCH_PER_LEVEL: 3}


def argument_names(kernel):
    return tuple(argument.label for argument in kernel.adj.args)


def phase_argument_names(phase_name):
    return argument_names(PHASE_KERNELS[phase_name][0])


def _validate_pass_name(phase_name, kernel_key, pass_count):
    if pass_count == 1:
        assert kernel_key == phase_name, \
            "phase %s runs a single pass so its kernel has to carry the name of the phase, " \
            "the row carries %s" % (phase_name, kernel_key)
        return
    assert kernel_key.startswith(phase_name + "_") and len(kernel_key) > len(phase_name) + 1, \
        "phase %s runs several passes so every pass kernel has to be named %s_<pass>, the " \
        "row carries %s" % (phase_name, phase_name, kernel_key)


def _validate_pass_slots(phase_name, rule, row, names):
    for slot_name in row[1:]:
        assert slot_name in SLOT_SOURCE, \
            "phase %s launches a pass over %r which is not a bound slot" \
            % (phase_name, slot_name)
        assert slot_name in names, \
            "phase %s launches a pass over %r but the kernel %s does not take that slot, a " \
            "launch extent comes from an array the pass itself reads" \
            % (phase_name, slot_name, row[0].key)
    if rule == _phases.PHASE_LAUNCH_PER_LEVEL:
        assert names[0] == _phases.LEVEL_SCALAR_NAME, \
            "phase %s runs one launch per level so the kernel %s has to take %s as its first " \
            "argument, it takes %s" \
            % (phase_name, row[0].key, _phases.LEVEL_SCALAR_NAME, names[0])
        return
    assert _phases.LEVEL_SCALAR_NAME not in names, \
        "phase %s runs one launch per pass so the kernel %s must not take %s" \
        % (phase_name, row[0].key, _phases.LEVEL_SCALAR_NAME)


def _validate_phase_table():
    seen = set()
    seen_kernels = set()
    for phase_name, rule, passes in _phases.PHASE_TABLE:
        assert phase_name not in seen, "phase %s is declared twice" % phase_name
        seen.add(phase_name)
        assert rule in _phases.PHASE_LAUNCH_RULES, \
            "phase %s declares the launch rule %r, only %r are defined" \
            % (phase_name, rule, _phases.PHASE_LAUNCH_RULES)
        assert passes, \
            "phase %s declares no pass, a phase is at least one kernel launch" % phase_name
        signature = argument_names(passes[0][0])
        for row in passes:
            assert len(row) == PASS_WIDTH[rule], \
                "phase %s runs under the launch rule %r so every pass row carries %d entries, " \
                "the row carries %d" % (phase_name, rule, PASS_WIDTH[rule], len(row))
            kernel = row[0]
            assert kernel.key not in seen_kernels, \
                "the kernel %s appears in more than one phase pass" % kernel.key
            seen_kernels.add(kernel.key)
            _validate_pass_name(phase_name, kernel.key, len(passes))
            names = argument_names(kernel)
            assert names == signature, \
                "phase %s pass %s takes %r while its first pass takes %r, every pass of a " \
                "phase carries the signature of the reference phase" \
                % (phase_name, kernel.key, list(names), list(signature))
            for name in names:
                assert (name in SLOT_SOURCE or name in SCALAR_NAMES
                        or name in LAUNCH_SCALAR_NAMES), \
                    "phase %s takes the argument %s which is neither a bound slot nor a " \
                    "declared scalar" % (phase_name, name)
            _validate_pass_slots(phase_name, rule, row, names)


_validate_phase_table()


def slot_extent(state, slot_name):
    storage_name, field_name = SLOT_SOURCE[slot_name]
    return state.plane_element_count(storage_name, field_name)


def phase_level_count(state, phase_name):
    declared = set()
    for _kernel, _entries_slot, offsets_slot in PHASE_PASSES[phase_name]:
        declared.add(slot_extent(state, offsets_slot) - 1)
    assert len(declared) == 1, \
        "phase %s runs its passes over the same levels so every offset plane holds the same " \
        "number of levels, the state holds %r" % (phase_name, sorted(declared))
    level_count = declared.pop()
    assert level_count >= 0, \
        "phase %s reads a level offset plane of %d elements, a level offset plane holds one " \
        "element more than the number of levels" % (phase_name, level_count + 1)
    return level_count


def phase_launches(state, phase_name):
    passes = PHASE_PASSES[phase_name]
    if PHASE_LAUNCH_RULE[phase_name] == _phases.PHASE_LAUNCH_ONCE:
        return tuple((kernel, _plan.launch_dimension(slot_extent(state, slot_name)), {})
                     for kernel, slot_name in passes)
    launches = []
    for level in range(phase_level_count(state, phase_name)):
        for kernel, entries_slot, _offsets_slot in passes:
            launches.append((kernel,
                             _plan.launch_dimension(slot_extent(state, entries_slot)),
                             {_phases.LEVEL_SCALAR_NAME: level}))
    return tuple(launches)


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
    for kernel, dimension, launch_scalars in phase_launches(state, phase_name):
        supplied = dict(scalars)
        supplied.update(launch_scalars)
        plan.record(kernel, dimension, phase_inputs(state, kernel, supplied))
