from . import phases as _phases
from . import state as _state

SLOT_BINDINGS = (
    ("csr_distance_offsets", "derived", "distance_csr_offsets"),
    ("csr_distance_order", "derived", "distance_csr_order"),
    ("p_attr_move", "particle", "attr_move"),
    ("p_base_positions", "particle", "base_positions"),
    ("p_depth", "particle", "depth"),
    ("p_friction", "particle", "friction"),
    ("p_local_normals", "particle", "local_normals"),
    ("p_local_positions", "particle", "local_positions"),
    ("p_local_tangents", "particle", "local_tangents"),
    ("p_next_positions", "particle", "next_positions"),
    ("p_positions", "particle", "positions"),
    ("p_rotations", "particle", "rotations"),
    ("p_skin_indices", "particle", "skin_indices"),
    ("p_skin_weights", "particle", "skin_weights"),
    ("p_step_basic_positions", "particle", "step_basic_positions"),
    ("p_team", "particle", "team"),
    ("p_velocity_positions", "particle", "velocity_positions"),
    ("p_vertex_root", "particle", "vertex_root"),
    ("sc_dcorr", "derived", "distance_correction"),
    ("st_distance_rest", "distance", "rest"),
    ("st_distance_target", "distance", "target"),
    ("st_tether_particle", "tether", "particle"),
    ("st_tether_team", "tether", "team"),
    ("t_animation_pose_ratio", "team", "animation_pose_ratio"),
    ("t_cws", "team", "component_world_scale"),
    ("t_distance_lut", "team", "distance_lut"),
    ("t_enabled", "team", "enabled"),
    ("t_frame_dt", "team", "frame_delta_time"),
    ("t_frame_old", "team", "frame_old_time"),
    ("t_frame_update", "team", "frame_update_time"),
    ("t_init_scale", "team", "init_scale"),
    ("t_is_spring", "team", "is_spring"),
    ("t_now_time_scale", "team", "now_time_scale"),
    ("t_now_update", "team", "now_update_time"),
    ("t_old_time", "team", "old_time"),
    ("t_old_update", "team", "old_update_time"),
    ("t_running", "team", "running"),
    ("t_scale_ratio", "team", "scale_ratio"),
    ("t_skip_count", "team", "skip_count"),
    ("t_tether_compression", "team", "tether_compression"),
    ("t_time", "team", "time"),
    ("t_time_reset", "team", "time_reset_pending"),
    ("t_time_scale", "team", "time_scale"),
    ("t_update_count", "team", "update_count"),
    ("t_valid", "team", "valid"),
    ("x_bind", "transform", "bind_pose"),
    ("x_world", "transform", "world"),
)

SCALAR_NAMES = ("fdt", "global_time_scale", "k", "max_sim_count", "power1", "sim_dt")


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
                       for phase_name, _kernel, launch_domain in _phases.PHASE_TABLE}

PHASE_KERNEL = {phase_name: kernel for phase_name, kernel, _launch_domain in _phases.PHASE_TABLE}


def argument_names(kernel):
    return tuple(argument.label for argument in kernel.adj.args)


def _validate_phase_table():
    seen = set()
    for phase_name, kernel, launch_domain in _phases.PHASE_TABLE:
        assert phase_name not in seen, "phase %s is declared twice" % phase_name
        seen.add(phase_name)
        assert kernel.key == phase_name, \
            "phase row %s carries the kernel %s, a row must carry the kernel of the same name" \
            % (phase_name, kernel.key)
        assert launch_domain in _state.DOMAIN_NAMES, \
            "phase %s launches over %r which is not a state domain" % (phase_name, launch_domain)
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
    plan.record(kernel, state.element_count(launch_domain),
                phase_inputs(state, kernel, scalars))
