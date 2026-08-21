CONTEXT_LOOP = "loop"
CONTEXT_IF = "if"
CONTEXT_ELSE = "else"

CONTEXT_KINDS = (CONTEXT_LOOP, CONTEXT_IF, CONTEXT_ELSE)

FLAG_SUBSTEP_COUNT = "sub_end"
FLAG_SELF_ITERATION_COUNT = "self_iterations"
FLAG_TOTAL_CONTACT_PAIRS = "total_ct"
FLAG_TOTAL_INTERSECT_PAIRS = "total_it"

FLAG_NAMES = (FLAG_SUBSTEP_COUNT, FLAG_SELF_ITERATION_COUNT, FLAG_TOTAL_CONTACT_PAIRS,
              FLAG_TOTAL_INTERSECT_PAIRS)

LOOP_VARIABLE_SUBSTEP = "_k"
LOOP_VARIABLE_SELF_ITERATION = "_sit"

LOOP_BINDINGS = {
    LOOP_VARIABLE_SUBSTEP: (FLAG_SUBSTEP_COUNT, 0),
    LOOP_VARIABLE_SELF_ITERATION: (FLAG_SELF_ITERATION_COUNT, 1),
}

PHASE_SEQUENCE = (
    ("phase_00", ()),
    ("phase_01", ()),
    ("phase_02", ()),
    ("phase_03", ()),
    ("phase_03b", ()),
    ("phase_04", ()),
    ("phase_05", ()),
    ("phase_07", (("if", "total_it > 0"),)),
    ("phase_08", (("if", "total_it > 0"),)),
    ("phase_10", (("loop", "_k"),)),
    ("phase_11", (("loop", "_k"),)),
    ("phase_12", (("loop", "_k"),)),
    ("phase_13", (("loop", "_k"),)),
    ("phase_14", (("loop", "_k"),)),
    ("phase_15", (("loop", "_k"),)),
    ("phase_16", (("loop", "_k"),)),
    ("phase_17", (("loop", "_k"),)),
    ("phase_18", (("loop", "_k"),)),
    ("phase_19", (("loop", "_k"),)),
    ("phase_20", (("loop", "_k"),)),
    ("phase_21", (("loop", "_k"),)),
    ("phase_22", (("loop", "_k"),)),
    ("phase_23", (("loop", "_k"),)),
    ("phase_24", (("loop", "_k"),)),
    ("phase_25", (("loop", "_k"),)),
    ("phase_26", (("loop", "_k"),)),
    ("phase_27", (("loop", "_k"),)),
    ("phase_28", (("loop", "_k"),)),
    ("phase_29", (("loop", "_k"),)),
    ("phase_30", (("loop", "_k"),)),
    ("phase_33", (("loop", "_k"), ("if", "total_ct > 0"), ("if", "_k == 0"))),
    ("phase_34", (("loop", "_k"), ("if", "total_ct > 0"), ("if", "_k == 0"))),
    ("phase_35", (("loop", "_k"), ("if", "total_ct > 0"), ("if", "_k == 0"))),
    ("phase_36", (("loop", "_k"), ("if", "total_ct > 0"), ("if", "_k == 0"))),
    ("phase_37", (("loop", "_k"), ("if", "total_ct > 0"), ("if", "_k == 0"))),
    ("phase_38", (("loop", "_k"), ("if", "total_ct > 0"), ("else", "_k == 0"))),
    ("phase_39", (("loop", "_k"), ("if", "total_ct > 0"))),
    ("phase_40", (("loop", "_k"), ("if", "total_ct > 0"), ("loop", "_sit"))),
    ("phase_41", (("loop", "_k"), ("if", "total_ct > 0"), ("loop", "_sit"))),
    ("phase_42", (("loop", "_k"),)),
    ("phase_43", (("loop", "_k"),)),
    ("phase_44", (("loop", "_k"),)),
    ("phase_45", (("if", "total_it > 0"),)),
    ("phase_46", (("if", "total_it > 0"),)),
    ("phase_47", ()),
    ("phase_48", ()),
    ("phase_49", ()),
    ("phase_50", ()),
    ("phase_51", ()),
    ("phase_52", ()),
    ("phase_53", ()),
)


def _predicate_total_contact_pairs(flags, substep_index, self_iteration_index):
    return flags[FLAG_TOTAL_CONTACT_PAIRS] > 0


def _predicate_total_intersect_pairs(flags, substep_index, self_iteration_index):
    return flags[FLAG_TOTAL_INTERSECT_PAIRS] > 0


def _predicate_first_substep(flags, substep_index, self_iteration_index):
    return substep_index == 0


PREDICATES = {
    "total_ct > 0": _predicate_total_contact_pairs,
    "total_it > 0": _predicate_total_intersect_pairs,
    "_k == 0": _predicate_first_substep,
}

PER_FRAME_PREDICATE_EXPRESSIONS = tuple(
    expression for expression in PREDICATES
    if not any(loop_variable in expression for loop_variable in LOOP_BINDINGS))


def _validate_phase_sequence():
    seen = set()
    for row in PHASE_SEQUENCE:
        assert len(row) == 2, \
            "a phase sequence row declares the phase name and its context, got %r" % (row,)
        phase_name, context = row
        assert isinstance(phase_name, str) and phase_name, \
            "a phase sequence row must name the phase, got %r" % (phase_name,)
        assert phase_name not in seen, "phase %s is declared twice" % phase_name
        seen.add(phase_name)
        assert isinstance(context, tuple), \
            "phase %s declares its context as a tuple of frames, got %r" % (phase_name, context)
        for frame in context:
            assert isinstance(frame, tuple) and len(frame) == 2, \
                "phase %s declares a context frame as kind and expression, got %r" \
                % (phase_name, (frame,))
            kind, expression = frame
            assert kind in CONTEXT_KINDS, \
                "phase %s declares the context kind %r, only %r are defined" \
                % (phase_name, kind, CONTEXT_KINDS)
            if kind == CONTEXT_LOOP:
                assert expression in LOOP_BINDINGS, \
                    "phase %s loops over %r which no loop binding declares, only %r are bound" \
                    % (phase_name, expression, tuple(LOOP_BINDINGS))
                continue
            assert expression in PREDICATES, \
                "phase %s tests %r which no predicate declares, only %r are defined" \
                % (phase_name, expression, tuple(PREDICATES))


def _validate_loop_bindings():
    positions = set()
    for loop_variable, (flag_name, position) in LOOP_BINDINGS.items():
        assert flag_name in FLAG_NAMES, \
            "the loop %s counts from the flag %s which the flag table does not declare" \
            % (loop_variable, flag_name)
        assert position not in positions, \
            "the loop %s reuses the loop state position %d" % (loop_variable, position)
        positions.add(position)
    assert positions == set(range(len(LOOP_BINDINGS))), \
        "the loop state positions have to be a dense range, they are %r" % (sorted(positions),)


_validate_phase_sequence()
_validate_loop_bindings()

PHASE_NAMES = tuple(phase_name for phase_name, _context in PHASE_SEQUENCE)

PHASE_CONTEXT = {phase_name: context for phase_name, context in PHASE_SEQUENCE}


def phase_tree(entries, depth):
    out = []
    index = 0
    while index < len(entries):
        phase_name, context = entries[index]
        if len(context) == depth:
            out.append((None, phase_name, None))
            index += 1
            continue
        head = context[depth]
        end = index
        while (end < len(entries) and len(entries[end][1]) > depth
               and entries[end][1][depth] == head):
            end += 1
        out.append((head, None, phase_tree(entries[index:end], depth + 1)))
        index = end
    return out


PHASE_TREE = phase_tree(PHASE_SEQUENCE, 0)


def flatten_plan(tree, substep_index, self_iteration_index, flags, out):
    for head, phase_name, subtree in tree:
        if phase_name is not None:
            out.append((phase_name, substep_index, self_iteration_index))
            continue
        kind, expression = head
        if kind == CONTEXT_LOOP:
            flag_name, position = LOOP_BINDINGS[expression]
            loop_state = [substep_index, self_iteration_index]
            for value in range(flags[flag_name]):
                loop_state[position] = value
                flatten_plan(subtree, loop_state[0], loop_state[1], flags, out)
            continue
        taken = PREDICATES[expression](flags, substep_index, self_iteration_index)
        if kind == CONTEXT_ELSE:
            taken = not taken
        if taken:
            flatten_plan(subtree, substep_index, self_iteration_index, flags, out)


def frame_plan(flags):
    out = []
    flatten_plan(PHASE_TREE, 0, 0, flags, out)
    return tuple(out)
