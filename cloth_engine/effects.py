import ast
import io
import os

import warp as wp
from warp import types as warp_types

SOURCE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

DEVICE_MODULE_NAMES = ("families", "kernels", "dmath")

STATE_ROOT_NAME = "state"

STATE_STRUCTURE_ANNOTATION = "ClothState"

SPATIAL_INDEX_MEMBER_SUFFIX = "_index"

SHAPE_ATTRIBUTE_NAME = "shape"

ARRAY_ANNOTATION_NAMES = ("array", "array2d", "array3d", "array4d")

EFFECT_READ = "read"

EFFECT_WRITE = "write"

EFFECT_ACCUMULATE = "accumulate"

EFFECT_CLEAR = "clear"

EFFECT_SHAPE = "shape"

EFFECT_KINDS = (EFFECT_READ, EFFECT_WRITE, EFFECT_ACCUMULATE, EFFECT_CLEAR, EFFECT_SHAPE)

MUTATING_EFFECT_KINDS = (EFFECT_WRITE, EFFECT_ACCUMULATE, EFFECT_CLEAR)

ATOMIC_SUM = "sum"

ATOMIC_REDUCE = "reduce"

ATOMIC_BUILTIN_OPERATIONS = {"atomic_add": ATOMIC_SUM, "atomic_sub": ATOMIC_SUM,
                             "atomic_max": ATOMIC_REDUCE, "atomic_min": ATOMIC_REDUCE}

SPATIAL_INDEX_BUILTIN_NAMES = ("bvh_query_aabb", "bvh_get_group_root")


def _warp_scalar_type_names():
    names = []
    for name in dir(wp):
        held = getattr(wp, name, None)
        if isinstance(held, type) and warp_types.type_is_scalar(held):
            names.append(name)
    return frozenset(names)


WARP_SCALAR_TYPE_NAMES = _warp_scalar_type_names()

WIDTH_CAST_REASON = (
    "a plane whose scalar type is not the default integer can only be written from a kernel "
    "through the constructor of its own width, so a clear reads as a call around the zero "
    "rather than as the bare zero it is; the set of spellings that count is asked of warp "
    "instead of listed here, because warp owns which scalar types exist and a list here "
    "would agree with it only until warp gains one; without this rule a widened accumulator "
    "plane still records the write and loses the clear, and an accumulation window that "
    "loses its clear silently merges with the window before it")

TERM_PLANE = "plane"

TERM_PARAMETER = "parameter"

TERM_ATOMIC = "atomic"

DYNAMIC_SLOT = None

EXTRACTION_REASON = (
    "the read and write set of a family is what the compiler actually does with the state "
    "structure, so it is read out of the kernel source instead of being declared a second "
    "time next to it; a declaration next to the code is a second source that drifts the "
    "first time somebody edits one of the two, and there is no build step that would catch "
    "the drift, whereas an extraction that fails to claim an access point refuses to build")


def _parent_map(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _module_constant_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _module_aliases(tree):
    aliases = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if alias.name in DEVICE_MODULE_NAMES and alias.asname is None:
                aliases[alias.name] = alias.name
    return aliases


def _is_array_annotation(annotation):
    if not isinstance(annotation, ast.Call):
        return False
    callee = annotation.func
    return isinstance(callee, ast.Attribute) and callee.attr in ARRAY_ANNOTATION_NAMES


def _is_state_annotation(annotation):
    return isinstance(annotation, ast.Name) and annotation.id == STATE_STRUCTURE_ANNOTATION


class DeviceSource:

    def __init__(self, module_name):
        self.module_name = module_name
        self.path = os.path.join(SOURCE_DIRECTORY, module_name + ".py")
        self.text = io.open(self.path, "r", encoding="utf-8").read()
        self.tree = ast.parse(self.text, filename=self.path)
        self.parents = _parent_map(self.tree)
        self.constant_names = _module_constant_names(self.tree)
        self.aliases = _module_aliases(self.tree)
        self.functions = {}
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                self.functions[node.name] = node

    def parent_of(self, node):
        return self.parents.get(id(node))

    def array_parameters(self, function):
        return tuple(argument.arg for argument in function.args.args
                     if _is_array_annotation(argument.annotation))

    def state_parameters(self, function):
        return tuple(argument.arg for argument in function.args.args
                     if _is_state_annotation(argument.annotation))

    def parameter_names(self, function):
        return tuple(argument.arg for argument in function.args.args)

    def state_member_nodes(self, function):
        found = []
        for node in ast.walk(function):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id == STATE_ROOT_NAME:
                found.append(node)
        return found

    def array_parameter_nodes(self, function):
        names = set(self.array_parameters(function))
        found = []
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and node.id in names:
                found.append(node)
        return found


SOURCES = {module_name: DeviceSource(module_name) for module_name in DEVICE_MODULE_NAMES}


class Unclaimed:

    __slots__ = ("module_name", "function_name", "line", "text", "reason")

    def __init__(self, module_name, function_name, line, text, reason):
        self.module_name = module_name
        self.function_name = function_name
        self.line = line
        self.text = text
        self.reason = reason

    def describe(self):
        return "%s.%s line %d : %s  (%s)" % (self.module_name, self.function_name, self.line,
                                             self.text, self.reason)


def segment(source, node):
    return " ".join((ast.get_source_segment(source.text, node) or "").split())[:120]


def _element_slot(source, index_node):
    element = index_node
    if isinstance(element, ast.Tuple) and element.elts:
        element = element.elts[0]
    if isinstance(element, ast.Name) and element.id in source.constant_names:
        return element.id
    if isinstance(element, ast.Constant) and isinstance(element.value, int) \
            and not isinstance(element.value, bool):
        return "index %d" % element.value
    return DYNAMIC_SLOT


def _is_width_cast(node):
    return isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords \
        and isinstance(node.func, ast.Attribute) and node.func.attr in WARP_SCALAR_TYPE_NAMES


def _is_constant_zero(node):
    if _is_width_cast(node):
        return _is_constant_zero(node.args[0])
    return isinstance(node, ast.Constant) and not isinstance(node.value, str) \
        and node.value == 0


def _resolve_callee(source, callee_node):
    if isinstance(callee_node, ast.Name):
        held = source.functions.get(callee_node.id)
        if held is None:
            return None
        return (source.module_name, callee_node.id)
    if not isinstance(callee_node, ast.Attribute) or not isinstance(callee_node.value, ast.Name):
        return None
    module_name = source.aliases.get(callee_node.value.id)
    if module_name is None:
        return None
    other = SOURCES[module_name]
    if callee_node.attr not in other.functions:
        return None
    return (module_name, callee_node.attr)


SPATIAL_INDEX_HANDOFF_REASON = (
    "a spatial index member is a handle and not a plane, so it carries no read or write of "
    "its own and the rule that claims it is the call it is handed to; it used to be claimed "
    "only where a query builtin names it directly, which held while every query sat in the "
    "family that owned it, and the collider surface is read from inside the shared field "
    "function instead, so the handle travels through a device call before any builtin sees "
    "it")


def _is_spatial_index_member(node):
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
        and node.value.id == STATE_ROOT_NAME \
        and node.attr.endswith(SPATIAL_INDEX_MEMBER_SUFFIX)


def _builtin_name(callee_node):
    if isinstance(callee_node, ast.Attribute) and isinstance(callee_node.value, ast.Name) \
            and callee_node.value.id == "wp":
        return callee_node.attr
    return None


class Analysis:

    def __init__(self):
        self.effects = {}
        self.claimed = {}
        self.unclaimed = []
        self.in_progress = set()

    def effects_of(self, module_name, function_name):
        key = (module_name, function_name)
        held = self.effects.get(key)
        if held is not None:
            return held
        if key in self.in_progress:
            return {}
        self.in_progress.add(key)
        collected = self._analyze(SOURCES[module_name], function_name)
        self.in_progress.discard(key)
        self.effects[key] = collected
        return collected

    def _analyze(self, source, function_name):
        function = source.functions[function_name]
        array_parameters = set(source.array_parameters(function))
        state_parameters = set(source.state_parameters(function))
        claimed = set()
        collected = {}

        def term_of(node):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute) \
                    and isinstance(node.value.value, ast.Name) \
                    and node.value.value.id == STATE_ROOT_NAME:
                return (TERM_PLANE, node.value.attr, node.attr)
            if isinstance(node, ast.Name) and node.id in array_parameters:
                return (TERM_PARAMETER, node.id)
            return None

        def record(term, effect_kind, slot):
            collected.setdefault(term, set()).add((effect_kind, slot))

        def claim(node):
            claimed.add(id(node))
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
                claimed.add(id(node.value))

        def refuse(node, reason):
            self.unclaimed.append(Unclaimed(source.module_name, function_name, node.lineno,
                                            segment(source, node), reason))

        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            builtin = _builtin_name(node.func)
            operation = ATOMIC_BUILTIN_OPERATIONS.get(builtin) if builtin else None
            if operation is not None:
                target = node.args[0]
                term = term_of(target)
                if term is None:
                    refuse(target, "an atomic writes into something that is not a state plane")
                    continue
                record(term, EFFECT_ACCUMULATE, _element_slot(source, node.args[1]))
                collected.setdefault((TERM_ATOMIC, term, operation), set()).add((builtin, None))
                claim(target)
                continue
            if builtin in SPATIAL_INDEX_BUILTIN_NAMES:
                target = node.args[0]
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
                        and target.value.id == STATE_ROOT_NAME \
                        and target.attr.endswith(SPATIAL_INDEX_MEMBER_SUFFIX):
                    claimed.add(id(target))
                continue
            callee = _resolve_callee(source, node.func)
            if callee is None:
                continue
            callee_source = SOURCES[callee[0]]
            callee_function = callee_source.functions[callee[1]]
            callee_effects = self.effects_of(callee[0], callee[1])
            callee_parameters = callee_source.parameter_names(callee_function)
            callee_state_parameters = set(callee_source.state_parameters(callee_function))
            for position, argument in enumerate(node.args):
                if position >= len(callee_parameters):
                    break
                parameter_name = callee_parameters[position]
                if parameter_name in callee_state_parameters:
                    if not (isinstance(argument, ast.Name)
                            and argument.id in state_parameters):
                        refuse(argument, "a device function takes the state structure and the "
                                         "caller hands it something else")
                        continue
                    for term, entries in callee_effects.items():
                        if term[0] != TERM_PLANE:
                            continue
                        for effect_kind, slot in entries:
                            record(term, effect_kind, slot)
                    for term, entries in callee_effects.items():
                        if term[0] != TERM_ATOMIC:
                            continue
                        collected.setdefault(term, set()).update(entries)
                    continue
                if _is_spatial_index_member(argument):
                    claim(argument)
                    continue
                term = term_of(argument)
                if term is None:
                    continue
                claim(argument)
                for effect_kind, slot in callee_effects.get((TERM_PARAMETER, parameter_name),
                                                            ()):
                    record(term, effect_kind, slot)
                for held, entries in callee_effects.items():
                    if held[0] != TERM_ATOMIC or held[1] != (TERM_PARAMETER, parameter_name):
                        continue
                    collected.setdefault((TERM_ATOMIC, term, held[2]), set()).update(entries)

        for node in ast.walk(function):
            term = term_of(node)
            if term is None:
                continue
            if id(node) in claimed:
                continue
            parent = source.parent_of(node)
            if isinstance(parent, ast.Attribute) and parent.value is node:
                if parent.attr == SHAPE_ATTRIBUTE_NAME:
                    record(term, EFFECT_SHAPE, DYNAMIC_SLOT)
                    claim(node)
                    continue
                refuse(node, "a state plane is read through the attribute %s" % parent.attr)
                claim(node)
                continue
            if isinstance(parent, ast.Subscript) and parent.value is node:
                slot = _element_slot(source, parent.slice)
                if isinstance(parent.ctx, ast.Store):
                    record(term, EFFECT_WRITE, slot)
                    assignment = source.parent_of(parent)
                    if isinstance(assignment, ast.Assign) and _is_constant_zero(assignment.value):
                        record(term, EFFECT_CLEAR, slot)
                    claim(node)
                    continue
                if isinstance(parent.ctx, ast.Load):
                    record(term, EFFECT_READ, slot)
                    claim(node)
                    continue
                refuse(node, "a state plane element is used in the context %s"
                       % type(parent.ctx).__name__)
                claim(node)
                continue
            refuse(node, "a state plane is used under %s and no rule claims that form"
                   % type(parent).__name__)
            claim(node)

        self.claimed[(source.module_name, function_name)] = claimed
        return collected


ANALYSIS = Analysis()


def _analyze_every_function():
    for module_name in DEVICE_MODULE_NAMES:
        for function_name in SOURCES[module_name].functions:
            ANALYSIS.effects_of(module_name, function_name)


_analyze_every_function()


ACCESS_POINT_STATE_MEMBER = "state member"

ACCESS_POINT_ARRAY_PARAMETER = "array parameter"


def _coverage_of(module_name):
    source = SOURCES[module_name]
    totals = {ACCESS_POINT_STATE_MEMBER: 0, ACCESS_POINT_ARRAY_PARAMETER: 0}
    claimed_counts = {ACCESS_POINT_STATE_MEMBER: 0, ACCESS_POINT_ARRAY_PARAMETER: 0}
    missing = []
    for function_name, function in source.functions.items():
        held = ANALYSIS.claimed.get((module_name, function_name), set())
        for kind, nodes in ((ACCESS_POINT_STATE_MEMBER,
                             source.state_member_nodes(function)),
                            (ACCESS_POINT_ARRAY_PARAMETER,
                             source.array_parameter_nodes(function))):
            for node in nodes:
                totals[kind] += 1
                if id(node) in held:
                    claimed_counts[kind] += 1
                    continue
                missing.append(Unclaimed(module_name, function_name, node.lineno,
                                         segment(source, node),
                                         "no rule claimed this %s access point" % kind))
    return totals, claimed_counts, missing


def coverage():
    rows = []
    missing = []
    for module_name in DEVICE_MODULE_NAMES:
        totals, claimed_counts, module_missing = _coverage_of(module_name)
        for kind in (ACCESS_POINT_STATE_MEMBER, ACCESS_POINT_ARRAY_PARAMETER):
            rows.append((module_name, kind, totals[kind], claimed_counts[kind]))
        missing.extend(module_missing)
    missing.extend(ANALYSIS.unclaimed)
    return tuple(rows), tuple(missing)


COVERAGE_ROWS, COVERAGE_MISSING = coverage()


def _assert_coverage():
    assert not COVERAGE_MISSING, \
        "the side effect extractor did not claim %d state access points and an unclaimed " \
        "access point is a read or a write that no derivation sees, so the frame data flow " \
        "would be derived from an incomplete picture; %s\n%s" \
        % (len(COVERAGE_MISSING), EXTRACTION_REASON,
           "\n".join("  " + entry.describe() for entry in COVERAGE_MISSING[:40]))


_assert_coverage()


def plane_effects(module_name, function_name):
    collected = ANALYSIS.effects_of(module_name, function_name)
    planes = {}
    for term, entries in collected.items():
        if term[0] != TERM_PLANE:
            continue
        planes[(term[1], term[2])] = set(entries)
    return planes


def summing_atomic_planes(module_name, function_name):
    collected = ANALYSIS.effects_of(module_name, function_name)
    planes = set()
    for term in collected:
        if term[0] != TERM_ATOMIC or term[2] != ATOMIC_SUM or term[1][0] != TERM_PLANE:
            continue
        planes.add((term[1][1], term[1][2]))
    return planes


def reducing_atomic_planes(module_name, function_name):
    collected = ANALYSIS.effects_of(module_name, function_name)
    planes = set()
    for term in collected:
        if term[0] != TERM_ATOMIC or term[2] != ATOMIC_REDUCE or term[1][0] != TERM_PLANE:
            continue
        planes.add((term[1][1], term[1][2]))
    return planes


def every_plane_slot():
    slots = {}
    for module_name in DEVICE_MODULE_NAMES:
        for function_name in SOURCES[module_name].functions:
            for plane, entries in plane_effects(module_name, function_name).items():
                for effect_kind, slot in entries:
                    if effect_kind == EFFECT_SHAPE:
                        continue
                    slots.setdefault(plane, set()).add(slot)
    return slots


PLANE_SLOTS = every_plane_slot()

SLOT_ADDRESSED_PLANES = tuple(sorted(plane for plane, slots in PLANE_SLOTS.items()
                                     if DYNAMIC_SLOT not in slots))


def addressed_slot(plane, slot):
    if plane in SLOT_ADDRESSED_PLANES:
        return slot
    return DYNAMIC_SLOT


def family_effects(family_names):
    table = {}
    for family_name in family_names:
        planes = plane_effects("families", family_name)
        rows = {}
        for plane, entries in planes.items():
            for effect_kind, slot in entries:
                rows.setdefault(effect_kind, set()).add((plane, addressed_slot(plane, slot)))
        table[family_name] = {effect_kind: frozenset(rows.get(effect_kind, ()))
                              for effect_kind in EFFECT_KINDS}
    return table


def family_summing_planes(family_names):
    return {family_name: frozenset(summing_atomic_planes("families", family_name))
            for family_name in family_names}


def family_reducing_planes(family_names):
    return {family_name: frozenset(reducing_atomic_planes("families", family_name))
            for family_name in family_names}
