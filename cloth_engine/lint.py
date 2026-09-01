import ast

from . import effects as _effects

UNUSED_LOCAL_REASON = (
    "a local that is assigned and never read is a computation the compiler will delete, and "
    "it is there because a transcription kept the left hand side after its reader moved or "
    "was dropped; whichever it is, the source says something the binary does not, and the "
    "next reader has to work out which of the two is the truth")

DUPLICATE_READ_REASON = (
    "two statements that read the same plane element into the same local with nothing "
    "between them that could change either the index or the plane are one statement written "
    "twice; the second read cannot produce a different value, so it is transcription "
    "residue rather than a load the algorithm asked for")


def _statement_blocks(function):
    blocks = []
    for node in ast.walk(function):
        for field_name in ("body", "orelse", "finalbody"):
            held = getattr(node, field_name, None)
            if isinstance(held, list) and held:
                blocks.append(held)
    return blocks


def _single_name_target(statement):
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    if not isinstance(target, ast.Name):
        return None
    return target.id


def _is_pure_read(statement):
    if _single_name_target(statement) is None:
        return False
    for node in ast.walk(statement.value):
        if isinstance(node, ast.Call):
            return False
    return True


def _loaded_names(node):
    return {inner.id for inner in ast.walk(node)
            if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load)}


def unused_local_assignments():
    rows = []
    for module_name in _effects.DEVICE_MODULE_NAMES:
        source = _effects.SOURCES[module_name]
        for function_name in sorted(source.functions):
            function = source.functions[function_name]
            loaded = _loaded_names(function)
            for statement in ast.walk(function):
                target_name = _single_name_target(statement)
                if target_name is None or target_name in loaded:
                    continue
                rows.append((module_name, function_name, statement.lineno, target_name,
                             _effects.segment(source, statement)))
    return tuple(sorted(rows))


def _duplicates_in_block(source, module_name, function_name, block):
    rows = []
    index = 0
    while index < len(block):
        if not _is_pure_read(block[index]):
            index += 1
            continue
        end = index
        while end < len(block) and _is_pure_read(block[end]):
            end += 1
        run = block[index:end]
        first_seen = {}
        for position, statement in enumerate(run):
            key = ast.dump(statement)
            earlier = first_seen.get(key)
            if earlier is None:
                first_seen[key] = position
                continue
            between = {_single_name_target(held) for held in run[earlier + 1:position]}
            guarded = _loaded_names(statement.value) | {_single_name_target(statement)}
            if between & guarded:
                first_seen[key] = position
                continue
            rows.append((module_name, function_name, run[earlier].lineno, statement.lineno,
                         _effects.segment(source, statement)))
        index = end
    return rows


def duplicate_adjacent_reads():
    rows = []
    for module_name in _effects.DEVICE_MODULE_NAMES:
        source = _effects.SOURCES[module_name]
        for function_name in sorted(source.functions):
            function = source.functions[function_name]
            for block in _statement_blocks(function):
                rows.extend(_duplicates_in_block(source, module_name, function_name, block))
    return tuple(sorted(rows))


UNUSED_LOCAL_ASSIGNMENTS = unused_local_assignments()

DUPLICATE_ADJACENT_READS = duplicate_adjacent_reads()


def assert_clean():
    assert not UNUSED_LOCAL_ASSIGNMENTS, \
        "%s\n%s" % (UNUSED_LOCAL_REASON,
                    "\n".join("  %s.%s line %d assigns %s and never reads it : %s"
                              % (module_name, function_name, line, target_name, text)
                              for module_name, function_name, line, target_name, text
                              in UNUSED_LOCAL_ASSIGNMENTS))
    assert not DUPLICATE_ADJACENT_READS, \
        "%s\n%s" % (DUPLICATE_READ_REASON,
                    "\n".join("  %s.%s line %d repeats line %d verbatim : %s"
                              % (module_name, function_name, later, first, text)
                              for module_name, function_name, first, later, text
                              in DUPLICATE_ADJACENT_READS))
