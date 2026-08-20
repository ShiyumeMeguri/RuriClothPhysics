import ast
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BODIES_PATH = os.path.join(HERE, "phase_bodies.py")
KERNELS_PATH = os.path.join(HERE, "kernels.py")
OUTPUT_PATH = os.path.join(HERE, "phases.py")

THREAD_SOURCES = (
    ("tid", "cuda.grid(1)"),
    ("stride", "cuda.gridsize(1)"),
    ("bid", "cuda.blockIdx.x"),
    ("bdim", "cuda.blockDim.x"),
)

ALIAS_SOURCES = (
    ("_k", "k"),
    ("_sit", "sit"),
)

SCALAR_SOURCES = (
    ("fdt", "scal_f[SCAL_FRAME_DT]"),
    ("sim_dt", "scal_f[SCAL_SIM_DT]"),
    ("global_time_scale", "scal_f[SCAL_TIME_SCALE]"),
    ("power1", "scal_f[SCAL_POWER1]"),
    ("power2", "scal_f[SCAL_POWER2]"),
    ("power3", "scal_f[SCAL_POWER3]"),
    ("max_sim_count", "scal_i[SCAL_MAX_SIM]"),
    ("n_zones", "scal_i[SCAL_N_ZONES]"),
)

COUNT_SOURCES = (
    ("num_teams", "t_enabled.shape[0]"),
    ("num_particles", "p_team.shape[0]"),
    ("num_colliders", "c_team.shape[0]"),
    ("num_triangles", "st_triangle_team.shape[0]"),
    ("num_self_points", "sfp_team.shape[0]"),
    ("num_self_edges", "sfe_team.shape[0]"),
    ("num_self_triangles", "sft_team.shape[0]"),
    ("num_fk_levels", "fk_yes_offsets.shape[0] - 1"),
    ("num_angle_passes", "angle_pass_offsets.shape[0] - 1"),
    ("num_postline_levels", "postline_entry_offsets.shape[0] - 1"),
    ("num_ct_slots", "ct_pair_off.shape[0] - 1"),
    ("num_it_slots", "it_pair_off.shape[0] - 1"),
    ("n_tether", "st_tether_particle.shape[0]"),
    ("n_move", "st_move_particle.shape[0]"),
    ("n_fixed", "st_fixed_particle.shape[0]"),
    ("n_spring", "st_spring_particle.shape[0]"),
    ("n_motion", "st_motion_particle.shape[0]"),
    ("n_bending", "st_bending_team.shape[0]"),
    ("n_baseline", "baseline_entries.shape[0]"),
    ("n_angle_buffered", "st_angle_buffered_particle.shape[0]"),
    ("ee_cap", "ee_my.shape[0]"),
    ("pt_cap", "pt_my.shape[0]"),
    ("ip_cap", "ip_edge.shape[0]"),
)

AMBIENT_NAMES = frozenset((
    "math", "cuda", "dmath", "libdevice",
    "float32", "float64", "int8", "int32", "uint8",
    "range", "abs", "len", "min", "max", "not",
))

HEADER = (
    "import math",
    "",
    "from numba import cuda, float32, float64, int8, int32, uint8",
    "from numba.cuda import libdevice",
    "",
    "from . import dmath",
    "from .slots import *",
)


def _module_constants(path):
    tree = ast.parse(io.open(path, "r", encoding="utf-8").read())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value
    return values


def _kernel_exports(path):
    tree = ast.parse(io.open(path, "r", encoding="utf-8").read())
    exported = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            exported.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    exported.add(target.id)
    return exported


def _signature():
    constants = _module_constants(KERNELS_PATH)
    resident = ast.literal_eval(constants["RESIDENT_BLOB_GROUPS"])
    zone = ast.literal_eval(constants["ZONE_BLOB_GROUPS"])
    return (["scal_f", "scal_i"]
            + ["blob_%s" % group for group in resident]
            + ["offs", "lens"]
            + ["zone_%s" % group for group in zone]
            + ["zone_offs", "zone_lens", "k", "sit"])


def _slot_bindings():
    constants = _module_constants(KERNELS_PATH)
    bindings = {}
    for layout, blob_prefix, offset_name, length_name in (
            ("RESIDENT_BLOB_LAYOUT", "blob_", "offs", "lens"),
            ("ZONE_BLOB_LAYOUT", "zone_", "zone_offs", "zone_lens")):
        for name, group, _per_row in ast.literal_eval(constants[layout]):
            assert name not in bindings, "duplicate slot %s" % name
            bindings[name] = "%s%s[%s[S_%s]:%s[S_%s] + %s[S_%s]]" % (
                blob_prefix, group, offset_name, name, offset_name, name, length_name, name)
    return bindings


def _expression_names(source):
    return {node.id for node in ast.walk(ast.parse(source, mode="eval"))
            if isinstance(node, ast.Name)}


def _body_names(function):
    assigned = set()
    loaded = []
    seen = set()
    for statement in function.body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Name):
                continue
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif node.id not in seen:
                seen.add(node.id)
                loaded.append(node.id)
    return [entry for entry in loaded if entry not in assigned]


def _phase_declarations():
    source = io.open(BODIES_PATH, "r", encoding="utf-8", newline="").read()
    eol = "\r\n" if "\r\n" in source else "\n"
    lines = source.split(eol)
    tree = ast.parse(source)
    declarations = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.decorator_list:
            continue
        decorator = node.decorator_list[0]
        assert isinstance(decorator, ast.Call) and decorator.func.id == "phase", node.name
        context = ast.literal_eval(decorator.args[0])
        grid_domain = ast.literal_eval(decorator.args[1])
        body_start = node.body[0].lineno
        body_lines = lines[body_start - 1:node.end_lineno]
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        declarations.append({
            "name": node.name,
            "context": context,
            "grid_domain": grid_domain,
            "parameters": [argument.arg for argument in node.args.args],
            "free": _body_names(node),
            "body": body_lines,
        })
    return declarations, eol


def render():
    signature = _signature()
    slot_bindings = _slot_bindings()
    kernel_exports = _kernel_exports(KERNELS_PATH)
    thread_sources = dict(THREAD_SOURCES)
    alias_sources = dict(ALIAS_SOURCES)
    scalar_sources = dict(SCALAR_SOURCES)
    count_sources = dict(COUNT_SOURCES)
    bindable = (set(thread_sources) | set(alias_sources) | set(scalar_sources)
                | set(count_sources) | set(slot_bindings))
    overlap = bindable & set(signature)
    assert not overlap, "binding names collide with the signature: %s" % sorted(overlap)

    declarations, eol = _phase_declarations()
    imported = set()
    blocks = []
    for declaration in declarations:
        name = declaration["name"]
        required = [entry for entry in declaration["free"] if entry in bindable]
        declared = declaration["parameters"]
        assert set(required) == set(declared), \
            "%s: declared %s but body needs %s" % (name, sorted(set(declared) - set(required)),
                                                   sorted(set(required) - set(declared)))
        unknown = [entry for entry in declaration["free"]
                   if entry not in bindable and entry not in AMBIENT_NAMES
                   and entry not in signature and not entry.startswith("S_")]
        for entry in unknown:
            assert entry in kernel_exports, "%s: unresolved name %s" % (name, entry)
            imported.add(entry)

        emitted = set(declared)
        for entry in list(emitted):
            if entry in count_sources:
                emitted |= _expression_names(count_sources[entry]) & bindable

        prologue = []
        for entry, expression in THREAD_SOURCES:
            if entry in emitted:
                prologue.append((entry, expression))
        for entry, expression in ALIAS_SOURCES:
            if entry in emitted:
                prologue.append((entry, expression))
        for group in (scalar_sources, slot_bindings, count_sources):
            for entry in sorted(group):
                if entry in emitted:
                    prologue.append((entry, group[entry]))

        for _entry, expression in prologue:
            for reference in _expression_names(expression):
                if reference in kernel_exports and reference not in bindable:
                    imported.add(reference)

        block = ["@cuda.jit(cache=True)",
                 "def %s(%s):" % (name, ", ".join(signature))]
        block.extend("    %s = %s" % pair for pair in prologue)
        block.extend(declaration["body"])
        blocks.append(eol.join(block))

    table = ["PHASE_TABLE = ("]
    for declaration in declarations:
        table.append("    (%r, %r, %r)," % (declaration["name"], declaration["context"],
                                            declaration["grid_domain"]))
    table.append(")")

    kernel_import = ["from .kernels import ("]
    for entry in sorted(imported):
        kernel_import.append("    %s," % entry)
    kernel_import.append(")")

    document = eol.join(list(HEADER) + kernel_import + ["", ""])
    document += eol + (eol + eol + eol).join(blocks) + eol + eol + eol
    document += eol.join(table) + eol
    return document


def main():
    document = render()
    ast.parse(document)
    io.open(OUTPUT_PATH, "w", encoding="utf-8", newline="").write(document)
    return document


if __name__ == "__main__":
    main()
