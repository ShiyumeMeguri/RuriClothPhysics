"""Which bones a config simulates, and in what role.

One truth source. The overlay colours bones from this, the operators decide what to add and remove
from this, and the compiler builds its topology from this -- previously the traversal lived inside
build_snapshot alone, so anything else that wanted to know "is this bone simulated" had to
re-derive it and could silently disagree with the solver.
"""

ROLE_ROOT = 'ROOT'
ROLE_CHILD = 'CHILD'


def is_selected(obj, name):
    """Blender 5.x keeps bone selection on PoseBone / EditBone -- Bone.select no longer exists.

    Every caller goes through here so the version-specific spelling lives in one place instead of
    being copied into the overlay, the operators and the property callbacks.
    """
    edit_bones = obj.data.edit_bones if obj.mode == 'EDIT' else None
    if edit_bones is not None:
        bone = edit_bones.get(name)
        return bool(bone and bone.select)
    bone = obj.pose.bones.get(name) if obj.pose else None
    return bool(bone and bone.select)


def selected_names(obj):
    if obj.mode == 'EDIT':
        return [bone.name for bone in obj.data.edit_bones if bone.select]
    if obj.pose is None:
        return []
    return [bone.name for bone in obj.pose.bones if bone.select]


def select(obj, names, active=None):
    """Replace the armature's bone selection with `names` and activate `active`."""
    wanted = set(names)
    if obj.mode == 'EDIT':
        for bone in obj.data.edit_bones:
            bone.select = bone.name in wanted
            bone.select_head = bone.select
            bone.select_tail = bone.select
        if active:
            bone = obj.data.edit_bones.get(active)
            if bone is not None:
                obj.data.edit_bones.active = bone
        return
    if obj.pose is not None:
        for bone in obj.pose.bones:
            bone.select = bone.name in wanted
    if active:
        bone = obj.data.bones.get(active)
        if bone is not None:
            obj.data.bones.active = bone


def root_names(obj, config):
    """Declared roots that still resolve to a real bone, de-duplicated, in list order."""
    bones = obj.data.bones
    seen = set()
    names = []
    for item in config.root_bones:
        name = item.bone
        if name and name in bones and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def descend(obj, roots):
    """Depth-first bone names under `roots`, roots included, each visited once."""
    bones = obj.data.bones
    visited = set()
    ordered = []

    def visit(bone):
        if bone.name in visited:
            return
        visited.add(bone.name)
        ordered.append(bone.name)
        for child in bone.children:
            visit(child)

    for name in roots:
        bone = bones.get(name)
        if bone is not None:
            visit(bone)
    return ordered


def config_chain(obj, config):
    roots = root_names(obj, config)
    return roots, descend(obj, roots)


def role_map(obj, config_indices=None):
    """{bone_name: (config_index, role)}.

    A bone claimed by several configs keeps the first claim, and a root always outranks a child --
    the same precedence the compiler applies, so the colours cannot say one thing while the solver
    does another.
    """
    settings = getattr(obj, "ruri_cloth_physics", None)
    if settings is None:
        return {}
    mapping = {}
    for index, config in enumerate(settings.configs):
        if config_indices is not None and index not in config_indices:
            continue
        roots, ordered = config_chain(obj, config)
        root_set = set(roots)
        for name in ordered:
            role = ROLE_ROOT if name in root_set else ROLE_CHILD
            current = mapping.get(name)
            if current is None or (current[1] == ROLE_CHILD and role == ROLE_ROOT):
                mapping[name] = (index, role)
    return mapping


# Calibrated against the two hubs on this rig, which look identical in shape and behave nothing
# alike. SleeveC_Jnt sits at 0.016-0.020 of its config's median gap and whips at 67 deg per frame;
# BowB_Jnt is also a 5 mm stub with two children but sits at 0.16-0.31 and is perfectly calm at
# 3.3 deg. 0.05 is three times above the broken one and three times below the healthy one.
DEGENERATE_RATIO = 0.05


def _head(bone):
    return bone.head_local


def degenerate_links(obj, config):
    """Child bones that start almost on top of their parent, which the solver cannot handle.

    A constraint whose rest length is a rounding error next to the particle radius fights itself,
    and a hub solved from near-coincident children has no stable orientation: measured on this rig,
    SleeveC_Jnt sat 1.0-1.3 mm from both its children against 60-90 mm everywhere else in the same
    config, and whipped at 67 deg per frame while every other chain stayed under half a degree.

    The test is scale free -- a gap is only suspicious against the OTHER gaps of the same config --
    so a legitimately dense chain, where every link is short, reports nothing.
    """
    import statistics

    roots, ordered = config_chain(obj, config)
    if not ordered:
        return []
    bones = obj.data.bones
    inside = set(ordered)
    gaps = []
    for name in ordered:
        bone = bones.get(name)
        parent = bone.parent if bone is not None else None
        if parent is None or parent.name not in inside:
            continue
        head = _head(bone)
        parent_head = _head(parent)
        distance = ((head[0] - parent_head[0]) ** 2 + (head[1] - parent_head[1]) ** 2
                    + (head[2] - parent_head[2]) ** 2) ** 0.5
        gaps.append((name, parent.name, distance))
    if len(gaps) < 2:
        return []
    median = statistics.median(distance for _, _, distance in gaps)
    if median <= 0.0:
        return []
    return [(name, parent, distance, median) for name, parent, distance in gaps
            if distance < median * DEGENERATE_RATIO]


def unpinned_roots(obj, config):
    """Roots whose attribute override drags them off FIXED.

    compile.py fills MOVE for every bone, then FIXED for the roots, then lets overrides REPLACE the
    result -- so an override saying MOVE on a root removes the only anchor the chain has and it
    drifts away under gravity. The override is usually harmless when it is written, because the bone
    is mid-chain and MOVE is what it would have been anyway; it only turns fatal later, when that
    bone is promoted to a root. Measured: promoting SleeveC_01/03 with their old MOVE overrides
    still in place sent the chain 2.6 m off the body over 240 frames.
    """
    declared = set(root_names(obj, config))
    return [(override.bone, override.attribute) for override in config.attribute_overrides
            if override.bone in declared and override.attribute != 'FIXED']


def owning_root(obj, bone_name):
    """Walk up from `bone_name` to the first bone that is a declared root of any config.

    Returns (config_index, root_name) or (None, None). This is what lets "remove" work when the
    user has a mid-chain bone selected: the thing they mean to delete is the chain they are
    standing in, and its handle is the root.
    """
    settings = getattr(obj, "ruri_cloth_physics", None)
    if settings is None:
        return None, None
    declared = {}
    for index, config in enumerate(settings.configs):
        for name in root_names(obj, config):
            declared.setdefault(name, index)
    bone = obj.data.bones.get(bone_name)
    while bone is not None:
        if bone.name in declared:
            return declared[bone.name], bone.name
        bone = bone.parent
    return None, None
