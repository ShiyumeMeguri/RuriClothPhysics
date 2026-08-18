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
