ROLE_ROOT = 'ROOT'
ROLE_CHILD = 'CHILD'


def active_config_index(settings):
    if len(settings.configs) == 0:
        return -1
    return min(settings.active_config_index, len(settings.configs) - 1)


def is_selected(obj, name):
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
    bones = obj.data.bones
    seen = set()
    names = []
    for item in config.root_bones:
        name = item.bone
        if name and name in bones and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def descend(obj, roots, stop=None):
    bones = obj.data.bones
    stop = stop or frozenset()
    visited = set()
    ordered = []

    def visit(bone):
        if bone.name in visited:
            return
        visited.add(bone.name)
        ordered.append(bone.name)
        for child in bone.children:
            if child.name in stop:
                continue
            visit(child)

    for name in roots:
        bone = bones.get(name)
        if bone is not None:
            visit(bone)
    return ordered


def excluded_names(obj, config):
    return {override.bone for override in config.attribute_overrides
            if override.attribute == 'IGNORE' and override.bone}


def boundary_names(obj, config):
    settings = getattr(obj, "ruri_cloth_physics", None)
    if settings is None:
        return excluded_names(obj, config)
    mine = set(root_names(obj, config))
    boundary = set()
    for other in settings.configs:
        if other == config:
            continue
        boundary.update(name for name in root_names(obj, other) if name not in mine)
    boundary.update(excluded_names(obj, config))
    return boundary


def config_chain(obj, config):
    roots = root_names(obj, config)
    return roots, descend(obj, roots, stop=boundary_names(obj, config))


def claimed_twice(obj):
    settings = getattr(obj, "ruri_cloth_physics", None)
    if settings is None:
        return []
    owners = {}
    for config in settings.configs:
        for name in root_names(obj, config):
            owners.setdefault(name, []).append(config.name)
    return [(name, names) for name, names in owners.items() if len(names) > 1]


def role_map(obj, config_indices=None):
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


CONNECTED_LINK_REASON = (
    "which bone places which head is asked here and nowhere else, because three readers "
    "need the same answer and a second copy of the rule is a second answer: the write path "
    "turns and stretches a parent so its tail lands on its connected child's particle, the "
    "bake has to carry the scale channel of exactly the bones that write does it through, "
    "and the panel has to name the bones the rig cannot place at all; the rule takes the "
    "bone table and the names in play rather than an object, so the write path can ask it "
    "about one solver batch and the panel about one armature and both get the same answer")


def connected_links(bones, names):
    position_of = {name: position for position, name in enumerate(names)}
    links = []
    for position, name in enumerate(names):
        bone = bones.get(name)
        if bone is None or not bone.use_connect or bone.parent is None:
            continue
        parent = position_of.get(bone.parent.name, -1)
        if parent < 0:
            continue
        links.append((parent, position))
    return links


def driven_names(obj):
    settings = getattr(obj, "ruri_cloth_physics", None)
    if settings is None:
        return []
    names = []
    seen = set()
    for config in settings.configs:
        if not config.enabled:
            continue
        for name in config_chain(obj, config)[1]:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def stretch_owners(obj):
    names = driven_names(obj)
    return {names[parent] for parent, _child in connected_links(obj.data.bones, names)}


def parent_owned_heads(obj):
    names = driven_names(obj)
    return {names[child] for _parent, child in connected_links(obj.data.bones, names)}


def forked_connections(obj):
    names = driven_names(obj)
    children_of = {}
    for parent, child in connected_links(obj.data.bones, names):
        children_of.setdefault(names[parent], []).append(names[child])
    return sorted((parent, children) for parent, children in children_of.items()
                  if len(children) > 1)


DEGENERATE_RATIO = 0.05


def _head(bone):
    return bone.head_local


def degenerate_links(obj, config):
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
    declared = set(root_names(obj, config))
    return [(override.bone, override.attribute) for override in config.attribute_overrides
            if override.bone in declared and override.attribute != 'FIXED']


def owning_root(obj, bone_name):
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
