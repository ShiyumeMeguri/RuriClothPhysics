import json
import os

import bpy

FORMAT_VERSION = 5

SCOPE_OBJECT = 'OBJECT'
SCOPE_CONFIG = 'CONFIG'
SCOPE_COLLIDERS = 'COLLIDERS'

MODE_REPLACE = 'REPLACE'
MODE_APPEND = 'APPEND'

COLLIDER_REFERENCE_GROUP = 'RCPColliderReference'

SKIP_EXACT = frozenset({
    "rna_type", "node_name", "param_serial", "rebuild_pending",
})


def _skip(identifier):
    if identifier in SKIP_EXACT:
        return True
    if identifier.endswith("_uid"):
        return True
    return identifier.startswith("active_") and identifier.endswith("_index")


def _is_id_pointer(prop):
    declared = getattr(bpy.types, prop.fixed_type.identifier, None)
    return declared is not None and issubclass(declared, bpy.types.ID)


def _read_value(owner, prop):
    value = getattr(owner, prop.identifier)
    if prop.type == 'POINTER':
        if _is_id_pointer(prop):
            return value.name if value is not None else None
        return read_group(value)
    if prop.type == 'COLLECTION':
        return [read_group(item) for item in value]
    if getattr(prop, "is_array", False):
        return [v for v in value]
    return value


def read_group(group):
    data = {}
    for prop in group.bl_rna.properties:
        if _skip(prop.identifier):
            continue
        data[prop.identifier] = _read_value(group, prop)
    return data


class LoadState:

    def __init__(self, armature_object, context):
        self.armature_object = armature_object
        self.context = context
        self.report = {}
        self.colliders = {}


def _resolve_pointer(owner, prop, value, state):
    target = bpy.data.objects.get(value) if value else None
    if value and target is None:
        state.report.setdefault("missing_objects", []).append(value)
    return target


COLLIDER_NAME_IS_PRESET_LOCAL_REASON = (
    "the name a preset writes for a collider is the name that collider had in the file the "
    "preset was taken from, and loading the preset builds new objects that blender may have "
    "had to rename, so the name is looked up in the table the build just filled rather than "
    "in the scene: taking it from the scene would bind the row to whatever object happened "
    "to already carry that name, which after one reload is the collider of the character "
    "the preset was copied from")


def _write_collider_reference(owner, value, state):
    from . import collider_binding
    target = state.colliders.get(value) if value else None
    if value and target is None:
        state.report.setdefault("missing_colliders", []).append(value)
    owner.collider_uid = collider_binding.bind(target) if target is not None else ""


def _write_value(owner, prop, value, state):
    identifier = prop.identifier
    if owner.bl_rna.identifier == COLLIDER_REFERENCE_GROUP and identifier == "collider":
        _write_collider_reference(owner, value, state)
        return
    if prop.type == 'POINTER':
        if _is_id_pointer(prop):
            setattr(owner, identifier, _resolve_pointer(owner, prop, value, state))
            return
        if isinstance(value, dict):
            write_group(getattr(owner, identifier), value, state)
        return
    if prop.type == 'COLLECTION':
        collection = getattr(owner, identifier)
        collection.clear()
        for entry in value or ():
            write_group(collection.add(), entry, state)
        return
    if getattr(prop, "is_array", False):
        setattr(owner, identifier, tuple(value))
        return
    setattr(owner, identifier, value)


def write_group(group, data, state):
    properties = {prop.identifier: prop for prop in group.bl_rna.properties}
    ordered = [key for key in data if key != "points_serialized"]
    if "points_serialized" in data:
        ordered.append("points_serialized")
    for key in ordered:
        prop = properties.get(key)
        if prop is None or _skip(key):
            state.report.setdefault("unknown_properties", []).append(key)
            continue
        if prop.is_readonly and not (prop.type == 'COLLECTION'
                                     or (prop.type == 'POINTER' and not _is_id_pointer(prop))):
            continue
        try:
            _write_value(group, prop, data[key], state)
        except (TypeError, ValueError) as error:
            state.report.setdefault("rejected", []).append("%s: %s" % (key, error))


def _referenced_colliders(configs):
    from . import collider_binding
    from . import collider_geom
    for config in configs:
        for reference in config.collider_collision.collider_references:
            target = collider_binding.resolve(reference.collider_uid)
            if collider_geom.is_collider(target):
                yield target


def collider_set(configs, scene, armature_object, include_owned):
    from . import collider_geom
    candidates = list(_referenced_colliders(configs))
    if include_owned:
        candidates.extend(collider_geom.owned_colliders(scene, armature_object))
    ordered = []
    seen = set()
    for target in candidates:
        if target.name in seen:
            continue
        seen.add(target.name)
        ordered.append(target)
    return ordered


def _collider_table(colliders, armature_object):
    from . import collider_geom
    return [collider_geom.serialize(target, armature_object) for target in colliders
            if not collider_geom.is_mesh_collider(target)]


def _assignments(configs):
    return {config.name: [target.name for target in _referenced_colliders([config])]
            for config in configs}


def serialize(settings, scope, scene, config_index=0):
    armature_object = settings.id_data
    configs = [settings.configs[config_index]] if scope == SCOPE_CONFIG else list(settings.configs)
    colliders = collider_set(configs, scene, armature_object, scope != SCOPE_CONFIG)
    payload = {"format": FORMAT_VERSION, "scope": scope,
               "addon": "RuriClothPhysics",
               "colliders": _collider_table(colliders, armature_object)}
    if scope == SCOPE_COLLIDERS:
        payload["assignments"] = _assignments(configs)
        return payload
    if scope == SCOPE_CONFIG:
        payload["config"] = read_group(configs[0])
        return payload
    payload["settings"] = read_group(settings)
    return payload


def _bone_names(settings):
    found = set()
    for config in settings.configs:
        found.update(item.bone for item in config.root_bones if item.bone)
    return found


def remove_owned_colliders(settings, scene, armature_object):
    """Delete every collider this rig's configurations use, and every one hanging under
    the rig whether or not something still points at it. Returns how many objects went.

    Read the configurations BEFORE emptying them -- the references are the only record of
    which colliders were in play."""
    from . import collider_geom
    doomed = {}
    for obj in collider_set(list(settings.configs), scene, armature_object, True):
        end = collider_geom.settings_of(obj).end_object
        if end is not None:
            doomed[end.name] = end
        doomed[obj.name] = obj
    for obj in doomed.values():
        bpy.data.objects.remove(obj)
    return len(doomed)


def _create_colliders(payload, state):
    from . import collider_geom
    collection = collider_geom.collection_for(state.context.scene, state.armature_object,
                                              create=True)
    for data in payload.get("colliders", ()):
        empty, missing = collider_geom.deserialize(data, state.armature_object, collection,
                                                   state.context.view_layer)
        state.colliders[data.get("name", empty.name)] = empty
        if missing:
            state.report.setdefault("missing_bones", []).extend(missing)
    state.report["created_colliders"] = len(state.colliders)


def _apply_assignments(settings, payload, mode, state):
    from . import collider_binding
    by_name = {config.name: config for config in settings.configs}
    for config_name, keys in (payload.get("assignments") or {}).items():
        config = by_name.get(config_name)
        if config is None:
            state.report.setdefault("missing_configs", []).append(config_name)
            continue
        references = config.collider_collision.collider_references
        if mode == MODE_REPLACE:
            references.clear()
        present = {reference.collider_uid for reference in references}
        for key in keys:
            target = state.colliders.get(key)
            if target is None:
                state.report.setdefault("missing_colliders", []).append(key)
                continue
            uid = collider_binding.bind(target)
            if uid not in present:
                references.add().collider_uid = uid
                present.add(uid)


def deserialize(settings, payload, mode, context):
    state = LoadState(settings.id_data, context)
    if payload.get("addon") != "RuriClothPhysics":
        state.report["error"] = "不是本插件的配置文件"
        return state.report
    if payload.get("format") != FORMAT_VERSION:
        state.report["error"] = "配置文件版本 %s 与本插件的 %d 不一致" % (
            payload.get("format"), FORMAT_VERSION)
        return state.report

    scope = payload.get("scope")
    if mode == MODE_REPLACE:
        state.report["removed_colliders"] = remove_owned_colliders(
            settings, state.context.scene, state.armature_object)
    _create_colliders(payload, state)
    if scope == SCOPE_COLLIDERS:
        _apply_assignments(settings, payload, mode, state)
    elif scope == SCOPE_CONFIG:
        if mode == MODE_REPLACE:
            settings.configs.clear()
        write_group(settings.configs.add(), payload["config"], state)
        settings.active_config_index = len(settings.configs) - 1
    else:
        data = dict(payload["settings"])
        if mode == MODE_APPEND:
            for entry in data.pop("configs", []):
                write_group(settings.configs.add(), entry, state)
        else:
            write_group(settings, data, state)
        settings.active_config_index = max(0, len(settings.configs) - 1)

    for config in settings.configs:
        if any(item.bone == "" for item in config.root_bones):
            state.report["unresolved_bones"] = True
        config.rebuild_pending = True
    settings.param_serial += 1
    return state.report


PRESET_FOLDER = "RuriClothPhysics"


def preset_directory(create=False):
    path = bpy.utils.user_resource('SCRIPTS', path="presets/" + PRESET_FOLDER, create=create)
    return path or os.path.join(os.path.dirname(__file__), "presets")


def preset_files():
    directory = preset_directory()
    if not directory or not os.path.isdir(directory):
        return []
    return sorted(name for name in os.listdir(directory) if name.lower().endswith(".json"))


def save(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False, sort_keys=True)


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
