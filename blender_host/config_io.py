import json
import os

import bpy

FORMAT_VERSION = 1

SCOPE_OBJECT = 'OBJECT'
SCOPE_CONFIG = 'CONFIG'

MODE_REPLACE = 'REPLACE'
MODE_APPEND = 'APPEND'

SKIP_EXACT = frozenset({
    "rna_type", "node_name", "param_serial", "collider_serial", "rebuild_pending",
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


def _write_value(owner, prop, value, report):
    identifier = prop.identifier
    if prop.type == 'POINTER':
        if _is_id_pointer(prop):
            target = bpy.data.objects.get(value) if value else None
            if value and target is None:
                report.setdefault("missing_objects", []).append(value)
            setattr(owner, identifier, target)
            return
        if isinstance(value, dict):
            write_group(getattr(owner, identifier), value, report)
        return
    if prop.type == 'COLLECTION':
        collection = getattr(owner, identifier)
        collection.clear()
        for entry in value or ():
            write_group(collection.add(), entry, report)
        return
    if getattr(prop, "is_array", False):
        setattr(owner, identifier, tuple(value))
        return
    setattr(owner, identifier, value)


def write_group(group, data, report):
    properties = {prop.identifier: prop for prop in group.bl_rna.properties}
    ordered = [key for key in data if key != "points_serialized"]
    if "points_serialized" in data:
        ordered.append("points_serialized")
    for key in ordered:
        prop = properties.get(key)
        if prop is None or _skip(key):
            report.setdefault("unknown_properties", []).append(key)
            continue
        if prop.is_readonly and not (prop.type == 'COLLECTION'
                                     or (prop.type == 'POINTER' and not _is_id_pointer(prop))):
            continue
        try:
            _write_value(group, prop, data[key], report)
        except (TypeError, ValueError) as error:
            report.setdefault("rejected", []).append("%s: %s" % (key, error))


def serialize(settings, scope, config_index=0):
    payload = {"format": FORMAT_VERSION, "scope": scope,
               "addon": "RuriClothPhysics"}
    if scope == SCOPE_CONFIG:
        payload["config"] = read_group(settings.configs[config_index])
        payload["colliders"] = [read_group(item) for item in settings.colliders]
        return payload
    payload["settings"] = read_group(settings)
    return payload


def _bone_names(settings):
    found = set()
    for config in settings.configs:
        found.update(item.bone for item in config.root_bones if item.bone)
    return found


def deserialize(settings, payload, mode):
    report = {}
    if payload.get("addon") != "RuriClothPhysics":
        report["error"] = "不是本插件的配置文件"
        return report
    if payload.get("format") != FORMAT_VERSION:
        report["error"] = "配置文件版本 %s 与本插件的 %d 不一致" % (
            payload.get("format"), FORMAT_VERSION)
        return report

    scope = payload.get("scope")
    if scope == SCOPE_CONFIG:
        if mode == MODE_REPLACE:
            settings.configs.clear()
        write_group(settings.configs.add(), payload["config"], report)
        existing = {item.name for item in settings.colliders}
        for entry in payload.get("colliders", ()):
            if entry.get("name") in existing:
                continue
            write_group(settings.colliders.add(), entry, report)
        settings.active_config_index = len(settings.configs) - 1
    else:
        data = dict(payload["settings"])
        if mode == MODE_APPEND:
            configs = data.pop("configs", [])
            colliders = data.pop("colliders", [])
            for entry in configs:
                write_group(settings.configs.add(), entry, report)
            existing = {item.name for item in settings.colliders}
            for entry in colliders:
                if entry.get("name") in existing:
                    continue
                write_group(settings.colliders.add(), entry, report)
        else:
            write_group(settings, data, report)
        settings.active_config_index = max(0, len(settings.configs) - 1)

    wanted = set()
    for config in settings.configs:
        for item in config.root_bones:
            wanted.add(item.bone)
    if "" in wanted:
        report["unresolved_bones"] = True
    for config in settings.configs:
        config.rebuild_pending = True
    settings.param_serial += 1
    settings.collider_serial += 1
    return report


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
