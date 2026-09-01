import bpy
from bpy.app.handlers import persistent

from . import chain

_last_active = {}
_suppressed = False


class suppressed:

    def __enter__(self):
        global _suppressed
        self._previous = _suppressed
        _suppressed = True

    def __exit__(self, *args):
        global _suppressed
        _suppressed = self._previous


def is_suppressed():
    return _suppressed


def _active_bone_name(obj):
    if obj.mode == 'EDIT':
        active = obj.data.edit_bones.active
        return active.name if active else ""
    active = obj.data.bones.active
    return active.name if active else ""


def sync(obj):
    settings = getattr(obj, "ruri_cloth_physics", None)
    if settings is None or not settings.sync_list_selection or len(settings.configs) == 0:
        return False
    name = _active_bone_name(obj)
    if not name:
        return False

    mapping = chain.role_map(obj)
    owner = mapping.get(name)
    if owner is None:
        return False

    config_index, _ = owner
    config = settings.configs[config_index]
    _, root_name = chain.owning_root(obj, name)
    moved = False
    with suppressed():
        if settings.active_config_index != config_index:
            settings.active_config_index = config_index
            moved = True
        if root_name:
            for row, item in enumerate(config.root_bones):
                if item.bone == root_name and config.active_root_bone_index != row:
                    config.active_root_bone_index = row
                    moved = True
                    break
    return moved


@persistent
def _on_depsgraph_update_post(scene, depsgraph=None):
    if _suppressed:
        return
    obj = bpy.context.view_layer.objects.active if bpy.context.view_layer else None
    if obj is None or obj.type != 'ARMATURE':
        return
    name = _active_bone_name(obj)
    if _last_active.get(obj.name) == name:
        return
    _last_active[obj.name] = name
    sync(obj)


@persistent
def _on_load_post(*args):
    _last_active.clear()


def register():
    if _on_depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update_post)
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)


def unregister():
    if _on_depsgraph_update_post in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update_post)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    _last_active.clear()
