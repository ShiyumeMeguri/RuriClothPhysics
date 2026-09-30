import bpy
from bpy.app.handlers import persistent

from . import handler_registry

SCENE_INDEX_REASON = (
    "which objects take part, the armatures, the colliders and the wind zones, is asked by "
    "every redraw, every gizmo poll and every frame, and the answer only moves when the file "
    "is edited; walking scene.objects to ask it read the add-on's property groups off every "
    "object in the scene four times per redraw and twice per frame, which on a shot linking a "
    "level of seven thousand nine hundred objects measured thirty nine milliseconds per "
    "redraw and four per frame with no physics switched on anywhere. The index is built by "
    "the first question after an edit and read by every other one, so what a redraw costs "
    "follows the number of objects taking part and not the size of the file; it is dropped "
    "by the things that can change membership and nothing else: an object entering or "
    "leaving the scene tags the collection holding it, a collider or a wind zone is marked "
    "through its own switch, and loading or undoing hands every object back under a new "
    "reference, while playback, orbiting and dragging an object touch none of those")

ARMATURE_TYPE = 'ARMATURE'
WIND_ZONE_TYPE = 'EMPTY'

_indices = {}


class SceneIndex:
    __slots__ = ("armatures", "colliders", "wind_zones")

    def __init__(self, scene):
        from . import collider_geom
        armatures = []
        colliders = []
        wind_zones = []
        for obj in scene.objects:
            kind = obj.type
            if kind == ARMATURE_TYPE:
                armatures.append(obj)
                continue
            if kind not in collider_geom.COLLIDER_OBJECT_TYPES:
                continue
            if obj.ruri_cloth_physics_collider.is_collider:
                colliders.append(obj)
            if kind == WIND_ZONE_TYPE and obj.ruri_cloth_physics_wind.is_wind_zone:
                wind_zones.append(obj)
        self.armatures = tuple(armatures)
        self.colliders = tuple(colliders)
        self.wind_zones = tuple(wind_zones)


def of(scene):
    key = scene.session_uid
    index = _indices.get(key)
    if index is None:
        index = SceneIndex(scene)
        _indices[key] = index
    return index


def armatures(scene):
    return of(scene).armatures


def colliders(scene):
    return of(scene).colliders


def wind_zones(scene):
    return of(scene).wind_zones


def invalidate():
    _indices.clear()


@persistent
def _on_depsgraph_update_post(scene, depsgraph=None):
    if depsgraph is None or depsgraph.id_type_updated('COLLECTION') \
            or depsgraph.id_type_updated('SCENE'):
        invalidate()


@persistent
def _on_reload(*args):
    invalidate()


_HANDLERS = (
    ("depsgraph_update_post", _on_depsgraph_update_post),
    ("load_post", _on_reload),
    ("undo_post", _on_reload),
    ("redo_post", _on_reload),
)


def register():
    for slot, function in _HANDLERS:
        handler_registry.install(getattr(bpy.app.handlers, slot), function)
    invalidate()


def unregister():
    for slot, function in _HANDLERS:
        handler_registry.uninstall(getattr(bpy.app.handlers, slot), function)
    invalidate()
