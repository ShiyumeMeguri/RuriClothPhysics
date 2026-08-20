from . import collider_geom
from . import shapes
from . import viewport
from ..cloth_kernel import defs
from ..cloth_kernel import math as pm

COLOR_ACTIVE = (1.0, 0.42, 0.05, 1.0)
COLOR_IDLE = (0.10, 0.45, 0.70, 0.55)
COLOR_OFF = (0.40, 0.40, 0.40, 0.30)


def _colliders(context):
    scene = context.scene
    if scene is None:
        return
    for obj in scene.objects:
        settings = collider_geom.settings_of(obj)
        if settings is None or not settings.is_collider or not obj.visible_get():
            continue
        yield obj, settings


def poll(context):
    for _ in _colliders(context):
        return True
    return False


def collect(context, canvas):
    active = context.object
    depsgraph = context.evaluated_depsgraph_get()
    for obj, settings in _colliders(context):
        kind, position, rotation, tip, radii, live = collider_geom.solve(obj, settings, depsgraph)
        if not live:
            color = COLOR_OFF
        elif obj is active:
            color = COLOR_ACTIVE
        else:
            color = COLOR_IDLE
        if kind == defs.COLLIDER_SPHERE:
            canvas.lines(*shapes.sphere(position, radii[0]), color=color)
        elif kind == defs.COLLIDER_PLANE:
            normal = pm.quat_to_tangent(rotation[None])[0]
            canvas.lines(*shapes.plane(position, normal, radii[0]), color=color)
        else:
            canvas.lines(*shapes.capsule(position, tip, radii[0], radii[1]), color=color)


LAYER = viewport.Layer("colliders", poll=poll, collect=collect, order=10)


def register():
    viewport.register_layer(LAYER)


def unregister():
    viewport.unregister_layer(LAYER.identifier)
