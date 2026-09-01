from . import collider_geom
from . import shapes
from . import viewport
from ..cloth_kernel import defs
from ..cloth_kernel import host_math

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


EVALUATED_ONCE_REASON = (
    "which objects the dependency graph still holds is one fact about the frame, not one "
    "fact per collider, so it is read once here and handed to every solve: letting each "
    "collider ask for it walks the whole graph again per collider, which on this file was "
    "27 walks per redraw and about half the cost of the entire overlay")


CIRCLE_SEGMENTS = 24

BATCH_BY_COLOUR_REASON = (
    "every collider used to build its own rings and hand them over as its own draw call, "
    "which is a few hundred small numpy allocations per redraw for a body's worth of "
    "capsules; the rings are the same three circles and the same four rails whoever asks, "
    "so they are built for all of them at once and grouped by the only thing that actually "
    "differs between two colliders in one pass, which is the colour -- three groups instead "
    "of one per collider, and the same lines come out")


def collect(context, canvas):
    active = context.object
    depsgraph = context.evaluated_depsgraph_get()
    evaluated = collider_geom.evaluated_objects(depsgraph)
    balls = {}
    capsules = {}
    for obj, settings in _colliders(context):
        kind, position, rotation, tip, radii, live = collider_geom.solve(
            obj, settings, depsgraph, evaluated)
        if not live:
            color = COLOR_OFF
        elif obj is active:
            color = COLOR_ACTIVE
        else:
            color = COLOR_IDLE
        if kind == defs.COLLIDER_MESH:
            continue
        if kind == defs.COLLIDER_SPHERE:
            centers, radiuses = balls.setdefault(color, ([], []))
            centers.append(position)
            radiuses.append(radii[0])
        elif kind == defs.COLLIDER_PLANE:
            normal = host_math.quat_to_tangent(rotation[None])[0]
            canvas.lines(*shapes.plane(position, normal, radii[0]), color=color)
        else:
            heads, tails, head_radii, tail_radii = capsules.setdefault(color, ([], [], [], []))
            heads.append(position)
            tails.append(tip)
            head_radii.append(radii[0])
            tail_radii.append(radii[1])

    for color, (centers, radiuses) in balls.items():
        canvas.lines(*shapes.spheres(centers, radiuses, CIRCLE_SEGMENTS), color=color)
    for color, (heads, tails, head_radii, tail_radii) in capsules.items():
        canvas.lines(*shapes.spheres(heads + tails, head_radii + tail_radii, CIRCLE_SEGMENTS),
                     color=color)
        canvas.lines(*shapes.swept_rails(heads, tails, head_radii, tail_radii), color=color)


LAYER = viewport.Layer("colliders", poll=poll, collect=collect, order=10)


def register():
    viewport.register_layer(LAYER)


def unregister():
    viewport.unregister_layer(LAYER.identifier)
