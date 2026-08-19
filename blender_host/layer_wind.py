import numpy as np

from . import shapes
from . import viewport
from . import wind_geom
from ..cloth_kernel import math as pm

COLOR_ACTIVE = (1.0, 0.62, 0.15, 1.0)
COLOR_IDLE = (0.35, 0.75, 1.0, 0.75)
COLOR_OFF = (0.40, 0.40, 0.40, 0.30)

GIZMO_SIZE = 0.5
ARROW_PROFILE = np.array([(0.0, 0.0, -1.0), (0.0, 0.5, -1.0), (0.0, 0.5, 0.0),
                          (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], dtype=np.float64)
ARROW_SCALE = np.array([GIZMO_SIZE, GIZMO_SIZE, GIZMO_SIZE * 2.0], dtype=np.float64)
ARROW_FINS = 4


def _zones(context):
    scene = context.scene
    if scene is None:
        return
    for obj in scene.objects:
        wind = getattr(obj, "ruri_cloth_physics_wind", None)
        if wind is None or not wind.is_wind_zone or obj.hide_viewport:
            continue
        yield obj, wind


def poll(context):
    for _ in _zones(context):
        return True
    return False


def _direction_arrow(canvas, origin, direction, color):
    heading = pm.axis_quaternion(np.asarray(direction, dtype=np.float32)[None])[0]
    profile = (ARROW_PROFILE * ARROW_SCALE).astype(np.float32)
    pairs = []
    for fin in range(ARROW_FINS):
        spin = pm.quat_from_axis_angle(pm.VEC_FORWARD,
                                       np.float32(fin * (2.0 * np.pi / ARROW_FINS)))
        points = pm.quat_rotate(pm.quat_mul(heading, spin), profile) + origin
        pairs.extend(zip(points[:-1], points[1:]))
    canvas.lines(*shapes.segments(pairs), color=color)


def _radial_cross(canvas, matrix, color):
    origin = matrix[:3, 3]
    offsets = (np.eye(3, dtype=np.float64) * GIZMO_SIZE) @ matrix[:3, :3].T
    pairs = [((origin - offset).astype(np.float32), (origin + offset).astype(np.float32))
             for offset in offsets]
    canvas.lines(*shapes.segments(pairs), color=color)


def collect(context, canvas):
    active = context.object
    depsgraph = context.evaluated_depsgraph_get()
    for obj, wind in _zones(context):
        matrix = wind_geom.zone_matrix(obj, depsgraph)
        if not wind.enabled:
            color = COLOR_OFF
        elif obj is active:
            color = COLOR_ACTIVE
        else:
            color = COLOR_IDLE
        if wind.mode == wind_geom.RADIAL_MODE:
            _radial_cross(canvas, matrix, color)
        else:
            _direction_arrow(canvas, matrix[:3, 3].astype(np.float64),
                             wind_geom.world_direction(matrix, wind), color)


LAYER = viewport.Layer("wind_zones", poll=poll, collect=collect, order=20)


def register():
    viewport.register_layer(LAYER)


def unregister():
    viewport.unregister_layer(LAYER.identifier)
