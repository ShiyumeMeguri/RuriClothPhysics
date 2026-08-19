import numpy as np

from . import armature
from . import curve_host
from . import shapes
from . import viewport
from . import wind_geom
from ..cloth_kernel import defs
from ..cloth_kernel import math as pm

COLOR_ACTIVE = (1.0, 0.62, 0.15, 1.0)
COLOR_IDLE = (0.35, 0.75, 1.0, 0.75)
COLOR_OFF = (0.40, 0.40, 0.40, 0.30)
COLOR_TURBULENCE = (0.35, 0.75, 1.0, 0.35)

HEAD_RATIO = 0.24
HEAD_WIDTH = 0.09
SPREAD_RATIO = 0.72
SPREAD_OFFSETS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                  (0.7, 0.7), (0.7, -0.7), (-0.7, 0.7), (-0.7, -0.7))

ATTENUATION_TICKS = (0.25, 0.5, 0.75)
TICK_WIDTH = 0.10


def _radial_directions():
    axes = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
    diagonal = 1.0 / np.sqrt(3.0)
    for sx in (-diagonal, diagonal):
        for sy in (-diagonal, diagonal):
            for sz in (-diagonal, diagonal):
                axes.append((sx, sy, sz))
    return np.asarray(axes, dtype=np.float64)


RADIAL_DIRECTIONS = _radial_directions()


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


def _arrow(canvas, origin, direction, length, color, head=True):
    tip = origin + direction * length
    pairs = [(origin, tip)]
    if head:
        axis, side_a, side_b = shapes.orthonormal_basis(direction)
        base = tip - direction * (length * HEAD_RATIO)
        width = length * HEAD_WIDTH
        for offset in (side_a, -side_a, side_b, -side_b):
            pairs.append((tip, base + offset * width))
    canvas.lines(*shapes.segments(pairs), color=color)


def _radial(canvas, display_size, matrix, origin, wind, color):
    lut = curve_host.get_lut(wind.attenuation)
    reach = float(wind_geom.local_extent(display_size, wind)[0])
    for unit in RADIAL_DIRECTIONS:
        boundary = matrix[:3, :3] @ (unit * reach) + origin
        offset = boundary - origin
        span = float(np.linalg.norm(offset))
        if span < 1e-9:
            continue
        direction = offset / span
        _arrow(canvas, origin, direction, span, color)
        _axis, side_a, side_b = shapes.orthonormal_basis(direction)
        pairs = []
        for depth in ATTENUATION_TICKS:
            weight = float(pm.evaluate_curve_lut_clamp01(lut, np.float32(depth)))
            if weight <= 1e-3:
                continue
            point = origin + direction * (span * depth)
            width = span * TICK_WIDTH * weight
            pairs.append((point - side_a * width, point + side_a * width))
            pairs.append((point - side_b * width, point + side_b * width))
        if pairs:
            canvas.lines(*shapes.segments(pairs), color=COLOR_TURBULENCE)


def collect(context, canvas):
    active = context.object
    depsgraph = context.evaluated_depsgraph_get()
    for obj, wind in _zones(context):
        matrix = wind_geom.zone_matrix(obj, depsgraph)
        display_size = wind_geom.zone_display_size(obj, depsgraph)
        origin = matrix[:3, 3].astype(np.float64)
        scale = armature.matrix_scale(matrix)
        span = display_size * float(scale[0])
        if span < 1e-5:
            span = 1.0
        if not wind.enabled:
            color = COLOR_OFF
        elif obj is active:
            color = COLOR_ACTIVE
        else:
            color = COLOR_IDLE

        speed = min(float(wind.main) / defs.WIND_BASE_SPEED, 2.0)
        length = span * (0.35 + 0.65 * speed)

        if wind.mode == wind_geom.RADIAL_MODE:
            _radial(canvas, display_size, matrix, origin, wind, color)
            continue

        base, spread = wind_geom.turbulence_spread(matrix, wind, SPREAD_OFFSETS)
        if float(wind.turbulence) > 1e-4:
            for direction in spread:
                _arrow(canvas, origin, direction, length * SPREAD_RATIO,
                       COLOR_TURBULENCE, head=False)
        _arrow(canvas, origin, base, length, color)


LAYER = viewport.Layer("wind_zones", poll=poll, collect=collect, order=20)


def register():
    viewport.register_layer(LAYER)


def unregister():
    viewport.unregister_layer(LAYER.identifier)
