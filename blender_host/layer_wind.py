import numpy as np

from . import shapes
from . import viewport
from . import wind_geom

COLOR_ACTIVE = (1.0, 0.62, 0.15, 1.0)
COLOR_IDLE = (0.35, 0.75, 1.0, 0.75)
COLOR_OFF = (0.40, 0.40, 0.40, 0.30)

GLOBAL_MODE = 'GLOBAL_DIRECTION'

ARROW_HEAD_RATIO = 0.08
ARROW_HEAD_TAPER = 2.5
ARROW_HEAD_BARBS = 4
ARROW_INSET = 0.9
RADIAL_AXES = np.concatenate([np.eye(3, dtype=np.float64), -np.eye(3, dtype=np.float64)])

ZONE_SHAPE_REASON = (
    "the empty draws the region and this layer draws the direction inside it, so each mode "
    "shows the one thing the other cannot: a box zone is the empty's cube plus the arrow "
    "that blows through it, a sphere zone is the empty's sphere plus that same arrow, and a "
    "radial zone is that sphere plus the six axis arrows leaving the centre, because it has "
    "a burst instead of a direction; a global wind is left entirely to the empty's own "
    "single arrow, see wind_geom.GLOBAL_ARROW_REASON")

ARROW_SHAPE_REASON = (
    "an arrow is a shaft and a head on its point, not a solid body with fins along its "
    "whole length; the body form drew sixteen segments per arrow and the radial zone draws "
    "six of them from one centre, so ninety six overlapping barbs filled the sphere and the "
    "one thing the picture had to say, which way each arrow points, was the thing it buried")

ARROW_REACH_REASON = (
    "the arrow is built in the zone's own local space and then carried out by the zone "
    "matrix, so the length that fits is measured against the same box the solver tests "
    "against and the arrow cannot leave the region it belongs to; measuring it in world "
    "space would need the region in world space, which a non uniform object scale turns "
    "from a box into a parallelepiped, and the arrow would then poke out of exactly the "
    "zones whose scale made them interesting; the head stands out sideways from the shaft, "
    "so the region is shrunk by the head radius before the distance to its boundary is "
    "measured, which is an exact containment and not a margin that happens to work: a box "
    "gives the smallest shrunk half extent divided by that axis of the direction and a "
    "sphere gives the leg of the right triangle the head radius and the radius make; "
    "bounding the head alone was not enough because a diagonal arrow reaches the corner "
    "where its head then sticks out through the side it is not pointing at")


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


def _to_world(matrix, points):
    return (np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T
            + matrix[:3, 3]).astype(np.float32)


def _arrow(direction, near, far, head_radius):
    length = far - near
    if length <= 1e-9:
        return []
    axis, side_a, side_b = shapes.orthonormal_basis(direction)
    tip = axis * far
    head_length = min(head_radius * ARROW_HEAD_TAPER, length)
    base = tip - axis * head_length
    rim = []
    for barb in range(ARROW_HEAD_BARBS):
        angle = barb * (2.0 * np.pi / ARROW_HEAD_BARBS)
        offset = side_a * np.cos(angle) + side_b * np.sin(angle)
        rim.append(base + offset * head_radius)
    pairs = [(axis * near, tip)]
    for index, corner in enumerate(rim):
        pairs.append((tip, corner))
        pairs.append((corner, rim[(index + 1) % ARROW_HEAD_BARBS]))
    return pairs


def _extent(size, wind, direction):
    span = wind_geom.local_reach(size, wind, direction)
    head_radius = span * ARROW_HEAD_RATIO
    reach = wind_geom.local_reach(size, wind, direction, head_radius) * ARROW_INSET
    return reach, head_radius


def _arrow_pairs(size, wind):
    if wind.mode == GLOBAL_MODE:
        return []
    if wind.mode == wind_geom.RADIAL_MODE:
        pairs = []
        for axis in RADIAL_AXES:
            reach, head_radius = _extent(size, wind, axis)
            pairs.extend(_arrow(axis, 0.0, reach, head_radius))
        return pairs
    direction = wind_geom.local_direction(wind)
    reach, head_radius = _extent(size, wind, direction)
    return _arrow(direction, -reach, reach, head_radius)


def _local_geometry(size, wind):
    pairs = _arrow_pairs(size, wind)
    if not pairs:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return shapes.segments(pairs)


def collect(context, canvas):
    active = context.object
    depsgraph = context.evaluated_depsgraph_get()
    for obj, wind in _zones(context):
        starts, ends = _local_geometry(wind_geom.zone_display_size(obj, depsgraph), wind)
        if starts.shape[0] == 0:
            continue
        if not wind.enabled:
            color = COLOR_OFF
        elif obj is active:
            color = COLOR_ACTIVE
        else:
            color = COLOR_IDLE
        matrix = wind_geom.zone_matrix(obj, depsgraph)
        canvas.lines(_to_world(matrix, starts), _to_world(matrix, ends), color=color)


LAYER = viewport.Layer("wind_zones", poll=poll, collect=collect, order=20)


def register():
    viewport.register_layer(LAYER)


def unregister():
    viewport.unregister_layer(LAYER.identifier)
