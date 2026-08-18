"""Viewport kernel: the record types and the layer registry.

The kernel knows nothing about colliders, bones or particles. A layer is one registered data row
that answers three questions -- should I draw, what lines/points, what draggable handles -- and both
the GPU pass and the gizmo group iterate the registry. Adding a visualisation is a new layer file
plus one register_layer call; nothing here changes.

That is the whole point of the split: the previous design hard-coded colliders and particles inside
the draw callback and had a second, entirely separate hand-written gizmo group that knew only about
colliders, so a third thing to show meant editing both and a draggable bone was not expressible.
"""

import numpy as np

ARROW = 'ARROW'
MOVE = 'MOVE'
DIAL = 'DIAL'
PICK = 'PICK'

_layers = []


def rgba(color):
    """Accept RGB or RGBA and always return RGBA.

    Gizmo colours are RGB while the shader wants RGBA, and a bare `colors[:] = color` turns that
    mismatch into a ValueError inside the draw handler -- which does not just drop one shape, it
    aborts the whole pass and blanks every overlay for that frame. Normalising here means no caller
    can reintroduce it.
    """
    values = tuple(color)
    return values if len(values) == 4 else (values[0], values[1], values[2], 1.0)


class Canvas:
    """Accumulates wireframe for one frame, split by whether it is depth tested."""

    def __init__(self):
        self._buckets = {False: ([], [], [], []), True: ([], [], [], [])}

    def lines(self, starts, ends, color, depth_test=False):
        count = len(starts)
        if count == 0:
            return
        line_positions, line_colors, _, _ = self._buckets[bool(depth_test)]
        interleaved = np.empty((count * 2, 3), dtype=np.float32)
        interleaved[0::2] = starts
        interleaved[1::2] = ends
        line_positions.append(interleaved)
        colors = np.empty((count * 2, 4), dtype=np.float32)
        colors[:] = rgba(color)
        line_colors.append(colors)

    def lines_colored(self, starts, ends, colors, depth_test=False):
        count = len(starts)
        if count == 0:
            return
        line_positions, line_colors, _, _ = self._buckets[bool(depth_test)]
        interleaved = np.empty((count * 2, 3), dtype=np.float32)
        interleaved[0::2] = starts
        interleaved[1::2] = ends
        line_positions.append(interleaved)
        line_colors.append(np.repeat(np.asarray(colors, dtype=np.float32), 2, axis=0))

    def points(self, positions, colors, depth_test=False):
        if len(positions) == 0:
            return
        _, _, point_positions, point_colors = self._buckets[bool(depth_test)]
        point_positions.append(np.asarray(positions, dtype=np.float32))
        point_colors.append(np.asarray(colors, dtype=np.float32))

    def batches(self):
        for depth_test, bucket in self._buckets.items():
            line_positions, line_colors, point_positions, point_colors = bucket
            lines = None
            points = None
            if line_positions:
                lines = (np.concatenate(line_positions), np.concatenate(line_colors))
            if point_positions:
                points = (np.concatenate(point_positions), np.concatenate(point_colors))
            if lines is not None or points is not None:
                yield depth_test, lines, points


class Handle:
    """One viewport control, expressed as data.

    `matrix` is a 4x4 whose +Z is the drag axis for ARROW. A value handle carries `read`/`write`
    callables over the underlying property; a PICK carries an operator to run instead, so the kernel
    never learns what is being edited or selected.
    """

    __slots__ = ("identifier", "kind", "matrix", "scale", "color", "read", "write", "minimum",
                 "operator", "properties")

    def __init__(self, identifier, kind, matrix, read=None, write=None, scale=0.05,
                 color=(1.0, 1.0, 1.0), minimum=None, operator=None, properties=None):
        self.identifier = identifier
        self.kind = kind
        self.matrix = matrix
        self.scale = scale
        self.color = color
        self.read = read
        self.write = write
        self.minimum = minimum
        self.operator = operator
        self.properties = properties or {}


class Layer:
    __slots__ = ("identifier", "order", "poll", "collect", "handles")

    def __init__(self, identifier, poll, collect, handles=None, order=0):
        self.identifier = identifier
        self.order = order
        self.poll = poll
        self.collect = collect
        self.handles = handles


def axis_frame(origin, direction):
    """4x4 at `origin` whose +Z is `direction`. ARROW handles slide along their own +Z."""
    import mathutils
    forward = mathutils.Vector(direction).normalized()
    reference = mathutils.Vector((0.0, 0.0, 1.0))
    if abs(forward.dot(reference)) > 0.99:
        reference = mathutils.Vector((1.0, 0.0, 0.0))
    side = reference.cross(forward).normalized()
    up = forward.cross(side)
    frame = mathutils.Matrix(((side.x, up.x, forward.x),
                              (side.y, up.y, forward.y),
                              (side.z, up.z, forward.z))).to_4x4()
    frame.translation = mathutils.Vector(origin)
    return frame


def tag_redraw():
    """Ask every 3D view to redraw.

    Editing an add-on PropertyGroup tags nothing on its own, so the viewport keeps showing the old
    wireframe until something unrelated forces a redraw -- which is why re-binding a collider's bone
    only appeared to take effect after toggling the collider display off and on again.
    """
    import bpy
    manager = bpy.context.window_manager
    if manager is None:
        return
    for window in manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def register_layer(layer):
    unregister_layer(layer.identifier)
    _layers.append(layer)
    _layers.sort(key=lambda entry: entry.order)


def unregister_layer(identifier):
    for index, entry in enumerate(_layers):
        if entry.identifier == identifier:
            del _layers[index]
            return


def active_layers(context):
    for layer in _layers:
        try:
            if layer.poll(context):
                yield layer
        except (AttributeError, KeyError, IndexError):
            continue


def collect(context):
    canvas = Canvas()
    for layer in active_layers(context):
        layer.collect(context, canvas)
    return canvas


def collect_handles(context):
    found = []
    for layer in active_layers(context):
        if layer.handles is None:
            continue
        found.extend(layer.handles(context))
    return found
