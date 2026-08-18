"""The only GPU code in the add-on: batch whatever the registered layers drew."""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from . import viewport

_handle = None


def _draw():
    context = bpy.context
    scene = context.scene
    if scene is None or not scene.ruri_cloth_physics.overlay_enabled:
        return

    canvas = viewport.collect(context)
    shader = gpu.shader.from_builtin('FLAT_COLOR')
    gpu.state.blend_set('ALPHA')
    for depth_test, lines, points in canvas.batches():
        gpu.state.depth_test_set('LESS_EQUAL' if depth_test else 'NONE')
        if lines is not None:
            positions, colors = lines
            gpu.state.line_width_set(1.5)
            batch_for_shader(shader, 'LINES', {"pos": positions, "color": colors}).draw(shader)
        if points is not None:
            positions, colors = points
            gpu.state.point_size_set(5.0)
            batch_for_shader(shader, 'POINTS', {"pos": positions, "color": colors}).draw(shader)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)


def register():
    global _handle
    if _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
