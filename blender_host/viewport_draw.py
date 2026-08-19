import bpy
import gpu
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader

from . import shapes
from . import viewport

GHOST_FACTOR = 0.02
GHOST_MINIMUM = 0.005
GHOST_MAXIMUM = 0.02

_handle = None


def _ghosts(context, canvas):
    for handle in viewport.collect_handles(context):
        origin = handle.matrix.translation
        radius = min(max(handle.scale * GHOST_FACTOR, GHOST_MINIMUM), GHOST_MAXIMUM)
        canvas.lines(*shapes.sphere(origin, radius, segments=12), color=handle.color)


def _draw():
    context = bpy.context
    scene = context.scene
    if scene is None or not scene.ruri_cloth_physics.overlay_enabled:
        return

    canvas = viewport.collect(context)
    screen = context.screen
    if screen is not None and screen.is_animation_playing:
        _ghosts(context, canvas)
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


@persistent
def _on_depsgraph_update(scene, depsgraph=None):
    if scene is None or depsgraph is None:
        return
    settings = getattr(scene, "ruri_cloth_physics", None)
    if settings is None or not settings.overlay_enabled:
        return
    screen = bpy.context.screen
    if screen is not None and screen.is_animation_playing:
        return
    for update in depsgraph.updates:
        if update.is_updated_transform or update.is_updated_geometry:
            viewport.tag_redraw()
            return


def register():
    global _handle
    if _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')
    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)


def unregister():
    global _handle
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
