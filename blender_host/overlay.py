import math

import numpy as np

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from . import collider_geom
from . import runtime
from ..cloth_kernel import defs
from ..cloth_kernel import math as pm

COLOR_MOVE = (0.8, 0.8, 0.8, 0.8)
COLOR_FIXED = (1.0, 0.0, 0.0, 0.8)
COLOR_INVALID = (0.0, 0.0, 0.0, 0.8)
COLOR_LINE = (0.0, 1.0, 1.0, 1.0)
COLOR_TRIANGLE = (1.0, 0.0, 0.8, 1.0)
COLOR_ANIMATED_LINE = (0.6, 0.8, 1.0, 1.0)
COLOR_ANIMATED_TRIANGLE = (0.8, 0.6, 1.0, 1.0)
COLOR_COLLIDER = (1.0, 0.27, 0.0, 1.0)
COLOR_COLLIDER_ACTIVE = (1.0, 0.85, 0.1, 1.0)
COLOR_COLLIDER_IDLE = (0.15, 0.75, 1.0, 0.75)
COLOR_COLLIDER_OFF = (0.45, 0.45, 0.45, 0.45)
COLOR_INERTIA = (1.0, 0.0, 1.0, 1.0)
COLOR_AXIS_X = (1.0, 0.2, 0.2, 1.0)
COLOR_AXIS_Y = (0.2, 1.0, 0.2, 1.0)
COLOR_AXIS_Z = (0.3, 0.3, 1.0, 1.0)

_draw_handle = None
_circle_cache = {}


def _unit_circle(segments=16):
    circle = _circle_cache.get(segments)
    if circle is None:
        angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
        circle = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        _circle_cache[segments] = circle
    return circle


class _Collector:
    def __init__(self):
        self.line_positions = []
        self.line_colors = []
        self.point_positions = []
        self.point_colors = []

    def add_segments(self, starts, ends, color):
        count = len(starts)
        if count == 0:
            return
        segments = np.empty((count * 2, 3), dtype=np.float32)
        segments[0::2] = starts
        segments[1::2] = ends
        self.line_positions.append(segments)
        colors = np.empty((count * 2, 4), dtype=np.float32)
        colors[:] = color
        self.line_colors.append(colors)

    def add_segments_colored(self, starts, ends, colors):
        count = len(starts)
        if count == 0:
            return
        segments = np.empty((count * 2, 3), dtype=np.float32)
        segments[0::2] = starts
        segments[1::2] = ends
        self.line_positions.append(segments)
        expanded = np.repeat(colors, 2, axis=0).astype(np.float32)
        self.line_colors.append(expanded)

    def add_points(self, points, colors):
        if len(points) == 0:
            return
        self.point_positions.append(points.astype(np.float32))
        self.point_colors.append(colors.astype(np.float32))


def _collect_points(collector, positions, attr_fixed, attr_move, depth, use_depth):
    n = len(positions)
    if use_depth:
        colors = np.empty((n, 4), dtype=np.float32)
        colors[:, 0] = depth
        colors[:, 1] = 0.2
        colors[:, 2] = 1.0 - depth
        colors[:, 3] = 1.0
    else:
        colors = np.empty((n, 4), dtype=np.float32)
        colors[:] = COLOR_INVALID
        colors[attr_move] = COLOR_MOVE
        colors[attr_fixed] = COLOR_FIXED
    collector.add_points(positions, colors)


def _collect_axes(collector, positions, rotations, scale=0.02):
    right = pm.quat_rotate(rotations, np.broadcast_to(pm.VEC_RIGHT, positions.shape).astype(np.float32))
    normal = pm.quat_to_normal(rotations)
    tangent = pm.quat_to_tangent(rotations)
    collector.add_segments(positions, positions + right * scale, COLOR_AXIS_X)
    collector.add_segments(positions, positions + normal * scale, COLOR_AXIS_Y)
    collector.add_segments(positions, positions + tangent * scale, COLOR_AXIS_Z)


def _collect_shape(collector, setup, positions, line_color, triangle_color):
    if len(setup.lines):
        collector.add_segments(positions[setup.lines[:, 0]], positions[setup.lines[:, 1]],
                               line_color)
    if len(setup.triangles):
        tri = setup.triangles
        collector.add_segments(positions[tri[:, 0]], positions[tri[:, 1]], triangle_color)
        collector.add_segments(positions[tri[:, 1]], positions[tri[:, 2]], triangle_color)
        collector.add_segments(positions[tri[:, 2]], positions[tri[:, 0]], triangle_color)


def _collect_baseline(collector, setup, positions):
    data = setup.baseline_data
    if len(data) == 0:
        return
    parents = setup.vertex_parent[data]
    keep = parents >= 0
    collector.add_segments(positions[parents[keep]], positions[data[keep]], COLOR_ANIMATED_LINE)


def _collect_circles(collector, centers, axis_a, axis_b, radii, color, segments=16):
    count = len(centers)
    if count == 0:
        return
    circle = _unit_circle(segments)
    points = centers[:, None, :] + axis_a[:, None, :] * (circle[None, :, 0:1] * radii[:, None, None]) \
        + axis_b[:, None, :] * (circle[None, :, 1:2] * radii[:, None, None])
    starts = points.reshape(-1, 3)
    ends = np.roll(points, -1, axis=1).reshape(-1, 3)
    collector.add_segments(starts, ends, color)


def _orthonormal_basis(axis):
    reference = np.array([0.0, 0.0, 1.0]) if abs(float(axis[2])) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    side_a = np.cross(axis, reference)
    length = float(np.linalg.norm(side_a))
    side_a = side_a / length if length > 1e-12 else np.array([1.0, 0.0, 0.0])
    return side_a, np.cross(axis, side_a)


_SPHERE_PLANES = (
    (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
)


def _collect_sphere_wire(collector, center, radius, color, segments=24):
    center32 = np.asarray(center, dtype=np.float32)[None]
    radii = np.array([radius], dtype=np.float32)
    for first, second in _SPHERE_PLANES:
        _collect_circles(collector, center32,
                         first.astype(np.float32)[None], second.astype(np.float32)[None],
                         radii, color, segments)


def collect_settings_colliders(collector, obj):
    settings = obj.ruri_cloth_physics
    if not settings.show_colliders or len(settings.colliders) == 0:
        return
    active_index = settings.active_collider_index
    for index, item in enumerate(settings.colliders):
        matrix = collider_geom.bone_matrix(obj, item.bone)
        kind, first, second, start_radius, end_radius = collider_geom.solve(item, matrix)
        if not item.enabled:
            color = COLOR_COLLIDER_OFF
        elif index == active_index:
            color = COLOR_COLLIDER_ACTIVE
        else:
            color = COLOR_COLLIDER_IDLE

        if kind == defs.COLLIDER_SPHERE:
            _collect_sphere_wire(collector, first, start_radius, color)
            continue
        if kind == defs.COLLIDER_PLANE:
            axis = second - first
            side_a, side_b = _orthonormal_basis(axis)
            size = 0.25
            corners = [first + side_a * size + side_b * size,
                       first - side_a * size + side_b * size,
                       first - side_a * size - side_b * size,
                       first + side_a * size - side_b * size]
            for k in range(4):
                collector.add_segments(np.asarray(corners[k], dtype=np.float32)[None],
                                       np.asarray(corners[(k + 1) % 4], dtype=np.float32)[None],
                                       color)
            collector.add_segments(np.asarray(first, dtype=np.float32)[None],
                                   np.asarray(first + axis * size * 0.5, dtype=np.float32)[None],
                                   color)
            continue

        _collect_sphere_wire(collector, first, start_radius, color)
        _collect_sphere_wire(collector, second, end_radius, color)
        axis = second - first
        length = float(np.linalg.norm(axis))
        axis = axis / length if length > 1e-12 else np.array([0.0, 1.0, 0.0])
        side_a, side_b = _orthonormal_basis(axis)
        for direction in (side_a, -side_a, side_b, -side_b):
            collector.add_segments(
                np.asarray(first + direction * start_radius, dtype=np.float32)[None],
                np.asarray(second + direction * end_radius, dtype=np.float32)[None], color)


def _collect_inertia_center(collector, center, size=0.02):
    axes = np.eye(3, dtype=np.float32) * size
    starts = center[None, :] - axes
    ends = center[None, :] + axes
    collector.add_segments(starts, ends, COLOR_INERTIA)


def _draw_callback():
    context = bpy.context
    scene = context.scene
    if scene is None or not scene.ruri_cloth_physics.overlay_enabled:
        return

    selected = {obj.name for obj in context.selected_objects}
    active = context.view_layer.objects.active
    if active is not None:
        selected.add(active.name)

    world = runtime.get_world()
    pa = world.particles
    tt = world.team
    no_depth = _Collector()
    with_depth = _Collector()

    seen = set()
    for obj in list(context.selected_objects) + ([active] if active is not None else []):
        if obj is None or obj.type != 'ARMATURE' or obj.name in seen:
            continue
        seen.add(obj.name)
        if getattr(obj, "ruri_cloth_physics", None) is not None:
            collect_settings_colliders(no_depth, obj)

    for obj, index, entry in runtime.iter_entries(scene):
        settings = obj.ruri_cloth_physics
        if index >= len(settings.configs):
            continue
        config = settings.configs[index]
        gizmos = config.gizmos
        if not gizmos.enable:
            continue
        if not gizmos.always and obj.name not in selected:
            continue

        collector = with_depth if gizmos.ztest else no_depth
        setup = entry.setup
        s = world.particle_slice(entry.team)
        positions = pa["positions"][s]
        rotations = pa["rotations"][s]
        base_positions = pa["base_positions"][s]
        base_rotations = pa["base_rotations"][s]
        attr_fixed = pa["attr_fixed"][s]
        attr_move = pa["attr_move"][s]
        depth = pa["depth"][s]

        if gizmos.position or gizmos.depth:
            _collect_points(collector, positions, attr_fixed, attr_move, depth, gizmos.depth)
        if gizmos.axis:
            _collect_axes(collector, positions, rotations)
        if gizmos.shape:
            _collect_shape(collector, setup, positions, COLOR_LINE, COLOR_TRIANGLE)
        if gizmos.base_line:
            _collect_baseline(collector, setup, positions)
        if gizmos.animated_position:
            colors = np.empty((len(base_positions), 4), dtype=np.float32)
            colors[:] = COLOR_ANIMATED_LINE
            collector.add_points(base_positions, colors)
        if gizmos.animated_axis:
            _collect_axes(collector, base_positions, base_rotations)
        if gizmos.animated_shape:
            _collect_shape(collector, setup, base_positions,
                           COLOR_ANIMATED_LINE, COLOR_ANIMATED_TRIANGLE)
        if gizmos.inertia_center:
            _collect_inertia_center(collector, tt["now_world_position"][entry.team])

    shader = gpu.shader.from_builtin('FLAT_COLOR')
    gpu.state.blend_set('ALPHA')

    for depth_test, collector in ((False, no_depth), (True, with_depth)):
        gpu.state.depth_test_set('LESS_EQUAL' if depth_test else 'NONE')
        if collector.line_positions:
            positions = np.concatenate(collector.line_positions)
            colors = np.concatenate(collector.line_colors)
            batch = batch_for_shader(shader, 'LINES', {"pos": positions, "color": colors})
            gpu.state.line_width_set(1.5)
            batch.draw(shader)
        if collector.point_positions:
            positions = np.concatenate(collector.point_positions)
            colors = np.concatenate(collector.point_colors)
            gpu.state.point_size_set(5.0)
            batch = batch_for_shader(shader, 'POINTS', {"pos": positions, "color": colors})
            batch.draw(shader)

    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)


def register():
    global _draw_handle
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_callback, (), 'WINDOW',
                                                              'POST_VIEW')


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
