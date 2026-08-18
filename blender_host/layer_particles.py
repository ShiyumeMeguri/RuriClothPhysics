"""Particle layer: the live solver state -- positions, axes, constraint shape, inertia centre."""

import numpy as np

from . import runtime
from . import shapes
from . import viewport
from ..cloth_kernel import math as pm

COLOR_MOVE = (0.8, 0.8, 0.8, 0.8)
COLOR_FIXED = (1.0, 0.0, 0.0, 0.8)
COLOR_INVALID = (0.0, 0.0, 0.0, 0.8)
COLOR_LINE = (0.0, 1.0, 1.0, 1.0)
COLOR_TRIANGLE = (1.0, 0.0, 0.8, 1.0)
COLOR_ANIMATED_LINE = (0.6, 0.8, 1.0, 1.0)
COLOR_ANIMATED_TRIANGLE = (0.8, 0.6, 1.0, 1.0)
COLOR_INERTIA = (1.0, 0.0, 1.0, 1.0)
COLOR_AXIS_X = (1.0, 0.2, 0.2, 1.0)
COLOR_AXIS_Y = (0.2, 1.0, 0.2, 1.0)
COLOR_AXIS_Z = (0.3, 0.3, 1.0, 1.0)


def poll(context):
    scene = context.scene
    if scene is None:
        return False
    for _, index, _ in runtime.iter_entries(scene):
        return True
    return False


def _points(canvas, positions, attr_fixed, attr_move, depth, use_depth, depth_test):
    count = len(positions)
    colors = np.empty((count, 4), dtype=np.float32)
    if use_depth:
        colors[:, 0] = depth
        colors[:, 1] = 0.2
        colors[:, 2] = 1.0 - depth
        colors[:, 3] = 1.0
    else:
        colors[:] = COLOR_INVALID
        colors[attr_move] = COLOR_MOVE
        colors[attr_fixed] = COLOR_FIXED
    canvas.points(positions, colors, depth_test=depth_test)


def _axes(canvas, positions, rotations, depth_test, scale=0.02):
    broadcast = np.broadcast_to(pm.VEC_RIGHT, positions.shape).astype(np.float32)
    right = pm.quat_rotate(rotations, broadcast)
    normal = pm.quat_to_normal(rotations)
    tangent = pm.quat_to_tangent(rotations)
    canvas.lines(positions, positions + right * scale, COLOR_AXIS_X, depth_test)
    canvas.lines(positions, positions + normal * scale, COLOR_AXIS_Y, depth_test)
    canvas.lines(positions, positions + tangent * scale, COLOR_AXIS_Z, depth_test)


def _shape(canvas, setup, positions, line_color, triangle_color, depth_test):
    if len(setup.lines):
        canvas.lines(positions[setup.lines[:, 0]], positions[setup.lines[:, 1]],
                     line_color, depth_test)
    if len(setup.triangles):
        triangles = setup.triangles
        for first, second in ((0, 1), (1, 2), (2, 0)):
            canvas.lines(positions[triangles[:, first]], positions[triangles[:, second]],
                         triangle_color, depth_test)


def _baseline(canvas, setup, positions, depth_test):
    data = setup.baseline_data
    if len(data) == 0:
        return
    parents = setup.vertex_parent[data]
    keep = parents >= 0
    canvas.lines(positions[parents[keep]], positions[data[keep]], COLOR_ANIMATED_LINE, depth_test)


def collect(context, canvas):
    scene = context.scene
    selected = {obj.name for obj in context.selected_objects}
    active = context.view_layer.objects.active
    if active is not None:
        selected.add(active.name)

    world = runtime.get_world()
    particles = world.particles
    team = world.team

    for obj, index, entry in runtime.iter_entries(scene):
        settings = obj.ruri_cloth_physics
        if index >= len(settings.configs):
            continue
        gizmos = settings.configs[index].gizmos
        if not gizmos.enable:
            continue
        if not gizmos.always and obj.name not in selected:
            continue

        depth_test = gizmos.ztest
        setup = entry.setup
        span = world.particle_slice(entry.team)
        positions = particles["positions"][span]
        base_positions = particles["base_positions"][span]

        if gizmos.position or gizmos.depth:
            _points(canvas, positions, particles["attr_fixed"][span],
                    particles["attr_move"][span], particles["depth"][span], gizmos.depth,
                    depth_test)
        if gizmos.axis:
            _axes(canvas, positions, particles["rotations"][span], depth_test)
        if gizmos.shape:
            _shape(canvas, setup, positions, COLOR_LINE, COLOR_TRIANGLE, depth_test)
        if gizmos.base_line:
            _baseline(canvas, setup, positions, depth_test)
        if gizmos.animated_position:
            colors = np.empty((len(base_positions), 4), dtype=np.float32)
            colors[:] = COLOR_ANIMATED_LINE
            canvas.points(base_positions, colors, depth_test=depth_test)
        if gizmos.animated_axis:
            _axes(canvas, base_positions, particles["base_rotations"][span], depth_test)
        if gizmos.animated_shape:
            _shape(canvas, setup, base_positions, COLOR_ANIMATED_LINE,
                   COLOR_ANIMATED_TRIANGLE, depth_test)
        if gizmos.inertia_center:
            canvas.lines(*shapes.cross(team["now_world_position"][entry.team]),
                         color=COLOR_INERTIA, depth_test=depth_test)


LAYER = viewport.Layer("particles", poll=poll, collect=collect, order=30)


def register():
    viewport.register_layer(LAYER)


def unregister():
    viewport.unregister_layer(LAYER.identifier)
