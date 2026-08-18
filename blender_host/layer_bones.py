"""Bone layer: show which bones a config drives, roots in one colour and driven children in another.

Reads the rest/pose hierarchy, not the solver, so it draws with the simulation switched off -- which
is when you are actually building the chain.
"""

import numpy as np

from . import chain
from . import shapes
from . import viewport

COLOR_ROOT = (1.00, 0.78, 0.10, 1.0)
COLOR_CHILD = (0.10, 0.85, 0.95, 1.0)
COLOR_ROOT_IDLE = (0.55, 0.44, 0.10, 0.55)
COLOR_CHILD_IDLE = (0.10, 0.42, 0.48, 0.45)
COLOR_SELECTED = (1.0, 1.0, 1.0, 1.0)


def _armatures(context):
    seen = set()
    candidates = list(context.selected_objects)
    active = context.view_layer.objects.active
    if active is not None:
        candidates.append(active)
    for obj in candidates:
        if obj is None or obj.type != 'ARMATURE' or obj.name in seen:
            continue
        seen.add(obj.name)
        if getattr(obj, "ruri_cloth_physics", None) is not None:
            yield obj


def _showing(obj):
    settings = obj.ruri_cloth_physics
    return settings.show_bones and len(settings.configs) > 0


def poll(context):
    return any(_showing(obj) for obj in _armatures(context))


def collect(context, canvas):
    for obj in _armatures(context):
        if not _showing(obj):
            continue
        settings = obj.ruri_cloth_physics
        active_index = min(settings.active_config_index, len(settings.configs) - 1)
        scope = None if settings.bone_display_scope == 'ALL' else {active_index}
        mapping = chain.role_map(obj, scope)
        if not mapping:
            continue

        pose = obj.pose.bones
        matrix_world = obj.matrix_world
        depth_test = settings.bone_display_depth
        for name, (config_index, role) in mapping.items():
            bone = pose.get(name)
            if bone is None:
                continue
            focused = config_index == active_index
            if role == chain.ROLE_ROOT:
                color = COLOR_ROOT if focused else COLOR_ROOT_IDLE
            else:
                color = COLOR_CHILD if focused else COLOR_CHILD_IDLE
            if chain.is_selected(obj, name):
                color = COLOR_SELECTED
            head = np.array(matrix_world @ bone.head, dtype=np.float32)
            tail = np.array(matrix_world @ bone.tail, dtype=np.float32)
            canvas.lines(*shapes.octahedron(head, tail), color=color, depth_test=depth_test)
            if role == chain.ROLE_ROOT:
                canvas.points(head[None, :], np.array([color], dtype=np.float32),
                              depth_test=depth_test)


LAYER = viewport.Layer("bones", poll=poll, collect=collect, order=20)


def register():
    viewport.register_layer(LAYER)


def unregister():
    viewport.unregister_layer(LAYER.identifier)
