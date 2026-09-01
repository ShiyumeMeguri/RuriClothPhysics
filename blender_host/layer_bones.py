import numpy as np

from . import chain
from . import shapes
from . import viewport

COLOR_ROOT = (1.00, 0.78, 0.10, 1.0)
COLOR_CHILD = (0.10, 0.85, 0.95, 1.0)
COLOR_ROOT_IDLE = (0.55, 0.44, 0.10, 0.55)
COLOR_CHILD_IDLE = (0.10, 0.42, 0.48, 0.45)
COLOR_SELECTED = (1.0, 1.0, 1.0, 1.0)
COLOR_EXCLUDED = (0.85, 0.15, 0.15, 0.60)


def _armatures(context):
    scene = context.scene
    if scene is None:
        return
    for obj in scene.objects:
        if obj.type != 'ARMATURE' or getattr(obj, "ruri_cloth_physics", None) is None:
            continue
        if not obj.visible_get():
            continue
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
        active_index = chain.active_config_index(settings)
        scope = None if settings.display_scope == 'ALL' else {active_index}
        mapping = chain.role_map(obj, scope)
        if not mapping:
            continue

        pose = obj.pose.bones
        matrix_world = obj.matrix_world
        depth_test = settings.display_depth
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

        for index, config in enumerate(settings.configs):
            if scope is not None and index not in scope:
                continue
            for name in chain.excluded_names(obj, config):
                bone = pose.get(name)
                if bone is None:
                    continue
                head = np.array(matrix_world @ bone.head, dtype=np.float32)
                tail = np.array(matrix_world @ bone.tail, dtype=np.float32)
                canvas.lines(*shapes.octahedron(head, tail), color=COLOR_EXCLUDED,
                             depth_test=depth_test)


LAYER = viewport.Layer("bones", poll=poll, collect=collect, order=20)


def register():
    viewport.register_layer(LAYER)


def unregister():
    viewport.unregister_layer(LAYER.identifier)
