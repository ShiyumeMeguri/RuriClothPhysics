bl_info = {
    "name": "Ruri Cloth Physics",
    "author": "ShiyumeMeguri",
    "version": (3, 0, 0),
    "blender": (4, 2, 0),
    "location": "Properties > Physics > Ruri 布料物理 (Armature) / Ruri 布料物理风区 (Empty)",
    "description": "骨架骨骼的布料与弹簧物理: 粒子图微内核全场景单竞技场模拟、碰撞体、风场、自碰撞与关键帧烘焙",
    "category": "Physics",
}

from . import dependencies

dependencies.ensure_installed()

from .blender_host import (
    properties,
    operators,
    bake,
    presets,
    ui,
    runtime,
    overlay,
    collider_gizmo,
)

_MODULES = (
    properties,
    operators,
    bake,
    presets,
    ui,
    runtime,
    overlay,
    collider_gizmo,
)


def register():
    for module in _MODULES:
        module.register()


def unregister():
    for module in reversed(_MODULES):
        module.unregister()
