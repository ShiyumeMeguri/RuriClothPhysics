"""Save and load the add-on state as JSON under scripts/presets/RuriClothPhysics.

Replaces the old .py preset system outright. That one wrote a hand-maintained list of parameter
paths and nothing else -- no root bones, no attribute overrides, no collider pool -- so restoring a
"preset" put numbers onto a config that simulated no bones at all. There is no migration path
because there was nothing to migrate: the preset folder was empty.
"""

import os

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import config_io
from . import runtime

SCOPE_ITEMS = (
    ('OBJECT', "整个骨架", "全部配置 + 碰撞体 + 骨架级开关"),
    ('CONFIG', "当前配置", "只存当前这一个配置(连同它引用的碰撞体)"),
)

MODE_ITEMS = (
    ('REPLACE', "替换", "清空现有内容后载入"),
    ('APPEND', "追加", "保留现有内容, 把文件里的配置接在后面"),
)


def _settings(context):
    obj = context.object
    if obj is None or obj.type != 'ARMATURE':
        return None
    return obj.ruri_cloth_physics


def _report_problems(operator, report):
    if report.get("error"):
        operator.report({'ERROR'}, report["error"])
        return False
    missing = report.get("missing_objects")
    if missing:
        operator.report({'WARNING'}, "找不到对象: %s" % ", ".join(sorted(set(missing))[:4]))
    if report.get("unresolved_bones"):
        operator.report({'WARNING'}, "有根骨骼在本骨架里不存在, 已留空")
    unknown = report.get("unknown_properties")
    if unknown:
        operator.report({'WARNING'}, "忽略了 %d 个本版本没有的字段" % len(set(unknown)))
    return True


class RCP_OT_preset_save(bpy.types.Operator, ExportHelper):
    bl_idname = "ruri_cloth_physics.preset_save"
    bl_label = "保存配置到 JSON"
    bl_description = "把布料设置写成 JSON 预设, 默认存到 scripts/presets/RuriClothPhysics"
    bl_options = {'REGISTER'}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    scope: EnumProperty(name="范围", items=SCOPE_ITEMS, default='OBJECT')

    @classmethod
    def poll(cls, context):
        settings = _settings(context)
        return settings is not None and len(settings.configs) > 0

    def invoke(self, context, event):
        settings = _settings(context)
        base = context.object.name if self.scope == 'OBJECT' \
            else settings.configs[settings.active_config_index].name
        self.filepath = os.path.join(config_io.preset_directory(create=True), "%s.json" % base)
        return super().invoke(context, event)

    def execute(self, context):
        settings = _settings(context)
        payload = config_io.serialize(settings, self.scope,
                                      min(settings.active_config_index,
                                          len(settings.configs) - 1))
        config_io.save(self.filepath, payload)
        self.report({'INFO'}, "已保存 %s" % os.path.basename(self.filepath))
        return {'FINISHED'}


class RCP_OT_preset_load(bpy.types.Operator, ImportHelper):
    bl_idname = "ruri_cloth_physics.preset_load"
    bl_label = "从 JSON 载入配置"
    bl_description = "载入 JSON 预设; 骨骼按名字重新绑定, 名字对不上的会留空并提示"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    mode: EnumProperty(name="方式", items=MODE_ITEMS, default='REPLACE')

    @classmethod
    def poll(cls, context):
        return _settings(context) is not None

    def invoke(self, context, event):
        self.filepath = os.path.join(config_io.preset_directory(create=True), "")
        return super().invoke(context, event)

    def execute(self, context):
        settings = _settings(context)
        try:
            payload = config_io.load(self.filepath)
        except (OSError, ValueError) as error:
            self.report({'ERROR'}, "读取失败: %s" % error)
            return {'CANCELLED'}
        report = config_io.deserialize(settings, payload, self.mode)
        if not _report_problems(self, report):
            return {'CANCELLED'}
        # The teams were built from the old topology; drop them so the next frame recompiles.
        runtime.clear_registry()
        self.report({'INFO'}, "已载入 %s (%d 配置 / %d 碰撞体)"
                    % (os.path.basename(self.filepath), len(settings.configs),
                       len(settings.colliders)))
        return {'FINISHED'}


def _preset_items(self, context):
    items = []
    for name in config_io.preset_files():
        items.append((name, os.path.splitext(name)[0], ""))
    if not items:
        items.append(('', "(没有预设)", ""))
    _preset_items.cache = items
    return items


class RCP_OT_preset_apply(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.preset_apply"
    bl_label = "应用预设"
    bl_description = "直接套用预设目录里的某个 JSON, 不用走文件浏览器"
    bl_options = {'REGISTER', 'UNDO'}

    preset: EnumProperty(name="预设", items=_preset_items)
    mode: EnumProperty(name="方式", items=MODE_ITEMS, default='REPLACE')

    @classmethod
    def poll(cls, context):
        return _settings(context) is not None

    def execute(self, context):
        if not self.preset:
            self.report({'WARNING'}, "预设目录里没有 JSON")
            return {'CANCELLED'}
        settings = _settings(context)
        path = os.path.join(config_io.preset_directory(), self.preset)
        try:
            payload = config_io.load(path)
        except (OSError, ValueError) as error:
            self.report({'ERROR'}, "读取失败: %s" % error)
            return {'CANCELLED'}
        report = config_io.deserialize(settings, payload, self.mode)
        if not _report_problems(self, report):
            return {'CANCELLED'}
        runtime.clear_registry()
        self.report({'INFO'}, "已应用 %s" % self.preset)
        return {'FINISHED'}


class RCP_OT_preset_reveal(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.preset_reveal"
    bl_label = "打开预设目录"
    bl_options = {'REGISTER'}

    def execute(self, context):
        directory = config_io.preset_directory(create=True)
        bpy.ops.wm.path_open(filepath=directory)
        self.report({'INFO'}, directory)
        return {'FINISHED'}


def draw_preset_row(layout, context):
    column = layout.column(align=True)
    row = column.row(align=True)
    row.operator("ruri_cloth_physics.preset_save", icon='FILE_TICK', text="保存 JSON")
    row.operator("ruri_cloth_physics.preset_load", icon='FILEBROWSER', text="载入 JSON")
    row.operator("ruri_cloth_physics.preset_reveal", icon='FILE_FOLDER', text="")
    if config_io.preset_files():
        column.operator_menu_enum("ruri_cloth_physics.preset_apply", "preset",
                                  text="应用预设", icon='PRESET')


_CLASSES = (
    RCP_OT_preset_save,
    RCP_OT_preset_load,
    RCP_OT_preset_apply,
    RCP_OT_preset_reveal,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
