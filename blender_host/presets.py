import os

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import config_io
from . import runtime

SCOPE_ITEMS = (
    ('OBJECT', "整个骨架", "全部配置 + 骨架级开关 + 它们用到的碰撞体定义"),
    ('CONFIG', "当前配置", "只存当前这一个配置, 连同它用到的碰撞体定义"),
    ('COLLIDERS', "只存碰撞体", "只存碰撞体本身与各配置的引用关系, 不动任何物理参数"),
)

MODE_ITEMS = (
    ('REPLACE', "替换", "清空现有内容后载入; 挂在本骨架下的旧碰撞体空物体会一并删除, 避免每次载入都多出一套"),
    ('APPEND', "追加", "保留现有内容, 配置接在后面, 碰撞体照样新建一套"),
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
    bones = report.get("missing_bones")
    if bones:
        operator.report({'WARNING'}, "碰撞体的父级骨骼不存在: %s"
                        % ", ".join(sorted(set(bones))[:4]))
    configs = report.get("missing_configs")
    if configs:
        operator.report({'WARNING'}, "本骨架没有这些配置, 引用未恢复: %s"
                        % ", ".join(sorted(set(configs))[:4]))
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
        if self.scope == 'CONFIG':
            base = settings.configs[settings.active_config_index].name
        elif self.scope == 'COLLIDERS':
            base = "%s.碰撞体" % context.object.name
        else:
            base = context.object.name
        self.filepath = os.path.join(config_io.preset_directory(create=True), "%s.json" % base)
        return super().invoke(context, event)

    def execute(self, context):
        settings = _settings(context)
        context.view_layer.update()
        payload = config_io.serialize(settings, self.scope, context.scene,
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
        report = config_io.deserialize(settings, payload, self.mode, context)
        if not _report_problems(self, report):
            return {'CANCELLED'}
        runtime.clear_registry()
        self.report({'INFO'}, "已载入 %s (%d 配置 / 新建 %d 碰撞体)"
                    % (os.path.basename(self.filepath), len(settings.configs),
                       report.get("created_colliders", 0)))
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
        report = config_io.deserialize(settings, payload, self.mode, context)
        if not _report_problems(self, report):
            return {'CANCELLED'}
        runtime.clear_registry()
        self.report({'INFO'}, "已应用 %s (新建 %d 碰撞体)"
                    % (self.preset, report.get("created_colliders", 0)))
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
