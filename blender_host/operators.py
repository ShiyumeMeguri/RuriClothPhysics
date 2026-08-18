import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty

from . import chain
from . import runtime


def _active_settings(context):
    obj = context.object
    if obj is None or obj.type != 'ARMATURE':
        return None
    return obj.ruri_cloth_physics


def _active_config(context):
    settings = _active_settings(context)
    if settings is None or len(settings.configs) == 0:
        return None
    index = min(settings.active_config_index, len(settings.configs) - 1)
    return settings.configs[index]


def _selected_bone_names(context):
    if context.mode == 'POSE':
        return [bone.name for bone in context.selected_pose_bones or ()]
    if context.mode == 'EDIT_ARMATURE':
        return [bone.name for bone in context.selected_bones or ()]
    obj = context.object
    if obj is not None and obj.type == 'ARMATURE':
        return chain.selected_names(obj)
    return []


def _unique_name(collection, base):
    if base not in collection:
        return base
    index = 1
    while "%s.%03d" % (base, index) in collection:
        index += 1
    return "%s.%03d" % (base, index)


def _mark_rebuild(config):
    config.rebuild_pending = True


class RCP_OT_config_add(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.config_add"
    bl_label = "添加配置"
    bl_description = "在配置栈中新建一个Ruri 布料物理配置"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_settings(context) is not None

    def execute(self, context):
        settings = _active_settings(context)
        config = settings.configs.add()
        config.name = _unique_name(settings.configs, "骨骼布料")
        settings.active_config_index = len(settings.configs) - 1
        return {'FINISHED'}


class RCP_OT_config_remove(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.config_remove"
    bl_label = "移除配置"
    bl_description = "删除当前选中的配置"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_config(context) is not None

    def execute(self, context):
        settings = _active_settings(context)
        index = settings.active_config_index
        config = settings.configs[index]
        if config.disable_mode == 'RESET':
            runtime.notify_config_enabled_changed(context.object, index)
        settings.configs.remove(index)
        settings.active_config_index = min(index, len(settings.configs) - 1)
        for other in settings.configs:
            other.rebuild_pending = True
        runtime.clear_registry()
        return {'FINISHED'}


class RCP_OT_config_move(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.config_move"
    bl_label = "移动配置"
    bl_description = "调整配置在栈中的顺序(顺序即执行顺序)"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(items=(('UP', "上移", ""), ('DOWN', "下移", "")))

    @classmethod
    def poll(cls, context):
        settings = _active_settings(context)
        return settings is not None and len(settings.configs) > 1

    def execute(self, context):
        settings = _active_settings(context)
        index = settings.active_config_index
        target = index - 1 if self.direction == 'UP' else index + 1
        if target < 0 or target >= len(settings.configs):
            return {'CANCELLED'}
        settings.configs.move(index, target)
        settings.configs[index].rebuild_pending = True
        settings.configs[target].rebuild_pending = True
        settings.active_config_index = target
        return {'FINISHED'}


class RCP_OT_collider_add(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.collider_add"
    bl_label = "添加碰撞体"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_settings(context) is not None

    def execute(self, context):
        settings = _active_settings(context)
        item = settings.colliders.add()
        item.name = _unique_name(settings.colliders, "碰撞体")
        settings.active_collider_index = len(settings.colliders) - 1
        settings.collider_serial += 1
        return {'FINISHED'}


class RCP_OT_collider_remove(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.collider_remove"
    bl_label = "移除碰撞体"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        settings = _active_settings(context)
        return settings is not None and len(settings.colliders) > 0

    def execute(self, context):
        settings = _active_settings(context)
        index = settings.active_collider_index
        settings.colliders.remove(index)
        settings.active_collider_index = min(index, len(settings.colliders) - 1)
        settings.collider_serial += 1
        return {'FINISHED'}


class RCP_OT_collider_move(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.collider_move"
    bl_label = "移动碰撞体"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(items=(('UP', "上移", ""), ('DOWN', "下移", "")))

    @classmethod
    def poll(cls, context):
        settings = _active_settings(context)
        return settings is not None and len(settings.colliders) > 1

    def execute(self, context):
        settings = _active_settings(context)
        index = settings.active_collider_index
        target = index - 1 if self.direction == 'UP' else index + 1
        if target < 0 or target >= len(settings.colliders):
            return {'CANCELLED'}
        settings.colliders.move(index, target)
        settings.active_collider_index = target
        settings.collider_serial += 1
        return {'FINISHED'}


class RCP_OT_collider_mirror(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.collider_mirror"
    bl_label = "镜像碰撞体"
    bl_description = "按左右命名规则(.L/.R)生成当前碰撞体的镜像副本"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        settings = _active_settings(context)
        return settings is not None and len(settings.colliders) > 0

    def execute(self, context):
        settings = _active_settings(context)
        source = settings.colliders[settings.active_collider_index]
        flipped_bone = bpy.utils.flip_name(source.bone) if source.bone else ""
        if not flipped_bone or flipped_bone == source.bone:
            self.report({'WARNING'}, "骨骼名没有可识别的左右后缀")
            return {'CANCELLED'}
        if flipped_bone not in context.object.data.bones:
            self.report({'WARNING'}, "镜像骨骼 %s 不存在" % flipped_bone)
            return {'CANCELLED'}

        item = settings.colliders.add()
        item.name = _unique_name(settings.colliders, bpy.utils.flip_name(source.name) or source.name)
        item.shape = source.shape
        item.bone = flipped_bone
        center = list(source.center)
        center[0] = -center[0]
        item.center = center
        item.radius = source.radius
        item.start_radius = source.start_radius
        item.end_radius = source.end_radius
        item.radius_separation = source.radius_separation
        item.length = source.length
        item.direction = source.direction
        item.reverse_direction = source.reverse_direction
        item.aligned_on_center = source.aligned_on_center
        settings.active_collider_index = len(settings.colliders) - 1
        settings.collider_serial += 1
        return {'FINISHED'}


class RCP_OT_colliders_from_selected(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.colliders_from_selected"
    bl_label = "从选中骨骼生成碰撞体"
    bl_description = "为每根选中骨骼创建一个沿骨骼方向的胶囊碰撞体, 并加入当前配置的引用列表"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_settings(context) is not None and len(_selected_bone_names(context)) > 0

    def execute(self, context):
        obj = context.object
        settings = _active_settings(context)
        config = _active_config(context)
        created = 0
        for name in _selected_bone_names(context):
            bone = obj.data.bones.get(name)
            if bone is None:
                continue
            item = settings.colliders.add()
            item.name = _unique_name(settings.colliders, name)
            item.shape = 'CAPSULE'
            item.bone = name
            item.direction = 'Y'
            item.aligned_on_center = True
            item.center = (0.0, bone.length * 0.5, 0.0)
            item.length = max(bone.length, 0.001)
            radius = max(bone.length * 0.2, 0.005)
            item.radius = radius
            item.start_radius = radius
            item.end_radius = radius
            if config is not None:
                reference = config.collider_collision.collider_references.add()
                reference.collider = item.name
            created += 1
        settings.collider_serial += 1
        self.report({'INFO'}, "已生成 %d 个碰撞体" % created)
        return {'FINISHED'}


LIST_IDS = (
    ('ROOT_BONES', "根骨骼", ""),
    ('ATTRIBUTE_OVERRIDES', "属性覆盖", ""),
    ('COLLIDER_REFERENCES', "碰撞体引用", ""),
    ('COLLISION_BONES', "碰撞骨骼", ""),
    ('SKINNING_BONES', "蒙皮骨骼", ""),
)


def _resolve_list(context, list_id):
    config = _active_config(context)
    if config is None:
        return None, None, None
    if list_id == 'ROOT_BONES':
        return config.root_bones, config, "active_root_bone_index"
    if list_id == 'ATTRIBUTE_OVERRIDES':
        return config.attribute_overrides, config, "active_attribute_override_index"
    if list_id == 'COLLIDER_REFERENCES':
        return (config.collider_collision.collider_references, config.collider_collision,
                "active_collider_reference_index")
    if list_id == 'COLLISION_BONES':
        return (config.collider_collision.collision_bones, config.collider_collision,
                "active_collision_bone_index")
    if list_id == 'SKINNING_BONES':
        return config.skinning_bones, config, "active_skinning_bone_index"
    return None, None, None


class RCP_OT_list_add(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.list_add"
    bl_label = "添加条目"
    bl_options = {'REGISTER', 'UNDO'}

    list_id: EnumProperty(items=LIST_IDS)

    @classmethod
    def poll(cls, context):
        return _active_config(context) is not None

    def execute(self, context):
        collection, owner, index_name = _resolve_list(context, self.list_id)
        if collection is None:
            return {'CANCELLED'}
        collection.add()
        setattr(owner, index_name, len(collection) - 1)
        config = _active_config(context)
        if self.list_id in {'ROOT_BONES', 'ATTRIBUTE_OVERRIDES', 'COLLISION_BONES',
                            'SKINNING_BONES'}:
            _mark_rebuild(config)
        return {'FINISHED'}


class RCP_OT_list_remove(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.list_remove"
    bl_label = "移除条目"
    bl_options = {'REGISTER', 'UNDO'}

    list_id: EnumProperty(items=LIST_IDS)

    @classmethod
    def poll(cls, context):
        return _active_config(context) is not None

    def execute(self, context):
        collection, owner, index_name = _resolve_list(context, self.list_id)
        if collection is None or len(collection) == 0:
            return {'CANCELLED'}
        index = getattr(owner, index_name)
        index = min(index, len(collection) - 1)
        collection.remove(index)
        setattr(owner, index_name, min(index, len(collection) - 1))
        config = _active_config(context)
        if self.list_id in {'ROOT_BONES', 'ATTRIBUTE_OVERRIDES', 'COLLISION_BONES',
                            'SKINNING_BONES'}:
            _mark_rebuild(config)
        else:
            context.object.ruri_cloth_physics.collider_serial += 1
        return {'FINISHED'}


class RCP_OT_list_move(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.list_move"
    bl_label = "移动条目"
    bl_options = {'REGISTER', 'UNDO'}

    list_id: EnumProperty(items=LIST_IDS)
    direction: EnumProperty(items=(('UP', "上移", ""), ('DOWN', "下移", "")))

    @classmethod
    def poll(cls, context):
        return _active_config(context) is not None

    def execute(self, context):
        collection, owner, index_name = _resolve_list(context, self.list_id)
        if collection is None or len(collection) < 2:
            return {'CANCELLED'}
        index = getattr(owner, index_name)
        target = index - 1 if self.direction == 'UP' else index + 1
        if target < 0 or target >= len(collection):
            return {'CANCELLED'}
        collection.move(index, target)
        setattr(owner, index_name, target)
        config = _active_config(context)
        if self.list_id == 'ROOT_BONES':
            _mark_rebuild(config)
        return {'FINISHED'}


class RCP_OT_bones_from_selected(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.bones_from_selected"
    bl_label = "从选中骨骼添加"
    bl_description = "把当前选中的骨骼批量加入列表(自动去重)"
    bl_options = {'REGISTER', 'UNDO'}

    list_id: EnumProperty(items=LIST_IDS)
    attribute: StringProperty(default="")

    @classmethod
    def poll(cls, context):
        return _active_config(context) is not None and len(_selected_bone_names(context)) > 0

    def execute(self, context):
        collection, owner, index_name = _resolve_list(context, self.list_id)
        if collection is None:
            return {'CANCELLED'}
        existing = {item.bone for item in collection}
        added = 0
        for name in _selected_bone_names(context):
            if name in existing:
                continue
            item = collection.add()
            item.bone = name
            if self.list_id == 'ATTRIBUTE_OVERRIDES' and self.attribute:
                item.attribute = self.attribute
            existing.add(name)
            added += 1
        if added:
            setattr(owner, index_name, len(collection) - 1)
            _mark_rebuild(_active_config(context))
        self.report({'INFO'}, "已添加 %d 根骨骼" % added)
        return {'FINISHED'}


class RCP_OT_wind_zone_add(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.wind_zone_add"
    bl_label = "新建风区"
    bl_description = "在游标处创建一个风区空物体"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        empty = bpy.data.objects.new("风区", None)
        empty.empty_display_type = 'SINGLE_ARROW'
        empty.empty_display_size = 0.5
        empty.location = context.scene.cursor.location
        context.collection.objects.link(empty)
        empty.ruri_cloth_physics_wind.is_wind_zone = True
        context.view_layer.objects.active = empty
        return {'FINISHED'}


class RCP_OT_wind_zone_convert(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.wind_zone_convert"
    bl_label = "设为风区"
    bl_description = "把当前空物体标记为风区"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'EMPTY'

    def execute(self, context):
        context.object.ruri_cloth_physics_wind.is_wind_zone = True
        return {'FINISHED'}


class RCP_OT_wind_zone_remove(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.wind_zone_remove"
    bl_label = "取消风区"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and getattr(obj, "ruri_cloth_physics_wind", None) is not None \
            and obj.ruri_cloth_physics_wind.is_wind_zone

    def execute(self, context):
        context.object.ruri_cloth_physics_wind.is_wind_zone = False
        return {'FINISHED'}


class RCP_OT_reset(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.reset"
    bl_label = "复位模拟"
    bl_description = "复位本骨架的全部布料模拟"
    bl_options = {'REGISTER'}

    mode: EnumProperty(
        name="模式",
        items=(('FULL', "完全清空", "回到动画姿势并清零速度"),
               ('KEEP', "保留位置", "保持当前形状, 只消除整体惯性")),
        default='FULL')

    @classmethod
    def poll(cls, context):
        return _active_settings(context) is not None

    def execute(self, context):
        runtime.request_reset(context.object, self.mode)
        return {'FINISHED'}


class RCP_OT_root_add_selected(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.root_add_selected"
    bl_label = "选中骨骼设为根骨骼"
    bl_description = "把视口里选中的骨骼加入当前配置的根骨骼列表(自动去重)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_config(context) is not None and len(_selected_bone_names(context)) > 0

    def execute(self, context):
        config = _active_config(context)
        obj = context.object
        existing = {item.bone for item in config.root_bones}
        added = []
        for name in _selected_bone_names(context):
            if name in existing:
                continue
            # A bone already driven as somebody's child would become a second, competing root of
            # the same chain, so say so instead of quietly building an ambiguous topology.
            owner_index, owner_root = chain.owning_root(obj, name)
            if owner_index is not None and owner_root != name:
                self.report({'WARNING'}, "%s 已属于 %s 的链, 跳过"
                            % (name, obj.ruri_cloth_physics.configs[owner_index].name))
                continue
            config.root_bones.add().bone = name
            existing.add(name)
            added.append(name)
        if not added:
            self.report({'INFO'}, "没有新增根骨骼")
            return {'CANCELLED'}
        config.active_root_bone_index = len(config.root_bones) - 1
        _mark_rebuild(config)
        self.report({'INFO'}, "已添加 %d 根: %s" % (len(added), ", ".join(added[:4])))
        return {'FINISHED'}


class RCP_OT_root_remove_selected(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.root_remove_selected"
    bl_label = "移除选中骨骼的链"
    bl_description = ("从根骨骼列表移除选中的骨骼; 选中的若是链中的子骨骼, "
                      "则向上找到第一个动态根骨骼并移除它")
    bl_options = {'REGISTER', 'UNDO'}

    all_configs: BoolProperty(
        name="搜索全部配置", default=True,
        description="关闭时只在当前配置里找; 打开时选中哪条链就删哪条, 不必先切到它的配置")

    @classmethod
    def poll(cls, context):
        return _active_settings(context) is not None and len(_selected_bone_names(context)) > 0

    def execute(self, context):
        obj = context.object
        settings = obj.ruri_cloth_physics
        active = _active_config(context)
        targets = {}
        for name in _selected_bone_names(context):
            config_index, root_name = chain.owning_root(obj, name)
            if root_name is None:
                continue
            if not self.all_configs and settings.configs[config_index] is not active:
                continue
            targets.setdefault(config_index, set()).add(root_name)
        if not targets:
            self.report({'WARNING'}, "选中的骨骼不属于任何布料链")
            return {'CANCELLED'}

        removed = []
        for config_index, roots in targets.items():
            config = settings.configs[config_index]
            for index in range(len(config.root_bones) - 1, -1, -1):
                if config.root_bones[index].bone in roots:
                    removed.append(config.root_bones[index].bone)
                    config.root_bones.remove(index)
            config.active_root_bone_index = max(0, min(config.active_root_bone_index,
                                                       len(config.root_bones) - 1))
            _mark_rebuild(config)
        self.report({'INFO'}, "已移除 %d 条链: %s" % (len(removed), ", ".join(removed[:4])))
        return {'FINISHED'}


class RCP_OT_chain_select(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.chain_select"
    bl_label = "选中本配置的全部骨骼"
    bl_description = "在视口里选中当前配置驱动的所有骨骼, 根骨骼设为激活骨骼"
    bl_options = {'REGISTER', 'UNDO'}

    roots_only: BoolProperty(name="只选根骨骼", default=False)

    @classmethod
    def poll(cls, context):
        return _active_config(context) is not None

    def execute(self, context):
        obj = context.object
        config = _active_config(context)
        roots, ordered = chain.config_chain(obj, config)
        wanted = roots if self.roots_only else ordered
        if not wanted:
            self.report({'WARNING'}, "本配置没有可用的根骨骼")
            return {'CANCELLED'}
        chain.select(obj, wanted, active=roots[0] if roots else None)
        self.report({'INFO'}, "已选中 %d 根骨骼" % len(wanted))
        return {'FINISHED'}


class RCP_OT_config_from_selected(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.config_from_selected"
    bl_label = "选中骨骼新建配置"
    bl_description = "用当前选中的骨骼直接新建一个配置并设为其根骨骼"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_settings(context) is not None and len(_selected_bone_names(context)) > 0

    def execute(self, context):
        settings = _active_settings(context)
        names = _selected_bone_names(context)
        config = settings.configs.add()
        config.name = _unique_name(settings.configs, names[0])
        for name in names:
            config.root_bones.add().bone = name
        settings.active_config_index = len(settings.configs) - 1
        config.active_root_bone_index = len(config.root_bones) - 1
        _mark_rebuild(config)
        settings.show_bones = True
        self.report({'INFO'}, "已新建配置 %s (%d 根)" % (config.name, len(names)))
        return {'FINISHED'}


class RCP_MT_bones(bpy.types.Menu):
    """Right-click entry point. Right-click any row here to put it on the Q menu."""

    bl_idname = "RCP_MT_bones"
    bl_label = "Ruri 布料物理"

    def draw(self, context):
        layout = self.layout
        settings = _active_settings(context)
        if settings is None:
            layout.label(text="当前对象不是骨架", icon='ERROR')
            return
        layout.prop(settings, "show_bones", toggle=True,
                    icon='HIDE_OFF' if settings.show_bones else 'HIDE_ON')
        layout.prop(settings, "show_colliders", toggle=True,
                    icon='MESH_CAPSULE')
        layout.separator()
        layout.operator("ruri_cloth_physics.root_add_selected", icon='ADD')
        layout.operator("ruri_cloth_physics.root_remove_selected", icon='REMOVE')
        layout.operator("ruri_cloth_physics.config_from_selected", icon='DUPLICATE')
        layout.separator()
        layout.operator("ruri_cloth_physics.chain_select", icon='RESTRICT_SELECT_OFF')
        operator = layout.operator("ruri_cloth_physics.chain_select", icon='PINNED',
                                   text="只选中根骨骼")
        operator.roots_only = True
        layout.separator()
        operator = layout.operator("ruri_cloth_physics.bones_from_selected", icon='PINNED',
                                   text="选中骨骼设为固定")
        operator.list_id = 'ATTRIBUTE_OVERRIDES'
        operator.attribute = 'FIXED'
        layout.operator("ruri_cloth_physics.colliders_from_selected", icon='MESH_CAPSULE')
        layout.separator()
        layout.operator("ruri_cloth_physics.reset", icon='FILE_REFRESH')


def _draw_menu(self, context):
    obj = context.object
    if obj is not None and obj.type == 'ARMATURE' \
            and getattr(obj, "ruri_cloth_physics", None) is not None:
        self.layout.separator()
        self.layout.menu(RCP_MT_bones.bl_idname, icon='MOD_CLOTH')


_MENUS = ("VIEW3D_MT_pose_context_menu", "VIEW3D_MT_armature_context_menu",
          "VIEW3D_MT_object_context_menu")


_CLASSES = (
    RCP_OT_config_add,
    RCP_OT_config_remove,
    RCP_OT_config_move,
    RCP_OT_collider_add,
    RCP_OT_collider_remove,
    RCP_OT_collider_move,
    RCP_OT_collider_mirror,
    RCP_OT_colliders_from_selected,
    RCP_OT_list_add,
    RCP_OT_list_remove,
    RCP_OT_list_move,
    RCP_OT_bones_from_selected,
    RCP_OT_root_add_selected,
    RCP_OT_root_remove_selected,
    RCP_OT_chain_select,
    RCP_OT_config_from_selected,
    RCP_MT_bones,
    RCP_OT_wind_zone_add,
    RCP_OT_wind_zone_convert,
    RCP_OT_wind_zone_remove,
    RCP_OT_reset,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    for name in _MENUS:
        menu = getattr(bpy.types, name, None)
        if menu is not None:
            menu.append(_draw_menu)


def unregister():
    for name in _MENUS:
        menu = getattr(bpy.types, name, None)
        if menu is not None:
            menu.remove(_draw_menu)
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
