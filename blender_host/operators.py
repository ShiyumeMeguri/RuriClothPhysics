import bpy
import mathutils
from bpy.props import BoolProperty, EnumProperty, StringProperty

from . import chain
from . import collider_geom
from . import properties
from . import runtime
from . import selection_sync
from . import wind_geom


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


def _set_active_index(owner, index_name, value):
    with selection_sync.suppressed():
        setattr(owner, index_name, value)


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


def _collider_collection(context, armature_object):
    return collider_geom.collection_for(context.scene, armature_object, create=True)


def _link_empty(context, collection, name, display_type, display_size, location):
    empty = collider_geom.new_empty(collection, name, display_type, display_size)
    empty.location = location
    return empty


def _reference_collider(config, empty):
    if config is None:
        return
    for reference in config.collider_collision.collider_references:
        if reference.object is empty:
            return
    with selection_sync.suppressed():
        config.collider_collision.collider_references.add().object = empty
        config.collider_collision.active_collider_reference_index = \
            len(config.collider_collision.collider_references) - 1


def _make_collider(context, collection, name, shape, location, radius):
    empty = _link_empty(context, collection, name, collider_geom.DISPLAY_TYPE[shape],
                        radius, location)
    settings = empty.ruri_cloth_physics_collider
    settings.is_collider = True
    settings.shape = shape
    return empty


def _make_capsule(context, collection, name, start_location, end_location, radius):
    empty = _make_collider(context, collection, name, 'CAPSULE', start_location, radius)
    end = _link_empty(context, collection, name + ".end", collider_geom.END_DISPLAY_TYPE,
                      radius, end_location)
    empty.ruri_cloth_physics_collider.end_object = end
    return empty, end


def _active_bone(context):
    obj = context.object
    if obj is None or obj.type != 'ARMATURE' or obj.pose is None:
        return None, None
    bone = obj.data.bones.active
    if bone is None:
        return obj, None
    return obj, obj.pose.bones.get(bone.name)


class RCP_OT_collider_create(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.collider_create"
    bl_label = "新建碰撞体"
    bl_description = ("在本骨架的碰撞体集合里新建一个碰撞体空物体; 有活动骨骼时贴合它并加子级约束跟随, "
                      "否则放在 3D 游标处。之后直接用 G/R/S 摆它就行")
    bl_options = {'REGISTER', 'UNDO'}

    shape: EnumProperty(name="形状", items=properties.COLLIDER_SHAPE_ITEMS, default='SPHERE')

    @classmethod
    def poll(cls, context):
        return _active_settings(context) is not None

    def execute(self, context):
        context.view_layer.update()
        armature_object, pose_bone = _active_bone(context)
        config = _active_config(context)
        collection = _collider_collection(context, armature_object)
        if pose_bone is not None:
            matrix = armature_object.matrix_world
            head = matrix @ pose_bone.head
            tail = matrix @ pose_bone.tail
            radius = max(pose_bone.length * 0.2, 0.005)
            base = pose_bone.name
        else:
            head = context.scene.cursor.location.copy()
            tail = head.copy()
            tail.z += 0.2
            radius = 0.05
            base = "碰撞体"
        parts = []
        if self.shape == 'CAPSULE':
            empty, end = _make_capsule(context, collection, base + ".胶囊", head, tail, radius)
            parts = [empty, end]
        else:
            center = head if self.shape == 'PLANE' else (head + tail) * 0.5
            empty = _make_collider(context, collection, base + ".碰撞体", self.shape,
                                   center, radius)
            parts = [empty]
        if pose_bone is not None:
            for part in parts:
                collider_geom.attach_to_bone(context.view_layer, part, armature_object,
                                             pose_bone.name)
        _reference_collider(config, empty)
        self.report({'INFO'}, "已新建 %s" % empty.name)
        return {'FINISHED'}


class RCP_OT_collider_convert(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.collider_convert"
    bl_label = "设为碰撞体"
    bl_description = "把当前空物体标记为碰撞体"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'EMPTY'

    def execute(self, context):
        settings = context.object.ruri_cloth_physics_collider
        settings.is_collider = True
        collider_geom.sync_display(context.object)
        return {'FINISHED'}


class RCP_OT_collider_clear(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.collider_clear"
    bl_label = "取消碰撞体"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return collider_geom.is_collider(context.object)

    def execute(self, context):
        context.object.ruri_cloth_physics_collider.is_collider = False
        return {'FINISHED'}


class RCP_OT_collider_end_create(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.collider_end_create"
    bl_label = "新建终点圆"
    bl_description = "为当前胶囊碰撞体创建终点圆空物体, 并作为子级放在起点圆上方"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not collider_geom.is_collider(context.object):
            return False
        settings = context.object.ruri_cloth_physics_collider
        return settings.shape == 'CAPSULE' and collider_geom.end_object(settings) is None

    def execute(self, context):
        context.view_layer.update()
        obj = context.object
        settings = obj.ruri_cloth_physics_collider
        offset = mathutils.Vector((0.0, 0.0, max(obj.empty_display_size, 0.001) * 4.0))
        collection = obj.users_collection[0] if obj.users_collection else context.collection
        end = _link_empty(context, collection, obj.name + ".end", collider_geom.END_DISPLAY_TYPE,
                          obj.empty_display_size, obj.matrix_world.translation + offset)
        constraint = collider_geom.bone_constraint(obj)
        if constraint is not None and constraint.target is not None:
            collider_geom.attach_to_bone(context.view_layer, end, constraint.target,
                                         constraint.subtarget)
        settings.end_object = end
        return {'FINISHED'}


class RCP_OT_collider_attach_bone(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.collider_attach_bone"
    bl_label = "挂到活动骨骼"
    bl_description = ("给选中的碰撞体加一个跟随活动骨骼的子级约束, 保持当前位置不变。"
                      "不改父级、不进骨架层级, 删掉约束就彻底脱开")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        _armature_object, pose_bone = _active_bone(context)
        return pose_bone is not None

    def execute(self, context):
        armature_object, pose_bone = _active_bone(context)
        attached = []
        for empty in context.selected_objects:
            if not collider_geom.is_collider(empty):
                continue
            collider_geom.attach_to_bone(context.view_layer, empty, armature_object,
                                         pose_bone.name)
            end = collider_geom.end_object(empty.ruri_cloth_physics_collider)
            if end is not None:
                collider_geom.attach_to_bone(context.view_layer, end, armature_object,
                                             pose_bone.name)
            attached.append(empty.name)
        if not attached:
            self.report({'WARNING'}, "没有选中任何碰撞体空物体")
            return {'CANCELLED'}
        self.report({'INFO'}, "已挂 %d 个到 %s" % (len(attached), pose_bone.name))
        return {'FINISHED'}


class RCP_OT_collider_detach(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.collider_detach"
    bl_label = "脱开骨骼"
    bl_description = "删掉选中碰撞体的跟随约束, 让它停在世界空间不再跟骨骼动"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return collider_geom.is_collider(context.object)

    def execute(self, context):
        context.view_layer.update()
        detached = 0
        for empty in context.selected_objects:
            if not collider_geom.is_collider(empty):
                continue
            for target in (empty, collider_geom.end_object(empty.ruri_cloth_physics_collider)):
                if target is None or collider_geom.bone_constraint(target) is None:
                    continue
                world = target.matrix_world.copy()
                collider_geom.detach(target)
                target.matrix_basis = world
                detached += 1
        context.view_layer.update()
        if not detached:
            self.report({'WARNING'}, "选中的碰撞体本来就没挂骨骼")
            return {'CANCELLED'}
        self.report({'INFO'}, "已脱开 %d 个" % detached)
        return {'FINISHED'}


def _mirror_sources(context):
    scene = context.scene
    found = []
    seen = set()
    candidates = list(context.selected_objects)
    if context.object is not None:
        candidates.append(context.object)
    for obj in candidates:
        collider = obj if collider_geom.is_collider(obj) else collider_geom.owner_of(scene, obj)
        if collider is None or collider.name in seen:
            continue
        seen.add(collider.name)
        found.append(collider)
    return found


def _mirror_space(context, obj):
    constraint = collider_geom.bone_constraint(obj)
    if constraint is not None and constraint.target is not None:
        return constraint.target
    active = context.object
    if active is not None and active.type == 'ARMATURE':
        return active
    return None


def _mirror_world(space, matrix):
    flip = mathutils.Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0))
    if space is None:
        return flip @ matrix @ flip
    basis = space.matrix_world
    return basis @ flip @ basis.inverted_safe() @ matrix @ flip


def _collection_of(context, obj):
    return obj.users_collection[0] if obj.users_collection else context.collection


class RCP_OT_collider_mirror(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.collider_mirror"
    bl_label = "镜像碰撞体"
    bl_description = ("按左右命名规则镜像选中的碰撞体, 胶囊连同它的两个圆一起。"
                      "对侧同名碰撞体已存在就直接更新它, 不会再堆一份; "
                      "镜像在骨架局部空间进行, 不使用负缩放")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(_mirror_sources(context))

    def _place(self, context, source, target):
        target.empty_display_size = source.empty_display_size
        constraint = collider_geom.bone_constraint(source)
        world = _mirror_world(_mirror_space(context, source), source.matrix_world)
        if constraint is None or constraint.target is None:
            collider_geom.detach(target)
            context.view_layer.update()
            target.matrix_basis = world
            context.view_layer.update()
            return
        bone_name = collider_geom.flip_side_name(constraint.subtarget)
        if bone_name not in constraint.target.pose.bones:
            bone_name = constraint.subtarget
        collider_geom.attach_to_bone(context.view_layer, target, constraint.target, bone_name)
        collider_geom.set_world(context.view_layer, target, world)

    def _mirrored_end(self, context, source_end, target, target_settings):
        end = target_settings.end_object
        if end is None or end is source_end or end is target or collider_geom.is_collider(end):
            end = collider_geom.new_empty(
                _collection_of(context, target),
                collider_geom.flip_side_name(source_end.name) or (target.name + ".end"),
                collider_geom.END_DISPLAY_TYPE, source_end.empty_display_size)
            target_settings.end_object = end
        return end

    def _apply(self, context, source, target):
        source_settings = source.ruri_cloth_physics_collider
        target_settings = target.ruri_cloth_physics_collider
        target_settings.is_collider = True
        target_settings.shape = source_settings.shape
        target_settings.enabled = source_settings.enabled
        self._place(context, source, target)
        source_end = source_settings.end_object
        if source_end is None:
            return
        self._place(context, source_end, self._mirrored_end(context, source_end, target,
                                                            target_settings))

    def _counterpart(self, source, name):
        target = bpy.data.objects.get(name) if name else None
        if target is None or target is source or not collider_geom.is_collider(target):
            return None
        return target

    def _create(self, context, source, name):
        copy = source.copy()
        copy.name = name or source.name
        _collection_of(context, source).objects.link(copy)
        copy.ruri_cloth_physics_collider.end_object = None
        return copy

    def execute(self, context):
        context.view_layer.update()
        config = _active_config(context)
        created = []
        updated = []
        unnamed = []
        for source in _mirror_sources(context):
            name = collider_geom.flip_side_name(source.name)
            if not name:
                unnamed.append(source.name)
            target = self._counterpart(source, name)
            if target is None:
                target = self._create(context, source, name)
                created.append(target.name)
            else:
                updated.append(target.name)
            self._apply(context, source, target)
            _reference_collider(config, target)
        if not created and not updated:
            self.report({'WARNING'}, "没有选中任何碰撞体")
            return {'CANCELLED'}
        if unnamed:
            self.report({'WARNING'}, "%d 个名字里没有左右标记, 只能新建副本: %s"
                        % (len(unnamed), ", ".join(sorted(unnamed)[:3])))
        self.report({'INFO'}, "已镜像 %d 个: 更新 %d, 新建 %d"
                    % (len(created) + len(updated), len(updated), len(created)))
        return {'FINISHED'}


class RCP_OT_colliders_from_selected(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.colliders_from_selected"
    bl_label = "从选中骨骼生成胶囊"
    bl_description = ("为每根选中骨骼生成一根贴合它的胶囊碰撞体(两个圆形空物体, 分别在骨骼首尾), "
                      "父级到该骨骼并加入当前配置的引用列表")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_settings(context) is not None and len(_selected_bone_names(context)) > 0

    def execute(self, context):
        context.view_layer.update()
        obj = context.object
        config = _active_config(context)
        collection = _collider_collection(context, obj)
        created = 0
        for name in _selected_bone_names(context):
            pose_bone = obj.pose.bones.get(name)
            if pose_bone is None:
                continue
            matrix = obj.matrix_world
            radius = max(pose_bone.length * 0.2, 0.005)
            empty, end = _make_capsule(context, collection, name + ".胶囊",
                                       matrix @ pose_bone.head, matrix @ pose_bone.tail, radius)
            for part in (empty, end):
                collider_geom.attach_to_bone(context.view_layer, part, obj, name)
            _reference_collider(config, empty)
            created += 1
        if not created:
            self.report({'WARNING'}, "选中的骨骼都不在姿态里")
            return {'CANCELLED'}
        self.report({'INFO'}, "已生成 %d 根胶囊" % created)
        return {'FINISHED'}


class RCP_OT_collider_reference_clean(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.collider_reference_clean"
    bl_label = "清理失效引用"
    bl_description = "删掉本骨架所有配置里指向空白或已不是碰撞体的引用行"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_settings(context) is not None

    def execute(self, context):
        removed = 0
        for config in _active_settings(context).configs:
            collision = config.collider_collision
            references = collision.collider_references
            for index in range(len(references) - 1, -1, -1):
                if not collider_geom.is_collider(references[index].object):
                    references.remove(index)
                    removed += 1
            collision.active_collider_reference_index = max(0, min(
                collision.active_collider_reference_index, len(references) - 1))
        if not removed:
            self.report({'INFO'}, "没有失效引用")
            return {'CANCELLED'}
        self.report({'INFO'}, "已清理 %d 行失效引用" % removed)
        return {'FINISHED'}


class RCP_OT_collider_reference_add_selected(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.collider_reference_add_selected"
    bl_label = "添加选中的碰撞体"
    bl_description = "把视口里选中的碰撞体空物体加入当前配置的引用列表(自动去重)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_config(context) is not None

    def execute(self, context):
        config = _active_config(context)
        added = 0
        for empty in context.selected_objects:
            if not collider_geom.is_collider(empty):
                continue
            before = len(config.collider_collision.collider_references)
            _reference_collider(config, empty)
            added += len(config.collider_collision.collider_references) - before
        if not added:
            self.report({'WARNING'}, "选中的对象里没有新的碰撞体")
            return {'CANCELLED'}
        self.report({'INFO'}, "已添加 %d 个碰撞体引用" % added)
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
        _set_active_index(owner, index_name, len(collection) - 1)
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
        _set_active_index(owner, index_name, min(index, len(collection) - 1))
        config = _active_config(context)
        if self.list_id in {'ROOT_BONES', 'ATTRIBUTE_OVERRIDES', 'COLLISION_BONES',
                            'SKINNING_BONES'}:
            _mark_rebuild(config)
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
        _set_active_index(owner, index_name, target)
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
            _set_active_index(owner, index_name, len(collection) - 1)
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
        empty.empty_display_size = 1.0
        empty.location = context.scene.cursor.location
        context.collection.objects.link(empty)
        wind = empty.ruri_cloth_physics_wind
        wind.is_wind_zone = True
        wind_geom.sync_display(empty, wind)
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


class RCP_OT_bone_exclude_selected(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.bone_exclude_selected"
    bl_label = "排除选中骨骼"
    bl_description = ("把选中的骨骼(连同其子级)踢出当前配置的模拟; "
                      "用于挂在身体骨下面却不属于该部位的东西, 比如挂在胸骨上的布料骨")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_config(context) is not None and len(_selected_bone_names(context)) > 0

    def execute(self, context):
        obj = context.object
        config = _active_config(context)
        _, ordered = chain.config_chain(obj, config)
        inside = set(ordered)
        roots = set(chain.root_names(obj, config))
        existing = {override.bone: override for override in config.attribute_overrides}
        excluded = []
        for name in _selected_bone_names(context):
            if name in roots:
                self.report({'WARNING'}, "%s 是本配置的根骨骼, 请改用移除选中" % name)
                continue
            if name not in inside:
                continue
            override = existing.get(name)
            if override is None:
                override = config.attribute_overrides.add()
                override.bone = name
            override.attribute = 'IGNORE'
            excluded.append(name)
        if not excluded:
            self.report({'WARNING'}, "选中的骨骼不在本配置的链里")
            return {'CANCELLED'}
        _mark_rebuild(config)
        remaining = len(chain.config_chain(obj, config)[1])
        self.report({'INFO'}, "已排除 %d 根, 链剩 %d 根骨骼" % (len(excluded), remaining))
        return {'FINISHED'}


class RCP_OT_bone_include_selected(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.bone_include_selected"
    bl_label = "取消排除"
    bl_description = "把选中骨骼上的排除标记去掉, 让它重新参与当前配置的模拟"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_config(context) is not None and len(_selected_bone_names(context)) > 0

    def execute(self, context):
        config = _active_config(context)
        wanted = set(_selected_bone_names(context))
        restored = []
        for index in range(len(config.attribute_overrides) - 1, -1, -1):
            override = config.attribute_overrides[index]
            if override.attribute == 'IGNORE' and override.bone in wanted:
                restored.append(override.bone)
                config.attribute_overrides.remove(index)
        if not restored:
            self.report({'WARNING'}, "选中骨骼上没有排除标记")
            return {'CANCELLED'}
        _mark_rebuild(config)
        self.report({'INFO'}, "已恢复 %d 根骨骼" % len(restored))
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


def _pin_roots(config):
    declared = {item.bone for item in config.root_bones}
    repaired = []
    for override in config.attribute_overrides:
        if override.bone in declared and override.attribute != 'FIXED':
            override.attribute = 'FIXED'
            repaired.append(override.bone)
    return repaired


class RCP_OT_repair_roots(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.repair_roots"
    bl_label = "修复被解除固定的根骨骼"
    bl_description = ("属性覆盖把根骨骼从固定改成了移动, 该链会失去锚点整条飘走; "
                      "点此把这些覆盖改回固定")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        config = _active_config(context)
        return config is not None and bool(chain.unpinned_roots(context.object, config))

    def execute(self, context):
        config = _active_config(context)
        repaired = _pin_roots(config)
        if not repaired:
            return {'CANCELLED'}
        _mark_rebuild(config)
        self.report({'INFO'}, "已重新固定: %s" % ", ".join(repaired))
        return {'FINISHED'}


class RCP_OT_promote_degenerate(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.promote_degenerate"
    bl_label = "修复退化链"
    bl_description = ("把与父骨几乎重合的子骨升为根骨骼, 并移除退化的中枢骨; "
                      "零长约束会让该链剧烈自转")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        config = _active_config(context)
        return config is not None and bool(chain.degenerate_links(context.object, config))

    def execute(self, context):
        obj = context.object
        config = _active_config(context)
        links = chain.degenerate_links(obj, config)
        if not links:
            return {'CANCELLED'}
        hubs = {parent for _, parent, _, _ in links}
        promoted = {name for name, _, _, _ in links}
        names = [item.bone for item in config.root_bones]
        rebuilt = []
        for name in names:
            if name in hubs:
                rebuilt.extend(sorted(child for child, parent, _, _ in links if parent == name))
            else:
                rebuilt.append(name)
        for child in sorted(promoted):
            if child not in rebuilt:
                rebuilt.append(child)
        config.root_bones.clear()
        for name in rebuilt:
            config.root_bones.add().bone = name
        config.active_root_bone_index = 0
        _pin_roots(config)
        _mark_rebuild(config)
        self.report({'INFO'}, "已升为根: %s" % ", ".join(sorted(promoted)))
        return {'FINISHED'}


class RCP_MT_bones(bpy.types.Menu):

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
        layout.separator()
        layout.operator("ruri_cloth_physics.root_add_selected", icon='ADD')
        layout.operator("ruri_cloth_physics.root_remove_selected", icon='REMOVE')
        layout.operator("ruri_cloth_physics.bone_exclude_selected", icon='X')
        layout.operator("ruri_cloth_physics.bone_include_selected", icon='CHECKMARK')
        layout.operator("ruri_cloth_physics.config_from_selected", icon='DUPLICATE')
        layout.separator()
        layout.operator("ruri_cloth_physics.promote_degenerate", icon='ERROR')
        layout.operator("ruri_cloth_physics.repair_roots", icon='PINNED')
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
        layout.operator("ruri_cloth_physics.collider_attach_bone", icon='BONE_DATA')
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
    RCP_OT_collider_create,
    RCP_OT_collider_convert,
    RCP_OT_collider_clear,
    RCP_OT_collider_end_create,
    RCP_OT_collider_attach_bone,
    RCP_OT_collider_detach,
    RCP_OT_collider_mirror,
    RCP_OT_colliders_from_selected,
    RCP_OT_collider_reference_add_selected,
    RCP_OT_collider_reference_clean,
    RCP_OT_list_add,
    RCP_OT_list_remove,
    RCP_OT_list_move,
    RCP_OT_bones_from_selected,
    RCP_OT_root_add_selected,
    RCP_OT_root_remove_selected,
    RCP_OT_chain_select,
    RCP_OT_bone_exclude_selected,
    RCP_OT_bone_include_selected,
    RCP_OT_config_from_selected,
    RCP_OT_promote_degenerate,
    RCP_OT_repair_roots,
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
