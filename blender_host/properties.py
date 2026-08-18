import re

import bpy
import mathutils
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from . import bone_binding
from . import curve_host

_CONFIG_PATH_PATTERN = re.compile(r"ruri_cloth_physics\.configs\[(\d+)\]")


def _owner_object(self):
    id_data = self.id_data
    return id_data if isinstance(id_data, bpy.types.Object) else None


def _redraw(self=None, context=None):
    from . import viewport
    viewport.tag_redraw()


def _param_update(self, context):
    obj = _owner_object(self)
    if obj is not None and obj.type == 'ARMATURE':
        obj.ruri_cloth_physics.param_serial += 1
    _redraw()


def _topology_update(self, context):
    obj = _owner_object(self)
    if obj is None or obj.type != 'ARMATURE':
        return
    settings = obj.ruri_cloth_physics
    match = _CONFIG_PATH_PATTERN.search(self.path_from_id())
    if match is not None:
        index = int(match.group(1))
        if 0 <= index < len(settings.configs):
            settings.configs[index].rebuild_pending = True
    else:
        for config in settings.configs:
            config.rebuild_pending = True
    settings.param_serial += 1
    _redraw()


def _collider_update(self, context):
    obj = _owner_object(self)
    if obj is not None and obj.type == 'ARMATURE':
        obj.ruri_cloth_physics.collider_serial += 1
    _redraw()


def _use_curve_update(self, context):
    if self.use_curve:
        curve_host.ensure_curve_node(self)
    _param_update(self, context)


def _enabled_update(self, context):
    obj = _owner_object(self)
    if obj is None or obj.type != 'ARMATURE':
        return
    from . import runtime
    match = _CONFIG_PATH_PATTERN.search(self.path_from_id())
    if match is not None:
        runtime.notify_config_enabled_changed(obj, int(match.group(1)))


_curve_classes = []


def _make_curve_type(type_name, default_value, min_value, max_value,
                     default_start=1.0, default_end=1.0, default_use=False,
                     soft_min=None, soft_max=None):
    def _points_get(self):
        return curve_host.serialize_points(self)

    def _points_set(self, text):
        curve_host.deserialize_points(self, text)

    annotations = {
        "value": FloatProperty(
            name="值", default=default_value,
            min=min_value, max=max_value,
            soft_min=min_value if soft_min is None else soft_min,
            soft_max=max_value if soft_max is None else soft_max,
            update=_param_update),
        "use_curve": BoolProperty(name="随深度曲线调制", default=default_use, update=_use_curve_update),
        "node_name": StringProperty(default="", options={'HIDDEN'}),
        "points_serialized": StringProperty(
            name="曲线控制点", get=_points_get, set=_points_set, options={'HIDDEN'}),
    }
    cls = type(type_name, (bpy.types.PropertyGroup,), {
        "__annotations__": annotations,
        "default_start": default_start,
        "default_end": default_end,
    })
    _curve_classes.append(cls)
    return cls


RCPCurveDamping = _make_curve_type("RCPCurveDamping", 0.05, 0.0, 1.0)
RCPCurveRadius = _make_curve_type("RCPCurveRadius", 0.02, 0.001, 1.0, soft_max=0.5)
RCPCurveDistanceStiffness = _make_curve_type("RCPCurveDistanceStiffness", 1.0, 0.0, 1.0, 1.0, 0.5, False)
RCPCurveAngleStiffness = _make_curve_type("RCPCurveAngleStiffness", 0.2, 0.0, 1.0, 1.0, 0.2, True)
RCPCurveLimitAngle = _make_curve_type("RCPCurveLimitAngle", 60.0, 0.0, 180.0, 0.0, 1.0, True)
RCPCurveMaxDistance = _make_curve_type("RCPCurveMaxDistance", 0.3, 0.0, 5.0)
RCPCurveBackstopDistance = _make_curve_type("RCPCurveBackstopDistance", 0.0, 0.0, 1.0)
RCPCurvePushoutDistance = _make_curve_type("RCPCurvePushoutDistance", 0.05, 0.0, 1.0)
RCPCurveSurfaceThickness = _make_curve_type("RCPCurveSurfaceThickness", 0.005, 0.001, 0.05, 0.5, 1.0, False)
RCPCurveWindAttenuation = _make_curve_type("RCPCurveWindAttenuation", 1.0, 1.0, 1.0, 1.0, 0.0, True)


_check_classes = []


def _make_check_type(type_name, default_use, default_value, min_value, max_value):
    annotations = {
        "use": BoolProperty(name="启用", default=default_use, update=_param_update),
        "value": FloatProperty(name="值", default=default_value, min=min_value, max=max_value,
                               update=_param_update),
    }
    cls = type(type_name, (bpy.types.PropertyGroup,), {"__annotations__": annotations})
    _check_classes.append(cls)
    return cls


RCPCheckMovementSpeed = _make_check_type("RCPCheckMovementSpeed", True, 5.0, 0.0, 10.0)
RCPCheckRotationSpeed = _make_check_type("RCPCheckRotationSpeed", True, 720.0, 0.0, 1440.0)
RCPCheckLocalMovementSpeed = _make_check_type("RCPCheckLocalMovementSpeed", False, 5.0, 0.0, 10.0)
RCPCheckLocalRotationSpeed = _make_check_type("RCPCheckLocalRotationSpeed", False, 720.0, 0.0, 1440.0)
RCPCheckParticleSpeed = _make_check_type("RCPCheckParticleSpeed", True, 4.0, 0.0, 10.0)
RCPCheckCullingLength = _make_check_type("RCPCheckCullingLength", False, 30.0, 0.0, 100.0)


def _bone_property(name, update, target=None, uid_attribute="bone_uid"):
    getter, setter = bone_binding.make_accessors(uid_attribute, update, target)
    return StringProperty(name=name, get=getter, set=setter)


def _focus_bone(obj, name):
    """Make `name` the selected + active bone, so the 3D view follows the list row."""
    if obj is None or obj.type != 'ARMATURE' or not name:
        return
    from . import chain
    if obj.data.bones.get(name) is None:
        return
    chain.select(obj, [name], active=name)


def _root_bone_index_update(self, context):
    """Selecting a root row drives the viewport selection.

    The list and the 3D view are two views of one selection, so picking a row here has to move the
    armature's own selection -- otherwise every operator that reads selected bones (add, remove,
    make fixed) silently acts on whatever was highlighted before the user touched the list.
    """
    obj = _owner_object(self)
    if obj is None or obj.type != 'ARMATURE':
        return
    settings = obj.ruri_cloth_physics
    if not settings.sync_list_selection:
        return
    if 0 <= self.active_root_bone_index < len(self.root_bones):
        _focus_bone(obj, self.root_bones[self.active_root_bone_index].bone)


class RCPBoneReference(bpy.types.PropertyGroup):
    bone_uid: StringProperty(default="", options={'HIDDEN'})
    bone: _bone_property("骨骼", _topology_update)


class RCPColliderReference(bpy.types.PropertyGroup):
    collider: StringProperty(name="碰撞体", default="", update=_collider_update)


ATTRIBUTE_OVERRIDE_ITEMS = (
    ('FIXED', "固定", "该骨骼作为固定点(运动学)", 'PINNED', 0),
    ('MOVE', "移动", "该骨骼参与模拟", 'UNPINNED', 1),
    ('IGNORE', "无效", "该骨骼从模拟中剔除", 'X', 2),
)


class RCPAttributeOverride(bpy.types.PropertyGroup):
    bone_uid: StringProperty(default="", options={'HIDDEN'})
    bone: _bone_property("骨骼", _topology_update)
    attribute: EnumProperty(name="属性", items=ATTRIBUTE_OVERRIDE_ITEMS, default='MOVE',
                            update=_topology_update)
    disable_collision: BoolProperty(name="禁用碰撞", default=False, update=_topology_update)
    exclude_motion: BoolProperty(name="排除移动限制", default=False, update=_topology_update)


def _collider_rotation(item):
    """Rotation taking the collider's bone-local space to world, or None if it cannot be resolved."""
    obj = item.id_data
    if obj is None or getattr(obj, "type", None) != 'ARMATURE':
        return None
    if item.bone and obj.pose is not None:
        pose_bone = obj.pose.bones.get(item.bone)
        if pose_bone is not None:
            return (obj.matrix_world @ pose_bone.matrix).to_3x3()
    return obj.matrix_world.to_3x3()


def _center_world_get(self):
    """The same offset, expressed on WORLD axes.

    `center` is bone-local because that is what the solver consumes, but a bone's Y is the bone
    direction, so on this rig the head bone maps local X to world -X and local Z to world +Y --
    typing into a field labelled Z moved the collider forwards, and left/right came out mirrored.
    This projection is a pure view of `center`; the stored value stays the single truth.
    """
    rotation = _collider_rotation(self)
    if rotation is None:
        return tuple(self.center)
    return tuple(rotation @ mathutils.Vector(self.center))


def _center_world_set(self, value):
    rotation = _collider_rotation(self)
    if rotation is None:
        self.center = value
        return
    inverse = rotation.copy()
    inverse.invert_safe()
    self.center = tuple(inverse @ mathutils.Vector(value))


COLLIDER_SHAPE_ITEMS = (
    ('SPHERE', "球", "球形碰撞体", 'MESH_UVSPHERE', 0),
    ('CAPSULE', "胶囊", "胶囊碰撞体", 'MESH_CAPSULE', 1),
    ('PLANE', "平面", "无限平面碰撞体", 'MESH_PLANE', 2),
)

CAPSULE_DIRECTION_ITEMS = (
    ('X', "X 轴", "沿骨骼局部 X 轴"),
    ('Y', "Y 轴", "沿骨骼局部 Y 轴(骨骼方向)"),
    ('Z', "Z 轴", "沿骨骼局部 Z 轴"),
)


class RCPColliderItem(bpy.types.PropertyGroup):
    enabled: BoolProperty(name="启用", default=True, update=_collider_update)
    shape: EnumProperty(name="形状", items=COLLIDER_SHAPE_ITEMS, default='SPHERE',
                        update=_collider_update)
    bone_uid: StringProperty(default="", options={'HIDDEN'})
    bone: _bone_property("骨骼", _collider_update)
    center: FloatVectorProperty(name="中心偏移", size=3, default=(0.0, 0.0, 0.0),
                                subtype='TRANSLATION', update=_collider_update)
    center_world: FloatVectorProperty(
        name="中心偏移(世界轴)", size=3, subtype='TRANSLATION',
        description="与上面同一个偏移, 但按世界 XYZ 表示: 上下就是上下, 左右就是左右",
        get=_center_world_get, set=_center_world_set)
    radius: FloatProperty(name="半径", default=0.05, min=0.001, soft_max=0.5,
                          update=_collider_update)
    start_radius: FloatProperty(name="始端半径", default=0.05, min=0.001, soft_max=0.5,
                                update=_collider_update)
    end_radius: FloatProperty(name="终端半径", default=0.05, min=0.001, soft_max=0.5,
                              update=_collider_update)
    radius_separation: BoolProperty(name="两端半径分离", default=False, update=_collider_update)
    length: FloatProperty(name="长度", default=0.2, min=0.001, soft_max=2.0,
                          update=_collider_update)
    direction: EnumProperty(name="方向", items=CAPSULE_DIRECTION_ITEMS, default='Y',
                            update=_collider_update)
    reverse_direction: BoolProperty(name="方向反转", default=False, update=_collider_update)
    aligned_on_center: BoolProperty(name="中心对齐", default=True, update=_collider_update)


class RCPTetherSettings(bpy.types.PropertyGroup):
    distance_compression: FloatProperty(name="收缩限界", default=0.4, min=0.0, max=1.0,
                                        update=_param_update)


class RCPDistanceSettings(bpy.types.PropertyGroup):
    stiffness: PointerProperty(type=RCPCurveDistanceStiffness)


class RCPTriangleBendingSettings(bpy.types.PropertyGroup):
    stiffness: FloatProperty(name="弯曲刚性", default=1.0, min=0.0, max=1.0,
                             update=_param_update)


class RCPAngleRestorationSettings(bpy.types.PropertyGroup):
    use: BoolProperty(name="启用角度复原", default=True, update=_param_update)
    stiffness: PointerProperty(type=RCPCurveAngleStiffness)
    velocity_attenuation: FloatProperty(name="速度衰减", default=0.8, min=0.0, max=1.0,
                                        update=_param_update)
    gravity_falloff: FloatProperty(name="重力方向衰减", default=0.0, min=0.0, max=1.0,
                                   update=_param_update)


class RCPAngleLimitSettings(bpy.types.PropertyGroup):
    use: BoolProperty(name="启用角度制限", default=False, update=_param_update)
    limit_angle: PointerProperty(type=RCPCurveLimitAngle)
    stiffness: FloatProperty(name="刚性", default=1.0, min=0.0, max=1.0, update=_param_update)


class RCPMotionSettings(bpy.types.PropertyGroup):
    use_max_distance: BoolProperty(name="启用最大移动距离", default=False, update=_param_update)
    max_distance: PointerProperty(type=RCPCurveMaxDistance)
    use_backstop: BoolProperty(name="启用背挡", default=False, update=_param_update)
    backstop_radius: FloatProperty(name="背挡球半径", default=10.0, min=0.1, max=10.0,
                                   update=_param_update)
    backstop_distance: PointerProperty(type=RCPCurveBackstopDistance)
    stiffness: FloatProperty(name="刚性", default=1.0, min=0.0, max=1.0, update=_param_update)


COLLISION_MODE_ITEMS = (
    ('NONE', "无", "不进行碰撞体碰撞"),
    ('POINT', "点", "以粒子球进行碰撞"),
    ('EDGE', "边", "以连接边进行碰撞(更精确, 稍重)"),
)


class RCPColliderCollisionSettings(bpy.types.PropertyGroup):
    mode: EnumProperty(name="模式", items=COLLISION_MODE_ITEMS, default='POINT',
                       update=_param_update)
    friction: FloatProperty(name="摩擦", default=0.05, min=0.0, max=0.5, update=_param_update)
    limit_distance: PointerProperty(type=RCPCurvePushoutDistance)
    collider_references: CollectionProperty(type=RCPColliderReference)
    active_collider_reference_index: IntProperty(default=0)
    collision_bones: CollectionProperty(type=RCPBoneReference)
    active_collision_bone_index: IntProperty(default=0)


SELF_COLLISION_MODE_ITEMS = (
    ('NONE', "无", "关闭"),
    ('FULL_MESH', "完全网格", "点-三角 + 边-边 + 交叉解决"),
)


class RCPSelfCollisionSettings(bpy.types.PropertyGroup):
    self_mode: EnumProperty(name="自碰撞模式", items=SELF_COLLISION_MODE_ITEMS, default='NONE',
                            update=_param_update)
    sync_mode: EnumProperty(name="相互碰撞模式", items=SELF_COLLISION_MODE_ITEMS, default='NONE',
                            update=_param_update)
    sync_partner: StringProperty(name="相互碰撞对象", default="", update=_param_update)
    surface_thickness: PointerProperty(type=RCPCurveSurfaceThickness)
    cloth_mass: FloatProperty(name="布料质量", default=0.0, min=0.0, max=1.0, update=_param_update)


TELEPORT_MODE_ITEMS = (
    ('NONE', "无", "不检测传送"),
    ('RESET', "复位", "检测到传送时完全复位模拟"),
    ('KEEP', "保持", "检测到传送时保持形状平移"),
)


class RCPInertiaSettings(bpy.types.PropertyGroup):
    anchor_object: PointerProperty(name="锚对象", type=bpy.types.Object, update=_param_update)
    anchor_bone_uid: StringProperty(default="", options={'HIDDEN'})
    anchor_bone: _bone_property("锚骨骼", _param_update, lambda self: self.anchor_object,
                                "anchor_bone_uid")
    anchor_inertia: FloatProperty(name="锚惯性", default=0.0, min=0.0, max=1.0,
                                  update=_param_update)
    world_inertia: FloatProperty(name="世界惯性", default=1.0, min=0.0, max=1.0,
                                 update=_param_update)
    movement_inertia_smoothing: FloatProperty(name="世界惯性平滑", default=0.4, min=0.0, max=1.0,
                                              update=_param_update)
    movement_speed_limit: PointerProperty(type=RCPCheckMovementSpeed)
    rotation_speed_limit: PointerProperty(type=RCPCheckRotationSpeed)
    local_inertia: FloatProperty(name="局部惯性", default=1.0, min=0.0, max=1.0,
                                 update=_param_update)
    local_movement_speed_limit: PointerProperty(type=RCPCheckLocalMovementSpeed)
    local_rotation_speed_limit: PointerProperty(type=RCPCheckLocalRotationSpeed)
    depth_inertia: FloatProperty(name="深度惯性", default=0.0, min=0.0, max=1.0,
                                 update=_param_update)
    centrifugal_acceleration: FloatProperty(name="离心加速", default=0.0, min=0.0, max=1.0,
                                            update=_param_update)
    particle_speed_limit: PointerProperty(type=RCPCheckParticleSpeed)
    teleport_mode: EnumProperty(name="传送模式", items=TELEPORT_MODE_ITEMS, default='NONE',
                                update=_param_update)
    teleport_distance: FloatProperty(name="传送判定距离", default=0.5, min=0.0,
                                     update=_param_update)
    teleport_rotation: FloatProperty(name="传送判定角度", default=90.0, min=0.0,
                                     update=_param_update)


class RCPWindSettings(bpy.types.PropertyGroup):
    influence: FloatProperty(name="影响率", default=1.0, min=0.0, max=2.0, update=_param_update)
    frequency: FloatProperty(name="频率", default=1.0, min=0.0, max=2.0, update=_param_update)
    turbulence: FloatProperty(name="乱流", default=1.0, min=0.0, max=2.0, update=_param_update)
    blend: FloatProperty(name="噪声混合", default=0.7, min=0.0, max=1.0, update=_param_update)
    synchronization: FloatProperty(name="同步率", default=0.7, min=0.0, max=1.0,
                                   update=_param_update)
    depth_weight: FloatProperty(name="深度影响", default=0.0, min=0.0, max=1.0,
                                update=_param_update)
    moving_wind: FloatProperty(name="移动风", default=0.0, min=0.0, max=10.0,
                               update=_param_update)


class RCPSpringSettings(bpy.types.PropertyGroup):
    use_spring: BoolProperty(name="启用弹簧", default=True, update=_param_update)
    spring_power: FloatProperty(name="弹簧强度", default=0.04, min=0.001, max=0.2,
                                update=_param_update)
    limit_distance: FloatProperty(name="移动限制距离", default=0.1, min=0.0, max=0.5,
                                  update=_param_update)
    normal_limit_ratio: FloatProperty(name="法线方向限制", default=1.0, min=0.0, max=1.0,
                                      update=_param_update)
    noise: FloatProperty(name="非同步化", default=0.0, min=0.0, max=1.0, update=_param_update)


CAMERA_CULLING_ITEMS = (
    ('OFF', "关闭", "永远模拟"),
    ('RESET', "复位", "对象不可见时停止, 恢复可见时复位"),
    ('KEEP', "保持", "对象不可见时停止, 恢复可见时继续"),
)


class RCPCullingSettings(bpy.types.PropertyGroup):
    camera_culling_mode: EnumProperty(name="视口剔除", items=CAMERA_CULLING_ITEMS, default='OFF',
                                      update=_param_update)
    distance_culling_length: PointerProperty(type=RCPCheckCullingLength)
    distance_culling_fade_ratio: FloatProperty(name="剔除渐隐比", default=0.2, min=0.0, max=1.0,
                                               update=_param_update)
    distance_culling_reference_object: PointerProperty(name="距离参考对象", type=bpy.types.Object,
                                                       update=_param_update)


NORMAL_ALIGNMENT_ITEMS = (
    ('NONE', "无", "使用骨骼自身姿势"),
    ('BOUNDING_BOX_CENTER', "包围盒中心", "法线从包围盒中心向外放射"),
    ('TRANSFORM', "指定对象", "法线从指定对象位置向外放射"),
)


class RCPGizmoSettings(bpy.types.PropertyGroup):
    always: BoolProperty(name="非选中也显示", default=False)
    enable: BoolProperty(name="启用覆盖显示", default=False)
    ztest: BoolProperty(name="深度测试", default=False)
    position: BoolProperty(name="粒子位置", default=True)
    axis: BoolProperty(name="粒子轴", default=False)
    shape: BoolProperty(name="形状(线/三角)", default=False)
    base_line: BoolProperty(name="基线", default=False)
    depth: BoolProperty(name="深度着色", default=False)
    animated_position: BoolProperty(name="动画位置", default=False)
    animated_axis: BoolProperty(name="动画轴", default=False)
    animated_shape: BoolProperty(name="动画形状", default=False)
    inertia_center: BoolProperty(name="惯性中心", default=True)


CLOTH_TYPE_ITEMS = (
    ('BONE_CLOTH', "骨骼布料", "骨骼链/网格布料模拟(裙摆·头发·尾巴)", 'MOD_CLOTH', 0),
    ('BONE_SPRING', "骨骼弹簧", "固定骨骼上的弹簧摆动(胸部等)", 'FORCE_HARMONIC', 1),
)

CONNECTION_MODE_ITEMS = (
    ('LINE', "线", "只按父子关系连成链"),
    ('AUTOMATIC_MESH', "自动网格", "根骨自动按最近邻排序并横向连接成网格"),
    ('SEQUENTIAL_LOOP_MESH', "顺序环状网格", "按根骨列表顺序横向连接并首尾成环"),
    ('SEQUENTIAL_NON_LOOP_MESH', "顺序开放网格", "按根骨列表顺序横向连接(不成环)"),
)

NORMAL_AXIS_ITEMS = (
    ('RIGHT', "X 轴", "骨骼局部 X 轴为法线"),
    ('UP', "Y 轴", "骨骼局部 Y 轴(骨骼方向)为法线"),
    ('FORWARD', "Z 轴", "骨骼局部 Z 轴为法线"),
    ('INVERSE_RIGHT', "-X 轴", "骨骼局部 -X 轴为法线"),
    ('INVERSE_UP', "-Y 轴", "骨骼局部 -Y 轴为法线"),
    ('INVERSE_FORWARD', "-Z 轴", "骨骼局部 -Z 轴为法线"),
)

DISABLE_MODE_ITEMS = (
    ('RESET', "复位", "禁用时恢复动画姿势, 重新启用时复位模拟"),
    ('KEEP', "保持", "禁用时冻结当前姿势, 重新启用时继续"),
)


class RCPClothConfig(bpy.types.PropertyGroup):
    enabled: BoolProperty(name="启用", default=True, update=_enabled_update)
    cloth_type: EnumProperty(name="类型", items=CLOTH_TYPE_ITEMS, default='BONE_CLOTH',
                             update=_topology_update)

    root_bones: CollectionProperty(type=RCPBoneReference)
    active_root_bone_index: IntProperty(default=0, update=_root_bone_index_update)
    attribute_overrides: CollectionProperty(type=RCPAttributeOverride)
    active_attribute_override_index: IntProperty(default=0)

    connection_mode: EnumProperty(name="连接模式", items=CONNECTION_MODE_ITEMS, default='LINE',
                                  update=_topology_update)
    root_rotation: FloatProperty(name="根骨旋转率", default=0.5, min=0.0, max=1.0,
                                 update=_param_update)
    rotational_interpolation: FloatProperty(name="旋转插值率", default=0.5, min=0.0, max=1.0,
                                            update=_param_update)
    animation_pose_ratio: FloatProperty(name="动画姿势比", default=0.0, min=0.0, max=1.0,
                                        update=_param_update)
    blend_weight: FloatProperty(name="混合权重", default=1.0, min=0.0, max=1.0,
                                update=_param_update)
    stablization_time: FloatProperty(name="复位后稳定时间", default=0.1, min=0.0, max=1.0,
                                     update=_param_update)
    time_scale: FloatProperty(name="时间缩放", default=1.0, min=0.0, max=1.0,
                              update=_param_update)
    disable_mode: EnumProperty(name="禁用行为", items=DISABLE_MODE_ITEMS, default='RESET',
                               update=_param_update)
    normal_axis: EnumProperty(name="法线轴", items=NORMAL_AXIS_ITEMS, default='UP',
                              update=_param_update)

    normal_alignment_mode: EnumProperty(name="法线对齐", items=NORMAL_ALIGNMENT_ITEMS,
                                        default='NONE', update=_topology_update)
    normal_alignment_object: PointerProperty(name="对齐参考对象", type=bpy.types.Object,
                                             update=_topology_update)
    normal_alignment_bone_uid: StringProperty(default="", options={'HIDDEN'})
    normal_alignment_bone: _bone_property("对齐参考骨骼", _topology_update,
                                          lambda self: self.normal_alignment_object,
                                          "normal_alignment_bone_uid")

    custom_skinning_enable: BoolProperty(name="启用自定义蒙皮", default=False,
                                         update=_topology_update)
    skinning_bones: CollectionProperty(type=RCPBoneReference)
    active_skinning_bone_index: IntProperty(default=0)

    culling: PointerProperty(type=RCPCullingSettings)

    gravity: FloatProperty(name="重力", default=5.0, min=0.0, max=10.0, update=_param_update)
    gravity_direction: FloatVectorProperty(name="重力方向", size=3, default=(0.0, 0.0, -1.0),
                                           subtype='DIRECTION', update=_param_update)
    gravity_falloff: FloatProperty(name="重力衰减", default=0.0, min=0.0, max=1.0,
                                   update=_param_update)
    damping: PointerProperty(type=RCPCurveDamping)
    radius: PointerProperty(type=RCPCurveRadius)

    tether: PointerProperty(type=RCPTetherSettings)
    distance: PointerProperty(type=RCPDistanceSettings)
    triangle_bending: PointerProperty(type=RCPTriangleBendingSettings)
    angle_restoration: PointerProperty(type=RCPAngleRestorationSettings)
    angle_limit: PointerProperty(type=RCPAngleLimitSettings)
    motion: PointerProperty(type=RCPMotionSettings)
    collider_collision: PointerProperty(type=RCPColliderCollisionSettings)
    self_collision: PointerProperty(type=RCPSelfCollisionSettings)
    inertia: PointerProperty(type=RCPInertiaSettings)
    wind: PointerProperty(type=RCPWindSettings)
    spring: PointerProperty(type=RCPSpringSettings)
    gizmos: PointerProperty(type=RCPGizmoSettings)

    rebuild_pending: BoolProperty(default=True, options={'HIDDEN'})


WIND_ZONE_MODE_ITEMS = (
    ('GLOBAL_DIRECTION', "全局方向风", "无范围, 影响所有布料", 'FORCE_WIND', 0),
    ('SPHERE_DIRECTION', "球形方向风", "球形范围内的方向风", 'SPHERE', 1),
    ('BOX_DIRECTION', "盒形方向风", "盒形范围内的方向风", 'MESH_CUBE', 2),
    ('SPHERE_RADIAL', "球形放射风", "从中心向外放射的风", 'PROP_ON', 3),
)


class RCPWindZoneSettings(bpy.types.PropertyGroup):
    is_wind_zone: BoolProperty(default=False, options={'HIDDEN'})
    enabled: BoolProperty(name="启用", default=True)
    mode: EnumProperty(name="模式", items=WIND_ZONE_MODE_ITEMS, default='GLOBAL_DIRECTION')
    size: FloatVectorProperty(name="盒尺寸", size=3, default=(10.0, 10.0, 10.0), min=0.0,
                              subtype='XYZ')
    radius: FloatProperty(name="半径", default=10.0, min=0.0)
    main: FloatProperty(name="风力", default=5.0, min=0.0, max=30.0)
    turbulence: FloatProperty(name="乱流", default=1.0, min=0.0, max=1.0)
    direction_angle_x: FloatProperty(name="方向角 X", default=0.0, min=-180.0, max=180.0)
    direction_angle_y: FloatProperty(name="方向角 Y", default=0.0, min=-180.0, max=180.0)
    attenuation: PointerProperty(type=RCPCurveWindAttenuation)
    is_addition: BoolProperty(name="加算风", default=False,
                              description="不抢占其他风区, 叠加生效(最多 3 个)")


class RCPObjectSettings(bpy.types.PropertyGroup):
    live: BoolProperty(name="实时模拟", default=False,
                       description="在播放/拖动时间轴时实时模拟布料")
    show_colliders: BoolProperty(
        name="编辑碰撞体", default=False, update=_redraw,
        description="打开后在视口里画出全部碰撞体并显示拖拽控制器, 平时关闭不干扰视图; "
                    "与模拟是否开启无关")
    show_collider_gizmo: BoolProperty(
        name="拖拽控制器", default=True, update=_redraw,
        description="编辑碰撞体时, 为当前碰撞体显示可拖拽的半径/长度/位置控制器")
    sync_list_selection: BoolProperty(
        name="列表联动视口选中", default=True,
        description="在根骨骼列表里选中一行时, 同步选中并激活视口里的那根骨骼")
    gizmo_size: FloatProperty(
        name="控制器大小", default=1.0, min=0.1, max=6.0, update=_redraw,
        description="碰撞体拖拽控制器的抓取块大小倍率, 觉得太小抓不住就调大")
    show_bones: BoolProperty(
        name="显示骨骼", default=False, update=_redraw,
        description="在视口里画出配置驱动的骨骼: 根骨骼与被带动的子骨骼两种颜色; "
                    "与模拟是否开启无关, 关掉模拟也能看")
    bone_display_scope: EnumProperty(
        name="显示范围", update=_redraw,
        items=(('ACTIVE', "当前配置", "只画当前配置的链"),
               ('ALL', "全部配置", "画出全部配置, 非当前配置压暗")),
        default='ACTIVE')
    show_collision_radius: BoolProperty(
        name="碰撞半径", default=False, update=_redraw,
        description="画出每根模拟骨骼实际参与碰撞的体积, 就是它和碰撞体交互用的那个球/扫掠体; "
                    "需要开启实时模拟才有数据")
    bone_display_depth: BoolProperty(
        name="骨骼深度测试", default=False, update=_redraw,
        description="关闭时骨骼画在模型前面, 不会被身体挡住")
    configs: CollectionProperty(type=RCPClothConfig)
    active_config_index: IntProperty(default=0)
    colliders: CollectionProperty(type=RCPColliderItem)
    active_collider_index: IntProperty(default=0, update=_redraw)
    param_serial: IntProperty(default=0, options={'HIDDEN'})
    collider_serial: IntProperty(default=0, options={'HIDDEN'})


class RCPSceneSettings(bpy.types.PropertyGroup):
    simulation_frequency: IntProperty(name="模拟频率", default=90, min=30, max=150,
                                      description="每秒模拟步数")
    max_simulation_count: IntProperty(name="单帧最大步数", default=3, min=1, max=5)
    global_time_scale: FloatProperty(name="全局时间缩放", default=1.0, min=0.0, max=1.0)
    overlay_enabled: BoolProperty(name="视口覆盖显示", default=True,
                                  description="覆盖线框总开关(每配置的细分开关在其 Viewport Display 子面板)")


_CLASSES = tuple(_curve_classes) + tuple(_check_classes) + (
    RCPBoneReference,
    RCPColliderReference,
    RCPAttributeOverride,
    RCPColliderItem,
    RCPTetherSettings,
    RCPDistanceSettings,
    RCPTriangleBendingSettings,
    RCPAngleRestorationSettings,
    RCPAngleLimitSettings,
    RCPMotionSettings,
    RCPColliderCollisionSettings,
    RCPSelfCollisionSettings,
    RCPInertiaSettings,
    RCPWindSettings,
    RCPSpringSettings,
    RCPCullingSettings,
    RCPGizmoSettings,
    RCPClothConfig,
    RCPWindZoneSettings,
    RCPObjectSettings,
    RCPSceneSettings,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.ruri_cloth_physics = PointerProperty(type=RCPObjectSettings)
    bpy.types.Object.ruri_cloth_physics_wind = PointerProperty(type=RCPWindZoneSettings)
    bpy.types.Scene.ruri_cloth_physics = PointerProperty(type=RCPSceneSettings)


def unregister():
    del bpy.types.Scene.ruri_cloth_physics
    del bpy.types.Object.ruri_cloth_physics_wind
    del bpy.types.Object.ruri_cloth_physics
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
