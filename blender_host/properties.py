import math
import re

import bpy
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


def _backend_update(self, context):
    from . import runtime
    runtime.notify_backend_changed()


def _wind_mode_update(self, context):
    from . import wind_geom
    obj = _owner_object(self)
    if obj is not None and obj.type == 'EMPTY':
        wind_geom.sync_display(obj, self)
    _redraw(self, context)


def _collider_shape_update(self, context):
    from . import collider_geom
    obj = _owner_object(self)
    if obj is not None and obj.type == 'EMPTY' and self.is_collider:
        collider_geom.sync_display(obj)
    _redraw(self, context)


_curve_classes = []


_DEPTH_NOTE = ("深度 = 该骨骼在链上的位置比例: 0 = 根部(靠近固定点), 1 = 末端(链尖)。")


def _make_curve_type(type_name, default_value, min_value, max_value,
                     default_start=1.0, default_end=1.0, default_use=False,
                     soft_min=None, soft_max=None,
                     value_label="值", value_desc="", curve_desc=""):
    def _points_get(self):
        return curve_host.serialize_points(self)

    def _points_set(self, text):
        curve_host.deserialize_points(self, text)

    curve_help = "开启后, 每根骨骼的实际值 = 上面那个值 × 曲线(该骨骼深度)。" + _DEPTH_NOTE
    if curve_desc:
        curve_help = curve_help + " " + curve_desc

    annotations = {
        "value": FloatProperty(
            name=value_label, default=default_value,
            description=value_desc,
            min=min_value, max=max_value,
            soft_min=min_value if soft_min is None else soft_min,
            soft_max=max_value if soft_max is None else soft_max,
            update=_param_update),
        "use_curve": BoolProperty(name="按深度曲线调制", default=default_use,
                                  description=curve_help, update=_use_curve_update),
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


RCPCurveDamping = _make_curve_type(
    "RCPCurveDamping", 0.05, 0.0, 1.0,
    value_label="空气阻尼",
    value_desc="每个模拟步按比例扣减速度: 越大越黏、摆动衰减越快, 0 = 不衰减(会一直晃)。"
               "注意实际生效值 = 本值 × 深度曲线 × 0.2, 默认 0.05 实际只有 0.01, "
               "所以这根滑块比看上去温和得多, 想要明显变黏要往大了拉。")
RCPCurveRadius = _make_curve_type(
    "RCPCurveRadius", 0.02, 0.001, 1.0, soft_max=0.5,
    value_label="粒子半径 (米)",
    value_desc="每根模拟骨骼参与碰撞时的球半径, 同时决定碰撞推出量和摩擦的作用范围。"
               "调大 = 布料被碰撞体顶得更远、更不容易穿模, 但整体轮廓也会被撑鼓。"
               "视口勾选'碰撞半径'可以直接看到这个球。")
RCPCurveDistanceStiffness = _make_curve_type(
    "RCPCurveDistanceStiffness", 1.0, 0.0, 1.0, 1.0, 0.5, False,
    value_label="距离刚性",
    value_desc="维持骨骼之间原始间距的强度: 1 = 几乎不可伸缩, 调低则链条像橡皮筋一样能被拉长。"
               "网格连接模式下的横向邻居只按本值的 50% 生效(纵向父子才是全额)。"
               "【骨骼弹簧类型下本项被固定为 0.5, 调了不生效】")
RCPCurveAngleStiffness = _make_curve_type(
    "RCPCurveAngleStiffness", 0.2, 0.0, 1.0, 1.0, 0.2, True,
    value_label="复原力",
    value_desc="把骨骼拉回它相对父骨的初始朝向, 是决定'甩出去以后回弹多快'的主力。"
               "【重要】实际刚性 = 本值 × 深度曲线 × 0.2, 即本值拉满 1.0 内部也只有 0.2, 这是硬上限, "
               "再拖不会有任何变化。而且默认曲线是 根部1.0 → 末端0.2, 末端只有本值的 1/5 —— "
               "嫌末端太软应该抬高曲线右端, 而不是拉这个值。"
               "拉满仍不够硬时, 请改用'角度限制'(硬钳位, 不是弹簧)。",
    curve_desc="默认根部 1.0 → 末端 0.2。")
RCPCurveLimitAngle = _make_curve_type(
    "RCPCurveLimitAngle", 60.0, 0.0, 180.0, 0.0, 1.0, True,
    value_label="最大弯曲角 (度)",
    value_desc="骨骼相对父骨允许偏离初始朝向的最大角度, 超出即被硬性掰回, 是比'复原力'更强硬的手段。"
               "【注意】默认曲线是 根部0 → 末端1, 也就是根部被完全锁死、只有末端能弯到这个角度; "
               "想让根部也能弯, 要抬高曲线左端。",
    curve_desc="默认根部 0 → 末端 1(根部锁死)。")
RCPCurveMaxDistance = _make_curve_type(
    "RCPCurveMaxDistance", 0.3, 0.0, 5.0,
    value_label="最大偏移距离 (米)",
    value_desc="粒子离它自身动画原位的最远距离, 超出即被拉回, 用来防止裙摆/头发飞得太夸张。"
               "始终以动画姿势为基准, 不受'动画姿势比'影响。"
               "【骨骼弹簧类型下整个移动限制被强制关闭】")
RCPCurveBackstopDistance = _make_curve_type(
    "RCPCurveBackstopDistance", 0.0, 0.0, 1.0,
    value_label="背挡距离 (米)",
    value_desc="背挡球球心相对动画原位、沿法线反方向的偏移量。"
               "背挡球是一个挡在布料身后的隐形球, 专门防止布料往身体里陷。"
               "【骨骼弹簧类型下整个移动限制被强制关闭】")
RCPCurvePushoutDistance = _make_curve_type(
    "RCPCurvePushoutDistance", 0.05, 0.0, 1.0,
    value_label="推出距离上限 (米)",
    value_desc="碰撞体最多能把骨骼从它的动画原位推开多远, 超过就不再推, 避免被挤飞。"
               "【仅骨骼弹簧类型生效, 骨骼布料类型下本项不接线】")
RCPCurveSurfaceThickness = _make_curve_type(
    "RCPCurveSurfaceThickness", 0.005, 0.001, 0.05, 0.5, 1.0, False,
    value_label="表面厚度 (米)",
    value_desc="自碰撞时每个点/边/三角的厚度, 两侧厚度相加就是它们之间的最小间隔。"
               "越大越不容易互相穿插, 但布料会显得鼓胀, 且开销上升。")
RCPCurveWindAttenuation = _make_curve_type(
    "RCPCurveWindAttenuation", 1.0, 1.0, 1.0, 1.0, 0.0, True,
    value_label="衰减 (仅曲线可调)",
    value_desc="放射风从中心到边界的衰减。本值被固定为 1, 只有曲线有意义。",
    curve_desc="左端 = 风源中心, 右端 = 范围边界; 默认 1 → 0 表示越远越弱。")


_check_classes = []


def _make_check_type(type_name, default_use, default_value, min_value, max_value,
                     value_label="值", value_desc=""):
    annotations = {
        "use": BoolProperty(name="启用限制", default=default_use,
                            description="关闭 = 完全不限制。" + value_desc,
                            update=_param_update),
        "value": FloatProperty(name=value_label, default=default_value,
                               description=value_desc,
                               min=min_value, max=max_value,
                               update=_param_update),
    }
    cls = type(type_name, (bpy.types.PropertyGroup,), {"__annotations__": annotations})
    _check_classes.append(cls)
    return cls


RCPCheckMovementSpeed = _make_check_type(
    "RCPCheckMovementSpeed", True, 5.0, 0.0, 10.0,
    value_label="世界移动限速 (米/秒)",
    value_desc="角色整体位移超过这个速度时, 超速的那部分不再作为惯性传给布料(相当于把布料整体一起搬过去)。"
               "用来防止瞬移、冲刺、镜头切换把布料甩爆。调小 = 布料更'跟得住'身体, 调大 = 甩动更狂野。")
RCPCheckRotationSpeed = _make_check_type(
    "RCPCheckRotationSpeed", True, 720.0, 0.0, 1440.0,
    value_label="世界旋转限速 (度/秒)",
    value_desc="同上, 针对角色整体旋转。急速转身时超速部分不传给布料, 防止头发被离心力抡飞。")
RCPCheckLocalMovementSpeed = _make_check_type(
    "RCPCheckLocalMovementSpeed", False, 5.0, 0.0, 10.0,
    value_label="局部移动限速 (米/秒)",
    value_desc="比世界限速更靠内层: 限制的是每个模拟步内惯性中心自身的位移速度。"
               "用于压制高速位移引起的高频抖动, 代价是布料对身体动作的跟随会变迟钝。")
RCPCheckLocalRotationSpeed = _make_check_type(
    "RCPCheckLocalRotationSpeed", False, 720.0, 0.0, 1440.0,
    value_label="局部旋转限速 (度/秒)",
    value_desc="同上, 针对每个模拟步内惯性中心自身的旋转速度。")
RCPCheckParticleSpeed = _make_check_type(
    "RCPCheckParticleSpeed", True, 4.0, 0.0, 10.0,
    value_label="粒子限速 (米/秒)",
    value_desc="单根骨骼自身速度的硬上限。适度限制会让头发/裙摆的甩动更柔和收敛; "
               "但限得太狠会让碰撞推出追不上运动, 反而更容易穿模。")
RCPCheckCullingLength = _make_check_type(
    "RCPCheckCullingLength", False, 30.0, 0.0, 100.0,
    value_label="剔除距离 (米)",
    value_desc="布料离参考对象超过这个距离就停止模拟以省性能, 靠近后恢复(会复位)。")


def _bone_property(name, update, target=None, uid_attribute="bone_uid"):
    getter, setter = bone_binding.make_accessors(uid_attribute, update, target)
    return StringProperty(name=name, get=getter, set=setter)


def _focus_bone(obj, name):
    if obj is None or obj.type != 'ARMATURE' or not name:
        return
    from . import chain
    if obj.data.bones.get(name) is None:
        return
    chain.select(obj, [name], active=name)


def _root_bone_index_update(self, context):
    obj = _owner_object(self)
    if obj is None or obj.type != 'ARMATURE':
        return
    settings = obj.ruri_cloth_physics
    if not settings.sync_list_selection:
        return
    from . import selection_sync
    if selection_sync.is_suppressed():
        return
    if 0 <= self.active_root_bone_index < len(self.root_bones):
        _focus_bone(obj, self.root_bones[self.active_root_bone_index].bone)


class RCPBoneReference(bpy.types.PropertyGroup):
    bone_uid: StringProperty(default="", options={'HIDDEN'})
    bone: _bone_property("骨骼", _topology_update)


def _collider_reference_poll(self, obj):
    from . import collider_geom
    return collider_geom.is_collider(obj)


def _collider_reference_index_update(self, context):
    from . import collider_geom
    from . import selection_sync
    obj = _owner_object(self)
    if obj is None or obj.type != 'ARMATURE':
        return
    if not obj.ruri_cloth_physics.sync_list_selection or selection_sync.is_suppressed():
        return
    index = self.active_collider_reference_index
    if not 0 <= index < len(self.collider_references):
        return
    target = self.collider_references[index].object
    view_layer = context.view_layer if context is not None else None
    if not collider_geom.is_collider(target) or view_layer is None:
        return
    if target.name not in view_layer.objects or not target.visible_get():
        return
    with selection_sync.suppressed():
        for other in context.selected_objects:
            other.select_set(False)
        target.select_set(True)


class RCPColliderReference(bpy.types.PropertyGroup):
    object: PointerProperty(
        name="碰撞体", type=bpy.types.Object, poll=_collider_reference_poll,
        description="参与本配置碰撞的碰撞体空物体。下拉框只列出已标记为碰撞体的空物体",
        update=_collider_update)


ATTRIBUTE_OVERRIDE_ITEMS = (
    ('FIXED', "固定", "完全跟随动画, 不参与模拟, 并成为下游骨骼摆动的锚点", 'PINNED', 0),
    ('MOVE', "可动", "参与模拟, 会被重力/风/碰撞带动", 'UNPINNED', 1),
    ('IGNORE', "排除", "该骨骼连同其全部子级一起退出本配置的模拟", 'X', 2),
)


class RCPAttributeOverride(bpy.types.PropertyGroup):
    bone_uid: StringProperty(default="", options={'HIDDEN'})
    bone: _bone_property("骨骼", _topology_update)
    attribute: EnumProperty(
        name="属性", items=ATTRIBUTE_OVERRIDE_ITEMS, default='MOVE',
        description="覆盖这根骨骼的默认属性。默认规则是: 根骨骼列表里的为'固定', 其余子骨为'可动'。",
        update=_topology_update)
    disable_collision: BoolProperty(
        name="不参与碰撞", default=False,
        description="让这根骨骼无视所有碰撞体, 可以自由穿过身体。"
                    "用于修那种'某一根骨被碰撞体顶得很怪'的局部问题。",
        update=_topology_update)
    exclude_motion: BoolProperty(
        name="不受移动限制", default=False,
        description="让这根骨骼跳过'最大偏移距离'和'背挡球'两项限制, 可以自由甩出限制范围。",
        update=_topology_update)


COLLIDER_SHAPE_ITEMS = (
    ('SPHERE', "球", "球形碰撞体: 空物体本身显示为球, 半径 = 显示尺寸 × 缩放", 'SPHERE', 0),
    ('CAPSULE', "胶囊", "胶囊碰撞体: 本空物体是起点圆, 另一个圆形空物体是终点圆; "
                        "两端各自的位置与半径直接决定胶囊形状", 'MESH_CAPSULE', 1),
    ('PLANE', "平面", "无限平面碰撞体: 平面法线 = 空物体的 +Z 轴", 'MESH_PLANE', 2),
)


def _collider_end_poll(self, obj):
    from . import collider_geom
    return (obj.type == 'EMPTY' and obj is not self.id_data
            and not collider_geom.is_collider(obj))


class RCPColliderSettings(bpy.types.PropertyGroup):
    is_collider: BoolProperty(default=False, options={'HIDDEN'}, update=_collider_shape_update)
    enabled: BoolProperty(
        name="启用", default=True,
        description="关闭后本碰撞体对所有配置立即失效, 中途重新启用是安全的: "
                    "求解器会在重新启用的那一帧自动重新快照它的历史位姿, "
                    "不会把它当成'从禁用前的位置一路扫过来'而把布料打飞。"
                    "另外, 当它被缩放到接近 0 时会自动挂起, 恢复正常缩放时同样重新快照。",
        update=_collider_update)
    shape: EnumProperty(
        name="形状", items=COLLIDER_SHAPE_ITEMS, default='SPHERE',
        description="碰撞体的几何形状。球最快, 胶囊适合四肢躯干, 平面用来做地面。"
                    "形状会同步切换空物体的显示方式。",
        update=_collider_shape_update)
    end_object: PointerProperty(
        name="终点圆", type=bpy.types.Object, poll=_collider_end_poll,
        description="胶囊另一端的圆形空物体: 它的位置 = 终点球心, 它的显示尺寸 × 缩放 = 终点半径。"
                    "两端可以分别父级到不同骨骼, 胶囊会跟着一起拉伸。"
                    "留空 = 胶囊退化成起点处的一个球。",
        update=_collider_shape_update)


class RCPTetherSettings(bpy.types.PropertyGroup):
    distance_compression: FloatProperty(
        name="允许收缩量", default=0.4, min=0.0, max=1.0,
        description="链条相对固定点最多能被压缩多少: 0 = 一点都不许缩(始终绷直, 像根绳子被拉紧), "
                    "0.4 = 允许缩到原长的 60%, 1.0 = 随便缩(可以完全堆叠折起)。"
                    "裙摆/头发下垂时想避免'穿过自己堆在一起'就调小。"
                    "拉伸方向是内部固定的(最多拉长 3%), 不开放调节。"
                    "【骨骼弹簧类型下本项被固定为 0.8, 调了不生效】",
        update=_param_update)


class RCPDistanceSettings(bpy.types.PropertyGroup):
    stiffness: PointerProperty(type=RCPCurveDistanceStiffness)


class RCPTriangleBendingSettings(bpy.types.PropertyGroup):
    stiffness: FloatProperty(
        name="抗弯折刚性", default=1.0, min=0.0, max=1.0,
        description="维持相邻三角面之间原始夹角的强度, 决定布料是'纸板'还是'丝绸': "
                    "调高 = 不易起皱、整片保持原形, 调低 = 更软更容易折。"
                    "【只有存在三角面时才有意义】连接模式选'线'时整条链没有三角面, "
                    "本项完全不接线; 要用它必须切到某种网格连接模式。"
                    "【骨骼弹簧类型下本项被固定为 0, 调了不生效】",
        update=_param_update)


class RCPAngleRestorationSettings(bpy.types.PropertyGroup):
    use: BoolProperty(
        name="启用角度复原", default=True,
        description="用弹簧的方式把骨骼拉回相对父骨的初始朝向。这是让头发/裙摆'甩出去还会回来'的主力, "
                    "关掉后链条只剩重力和距离约束, 会一直瘫着。",
        update=_param_update)
    stiffness: PointerProperty(type=RCPCurveAngleStiffness)
    velocity_attenuation: FloatProperty(
        name="复原速度吸收", default=0.8, min=0.0, max=1.0,
        description="复原动作产生的位移有多少不计入速度: "
                    "1 = 完全吸收, 复原只挪位置不加速, 收敛最稳、不来回振; "
                    "0 = 完全不吸收, 复原会转成速度, 弹性更强但容易反复过冲抖动。"
                    "觉得回弹后'还在抖'就往大调。",
        update=_param_update)
    gravity_falloff: FloatProperty(
        name="逆重力时削弱", default=0.0, min=0.0, max=1.0,
        description="当整条链被翻到与重力相反的朝向时(角色倒立、躺下、大幅仰身), "
                    "把复原力削弱多少, 让头发顺着重力自然垂落而不是硬撑回原姿势。"
                    "0 = 任何朝向复原力都一样; 1 = 完全翻转时复原力削到 0。"
                    "注意这条只看朝向, 即使重力设为 0 也照样生效。"
                    "【骨骼弹簧类型下本项被固定为 0, 调了不生效】",
        update=_param_update)


class RCPAngleLimitSettings(bpy.types.PropertyGroup):
    use: BoolProperty(
        name="启用角度限制", default=False,
        description="硬性钳制骨骼相对父骨的弯曲角度, 超出直接掰回 —— 和'角度复原'的弹簧不同, "
                    "这是刚性上限。当复原力已经拉满(内部上限 0.2)仍嫌不够硬时, 用这个。",
        update=_param_update)
    limit_angle: PointerProperty(type=RCPCurveLimitAngle)
    stiffness: FloatProperty(
        name="钳位强度", default=1.0, min=0.0, max=1.0,
        description="超出限制角时掰回去的力度: 1 = 一步到位压回限制角内(最硬), "
                    "调低 = 每步只掰回一部分, 观感更软, 但允许暂时超出限制角。",
        update=_param_update)


class RCPMotionSettings(bpy.types.PropertyGroup):
    use_max_distance: BoolProperty(
        name="启用最大偏移距离", default=False,
        description="限制每根骨骼离它自身动画原位的最远距离, 防止甩得太夸张。"
                    "【骨骼弹簧类型下强制关闭】",
        update=_param_update)
    max_distance: PointerProperty(type=RCPCurveMaxDistance)
    use_backstop: BoolProperty(
        name="启用背挡球", default=False,
        description="在每根骨骼身后放一个隐形球挡住它, 专治布料往身体里陷。"
                    "【骨骼弹簧类型下强制关闭】",
        update=_param_update)
    backstop_radius: FloatProperty(
        name="背挡球半径", default=10.0, min=0.1, max=10.0,
        description="隐形挡球的半径: 调大越接近一块平板(挡得平整), 调小越像个球顶在背后(局部凸起)。",
        update=_param_update)
    backstop_distance: PointerProperty(type=RCPCurveBackstopDistance)
    stiffness: FloatProperty(
        name="限制强度", default=1.0, min=0.0, max=1.0,
        description="上面两项限制的执行力度: 1 = 严格执行, 调低 = 只执行一部分, 允许溢出一些, 观感更软。",
        update=_param_update)


COLLISION_MODE_ITEMS = (
    ('NONE', "无", "完全不与碰撞体交互"),
    ('POINT', "逐点", "只把每根骨骼当成一个球去碰。快, 但骨骼间距大时中间那段会漏进碰撞体"),
    ('EDGE', "逐边", "把骨骼之间的连线当成扫掠体去碰。骨骼稀疏时不会漏穿, 代价是更重"),
)


class RCPColliderCollisionSettings(bpy.types.PropertyGroup):
    mode: EnumProperty(
        name="碰撞方式", items=COLLISION_MODE_ITEMS, default='POINT',
        description="骨骼用什么形状去和碰撞体求交。骨骼数量少/间距大又要贴身时选'逐边'。"
                    "【骨骼弹簧类型下强制为逐点, 选了逐边也不生效】",
        update=_param_update)
    friction: FloatProperty(
        name="摩擦力", default=0.05, min=0.0, max=0.5,
        description="布料贴着碰撞体滑动时的阻力: 调大 = 更容易'挂'在身上跟着走, 调小 = 更顺滑地溜下去。"
                    "内部同时驱动动摩擦(按接触角度削减切向速度)和静摩擦(低速时把布料粘住不滑)。"
                    "【骨骼弹簧类型下被固定为 0.5, 调了不生效】",
        update=_param_update)
    limit_distance: PointerProperty(type=RCPCurvePushoutDistance)
    collider_references: CollectionProperty(type=RCPColliderReference)
    active_collider_reference_index: IntProperty(default=0,
                                                 update=_collider_reference_index_update)
    collision_bones: CollectionProperty(type=RCPBoneReference)
    active_collision_bone_index: IntProperty(default=0)


SELF_COLLISION_MODE_ITEMS = (
    ('NONE', "关闭", "不做自碰撞(默认, 开销最低)"),
    ('FULL_MESH', "完全网格", "点-三角 + 边-边 + 穿插解除三件套全开; 效果最好, 也是整套里最贵的功能"),
)


class RCPSelfCollisionSettings(bpy.types.PropertyGroup):
    self_mode: EnumProperty(
        name="自身穿插", items=SELF_COLLISION_MODE_ITEMS, default='NONE',
        description="让这一条布料自己不穿过自己(裙子前后片、双马尾互相穿)。"
                    "【这是整套里最耗时的功能, 开之前先确认真的需要】"
                    "【骨骼弹簧类型下强制关闭】",
        update=_param_update)
    sync_mode: EnumProperty(
        name="与另一条互穿", items=SELF_COLLISION_MODE_ITEMS, default='NONE',
        description="让本配置和下面指定的另一条布料互相不穿透(例如外裙 vs 内衬)。"
                    "【骨骼弹簧类型下强制关闭】",
        update=_param_update)
    sync_partner: StringProperty(
        name="互穿对象", default="",
        description="要与之互相碰撞的另一条配置名。两边最好都指向对方以获得对称结果。",
        update=_param_update)
    surface_thickness: PointerProperty(type=RCPCurveSurfaceThickness)
    cloth_mass: FloatProperty(
        name="布料重量", default=0.0, min=0.0, max=1.0,
        description="自碰撞里这条布料的相对重量: 调大 = 更'沉', 被别的布料推开时更纹丝不动, "
                    "反过来会把对方顶开。只在自碰撞/互穿开启时有意义。",
        update=_param_update)


TELEPORT_MODE_ITEMS = (
    ('NONE', "不检测", "不做瞬移检测; 一帧跨越大距离时布料会被狠狠甩一下"),
    ('RESET', "复位", "检测到瞬移就把布料完全复位成动画姿势(最干净, 但会丢掉当前摆动)"),
    ('KEEP', "保持", "检测到瞬移就带着当前形状整体搬过去(保留摆动姿态)"),
)


class RCPInertiaSettings(bpy.types.PropertyGroup):
    anchor_object: PointerProperty(
        name="锚对象", type=bpy.types.Object,
        description="把这个对象的运动从布料受到的惯性里扣掉。典型用途: 角色站在移动的载具/平台上, "
                    "指定载具为锚, 头发就不会因为载具行进而一直向后飘。留空 = 不使用。",
        update=_param_update)
    anchor_bone_uid: StringProperty(default="", options={'HIDDEN'})
    anchor_bone: _bone_property("锚骨骼", _param_update, lambda self: self.anchor_object,
                                "anchor_bone_uid")
    anchor_inertia: FloatProperty(
        name="锚运动保留", default=0.0, min=0.0, max=1.0,
        description="锚的运动有多少仍然传给布料: 0 = 完全扣掉(坐车时头发像车没动一样, 最常用), "
                    "1 = 完全不扣(等于没设锚)。只在指定了锚对象时有效。",
        update=_param_update)
    world_inertia: FloatProperty(
        name="世界惯性", default=1.0, min=0.0, max=1.0,
        description="整体位移/旋转带来的甩动有多强: 1 = 全额甩动(角色一跑头发就往后飘), "
                    "0 = 完全跟随身体走、几乎不甩。想让布料'贴身安静'就调小。",
        update=_param_update)
    movement_inertia_smoothing: FloatProperty(
        name="世界惯性平滑", default=0.4, min=0.0, max=1.0,
        description="对上面那股惯性做时间平滑, 缓和急起步/急刹车/抽帧带来的突然一甩。"
                    "调大 = 反应更迟钝但更稳(不易抽搐), 0 = 立即响应每一帧的速度变化。",
        update=_param_update)
    movement_speed_limit: PointerProperty(type=RCPCheckMovementSpeed)
    rotation_speed_limit: PointerProperty(type=RCPCheckRotationSpeed)
    local_inertia: FloatProperty(
        name="局部惯性", default=1.0, min=0.0, max=1.0,
        description="每个模拟步内部的甩动强度, 作用在比'世界惯性'更细的时间尺度上: "
                    "1 = 全额, 0 = 每步都把布料整体搬到位、不产生相对运动。"
                    "世界惯性管'整体飘不飘', 这个管'步与步之间抖不抖'。",
        update=_param_update)
    local_movement_speed_limit: PointerProperty(type=RCPCheckLocalMovementSpeed)
    local_rotation_speed_limit: PointerProperty(type=RCPCheckLocalRotationSpeed)
    depth_inertia: FloatProperty(
        name="根部抑制", default=0.0, min=0.0, max=1.0,
        description="按深度削弱惯性, 越靠近根部削得越狠(末端几乎不受影响)。"
                    "调大 = 发根/裙腰稳稳跟着身体, 只有发梢裙摆在飘, 是做'服帖但有动感'的关键旋钮。"
                    "0 = 整条链一视同仁。",
        update=_param_update)
    centrifugal_acceleration: FloatProperty(
        name="离心力增强", default=0.0, min=0.0, max=1.0,
        description="角色旋转时额外给一份向外的离心力, 让转身时头发/裙摆更明显地张开。"
                    "只在确实存在旋转、且骨骼运动方向与旋转方向一致时才生效, 0 = 不额外加。",
        update=_param_update)
    particle_speed_limit: PointerProperty(type=RCPCheckParticleSpeed)
    teleport_mode: EnumProperty(
        name="瞬移处理", items=TELEPORT_MODE_ITEMS, default='NONE',
        description="一帧之内位移或旋转超过下面阈值时怎么办。切镜头、换场景、角色被拉走时用得上。",
        update=_param_update)
    teleport_distance: FloatProperty(
        name="瞬移判定距离 (米)", default=0.5, min=0.0,
        description="单帧位移超过这个距离即判定为瞬移。设太小会把正常疾跑误判成瞬移。",
        update=_param_update)
    teleport_rotation: FloatProperty(
        name="瞬移判定角度 (度)", default=90.0, min=0.0,
        description="单帧旋转超过这个角度即判定为瞬移。设太小会把正常急转身误判成瞬移。",
        update=_param_update)


class RCPWindSettings(bpy.types.PropertyGroup):
    influence: FloatProperty(
        name="受风程度", default=1.0, min=0.0, max=2.0,
        description="这条布料对场景里风区的总响应倍率, 0 = 完全不吃风。"
                    "【前提是场景里存在启用的风区对象, 否则调它没有任何效果】",
        update=_param_update)
    frequency: FloatProperty(
        name="摆动快慢", default=1.0, min=0.0, max=2.0,
        description="风致摆动的周期倍率: 调大 = 抖得更急促, 调小 = 大开大合地缓慢摆。",
        update=_param_update)
    turbulence: FloatProperty(
        name="乱流倍率", default=1.0, min=0.0, max=2.0,
        description="风向抖动的强度倍率, 会和风区自身的乱流值相乘。"
                    "调大 = 风向乱窜、飘得不规则; 0 = 风向恒定, 布料被稳稳吹向一侧。",
        update=_param_update)
    blend: FloatProperty(
        name="随机感", default=0.7, min=0.0, max=1.0,
        description="摆动波形在'正弦'和'噪声'之间的混合: 0 = 纯正弦, 规律得像节拍器; "
                    "1 = 纯噪声, 完全不规则。默认 0.7 偏自然。",
        update=_param_update)
    synchronization: FloatProperty(
        name="各链同步率", default=0.7, min=0.0, max=1.0,
        description="同一配置下不同链条(不同根骨)之间的摆动相位是否一致: "
                    "1 = 全部整齐划一地一起摆(像一整片旗), 0 = 每条链各摆各的(像一把散落的发丝)。",
        update=_param_update)
    depth_weight: FloatProperty(
        name="只吹末端", default=0.0, min=0.0, max=1.0,
        description="按深度削弱风力, 越靠近根部削得越狠: 调大 = 发根不动、只有发梢被风吹起。"
                    "0 = 整条链均匀受风。",
        update=_param_update)
    moving_wind: FloatProperty(
        name="移动迎面风", default=0.0, min=0.0, max=10.0,
        description="把角色自身的移动速度换算成一股迎面吹来的风, 不需要场景里有风区也能生效。"
                    "跑步时头发向后飘就靠它。0 = 关闭。",
        update=_param_update)


class RCPSpringSettings(bpy.types.PropertyGroup):
    use_spring: BoolProperty(
        name="启用弹簧", default=True,
        description="让被标记为固定的骨骼自己在原位附近做弹性摆动(胸部、软组织)。"
                    "【整个弹簧面板仅在类型 = 骨骼弹簧 时接线; 骨骼布料类型下这里调什么都没用】",
        update=_param_update)
    spring_power: FloatProperty(
        name="回位力", default=0.04, min=0.001, max=0.2,
        description="每个模拟步把骨骼从当前偏移拉回原位的比例 —— 注意它不是'摆得多大'的旋钮。"
                    "调大 = 回位更快、停下来更利索、残留的余震明显减少, 但整体变硬, "
                    "受到冲击的瞬间反而更容易过冲, 峰值会略微变大; "
                    "调到最小 ≈ 几乎不回位, 会一直晃个不停、残留最大。"
                    "想改'摆动幅度'请用下面的最大摆幅。"
                    "【仅骨骼弹簧类型生效】",
        update=_param_update)
    limit_distance: FloatProperty(
        name="最大摆幅 (米)", default=0.1, min=0.0, max=0.5,
        description="骨骼能离开原位的最远距离, 这才是控制'摆动幅度'的主旋钮。"
                    "0 = 完全锁死不动。【仅骨骼弹簧类型生效】",
        update=_param_update)
    normal_limit_ratio: FloatProperty(
        name="法线方向压扁", default=1.0, min=0.0, max=1.0,
        description="把上面那个球形摆动范围沿法线轴压扁成椭球: "
                    "1 = 正球, 各方向摆幅相同; 调小 = 沿法线方向摆得更少, 摆动更多发生在横向。"
                    "法线轴由主面板的'法线轴'决定。【仅骨骼弹簧类型生效】",
        update=_param_update)
    noise: FloatProperty(
        name="相位错开", default=0.0, min=0.0, max=1.0,
        description="给每根弹簧骨骼的回位力叠加不同相位的扰动, 避免所有骨骼整齐划一地同步弹动。"
                    "调大 = 各自为政更自然, 0 = 完全同步。【仅骨骼弹簧类型生效】",
        update=_param_update)


CAMERA_CULLING_ITEMS = (
    ('OFF', "不剔除", "永远模拟(烘焙时应保持这个)"),
    ('RESET', "复位", "对象不可见时停止模拟, 重新可见时复位成动画姿势"),
    ('KEEP', "保持", "对象不可见时停止模拟, 重新可见时接着原姿势继续"),
)


class RCPCullingSettings(bpy.types.PropertyGroup):
    camera_culling_mode: EnumProperty(
        name="不可见时剔除", items=CAMERA_CULLING_ITEMS, default='OFF',
        description="对象在视口里不可见时是否停止模拟以省性能。"
                    "【烘焙请保持'不剔除': 一旦中途剔除, 烘出来的曲线会出现断裂或复位跳变】",
        update=_param_update)
    distance_culling_length: PointerProperty(type=RCPCheckCullingLength)
    distance_culling_fade_ratio: FloatProperty(
        name="剔除渐隐区间", default=0.2, min=0.0, max=1.0,
        description="在到达剔除距离之前的多大一段范围内, 逐渐把模拟结果淡回动画姿势, "
                    "避免走到临界距离时'啪'地一下弹回去。占剔除距离的比例, 0 = 不渐隐, 硬切。",
        update=_param_update)
    distance_culling_reference_object: PointerProperty(
        name="距离参考对象", type=bpy.types.Object,
        description="拿谁来量距离(通常是摄像机)。留空 = 使用场景当前摄像机。",
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
    ('BONE_CLOTH', "骨骼布料",
     "链状/网格状骨骼的布料模拟(裙摆·头发·尾巴·飘带)。"
     "本类型下弹簧面板整块不接线", 'MOD_CLOTH', 0),
    ('BONE_SPRING', "骨骼弹簧",
     "固定骨骼在原位附近的弹性摆动(胸部·软组织)。"
     "本类型下重力/距离刚性/摩擦/允许收缩量/移动限制/自碰撞/抗弯折 全部被内部常量覆写, 调了不生效",
     'FORCE_HARMONIC', 1),
)

CONNECTION_MODE_ITEMS = (
    ('LINE', "线", "只按父子关系连成一条条独立的链。没有三角面, 因此抗弯折与自碰撞都不会生效"),
    ('AUTOMATIC_MESH', "自动网格", "自动按最近邻给根骨排序并横向织成网格; 根骨顺序乱也能用"),
    ('SEQUENTIAL_LOOP_MESH', "顺序环状网格", "严格按根骨列表顺序横向连接, 并首尾相接成筒(整圈裙子)"),
    ('SEQUENTIAL_NON_LOOP_MESH', "顺序开放网格", "严格按根骨列表顺序横向连接, 不闭合(开衩裙·披风)"),
)

NORMAL_AXIS_ITEMS = (
    ('RIGHT', "X 轴", "骨骼局部 X 轴为法线"),
    ('UP', "Y 轴", "骨骼局部 Y 轴(骨骼指向)为法线"),
    ('FORWARD', "Z 轴", "骨骼局部 Z 轴为法线"),
    ('INVERSE_RIGHT', "-X 轴", "骨骼局部 -X 轴为法线"),
    ('INVERSE_UP', "-Y 轴", "骨骼局部 -Y 轴(骨骼反向)为法线"),
    ('INVERSE_FORWARD', "-Z 轴", "骨骼局部 -Z 轴为法线"),
)

DISABLE_MODE_ITEMS = (
    ('RESET', "复位", "禁用时恢复动画姿势, 重新启用时从动画姿势重新开始"),
    ('KEEP', "保持", "禁用时冻结当前姿势, 重新启用时接着这个姿势继续"),
)


class RCPClothConfig(bpy.types.PropertyGroup):
    enabled: BoolProperty(name="启用", default=True, update=_enabled_update)
    cloth_type: EnumProperty(name="类型", items=CLOTH_TYPE_ITEMS, default='BONE_CLOTH',
                             update=_topology_update)

    root_bones: CollectionProperty(type=RCPBoneReference)
    active_root_bone_index: IntProperty(default=0, update=_root_bone_index_update)
    attribute_overrides: CollectionProperty(type=RCPAttributeOverride)
    active_attribute_override_index: IntProperty(default=0)

    connection_mode: EnumProperty(
        name="连接方式", items=CONNECTION_MODE_ITEMS, default='LINE',
        description="根骨之间要不要横向连起来。选'线'时每条链彼此独立(头发一缕缕各飘各的); "
                    "选任意网格模式才会生成三角面, 让整片布料互相牵制。"
                    "【抗弯折刚性和自碰撞都依赖三角面, 在'线'模式下形同虚设】",
        update=_topology_update)
    root_rotation: FloatProperty(
        name="根骨朝向跟随", default=0.5, min=0.0, max=1.0,
        description="固定的根骨自身朝向, 有多大比例转去对齐它子骨的实际方向: "
                    "0 = 根骨严格保持动画朝向(发根不动), 1 = 完全跟着子骨甩。"
                    "【只影响不含三角面的链; 有三角面的部分朝向由三角面决定, 本项无效】",
        update=_param_update)
    rotational_interpolation: FloatProperty(
        name="中段朝向跟随", default=0.5, min=0.0, max=1.0,
        description="同上, 但作用于链条中间的可动骨: 数值越大, 骨骼朝向越贴合子骨的实际走向, "
                    "链条看起来越顺滑; 越小则各骨更保留自身姿态, 转折更硬。"
                    "【只影响不含三角面的链】",
        update=_param_update)
    animation_pose_ratio: FloatProperty(
        name="跟随动画姿势", default=0.0, min=0.0, max=1.0,
        description="约束求解时以哪个姿势为'原形': 0 = 以绑定时的初始姿势为准(K 了新 pose 布料会想回到原始形状), "
                    "1 = 以当前动画姿势为准(布料只对动画的变化量作出反应, 骨骼动画摆到哪就以哪为原形)。"
                    "给骨骼手 K 了姿势动画又希望布料尊重它, 就往大调。",
        update=_param_update)
    blend_weight: FloatProperty(
        name="模拟混合量", default=1.0, min=0.0, max=1.0,
        description="最终结果里模拟占多少: 1 = 完全用模拟结果, 0 = 完全用原动画(相当于关闭效果)。"
                    "可以 K 关键帧做效果的淡入淡出。",
        update=_param_update)
    stablization_time: FloatProperty(
        name="复位后缓冲时间 (秒)", default=0.1, min=0.0, max=1.0,
        description="每次复位(开始播放、瞬移复位、剔除恢复)之后, 用多长时间把模拟强度从 0 平滑升到满。"
                    "防止第一帧因为姿势突变而炸开。调大 = 起步更温柔, 0 = 立刻满强度。",
        update=_param_update)
    time_scale: FloatProperty(
        name="时间流速", default=1.0, min=0.0, max=1.0,
        description="这条布料自己的时间倍率: 1 = 正常, 调小 = 像在慢动作/水里(整体变慢变黏), 0 = 冻结。",
        update=_param_update)
    disable_mode: EnumProperty(
        name="禁用时行为", items=DISABLE_MODE_ITEMS, default='RESET',
        description="这条配置被关掉时怎么处理已有的模拟状态。",
        update=_param_update)
    normal_axis: EnumProperty(
        name="法线轴", items=NORMAL_AXIS_ITEMS, default='UP',
        description="把骨骼的哪根局部轴当作'布料表面朝外'的方向。"
                    "它决定了背挡球往哪边挡、弹簧的压扁方向往哪压。"
                    "选错的典型症状: 背挡球挡在了布料正面, 或弹簧压扁的方向和预期垂直。",
        update=_param_update)

    normal_alignment_mode: EnumProperty(
        name="法线重定向", items=NORMAL_ALIGNMENT_ITEMS, default='NONE',
        description="强制把所有骨骼的法线改成从某个中心向外放射, 而不是用骨骼自身姿态推出来的法线。"
                    "适合筒状裙子这类'法线应该一致朝外'的形状; 改法线会连带影响背挡方向与弹簧压扁方向。",
        update=_topology_update)
    normal_alignment_object: PointerProperty(
        name="对齐参考对象", type=bpy.types.Object,
        description="法线重定向选'指定对象'时, 以这个对象的位置作为放射中心。",
        update=_topology_update)
    normal_alignment_bone_uid: StringProperty(default="", options={'HIDDEN'})
    normal_alignment_bone: _bone_property("对齐参考骨骼", _topology_update,
                                          lambda self: self.normal_alignment_object,
                                          "normal_alignment_bone_uid")

    custom_skinning_enable: BoolProperty(
        name="启用自定义蒙皮", default=False,
        description="改用下面指定的骨骼来驱动布料的基础姿势, 而不是让每根模拟骨骼只跟随自己。"
                    "适合让裙摆跟着大腿走而不是只跟着腰。"
                    "【骨骼弹簧类型下不生效】",
        update=_topology_update)
    skinning_bones: CollectionProperty(type=RCPBoneReference)
    active_skinning_bone_index: IntProperty(default=0)

    culling: PointerProperty(type=RCPCullingSettings)

    gravity: FloatProperty(
        name="重力强度", default=5.0, min=0.0, max=10.0,
        description="沿下方重力方向施加的加速度大小。注意这是艺术化数值, 不是 9.8 那种物理单位: "
                    "调大 = 下坠感更沉、摆动更靠向重力方向。"
                    "【骨骼弹簧类型下被固定为 0, 整个重力组不生效】",
        update=_param_update)
    gravity_direction: FloatVectorProperty(
        name="重力方向", size=3, default=(0.0, 0.0, -1.0), subtype='DIRECTION',
        description="世界空间下的重力朝向, 默认 -Z(向下)。改成水平方向可以做'一直被风压向一侧'的效果。",
        update=_param_update)
    gravity_falloff: FloatProperty(
        name="垂下时减重力", default=0.0, min=0.0, max=1.0,
        description="当布料已经垂到接近它的初始朝向时, 把重力削弱多少, 防止长发/长裙被重力越拉越长、"
                    "永远回不到造型。1 = 完全垂下时重力归零(最能保持造型), 0 = 任何姿态都是全额重力。"
                    "被举起/翻转时始终是全额重力, 不受本项影响。",
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
    is_wind_zone: BoolProperty(default=False, options={'HIDDEN'}, update=_redraw)
    enabled: BoolProperty(name="启用", default=True, update=_redraw,
                          description="关闭后本风区不再影响任何布料。")
    mode: EnumProperty(name="模式", items=WIND_ZONE_MODE_ITEMS, default='GLOBAL_DIRECTION',
                       update=_wind_mode_update,
                       description="风区的作用范围与吹法。范围风只影响落在范围内的布料。"
                                   "范围大小直接由空物体的显示尺寸与缩放决定, 拖物体即可调。")
    main: FloatProperty(name="风力强度", default=5.0, min=0.0, max=30.0, soft_max=10.0,
                        update=_redraw,
                        description="风推动布料的力度。注意这是艺术化数值, 不是米/秒那种物理单位: "
                                    "实测超过 10 之后响应迅速趋平(位置约束把多余的力投影掉了), "
                                    "20 与 30 只差 8%, 50 以上完全没有区别 —— 滑块因此只拖到 10, "
                                    "更大的值仍可手动输入。"
                                    "【另: 力度越高, 乱流造成的忽强忽弱越不明显, "
                                    "低力度才会呈现明显的一阵一阵】")
    turbulence: FloatProperty(name="乱流", default=1.0, min=0.0, max=1.0, update=_redraw,
                              description="本风区的风向抖动强度, 会与每条布料自己的'乱流倍率'相乘。"
                                          "0 = 风向恒定不变。")
    direction_angle_x: FloatProperty(
        name="风向俯仰", default=0.0, min=-math.pi, max=math.pi, subtype='ANGLE',
        update=_redraw,
        description="在空物体 +Z 的基础上再把风向往下/往上掰。"
                    "范围风靠这两项把'风往哪吹'和'盒子朝哪摆'拆开: "
                    "想让轴对齐的盒子里刮斜风, 调这里而不是转物体。")
    direction_angle_y: FloatProperty(
        name="风向偏航", default=0.0, min=-math.pi, max=math.pi, subtype='ANGLE',
        update=_redraw,
        description="在空物体 +Z 的基础上再把风向左右转。视口箭头会同步显示最终风向。")
    attenuation: PointerProperty(type=RCPCurveWindAttenuation)
    is_addition: BoolProperty(
        name="叠加风", default=False, update=_redraw,
        description="默认情况下多个范围风互相抢占, 只有体积最小的那个生效(小范围优先); "
                    "勾选本项后它改为叠加, 不参与抢占。最多同时叠加 3 个, 超出的被忽略。")


def _collider_layer(self):
    from . import collider_geom
    obj = _owner_object(self)
    scene = bpy.context.scene
    if obj is None or scene is None:
        return None
    collection = collider_geom.collection_for(scene, obj)
    return collider_geom.layer_collection_for(bpy.context.view_layer, collection)


def _show_colliders_get(self):
    layer = _collider_layer(self)
    return True if layer is None else not layer.hide_viewport


def _show_colliders_set(self, value):
    layer = _collider_layer(self)
    if layer is not None:
        layer.hide_viewport = not value


class RCPObjectSettings(bpy.types.PropertyGroup):
    live: BoolProperty(name="实时模拟", default=False,
                       description="在播放/拖动时间轴时实时模拟布料")
    show_colliders: BoolProperty(
        name="显示碰撞体", get=_show_colliders_get, set=_show_colliders_set,
        description="开关本骨架碰撞体集合的显示。这只是视图里的眼睛, 碰撞照常参与模拟; "
                    "要让某个碰撞体不参与请用它自己的启用开关")
    sync_list_selection: BoolProperty(
        name="列表联动视口选中", default=True,
        description="在根骨骼列表里选中一行时同步选中视口里的那根骨骼; "
                    "在碰撞体列表里选中一行时同步选中对应的碰撞体空物体, 可以直接 G/R/S 摆它")
    gizmo_size: FloatProperty(
        name="控制器大小", default=1.0, min=0.1, max=6.0, update=_redraw,
        description="粒子半径拖拽控制器的抓取块大小倍率, 觉得太小抓不住就调大")
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
    param_serial: IntProperty(default=0, options={'HIDDEN'})


BACKEND_ITEMS = (
    ('GPU', "GPU (numba.cuda)", "生产路径: 常驻显存的协作式巨核, 整帧一次启动"),
    ('CPU', "CPU (numpy 参考)", "参考实现: 同一套算法的 numpy 版, 用来对拍 GPU 结果"),
)


class RCPSceneSettings(bpy.types.PropertyGroup):
    backend: EnumProperty(
        name="求解后端", items=BACKEND_ITEMS, default='GPU',
        description="跑物理的引擎。默认 GPU; CPU 是同一套算法的 numpy 参考实现, 只在需要交叉"
                    "验证 GPU 结果时切过去, 会慢一到两个数量级。"
                    "【切换是安全的: GPU 常驻显存里的模拟状态会先回读回场景再交接, 不会跳变】",
        update=_backend_update)
    simulation_frequency: IntProperty(
        name="模拟频率 (步/秒)", default=90, min=30, max=150,
        description="每秒执行多少次物理步。调高 = 更精确、更不容易穿模和抖动, 但线性变慢。"
                    "【改这个会连带改变布料手感: 内部会按频率补偿各项刚性, 但补偿不是完全等效的, "
                    "调完频率通常需要重新微调刚性类参数】低于 60 要特别当心精度问题。")
    max_simulation_count: IntProperty(
        name="单帧最大步数", default=3, min=1, max=5,
        description="一帧之内最多跑几步物理。帧率低时需要的步数会变多, 这个上限用来防止卡顿雪崩; "
                    "被截断的步数会通过整体平移补偿, 不会导致布料落后于身体。")
    global_time_scale: FloatProperty(
        name="全局时间流速", default=1.0, min=0.0, max=1.0,
        description="所有布料的统一时间倍率, 会与每条配置自己的'时间流速'相乘。0 = 全部冻结。")
    overlay_enabled: BoolProperty(name="视口覆盖显示", default=True,
                                  description="覆盖线框总开关(每配置的细分开关在其 Viewport Display 子面板)")


_CLASSES = tuple(_curve_classes) + tuple(_check_classes) + (
    RCPBoneReference,
    RCPColliderReference,
    RCPAttributeOverride,
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
    RCPColliderSettings,
    RCPObjectSettings,
    RCPSceneSettings,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Object.ruri_cloth_physics = PointerProperty(type=RCPObjectSettings)
    bpy.types.Object.ruri_cloth_physics_wind = PointerProperty(type=RCPWindZoneSettings)
    bpy.types.Object.ruri_cloth_physics_collider = PointerProperty(type=RCPColliderSettings)
    bpy.types.Scene.ruri_cloth_physics = PointerProperty(type=RCPSceneSettings)


def unregister():
    del bpy.types.Scene.ruri_cloth_physics
    del bpy.types.Object.ruri_cloth_physics_collider
    del bpy.types.Object.ruri_cloth_physics_wind
    del bpy.types.Object.ruri_cloth_physics
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
