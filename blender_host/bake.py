import numpy as np

import bpy
from bpy.props import BoolProperty, IntProperty

from . import chain
from . import compat
from . import runtime

_BONE_PATH = 'pose.bones["%s"].%s'

SCALE_CHANNEL_REASON = (
    "a bone whose connected child the solver moves reaches that child by turning and by "
    "stretching, so its scale is a result of the simulation and not a constant, and a bake "
    "that carries only location and rotation drops the stretch and plays back a chain that "
    "is the wrong length; the bones that need the channel are the ones the write path "
    "stretches, asked of the same rule the write path asks, and the bones whose head their "
    "parent owns get no location channel because blender would ignore those keyframes")


def _collect_target_bones(obj):
    move_bones = set()
    rotation_bones = set()
    settings = obj.ruri_cloth_physics
    for index, config in enumerate(settings.configs):
        if not config.enabled:
            continue
        entry = runtime.ensure_entry(obj, index, config,
                                     bpy.context.evaluated_depsgraph_get())
        if entry.setup is None:
            continue
        setup = entry.setup
        attrs = setup.attributes
        move = (attrs & runtime.defs.ATTR_MOVE) != 0
        fixed = (attrs & runtime.defs.ATTR_FIXED) != 0
        for i, name in enumerate(setup.bone_names):
            if move[i]:
                move_bones.add(name)
            elif fixed[i]:
                rotation_bones.add(name)
    rotation_bones -= move_bones
    owned_heads = chain.parent_owned_heads(obj)
    stretch_owners = chain.stretch_owners(obj)
    return {name: (name in move_bones and name not in owned_heads, name in stretch_owners)
            for name in sorted(move_bones | rotation_bones)}


def _channel_list(pose_bone, include_location, include_rotation, include_scale,
                  location_allowed, scale_allowed):
    channels = []
    if include_location and location_allowed:
        channels.append(("location", 3))
    if include_rotation:
        mode = pose_bone.rotation_mode
        if mode == 'QUATERNION':
            channels.append(("rotation_quaternion", 4))
        elif mode == 'AXIS_ANGLE':
            channels.append(("rotation_axis_angle", 4))
        else:
            channels.append(("rotation_euler", 3))
    if include_scale and scale_allowed:
        channels.append(("scale", 3))
    return channels


class RCP_OT_bake(bpy.types.Operator):
    bl_idname = "ruri_cloth_physics.bake"
    bl_label = "烘焙为关键帧"
    bl_description = "在帧范围内模拟并把Ruri 布料物理结果写入动画关键帧"
    bl_options = {'REGISTER', 'UNDO'}

    frame_start: IntProperty(name="起始帧", default=1)
    frame_end: IntProperty(name="结束帧", default=250)
    channel_location: BoolProperty(name="位置通道", default=True)
    channel_rotation: BoolProperty(name="旋转通道", default=True)
    channel_scale: BoolProperty(
        name="缩放通道", default=True,
        description="连接骨的头由父骨的尾摆到位, 父骨靠拉伸够到子骨的粒子; "
                    "关掉这个通道会让烘焙出来的链长度不对")
    mute_existing: BoolProperty(
        name="静音原始通道", default=True,
        description="采样前静音目标骨既有的动画通道, 防止旧关键帧与模拟输入串扰")
    disable_live: BoolProperty(
        name="完成后关闭实时模拟", default=True,
        description="烘焙后关闭实时模拟, 使播放结果与烘焙一致")

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (obj is not None and obj.type == 'ARMATURE'
                and len(obj.ruri_cloth_physics.configs) > 0)

    def invoke(self, context, event):
        scene = context.scene
        self.frame_start = scene.frame_start
        self.frame_end = scene.frame_end
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(self, "frame_start")
        layout.prop(self, "frame_end")
        layout.prop(self, "channel_location")
        layout.prop(self, "channel_rotation")
        layout.prop(self, "channel_scale")
        layout.prop(self, "mute_existing")
        layout.prop(self, "disable_live")

    def execute(self, context):
        obj = context.object
        scene = context.scene
        settings = obj.ruri_cloth_physics
        if self.frame_end < self.frame_start:
            self.report({'ERROR'}, "帧范围无效")
            return {'CANCELLED'}
        if not (self.channel_location or self.channel_rotation or self.channel_scale):
            self.report({'ERROR'}, "至少选择一个通道")
            return {'CANCELLED'}

        targets = _collect_target_bones(obj)
        if not targets:
            self.report({'ERROR'}, "没有可烘焙的骨骼(检查配置与根骨骼)")
            return {'CANCELLED'}

        pose_bones = obj.pose.bones
        channel_table = {}
        for name, (location_allowed, scale_allowed) in targets.items():
            pose_bone = pose_bones.get(name)
            if pose_bone is None:
                continue
            channels = _channel_list(pose_bone, self.channel_location, self.channel_rotation,
                                     self.channel_scale, location_allowed, scale_allowed)
            if channels:
                channel_table[name] = channels

        muted = []
        if self.mute_existing:
            targets = {(_BONE_PATH % (name, channel)) for name, channels in channel_table.items()
                       for channel, _ in channels}
            for fcurve in compat.iter_object_fcurves(obj):
                if fcurve.data_path in targets and not fcurve.mute:
                    fcurve.mute = True
                    muted.append(fcurve)

        original_frame = scene.frame_current
        original_live = settings.live
        settings.live = True
        runtime.request_reset(obj, 'FULL')

        frame_count = self.frame_end - self.frame_start + 1
        frames = list(range(self.frame_start, self.frame_end + 1))
        samples = {name: np.zeros((frame_count, sum(c[1] for c in channels)), dtype=np.float64)
                   for name, channels in channel_table.items()}

        try:
            for frame_index, frame in enumerate(frames):
                scene.frame_set(frame)
                runtime.flush_pose_writes(None)
                for name, channels in channel_table.items():
                    pose_bone = pose_bones.get(name)
                    if pose_bone is None:
                        continue
                    values = []
                    for channel, size in channels:
                        values.extend(getattr(pose_bone, channel)[:size])
                    samples[name][frame_index] = values
        finally:
            action, container = compat.ensure_action(obj, obj.name + "_物理烘焙")
            written = []
            for name, channels in channel_table.items():
                offset = 0
                for channel, size in channels:
                    data_path = _BONE_PATH % (name, channel)
                    for axis in range(size):
                        fcurve = compat.ensure_fcurve(container, data_path, axis, name)
                        compat.bulk_write_fcurve(
                            fcurve, frames, samples[name][:, offset + axis].tolist(),
                            self.frame_start, self.frame_end)
                        written.append(fcurve)
                    offset += size

            for fcurve in muted:
                fcurve.mute = False
            for fcurve in written:
                fcurve.mute = False

            settings.live = original_live and not self.disable_live
            scene.frame_set(original_frame)

        self.report({'INFO'}, "已烘焙 %d 根骨骼 × %d 帧" % (len(channel_table), frame_count))
        return {'FINISHED'}


def register():
    bpy.utils.register_class(RCP_OT_bake)


def unregister():
    bpy.utils.unregister_class(RCP_OT_bake)
