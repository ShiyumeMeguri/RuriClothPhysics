import bpy

from . import curve_host
from . import frame_cache

CLOTH_TYPE_ICONS = {'BONE_CLOTH': 'MOD_CLOTH', 'BONE_SPRING': 'FORCE_HARMONIC'}
COLLIDER_SHAPE_ICONS = {'SPHERE': 'MESH_UVSPHERE', 'CAPSULE': 'MESH_CAPSULE',
                        'PLANE': 'MESH_PLANE'}
ATTRIBUTE_ICONS = {'FIXED': 'PINNED', 'MOVE': 'UNPINNED', 'IGNORE': 'X'}


def _is_armature(context):
    return context.object is not None and context.object.type == 'ARMATURE'


def _active_config(context):
    settings = context.object.ruri_cloth_physics
    if len(settings.configs) == 0:
        return None
    return settings.configs[min(settings.active_config_index, len(settings.configs) - 1)]


def _setup_layout(layout):
    layout.use_property_split = True
    layout.use_property_decorate = False


def _draw_side_buttons(column, add_operator, remove_operator, move_operator=None, list_id=None):
    def _set(op):
        if list_id is not None:
            op.list_id = list_id
        return op

    _set(column.operator(add_operator, icon='ADD', text=""))
    _set(column.operator(remove_operator, icon='REMOVE', text=""))
    if move_operator is not None:
        column.separator()
        op = _set(column.operator(move_operator, icon='TRIA_UP', text=""))
        op.direction = 'UP'
        op = _set(column.operator(move_operator, icon='TRIA_DOWN', text=""))
        op.direction = 'DOWN'


def draw_curve(layout, owner, prop_name, label):
    curve = getattr(owner, prop_name)
    row = layout.row(align=True)
    row.prop(curve, "value", text=label)
    row.prop(curve, "use_curve", text="", icon='FCURVE')
    if curve.use_curve:
        node = curve_host.get_curve_node(curve)
        if node is not None:
            box = layout.box()
            box.template_curve_mapping(node, "mapping")
        else:
            layout.label(text="曲线节点将在下次更新时创建", icon='INFO')


def draw_check_slider(layout, owner, prop_name, label):
    check = getattr(owner, prop_name)
    row = layout.row(align=True)
    row.prop(check, "use", text="")
    sub = row.row(align=True)
    sub.enabled = check.use
    sub.prop(check, "value", text=label)


class RCP_UL_configs(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property,
                  index=0, flt_flag=0):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="", emboss=False,
                 icon='CHECKBOX_HLT' if item.enabled else 'CHECKBOX_DEHLT')
        row.label(icon=CLOTH_TYPE_ICONS.get(item.cloth_type, 'PHYSICS'))
        row.prop(item, "name", text="", emboss=False)


class RCP_UL_colliders(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property,
                  index=0, flt_flag=0):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="", emboss=False,
                 icon='CHECKBOX_HLT' if item.enabled else 'CHECKBOX_DEHLT')
        row.label(icon=COLLIDER_SHAPE_ICONS.get(item.shape, 'MESH_UVSPHERE'))
        row.prop(item, "name", text="", emboss=False)
        if item.bone:
            row.label(text=item.bone, icon='BONE_DATA')


class RCP_UL_bone_references(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property,
                  index=0, flt_flag=0):
        obj = item.id_data
        layout.prop_search(item, "bone", obj.data, "bones", text="", icon='BONE_DATA')


class RCP_UL_attribute_overrides(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property,
                  index=0, flt_flag=0):
        obj = item.id_data
        row = layout.row(align=True)
        row.label(icon=ATTRIBUTE_ICONS.get(item.attribute, 'DOT'))
        row.prop_search(item, "bone", obj.data, "bones", text="")
        row.prop(item, "attribute", text="")
        row.prop(item, "disable_collision", text="", icon='MOD_PHYSICS')
        row.prop(item, "exclude_motion", text="", icon='CON_LOCLIMIT')


class RCP_UL_collider_references(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property,
                  index=0, flt_flag=0):
        obj = item.id_data
        pool = obj.ruri_cloth_physics.colliders
        entry = pool.get(item.collider) if item.collider else None
        shape_icon = COLLIDER_SHAPE_ICONS.get(entry.shape, 'MESH_UVSPHERE') if entry else 'ERROR'
        row = layout.row(align=True)
        row.label(icon=shape_icon)
        row.prop_search(item, "collider", obj.ruri_cloth_physics, "colliders", text="")


class RCP_PT_ruri_cloth_physics(bpy.types.Panel):
    bl_label = "Ruri 布料物理"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "physics"
    bl_idname = "RCP_PT_ruri_cloth_physics"

    @classmethod
    def poll(cls, context):
        return _is_armature(context)

    def draw_header(self, context):
        settings = context.object.ruri_cloth_physics
        self.layout.prop(settings, "live", text="", icon='PLAY')

    def draw(self, context):
        layout = self.layout
        settings = context.object.ruri_cloth_physics

        row = layout.row()
        row.template_list("RCP_UL_configs", "", settings, "configs", settings,
                          "active_config_index", rows=3)
        side = row.column(align=True)
        _draw_side_buttons(side, "ruri_cloth_physics.config_add", "ruri_cloth_physics.config_remove",
                           "ruri_cloth_physics.config_move")

        row = layout.row(align=True)
        op = row.operator("ruri_cloth_physics.reset", text="复位(清空)", icon='FILE_REFRESH')
        op.mode = 'FULL'
        op = row.operator("ruri_cloth_physics.reset", text="复位(保形)", icon='ORIENTATION_LOCAL')
        op.mode = 'KEEP'
        layout.operator("ruri_cloth_physics.bake", icon='ACTION')


class RCP_PT_scene_settings(bpy.types.Panel):
    bl_label = "全局模拟设置"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "physics"
    bl_parent_id = "RCP_PT_ruri_cloth_physics"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        scene_settings = context.scene.ruri_cloth_physics
        layout.prop(scene_settings, "simulation_frequency")
        layout.prop(scene_settings, "max_simulation_count")
        layout.prop(scene_settings, "global_time_scale")
        layout.prop(scene_settings, "overlay_enabled")

        layout.separator()
        layout.prop(scene_settings, "cache_mode")
        row = layout.row(align=True)
        cached = frame_cache.cached_range(context.object.session_uid)
        if cached is None:
            row.label(text="缓存: 空", icon='TRASH')
        else:
            row.label(text="缓存: %d-%d 帧 (共 %d)" % cached, icon='SEQUENCE')
        row.operator("ruri_cloth_physics.cache_clear", text="清空缓存", icon='X')


class _ConfigPanel:
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "physics"

    @classmethod
    def poll(cls, context):
        return _is_armature(context) and _active_config(context) is not None


class _NonSpringPanel(_ConfigPanel):
    @classmethod
    def poll(cls, context):
        if not _ConfigPanel.poll(context):
            return False
        return _active_config(context).cloth_type != 'BONE_SPRING'


class _SpringPanel(_ConfigPanel):
    @classmethod
    def poll(cls, context):
        if not _ConfigPanel.poll(context):
            return False
        return _active_config(context).cloth_type == 'BONE_SPRING'


class RCP_PT_config_main(_ConfigPanel, bpy.types.Panel):
    bl_label = "主设置"
    bl_parent_id = "RCP_PT_ruri_cloth_physics"

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        config = _active_config(context)
        is_spring = config.cloth_type == 'BONE_SPRING'

        layout.prop(config, "cloth_type")

        layout.label(text="根骨骼")
        row = layout.row()
        row.template_list("RCP_UL_bone_references", "pb_roots", config, "root_bones", config,
                          "active_root_bone_index", rows=3)
        side = row.column(align=True)
        _draw_side_buttons(side, "ruri_cloth_physics.list_add", "ruri_cloth_physics.list_remove",
                           "ruri_cloth_physics.list_move", 'ROOT_BONES')
        op = layout.operator("ruri_cloth_physics.bones_from_selected", icon='BONE_DATA',
                             text="从选中骨骼添加根骨")
        op.list_id = 'ROOT_BONES'

        if is_spring:
            row = layout.row()
            row.enabled = False
            row.label(text="连接模式: 线(固定)")
        else:
            layout.prop(config, "connection_mode")
        layout.prop(config, "root_rotation")
        layout.prop(config, "rotational_interpolation")
        layout.separator()
        layout.prop(config, "animation_pose_ratio")
        layout.prop(config, "blend_weight")
        layout.prop(config, "time_scale")
        layout.prop(config, "disable_mode")
        layout.prop(config, "normal_axis")
        layout.prop(config, "normal_alignment_mode")
        if config.normal_alignment_mode == 'TRANSFORM':
            layout.prop(config, "normal_alignment_object")
            if config.normal_alignment_object is not None \
                    and config.normal_alignment_object.type == 'ARMATURE':
                layout.prop_search(config, "normal_alignment_bone",
                                   config.normal_alignment_object.data, "bones")

        layout.separator()
        layout.label(text="属性覆盖(固定/移动/无效)")
        row = layout.row()
        row.template_list("RCP_UL_attribute_overrides", "", config, "attribute_overrides",
                          config, "active_attribute_override_index", rows=2)
        side = row.column(align=True)
        _draw_side_buttons(side, "ruri_cloth_physics.list_add", "ruri_cloth_physics.list_remove",
                           None, 'ATTRIBUTE_OVERRIDES')
        op = layout.operator("ruri_cloth_physics.bones_from_selected", icon='PINNED',
                             text="选中骨骼设为固定")
        op.list_id = 'ATTRIBUTE_OVERRIDES'
        op.attribute = 'FIXED'


class RCP_PT_custom_skinning(_NonSpringPanel, bpy.types.Panel):
    bl_label = "自定义蒙皮"
    bl_parent_id = "RCP_PT_ruri_cloth_physics"
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        config = _active_config(context)
        self.layout.prop(config, "custom_skinning_enable", text="")

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        config = _active_config(context)
        layout.enabled = config.custom_skinning_enable
        row = layout.row()
        row.template_list("RCP_UL_bone_references", "pb_skinning", config, "skinning_bones",
                          config, "active_skinning_bone_index", rows=2)
        side = row.column(align=True)
        _draw_side_buttons(side, "ruri_cloth_physics.list_add", "ruri_cloth_physics.list_remove",
                           None, 'SKINNING_BONES')
        op = layout.operator("ruri_cloth_physics.bones_from_selected", icon='BONE_DATA')
        op.list_id = 'SKINNING_BONES'


class RCP_PT_culling(_ConfigPanel, bpy.types.Panel):
    bl_label = "剔除"
    bl_parent_id = "RCP_PT_ruri_cloth_physics"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        config = _active_config(context)
        culling = config.culling
        layout.prop(culling, "camera_culling_mode")
        draw_check_slider(layout, culling, "distance_culling_length", "剔除距离")
        sub = layout.column()
        sub.enabled = culling.distance_culling_length.use
        sub.prop(culling, "distance_culling_fade_ratio")
        sub.prop(culling, "distance_culling_reference_object")
        sub.label(text="参考对象为空时使用视口视点", icon='INFO')


class RCP_PT_parameters(_ConfigPanel, bpy.types.Panel):
    bl_label = "物理参数"
    bl_parent_id = "RCP_PT_ruri_cloth_physics"
    bl_idname = "RCP_PT_parameters"

    def draw_header_preset(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.menu("RCP_MT_config_presets", text="预设")
        row.operator("ruri_cloth_physics.config_preset_add", text="", icon='ADD')
        op = row.operator("ruri_cloth_physics.config_preset_add", text="", icon='REMOVE')
        op.remove_active = True

    def draw(self, context):
        pass


class RCP_PT_force(_ConfigPanel, bpy.types.Panel):
    bl_label = "力"
    bl_parent_id = "RCP_PT_parameters"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        config = _active_config(context)
        if config.cloth_type != 'BONE_SPRING':
            layout.prop(config, "gravity")
            layout.prop(config, "gravity_direction")
            layout.prop(config, "gravity_falloff")
        draw_curve(layout, config, "damping", "阻尼")
        layout.prop(config, "stablization_time")


class RCP_PT_spring(_SpringPanel, bpy.types.Panel):
    bl_label = "弹簧"
    bl_parent_id = "RCP_PT_parameters"

    def draw_header(self, context):
        config = _active_config(context)
        self.layout.prop(config.spring, "use_spring", text="")

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        spring = _active_config(context).spring
        layout.enabled = spring.use_spring
        layout.prop(spring, "spring_power")
        layout.prop(spring, "limit_distance")
        layout.prop(spring, "normal_limit_ratio")
        layout.prop(spring, "noise")


class RCP_PT_angle_restoration(_ConfigPanel, bpy.types.Panel):
    bl_label = "角度复原"
    bl_parent_id = "RCP_PT_parameters"
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        config = _active_config(context)
        self.layout.prop(config.angle_restoration, "use", text="")

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        config = _active_config(context)
        restoration = config.angle_restoration
        layout.enabled = restoration.use
        draw_curve(layout, restoration, "stiffness", "复原力")
        layout.prop(restoration, "velocity_attenuation")
        if config.cloth_type != 'BONE_SPRING':
            layout.prop(restoration, "gravity_falloff")


class RCP_PT_angle_limit(_ConfigPanel, bpy.types.Panel):
    bl_label = "角度制限"
    bl_parent_id = "RCP_PT_parameters"
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        config = _active_config(context)
        self.layout.prop(config.angle_limit, "use", text="")

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        limit = _active_config(context).angle_limit
        layout.enabled = limit.use
        draw_curve(layout, limit, "limit_angle", "制限角度")
        layout.prop(limit, "stiffness")


class RCP_PT_shape_restoration(_NonSpringPanel, bpy.types.Panel):
    bl_label = "形状复原"
    bl_parent_id = "RCP_PT_parameters"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        config = _active_config(context)
        draw_curve(layout, config.distance, "stiffness", "距离刚性")
        layout.prop(config.tether, "distance_compression", text="系绳收缩限界")
        layout.prop(config.triangle_bending, "stiffness", text="三角弯曲刚性")


class RCP_PT_inertia(_ConfigPanel, bpy.types.Panel):
    bl_label = "惯性"
    bl_parent_id = "RCP_PT_parameters"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        inertia = _active_config(context).inertia
        layout.prop(inertia, "anchor_object")
        if inertia.anchor_object is not None:
            if inertia.anchor_object.type == 'ARMATURE':
                layout.prop_search(inertia, "anchor_bone", inertia.anchor_object.data, "bones")
            layout.prop(inertia, "anchor_inertia")
        layout.separator()
        layout.prop(inertia, "world_inertia")
        layout.prop(inertia, "movement_inertia_smoothing")
        draw_check_slider(layout, inertia, "movement_speed_limit", "世界移动限速")
        draw_check_slider(layout, inertia, "rotation_speed_limit", "世界旋转限速")
        layout.prop(inertia, "teleport_mode")
        sub = layout.column()
        sub.enabled = inertia.teleport_mode != 'NONE'
        sub.prop(inertia, "teleport_distance")
        sub.prop(inertia, "teleport_rotation")
        layout.separator()
        layout.prop(inertia, "local_inertia")
        draw_check_slider(layout, inertia, "local_movement_speed_limit", "局部移动限速")
        draw_check_slider(layout, inertia, "local_rotation_speed_limit", "局部旋转限速")
        layout.prop(inertia, "depth_inertia")
        layout.prop(inertia, "centrifugal_acceleration")
        draw_check_slider(layout, inertia, "particle_speed_limit", "粒子限速")


class RCP_PT_movement_limit(_NonSpringPanel, bpy.types.Panel):
    bl_label = "移动限制"
    bl_parent_id = "RCP_PT_parameters"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        motion = _active_config(context).motion
        layout.prop(motion, "use_max_distance")
        sub = layout.column()
        sub.enabled = motion.use_max_distance
        draw_curve(sub, motion, "max_distance", "最大距离")
        layout.prop(motion, "use_backstop")
        sub = layout.column()
        sub.enabled = motion.use_backstop
        sub.prop(motion, "backstop_radius")
        draw_curve(sub, motion, "backstop_distance", "背挡距离")
        layout.prop(motion, "stiffness")


class RCP_PT_collider_collision(_ConfigPanel, bpy.types.Panel):
    bl_label = "碰撞体碰撞"
    bl_parent_id = "RCP_PT_parameters"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        config = _active_config(context)
        collision = config.collider_collision
        is_spring = config.cloth_type == 'BONE_SPRING'

        if is_spring:
            row = layout.row()
            row.enabled = False
            row.label(text="模式: 点(固定)")
        else:
            layout.prop(collision, "mode")
        draw_curve(layout, config, "radius", "粒子半径")
        if is_spring:
            draw_curve(layout, collision, "limit_distance", "推出限距")
            layout.label(text="碰撞骨骼")
            row = layout.row()
            row.template_list("RCP_UL_bone_references", "pb_collision_bones", collision,
                              "collision_bones", collision, "active_collision_bone_index",
                              rows=2)
            side = row.column(align=True)
            _draw_side_buttons(side, "ruri_cloth_physics.list_add", "ruri_cloth_physics.list_remove",
                               None, 'COLLISION_BONES')
            op = layout.operator("ruri_cloth_physics.bones_from_selected", icon='BONE_DATA')
            op.list_id = 'COLLISION_BONES'
        else:
            layout.prop(collision, "friction")

        layout.label(text="碰撞体引用")
        row = layout.row()
        row.template_list("RCP_UL_collider_references", "", collision, "collider_references",
                          collision, "active_collider_reference_index", rows=3)
        side = row.column(align=True)
        _draw_side_buttons(side, "ruri_cloth_physics.list_add", "ruri_cloth_physics.list_remove",
                           "ruri_cloth_physics.list_move", 'COLLIDER_REFERENCES')


class RCP_PT_self_collision(_NonSpringPanel, bpy.types.Panel):
    bl_label = "自碰撞"
    bl_parent_id = "RCP_PT_parameters"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        obj = context.object
        self_collision = _active_config(context).self_collision
        layout.prop(self_collision, "self_mode")
        layout.prop(self_collision, "sync_mode")
        if self_collision.sync_mode != 'NONE':
            layout.prop_search(self_collision, "sync_partner", obj.ruri_cloth_physics, "configs")
        draw_curve(layout, self_collision, "surface_thickness", "表面厚度")
        layout.prop(self_collision, "cloth_mass")


class RCP_PT_wind(_ConfigPanel, bpy.types.Panel):
    bl_label = "风"
    bl_parent_id = "RCP_PT_parameters"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        wind = _active_config(context).wind
        layout.prop(wind, "influence")
        layout.prop(wind, "frequency")
        layout.prop(wind, "turbulence")
        layout.prop(wind, "blend")
        layout.prop(wind, "synchronization")
        layout.prop(wind, "depth_weight")
        layout.prop(wind, "moving_wind")
        layout.operator("ruri_cloth_physics.wind_zone_add", icon='FORCE_WIND')


class RCP_PT_colliders(_ConfigPanel, bpy.types.Panel):
    bl_label = "碰撞体池"
    bl_parent_id = "RCP_PT_ruri_cloth_physics"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return _is_armature(context)

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        obj = context.object
        settings = obj.ruri_cloth_physics

        row = layout.row()
        row.template_list("RCP_UL_colliders", "", settings, "colliders", settings,
                          "active_collider_index", rows=3)
        side = row.column(align=True)
        _draw_side_buttons(side, "ruri_cloth_physics.collider_add", "ruri_cloth_physics.collider_remove",
                           "ruri_cloth_physics.collider_move")

        row = layout.row(align=True)
        row.operator("ruri_cloth_physics.colliders_from_selected", icon='BONE_DATA',
                     text="从选中骨骼生成")
        row.operator("ruri_cloth_physics.collider_mirror", icon='MOD_MIRROR', text="镜像")

        if len(settings.colliders) == 0:
            return
        item = settings.colliders[min(settings.active_collider_index,
                                      len(settings.colliders) - 1)]
        layout.separator()
        layout.prop(item, "shape")
        layout.prop_search(item, "bone", obj.data, "bones")
        layout.prop(item, "center")
        if item.shape == 'SPHERE':
            layout.prop(item, "radius")
        elif item.shape == 'CAPSULE':
            layout.prop(item, "direction")
            layout.prop(item, "reverse_direction")
            layout.prop(item, "aligned_on_center")
            layout.prop(item, "length")
            layout.prop(item, "radius_separation")
            if item.radius_separation:
                layout.prop(item, "start_radius")
                layout.prop(item, "end_radius")
            else:
                layout.prop(item, "start_radius", text="半径")


class RCP_PT_viewport_display(_ConfigPanel, bpy.types.Panel):
    bl_label = "视口显示"
    bl_parent_id = "RCP_PT_ruri_cloth_physics"
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        config = _active_config(context)
        self.layout.prop(config.gizmos, "enable", text="")

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        gizmos = _active_config(context).gizmos
        layout.prop(context.scene.ruri_cloth_physics, "overlay_enabled", text="场景总开关")
        column = layout.column()
        column.enabled = gizmos.enable and context.scene.ruri_cloth_physics.overlay_enabled
        column.prop(gizmos, "always")
        column.prop(gizmos, "ztest")
        column.prop(gizmos, "position")
        column.prop(gizmos, "axis")
        column.prop(gizmos, "shape")
        column.prop(gizmos, "base_line")
        column.prop(gizmos, "depth")
        column.prop(gizmos, "collider")
        column.prop(gizmos, "animated_position")
        column.prop(gizmos, "animated_axis")
        column.prop(gizmos, "animated_shape")
        column.prop(gizmos, "inertia_center")


class RCP_PT_wind_zone(bpy.types.Panel):
    bl_label = "Ruri 布料物理风区"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "physics"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'EMPTY'

    def draw_header(self, context):
        wind = context.object.ruri_cloth_physics_wind
        if wind.is_wind_zone:
            self.layout.prop(wind, "enabled", text="", icon='FORCE_WIND')

    def draw(self, context):
        layout = self.layout
        _setup_layout(layout)
        wind = context.object.ruri_cloth_physics_wind
        if not wind.is_wind_zone:
            layout.operator("ruri_cloth_physics.wind_zone_convert", icon='FORCE_WIND')
            return
        layout.prop(wind, "mode")
        if wind.mode == 'BOX_DIRECTION':
            layout.prop(wind, "size")
        elif wind.mode in {'SPHERE_DIRECTION', 'SPHERE_RADIAL'}:
            layout.prop(wind, "radius")
        layout.prop(wind, "main")
        layout.prop(wind, "turbulence")
        if wind.mode == 'SPHERE_RADIAL':
            draw_curve(layout, wind, "attenuation", "衰减")
        else:
            layout.prop(wind, "direction_angle_x")
            layout.prop(wind, "direction_angle_y")
        layout.prop(wind, "is_addition")
        layout.operator("ruri_cloth_physics.wind_zone_remove", icon='X')


_CLASSES = (
    RCP_UL_configs,
    RCP_UL_colliders,
    RCP_UL_bone_references,
    RCP_UL_attribute_overrides,
    RCP_UL_collider_references,
    RCP_PT_ruri_cloth_physics,
    RCP_PT_scene_settings,
    RCP_PT_config_main,
    RCP_PT_custom_skinning,
    RCP_PT_culling,
    RCP_PT_parameters,
    RCP_PT_force,
    RCP_PT_spring,
    RCP_PT_angle_restoration,
    RCP_PT_angle_limit,
    RCP_PT_shape_restoration,
    RCP_PT_inertia,
    RCP_PT_movement_limit,
    RCP_PT_collider_collision,
    RCP_PT_self_collision,
    RCP_PT_wind,
    RCP_PT_colliders,
    RCP_PT_viewport_display,
    RCP_PT_wind_zone,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
