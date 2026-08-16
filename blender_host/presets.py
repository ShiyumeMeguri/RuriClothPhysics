import bpy
from bl_operators.presets import AddPresetBase
from bpy.types import Menu, Operator

PRESET_SUBDIR = "ruri_cloth_physics"


class RCP_MT_config_presets(Menu):
    bl_label = "物理参数预设"
    preset_subdir = PRESET_SUBDIR
    preset_operator = "script.execute_preset"
    draw = Menu.draw_preset


def _curve_values(path):
    return [
        "%s.value" % path,
        "%s.use_curve" % path,
        "%s.points_serialized" % path,
    ]


def _check_values(path):
    return [
        "%s.use" % path,
        "%s.value" % path,
    ]


class RCP_OT_config_preset_add(AddPresetBase, Operator):
    bl_idname = "ruri_cloth_physics.config_preset_add"
    bl_label = "保存物理参数预设"
    preset_menu = "RCP_MT_config_presets"
    preset_subdir = PRESET_SUBDIR

    preset_defines = [
        "config = bpy.context.object.ruri_cloth_physics.configs["
        "bpy.context.object.ruri_cloth_physics.active_config_index]",
    ]

    preset_values = (
        [
            "config.gravity",
            "config.gravity_direction",
            "config.gravity_falloff",
        ]
        + _curve_values("config.damping")
        + _curve_values("config.radius")
        + [
            "config.tether.distance_compression",
        ]
        + _curve_values("config.distance.stiffness")
        + [
            "config.triangle_bending.stiffness",
            "config.angle_restoration.use",
        ]
        + _curve_values("config.angle_restoration.stiffness")
        + [
            "config.angle_restoration.velocity_attenuation",
            "config.angle_restoration.gravity_falloff",
            "config.angle_limit.use",
        ]
        + _curve_values("config.angle_limit.limit_angle")
        + [
            "config.angle_limit.stiffness",
            "config.motion.use_max_distance",
        ]
        + _curve_values("config.motion.max_distance")
        + [
            "config.motion.use_backstop",
            "config.motion.backstop_radius",
        ]
        + _curve_values("config.motion.backstop_distance")
        + [
            "config.motion.stiffness",
            "config.collider_collision.mode",
            "config.collider_collision.friction",
        ]
        + _curve_values("config.collider_collision.limit_distance")
        + [
            "config.self_collision.self_mode",
            "config.self_collision.sync_mode",
        ]
        + _curve_values("config.self_collision.surface_thickness")
        + [
            "config.self_collision.cloth_mass",
            "config.inertia.world_inertia",
            "config.inertia.movement_inertia_smoothing",
        ]
        + _check_values("config.inertia.movement_speed_limit")
        + _check_values("config.inertia.rotation_speed_limit")
        + [
            "config.inertia.local_inertia",
        ]
        + _check_values("config.inertia.local_movement_speed_limit")
        + _check_values("config.inertia.local_rotation_speed_limit")
        + [
            "config.inertia.depth_inertia",
            "config.inertia.centrifugal_acceleration",
        ]
        + _check_values("config.inertia.particle_speed_limit")
        + [
            "config.inertia.teleport_mode",
            "config.inertia.teleport_distance",
            "config.inertia.teleport_rotation",
            "config.wind.influence",
            "config.wind.frequency",
            "config.wind.turbulence",
            "config.wind.blend",
            "config.wind.synchronization",
            "config.wind.depth_weight",
            "config.wind.moving_wind",
            "config.spring.use_spring",
            "config.spring.spring_power",
            "config.spring.limit_distance",
            "config.spring.normal_limit_ratio",
            "config.spring.noise",
        ]
    )


_CLASSES = (
    RCP_MT_config_presets,
    RCP_OT_config_preset_add,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
