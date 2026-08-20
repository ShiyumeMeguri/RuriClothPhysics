import zlib

import numpy as np

import bpy
from bpy.app.handlers import persistent

from . import armature
from . import bone_binding
from . import collider_geom
from . import curve_host
from . import wind_geom
from ..cloth_kernel import compile as kernel_compile
from ..cloth_kernel import defs
from ..cloth_kernel import io as kernel_io
import sys as _sys
from .. import cloth_kernel as _cloth_kernel_pkg
_sys.modules.setdefault("cloth_kernel", _cloth_kernel_pkg)
from ..cloth_kernel import world as kernel_world

_world = kernel_world.World()
_registry = {}
_batch_registry = {}
_last_frame = None
_last_display = None

DEFAULT_BACKEND = 'GPU'
_backends = {}


def _backend_module(name):
    backend = _backends.get(name)
    if backend is not None:
        return backend
    if name == 'CPU':
        from ..cloth_engine_cpu import pipeline as backend
    else:
        from ..cloth_engine_gpu import pipeline as backend
    _backends[name] = backend
    return backend


def scene_backend(scene):
    settings = getattr(scene, "ruri_cloth_physics", None)
    if settings is None:
        return DEFAULT_BACKEND
    return getattr(settings, "backend", DEFAULT_BACKEND)


def notify_backend_changed():
    for backend in _backends.values():
        backend.release(_world)


def _flush_backends():
    for backend in _backends.values():
        backend.flush(_world)

NORMAL_AXIS_VECTORS = {
    'RIGHT': (1.0, 0.0, 0.0),
    'UP': (0.0, 1.0, 0.0),
    'FORWARD': (0.0, 0.0, 1.0),
    'INVERSE_RIGHT': (-1.0, 0.0, 0.0),
    'INVERSE_UP': (0.0, -1.0, 0.0),
    'INVERSE_FORWARD': (0.0, 0.0, -1.0),
}

COLLISION_MODE_VALUES = {'NONE': defs.COLLISION_NONE, 'POINT': defs.COLLISION_POINT,
                         'EDGE': defs.COLLISION_EDGE}
SELF_MODE_VALUES = {'NONE': defs.SELF_MODE_NONE, 'FULL_MESH': defs.SELF_MODE_FULL_MESH}
TELEPORT_MODE_VALUES = {'NONE': defs.TELEPORT_NONE, 'RESET': defs.TELEPORT_RESET,
                        'KEEP': defs.TELEPORT_KEEP}
FORCE_MODE_VALUES = {'VELOCITY_ADD': defs.FORCE_VELOCITY_ADD,
                     'VELOCITY_ADD_WITHOUT_DEPTH': defs.FORCE_VELOCITY_ADD_WITHOUT_DEPTH,
                     'VELOCITY_CHANGE': defs.FORCE_VELOCITY_CHANGE,
                     'VELOCITY_CHANGE_WITHOUT_DEPTH': defs.FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH}


class ColliderBinding:
    def __init__(self):
        self.count = 0
        self.kinds = np.zeros(0, dtype=np.int32)


class RuntimeEntry:
    def __init__(self):
        self.team = None
        self.setup = None
        self.kinematics = None
        self.binding = None
        self.topology_token = None
        self.collider_token = None
        self.params_token = None
        self.host = None
        self.write_mask = None
        self.position_mask = None
        self.culling_invisible = False
        self.signature_rev = None


def get_world():
    return _world


def clear_registry():
    for entry in _registry.values():
        if entry.team is not None:
            _world.unregister_team(entry.team)
    _registry.clear()
    _batch_registry.clear()
    global _last_frame, _last_display
    _last_frame = None
    _last_display = None


def get_entry(obj, config_index):
    return _registry.get((obj.session_uid, config_index))


def iter_entries(scene):
    for obj in scene.objects:
        if obj.type != 'ARMATURE':
            continue
        settings = getattr(obj, "ruri_cloth_physics", None)
        if settings is None:
            continue
        for index in range(len(settings.configs)):
            entry = _registry.get((obj.session_uid, index))
            if entry is not None and entry.team is not None:
                yield obj, index, entry


def request_reset(obj, mode):
    for (uid, index), entry in _registry.items():
        if uid != obj.session_uid or entry.team is None:
            continue
        _world.request_reset(entry.team, mode)


def add_force(obj, config_index, force_direction, force_velocity, force_mode='VELOCITY_ADD'):
    entry = _registry.get((obj.session_uid, config_index))
    if entry is None or entry.team is None:
        return
    direction = np.array(force_direction, dtype=np.float32)
    length = float(np.linalg.norm(direction))
    mode = FORCE_MODE_VALUES.get(force_mode, defs.FORCE_NONE)
    if length <= 0.0 or force_velocity <= 0.0 or mode == defs.FORCE_NONE:
        return
    _world.add_force(entry.team, mode, direction / length * force_velocity)


def notify_config_enabled_changed(obj, config_index):
    settings = obj.ruri_cloth_physics
    if config_index >= len(settings.configs):
        return
    config = settings.configs[config_index]
    entry = _registry.get((obj.session_uid, config_index))
    if config.enabled:
        if entry is not None and entry.team is not None and config.disable_mode == 'RESET':
            _world.request_reset(entry.team, 'FULL')
    else:
        if entry is not None and entry.setup is not None and config.disable_mode == 'RESET':
            armature.clear_pose_basis(obj, entry.setup.bone_names)


def _topology_token(config):
    return (
        config.cloth_type,
        config.spring.use_spring,
        config.connection_mode,
        tuple(item.bone for item in config.root_bones),
        tuple((o.bone, o.attribute, o.disable_collision, o.exclude_motion)
              for o in config.attribute_overrides),
        tuple(item.bone for item in config.collider_collision.collision_bones),
        config.custom_skinning_enable,
        tuple(item.bone for item in config.skinning_bones),
        config.normal_alignment_mode,
        config.normal_alignment_object.name if config.normal_alignment_object else "",
        config.normal_alignment_bone,
        tuple(np.round(np.array(config.gravity_direction, dtype=np.float64), 6)),
    )


def _params_token(obj, config):
    curves = (
        curve_host.curve_content_token(config.damping),
        curve_host.curve_content_token(config.radius),
        curve_host.curve_content_token(config.distance.stiffness),
        curve_host.curve_content_token(config.angle_restoration.stiffness),
        curve_host.curve_content_token(config.angle_limit.limit_angle),
        curve_host.curve_content_token(config.motion.max_distance),
        curve_host.curve_content_token(config.motion.backstop_distance),
        curve_host.curve_content_token(config.collider_collision.limit_distance),
        curve_host.curve_content_token(config.self_collision.surface_thickness),
    )
    return (obj.ruri_cloth_physics.param_serial, curves)


def _build_params(config):
    is_spring = config.cloth_type == 'BONE_SPRING'
    gravity_direction = np.array(config.gravity_direction, dtype=np.float32)
    length = float(np.linalg.norm(gravity_direction))
    gravity_direction = gravity_direction / length if length > defs.EPSILON \
        else np.zeros(3, dtype=np.float32)

    collision = config.collider_collision
    if is_spring:
        collision_mode = defs.COLLISION_POINT
        dynamic_friction = defs.BONE_SPRING_COLLISION_FRICTION
        static_friction = defs.BONE_SPRING_COLLISION_FRICTION
        distance_lut = np.full(defs.CURVE_LUT_SAMPLES, defs.BONE_SPRING_DISTANCE_STIFFNESS,
                               dtype=np.float32)
        tether_compression = defs.BONE_SPRING_TETHER_COMPRESSION_LIMIT
        use_max_distance = False
        use_backstop = False
        gravity = 0.0
        self_mode = defs.SELF_MODE_NONE
        sync_mode = defs.SELF_MODE_NONE
    else:
        collision_mode = COLLISION_MODE_VALUES[collision.mode]
        dynamic_friction = collision.friction * defs.COLLIDER_DYNAMIC_FRICTION_RATIO
        static_friction = collision.friction * defs.COLLIDER_STATIC_FRICTION_RATIO
        distance_lut = curve_host.get_lut(config.distance.stiffness)
        tether_compression = config.tether.distance_compression
        use_max_distance = config.motion.use_max_distance
        use_backstop = config.motion.use_backstop
        gravity = config.gravity
        self_mode = SELF_MODE_VALUES[config.self_collision.self_mode]
        sync_mode = SELF_MODE_VALUES[config.self_collision.sync_mode]

    inertia = config.inertia
    return {
        "gravity": gravity,
        "gravity_direction": gravity_direction,
        "gravity_falloff": config.gravity_falloff,
        "stablization_time": config.stablization_time,
        "blend_weight_param": config.blend_weight,
        "damping_lut": curve_host.get_lut(config.damping) * 0.2,
        "radius_lut": curve_host.get_lut(config.radius),
        "normal_axis_vector": NORMAL_AXIS_VECTORS[config.normal_axis],
        "rotational_interpolation": config.rotational_interpolation,
        "root_rotation": config.root_rotation,
        "animation_pose_ratio": config.animation_pose_ratio,
        "time_scale": config.time_scale,
        "tether_compression": tether_compression,
        "distance_lut": distance_lut,
        "bending_stiffness": 0.0 if is_spring else config.triangle_bending.stiffness,
        "angle_use_restoration": config.angle_restoration.use,
        "angle_restoration_lut": curve_host.get_lut(config.angle_restoration.stiffness) * 0.2,
        "angle_restoration_attenuation": config.angle_restoration.velocity_attenuation,
        "angle_restoration_gravity_falloff": 0.0 if is_spring else config.angle_restoration.gravity_falloff,
        "angle_use_limit": config.angle_limit.use,
        "angle_limit_lut": curve_host.get_lut(config.angle_limit.limit_angle),
        "angle_limit_stiffness": config.angle_limit.stiffness,
        "motion_use_max_distance": use_max_distance,
        "motion_max_distance_lut": curve_host.get_lut(config.motion.max_distance),
        "motion_use_backstop": use_backstop,
        "motion_backstop_radius": config.motion.backstop_radius,
        "motion_backstop_lut": curve_host.get_lut(config.motion.backstop_distance),
        "motion_stiffness": config.motion.stiffness,
        "collision_mode": collision_mode,
        "dynamic_friction": dynamic_friction,
        "static_friction": static_friction,
        "limit_distance_lut": curve_host.get_lut(collision.limit_distance),
        "self_mode": self_mode,
        "sync_mode": sync_mode,
        "self_thickness_lut": curve_host.get_lut(config.self_collision.surface_thickness),
        "self_cloth_mass": config.self_collision.cloth_mass,
        "anchor_inertia": inertia.anchor_inertia,
        "world_inertia": inertia.world_inertia,
        "movement_inertia_smoothing": inertia.movement_inertia_smoothing,
        "movement_speed_limit": inertia.movement_speed_limit.value if inertia.movement_speed_limit.use else -1.0,
        "rotation_speed_limit": inertia.rotation_speed_limit.value if inertia.rotation_speed_limit.use else -1.0,
        "local_inertia": inertia.local_inertia,
        "local_movement_speed_limit": inertia.local_movement_speed_limit.value if inertia.local_movement_speed_limit.use else -1.0,
        "local_rotation_speed_limit": inertia.local_rotation_speed_limit.value if inertia.local_rotation_speed_limit.use else -1.0,
        "depth_inertia": inertia.depth_inertia,
        "centrifugal_acceleration": inertia.centrifugal_acceleration,
        "particle_speed_limit": inertia.particle_speed_limit.value if inertia.particle_speed_limit.use else -1.0,
        "teleport_mode": TELEPORT_MODE_VALUES[inertia.teleport_mode],
        "teleport_distance": inertia.teleport_distance,
        "teleport_rotation": inertia.teleport_rotation,
        "wind_influence": config.wind.influence,
        "wind_frequency": config.wind.frequency,
        "wind_turbulence": config.wind.turbulence,
        "wind_blend": config.wind.blend,
        "wind_synchronization": config.wind.synchronization,
        "wind_depth_weight": config.wind.depth_weight,
        "wind_moving": config.wind.moving_wind,
        "spring_power": config.spring.spring_power if (is_spring and config.spring.use_spring) else 0.0,
        "spring_limit_distance": config.spring.limit_distance,
        "spring_normal_limit_ratio": config.spring.normal_limit_ratio,
        "spring_noise": config.spring.noise,
    }


def _collider_objects(config):
    for reference in config.collider_collision.collider_references:
        target = reference.object
        settings = collider_geom.settings_of(target)
        if settings is None or not settings.is_collider:
            continue
        yield target, settings


def _collider_token(config):
    entries = []
    for target, settings in _collider_objects(config):
        end = collider_geom.end_object(settings)
        entries.append((target.session_uid, settings.shape,
                        end.session_uid if end is not None else 0))
    return tuple(entries)


def collider_objects(config):
    return [target for target, _settings in _collider_objects(config)]


def build_collider_binding(config):
    binding = ColliderBinding()
    kinds = [collider_geom.KIND_VALUES[settings.shape]
             for _target, settings in _collider_objects(config)]
    binding.count = len(kinds)
    binding.kinds = np.array(kinds, dtype=np.int32)
    return binding


def ensure_entry(obj, config_index, config):
    key = (obj.session_uid, config_index)
    entry = _registry.get(key)
    if entry is None:
        entry = RuntimeEntry()
        _registry[key] = entry

    topology_token = _topology_token(config)
    params_token = _params_token(obj, config)
    collider_token = _collider_token(config)
    signature_rev = (topology_token, params_token, collider_token, obj.mode)
    if (entry.setup is not None and not config.rebuild_pending
            and entry.signature_rev == signature_rev):
        return entry

    if entry.setup is None or entry.topology_token != topology_token or config.rebuild_pending:
        _flush_backends()
        if entry.team is not None:
            _world.unregister_team(entry.team)
            entry.team = None
        cache_key = (obj.session_uid, config_index, topology_token)
        if config.rebuild_pending:
            kernel_compile._cache.pop(cache_key, None)
        snapshot = armature.build_snapshot(obj, config)
        snapshot.token = cache_key
        snapshot.wind_seed = 1 + (zlib.crc32(obj.name.encode("utf-8")) + config_index) % 997
        setup = kernel_compile.build_setup(snapshot)
        entry.topology_token = topology_token
        config.rebuild_pending = False
        if not setup.valid:
            entry.setup = None
            entry.kinematics = None
            return entry
        entry.setup = setup
        entry.kinematics = armature.KinematicsHost(setup)
        entry.binding = build_collider_binding(config)
        entry.collider_token = collider_token
        entry.params_token = None
        attrs = setup.attributes
        invalid = ((attrs & defs.ATTR_MOVE) == 0) & ((attrs & defs.ATTR_FIXED) == 0)
        move = (attrs & defs.ATTR_MOVE) != 0
        entry.write_mask = ~invalid
        entry.position_mask = (move | setup.spring_active) & ~invalid
        params = _build_params(config)
        entry.params_token = params_token
        entry.team = _world.register_team(setup, params, entry.binding)
    elif entry.team is not None:
        if entry.collider_token != collider_token:
            _flush_backends()
            entry.binding = build_collider_binding(config)
            entry.collider_token = collider_token
            _world.update_colliders(entry.team, entry.binding)

    if entry.team is not None:
        if entry.params_token != params_token:
            _world.update_params(entry.team, _build_params(config))
            entry.params_token = params_token
    entry.signature_rev = signature_rev
    return entry


def _component_pose(obj):
    world = armature.read_matrix(obj.matrix_world)
    rotation = armature.matrix_to_quat(world)
    r3 = armature.quat_to_matrix3(rotation)
    signed = np.diag(r3.T @ world[:3, :3])
    return world[:3, 3].astype(np.float32), rotation.astype(np.float32), signed.astype(np.float32)


def _anchor_world(obj, config):
    inertia = config.inertia
    anchor = inertia.anchor_object
    if anchor is None:
        return None
    if inertia.anchor_bone and anchor.type == 'ARMATURE':
        world = armature.evaluate_live_bone_world(anchor, inertia.anchor_bone)
        if world is None:
            world = armature.read_matrix(anchor.matrix_world)
    else:
        world = armature.read_matrix(anchor.matrix_world)
    return (world[:3, 3].astype(np.float32),
            armature.matrix_to_quat(world).astype(np.float32))


def _viewer_position(scene, config):
    reference = config.culling.distance_culling_reference_object
    if reference is not None:
        return armature.read_matrix(reference.matrix_world)[:3, 3]
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D' and space.region_3d is not None:
                            view = np.array(space.region_3d.view_matrix.inverted(),
                                            dtype=np.float64)
                            return view[:3, 3]
    except (AttributeError, RuntimeError):
        pass
    if scene.camera is not None:
        return armature.read_matrix(scene.camera.matrix_world)[:3, 3]
    return None


def _evaluate_culling(obj, entry, config, scene):
    invisible = False
    mode = config.culling.camera_culling_mode
    if mode != 'OFF':
        try:
            visible = obj.visible_get()
        except RuntimeError:
            visible = True
        if not visible:
            invisible = True

    distance_weight = 1.0
    distance_use = config.culling.distance_culling_length.use
    if distance_use:
        viewer = _viewer_position(scene, config)
        if viewer is not None:
            component = armature.read_matrix(obj.matrix_world)[:3, 3]
            dist = float(np.linalg.norm(component - viewer))
            length = config.culling.distance_culling_length.value
            if dist >= length:
                invisible = True
            fade = min(max(config.culling.distance_culling_fade_ratio, 0.0), 1.0) * length
            if fade > 0.0:
                distance_weight = 1.0 - min(max((dist - (length - fade)) / fade, 0.0), 1.0)
            else:
                distance_weight = 0.0 if dist >= length else 1.0

    was_invisible = entry.culling_invisible
    entry.culling_invisible = invisible
    if invisible != was_invisible and not invisible:
        if distance_use or mode == 'RESET':
            _world.request_reset(entry.team, 'FULL')
        elif mode == 'KEEP':
            _world.request_reset(entry.team, 'KEEP')
    return invisible, distance_weight


def _gather_wind_zones(scene, depsgraph=None):
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    zones = []
    for obj in scene.objects:
        wind = getattr(obj, "ruri_cloth_physics_wind", None)
        if wind is None or not wind.is_wind_zone or not wind.enabled:
            continue
        if obj.hide_viewport:
            continue
        zone = kernel_io.WindZoneInput()
        zone.zone_id = obj.session_uid
        zone.mode = wind.mode
        zone.main = wind.main
        zone.turbulence = wind.turbulence
        zone.is_addition = wind.is_addition
        world = wind_geom.zone_matrix(obj, depsgraph)
        display_size = wind_geom.zone_display_size(obj, depsgraph)
        zone.world_position = world[:3, 3].astype(np.float32)
        zone.world_to_local = np.linalg.inv(world)
        world_scale = armature.matrix_scale(world).astype(np.float32)

        zone.size = wind_geom.local_extent(display_size, wind)
        zone.zone_volume = wind_geom.zone_volume(display_size, wind, world_scale)

        if wind.mode == wind_geom.RADIAL_MODE:
            zone.attenuation_lut = curve_host.get_lut(wind.attenuation)
        else:
            zone.world_direction = wind_geom.world_direction(world, wind)
        zones.append(zone)
    zones.sort(key=lambda entry: entry.zone_id)
    return zones


def _active_armatures(scene):
    result = []
    for obj in scene.objects:
        if obj.type != 'ARMATURE':
            continue
        settings = getattr(obj, "ruri_cloth_physics", None)
        if settings is None or not settings.live or len(settings.configs) == 0:
            continue
        result.append(obj)
    result.sort(key=lambda o: o.name)
    return result


def _ensure_batch(obj, entries):
    key = obj.session_uid
    token = tuple((id(entry), id(entry.setup), entry.signature_rev) for entry in entries)
    cached = _batch_registry.get(key)
    if cached is None or cached[0] != token:
        cached = (token, armature.BatchedKinematics(obj, entries))
        _batch_registry[key] = cached
    return cached[1]


def sync_params(obj, config_index):
    entry = _registry.get((obj.session_uid, config_index))
    if entry is None or entry.team is None or config_index >= len(obj.ruri_cloth_physics.configs):
        return False
    config = obj.ruri_cloth_physics.configs[config_index]
    _world.update_params(entry.team, _build_params(config))
    entry.params_token = _params_token(obj, config)
    return True


def run_frame(scene, frame_delta_time):
    scene_settings = scene.ruri_cloth_physics
    frame_globals = kernel_io.FrameGlobals()
    frame_globals.frame_delta_time = frame_delta_time
    frame_globals.simulation_frequency = int(scene_settings.simulation_frequency)
    frame_globals.max_simulation_count = int(scene_settings.max_simulation_count)
    frame_globals.global_time_scale = scene_settings.global_time_scale
    frame_globals.frame_index = scene.frame_current
    frame_globals.zones = None

    collected = []
    name_maps = {}
    depsgraph = None
    evaluated_objects = None
    collider_cache = {}
    for obj in _active_armatures(scene):
        settings = obj.ruri_cloth_physics
        if depsgraph is None:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            evaluated_objects = collider_geom.evaluated_objects(depsgraph)
        if frame_globals.zones is None:
            frame_globals.zones = _gather_wind_zones(scene, depsgraph)

        matrix_world = armature.read_matrix(obj.matrix_world)
        matrix_world_inverse = np.linalg.inv(matrix_world)
        component_position, component_rotation, component_scale = _component_pose(obj)
        name_map = {}
        name_maps[obj.session_uid] = name_map

        active = []
        for index, config in enumerate(settings.configs):
            if not config.enabled:
                continue
            entry = ensure_entry(obj, index, config)
            if entry.setup is None or entry.team is None:
                continue
            active.append((index, config, entry))
        if not active:
            continue

        batch = _ensure_batch(obj, [entry for _, _, entry in active])
        all_basis = armature.read_pose_matrices(obj, "matrix_basis")
        all_matrix = armature.read_pose_matrices(obj, "matrix")
        gathered = batch.gather(all_basis)
        anim_world_all = batch.compute_world(matrix_world, all_matrix, gathered)
        batch._frame_all_matrix = all_matrix
        batch._frame_anim_world = anim_world_all
        batch._frame_matrix_world_inverse = matrix_world_inverse

        for entry_index, (index, config, entry) in enumerate(active):
            invisible, distance_weight = _evaluate_culling(obj, entry, config, scene)
            if invisible:
                continue

            start, stop = batch.slices[entry_index]
            anim_world = anim_world_all[start:stop]
            transform_worlds = batch.entry_transform_worlds(entry_index, matrix_world,
                                                            all_matrix, anim_world)

            anchor = _anchor_world(obj, config)
            kernel_io.set_team_frame_input(_world, entry.team, component_position,
                                           component_rotation, component_scale, anchor,
                                           invisible, distance_weight, 0)
            kernel_io.set_team_transform_worlds(_world, entry.team, transform_worlds)

            if entry.binding.count:
                positions, rotations, tips, radii, enabled = collider_geom.gather(
                    collider_objects(config), depsgraph, evaluated_objects, collider_cache)
                kernel_io.set_team_collider_input(_world, entry.team, positions, rotations,
                                                  tips, radii, enabled)

            name_map[config.name] = entry.team
            collected.append((entry, obj, config, batch, entry_index, start, stop,
                              obj.session_uid))

    for entry, obj, config, batch, entry_index, start, stop, uid in collected:
        partner_name = config.self_collision.sync_partner
        partner = 0
        if partner_name and partner_name != config.name:
            partner = name_maps.get(uid, {}).get(partner_name, 0)
            if partner == entry.team:
                partner = 0
        _world.team["sync_target"][entry.team] = partner

    if frame_globals.zones is None:
        frame_globals.zones = []
    _backend_module(scene_backend(scene)).run_frame(_world, frame_globals)

    global _last_display
    _last_display = collected
    _emit_display(collected)


def _emit_display(collected):
    index_in = 0
    while index_in < len(collected):
        batch = collected[index_in][3]
        obj = collected[index_in][1]
        group = []
        while index_in < len(collected) and collected[index_in][3] is batch:
            group.append(collected[index_in])
            index_in += 1

        out_positions = np.zeros((batch.count, 3), dtype=np.float64)
        out_rotations = np.zeros((batch.count, 4), dtype=np.float64)
        out_rotations[:, 3] = 1.0
        write_select = np.zeros(batch.count, dtype=bool)
        for entry, _, _, _, _, start, stop, _ in group:
            team_positions, team_rotations = kernel_io.team_output(_world, entry.team)
            out_positions[start:stop] = team_positions.astype(np.float64)
            out_rotations[start:stop] = team_rotations
            write_select[start:stop] = entry.write_mask

        basis = batch.write_basis(batch._frame_matrix_world_inverse, batch._frame_all_matrix,
                                  out_positions, out_rotations, batch._frame_anim_world)

        pose_bones = obj.pose.bones
        count = len(pose_bones)
        flat = np.empty(count * 16, dtype=np.float64)
        pose_bones.foreach_get("matrix_basis", flat)
        stored = flat.reshape(count, 4, 4)
        selected = np.flatnonzero(write_select & (batch.pose_index >= 0))
        stored[batch.pose_index[selected]] = basis[selected].transpose(0, 2, 1)
        pose_bones.foreach_set("matrix_basis", flat)
        obj.update_tag(refresh={'DATA'})


@persistent
def _on_frame_change_post(scene, depsgraph=None):
    global _last_frame
    if scene is None:
        return
    frame = scene.frame_current
    fps = scene.render.fps / scene.render.fps_base
    if _last_frame is None or frame <= _last_frame:
        for entry in _registry.values():
            if entry.team is not None:
                _world.request_reset(entry.team, 'FULL')
        _last_frame = frame
        delta = 1.0 / fps
    else:
        delta = (frame - _last_frame) / fps
        _last_frame = frame

    try:
        run_frame(scene, delta)
    except Exception:
        import traceback
        traceback.print_exc()


@persistent
def _on_load_post(*args):
    clear_registry()
    bone_binding.invalidate()


@persistent
def _on_undo_post(*args):
    global _last_display
    _last_display = None
    bone_binding.invalidate()


def _solver_driven_bones(scene):
    for obj in scene.objects if scene is not None else ():
        if obj.type != 'ARMATURE':
            continue
        settings = getattr(obj, "ruri_cloth_physics", None)
        if settings is None or not settings.live or len(settings.configs) == 0:
            continue
        names = []
        for index in range(len(settings.configs)):
            entry = _registry.get((obj.session_uid, index))
            if entry is not None and entry.setup is not None:
                names.extend(entry.setup.bone_names)
        if names:
            yield obj, names


@persistent
def _on_frame_change_pre(scene, depsgraph=None):
    for obj, names in _solver_driven_bones(scene):
        armature.clear_pose_basis(obj, names)


@persistent
def _on_save_pre(*args):
    _on_frame_change_pre(bpy.context.scene)


@persistent
def _on_save_post(*args):
    if _last_display is None:
        return
    try:
        _emit_display(_last_display)
    except Exception:
        import traceback
        traceback.print_exc()


def register():
    if _on_frame_change_pre not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(_on_frame_change_pre)
    if _on_frame_change_post not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_on_frame_change_post)
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    for handlers in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if _on_undo_post not in handlers:
            handlers.append(_on_undo_post)
    if _on_save_pre not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(_on_save_pre)
    if _on_save_post not in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.append(_on_save_post)


def unregister():
    if _on_frame_change_pre in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(_on_frame_change_pre)
    if _on_frame_change_post in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_on_frame_change_post)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    for handlers in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if _on_undo_post in handlers:
            handlers.remove(_on_undo_post)
    if _on_save_pre in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(_on_save_pre)
    if _on_save_post in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.remove(_on_save_post)
    clear_registry()
