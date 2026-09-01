import zlib

import numpy as np

import bpy
from bpy.app.handlers import persistent

from . import armature
from . import bone_binding
from . import collider_binding
from . import collider_geom
from . import curve_host
from . import viewport
from . import wind_geom
from ..cloth_engine import pipeline as cloth_pipeline
from ..cloth_engine import target as cloth_target
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
_pose_buffers = {}
_pending_writes = []
_collider_follow_cache = {}
_animation_revision = 0

EDIT_REVISION_REASON = (
    "which bones an armature animates, and which colliders ride which bone, are both read "
    "by walking data that grows with the file rather than with the cloth, so both are kept "
    "until something could have changed them; keying a bone, muting a strip, reparenting a "
    "collider, loading a file and undoing all reach the dependency graph, and playback does "
    "not, which is exactly the split this counter needs: the walk happens on the edit and "
    "never on the frame")


@persistent
def _on_depsgraph_update_post(scene, depsgraph=None):
    global _animation_revision
    _animation_revision += 1
    _collider_follow_cache.clear()
_last_frame = None
_last_display = None
_display_planes = None


def pose_buffer(obj):
    buffer = _pose_buffers.get(obj.session_uid)
    if buffer is None:
        buffer = armature.PoseBuffer()
        _pose_buffers[obj.session_uid] = buffer
    return buffer


def scene_compile_target(scene):
    settings = getattr(scene, "ruri_cloth_physics", None)
    if settings is None:
        return cloth_target.DEFAULT_TARGET
    return getattr(settings, "compile_target", cloth_target.DEFAULT_TARGET)


def notify_compile_target_changed():
    cloth_pipeline.release(_world)


def _flush_solver():
    cloth_pipeline.flush(_world)


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
        self.names = ()
        self.meshes = ()
        self.has_mesh = False


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
        self.gate_signature = None


def get_world():
    return _world


DISPLAY_CATCH_UP_REASON = (
    "the frame downloads what the layers declared at the moment it ran, so a switch turned "
    "on after that frame has no data to draw until the next one, and the viewport is not "
    "playing while somebody is clicking switches; the switch therefore hands its own new "
    "declaration to the world and pulls those planes down once, which is the whole cost of "
    "turning an overlay on and is paid on the click rather than on every frame")


def _sync_display_planes(scene):
    global _display_planes
    planes = viewport.collect_planes(scene) if scene is not None else ()
    if planes == _display_planes:
        return False
    _display_planes = planes
    _world.set_display_planes(planes)
    return True


def refresh_display_planes(scene):
    if _sync_display_planes(scene):
        cloth_pipeline.download_display(_world)


def clear_registry():
    release_driven_pose(getattr(bpy.context, "scene", None))
    for entry in _registry.values():
        if entry.team is not None:
            _world.unregister_team(entry.team)
    _registry.clear()
    _batch_registry.clear()
    _pose_buffers.clear()
    _pending_writes.clear()
    _collider_follow_cache.clear()
    global _last_frame, _last_display, _display_planes
    _last_frame = None
    _last_display = None
    _display_planes = None


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
        if config.disable_mode == 'RESET':
            release_entry_pose(obj, entry)


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


FRAME_COLLIDER_RESOLUTION_REASON = (
    "which colliders a config collides against is the config's own references plus every "
    "collider the scene declared global, globals last in the stable order global_colliders "
    "already sorts them and a collider that is both counted once, and it is resolved here "
    "and nowhere else because four readers lay out that same block in that same order or the "
    "solver reads one collider's history against another's shape: the identity token that "
    "decides whether the binding still stands, the binding itself, the per frame poses and "
    "the mesh vertices. The scene half of that answer is the same for every config, so the "
    "frame reads the global colliders once and resolves each config's list once rather than "
    "walking the whole scene three times a config every frame for the same answer, which on "
    "a scene of a few dozen colliders and a couple dozen configs was one eighty object scan "
    "run seventy times a frame; the resolution is rebuilt from scratch every frame, so a "
    "collider added, removed, retargeted or made global between two frames is seen the frame "
    "the per-call scan used to see it")


def _resolve_config_colliders(config, global_colliders):
    taken = set()
    resolved = []
    for reference in config.collider_collision.collider_references:
        target = collider_binding.resolve(reference.collider_uid)
        settings = collider_geom.settings_of(target)
        if settings is None or not settings.is_collider:
            continue
        taken.add(target.name)
        resolved.append((target, settings))
    for target, settings in global_colliders:
        if target.name not in taken:
            resolved.append((target, settings))
    return resolved


class FrameColliders:
    def __init__(self, scene):
        self._global = collider_geom.global_colliders(scene)
        self._per_config = {}

    def resolve(self, obj, config_index, config):
        key = (obj.session_uid, config_index)
        cached = self._per_config.get(key)
        if cached is None:
            cached = _resolve_config_colliders(config, self._global)
            self._per_config[key] = cached
        return cached


COLLIDER_MESH_CACHE_REASON = (
    "reading the triangles off an evaluated object costs an evaluated mesh, and a frame "
    "asks for the same object twice, once to decide whether the binding still stands and "
    "once to hand the vertices to the world; the frame therefore evaluates each collider "
    "once into a table it carries and both readers take it from there, which also makes "
    "the two of them answer about the same evaluation rather than about two")


def _collider_mesh_geometry(target, depsgraph, mesh_cache):
    key = target.as_pointer()
    held = mesh_cache.get(key)
    if held is None:
        held = collider_geom.mesh_geometry(target, depsgraph)
        mesh_cache[key] = held
    return held


def _collider_token(resolved, depsgraph, mesh_cache):
    entries = []
    for target, settings in resolved:
        end = collider_geom.end_object(settings)
        shape_token = ()
        if collider_geom.KIND_VALUES[settings.shape] == defs.COLLIDER_MESH:
            vertices, triangles = _collider_mesh_geometry(target, depsgraph, mesh_cache)
            shape_token = collider_geom.mesh_topology_token(vertices, triangles)
        entries.append((target.session_uid, settings.shape,
                        end.session_uid if end is not None else 0, shape_token))
    return tuple(entries)


def collider_mesh_vertices(resolved, depsgraph, mesh_cache):
    values = []
    for target, settings in resolved:
        if collider_geom.KIND_VALUES[settings.shape] != defs.COLLIDER_MESH:
            values.append(None)
            continue
        values.append(_collider_mesh_geometry(target, depsgraph, mesh_cache)[0])
    return values


def build_collider_binding(resolved, depsgraph, mesh_cache):
    binding = ColliderBinding()
    kinds = []
    names = []
    meshes = []
    for target, settings in resolved:
        kind = collider_geom.KIND_VALUES[settings.shape]
        kinds.append(kind)
        names.append(target.name)
        meshes.append(_collider_mesh_geometry(target, depsgraph, mesh_cache)
                      if kind == defs.COLLIDER_MESH else None)
    binding.count = len(kinds)
    binding.kinds = np.array(kinds, dtype=np.int32)
    binding.names = tuple(names)
    binding.meshes = tuple(meshes)
    binding.has_mesh = defs.COLLIDER_MESH in kinds
    return binding


TOPOLOGY_GATE_REASON = (
    "the topology token is the one part of a config's setup signature that the frame does "
    "not have to read to decide nothing changed, because every property it is built from "
    "raises param_serial when it is edited and param_serial rides inside the params token: "
    "the bones a config names, its connection and cloth type, its overrides, its skinning "
    "and normal alignment and gravity direction all reach the world through _topology_update "
    "or _param_update, both of which bump the serial, so a frame on which the serial, the "
    "curve hashes, the collider identities and the object's mode all stand still is a frame "
    "on which the topology token stands still too, and reading it back off the armature "
    "every frame for twenty four configs only to compare it against itself is the cost this "
    "gate removes; the token is still read on the frame the gate opens, where it decides "
    "whether the structure is rebuilt, so a real topology change -- which arrives with "
    "rebuild_pending set and a bumped serial -- is answered exactly as before, and the one "
    "edit the serial cannot see, renaming the object a normal alignment points at, is the "
    "same edit the byte signature already could not see through the pointer it stored")


def ensure_entry(obj, config_index, config, depsgraph, frame_colliders=None, mesh_cache=None):
    key = (obj.session_uid, config_index)
    entry = _registry.get(key)
    if entry is None:
        entry = RuntimeEntry()
        _registry[key] = entry
    if mesh_cache is None:
        mesh_cache = {}
    if frame_colliders is None:
        frame_colliders = FrameColliders(depsgraph.scene)

    resolved = frame_colliders.resolve(obj, config_index, config)
    params_token = _params_token(obj, config)
    collider_token = _collider_token(resolved, depsgraph, mesh_cache)
    gate_signature = (params_token, collider_token, obj.mode)
    if (entry.setup is not None and not config.rebuild_pending
            and entry.gate_signature == gate_signature):
        return entry

    topology_token = _topology_token(config)
    if entry.setup is None or entry.topology_token != topology_token or config.rebuild_pending:
        _flush_solver()
        release_entry_pose(obj, entry)
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
        entry.binding = build_collider_binding(resolved, depsgraph, mesh_cache)
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
            _flush_solver()
            entry.binding = build_collider_binding(resolved, depsgraph, mesh_cache)
            entry.collider_token = collider_token
            _world.update_colliders(entry.team, entry.binding)

    if entry.team is not None:
        if entry.params_token != params_token:
            _world.update_params(entry.team, _build_params(config))
            entry.params_token = params_token
    entry.gate_signature = gate_signature
    return entry


COLLIDER_FOLLOW_REASON = (
    "which colliders ride a bone this armature's solver drives is walked once and kept, "
    "because the walk is over every object in the scene and every constraint on it while "
    "the answer changes only when somebody reparents a collider or edits a configuration; "
    "the empty answer is never kept, because the first frame asks before the registry that "
    "names the driven bones has been filled and would otherwise pin an empty answer over "
    "the whole of playback")


def _collider_followers(scene, obj, driven):
    key = obj.session_uid
    held = _collider_follow_cache.get(key)
    if held is not None:
        return held
    rows = []
    if driven:
        for candidate in scene.objects:
            settings = collider_geom.settings_of(candidate)
            if settings is None or not settings.is_collider:
                continue
            for target in (candidate, collider_geom.end_object(settings)):
                if target is None:
                    continue
                constraint = collider_geom.bone_constraint(target)
                if constraint is None or constraint.target is not obj:
                    continue
                if constraint.subtarget in driven:
                    rows.append((target, constraint.subtarget, constraint))
        _collider_follow_cache[key] = rows
    return rows


def _collider_world_overrides(scene, obj, batch, anim_world):
    followers = _collider_followers(scene, obj, batch.bone_position)
    if not followers:
        return None
    override = {}
    for target, bone_name, constraint in followers:
        position = batch.bone_position.get(bone_name)
        if position is None:
            continue
        override[target.as_pointer()] = (
            anim_world[position] @ armature.read_matrix(constraint.inverse_matrix)
            @ armature.read_matrix(target.matrix_basis))
    return override


def _component_pose(obj):
    world = armature.read_matrix(obj.matrix_world)
    rotation, component_scale, reflected = armature.decompose_component_basis(world)
    return (world[:3, 3].astype(np.float32), rotation.astype(np.float32),
            component_scale.astype(np.float32), reflected)


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
    token = tuple((id(entry), id(entry.setup), entry.gate_signature) for entry in entries)
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
    frame_colliders = None
    collider_cache = {}
    collider_mesh_cache = {}
    for obj in _active_armatures(scene):
        settings = obj.ruri_cloth_physics
        if depsgraph is None:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            evaluated_objects = collider_geom.evaluated_objects(depsgraph)
            frame_colliders = FrameColliders(scene)
        if frame_globals.zones is None:
            frame_globals.zones = _gather_wind_zones(scene, depsgraph)

        matrix_world = armature.read_matrix(obj.matrix_world)
        matrix_world_inverse = np.linalg.inv(matrix_world)
        component_position, component_rotation, component_scale, component_reflected = \
            _component_pose(obj)
        name_map = {}
        name_maps[obj.session_uid] = name_map

        active = []
        for index, config in enumerate(settings.configs):
            if not config.enabled:
                continue
            entry = ensure_entry(obj, index, config, depsgraph, frame_colliders,
                                 collider_mesh_cache)
            if entry.setup is None or entry.team is None:
                continue
            active.append((index, config, entry))
        if not active:
            continue

        batch = _ensure_batch(obj, [entry for _, _, entry in active])
        buffer = pose_buffer(obj)
        rest_rows = buffer.rest_driven_rows(obj, batch.bone_names, batch.pose_index,
                                            (id(batch), _animation_revision))
        all_basis = buffer.read_animation(obj, rest_rows)
        all_matrix = armature.read_pose_matrices(obj, "matrix")
        gathered = batch.gather(all_basis)
        anim_world_all = batch.compute_world(matrix_world, all_matrix, gathered)
        batch._frame_all_matrix = all_matrix
        batch._frame_anim_world = anim_world_all
        batch._frame_matrix_world_inverse = matrix_world_inverse
        collider_override = _collider_world_overrides(scene, obj, batch, anim_world_all)

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
                                           component_rotation, component_scale,
                                           component_reflected, anchor,
                                           invisible, distance_weight, 0)
            kernel_io.set_team_transform_worlds(_world, entry.team, transform_worlds)

            if entry.binding.count:
                resolved = frame_colliders.resolve(obj, index, config)
                positions, rotations, tips, radii, enabled = collider_geom.gather(
                    [target for target, _settings in resolved], depsgraph,
                    evaluated_objects, collider_cache, collider_override)
                kernel_io.set_team_collider_input(_world, entry.team, positions, rotations,
                                                  tips, radii, enabled)
                if entry.binding.has_mesh:
                    kernel_io.set_team_collider_mesh_vertices(
                        _world, entry.team,
                        collider_mesh_vertices(resolved, depsgraph, collider_mesh_cache))

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
        targets = []
        if int(_world.team["self_mode"][entry.team]) == defs.SELF_MODE_FULL_MESH:
            targets.append(entry.team)
        if int(_world.team["sync_mode"][entry.team]) == defs.SELF_MODE_FULL_MESH and partner:
            targets.append(partner)
        kernel_io.set_team_contact_links(_world, entry.team, targets)

    if frame_globals.zones is None:
        frame_globals.zones = []
    _sync_display_planes(scene)
    cloth_pipeline.run_frame(_world, frame_globals, scene_compile_target(scene))

    global _last_display
    _last_display = collected
    _solve_pose_writes(collected)


DEFERRED_WRITE_REASON = (
    "the solved pose is computed in the frame change handler and written into the armature "
    "at the start of the next frame rather than at the end of this one, because a write "
    "after the frame has been evaluated is a second evaluation: blender re-poses the rig "
    "and re-runs every modifier stack that depends on it, and a stack is re-run from its "
    "first modifier whatever the change was, so a mirror sitting in front of an armature "
    "modifier is rebuilt a second time for a change that only moved bones. Writing before "
    "the frame is evaluated folds the pose into the evaluation the frame was going to do "
    "anyway -- one evaluation instead of two, measured here as 11.6 of the 15.9 "
    "milliseconds a frame took. The cloth is therefore one frame behind the animation, "
    "which at any playback rate is smaller than the delay of the solver itself, and the "
    "bake asks for the flush explicitly because it samples the pose the instant the frame "
    "is set and cannot wait for a frame that never comes")


def _solve_pose_writes(collected):
    index_in = 0
    while index_in < len(collected):
        batch = collected[index_in][3]
        obj = collected[index_in][1]
        group = []
        while index_in < len(collected) and collected[index_in][3] is batch:
            group.append(collected[index_in])
            index_in += 1

        final_world = np.array(batch._frame_anim_world, dtype=np.float64)
        write_select = np.zeros(batch.count, dtype=bool)
        solved = _world.transforms["solved"]
        for entry, _, _, _, _, start, stop, _ in group:
            rows = int(_world.team[entry.team]["t_start"]) + np.arange(stop - start)
            taken = solved[rows].astype(np.float64)
            keep = entry.write_mask
            final_world[start:stop][keep] = taken[keep]
            write_select[start:stop] = keep

        parent_matrix = np.array(batch._frame_all_matrix, dtype=np.float64)
        placed = np.flatnonzero(write_select & (batch.pose_index >= 0))
        parent_matrix[batch.pose_index[placed]] = np.einsum(
            'ij,njk->nik', batch._frame_matrix_world_inverse, final_world[placed])
        basis = batch.write_basis(batch._frame_matrix_world_inverse, parent_matrix,
                                  final_world)

        selected = np.flatnonzero(write_select & (batch.pose_index >= 0))
        _pending_writes.append((obj, batch.pose_index[selected],
                                basis[selected].transpose(0, 2, 1)))


POSE_WRITE_REFRESH_REASON = (
    "writing the pose is announced as a transform change and not a geometry one, because "
    "the geometry of an armature is its rest bones and the solver did not touch those: "
    "asking for both makes blender rebuild the armature datablock as well as re-pose it, "
    "which measured 2.7 milliseconds a frame more for an evaluated pose and an evaluated "
    "mesh that came out bit for bit identical. The announcement cannot be dropped or "
    "narrowed further -- 'TIME' looks three times cheaper again and is not doing the work, "
    "measured as the evaluated pose standing still for a hundred and nineteen of a hundred "
    "and twenty frames, which is the same silence as not announcing at all")

POSE_WRITE_REFRESH = {'OBJECT'}


def flush_pose_writes(refresh):
    if not _pending_writes:
        return
    for obj, rows, values in _pending_writes:
        if not rows.size:
            continue
        buffer = pose_buffer(obj)
        buffer.read(obj)
        buffer.matrices()[rows] = values
        obj.pose.bones.foreach_set("matrix_basis", buffer.flat)
        if refresh is not None:
            obj.update_tag(refresh=refresh)
    _pending_writes.clear()

HANDLER_FAILURE_REASON = (
    "an exception raised while a frame is being solved used to be caught here and written "
    "to the console, which is the shape of defect the migration already removed once from "
    "the device side, a signal that is raised and never read: blender goes on drawing, the "
    "cloth simply stops moving, and nothing tells anybody that the engine refused the "
    "frame; the exception is therefore let out of the handler, where blender reports it "
    "through the channel it already has for a failing handler, and it is let out once per "
    "distinct failure rather than once per frame, because the same traceback repeated a "
    "hundred and twenty times during one playback buries the first one; the signature is "
    "the type, the message and the innermost source line, so a different failure is always "
    "reported and the same failure is reported again after any frame that succeeds")

_handler_failure = None


def _failure_signature(error):
    innermost = error.__traceback__
    while innermost is not None and innermost.tb_next is not None:
        innermost = innermost.tb_next
    if innermost is None:
        return (type(error).__name__, str(error), "", 0)
    return (type(error).__name__, str(error), innermost.tb_frame.f_code.co_filename,
            innermost.tb_lineno)


def _report_once(action, *arguments):
    global _handler_failure
    try:
        action(*arguments)
    except Exception as error:
        signature = _failure_signature(error)
        repeated = signature == _handler_failure
        _handler_failure = signature
        if not repeated:
            raise
        return
    _handler_failure = None


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

    _report_once(run_frame, scene, delta)


@persistent
def _on_load_post(*args):
    clear_registry()
    bone_binding.invalidate()
    collider_binding.invalidate()


@persistent
def _on_undo_post(*args):
    global _last_display
    _last_display = None
    bone_binding.invalidate()
    collider_binding.invalidate()


def _entry_driven_bones(entry):
    if entry is None or entry.setup is None:
        return ()
    return entry.setup.bone_names


def _solver_driven_entries(scene):
    for obj in scene.objects if scene is not None else ():
        if obj.type != 'ARMATURE':
            continue
        settings = getattr(obj, "ruri_cloth_physics", None)
        if settings is None:
            continue
        for index in range(len(settings.configs)):
            entry = _registry.get((obj.session_uid, index))
            if entry is not None:
                yield obj, entry


DRIVEN_POSE_RELEASE_REASON = (
    "the pose basis of a bone the solver drives is the solver's own output and nobody "
    "else's input, so it is given back before every frame rather than left standing: the "
    "animation, the constraints that make a collider follow a bone, and the collider poses "
    "the host gathers off those constraints are all read out of the pose, so a basis left "
    "over from an earlier solve is read by the next frame as though the artist had put it "
    "there; the list of which bones those are lives in the registry and nowhere else, which "
    "is why the registry cannot be dropped before the pose it is the only record of has "
    "been given back, or the wipe loses its list, stops happening, and the next frame is "
    "solved on top of a run that is over. Dropping the wipe to save the evaluation it costs "
    "was tried and is why this paragraph names the collider: a collider parented to a bone "
    "the solver drives then followed the solver's own output, every frame pushed the cloth "
    "further than the last, and a spring chain on a chest read as though the body had grown "
    "by a factor of ten")

LIVE_SWITCH_SCOPE_REASON = (
    "the live switch is read where the new values are written and not where the old ones are "
    "given back, so the wipe covers every registered config whether the switch is on or off: "
    "turning it off says this object is no longer simulated, and a rig that is not simulated "
    "is the animation, while a pose the solver wrote and then stopped maintaining is neither "
    "the animation nor a live simulation and nothing in the file corresponds to it; it is the "
    "same state the registry clear used to leave, reached by a different switch, and it "
    "survives scrubbing, saving and rendering with no way back but turning the switch on "
    "again; freezing a pose is what the bake is for, and an enable switch that freezes one as "
    "a side effect is a switch that does two things and says one. The switch also hands the "
    "bones back where it is thrown, because the frame that would otherwise do it is a frame "
    "nobody has asked for yet when the timeline is standing still")

ENTRY_POSE_RELEASE_REASON = (
    "every place that stops owning a bone gives that bone back through this one call, and "
    "reads which bones those are out of the registry entry rather than off a list of its "
    "own: the entry's setup is the only record of what a config drives, and the whole shape "
    "of this defect is a second record, kept somewhere else, that disagrees with it or is "
    "simply missing when the first one is thrown away; the callers are the frame handler, "
    "which gives back everything registered before the frame it is about to solve, the "
    "rebuild, which stops owning every bone its new setup does not name, and the enable "
    "switch, which stops owning the whole config")


def release_entry_pose(obj, entry):
    armature.clear_pose_basis(obj, _entry_driven_bones(entry))


def release_driven_pose(scene):
    for obj, entry in _solver_driven_entries(scene):
        release_entry_pose(obj, entry)


def release_armature_pose(obj):
    for (uid, _index), entry in _registry.items():
        if uid == obj.session_uid:
            release_entry_pose(obj, entry)


PRE_FRAME_TAG_REASON = (
    "the flush announces the change even though it runs before the frame is evaluated, "
    "because writing the plane through foreach_set does not invalidate anything on its own "
    "-- the basis reads back correctly and the evaluated pose does not move, which is the "
    "same silence that made a still timeline stop updating. Announcing here is free: the "
    "evaluation that consumes the tag is the one the frame change was going to run anyway, "
    "so the rig is posed once. Announcing after the frame, which is where this used to "
    "happen, is what cost a second evaluation of every modifier stack hanging off the rig")


@persistent
def _on_frame_change_pre(scene, depsgraph=None):
    flush_pose_writes(POSE_WRITE_REFRESH)


@persistent
def _on_save_pre(*args):
    _pending_writes.clear()
    release_driven_pose(bpy.context.scene)


@persistent
def _on_save_post(*args):
    if _last_display is None:
        return
    _report_once(_solve_pose_writes, _last_display)
    flush_pose_writes(POSE_WRITE_REFRESH)


def register():
    if _on_frame_change_pre not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(_on_frame_change_pre)
    if _on_frame_change_post not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_on_frame_change_post)
    names = {getattr(handler, "__name__", "") for handler in
             bpy.app.handlers.depsgraph_update_post}
    if _on_depsgraph_update_post.__name__ not in names:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update_post)
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
    for handler in list(bpy.app.handlers.depsgraph_update_post):
        if getattr(handler, "__name__", "") == _on_depsgraph_update_post.__name__:
            bpy.app.handlers.depsgraph_update_post.remove(handler)
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
