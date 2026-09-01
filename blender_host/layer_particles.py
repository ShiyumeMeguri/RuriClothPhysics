import numpy as np

from . import chain
from . import runtime
from . import shapes
from . import viewport
from ..cloth_kernel import defs
from ..cloth_kernel import host_math
from ..cloth_kernel import io as kernel_io

COLOR_MOVE = (0.8, 0.8, 0.8, 0.8)
COLOR_FIXED = (1.0, 0.0, 0.0, 0.8)
COLOR_INVALID = (0.0, 0.0, 0.0, 0.8)
COLOR_LINE = (0.0, 1.0, 1.0, 1.0)
COLOR_TRIANGLE = (1.0, 0.0, 0.8, 1.0)
COLOR_ANIMATED_LINE = (0.6, 0.8, 1.0, 1.0)
COLOR_ANIMATED_TRIANGLE = (0.8, 0.6, 1.0, 1.0)
COLOR_INERTIA = (1.0, 0.0, 1.0, 1.0)
COLOR_AXIS_X = (1.0, 0.2, 0.2, 1.0)
COLOR_AXIS_Y = (0.2, 1.0, 0.2, 1.0)
COLOR_AXIS_Z = (0.3, 0.3, 1.0, 1.0)
COLOR_COLLISION = (1.0, 0.78, 0.10, 0.75)
COLOR_COLLISION_EDGE = (1.0, 0.55, 0.10, 0.65)


RADIUS_PLANES = (("team", "scale_ratio"),)

GIZMO_PLANE_TABLE = (
    ("axis", (("particle", "rotations"),)),
    ("animated_position", (("particle", "base_positions"),)),
    ("animated_axis", (("particle", "base_positions"), ("particle", "base_rotations"))),
    ("animated_shape", (("particle", "base_positions"),)),
    ("inertia_center", (("team", "now_world_position"),)),
)


def planes(scene):
    found = set()
    for obj in scene.objects:
        if obj.type != 'ARMATURE':
            continue
        settings = getattr(obj, "ruri_cloth_physics", None)
        if settings is None:
            continue
        if settings.show_collision_radius:
            found.update(RADIUS_PLANES)
        for config in settings.configs:
            gizmos = config.gizmos
            if not gizmos.enable:
                continue
            for switch_name, needed in GIZMO_PLANE_TABLE:
                if getattr(gizmos, switch_name):
                    found.update(needed)
    return found


def poll(context):
    scene = context.scene
    if scene is None:
        return False
    for _, index, _ in runtime.iter_entries(scene):
        return True
    return False


def _points(canvas, positions, attr_fixed, attr_move, depth, use_depth, depth_test):
    count = len(positions)
    colors = np.empty((count, 4), dtype=np.float32)
    if use_depth:
        colors[:, 0] = depth
        colors[:, 1] = 0.2
        colors[:, 2] = 1.0 - depth
        colors[:, 3] = 1.0
    else:
        colors[:] = COLOR_INVALID
        colors[attr_move] = COLOR_MOVE
        colors[attr_fixed] = COLOR_FIXED
    canvas.points(positions, colors, depth_test=depth_test)


def _axes(canvas, positions, rotations, depth_test, scale=0.02):
    broadcast = np.broadcast_to(host_math.VEC_RIGHT, positions.shape).astype(np.float32)
    right = host_math.quat_rotate(rotations, broadcast)
    normal = host_math.quat_to_normal(rotations)
    tangent = host_math.quat_to_tangent(rotations)
    canvas.lines(positions, positions + right * scale, COLOR_AXIS_X, depth_test)
    canvas.lines(positions, positions + normal * scale, COLOR_AXIS_Y, depth_test)
    canvas.lines(positions, positions + tangent * scale, COLOR_AXIS_Z, depth_test)


def _shape(canvas, setup, positions, line_color, triangle_color, depth_test):
    if len(setup.lines):
        canvas.lines(positions[setup.lines[:, 0]], positions[setup.lines[:, 1]],
                     line_color, depth_test)
    if len(setup.triangles):
        triangles = setup.triangles
        for first, second in ((0, 1), (1, 2), (2, 0)):
            canvas.lines(positions[triangles[:, first]], positions[triangles[:, second]],
                         triangle_color, depth_test)


def particle_radii(world, slot):
    depth = kernel_io.particle_display(world, slot, "depth")
    index = np.full(len(depth), slot, dtype=np.int64)
    radius = host_math.evaluate_team_lut(kernel_io.team_plane(world, "radius_lut"),
                                         index, depth)
    return np.maximum(radius, 0.0001) * float(
        kernel_io.team_display(world, slot, "scale_ratio"))


RADIUS_ABSENT_NOT_LIVE = "碰撞半径需开启实时模拟"
RADIUS_ABSENT_NO_DATA = "这条配置还没有模拟数据, 走一帧即可"
RADIUS_ABSENT_NO_COLLISION = "这条配置的碰撞方式是不碰撞, 没有碰撞体积可画"
RADIUS_ABSENT_NO_BONES = "这条配置没有骨骼参与碰撞"


def collision_volume_absence(settings, entry):
    if not settings.live:
        return RADIUS_ABSENT_NOT_LIVE
    if entry is None or entry.setup is None or entry.team is None:
        return RADIUS_ABSENT_NO_DATA
    world = runtime.get_world()
    if int(kernel_io.team_display(world, entry.team, "collision_mode")) == defs.COLLISION_NONE:
        return RADIUS_ABSENT_NO_COLLISION
    if not len(entry.setup.collision_process_index):
        return RADIUS_ABSENT_NO_BONES
    return ""


def _collision_volume(canvas, world, entry, setup, positions, depth_test):
    radius = particle_radii(world, entry.team)
    selected = setup.collision_process_index
    canvas.lines(*shapes.spheres(positions[selected], radius[selected]),
                 color=COLOR_COLLISION, depth_test=depth_test)
    if int(kernel_io.team_display(world, entry.team, "collision_mode")) != defs.COLLISION_EDGE \
            or not len(setup.collision_edge_index):
        return
    edges = setup.edges[setup.collision_edge_index]
    first, second = edges[:, 0], edges[:, 1]
    canvas.lines(*shapes.swept_rails(positions[first], positions[second],
                                     radius[first], radius[second]),
                 color=COLOR_COLLISION_EDGE, depth_test=depth_test)


def _baseline(canvas, setup, positions, depth_test):
    data = setup.baseline_data
    if len(data) == 0:
        return
    parents = setup.vertex_parent[data]
    keep = parents >= 0
    canvas.lines(positions[parents[keep]], positions[data[keep]], COLOR_ANIMATED_LINE, depth_test)


DISPLAY_SELECTION_REASON = (
    "the collision volume is drawn for every armature whose own switch asks for it and not "
    "only for the selected one, because the thing it is looked at against is the collider, "
    "and moving a collider means selecting the collider: gating the drawing on the armature "
    "being selected turns the overlay off at exactly the moment it is being used, and the "
    "switch already says which armatures want it; the per-config gizmo block keeps its own "
    "selection switch, which is a debug overlay somebody turns on for one rig at a time")


def collect(context, canvas):
    scene = context.scene
    selected = {obj.name for obj in context.selected_objects}
    active = context.view_layer.objects.active
    if active is not None:
        selected.add(active.name)

    world = runtime.get_world()

    for obj, index, entry in runtime.iter_entries(scene):
        settings = obj.ruri_cloth_physics
        if index >= len(settings.configs) or not obj.visible_get():
            continue
        gizmos = settings.configs[index].gizmos
        in_scope = (settings.display_scope == 'ALL'
                    or index == chain.active_config_index(settings))
        want_radius = (settings.show_collision_radius and in_scope
                       and not collision_volume_absence(settings, entry))
        want_gizmos = gizmos.enable and (gizmos.always or obj.name in selected)
        if not want_radius and not want_gizmos:
            continue

        depth_test = gizmos.ztest
        setup = entry.setup
        positions = kernel_io.particle_display(world, entry.team, "positions")
        if want_radius:
            _collision_volume(canvas, world, entry, setup, positions, settings.display_depth)
        if not want_gizmos:
            continue

        if gizmos.position or gizmos.depth:
            _points(canvas, positions,
                    kernel_io.particle_display(world, entry.team, "attr_fixed"),
                    kernel_io.particle_display(world, entry.team, "attr_move"),
                    kernel_io.particle_display(world, entry.team, "depth"),
                    gizmos.depth, depth_test)
        if gizmos.axis:
            _axes(canvas, positions,
                  kernel_io.particle_display(world, entry.team, "rotations"), depth_test)
        if gizmos.shape:
            _shape(canvas, setup, positions, COLOR_LINE, COLOR_TRIANGLE, depth_test)
        if gizmos.base_line:
            _baseline(canvas, setup, positions, depth_test)
        if gizmos.animated_position:
            base_positions = kernel_io.particle_display(world, entry.team, "base_positions")
            colors = np.empty((len(base_positions), 4), dtype=np.float32)
            colors[:] = COLOR_ANIMATED_LINE
            canvas.points(base_positions, colors, depth_test=depth_test)
        if gizmos.animated_axis:
            _axes(canvas, kernel_io.particle_display(world, entry.team, "base_positions"),
                  kernel_io.particle_display(world, entry.team, "base_rotations"), depth_test)
        if gizmos.animated_shape:
            _shape(canvas, setup,
                   kernel_io.particle_display(world, entry.team, "base_positions"),
                   COLOR_ANIMATED_LINE, COLOR_ANIMATED_TRIANGLE, depth_test)
        if gizmos.inertia_center:
            canvas.lines(*shapes.cross(
                kernel_io.team_display(world, entry.team, "now_world_position")),
                color=COLOR_INERTIA, depth_test=depth_test)


COLOR_RADIUS_HANDLE = (1.0, 0.78, 0.10)
RADIUS_MINIMUM = 0.001
GLYPH_FACTOR = 12.0
MINIMUM_GLYPH = 0.18


def handles(context):
    obj = context.object
    if obj is None or obj.type != 'ARMATURE':
        return ()
    settings = getattr(obj, "ruri_cloth_physics", None)
    if settings is None or not settings.show_collision_radius or not settings.configs:
        return ()
    index = chain.active_config_index(settings)
    entry = runtime.get_entry(obj, index)
    if collision_volume_absence(settings, entry):
        return ()

    config = settings.configs[index]
    world = runtime.get_world()
    names = list(entry.setup.bone_names)
    allowed = list(entry.setup.collision_process_index)
    chosen = allowed[len(allowed) // 2]
    for slot in allowed:
        if chain.is_selected(obj, names[slot]):
            chosen = slot
            break

    position = np.array(
        kernel_io.particle_display(world, entry.team, "positions")[chosen], dtype=np.float64)

    def read():
        return float(particle_radii(world, entry.team)[chosen])

    def write(value):
        current = read()
        if current <= 1e-9:
            return
        scaled = config.radius.value * (float(value) / current)
        config.radius.value = max(min(scaled, 1.0), 0.001)
        runtime.sync_params(obj, index)

    glyph = max(read() * GLYPH_FACTOR, MINIMUM_GLYPH) * settings.gizmo_size
    return (viewport.Handle(
        "particle.radius", viewport.ARROW,
        viewport.axis_frame(position, (1.0, 0.0, 0.0)),
        read=read, write=write, scale=glyph, color=COLOR_RADIUS_HANDLE,
        minimum=RADIUS_MINIMUM),)


LAYER = viewport.Layer("particles", poll=poll, collect=collect, handles=handles,
                       planes=planes, order=30)


def register():
    viewport.register_layer(LAYER)


def unregister():
    viewport.unregister_layer(LAYER.identifier)
