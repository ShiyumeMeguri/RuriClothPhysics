"""Collider layer: wireframe plus the radius / length / centre drag handles."""

import mathutils
import numpy as np

from . import collider_geom
from . import shapes
from . import viewport
from ..cloth_kernel import defs

COLOR_ACTIVE = (1.0, 0.42, 0.05, 1.0)
COLOR_IDLE = (0.10, 0.45, 0.70, 0.55)
COLOR_OFF = (0.40, 0.40, 0.40, 0.30)

COLOR_RADIUS = (1.0, 1.0, 1.0)
COLOR_END_RADIUS = (0.65, 0.65, 0.68)
COLOR_LENGTH = (0.25, 0.95, 0.45)
COLOR_CENTER = (1.0, 0.35, 0.85)
COLOR_PICK = (0.35, 0.75, 1.0)
COLOR_AXIS_X = (0.95, 0.25, 0.30)
COLOR_AXIS_Y = (0.35, 0.85, 0.30)
COLOR_AXIS_Z = (0.30, 0.50, 0.95)

MINIMUM_RADIUS = 0.001
# Calibrated against a real viewport: scale_basis feeds the glyph, not the shaft, and the glyph
# comes out roughly a twentieth of the value handed in.
GLYPH_FACTOR = 12.0
MINIMUM_GLYPH = 0.18


def _armatures(context):
    seen = set()
    candidates = list(context.selected_objects)
    active = context.view_layer.objects.active
    if active is not None:
        candidates.append(active)
    for obj in candidates:
        if obj is None or obj.type != 'ARMATURE' or obj.name in seen:
            continue
        seen.add(obj.name)
        if getattr(obj, "ruri_cloth_physics", None) is not None:
            yield obj


def _showing(obj):
    settings = obj.ruri_cloth_physics
    return settings.show_colliders and len(settings.colliders) > 0


def poll(context):
    return any(_showing(obj) for obj in _armatures(context))


def collect(context, canvas):
    for obj in _armatures(context):
        if not _showing(obj):
            continue
        settings = obj.ruri_cloth_physics
        active_index = settings.active_collider_index
        for index, item in enumerate(settings.colliders):
            matrix = collider_geom.bone_matrix(obj, item.bone)
            kind, first, second, start_radius, end_radius = collider_geom.solve(item, matrix)
            if not item.enabled:
                color = COLOR_OFF
            elif index == active_index:
                color = COLOR_ACTIVE
            else:
                color = COLOR_IDLE
            if kind == defs.COLLIDER_SPHERE:
                canvas.lines(*shapes.sphere(first, start_radius), color=color)
            elif kind == defs.COLLIDER_PLANE:
                canvas.lines(*shapes.plane(first, second - first), color=color)
            else:
                canvas.lines(*shapes.capsule(first, second, start_radius, end_radius), color=color)


def _active(context):
    obj = context.object
    if obj is None or obj.type != 'ARMATURE':
        return None, None
    settings = getattr(obj, "ruri_cloth_physics", None)
    if settings is None or not settings.show_colliders or not settings.show_collider_gizmo:
        return None, None
    if not 0 <= settings.active_collider_index < len(settings.colliders):
        return None, None
    return obj, settings.colliders[settings.active_collider_index]


def _axis_frame(matrix, center, column):
    """Frame whose +Z is bone axis `column`; ARROW gizmos slide along their own +Z."""
    basis = matrix.to_3x3().normalized()
    forward = basis.col[column].normalized()
    reference = basis.col[(column + 1) % 3]
    side = reference.cross(forward)
    if side.length < 1e-6:
        side = mathutils.Vector((1.0, 0.0, 0.0))
    side.normalize()
    up = forward.cross(side)
    result = mathutils.Matrix(((side.x, up.x, forward.x),
                               (side.y, up.y, forward.y),
                               (side.z, up.z, forward.z))).to_4x4()
    result.translation = matrix @ mathutils.Vector(center)
    return result


def _glyph_scale(settings, reference):
    """Grab-glyph size for a handle.

    scale_basis sizes the gizmo GLYPH, while the arrow's shaft length comes from its target value --
    so feeding scale_basis the raw radius drew a grab box about a twentieth of the collider, which
    is the "tiny tick at the sphere edge" this replaced. The reference is scaled up and floored so a
    17 mm toe collider is still grabbable, and gizmo_size is the user's escape hatch.
    """
    return max(reference * GLYPH_FACTOR, MINIMUM_GLYPH) * settings.gizmo_size


def picks(context, obj):
    """One click target per non-active collider, so every collider can be reached in the viewport.

    Without these only the active collider carried any gizmo, which reads as "some colliders have no
    handles at all" -- the other twenty-three were only reachable through the properties list.
    """
    settings = obj.ruri_cloth_physics
    found = []
    for index, item in enumerate(settings.colliders):
        if index == settings.active_collider_index or not item.enabled:
            continue
        matrix = collider_geom.bone_matrix(obj, item.bone)
        kind, first, second, start_radius, end_radius = collider_geom.solve(item, matrix)
        frame = mathutils.Matrix.Translation(mathutils.Vector(first))
        found.append(viewport.Handle(
            "collider.pick.%d" % index, viewport.PICK, frame,
            scale=_glyph_scale(settings, start_radius) * 0.6, color=COLOR_PICK,
            operator="ruri_cloth_physics.collider_activate", properties={"index": index}))
    return found


def handles(context):
    obj, item = _active(context)
    if item is None:
        obj = context.object
        if obj is None or obj.type != 'ARMATURE':
            return ()
        settings = getattr(obj, "ruri_cloth_physics", None)
        if settings is None or not settings.show_colliders or not settings.show_collider_gizmo:
            return ()
        return picks(context, obj)
    settings = obj.ruri_cloth_physics

    matrix = collider_geom.bone_matrix(obj, item.bone)
    matrix = mathutils.Matrix([list(row) for row in np.asarray(matrix)])
    center = tuple(item.center)
    is_capsule = item.shape == 'CAPSULE'
    radius = item.start_radius if is_capsule else item.radius
    found = list(picks(context, obj))

    def write_radius(value):
        if is_capsule:
            item.start_radius = value
            if not item.radius_separation:
                item.end_radius = value
        else:
            item.radius = value

    if item.shape != 'PLANE':
        found.append(viewport.Handle(
            "collider.radius", viewport.ARROW, _axis_frame(matrix, center, 0),
            read=lambda: item.start_radius if is_capsule else item.radius,
            write=write_radius, scale=_glyph_scale(settings, radius), color=COLOR_RADIUS,
            minimum=MINIMUM_RADIUS))

    if is_capsule:
        found.append(viewport.Handle(
            "collider.length", viewport.ARROW, _axis_frame(matrix, center, 1),
            read=lambda: item.length * 0.5,
            write=lambda value: setattr(item, "length", max(float(value) * 2.0, MINIMUM_RADIUS)),
            scale=_glyph_scale(settings, item.length * 0.5), color=COLOR_LENGTH,
            minimum=MINIMUM_RADIUS))
        if item.radius_separation:
            found.append(viewport.Handle(
                "collider.end_radius", viewport.ARROW, _axis_frame(matrix, center, 2),
                read=lambda: item.end_radius,
                write=lambda value: setattr(item, "end_radius", value),
                scale=_glyph_scale(settings, item.end_radius), color=COLOR_END_RADIUS,
                minimum=MINIMUM_RADIUS))

    # move_3d draws at matrix_basis + rotation @ offset and the offset IS item.center, so this
    # frame must stay at the bone origin or the centre would be applied twice.
    origin = matrix.to_3x3().normalized().to_4x4()
    origin.translation = matrix.translation
    found.append(viewport.Handle(
        "collider.center", viewport.MOVE, origin,
        read=lambda: tuple(item.center),
        write=lambda value: setattr(item, "center",
                                    (float(value[0]), float(value[1]), float(value[2]))),
        scale=_glyph_scale(settings, radius) * 1.2, color=COLOR_CENTER))

    # Three WORLD-aligned arrows for the centre. The free-drag ball alone gives no clue which way
    # the offset runs, and the stored offset is bone-local -- on an upright head bone that is world
    # -X / +Z / +Y, so "up" and "forward" are swapped and left/right is mirrored against every
    # expectation. These arrows are labelled by world axis and move along it, whatever the bone does.
    rotation = matrix.to_3x3()
    inverse = rotation.copy()
    inverse.invert_safe()

    def world_offset():
        return rotation @ mathutils.Vector(item.center)

    def make_axis(index, direction, color, name):
        basis = viewport.axis_frame(matrix.translation, direction)

        def write(value):
            current = world_offset()
            current[index] = float(value)
            item.center = tuple(inverse @ current)

        return viewport.Handle(
            "collider.center.%s" % name, viewport.ARROW, basis,
            read=lambda: world_offset()[index], write=write,
            scale=_glyph_scale(settings, radius) * 0.8, color=color)

    for index, (direction, color, name) in enumerate((
            (mathutils.Vector((1.0, 0.0, 0.0)), COLOR_AXIS_X, "x"),
            (mathutils.Vector((0.0, 1.0, 0.0)), COLOR_AXIS_Y, "y"),
            (mathutils.Vector((0.0, 0.0, 1.0)), COLOR_AXIS_Z, "z"))):
        found.append(make_axis(index, direction, color, name))
    return found


LAYER = viewport.Layer("colliders", poll=poll, collect=collect, handles=handles, order=10)


def register():
    viewport.register_layer(LAYER)


def unregister():
    viewport.unregister_layer(LAYER.identifier)
