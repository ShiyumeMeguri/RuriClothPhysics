import bpy
import mathutils

COLOR_RADIUS = (1.0, 1.0, 1.0)
COLOR_END_RADIUS = (0.65, 0.65, 0.68)
COLOR_LENGTH = (0.25, 0.95, 0.45)
COLOR_CENTER = (1.0, 0.35, 0.85)
COLOR_HIGHLIGHT = (1.0, 0.95, 0.55)

MINIMUM_HANDLE = 0.008
MAXIMUM_HANDLE = 0.06


def active_collider(context):
    obj = context.object
    if obj is None or obj.type != 'ARMATURE':
        return None, None
    settings = getattr(obj, "ruri_cloth_physics", None)
    if settings is None or not settings.show_colliders or not settings.show_collider_gizmo:
        return None, None
    if not 0 <= settings.active_collider_index < len(settings.colliders):
        return None, None
    return obj, settings.colliders[settings.active_collider_index]


def _live():
    return active_collider(bpy.context)[1]


def _bone_matrix(obj, item):
    if item.bone:
        pose_bone = obj.pose.bones.get(item.bone)
        if pose_bone is not None:
            return obj.matrix_world @ pose_bone.matrix
    return obj.matrix_world.copy()


def _oriented(matrix, center, column):
    """Frame whose +Z (the axis a handle slides along) is bone axis `column`."""
    basis = matrix.to_3x3().normalized()
    forward = basis.col[column].normalized()
    reference = basis.col[(column + 1) % 3]
    side = reference.cross(forward)
    if side.length < 1e-6:
        side = mathutils.Vector((1.0, 0.0, 0.0))
    side.normalize()
    up = forward.cross(side)
    result = mathutils.Matrix((
        (side.x, up.x, forward.x),
        (side.y, up.y, forward.y),
        (side.z, up.z, forward.z),
    )).to_4x4()
    result.translation = matrix @ mathutils.Vector(center)
    return result


def _handle_size(value):
    return min(max(value * 0.35, MINIMUM_HANDLE), MAXIMUM_HANDLE)


class RCP_GGT_collider(bpy.types.GizmoGroup):
    bl_idname = "RCP_GGT_collider"
    bl_label = "碰撞体调整"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT', 'SCALE'}

    @classmethod
    def poll(cls, context):
        return active_collider(context)[1] is not None

    def _make_handle(self, style, color):
        gizmo = self.gizmos.new("GIZMO_GT_arrow_3d")
        gizmo.draw_style = style
        gizmo.color = color
        gizmo.alpha = 0.9
        gizmo.color_highlight = COLOR_HIGHLIGHT
        gizmo.alpha_highlight = 1.0
        gizmo.use_draw_modal = True
        gizmo.line_width = 3.0
        return gizmo

    def setup(self, context):
        self.radius_gizmo = self._make_handle('BOX', COLOR_RADIUS)
        self.end_radius_gizmo = self._make_handle('BOX', COLOR_END_RADIUS)
        self.length_gizmo = self._make_handle('NORMAL', COLOR_LENGTH)

        self.center_gizmo = self.gizmos.new("GIZMO_GT_move_3d")
        self.center_gizmo.draw_options = {'ALIGN_VIEW'}
        self.center_gizmo.color = COLOR_CENTER
        self.center_gizmo.alpha = 0.9
        self.center_gizmo.color_highlight = COLOR_HIGHLIGHT
        self.center_gizmo.alpha_highlight = 1.0

        # Bound once. The handlers read whichever collider is active at call time, so the
        # group never has to rebind and can never point at a stale index.
        self.radius_gizmo.target_set_handler("offset", get=_get_radius, set=_set_radius)
        self.end_radius_gizmo.target_set_handler("offset", get=_get_end_radius,
                                                 set=_set_end_radius)
        self.length_gizmo.target_set_handler("offset", get=_get_half_length,
                                             set=_set_half_length)
        self.center_gizmo.target_set_handler("offset", get=_get_center, set=_set_center)

    def refresh(self, context):
        obj, item = active_collider(context)
        if item is None:
            return

        matrix = _bone_matrix(obj, item)
        center = tuple(item.center)
        is_capsule = item.shape == 'CAPSULE'
        radius = item.start_radius if is_capsule else item.radius

        self.radius_gizmo.matrix_basis = _oriented(matrix, center, 0)
        self.radius_gizmo.scale_basis = _handle_size(radius)
        self.radius_gizmo.hide = item.shape == 'PLANE'

        self.length_gizmo.matrix_basis = _oriented(matrix, center, 1)
        self.length_gizmo.scale_basis = _handle_size(item.length * 0.5)
        self.length_gizmo.hide = not is_capsule

        self.end_radius_gizmo.matrix_basis = _oriented(matrix, center, 2)
        self.end_radius_gizmo.scale_basis = _handle_size(item.end_radius)
        self.end_radius_gizmo.hide = not (is_capsule and item.radius_separation)

        # move_3d draws at matrix_basis + rotation @ offset, and offset IS item.center,
        # so this must stay at the bone origin or the centre gets counted twice.
        basis = matrix.to_3x3().normalized().to_4x4()
        basis.translation = matrix.translation
        self.center_gizmo.matrix_basis = basis
        self.center_gizmo.scale_basis = _handle_size(radius) * 1.2


def _clamped(value):
    return max(float(value), 0.001)


def _get_radius():
    item = _live()
    if item is None:
        return 0.05
    return item.start_radius if item.shape == 'CAPSULE' else item.radius


def _set_radius(value):
    item = _live()
    if item is None:
        return
    if item.shape == 'CAPSULE':
        item.start_radius = _clamped(value)
        if not item.radius_separation:
            item.end_radius = _clamped(value)
    else:
        item.radius = _clamped(value)


def _get_end_radius():
    item = _live()
    return item.end_radius if item is not None else 0.05


def _set_end_radius(value):
    item = _live()
    if item is not None:
        item.end_radius = _clamped(value)


def _get_half_length():
    item = _live()
    return item.length * 0.5 if item is not None else 0.1


def _set_half_length(value):
    item = _live()
    if item is not None:
        item.length = max(float(value) * 2.0, 0.001)


def _get_center():
    item = _live()
    return tuple(item.center) if item is not None else (0.0, 0.0, 0.0)


def _set_center(value):
    item = _live()
    if item is not None:
        item.center = (float(value[0]), float(value[1]), float(value[2]))


_CLASSES = (RCP_GGT_collider,)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
