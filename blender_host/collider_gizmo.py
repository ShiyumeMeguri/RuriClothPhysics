import bpy
import mathutils

COLOR_RADIUS = (1.0, 0.85, 0.1)
COLOR_LENGTH = (0.2, 0.9, 0.4)
COLOR_CENTER = (0.15, 0.75, 1.0)


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


def _bone_matrix(obj, item):
    if item.bone:
        pose_bone = obj.pose.bones.get(item.bone)
        if pose_bone is not None:
            return obj.matrix_world @ pose_bone.matrix
    return obj.matrix_world.copy()


def _oriented(matrix, center, column):
    """Frame whose +Z (the axis an arrow gizmo extends along) is bone axis `column`."""
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


class RCP_GGT_collider(bpy.types.GizmoGroup):
    bl_idname = "RCP_GGT_collider"
    bl_label = "碰撞体调整"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT', 'SCALE'}

    @classmethod
    def poll(cls, context):
        obj, item = active_collider(context)
        return item is not None

    def setup(self, context):
        def arrow(color):
            gizmo = self.gizmos.new("GIZMO_GT_arrow_3d")
            gizmo.draw_style = 'NORMAL'
            gizmo.color = color
            gizmo.alpha = 0.8
            gizmo.color_highlight = (1.0, 1.0, 1.0)
            gizmo.alpha_highlight = 1.0
            gizmo.use_draw_modal = True
            return gizmo

        self.radius_gizmo = arrow(COLOR_RADIUS)
        self.end_radius_gizmo = arrow(COLOR_RADIUS)
        self.length_gizmo = arrow(COLOR_LENGTH)

        self.center_gizmo = self.gizmos.new("GIZMO_GT_move_3d")
        self.center_gizmo.draw_options = {'ALIGN_VIEW'}
        self.center_gizmo.color = COLOR_CENTER
        self.center_gizmo.alpha = 0.8
        self.center_gizmo.color_highlight = (1.0, 1.0, 1.0)
        self.center_gizmo.alpha_highlight = 1.0
        self.center_gizmo.scale_basis = 0.12

        self._bind(context)

    def _bind(self, context):
        obj, item = active_collider(context)
        if item is None:
            return
        settings = obj.ruri_cloth_physics
        index = settings.active_collider_index

        def current():
            pool = obj.ruri_cloth_physics.colliders
            if 0 <= index < len(pool):
                return pool[index]
            return None

        def bounded(value):
            return max(float(value), 0.001)

        def make(attribute):
            def get():
                target = current()
                return getattr(target, attribute) if target is not None else 0.05

            def put(value):
                target = current()
                if target is not None:
                    setattr(target, attribute, bounded(value))
            return get, put

        end_get, end_put = make("end_radius")

        def capsule_radius_get():
            target = current()
            if target is None:
                return 0.05
            return target.start_radius if target.shape == 'CAPSULE' else target.radius

        def capsule_radius_put(value):
            target = current()
            if target is None:
                return
            if target.shape == 'CAPSULE':
                target.start_radius = bounded(value)
                if not target.radius_separation:
                    target.end_radius = bounded(value)
            else:
                target.radius = bounded(value)

        self.radius_gizmo.target_set_handler("offset", get=capsule_radius_get,
                                             set=capsule_radius_put)
        self.end_radius_gizmo.target_set_handler("offset", get=end_get, set=end_put)

        def half_length_get():
            target = current()
            return target.length * 0.5 if target is not None else 0.1

        def half_length_put(value):
            target = current()
            if target is not None:
                target.length = max(float(value) * 2.0, 0.001)

        self.length_gizmo.target_set_handler("offset", get=half_length_get, set=half_length_put)

        def center_get():
            target = current()
            return tuple(target.center) if target is not None else (0.0, 0.0, 0.0)

        def center_put(value):
            target = current()
            if target is not None:
                target.center = (float(value[0]), float(value[1]), float(value[2]))

        self.center_gizmo.target_set_handler("offset", get=center_get, set=center_put)

    def refresh(self, context):
        obj, item = active_collider(context)
        if item is None:
            return
        self._bind(context)

        matrix = _bone_matrix(obj, item)
        center = tuple(item.center)
        is_capsule = item.shape == 'CAPSULE'
        radius = item.start_radius if is_capsule else item.radius
        handle = min(max(radius * 0.8, 0.01), 0.12)

        self.radius_gizmo.matrix_basis = _oriented(matrix, center, 0)
        self.radius_gizmo.scale_basis = handle
        self.radius_gizmo.hide = item.shape == 'PLANE'

        self.length_gizmo.matrix_basis = _oriented(matrix, center, 1)
        self.length_gizmo.scale_basis = min(max(item.length * 0.2, 0.01), 0.12)
        self.length_gizmo.hide = not is_capsule

        self.end_radius_gizmo.matrix_basis = _oriented(matrix, center, 2)
        self.end_radius_gizmo.scale_basis = min(max(item.end_radius * 0.8, 0.01), 0.12)
        self.end_radius_gizmo.hide = not (is_capsule and item.radius_separation)

        # move_3d draws at matrix_basis + rotation @ offset, and offset IS item.center,
        # so this must stay at the bone origin or the centre gets counted twice.
        basis = matrix.to_3x3().normalized().to_4x4()
        basis.translation = matrix.translation
        self.center_gizmo.matrix_basis = basis
        self.center_gizmo.scale_basis = handle * 0.8


_CLASSES = (RCP_GGT_collider,)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
