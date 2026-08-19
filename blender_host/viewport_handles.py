import bpy
from bpy.props import StringProperty

from . import viewport

GIZMO_TYPES = {
    viewport.ARROW: "GIZMO_GT_arrow_3d",
    viewport.MOVE: "GIZMO_GT_move_3d",
    viewport.DIAL: "GIZMO_GT_dial_3d",
    viewport.PICK: "GIZMO_GT_move_3d",
}

POOL_SIZES = {
    viewport.ARROW: 16,
    viewport.MOVE: 4,
    viewport.DIAL: 4,
    viewport.PICK: 128,
}

COLOR_HIGHLIGHT = (1.0, 0.95, 0.55)


def _slot_key(kind, index):
    return "%s:%d" % (kind, index)


class RCP_OT_viewport_handle(bpy.types.Operator):
    """Dispatcher for click-style handles.

    A pool slot is bound to this once and forever; the Handle occupying the slot right now decides
    which operator actually runs, so the kernel never names a domain operator.
    """

    bl_idname = "ruri_cloth_physics.viewport_handle"
    bl_label = "视口手柄"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    slot: StringProperty(default="")

    def execute(self, context):
        handle = _live_slots.get(self.slot)
        if handle is None or handle.operator is None:
            return {'CANCELLED'}
        module, _, name = handle.operator.partition(".")
        target = getattr(getattr(bpy.ops, module, None), name, None)
        if target is None:
            return {'CANCELLED'}
        return target(**handle.properties)


_live_slots = {}


VECTOR_KINDS = {viewport.MOVE, viewport.PICK}


def _reader(slot, kind):
    idle = (0.0, 0.0, 0.0) if kind in VECTOR_KINDS else 0.0

    def read():
        handle = _live_slots.get(slot)
        if handle is None or handle.read is None:
            return idle
        return handle.read()
    return read


def _writer(slot):
    def write(value):
        handle = _live_slots.get(slot)
        if handle is None or handle.write is None:
            return
        if handle.minimum is not None:
            try:
                value = max(float(value), handle.minimum)
            except TypeError:
                pass
        handle.write(value)
    return write


class RCP_GGT_handles(bpy.types.GizmoGroup):
    bl_idname = "RCP_GGT_handles"
    bl_label = "Ruri 布料物理控制器"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}

    @classmethod
    def poll(cls, context):
        return bool(viewport.collect_handles(context))

    def setup(self, context):
        self._pool = {}
        for kind, count in POOL_SIZES.items():
            gizmos = []
            for index in range(count):
                slot = _slot_key(kind, index)
                gizmo = self.gizmos.new(GIZMO_TYPES[kind])
                if kind == viewport.ARROW:
                    gizmo.draw_style = 'BOX'
                elif kind in {viewport.MOVE, viewport.PICK}:
                    gizmo.draw_options = {'ALIGN_VIEW'}
                gizmo.use_draw_modal = True
                gizmo.line_width = 3.0
                gizmo.alpha = 0.9
                gizmo.color_highlight = COLOR_HIGHLIGHT
                gizmo.alpha_highlight = 1.0
                gizmo.hide = True
                if kind == viewport.PICK:
                    properties = gizmo.target_set_operator(RCP_OT_viewport_handle.bl_idname)
                    properties.slot = slot
                else:
                    gizmo.target_set_handler("offset", get=_reader(slot, kind), set=_writer(slot))
                gizmos.append(gizmo)
            self._pool[kind] = gizmos

    def refresh(self, context):
        self._sync(context)

    def draw_prepare(self, context):
        self._sync(context)

    def _sync(self, context):
        handles = viewport.collect_handles(context)
        _live_slots.clear()
        used = dict.fromkeys(POOL_SIZES, 0)
        overflow = 0
        for handle in handles:
            gizmos = self._pool.get(handle.kind)
            if gizmos is None:
                continue
            index = used[handle.kind]
            if index >= len(gizmos):
                overflow += 1
                continue
            slot = _slot_key(handle.kind, index)
            _live_slots[slot] = handle
            gizmo = gizmos[index]
            gizmo.hide = False
            gizmo.matrix_basis = handle.matrix
            gizmo.scale_basis = handle.scale
            gizmo.color = handle.color
            used[handle.kind] = index + 1
        for kind, gizmos in self._pool.items():
            for gizmo in gizmos[used[kind]:]:
                gizmo.hide = True
        if overflow:
            print("RuriClothPhysics: %d viewport handles exceeded the gizmo pool" % overflow)


_CLASSES = (RCP_OT_viewport_handle, RCP_GGT_handles)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    _live_slots.clear()
