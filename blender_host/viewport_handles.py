import bpy

from . import viewport

GIZMO_TYPES = {
    viewport.ARROW: "GIZMO_GT_arrow_3d",
}

POOL_SIZES = {
    viewport.ARROW: 16,
}

COLOR_HIGHLIGHT = (1.0, 0.95, 0.55)


def _slot_key(kind, index):
    return "%s:%d" % (kind, index)


_live_slots = {}


def _reader(slot):
    def read():
        handle = _live_slots.get(slot)
        if handle is None or handle.read is None:
            return 0.0
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
                gizmo.draw_style = 'BOX'
                gizmo.use_draw_modal = True
                gizmo.line_width = 3.0
                gizmo.alpha = 0.9
                gizmo.color_highlight = COLOR_HIGHLIGHT
                gizmo.alpha_highlight = 1.0
                gizmo.hide = True
                gizmo.target_set_handler("offset", get=_reader(slot), set=_writer(slot))
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


def register():
    bpy.utils.register_class(RCP_GGT_handles)


def unregister():
    bpy.utils.unregister_class(RCP_GGT_handles)
    _live_slots.clear()
