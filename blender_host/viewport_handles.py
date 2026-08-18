"""The only gizmo code in the add-on: turn Handle records into draggable Blender gizmos.

Rebuilt from the registry every refresh, so any layer can offer handles without touching this file.

Why the old collider gizmos were invisible: the group declared bl_options {'3D','PERSISTENT',
'SCALE'} while feeding scale_basis world-space sizes clamped to 0.008..0.06. With 'SCALE' the gizmo
is drawn at a fixed screen size and scale_basis is a MULTIPLIER of it, so every handle rendered at
under a hundredth of its intended size -- present, pickable in principle, and invisible in practice.
Handles here are world-scaled on purpose (they should grow with the collider they edit), so 'SCALE'
is gone and scale_basis stays in world units.
"""

import bpy

from . import viewport

GIZMO_TYPES = {
    viewport.ARROW: "GIZMO_GT_arrow_3d",
    viewport.MOVE: "GIZMO_GT_move_3d",
    viewport.DIAL: "GIZMO_GT_dial_3d",
}

COLOR_HIGHLIGHT = (1.0, 0.95, 0.55)

_current = {}


def _reader(identifier):
    def read():
        handle = _current.get(identifier)
        return handle.read() if handle is not None else 0.0
    return read


def _writer(identifier):
    def write(value):
        handle = _current.get(identifier)
        if handle is None:
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
        self._signature = None
        self._built = {}

    def _build(self, handles):
        self.gizmos.clear()
        self._built = {}
        for handle in handles:
            gizmo = self.gizmos.new(GIZMO_TYPES[handle.kind])
            if handle.kind == viewport.ARROW:
                gizmo.draw_style = 'BOX'
            elif handle.kind == viewport.MOVE:
                gizmo.draw_options = {'ALIGN_VIEW'}
            gizmo.use_draw_modal = True
            gizmo.line_width = 3.0
            gizmo.alpha = 0.9
            gizmo.color_highlight = COLOR_HIGHLIGHT
            gizmo.alpha_highlight = 1.0
            gizmo.target_set_handler("offset", get=_reader(handle.identifier),
                                     set=_writer(handle.identifier))
            self._built[handle.identifier] = gizmo

    def refresh(self, context):
        handles = viewport.collect_handles(context)
        _current.clear()
        for handle in handles:
            _current[handle.identifier] = handle
        signature = tuple(handle.identifier for handle in handles)
        if signature != getattr(self, "_signature", None):
            self._build(handles)
            self._signature = signature
        for handle in handles:
            gizmo = self._built.get(handle.identifier)
            if gizmo is None:
                continue
            gizmo.matrix_basis = handle.matrix
            gizmo.scale_basis = handle.scale
            gizmo.color = handle.color


_CLASSES = (RCP_GGT_handles,)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    _current.clear()
