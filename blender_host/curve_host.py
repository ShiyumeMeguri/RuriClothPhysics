import uuid

import bpy
import numpy as np

from ..cloth_kernel import defs

TREE_NAME = ".ruri_cloth_physics_curves"

_lut_cache = {}


def _get_tree(create=True):
    tree = bpy.data.node_groups.get(TREE_NAME)
    if tree is None and create:
        tree = bpy.data.node_groups.new(TREE_NAME, "ShaderNodeTree")
        tree.use_fake_user = True
    return tree


def ensure_curve_node(curve_prop):
    if not curve_prop.node_name:
        curve_prop.node_name = "ruri_cloth_curve_" + uuid.uuid4().hex
    tree = _get_tree(create=True)
    node = tree.nodes.get(curve_prop.node_name)
    if node is None:
        node = tree.nodes.new("ShaderNodeFloatCurve")
        node.name = curve_prop.node_name
        node.label = curve_prop.node_name
        mapping = node.mapping
        mapping.use_clip = True
        mapping.clip_min_x = 0.0
        mapping.clip_min_y = 0.0
        mapping.clip_max_x = 1.0
        mapping.clip_max_y = 1.0
        curve = mapping.curves[0]
        curve.points[0].location = (0.0, curve_prop.default_start)
        curve.points[1].location = (1.0, curve_prop.default_end)
        for point in curve.points:
            point.handle_type = 'VECTOR'
        mapping.update()
    return node


def get_curve_node(curve_prop):
    if not curve_prop.node_name:
        return None
    tree = _get_tree(create=False)
    if tree is None:
        return None
    return tree.nodes.get(curve_prop.node_name)


def _points_hash(node):
    curve = node.mapping.curves[0]
    return tuple((p.location[0], p.location[1], p.handle_type) for p in curve.points)


def get_lut(curve_prop):
    value = curve_prop.value
    if not curve_prop.use_curve:
        key = curve_prop.node_name or id(curve_prop)
        entry = _lut_cache.get(key)
        if entry is not None and entry["use"] is False and entry["value"] == value:
            return entry["lut"]
        lut = np.full(defs.CURVE_LUT_SAMPLES, value, dtype=np.float32)
        _lut_cache[key] = {"hash": None, "value": value, "use": False, "lut": lut}
        return lut

    node = ensure_curve_node(curve_prop)
    key = curve_prop.node_name
    points = _points_hash(node)
    entry = _lut_cache.get(key)
    if entry is not None and entry["use"] is True and entry["value"] == value and entry["hash"] == points:
        return entry["lut"]

    mapping = node.mapping
    mapping.update()
    curve = mapping.curves[0]
    lut = np.empty(defs.CURVE_LUT_SAMPLES, dtype=np.float32)
    for i in range(defs.CURVE_LUT_SAMPLES):
        lut[i] = mapping.evaluate(curve, i / 15.0) * value
    _lut_cache[key] = {"hash": points, "value": value, "use": True, "lut": lut}
    return lut


def curve_content_token(curve_prop):
    if not curve_prop.use_curve:
        return (curve_prop.value, False)
    node = get_curve_node(curve_prop)
    if node is None:
        return (curve_prop.value, True)
    return (curve_prop.value, True, _points_hash(node))


def serialize_points(curve_prop):
    node = get_curve_node(curve_prop)
    if node is None:
        return ""
    curve = node.mapping.curves[0]
    parts = []
    for point in curve.points:
        parts.append("%.6g,%.6g,%s" % (point.location[0], point.location[1], point.handle_type))
    return ";".join(parts)


def deserialize_points(curve_prop, text):
    if not text:
        return
    node = ensure_curve_node(curve_prop)
    curve = node.mapping.curves[0]
    entries = []
    for chunk in text.split(";"):
        fields = chunk.split(",")
        if len(fields) < 2:
            continue
        try:
            x = float(fields[0])
            y = float(fields[1])
        except ValueError:
            continue
        handle = fields[2] if len(fields) >= 3 else 'AUTO'
        entries.append((x, y, handle))
    if len(entries) < 2:
        return

    while len(curve.points) > 2:
        curve.points.remove(curve.points[-1])
    while len(curve.points) < len(entries):
        curve.points.new(0.5, 0.5)
    for point, (x, y, handle) in zip(curve.points, entries):
        point.location = (x, y)
        try:
            point.handle_type = handle
        except TypeError:
            point.handle_type = 'AUTO'
    node.mapping.update()
    _lut_cache.pop(curve_prop.node_name, None)


def remove_curve_node(curve_prop):
    node = get_curve_node(curve_prop)
    if node is not None:
        tree = _get_tree(create=False)
        if tree is not None:
            tree.nodes.remove(node)
    _lut_cache.pop(curve_prop.node_name, None)
