import re

import bpy
import mathutils
import numpy as np

from . import armature
from ..cloth_kernel import defs

SIDE_FLIP = {"l": "r", "r": "l", "left": "right", "right": "left"}
_NAME_PARTS = re.compile(r"([._\- ])")

KIND_VALUES = {'SPHERE': defs.COLLIDER_SPHERE, 'CAPSULE': defs.COLLIDER_CAPSULE,
               'PLANE': defs.COLLIDER_PLANE}

DISPLAY_TYPE = {'SPHERE': 'SPHERE', 'CAPSULE': 'CIRCLE', 'PLANE': 'SINGLE_ARROW'}
END_DISPLAY_TYPE = 'CIRCLE'

MINIMUM_RADIUS = 1e-5

COLLECTION_SUFFIX = "Colliders"
BONE_CONSTRAINT = "Ruri Follow Bone"
SHAPE_PREFIX = {'SPHERE': "Sphere", 'CAPSULE': "Capsule", 'PLANE': "Plane"}
CAPSULE_START = "01"
CAPSULE_END = "02"

FOLLOW_NONE = 'NONE'
FOLLOW_BONE = 'BONE'


def _match_case(sample, word):
    if sample.isupper():
        return word.upper()
    if sample[:1].isupper():
        return word.capitalize()
    return word


def flip_side_name(name):
    parts = _NAME_PARTS.split(name)
    flipped = False
    for index, part in enumerate(parts):
        word = SIDE_FLIP.get(part.lower())
        if word is None:
            continue
        parts[index] = _match_case(part, word)
        flipped = True
    return "".join(parts) if flipped else ""


def _stem(shape, base):
    return "%s.%s" % (SHAPE_PREFIX[shape], base) if base else SHAPE_PREFIX[shape]


def collider_name(shape, base):
    stem = _stem(shape, base)
    return "%s.%s" % (stem, CAPSULE_START) if shape == 'CAPSULE' else stem


def end_name(base):
    return "%s.%s" % (_stem('CAPSULE', base), CAPSULE_END)


def base_name(obj):
    name = obj.name
    for tail in ("." + CAPSULE_START, "." + CAPSULE_END):
        if name.endswith(tail):
            name = name[:-len(tail)]
            break
    for prefix in SHAPE_PREFIX.values():
        head = prefix + "."
        if name.startswith(head):
            return name[len(head):]
    return name


def settings_of(obj):
    if obj is None:
        return None
    return getattr(obj, "ruri_cloth_physics_collider", None)


def is_collider(obj):
    settings = settings_of(obj)
    return settings is not None and settings.is_collider


def end_object(settings):
    if settings.shape != 'CAPSULE':
        return None
    end = settings.end_object
    if end is None or end.type != 'EMPTY':
        return None
    return end


def owner_of(scene, obj):
    if scene is None or obj is None:
        return None
    for candidate in scene.objects:
        settings = settings_of(candidate)
        if settings is None or not settings.is_collider:
            continue
        if settings.end_object is obj:
            return candidate
    return None


def sync_display(obj):
    settings = settings_of(obj)
    obj.empty_display_type = DISPLAY_TYPE[settings.shape]
    end = end_object(settings)
    if end is not None:
        end.empty_display_type = END_DISPLAY_TYPE


def collection_name(armature_object):
    return "%s.%s" % (armature_object.name, COLLECTION_SUFFIX)


def collection_for(scene, armature_object, create=False):
    name = collection_name(armature_object)
    for collection in scene.collection.children_recursive:
        if collection.name == name:
            return collection
    for obj in scene.objects:
        constraint = bone_constraint(obj) if is_collider(obj) else None
        if constraint is not None and constraint.target is armature_object and obj.users_collection:
            return obj.users_collection[0]
    if not create:
        return None
    collection = bpy.data.collections.new(name)
    scene.collection.children.link(collection)
    return collection


def layer_collection_for(view_layer, collection):
    if collection is None:
        return None

    def walk(layer):
        if layer.collection is collection:
            return layer
        for child in layer.children:
            found = walk(child)
            if found is not None:
                return found
        return None

    return walk(view_layer.layer_collection)


def owned_colliders(scene, armature_object):
    collection = collection_for(scene, armature_object)
    if collection is None:
        return []
    return sorted((obj for obj in collection.objects if is_collider(obj)),
                  key=lambda obj: obj.name)


def new_empty(collection, name, display_type, display_size):
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = display_type
    empty.empty_display_size = max(float(display_size), MINIMUM_RADIUS)
    collection.objects.link(empty)
    return empty


def bone_constraint(obj):
    found = obj.constraints.get(BONE_CONSTRAINT)
    if found is not None:
        return found
    for constraint in obj.constraints:
        if constraint.type == 'CHILD_OF' and constraint.target is not None:
            return constraint
    return None


def ensure_bone_constraint(obj):
    constraint = bone_constraint(obj)
    if constraint is None:
        constraint = obj.constraints.new('CHILD_OF')
    constraint.name = BONE_CONSTRAINT
    return constraint


def detach(obj):
    constraint = bone_constraint(obj)
    if constraint is not None:
        obj.constraints.remove(constraint)


def bone_world(armature_object, bone_name):
    pose_bone = armature_object.pose.bones.get(bone_name)
    if pose_bone is None:
        return None
    return armature_object.matrix_world @ pose_bone.matrix


def attach_to_bone(view_layer, obj, armature_object, bone_name):
    view_layer.update()
    world = obj.matrix_world.copy()
    constraint = ensure_bone_constraint(obj)
    constraint.target = armature_object
    constraint.subtarget = bone_name
    view_layer.update()
    space = bone_world(armature_object, bone_name)
    constraint.inverse_matrix = space.inverted_safe() if space is not None \
        else mathutils.Matrix.Identity(4)
    obj.matrix_basis = world
    view_layer.update()


def set_world(view_layer, obj, matrix):
    view_layer.update()
    constraint = bone_constraint(obj)
    space = None if constraint is None or constraint.target is None \
        else bone_world(constraint.target, constraint.subtarget)
    if space is None:
        obj.matrix_basis = matrix
    else:
        obj.matrix_basis = (space @ constraint.inverse_matrix).inverted_safe() @ matrix
    view_layer.update()


def follow_reference(obj, armature_object):
    constraint = bone_constraint(obj)
    if constraint is not None and constraint.target is armature_object and constraint.subtarget:
        return {"kind": FOLLOW_BONE, "bone": constraint.subtarget}
    return {"kind": FOLLOW_NONE, "bone": ""}


def follow_space(armature_object, reference):
    if reference.get("kind", FOLLOW_NONE) == FOLLOW_BONE:
        space = bone_world(armature_object, reference.get("bone", ""))
        if space is not None:
            return space
    return armature_object.matrix_world.copy()


def _serialize_empty(obj, armature_object):
    reference = follow_reference(obj, armature_object)
    space = follow_space(armature_object, reference)
    return {"name": obj.name,
            "display_size": float(obj.empty_display_size),
            "follow": reference,
            "matrix": [list(row) for row in space.inverted_safe() @ obj.matrix_world]}


def serialize(obj, armature_object):
    settings = settings_of(obj)
    data = _serialize_empty(obj, armature_object)
    data["shape"] = settings.shape
    data["enabled"] = bool(settings.enabled)
    end = end_object(settings)
    if end is not None:
        data["end"] = _serialize_empty(end, armature_object)
    return data


def _place(obj, armature_object, data, view_layer):
    reference = data.get("follow") or {}
    space = follow_space(armature_object, reference)
    missing = []
    if reference.get("kind", FOLLOW_NONE) == FOLLOW_BONE:
        bone_name = reference.get("bone", "")
        if bone_name in armature_object.pose.bones:
            constraint = ensure_bone_constraint(obj)
            constraint.target = armature_object
            constraint.subtarget = bone_name
            constraint.inverse_matrix = space.inverted_safe()
        else:
            missing.append(bone_name)
    obj.matrix_basis = space @ mathutils.Matrix(data["matrix"])
    view_layer.update()
    return missing


def deserialize(data, armature_object, collection, view_layer):
    shape = data.get("shape", 'SPHERE')
    empty = new_empty(collection, data.get("name", collider_name(shape, "")), DISPLAY_TYPE[shape],
                      data.get("display_size", 0.05))
    settings = settings_of(empty)
    settings.is_collider = True
    settings.shape = shape
    settings.enabled = bool(data.get("enabled", True))
    missing = _place(empty, armature_object, data, view_layer)
    end_data = data.get("end")
    if end_data is not None:
        end = new_empty(collection, end_data.get("name", end_name(base_name(empty))),
                        END_DISPLAY_TYPE, end_data.get("display_size", 0.05))
        missing.extend(_place(end, armature_object, end_data, view_layer))
        settings.end_object = end
    return empty, missing


def _evaluated(obj, depsgraph):
    return obj.evaluated_get(depsgraph) if depsgraph is not None else obj


def evaluated_objects(depsgraph):
    if depsgraph is None:
        return None
    return {block.original.as_pointer() for block in depsgraph.ids
            if isinstance(block, bpy.types.Object)}


def is_live(obj, depsgraph, evaluated=None):
    if depsgraph is None:
        return True
    if evaluated is None:
        evaluated = evaluated_objects(depsgraph)
    return obj.as_pointer() in evaluated


def _circle(obj, depsgraph):
    source = _evaluated(obj, depsgraph)
    matrix = armature.read_matrix(source.matrix_world)
    radius = float(source.empty_display_size) * float(armature.matrix_scale(matrix).max())
    return matrix, radius


def solve(obj, settings, depsgraph, evaluated=None):
    matrix, radius = _circle(obj, depsgraph)
    position = matrix[:3, 3].astype(np.float32)
    rotation = armature.matrix_to_quat(matrix).astype(np.float32)
    kind = KIND_VALUES[settings.shape]
    if evaluated is None:
        evaluated = evaluated_objects(depsgraph)
    live = settings.enabled and is_live(obj, depsgraph, evaluated)
    if kind == defs.COLLIDER_PLANE:
        return (kind, position, rotation, position,
                np.array([radius, radius], dtype=np.float32), live)

    end = end_object(settings)
    if end is None:
        tip = position
        end_radius = radius
    else:
        end_matrix, end_radius = _circle(end, depsgraph)
        tip = end_matrix[:3, 3].astype(np.float32)
        live = live and is_live(end, depsgraph, evaluated)
    radii = np.array([radius, end_radius], dtype=np.float32)
    live = live and radius >= MINIMUM_RADIUS and end_radius >= MINIMUM_RADIUS
    return kind, position, rotation, tip, radii, live


def gather(objects, depsgraph, evaluated, cache):
    count = len(objects)
    positions = np.zeros((count, 3), dtype=np.float32)
    rotations = np.zeros((count, 4), dtype=np.float32)
    rotations[:, 3] = 1.0
    tips = np.zeros((count, 3), dtype=np.float32)
    radii = np.zeros((count, 2), dtype=np.float32)
    enabled = np.zeros(count, dtype=bool)
    for index, obj in enumerate(objects):
        key = obj.as_pointer()
        resolved = cache.get(key)
        if resolved is None:
            resolved = solve(obj, settings_of(obj), depsgraph, evaluated)
            cache[key] = resolved
        _kind, position, rotation, tip, radius, live = resolved
        positions[index] = position
        rotations[index] = rotation
        tips[index] = tip
        radii[index] = radius
        enabled[index] = live
    return positions, rotations, tips, radii, enabled
