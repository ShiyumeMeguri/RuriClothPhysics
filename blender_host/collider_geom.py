import re
import zlib

import bpy
import mathutils
import numpy as np

from . import armature
from ..cloth_kernel import defs

SIDE_FLIP = {"l": "r", "r": "l", "left": "right", "right": "left"}
_NAME_PARTS = re.compile(r"([._\- ])")

KIND_VALUES = {'SPHERE': defs.COLLIDER_SPHERE, 'CAPSULE': defs.COLLIDER_CAPSULE,
               'PLANE': defs.COLLIDER_PLANE, 'MESH': defs.COLLIDER_MESH}

DISPLAY_TYPE = {'SPHERE': 'SPHERE', 'CAPSULE': 'CIRCLE', 'PLANE': 'SINGLE_ARROW',
                'MESH': 'CUBE'}
END_DISPLAY_TYPE = 'CIRCLE'

MINIMUM_RADIUS = 1e-5

MESH_OBJECT_TYPE = 'MESH'

EMPTY_OBJECT_TYPE = 'EMPTY'

MESH_COLLIDER_SOURCE_REASON = (
    "a mesh collider is the mesh object itself rather than an empty pointing at one, "
    "because everything the collider layer already does with a collider, following a bone, "
    "being saved into a preset, being switched off, is done through the object it is, and a "
    "second object in between would give the same collider two poses that can disagree; the "
    "triangles are read from the evaluated object once when the binding is built and are "
    "stored in the frame of the collider with the world scale of that moment folded in, so "
    "the pose the frame interpolates stays a rotation and a position, which is what the "
    "distance the collider reports is defined against")

COLLECTION_SUFFIX = "Colliders"
BONE_CONSTRAINT = "Ruri Follow Bone"
SHAPE_PREFIX = {'SPHERE': "Sphere", 'CAPSULE': "Capsule", 'PLANE': "Plane", 'MESH': "Mesh"}
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


def is_mesh_collider(obj):
    settings = settings_of(obj)
    return settings is not None and settings.is_collider and settings.shape == 'MESH'


COLLIDER_OBJECT_TYPES = (EMPTY_OBJECT_TYPE, MESH_OBJECT_TYPE)

SHAPE_FOR_OBJECT_REASON = (
    "the shape of a collider and the type of the object carrying it are one fact and not "
    "two: the mesh shape is the mesh object itself, and every other shape is an empty whose "
    "display size is its radius, so a mesh object on the sphere shape would ask for a "
    "display size a mesh has not got and an empty on the mesh shape would ask for triangles "
    "an empty has not got; the panel therefore offers each object only the shapes it can "
    "be, and this is the one place that coerces a shape which arrived from somewhere the "
    "panel does not control, a script, a preset, or a file saved before the mesh shape "
    "existed")


def shape_for_object(obj, shape):
    if obj is None:
        return shape
    if obj.type == MESH_OBJECT_TYPE:
        return 'MESH'
    if shape == 'MESH':
        return 'SPHERE'
    return shape


def can_be_collider(obj):
    return obj is not None and obj.type in COLLIDER_OBJECT_TYPES


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
    if obj.type == EMPTY_OBJECT_TYPE:
        obj.empty_display_type = DISPLAY_TYPE[settings.shape]
    end = end_object(settings)
    if end is not None:
        end.empty_display_type = END_DISPLAY_TYPE


MARK_SINGLE_PLACE_REASON = (
    "telling an object it is a collider is three facts that have to land together, the mark "
    "itself, a shape the object can actually carry and a display that shows that shape, and "
    "two buttons do it, the convert operator and picking an object in a configuration's "
    "list; written twice they drift, so they are written here and both call it")


def mark(obj):
    settings = settings_of(obj)
    if settings is None:
        return False
    settings.is_collider = True
    settings.shape = shape_for_object(obj, settings.shape)
    sync_display(obj)
    return True


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


GLOBAL_COLLIDER_REASON = (
    "a wall is not a property of one piece of cloth, so it is stated once on the collider "
    "itself rather than as a row in every configuration that must not pass through it: the "
    "alternative is a list that has to be edited again for every configuration anyone adds "
    "later, and a wall silently missing from the one nobody remembered. The order is by "
    "name and not by whatever order the scene happens to hold its objects in, because this "
    "list is concatenated onto each configuration's own references to lay out the collider "
    "block, and a block whose rows move around between two frames is a collider swapping "
    "identity underneath the solver's stored history")


def global_colliders(scene):
    found = []
    for obj in scene.objects:
        settings = settings_of(obj)
        if settings is not None and settings.is_collider and settings.is_global:
            found.append((obj, settings))
    found.sort(key=lambda pair: pair[0].name)
    return found


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


EVALUATED_OBJECT_SOURCE_REASON = (
    "what this answers is whether an object is still in the dependency graph, so it is read "
    "off the graph's objects and not off every datablock it holds: the graph carries meshes, "
    "materials, images and node groups beside the objects, and asking the whole pile means "
    "a type test on each one to throw most of them away -- on this file 598 datablocks "
    "tested to keep 136 objects, four times the work for the same set")


def evaluated_objects(depsgraph):
    if depsgraph is None:
        return None
    return {block.original.as_pointer() for block in depsgraph.objects}


def is_live(obj, depsgraph, evaluated=None):
    if depsgraph is None:
        return True
    if evaluated is None:
        evaluated = evaluated_objects(depsgraph)
    return obj.as_pointer() in evaluated


WORLD_OVERRIDE_REASON = (
    "a collider that follows a bone the solver drives is placed from the animation pose the "
    "host already computed rather than read back off the dependency graph, because reading "
    "it back is what forces the armature to be wiped and evaluated a second time every "
    "frame: the collider has to see the body where the animation puts it and not where the "
    "last solve left the cloth, or it chases the cloth it is meant to stop. The pose is the "
    "same one either way -- the host's animation world matrix for a bone was measured equal "
    "to the graph's to the bit -- so the override changes where the answer comes from and "
    "not what it is. Only the pose is overridden; the display size a radius is read from is "
    "still the evaluated object's, because that is what a driver or an animation on the "
    "collider itself would change")


def _circle(obj, depsgraph, world_override=None):
    source = _evaluated(obj, depsgraph)
    matrix = None
    if world_override is not None:
        matrix = world_override.get(obj.as_pointer())
    if matrix is None:
        matrix = armature.read_matrix(source.matrix_world)
    radius = float(source.empty_display_size) * float(armature.matrix_scale(matrix).max())
    return matrix, radius


def mesh_geometry(obj, depsgraph):
    source = _evaluated(obj, depsgraph)
    if source.type != MESH_OBJECT_TYPE:
        raise ValueError(
            "%s\nthe collider %s is set to the mesh shape and it is a %s object, a mesh "
            "collider is the mesh object itself"
            % (MESH_COLLIDER_SOURCE_REASON, obj.name, source.type))
    matrix = armature.read_matrix(source.matrix_world)
    _rotation, magnitude, reflected = armature.decompose_component_basis(matrix)
    mesh = source.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertex_count = len(mesh.vertices)
        triangle_count = len(mesh.loop_triangles)
        coordinates = np.zeros(vertex_count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coordinates)
        corners = np.zeros(triangle_count * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", corners)
    finally:
        source.to_mesh_clear()
    sign = -1.0 if reflected else 1.0
    vertices = coordinates.reshape(vertex_count, 3) * magnitude.astype(np.float32) * sign
    triangles = corners.reshape(triangle_count, 3)
    if reflected:
        triangles = triangles[:, (0, 2, 1)]
    return (np.ascontiguousarray(vertices, dtype=np.float32),
            np.ascontiguousarray(triangles, dtype=np.int32))


MESH_COLLIDER_EARLY_REFUSAL_REASON = (
    "the triangles a mesh collider is asked for are checked at the button that marks the "
    "object rather than only at the frame that builds the world, because the build runs "
    "inside a frame change handler where the only place a refusal can land is the system "
    "console, and a surface with no consistent outward side is a property of the object that "
    "is knowable the moment it is picked; the check itself is the kernel's, called rather than "
    "restated, so a mesh the button accepts is exactly a mesh the world accepts")


MESH_COLLIDER_INWARD_NOTICE_REASON = (
    "a closed surface wound outward side in is taken rather than refused, so the one place a "
    "person can be told about it is the moment they made the object a collider: the world "
    "cannot say it, because the world is rebuilt inside a frame change handler and would say "
    "it again on every rebuild into a console nobody is reading, and the engine cannot say "
    "it, because the field it reads carries whichever sign the triangles were wound to carry "
    "and no column anywhere records which way that was, which is the whole reason the thing "
    "is silent; the notice is therefore read here, once, beside the refusal and off the same "
    "triangles and the same repaired winding, and it is the kernel's own reading called "
    "rather than restated")


def mesh_collider_reading(obj, depsgraph):
    from ..cloth_kernel import world as kernel_world
    try:
        vertices, faces = mesh_geometry(obj, depsgraph)
        held = kernel_world.assert_collider_mesh_is_orientable(obj.name, vertices, faces)
    except (kernel_world.ColliderMeshRefused, AssertionError, ValueError) as refusal:
        return str(refusal), None
    return None, kernel_world.collider_mesh_inward_notice(obj.name, held[0], held[1])


MESH_TOPOLOGY_TOKEN_REASON = (
    "what a mesh collider has to be rebound for is its topology, because the corners of "
    "every triangle, the pairing of the half edges and the size of the block the world "
    "reserves are all derived from it once and stored, while the vertices are carried on "
    "every frame; the token this returns therefore reads the corners and never a "
    "coordinate, so a body that bends, is skinned or carries a shape key does not rebuild "
    "anything, and it reads all of them rather than only how many there are, because "
    "retriangulating a surface into the same number of faces changes every corner and "
    "leaves both counts where they were")


def mesh_topology_token(vertices, triangles):
    return (int(vertices.shape[0]), int(triangles.shape[0]),
            zlib.crc32(triangles.tobytes()))


def solve(obj, settings, depsgraph, evaluated=None, world_override=None):
    kind = KIND_VALUES[settings.shape]
    if evaluated is None:
        evaluated = evaluated_objects(depsgraph)
    if kind == defs.COLLIDER_MESH:
        matrix = None
        if world_override is not None:
            matrix = world_override.get(obj.as_pointer())
        if matrix is None:
            matrix = armature.read_matrix(_evaluated(obj, depsgraph).matrix_world)
        position = matrix[:3, 3].astype(np.float32)
        rotation = armature.matrix_to_quat(matrix).astype(np.float32)
        return (kind, position, rotation, position, np.zeros(2, dtype=np.float32),
                settings.enabled and is_live(obj, depsgraph, evaluated))
    matrix, radius = _circle(obj, depsgraph, world_override)
    position = matrix[:3, 3].astype(np.float32)
    rotation = armature.matrix_to_quat(matrix).astype(np.float32)
    live = settings.enabled and is_live(obj, depsgraph, evaluated)
    if kind == defs.COLLIDER_PLANE:
        return (kind, position, rotation, position,
                np.array([radius, radius], dtype=np.float32), live)

    end = end_object(settings)
    if end is None:
        tip = position
        end_radius = radius
    else:
        end_matrix, end_radius = _circle(end, depsgraph, world_override)
        tip = end_matrix[:3, 3].astype(np.float32)
        live = live and is_live(end, depsgraph, evaluated)
    radii = np.array([radius, end_radius], dtype=np.float32)
    live = live and radius >= MINIMUM_RADIUS and end_radius >= MINIMUM_RADIUS
    return kind, position, rotation, tip, radii, live


def gather(objects, depsgraph, evaluated, cache, world_override=None):
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
            resolved = solve(obj, settings_of(obj), depsgraph, evaluated, world_override)
            cache[key] = resolved
        _kind, position, rotation, tip, radius, live = resolved
        positions[index] = position
        rotations[index] = rotation
        tips[index] = tip
        radii[index] = radius
        enabled[index] = live
    return positions, rotations, tips, radii, enabled
