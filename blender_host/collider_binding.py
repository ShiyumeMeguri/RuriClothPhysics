import uuid

import bpy

UID_KEY = "ruri_cloth_collider_uid"

WEAK_REFERENCE_REASON = (
    "a configuration names the colliders it hits the same way it names the bones it drives, "
    "by a mark written on the thing it names rather than by a pointer at it, because a "
    "pointer is a dependency and blender carries dependencies wherever it carries the "
    "datablock that holds them: with a pointer on the armature, copying that armature, or "
    "any mesh under it, or appending it into another file, dragged every collider the "
    "configurations referenced along with it, and every capsule end circle behind those, "
    "which on one character was twenty seven objects nobody asked for. The dependency a "
    "collider legitimately has is the one it declares itself, the child of constraint that "
    "makes it follow a bone, and that one points from the collider at the armature, so the "
    "armature stays copyable on its own and a collider copied on its own still knows what "
    "it follows. What a name would not survive is being renamed, which is why the mark is a "
    "generated identifier and the name shown in the list is read back from it")


_mapping = None


def _rebuild():
    global _mapping
    found = {}
    for obj in bpy.data.objects:
        uid = obj.get(UID_KEY)
        if uid and uid not in found:
            found[uid] = obj.name
    _mapping = found
    return found


def _lookup():
    return _mapping if _mapping is not None else _rebuild()


def invalidate():
    global _mapping
    _mapping = None


def _object_carrying(uid, mapping):
    name = mapping.get(uid)
    if name is None:
        return None
    obj = bpy.data.objects.get(name)
    return obj if obj is not None and obj.get(UID_KEY) == uid else None


def resolve(uid):
    if not uid:
        return None
    found = _object_carrying(uid, _lookup())
    if found is not None:
        return found
    invalidate()
    return _object_carrying(uid, _lookup())


def bind(obj):
    if obj is None:
        return ""
    uid = obj.get(UID_KEY)
    if uid:
        for other in bpy.data.objects:
            if other is not obj and other.get(UID_KEY) == uid:
                uid = ""
                break
    if not uid:
        uid = uuid.uuid4().hex
        obj[UID_KEY] = uid
    invalidate()
    return uid


def make_accessors(uid_attribute, adopt):
    def getter(self):
        target = resolve(getattr(self, uid_attribute))
        return target.name if target is not None else ""

    def setter(self, value):
        target = bpy.data.objects.get(value) if value else None
        if target is None:
            setattr(self, uid_attribute, "")
            return
        if not adopt(target):
            return
        setattr(self, uid_attribute, bind(target))

    return getter, setter
