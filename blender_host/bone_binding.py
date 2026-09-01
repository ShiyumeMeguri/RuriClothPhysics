import uuid

UID_KEY = "ruri_cloth_bone_uid"

_cache = {}


def _armature_data(obj):
    if obj is None or getattr(obj, "type", None) != 'ARMATURE':
        return None
    return obj.data


def _mapping(data):
    entry = _cache.get(data.session_uid)
    if entry is not None:
        return entry
    mapping = {}
    for bone in data.bones:
        uid = bone.get(UID_KEY)
        if uid and uid not in mapping:
            mapping[uid] = bone.name
    _cache[data.session_uid] = mapping
    return mapping


def invalidate(data=None):
    if data is None:
        _cache.clear()
    else:
        _cache.pop(data.session_uid, None)


def resolve(obj, uid):
    data = _armature_data(obj)
    if data is None or not uid:
        return ""
    name = _mapping(data).get(uid)
    if name is None:
        return ""
    bone = data.bones.get(name)
    if bone is not None and bone.get(UID_KEY) == uid:
        return name
    invalidate(data)
    return _mapping(data).get(uid, "")


def bind(obj, name):
    data = _armature_data(obj)
    if data is None or not name:
        return ""
    bone = data.bones.get(name)
    if bone is None:
        return ""
    uid = bone.get(UID_KEY)
    if uid:
        for other in data.bones:
            if other.name != name and other.get(UID_KEY) == uid:
                uid = ""
                break
    if not uid:
        uid = uuid.uuid4().hex
        bone[UID_KEY] = uid
    invalidate(data)
    return uid


def make_accessors(uid_attribute, update, target=None):
    resolver = target if target is not None else (lambda self: self.id_data)

    def getter(self):
        return resolve(resolver(self), getattr(self, uid_attribute))

    def setter(self, value):
        setattr(self, uid_attribute, bind(resolver(self), value))
        update(self, None)

    return getter, setter
