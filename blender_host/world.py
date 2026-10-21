ARMATURE_TYPE = 'ARMATURE'

WORLD_REGISTRY_REASON = (
    "who takes part in a scene's simulation is a fact the scene states, not one the add-on "
    "finds out: the scene keeps three explicit lists -- the bodies it simulates, the "
    "colliders every cloth in it collides with, the wind zones blowing through it -- and a "
    "configuration names the colliders it hits by pointing at them; the frame, the overlays "
    "and the panels only ever walk outward from these lists, so nothing anywhere enumerates "
    "the scene, an object nobody registered is never read, and a scene with empty lists "
    "costs nothing at all. An entry whose object was deleted reads as no object and is "
    "skipped where it is read and dropped where the lists are written. A configuration "
    "names a collider by the uid written on that collider and the scene's pool resolves "
    "the uid, so nothing on the rig points at a collider object and copying or appending "
    "the rig never carries one with it")


def of(scene):
    return scene.ruri_cloth_physics if scene is not None else None


def _present(entries):
    for entry in entries:
        obj = entry.object
        if obj is not None:
            yield entry, obj


def _is_body(obj):
    return obj.type == ARMATURE_TYPE and len(obj.ruri_cloth_physics.configs) > 0


def bodies(scene):
    world = of(scene)
    if world is None:
        return ()
    return tuple(obj for _entry, obj in _present(world.bodies) if _is_body(obj))


def live_bodies(scene):
    world = of(scene)
    if world is None:
        return ()
    found = [obj for entry, obj in _present(world.bodies) if entry.live and _is_body(obj)]
    found.sort(key=lambda obj: obj.name)
    return found


def body_entry(scene, obj):
    world = of(scene)
    if world is None or obj is None:
        return None
    for entry in world.bodies:
        if entry.object == obj:
            return entry
    return None


def _drop_vacant(entries):
    for index in range(len(entries) - 1, -1, -1):
        if entries[index].object is None:
            entries.remove(index)


def _enroll(entries, obj):
    for entry in entries:
        if entry.object == obj:
            return entry
    _drop_vacant(entries)
    entry = entries.add()
    entry.object = obj
    return entry


def _withdraw(entries, obj):
    removed = False
    for index in range(len(entries) - 1, -1, -1):
        if entries[index].object == obj:
            entries.remove(index)
            removed = True
    _drop_vacant(entries)
    return removed


def enroll_body(scene, obj):
    return _enroll(of(scene).bodies, obj)


def withdraw_body(scene, obj):
    return _withdraw(of(scene).bodies, obj)


def scene_colliders(scene):
    world = of(scene)
    if world is None:
        return ()
    found = [(obj, obj.ruri_cloth_physics_collider) for _entry, obj in _present(world.colliders)
             if obj.ruri_cloth_physics_collider.is_collider]
    found.sort(key=lambda pair: pair[0].name)
    return found


def has_collider(scene, obj):
    world = of(scene)
    return world is not None and obj is not None \
        and any(entry.object == obj for entry in world.colliders)


def enroll_collider(scene, obj):
    return _enroll(of(scene).colliders, obj)


def withdraw_collider(scene, obj):
    return _withdraw(of(scene).colliders, obj)


def winds(scene):
    world = of(scene)
    if world is None:
        return ()
    return tuple(obj for _entry, obj in _present(world.winds))


def has_wind(scene, obj):
    world = of(scene)
    return world is not None and obj is not None \
        and any(entry.object == obj for entry in world.winds)


def enroll_wind(scene, obj):
    return _enroll(of(scene).winds, obj)


def withdraw_wind(scene, obj):
    return _withdraw(of(scene).winds, obj)


def pool_targets(scene):
    world = of(scene)
    if world is None:
        return ()
    return tuple(obj for _entry, obj in _present(world.pool))


def pool_index(scene):
    index = {}
    for obj in pool_targets(scene):
        uid = obj.ruri_cloth_physics_collider.uid
        if uid and uid not in index:
            index[uid] = obj
    return index


def enroll_pooled(scene, obj):
    return _enroll(of(scene).pool, obj)


def reference_target(scene, reference, index=None):
    uid = reference.collider_uid
    if not uid:
        return None
    target = (index if index is not None else pool_index(scene)).get(uid)
    if target is not None and target.ruri_cloth_physics_collider.is_collider:
        return target
    return None


def config_colliders(scene, config, index=None):
    if index is None:
        index = pool_index(scene)
    for reference in config.collider_collision.collider_references:
        target = reference_target(scene, reference, index)
        if target is not None:
            yield target


def reachable_colliders(scene):
    seen = set()
    ordered = []
    index = pool_index(scene)
    candidates = [target for obj in bodies(scene) for config in obj.ruri_cloth_physics.configs
                  for target in config_colliders(scene, config, index)]
    candidates.extend(obj for obj, _settings in scene_colliders(scene))
    for target in candidates:
        key = target.as_pointer()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(target)
    return ordered
