ARMATURE_TYPE = 'ARMATURE'

WORLD_REGISTRY_REASON = (
    "who takes part in a scene's simulation is a fact the scene states, not one the add-on "
    "finds out: the scene keeps three explicit lists -- the bodies it simulates, the "
    "colliders every cloth in it collides with, the wind zones blowing through it -- and a "
    "configuration names the colliders it hits by pointing at them; the frame, the overlays "
    "and the panels only ever walk outward from these lists, so nothing anywhere enumerates "
    "the scene, an object nobody registered is never read, and a scene with empty lists "
    "costs nothing at all. An entry whose object was deleted reads as no object and is "
    "skipped where it is read and dropped where the lists are written")


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


def config_colliders(config):
    for reference in config.collider_collision.collider_references:
        target = reference.collider
        if target is not None and target.ruri_cloth_physics_collider.is_collider:
            yield target


def reachable_colliders(scene):
    seen = set()
    ordered = []
    candidates = [target for obj in bodies(scene) for config in obj.ruri_cloth_physics.configs
                  for target in config_colliders(config)]
    candidates.extend(obj for obj, _settings in scene_colliders(scene))
    for target in candidates:
        key = target.as_pointer()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(target)
    return ordered
