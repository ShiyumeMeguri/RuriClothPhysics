import bpy


def action_supports_slots(action):
    return hasattr(action, "slots")


def ensure_action(obj, name):
    animation_data = obj.animation_data
    if animation_data is None:
        animation_data = obj.animation_data_create()

    action = animation_data.action
    if action is None:
        action = bpy.data.actions.new(name)
        animation_data.action = action

    if action_supports_slots(action):
        slot = animation_data.action_slot
        if slot is None:
            for candidate in action.slots:
                if candidate.target_id_type in {'OBJECT', 'UNSPECIFIED'}:
                    slot = candidate
                    break
            if slot is None:
                slot = action.slots.new(id_type='OBJECT', name=obj.name)
            animation_data.action_slot = slot

        if len(action.layers) == 0:
            layer = action.layers.new("Layer")
        else:
            layer = action.layers[0]

        if len(layer.strips) == 0:
            strip = layer.strips.new(type='KEYFRAME')
        else:
            strip = layer.strips[0]

        channelbag = strip.channelbag(slot, ensure=True)
        return action, channelbag

    return action, action


def iter_object_fcurves(obj):
    animation_data = obj.animation_data
    if animation_data is None or animation_data.action is None:
        return
    action = animation_data.action
    if action_supports_slots(action):
        slot = animation_data.action_slot
        if slot is not None:
            for layer in action.layers:
                for strip in layer.strips:
                    bag = strip.channelbag(slot)
                    if bag is not None:
                        yield from bag.fcurves
            return
    if hasattr(action, "fcurves"):
        yield from action.fcurves


def iter_nla_fcurves(obj):
    animation_data = obj.animation_data
    if animation_data is None:
        return
    for track in getattr(animation_data, "nla_tracks", ()):
        if track.mute:
            continue
        for strip in track.strips:
            action = strip.action
            if action is None:
                continue
            if action_supports_slots(action):
                for layer in action.layers:
                    for astrip in layer.strips:
                        for bag in getattr(astrip, "channelbags", ()):
                            yield from bag.fcurves
            elif hasattr(action, "fcurves"):
                yield from action.fcurves


def ensure_fcurve(channel_container, data_path, index, group_name):
    fcurve = channel_container.fcurves.find(data_path, index=index)
    if fcurve is None:
        try:
            fcurve = channel_container.fcurves.new(data_path, index=index, action_group=group_name)
        except TypeError:
            fcurve = channel_container.fcurves.new(data_path, index=index)
    return fcurve


_LINEAR_ENUM_VALUE = None


def linear_interpolation_value():
    global _LINEAR_ENUM_VALUE
    if _LINEAR_ENUM_VALUE is None:
        _LINEAR_ENUM_VALUE = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value
    return _LINEAR_ENUM_VALUE


def bulk_write_fcurve(fcurve, frames, values, frame_start, frame_end):
    kept = []
    for keyframe in fcurve.keyframe_points:
        frame = keyframe.co[0]
        if frame < frame_start - 1e-6 or frame > frame_end + 1e-6:
            kept.append((frame, keyframe.co[1]))

    merged = kept + list(zip(frames, values))
    merged.sort(key=lambda item: item[0])
    total = len(merged)

    points = fcurve.keyframe_points
    points.clear()
    points.add(total)
    flat = [0.0] * (total * 2)
    for i, (frame, value) in enumerate(merged):
        flat[i * 2] = frame
        flat[i * 2 + 1] = value
    points.foreach_set("co", flat)
    points.foreach_set("interpolation", [linear_interpolation_value()] * total)
    fcurve.update()
