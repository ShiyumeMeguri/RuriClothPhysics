import numpy as np

from ...cloth_kernel import defs
from ...cloth_kernel import math as pm

TIME_RESET_FIELDS = ("time", "old_time", "now_update_time", "old_update_time",
                     "frame_update_time", "frame_old_time")
SYNC_TIME_FIELDS = ("time", "old_time", "now_update_time", "old_update_time",
                    "frame_update_time", "frame_old_time", "time_scale")
SYNC_PARAM_FIELDS = ("anchor_inertia", "world_inertia", "movement_inertia_smoothing",
                     "movement_speed_limit", "rotation_speed_limit",
                     "teleport_mode", "teleport_distance", "teleport_rotation")


def resolve_sync(world):
    tt = world.team
    active = np.flatnonzero(tt["enabled"])
    for slot in active:
        target = int(tt["sync_target"][slot])
        if target <= 0 or not tt["valid"][target] or not tt["enabled"][target]:
            tt["sync_top"][slot] = 0
            continue
        top = target
        for _ in range(8):
            upper = int(tt["sync_target"][top])
            if upper <= 0 or upper == slot or not tt["valid"][upper] or not tt["enabled"][upper]:
                break
            top = upper
        tt["sync_top"][slot] = top
    children = active[tt["sync_top"][active] > 0]
    if len(children) == 0:
        return
    tops = tt["sync_top"][children]
    for name in SYNC_TIME_FIELDS:
        tt[name][children] = tt[name][tops]
    for name in SYNC_PARAM_FIELDS:
        tt[name][children] = tt[name][tops]
    tt["component_world_position"][children] = tt["component_world_position"][tops]
    tt["component_world_rotation"][children] = tt["component_world_rotation"][tops]


def advance(world, ctx):
    tt = world.team
    i = ctx.frame_team_index
    if len(i) == 0:
        return
    time_reset = tt["time_reset_pending"][i]
    for name in TIME_RESET_FIELDS:
        tt[name][i] = np.where(time_reset, 0.0, tt[name][i])

    tt["frame_delta_time"][i] = ctx.frame_delta_time
    time_scale = tt["time_scale"][i] * ctx.global_time_scale
    tt["now_time_scale"][i] = time_scale

    add_time = np.float32(ctx.frame_delta_time) * time_scale
    time = tt["time"][i] + add_time
    interval = time - tt["now_update_time"][i]
    update_count = (interval / ctx.simulation_delta_time).astype(np.int32)
    clamped = np.minimum(update_count, ctx.max_simulation_count)
    skip = update_count - clamped
    time = np.where(skip > 0, time - np.float32(ctx.simulation_delta_time) * skip, time)
    guard = (clamped > 0) & (add_time == 0.0)
    clamped = np.where(guard, 0, clamped)
    skip = np.where(guard, 0, skip)
    tt["now_update_time"][i] = np.where(guard, time - ctx.simulation_delta_time + 0.0001,
                                        tt["now_update_time"][i])

    updated = clamped > 0
    tt["frame_old_time"][i] = np.where(updated, tt["frame_update_time"][i], tt["frame_old_time"][i])
    tt["frame_update_time"][i] = np.where(updated, time, tt["frame_update_time"][i])
    tt["old_update_time"][i] = np.where(updated, tt["now_update_time"][i], tt["old_update_time"][i])
    tt["old_time"][i] = tt["time"][i]
    tt["time"][i] = time
    tt["update_count"][i] = clamped
    tt["skip_count"][i] = skip
    tt["running"][i] = updated


def step_update(world, ctx):
    tt = world.team
    i = ctx.step_team_index
    if len(i) == 0:
        return
    sdt = np.float32(ctx.simulation_delta_time)

    tt["now_update_time"][i] = tt["now_update_time"][i] + sdt
    span = tt["time"][i] - tt["frame_old_time"][i]
    interpolation = np.where(span > 0,
                             pm.saturate((tt["now_update_time"][i] - tt["frame_old_time"][i])
                                         / np.where(span > 0, span, 1.0)),
                             1.0).astype(np.float32)
    tt["frame_interpolation"][i] = interpolation

    tt["old_world_position"][i] = tt["now_world_position"][i]
    tt["old_world_rotation"][i] = tt["now_world_rotation"][i]
    t = interpolation
    tt["now_world_position"][i] = pm.lerp(tt["old_frame_world_position"][i],
                                          tt["frame_world_position"][i], t[:, None])
    tt["now_world_rotation"][i] = pm.quat_slerp(tt["old_frame_world_rotation"][i],
                                                tt["frame_world_rotation"][i], t)
    world_scale = pm.lerp(tt["old_frame_world_scale"][i], tt["frame_world_scale"][i], t[:, None])

    step_vector = tt["now_world_position"][i] - tt["old_world_position"][i]
    step_rotation = pm.quat_mul(tt["now_world_rotation"][i],
                                pm.quat_inverse(tt["old_world_rotation"][i]))
    step_angle = pm.quat_angle(tt["old_world_rotation"][i], tt["now_world_rotation"][i])
    tt["step_vector"][i] = step_vector
    tt["step_rotation"][i] = step_rotation

    local_movement_inertia = 1.0 - tt["local_inertia"][i]
    local_rotation_inertia = 1.0 - tt["local_inertia"][i]
    local_vector = step_vector * (1.0 - local_movement_inertia)[:, None]
    local_speed = pm.length(local_vector) / sdt
    limit = tt["local_movement_speed_limit"][i]
    over = (local_speed > limit) & (limit >= 0.0)
    ratio = np.where(over, limit / np.where(local_speed > 0, local_speed, 1.0), 1.0)
    local_movement_inertia = np.where(over, 1.0 + (local_movement_inertia - 1.0) * ratio,
                                      local_movement_inertia)
    local_angle = step_angle * (1.0 - local_rotation_inertia)
    local_angle_speed = np.degrees(local_angle / sdt)
    limit = tt["local_rotation_speed_limit"][i]
    over = (local_angle_speed > limit) & (limit >= 0.0)
    ratio = np.where(over, limit / np.where(local_angle_speed > 0, local_angle_speed, 1.0), 1.0)
    local_rotation_inertia = np.where(over, 1.0 + (local_rotation_inertia - 1.0) * ratio,
                                      local_rotation_inertia)
    tt["step_move_inertia_ratio"][i] = local_movement_inertia
    tt["step_rotation_inertia_ratio"][i] = local_rotation_inertia

    tt["inertia_vector"][i] = step_vector * local_movement_inertia[:, None]
    identity = pm.IDENTITY_QUAT
    tt["inertia_rotation"][i] = pm.quat_slerp(identity, step_rotation, local_rotation_inertia)

    angular_velocity = step_angle / sdt
    tt["angular_velocity"][i] = angular_velocity
    _, axis = pm.quat_to_angle_axis(step_rotation)
    tt["rotation_axis"][i] = np.where((angular_velocity > defs.EPSILON)[:, None], axis, 0.0)

    init_scale_length = np.maximum(pm.length(tt["init_scale"][i]), 1e-30)
    tt["scale_ratio"][i] = np.maximum(pm.length(world_scale) / init_scale_length, 1e-6)

    gravity_dot = np.ones(len(i), dtype=np.float32)
    has_gravity = pm.length(tt["gravity_direction"][i]) ** 2 > defs.EPSILON
    init_local = tt["init_local_gravity_direction"][i].copy()
    init_local[:, 1] *= tt["negative_scale_direction"][i][:, 1]
    world_falloff = pm.quat_rotate(tt["now_world_rotation"][i], init_local)
    dot = pm.dot(world_falloff, tt["gravity_direction"][i])
    gravity_dot = np.where(has_gravity, pm.saturate(dot * 0.5 + 0.5), gravity_dot)
    tt["gravity_dot"][i] = gravity_dot

    gravity_ratio = np.ones(len(i), dtype=np.float32)
    falloff_active = (tt["gravity"][i] > 1e-6) & (tt["gravity_falloff"][i] > 1e-6)
    low = pm.saturate(1.0 - tt["gravity_falloff"][i])
    gravity_ratio = np.where(falloff_active,
                             low + (1.0 - low) * pm.saturate(1.0 - gravity_dot),
                             gravity_ratio)
    tt["gravity_ratio"][i] = gravity_ratio

    below = tt["velocity_weight"][i] < 1.0
    stab = tt["stablization_time"][i]
    add = np.where(stab > 1e-6, sdt / np.where(stab > 1e-6, stab, 1.0), 1.0)
    tt["velocity_weight"][i] = np.where(below, np.minimum(tt["velocity_weight"][i] + add, 1.0),
                                        tt["velocity_weight"][i])
    tt["blend_weight"][i] = pm.saturate(tt["velocity_weight"][i] * tt["blend_weight_param"][i]
                                        * tt["distance_weight"][i])

    moving_active = tt["wind_moving"][i] > 0.01
    tt["moving_wind_main"][i] = np.where(
        moving_active,
        (tt["frame_moving_speed"][i] * tt["wind_moving"][i])
        / np.where(tt["scale_ratio"][i] > 0, tt["scale_ratio"][i], 1.0),
        0.0)
    tt["moving_wind_direction"][i] = np.where(moving_active[:, None],
                                              -tt["frame_moving_direction"][i],
                                              tt["moving_wind_direction"][i])
    tt["moving_wind_dirq"][i] = np.where(
        moving_active[:, None],
        pm.axis_quaternion(tt["moving_wind_direction"][i]).astype(np.float32),
        tt["moving_wind_dirq"][i])

    main_ratio = tt["wind_main"][i] / defs.WIND_BASE_SPEED
    frequency = 0.2 + main_ratio * 0.5
    frequency = frequency * tt["wind_frequency"][i][:, None]
    frequency = np.minimum(frequency, 1.5) * sdt
    slot_active = np.arange(defs.WIND_ZONE_SLOTS)[None, :] < tt["wind_count"][i][:, None]
    new_time = tt["wind_time"][i] + frequency
    new_time = np.where(new_time > defs.WIND_MAX_TIME, new_time - defs.WIND_MAX_TIME * 2.0, new_time)
    tt["wind_time"][i] = np.where(slot_active, new_time, tt["wind_time"][i])

    move_ratio = tt["moving_wind_main"][i] / defs.WIND_BASE_SPEED
    move_frequency = np.minimum((0.2 + move_ratio * 0.5) * tt["wind_frequency"][i], 1.5) * sdt
    move_time = tt["moving_wind_time"][i] + move_frequency
    move_time = np.where(move_time > defs.WIND_MAX_TIME, move_time - defs.WIND_MAX_TIME * 2.0, move_time)
    tt["moving_wind_time"][i] = np.where(moving_active, move_time, tt["moving_wind_time"][i])


def frame_post(world, ctx):
    tt = world.team
    i = ctx.frame_team_index
    if len(i) == 0:
        return
    tt["old_component_world_position"][i] = tt["component_world_position"][i]
    tt["old_component_world_rotation"][i] = tt["component_world_rotation"][i]
    tt["old_component_world_scale"][i] = tt["component_world_scale"][i]

    running = tt["running"][i]
    tt["old_frame_world_position"][i] = np.where(running[:, None], tt["frame_world_position"][i],
                                                 tt["old_frame_world_position"][i])
    tt["old_frame_world_rotation"][i] = np.where(running[:, None], tt["frame_world_rotation"][i],
                                                 tt["old_frame_world_rotation"][i])
    tt["old_frame_world_scale"][i] = np.where(running[:, None], tt["frame_world_scale"][i],
                                              tt["old_frame_world_scale"][i])
    tt["skip_count"][i] = np.where(running, 0, tt["skip_count"][i])
    tt["force_mode"][i] = np.where(running, defs.FORCE_NONE, tt["force_mode"][i])
    tt["impact_force"][i] = np.where(running[:, None], 0.0, tt["impact_force"][i])

    tt["old_anchor_position"][i] = tt["anchor_position"][i]
    tt["old_anchor_rotation"][i] = tt["anchor_rotation"][i]
    tt["anchor_component_local_position"][i] = pm.quat_rotate(
        pm.quat_inverse(tt["anchor_rotation"][i]),
        tt["component_world_position"][i] - tt["anchor_position"][i])

    tt["reset_pending"][i] = False
    tt["time_reset_pending"][i] = False
    tt["running"][i] = False
    tt["keep_teleport_pending"][i] = False
    tt["inertia_shift"][i] = False
    tt["negative_scale_teleport"][i] = False

    limit_time = 3600.0
    over = tt["time"][i] > limit_time * 2
    for name in TIME_RESET_FIELDS:
        tt[name][i] = np.where(over, tt[name][i] - limit_time, tt[name][i])
