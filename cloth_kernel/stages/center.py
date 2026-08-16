import numpy as np

from .. import defs
from .. import math as pm
from . import wind as wind_stage


def run(world, ctx):
    tt = world.team
    i = ctx.frame_team_index
    if len(i) == 0:
        return

    component_pos = tt["component_world_position"][i]
    component_rot = tt["component_world_rotation"][i]
    component_scl = tt["component_world_scale"][i]

    init_scale_length = np.maximum(pm.length(tt["init_scale"][i]), 1e-30)
    component_scale_ratio = pm.length(component_scl) / init_scale_length

    old_direction = tt["negative_scale_direction"][i].copy()
    direction = np.sign(np.where(component_scl == 0.0, 1.0, component_scl)).astype(np.float32)
    tt["negative_scale_direction"][i] = direction
    tt["negative_scale_change"][i] = old_direction * direction
    is_negative = np.any(component_scl < 0.0, axis=1)
    tt["is_negative_scale"][i] = is_negative
    tt["negative_scale_sign"][i] = np.where(is_negative, -1.0, 1.0)
    quaternion_flip = np.concatenate([-direction, np.ones((len(i), 1), dtype=np.float32)], axis=1)
    tt["negative_scale_quaternion"][i] = np.where(is_negative[:, None], quaternion_flip, 1.0)
    triangle_sign = np.stack([
        np.where((component_scl[:, 0] < 0) | (component_scl[:, 2] < 0), -1.0, 1.0),
        np.where(component_scl[:, 0] < 0, -1.0, 1.0),
    ], axis=1).astype(np.float32)
    tt["negative_scale_triangle_sign"][i] = np.where(is_negative[:, None], triangle_sign, 1.0)
    teleport = np.any(old_direction != direction, axis=1)
    tt["negative_scale_teleport"][i] = teleport

    if np.any(teleport):
        j = i[teleport]
        now_lw = pm.trs(tt["component_world_position"][j], tt["component_world_rotation"][j],
                        tt["component_world_scale"][j])
        old_lw = pm.trs(tt["old_component_world_position"][j],
                        tt["old_component_world_rotation"][j],
                        tt["old_component_world_scale"][j])
        negative_component = now_lw @ np.linalg.inv(old_lw)
        tt["old_component_world_position"][j] = pm.transform_points(
            tt["old_component_world_position"][j], negative_component)
        tt["old_component_world_scale"][j] = tt["component_world_scale"][j]
        tt["old_anchor_position"][j] = pm.transform_points(tt["old_anchor_position"][j],
                                                           negative_component)
        tt["smoothing_velocity"][j] = pm.transform_vectors(tt["smoothing_velocity"][j],
                                                           negative_component)

    old_component_position = tt["old_component_world_position"][i].copy()
    old_component_rotation = tt["old_component_world_rotation"][i].copy()

    center_world_pos = component_pos.copy()
    center_world_rot = component_rot.copy()
    fa = world.center_fixed
    fixed_particle = fa["particle"][:len(fa["team"])]
    fixed_team = fa["team"]
    frame_mask = np.zeros(len(tt), dtype=bool)
    frame_mask[i] = True
    fixed_active = np.flatnonzero(frame_mask[fixed_team])
    if len(fixed_active):
        fp = fixed_particle[fixed_active]
        ft = fixed_team[fixed_active]
        pa = world.particles
        rot = pa["rotations"][fp]
        negative_entry = tt["negative_scale_sign"][ft] < 0
        if np.any(negative_entry):
            normals = pm.quat_to_normal(rot)
            tangents = pm.quat_to_tangent(rot)
            flipped = pm.to_rotation(-normals, -tangents)
            rot = np.where(negative_entry[:, None], flipped, rot)
        rot = pm.quat_mul(rot, pa["vertex_bind_pose_rotations"][fp])
        nor = pm.quat_to_normal(rot)
        tan = pm.quat_to_tangent(rot)
        entry_direction = tt["negative_scale_direction"][ft]
        nor = nor * np.where((entry_direction[:, 0] < 0) | (entry_direction[:, 2] < 0), -1.0, 1.0)[:, None]
        tan = tan * np.where((entry_direction[:, 0] < 0) | (entry_direction[:, 1] < 0), -1.0, 1.0)[:, None]

        team_count = len(tt)
        nor_sum = np.zeros((team_count, 3), dtype=np.float64)
        tan_sum = np.zeros((team_count, 3), dtype=np.float64)
        pos_sum = np.zeros((team_count, 3), dtype=np.float64)
        count = np.zeros(team_count, dtype=np.int64)
        np.add.at(nor_sum, ft, nor)
        np.add.at(tan_sum, ft, tan)
        np.add.at(pos_sum, ft, pa["positions"][fp])
        np.add.at(count, ft, 1)

        team_nor = nor_sum[i]
        team_tan = tan_sum[i]
        nl = np.linalg.norm(team_nor, axis=1)
        tl = np.linalg.norm(team_tan, axis=1)
        team_ok = (count[i] > 0) & (nl > 1e-30) & (tl > 1e-30)
        safe_count = np.maximum(count[i], 1)[:, None]
        mean_pos = (pos_sum[i] / safe_count).astype(np.float32)
        rot_from_fixed = pm.to_rotation(
            (team_nor / np.where(nl > 1e-30, nl, 1.0)[:, None]).astype(np.float32),
            (team_tan / np.where(tl > 1e-30, tl, 1.0)[:, None]).astype(np.float32))
        center_world_pos = np.where(team_ok[:, None], mean_pos, center_world_pos)
        center_world_rot = np.where(team_ok[:, None], rot_from_fixed, center_world_rot)

    if np.any(teleport):
        j = np.flatnonzero(teleport)
        now_lw = pm.trs(center_world_pos[j], center_world_rot[j], component_scl[j])
        old_lw = pm.trs(tt["old_frame_world_position"][i[j]],
                        tt["old_frame_world_rotation"][i[j]],
                        tt["old_frame_world_scale"][i[j]])
        tt["negative_scale_matrix"][i[j]] = now_lw @ np.linalg.inv(old_lw)

    anchor_delta_vector = np.zeros((len(i), 3), dtype=np.float32)
    anchor_delta_rotation = np.broadcast_to(np.array([0, 0, 0, 1], dtype=np.float32),
                                            (len(i), 4)).copy()
    has_anchor = tt["has_anchor"][i]
    anchor_reset = (has_anchor != tt["had_anchor"][i]) | tt["reset_pending"][i]
    tt["had_anchor"][i] = has_anchor
    if np.any(anchor_reset):
        j = i[anchor_reset]
        tt["old_anchor_position"][j] = tt["anchor_position"][j]
        tt["old_anchor_rotation"][j] = tt["anchor_rotation"][j]
        tt["anchor_component_local_position"][j] = pm.quat_rotate(
            pm.quat_inverse(tt["anchor_rotation"][j]),
            tt["component_world_position"][j] - tt["anchor_position"][j])
    if np.any(has_anchor):
        j = np.flatnonzero(has_anchor)
        gi = i[j]
        anchor_center = pm.quat_rotate(tt["anchor_rotation"][gi],
                                       tt["anchor_component_local_position"][gi]) \
            + tt["anchor_position"][gi]
        delta_vector = anchor_center - old_component_position[j]
        delta_rotation = pm.quat_mul(tt["anchor_rotation"][gi],
                                     pm.quat_inverse(tt["old_anchor_rotation"][gi]))
        anchor_ratio = (1.0 - tt["anchor_inertia"][gi])
        delta_vector = delta_vector * anchor_ratio[:, None]
        identity = np.broadcast_to(np.array([0, 0, 0, 1], dtype=np.float32), delta_rotation.shape)
        delta_rotation = pm.quat_slerp(identity, delta_rotation, anchor_ratio)
        anchor_delta_vector[j] = delta_vector
        anchor_delta_rotation[j] = delta_rotation
        old_component_position[j] = old_component_position[j] + delta_vector
        old_component_rotation[j] = pm.quat_mul(delta_rotation, old_component_rotation[j])
        tt["inertia_shift"][gi] = True

    frame_delta_vector = component_pos - old_component_position
    frame_delta_angle = pm.quat_angle(old_component_rotation, component_rot)
    teleport_check = (tt["teleport_mode"][i] != defs.TELEPORT_NONE) & ~tt["reset_pending"][i]
    if np.any(teleport_check):
        far = pm.length(frame_delta_vector) >= tt["teleport_distance"][i] * component_scale_ratio
        spun = np.degrees(frame_delta_angle) >= tt["teleport_rotation"][i]
        hit = teleport_check & (far | spun)
        reset_hit = hit & (tt["teleport_mode"][i] == defs.TELEPORT_RESET)
        keep_hit = hit & (tt["teleport_mode"][i] != defs.TELEPORT_RESET)
        tt["reset_pending"][i] |= reset_hit
        tt["keep_teleport_pending"][i] |= keep_hit

    reset = tt["reset_pending"][i]
    keep = tt["keep_teleport_pending"][i]
    smooth_delta_vector = np.zeros((len(i), 3), dtype=np.float32)
    smoothing_on = (tt["movement_inertia_smoothing"][i] >= 1e-6) & ~(keep | reset)
    if np.any(smoothing_on):
        j = np.flatnonzero(smoothing_on)
        gi = i[j]
        running = tt["running"][gi]
        frame_dt = tt["frame_delta_time"][gi]
        delta_velocity = np.where((frame_dt > 0.0)[:, None],
                                  frame_delta_vector[j] / np.where(frame_dt > 0, frame_dt, 1.0)[:, None],
                                  0.0)
        limit = tt["movement_speed_limit"][gi] * component_scale_ratio[j]
        limited = pm.clamp_vector(delta_velocity, np.maximum(limit, 0.0))
        delta_velocity = np.where((limit >= 0.0)[:, None], limited, delta_velocity)
        average_ratio = pm.saturate((1.0 - tt["movement_inertia_smoothing"][gi]) ** 3 * 0.99 + 0.01)
        smoothed = pm.lerp(tt["smoothing_velocity"][gi], delta_velocity,
                           average_ratio[:, None].astype(np.float32))
        tt["smoothing_velocity"][gi] = np.where(running[:, None], smoothed,
                                                tt["smoothing_velocity"][gi])
        smooth_pos = component_pos[j] - tt["smoothing_velocity"][gi] * frame_dt[:, None]
        smooth_delta_vector[j] = smooth_pos - old_component_position[j]
        old_component_position[j] = smooth_pos
        tt["inertia_shift"][gi] = True

    tt["frame_world_position"][i] = center_world_pos
    tt["frame_world_rotation"][i] = center_world_rot
    tt["frame_world_scale"][i] = component_scl
    if np.any(reset):
        j = i[reset]
        tt["old_component_world_position"][j] = tt["component_world_position"][j]
        tt["old_component_world_rotation"][j] = tt["component_world_rotation"][j]
        tt["old_component_world_scale"][j] = tt["component_world_scale"][j]
        old_component_position[reset] = component_pos[reset]
        old_component_rotation[reset] = component_rot[reset]
        tt["old_frame_world_position"][j] = tt["frame_world_position"][j]
        tt["old_frame_world_rotation"][j] = tt["frame_world_rotation"][j]
        tt["old_frame_world_scale"][j] = component_scl[reset]
        tt["now_world_position"][j] = tt["frame_world_position"][j]
        tt["now_world_rotation"][j] = tt["frame_world_rotation"][j]
        tt["old_world_position"][j] = tt["frame_world_position"][j]
        tt["old_world_rotation"][j] = tt["frame_world_rotation"][j]
    neg_only = teleport & ~reset
    if np.any(neg_only):
        j = i[neg_only]
        tt["old_frame_world_position"][j] = tt["frame_world_position"][j]
        tt["old_frame_world_rotation"][j] = tt["frame_world_rotation"][j]
        tt["old_frame_world_scale"][j] = component_scl[neg_only]
        tt["now_world_position"][j] = tt["frame_world_position"][j]
        tt["now_world_rotation"][j] = tt["frame_world_rotation"][j]
        tt["old_world_position"][j] = tt["frame_world_position"][j]
        tt["old_world_rotation"][j] = tt["frame_world_rotation"][j]

    work_old_position = old_component_position.copy()
    work_old_rotation = old_component_rotation.copy()
    shift_vector = np.zeros((len(i), 3), dtype=np.float32)
    shift_rotation = np.broadcast_to(np.array([0, 0, 0, 1], dtype=np.float32), (len(i), 4)).copy()
    if np.any(reset):
        tt["smoothing_velocity"][i[reset]] = 0.0
        smooth_delta_vector[reset] = 0.0

    live = ~reset
    if np.any(live):
        j = np.flatnonzero(live)
        gi = i[j]
        shift_vector[j] = component_pos[j] - old_component_position[j]
        shift_rotation[j] = pm.quat_mul(component_rot[j],
                                        pm.quat_inverse(old_component_rotation[j]))
        move_shift_ratio = np.zeros(len(j), dtype=np.float32)
        rotation_shift_ratio = np.zeros(len(j), dtype=np.float32)

        keep_now = keep[j] | tt["culling_invisible"][gi]
        movement_shift = np.where(keep_now, 1.0, 1.0 - tt["world_inertia"][gi]).astype(np.float32)
        rotation_shift = movement_shift.copy()

        shifting = (movement_shift > defs.EPSILON) | (rotation_shift > defs.EPSILON)
        if np.any(shifting):
            tt["inertia_shift"][gi[shifting]] = True
        move_shift_ratio = np.where(shifting, movement_shift, move_shift_ratio)
        rotation_shift_ratio = np.where(shifting, rotation_shift, rotation_shift_ratio)
        work_old_position[j] = np.where(shifting[:, None],
                                        pm.lerp(work_old_position[j], component_pos[j],
                                                movement_shift[:, None]),
                                        work_old_position[j])
        slerped = pm.quat_slerp(work_old_rotation[j], component_rot[j], rotation_shift)
        work_old_rotation[j] = np.where(shifting[:, None], slerped, work_old_rotation[j])

        movement_limit = tt["movement_speed_limit"][gi] * component_scale_ratio[j]
        rotation_limit = tt["rotation_speed_limit"][gi]
        delta_vector = component_pos[j] - work_old_position[j]
        delta_angle = pm.quat_angle(work_old_rotation[j], component_rot[j])
        frame_dt = tt["frame_delta_time"][gi]
        frame_speed = np.where(frame_dt > 0, pm.length(delta_vector) / np.where(frame_dt > 0, frame_dt, 1.0), 0.0)
        frame_rotation_speed = np.where(frame_dt > 0, np.degrees(delta_angle) / np.where(frame_dt > 0, frame_dt, 1.0), 0.0)

        over_move = (frame_speed > movement_limit) & (movement_limit >= 0.0)
        if np.any(over_move):
            tt["inertia_shift"][gi[over_move]] = True
        move_limit_ratio = pm.saturate(np.maximum(frame_speed - movement_limit, 0.0)
                                       / np.where(frame_speed > 0, frame_speed, 1.0))
        move_limit_ratio = np.where(over_move, move_limit_ratio, 0.0)
        move_shift_ratio = move_shift_ratio + (1.0 - move_shift_ratio) * move_limit_ratio
        work_old_position[j] = np.where(over_move[:, None],
                                        pm.lerp(work_old_position[j], component_pos[j],
                                                move_limit_ratio[:, None].astype(np.float32)),
                                        work_old_position[j])
        over_rot = (frame_rotation_speed > rotation_limit) & (rotation_limit >= 0.0)
        if np.any(over_rot):
            tt["inertia_shift"][gi[over_rot]] = True
        rotation_limit_ratio = pm.saturate(np.maximum(frame_rotation_speed - rotation_limit, 0.0)
                                           / np.where(frame_rotation_speed > 0, frame_rotation_speed, 1.0))
        rotation_limit_ratio = np.where(over_rot, rotation_limit_ratio, 0.0)
        rotation_shift_ratio = rotation_shift_ratio + (1.0 - rotation_shift_ratio) * rotation_limit_ratio
        slerped = pm.quat_slerp(work_old_rotation[j], component_rot[j], rotation_limit_ratio)
        work_old_rotation[j] = np.where(over_rot[:, None], slerped, work_old_rotation[j])

        other_shift_ratio = np.zeros(len(j), dtype=np.float32)
        skip = tt["skip_count"][gi]
        scaled_dt = frame_dt * tt["now_time_scale"][gi]
        skip_mask = (skip > 0) & (scaled_dt > 0.0)
        skip_ratio = pm.saturate((skip * ctx.simulation_delta_time)
                                 / np.where(scaled_dt > 0, scaled_dt, 1.0))
        other_shift_ratio = np.where(skip_mask,
                                     other_shift_ratio + (1.0 - other_shift_ratio) * skip_ratio,
                                     other_shift_ratio)
        vw = tt["velocity_weight"][gi]
        other_shift_ratio = np.where(vw < 1.0,
                                     other_shift_ratio + (1.0 - other_shift_ratio) * (1.0 - vw),
                                     other_shift_ratio)
        nts = tt["now_time_scale"][gi]
        other_shift_ratio = np.where(nts < 1.0,
                                     other_shift_ratio + (1.0 - other_shift_ratio) * (1.0 - nts),
                                     other_shift_ratio)
        other_on = other_shift_ratio > 0.0
        if np.any(other_on):
            tt["inertia_shift"][gi[other_on]] = True
        move_shift_ratio = np.where(other_on,
                                    move_shift_ratio + (1.0 - move_shift_ratio) * other_shift_ratio,
                                    move_shift_ratio)
        work_old_position[j] = np.where(other_on[:, None],
                                        pm.lerp(work_old_position[j], component_pos[j],
                                                other_shift_ratio[:, None].astype(np.float32)),
                                        work_old_position[j])
        rotation_shift_ratio = np.where(other_on,
                                        rotation_shift_ratio + (1.0 - rotation_shift_ratio) * other_shift_ratio,
                                        rotation_shift_ratio)
        slerped = pm.quat_slerp(work_old_rotation[j], component_rot[j], other_shift_ratio)
        work_old_rotation[j] = np.where(other_on[:, None], slerped, work_old_rotation[j])

        applied = tt["inertia_shift"][gi]
        if np.any(applied):
            k = np.flatnonzero(applied)
            gk = gi[k]
            vec = shift_vector[j[k]] * move_shift_ratio[k][:, None]
            identity = np.broadcast_to(np.array([0, 0, 0, 1], dtype=np.float32), (len(k), 4))
            rot = pm.quat_slerp(identity, shift_rotation[j[k]], rotation_shift_ratio[k])
            vec = vec + anchor_delta_vector[j[k]]
            rot = pm.quat_mul(anchor_delta_rotation[j[k]], rot)
            vec = vec + smooth_delta_vector[j[k]]
            tt["frame_component_shift_vector"][gk] = vec
            tt["frame_component_shift_rotation"][gk] = rot

            old_center = tt["old_component_world_position"][gk]
            tt["old_frame_world_position"][gk] = pm.shift_points(
                tt["old_frame_world_position"][gk], old_center, vec, rot)
            tt["old_frame_world_rotation"][gk] = pm.quat_mul(rot, tt["old_frame_world_rotation"][gk])
            tt["now_world_position"][gk] = pm.shift_points(
                tt["now_world_position"][gk], old_center, vec, rot)
            tt["now_world_rotation"][gk] = pm.quat_mul(rot, tt["now_world_rotation"][gk])
        not_applied = ~applied
        if np.any(not_applied):
            gk = gi[not_applied]
            tt["frame_component_shift_vector"][gk] = shift_vector[j[not_applied]]
            tt["frame_component_shift_rotation"][gk] = shift_rotation[j[not_applied]]
    if np.any(reset):
        gk = i[reset]
        tt["frame_component_shift_vector"][gk] = 0.0
        tt["frame_component_shift_rotation"][gk] = (0, 0, 0, 1)

    moving_vector = component_pos - work_old_position
    moving_length = pm.length(moving_vector)
    frame_dt = tt["frame_delta_time"][i]
    speed = np.where(frame_dt > 0, moving_length / np.where(frame_dt > 0, frame_dt, 1.0), 0.0)
    nts = tt["now_time_scale"][i]
    speed = speed * np.where(nts > 1e-6, 1.0 / np.where(nts > 1e-6, nts, 1.0), 0.0)
    tt["frame_moving_speed"][i] = speed
    tt["frame_moving_direction"][i] = np.where((moving_length > 1e-6)[:, None],
                                               moving_vector / np.where(moving_length > 1e-6,
                                                                        moving_length, 1.0)[:, None],
                                               0.0)

    stabilize = tt["reset_pending"][i] | tt["time_reset_pending"][i]
    if np.any(stabilize):
        gk = i[stabilize]
        weight = np.where(tt["stablization_time"][gk] > 1e-6, 0.0, 1.0).astype(np.float32)
        tt["velocity_weight"][gk] = weight
        tt["blend_weight"][gk] = weight

    wind_stage.select_team_wind(world, ctx, i, center_world_pos)
