import numpy as np

from .. import defs
from .. import math as pm


def select_team_wind(world, ctx, team_index, center_world_positions):
    tt = world.team
    zones = ctx.zones
    for k, slot in enumerate(team_index):
        row = tt[slot]
        old_ids = row["wind_zone_id"][:row["wind_count"]].tolist()
        old_times = row["wind_time"][:row["wind_count"]].tolist()
        old_map = {zone_id: time for zone_id, time in zip(old_ids, old_times)}
        result = []
        if zones and row["wind_influence"] > defs.EPSILON:
            center = center_world_positions[k].astype(np.float64)
            min_volume = float("inf")
            addition_count = 0
            latest_id = None
            for zone in zones:
                if zone.is_addition and addition_count >= 3:
                    continue
                zone_volume = float("inf") if zone.mode == 'GLOBAL_DIRECTION' else float(zone.zone_volume)
                lpos = zone.world_to_local[:3, :3] @ center + zone.world_to_local[:3, 3]
                llen = float(np.linalg.norm(lpos))

                if zone.mode == 'BOX_DIRECTION':
                    lv = np.abs(lpos) * 2.0
                    if lv[0] > zone.size[0] or lv[1] > zone.size[1] or lv[2] > zone.size[2]:
                        continue
                elif zone.mode in {'SPHERE_DIRECTION', 'SPHERE_RADIAL'}:
                    if llen > zone.size[0]:
                        continue

                if not zone.is_addition and zone_volume > min_volume:
                    continue

                direction = zone.world_direction
                main = zone.main
                if zone.mode == 'SPHERE_RADIAL':
                    if llen <= 1e-6:
                        continue
                    v = center - zone.world_position
                    direction = (v / np.linalg.norm(v)).astype(np.float32)
                    depth = min(max(llen / float(zone.size[0]), 0.0), 1.0)
                    attenuation = float(pm.evaluate_curve_lut_clamp01(zone.attenuation_lut,
                                                                      np.float32(depth)))
                    main = main * attenuation

                info = (zone.zone_id, old_map.get(zone.zone_id, -defs.WIND_MAX_TIME), main,
                        np.asarray(direction, dtype=np.float32), zone.turbulence)
                if zone.is_addition:
                    result.append(info)
                    addition_count += 1
                else:
                    if latest_id is not None:
                        result = [r for r in result if r[0] != latest_id]
                    result.append(info)
                    min_volume = zone_volume
                    latest_id = zone.zone_id
        result = result[:defs.WIND_ZONE_SLOTS]
        row["wind_count"] = len(result)
        for s, (zone_id, time, main, direction, turbulence) in enumerate(result):
            row["wind_zone_id"][s] = zone_id
            row["wind_time"][s] = time
            row["wind_main"][s] = main
            row["wind_direction"][s] = direction
            row["wind_zone_turbulence"][s] = turbulence
            row["wind_dirq"][s] = pm.axis_quaternion(direction[None].astype(np.float32))[0]


def calc_wind_force(world, ctx, particle_index, particle_team, depth, friction):
    tt = world.team
    pa = world.particles
    count = len(particle_index)
    force = np.zeros((count, 3), dtype=np.float32)
    if count == 0:
        return force
    roots = pa["vertex_root_local"][particle_index].astype(np.float32)
    seed = tt["wind_seed"][particle_team].astype(np.float32)
    synchronization = tt["wind_synchronization"][particle_team]
    wind_positions = (seed + 1.0) * 4.19230645 + roots * 0.0023963 * (1.0 - synchronization) * 100.0

    blend = tt["wind_blend"][particle_team]
    turbulence_param = tt["wind_turbulence"][particle_team]

    for s in range(defs.WIND_ZONE_SLOTS):
        slot_on = s < tt["wind_count"][particle_team]
        if not np.any(slot_on):
            continue
        main = tt["wind_main"][particle_team, s]
        contribution = _wind_force_blend(main, tt["wind_time"][particle_team, s],
                                         tt["wind_dirq"][particle_team, s],
                                         tt["wind_zone_turbulence"][particle_team, s],
                                         blend, turbulence_param, wind_positions)
        force = force + np.where(slot_on[:, None], contribution, 0.0)

    moving_on = (tt["moving_wind_main"][particle_team] > 0.01) \
        & (tt["wind_moving"][particle_team] > 0.01)
    if np.any(moving_on):
        contribution = _wind_force_blend(tt["moving_wind_main"][particle_team],
                                         tt["moving_wind_time"][particle_team],
                                         tt["moving_wind_dirq"][particle_team],
                                         np.ones(count, dtype=np.float32),
                                         blend, turbulence_param, wind_positions)
        force = force + np.where(moving_on[:, None], contribution, 0.0)

    influence = tt["wind_influence"][particle_team] * (1.0 - friction)
    depth_scale = depth * depth
    influence = influence * pm.lerp(1.0, depth_scale, tt["wind_depth_weight"][particle_team])
    return force * influence[:, None]


def _wind_force_blend(wind_main, time, dirq, zone_turbulence, blend, turbulence_param,
                      wind_positions):
    active = wind_main >= 0.01
    main_ratio = wind_main / defs.WIND_BASE_SPEED

    sin_pos = wind_positions + time * 10.0
    sin_wave = np.sin(sin_pos)

    noise_pos = wind_positions + time * 2.3132
    noise_wave = pm.cnoise2(noise_pos, noise_pos) * 2.3

    wave_x = sin_wave + (noise_wave - sin_wave) * blend
    wave_y = wave_x

    turbulence = zone_turbulence * turbulence_param

    angle_x = np.radians(wave_x * defs.WIND_TURBULENCE_ANGLE)
    angle_y = np.radians(wave_y * defs.WIND_TURBULENCE_ANGLE)
    angle_y = angle_y * (0.1 + (0.5 - 0.1) * blend)
    angle_x = angle_x * turbulence
    angle_y = angle_y * turbulence

    rq = pm.euler_yx(angle_x.astype(np.float32), angle_y.astype(np.float32))
    combined = pm.quat_mul(dirq.astype(np.float32), rq)
    wind_direction = pm.quat_to_tangent(combined)

    main_scale = pm.saturate(1.0 - main_ratio)
    main_wave = (wave_x + 1.0) * 0.5
    main_wave = main_wave * main_scale * turbulence
    strength = wind_main - wind_main * main_wave
    strength = np.where(active, strength, 0.0)
    return wind_direction * strength[..., None].astype(np.float32)
