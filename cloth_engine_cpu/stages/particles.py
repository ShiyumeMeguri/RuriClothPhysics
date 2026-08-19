import numpy as np

from ...cloth_kernel import defs
from ...cloth_kernel import math as pm
from . import wind as wind_stage


def compute_base_pose(world, ctx):
    fp = ctx.frame_particle_index
    if len(fp) == 0:
        return
    pa = world.particles
    ta = world.transforms
    skin = np.einsum('tij,tjk->tik', ta["world"], ta["bind_pose"])

    lpos = np.concatenate([pa["local_positions"][fp],
                           np.ones((len(fp), 1), dtype=np.float32)], axis=1)
    lnor = pa["local_normals"][fp]
    ltan = pa["local_tangents"][fp]

    wpos = np.zeros((len(fp), 3), dtype=np.float32)
    wnor = np.zeros((len(fp), 3), dtype=np.float32)
    wtan = np.zeros((len(fp), 3), dtype=np.float32)
    indices = pa["skin_indices"][fp]
    weights = pa["skin_weights"][fp]
    for j in range(4):
        w = weights[:, j]
        if not np.any(w > 0.0):
            continue
        m = skin[indices[:, j]]
        wj = w[:, None]
        wpos += np.einsum('nij,nj->ni', m[:, :3, :4], lpos) * wj
        wnor += np.einsum('nij,nj->ni', m[:, :3, :3], lnor) * wj
        wtan += np.einsum('nij,nj->ni', m[:, :3, :3], ltan) * wj

    pa["positions"][fp] = wpos
    wnor = pm.normalize(wnor, fallback=pm.VEC_UP)
    wtan = pm.normalize(wtan, fallback=pm.VEC_FORWARD)
    pa["rotations"][fp] = pm.to_rotation(wnor, wtan)


def frame_pre(world, ctx):
    fp = ctx.frame_particle_index
    if len(fp) == 0:
        return
    pa = world.particles
    tt = world.team
    pt = pa["team"][fp]

    reset = tt["reset_pending"][pt]
    if np.any(reset):
        r = fp[reset]
        pa["next_positions"][r] = pa["positions"][r]
        pa["old_positions"][r] = pa["positions"][r]
        pa["old_rotations"][r] = pa["rotations"][r]
        pa["base_positions"][r] = pa["positions"][r]
        pa["base_rotations"][r] = pa["rotations"][r]
        pa["old_anim_positions"][r] = pa["positions"][r]
        pa["old_anim_rotations"][r] = pa["rotations"][r]
        pa["velocity_positions"][r] = pa["positions"][r]
        pa["display_positions"][r] = pa["positions"][r]
        pa["velocities"][r] = 0.0
        pa["real_velocities"][r] = 0.0
        pa["friction"][r] = 0.0
        pa["static_friction"][r] = 0.0
        pa["collision_normals"][r] = 0.0

    moved = ~reset & (tt["inertia_shift"][pt] | tt["negative_scale_teleport"][pt])
    if not np.any(moved):
        return
    m = fp[moved]
    mt = pa["team"][m]

    negative = tt["negative_scale_teleport"][mt]
    if np.any(negative):
        g = m[negative]
        gt = pa["team"][g]
        matrix = tt["negative_scale_matrix"][gt]
        flip = np.ones((len(g), 3), dtype=np.float32)
        pa["old_positions"][g] = pm.transform_points(pa["old_positions"][g], matrix)
        pa["old_rotations"][g] = pm.transform_rotations(pa["old_rotations"][g], matrix, flip)
        pa["old_anim_positions"][g] = pm.transform_points(pa["old_anim_positions"][g], matrix)
        pa["old_anim_rotations"][g] = pm.transform_rotations(pa["old_anim_rotations"][g], matrix, flip)
        pa["display_positions"][g] = pm.transform_points(pa["display_positions"][g], matrix)
        pa["velocities"][g] = pm.transform_vectors(pa["velocities"][g], matrix)
        pa["real_velocities"][g] = pm.transform_vectors(pa["real_velocities"][g], matrix)

    shifted = tt["inertia_shift"][mt]
    if np.any(shifted):
        g = m[shifted]
        gt = pa["team"][g]
        shift_vector = tt["frame_component_shift_vector"][gt]
        shift_rotation = tt["frame_component_shift_rotation"][gt]
        center = tt["old_component_world_position"][gt]
        for name in ("old_positions", "old_anim_positions", "display_positions"):
            local = pa[name][g] - center
            pa[name][g] = pm.quat_rotate(shift_rotation, local) + center + shift_vector
        pa["old_rotations"][g] = pm.quat_mul(shift_rotation, pa["old_rotations"][g])
        pa["old_anim_rotations"][g] = pm.quat_mul(shift_rotation, pa["old_anim_rotations"][g])
        pa["velocities"][g] = pm.quat_rotate(shift_rotation, pa["velocities"][g])
        pa["real_velocities"][g] = pm.quat_rotate(shift_rotation, pa["real_velocities"][g])


def step_update(world, ctx):
    sp = ctx.step_particle_index
    if len(sp) == 0:
        return
    pa = world.particles
    tt = world.team
    pt = pa["team"][sp]

    t = tt["frame_interpolation"][pt]
    base_pos = pm.lerp(pa["old_anim_positions"][sp], pa["positions"][sp], t[:, None])
    base_rot = pm.quat_slerp(pa["old_anim_rotations"][sp], pa["rotations"][sp], t)
    pa["base_positions"][sp] = base_pos
    pa["base_rotations"][sp] = base_rot
    pa["step_basic_positions"][sp] = base_pos
    pa["step_basic_rotations"][sp] = base_rot

    move = ctx.entry_index("update_move", world.update_move)
    if len(move):
        pmi = world.update_move["particle"][move]
        mt = pa["team"][pmi]
        depth = pa["depth"][pmi]
        velocity = pa["velocities"][pmi]
        old_pos = pa["old_positions"][pmi]

        inertia_depth = tt["depth_inertia"][mt] * (1.0 - depth * depth)
        inertia_vector = pm.lerp(tt["inertia_vector"][mt], tt["step_vector"][mt],
                                 inertia_depth[:, None])
        inertia_rotation = pm.quat_slerp(tt["inertia_rotation"][mt], tt["step_rotation"][mt],
                                         inertia_depth)
        old_world = tt["old_world_position"][mt]
        local = old_pos - old_world
        local = pm.quat_rotate(inertia_rotation, local) + inertia_vector
        world_pos = old_world + local
        next_pos = world_pos
        velocity_pos = old_pos + (world_pos - old_pos)
        velocity = pm.quat_rotate(inertia_rotation, velocity)

        velocity = velocity * tt["velocity_weight"][mt][:, None]
        damping = pm.evaluate_team_lut_clamp01(tt["damping_lut"], mt, depth)
        velocity = velocity * pm.saturate(1.0 - damping * ctx.power[2])[:, None]

        force_mode = tt["force_mode"][mt]
        change = (force_mode == defs.FORCE_VELOCITY_CHANGE) \
            | (force_mode == defs.FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH)
        velocity = np.where(change[:, None], 0.0, velocity)

        force = tt["gravity_direction"][mt] * (tt["gravity"][mt] * tt["gravity_ratio"][mt])[:, None]
        mass = pm.calc_mass(depth)
        impact = tt["impact_force"][mt]
        with_depth = (force_mode == defs.FORCE_VELOCITY_ADD) \
            | (force_mode == defs.FORCE_VELOCITY_CHANGE)
        without_depth = (force_mode == defs.FORCE_VELOCITY_ADD_WITHOUT_DEPTH) \
            | (force_mode == defs.FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH)
        force = force + np.where(with_depth[:, None], impact / mass[:, None], 0.0)
        force = force + np.where(without_depth[:, None], impact, 0.0)
        force = force + wind_stage.calc_wind_force(world, ctx, pmi, mt, depth,
                                                   pa["friction"][pmi])
        force = force * tt["scale_ratio"][mt][:, None]

        velocity = velocity + force * ctx.simulation_delta_time
        next_pos = next_pos + velocity * ctx.simulation_delta_time

        pa["velocities"][pmi] = velocity
        pa["next_positions"][pmi] = next_pos
        pa["velocity_positions"][pmi] = velocity_pos

    fixed = ctx.entry_index("update_fixed", world.update_fixed)
    if len(fixed):
        pfi = world.update_fixed["particle"][fixed]
        pa["next_positions"][pfi] = pa["base_positions"][pfi]
        pa["velocity_positions"][pfi] = pa["base_positions"][pfi]

    spring = ctx.entry_index("spring", world.spring)
    if len(spring):
        psi = world.spring["particle"][spring]
        st = pa["team"][psi]
        powered = tt["spring_power"][st] > 0.0
        if np.any(powered):
            _apply_spring(world, ctx, psi[powered])


def _apply_spring(world, ctx, spring_particles):
    pa = world.particles
    tt = world.team
    st = pa["team"][spring_particles]
    next_pos = pa["next_positions"][spring_particles]
    base_pos = pa["base_positions"][spring_particles]
    base_rot = pa["base_rotations"][spring_particles]
    v = next_pos - base_pos

    direction = pm.quat_rotate(base_rot, tt["normal_axis_vector"][st])
    limit = tt["spring_limit_distance"][st] * tt["scale_ratio"][st]
    clampable = limit > 1e-8
    l = pm.length(v)
    over = clampable & (l > limit)
    scale = np.where(over & (l > 1e-30), limit / np.where(l > 1e-30, l, 1.0), 1.0)
    v = v * scale[:, None]
    ratio = tt["spring_normal_limit_ratio"][st]
    elliptic = clampable & (ratio < 1.0)
    ylen = pm.dot(direction, v)
    vx = v - direction * ylen[:, None]
    xlen = pm.length(vx)
    t = pm.saturate(xlen / np.where(limit > 1e-30, limit, 1.0))
    y = np.cos(np.arcsin(np.clip(t, -1.0, 1.0))) * (limit * ratio)
    exceed = elliptic & (np.abs(ylen) > y)
    v = np.where(exceed[:, None],
                 v - ((np.abs(ylen) - y) * np.sign(ylen))[:, None] * direction, v)
    v = np.where(clampable[:, None], v, 0.0)

    power = tt["spring_power"][st].copy()
    noise_param = tt["spring_noise"][st]
    noisy = noise_param > 0.0
    noise_time = (tt["time"][st] + spring_particles.astype(np.float32) * 49.6198) * 2.4512 \
        + pa["next_positions"][spring_particles].sum(axis=1)
    noise = np.sin(noise_time) * (noise_param * 0.6)
    power = np.where(noisy, np.maximum(power + power * noise, 0.0), power)

    v = v - v * power[:, None]
    pa["next_positions"][spring_particles] = base_pos + v


def step_post(world, ctx):
    sp = ctx.step_particle_index
    if len(sp) == 0:
        return
    pa = world.particles
    tt = world.team
    sdt = np.float32(ctx.simulation_delta_time)

    move = ctx.entry_index("update_move", world.update_move)
    if len(move):
        pmi = world.update_move["particle"][move]
        mt = pa["team"][pmi]
        next_pos = pa["next_positions"][pmi]
        old_pos = pa["old_positions"][pmi]
        velocity_old = pa["velocity_positions"][pmi]
        depth = pa["depth"][pmi]

        friction = pa["friction"][pmi]
        cn = pa["collision_normals"][pmi]
        is_collision = (pm.length(cn) ** 2 > defs.EPSILON) & (friction > defs.EPSILON)
        static_param = tt["static_friction"][mt] * tt["scale_ratio"][mt]
        dynamic_param = tt["dynamic_friction"][mt]

        static_friction = pa["static_friction"][pmi]
        static_on = static_param > 0.0
        v = next_pos - old_pos
        tangent = v - pm.dot(v, cn)[:, None] * cn
        tangent_velocity = pm.length(tangent) / sdt
        slow = tangent_velocity < static_param
        increase = pm.saturate(static_friction + 0.04)
        decrease = pm.saturate(static_friction - np.maximum(
            (tangent_velocity - static_param) / 0.2, 0.05))
        new_static = np.where(slow, increase, decrease)
        decayed = pm.saturate(static_friction - 0.05)
        updated = np.where(is_collision, new_static, decayed)
        static_friction = np.where(static_on, updated, decayed)
        rollback = tangent * static_friction[:, None]
        rollback = np.where((static_on & is_collision)[:, None], rollback, 0.0)
        next_pos = next_pos - rollback
        velocity_old = velocity_old - rollback
        pa["static_friction"][pmi] = static_friction

        velocity = (next_pos - velocity_old) / sdt
        sq_velocity = pm.length(velocity) ** 2
        normal_velocity = pm.normalize(velocity)
        normal_velocity = np.where((sq_velocity > defs.EPSILON)[:, None], normal_velocity, 0.0)

        dynamic_on = dynamic_param > 0.0
        d = pm.dot(cn, normal_velocity)
        d = 0.5 + 0.5 * d
        d = d * d
        d = 1.0 - d
        damp = d * pm.saturate(friction * dynamic_param)
        apply = dynamic_on & is_collision & (sq_velocity >= defs.EPSILON)
        velocity = velocity - velocity * np.where(apply, damp, 0.0)[:, None]
        pa["friction"][pmi] = friction * defs.FRICTION_DAMPING_RATE

        speed_limit = tt["particle_speed_limit"][mt]
        limited = pm.clamp_vector(velocity, np.maximum(speed_limit * tt["scale_ratio"][mt], 0.0))
        velocity = np.where((speed_limit >= 0.0)[:, None], limited, velocity)

        angular = tt["angular_velocity"][mt]
        centrifugal = tt["centrifugal_acceleration"][mt]
        spin = (angular > defs.EPSILON) & (centrifugal > defs.EPSILON)
        if np.any(spin):
            axis = tt["rotation_axis"][mt]
            lpos = next_pos - tt["now_world_position"][mt]
            v2 = lpos - pm.dot(lpos, axis)[:, None] * axis
            r = pm.length(v2)
            valid = spin & (r > defs.EPSILON) & (sq_velocity >= defs.EPSILON)
            n = v2 / np.where(r > 1e-30, r, 1.0)[:, None]
            m = 1.0 + (1.0 - depth)
            f = m * angular * angular * r
            u = pm.normalize(pm.cross(axis, n))
            f = f * pm.saturate(pm.dot(normal_velocity, u))
            add = n * (f * centrifugal * 0.02)[:, None]
            velocity = velocity + np.where(valid[:, None], add, 0.0)

        velocity = velocity * tt["velocity_weight"][mt][:, None]
        pa["velocities"][pmi] = velocity
        pa["next_positions"][pmi] = next_pos

    pa["real_velocities"][sp] = (pa["next_positions"][sp] - pa["old_positions"][sp]) / sdt
    pa["old_positions"][sp] = pa["next_positions"][sp]
