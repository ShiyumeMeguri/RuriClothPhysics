import warp as wp

from ..cloth_kernel import defs as _defs
from . import dmath
from . import kernels
from . import policy
from .device_state import ClothState
from .dmath import FRICTION_MASS


from .kernels import ANGLE_LIMIT_ITERATION
from .kernels import BENDING_FIXED_INVERSE_MASS
from .kernels import COLLISION_EDGE
from .kernels import COLLISION_POINT
from .kernels import CONTACT_PATH_COLLIDER
from .kernels import CONTACT_PATH_SELF_COLLISION
from .kernels import DISTANCE_VELOCITY_ATTENUATION
from .kernels import EPSILON
from .kernels import FORCE_VELOCITY_ADD
from .kernels import FORCE_VELOCITY_ADD_WITHOUT_DEPTH
from .kernels import FORCE_VELOCITY_CHANGE
from .kernels import FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH
from .kernels import ONE_SIXTH
from .kernels import RAD2DEG
from .kernels import SCAL_FRAME_DT
from .kernels import SCAL_MAX_SIM
from .kernels import SCAL_N_ZONES
from .kernels import SCAL_POWER1
from .kernels import SCAL_POWER2
from .kernels import SCAL_POWER3
from .kernels import SCAL_SIM_DT
from .kernels import SCAL_TIME_SCALE
from .kernels import SCL_EE_COUNT
from .kernels import SCL_ERROR
from .kernels import SCL_FRAME_INDEX
from .kernels import SCL_IP_COUNT
from .kernels import SCL_PT_COUNT
from .kernels import SELF_COLLISION_INTERSECT_DIV
from .kernels import SELF_COLLISION_UNIFORM_GRID_SCALE
from .kernels import TELEPORT_RESET
from .kernels import TO_FIXED
from .kernels import VOLUME_SCALE
from .kernels import VOLUME_SIGN
from .kernels import WIND_MAX_TIME
from .kernels import WIND_MIN_SPEED
from .kernels import WIND_ZONE_MIN_MAIN
from .kernels import WIND_ZONE_RESULT_SLOTS
from .kernels import WIND_ZONE_SLOTS
from .kernels import ZONE_BOX
from .kernels import ZONE_SPHERE_DIR
from .kernels import ZONE_SPHERE_RADIAL


wp.set_module_options(policy.MODULE_OPTIONS)


ZONE_RESULT_INDICES = wp.types.vector(length=_defs.WIND_ZONE_RESULT_SLOTS, dtype=wp.int32)


ZONE_RESULT_VALUES = wp.types.vector(length=_defs.WIND_ZONE_RESULT_SLOTS, dtype=wp.float32)


WIND_SLOT_INDICES = wp.types.vector(length=_defs.WIND_ZONE_SLOTS, dtype=wp.int32)


WIND_SLOT_VALUES = wp.types.vector(length=_defs.WIND_ZONE_SLOTS, dtype=wp.float32)


CARRY_OLD_COMPONENT_POSITION = wp.constant(int(_defs.CARRY_OLD_COMPONENT_POSITION))


CARRY_OLD_COMPONENT_ROTATION = wp.constant(int(_defs.CARRY_OLD_COMPONENT_ROTATION))


CARRY_ANCHOR_SHIFT_VECTOR = wp.constant(int(_defs.CARRY_ANCHOR_SHIFT_VECTOR))


CARRY_ANCHOR_SHIFT_ROTATION = wp.constant(int(_defs.CARRY_ANCHOR_SHIFT_ROTATION))


CARRY_SMOOTHING_SHIFT_VECTOR = wp.constant(int(_defs.CARRY_SMOOTHING_SHIFT_VECTOR))


@wp.func
def resolve_team_synchronization_top_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    i = thread_index
    if state.team.enabled[i] != 0:
        target = state.team.sync_target[i]
        if target <= 0 or state.team.valid[target] == 0 or state.team.enabled[target] == 0:
            state.team.sync_top[i] = 0
        else:
            top = target
            for _hop in range(8):
                upper = state.team.sync_target[top]
                if upper <= 0 or upper == i or state.team.valid[upper] == 0 or (state.team.enabled
                        [upper] == 0):
                    break
                top = upper
            state.team.sync_top[i] = top


@wp.kernel
def resolve_team_synchronization_top(state: ClothState, substep: int, level: int,
        iteration: int):
    resolve_team_synchronization_top_element(state, wp.tid(), substep, level, iteration)


@wp.func
def snapshot_team_synchronization_clock_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    i = thread_index
    if state.team.enabled[i] != 0 and state.team.sync_top[i] > 0:
        top = state.team.sync_top[i]
        state.derived.synchronization_snapshot[i, 0] = state.team.time[top]
        state.derived.synchronization_snapshot[i, 1] = state.team.old_time[top]
        state.derived.synchronization_snapshot[i, 2] = state.team.now_update_time[top]
        state.derived.synchronization_snapshot[i, 3] = state.team.old_update_time[top]
        state.derived.synchronization_snapshot[i, 4] = state.team.frame_update_time[top]
        state.derived.synchronization_snapshot[i, 5] = state.team.frame_old_time[top]
        state.derived.synchronization_snapshot[i, 6] = state.team.time_scale[top]
        state.derived.synchronization_snapshot[i, 7] = state.team.anchor_inertia[top]
        state.derived.synchronization_snapshot[i, 8] = state.team.world_inertia[top]
        state.derived.synchronization_snapshot[i,
                9] = state.team.movement_inertia_smoothing[top]
        state.derived.synchronization_snapshot[i, 10] = state.team.movement_speed_limit[top]
        state.derived.synchronization_snapshot[i, 11] = state.team.rotation_speed_limit[top]
        state.derived.synchronization_snapshot[i, 12] = float(state.team.teleport_mode[top])
        state.derived.synchronization_snapshot[i, 13] = state.team.teleport_distance[top]
        state.derived.synchronization_snapshot[i, 14] = state.team.teleport_rotation[top]
        state.derived.synchronization_snapshot[i, 15] = state.team.component_world_position[top,
                0]
        state.derived.synchronization_snapshot[i, 16] = state.team.component_world_position[top,
                1]
        state.derived.synchronization_snapshot[i, 17] = state.team.component_world_position[top,
                2]
        state.derived.synchronization_snapshot[i, 18] = state.team.component_world_rotation[top,
                0]
        state.derived.synchronization_snapshot[i, 19] = state.team.component_world_rotation[top,
                1]
        state.derived.synchronization_snapshot[i, 20] = state.team.component_world_rotation[top,
                2]
        state.derived.synchronization_snapshot[i, 21] = state.team.component_world_rotation[top,
                3]


@wp.kernel
def snapshot_team_synchronization_clock(state: ClothState, substep: int, level: int,
        iteration: int):
    snapshot_team_synchronization_clock_element(state, wp.tid(), substep, level, iteration)


@wp.func
def apply_team_synchronization_clock_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    i = thread_index
    if state.team.enabled[i] != 0 and state.team.sync_top[i] > 0:
        state.team.time[i] = state.derived.synchronization_snapshot[i, 0]
        state.team.old_time[i] = state.derived.synchronization_snapshot[i, 1]
        state.team.now_update_time[i] = state.derived.synchronization_snapshot[i, 2]
        state.team.old_update_time[i] = state.derived.synchronization_snapshot[i, 3]
        state.team.frame_update_time[i] = state.derived.synchronization_snapshot[i, 4]
        state.team.frame_old_time[i] = state.derived.synchronization_snapshot[i, 5]
        state.team.time_scale[i] = state.derived.synchronization_snapshot[i, 6]
        state.team.anchor_inertia[i] = state.derived.synchronization_snapshot[i, 7]
        state.team.world_inertia[i] = state.derived.synchronization_snapshot[i, 8]
        state.team.movement_inertia_smoothing[i] = state.derived.synchronization_snapshot[i, 9]
        state.team.movement_speed_limit[i] = state.derived.synchronization_snapshot[i, 10]
        state.team.rotation_speed_limit[i] = state.derived.synchronization_snapshot[i, 11]
        state.team.teleport_mode[i] = int(state.derived.synchronization_snapshot[i, 12])
        state.team.teleport_distance[i] = state.derived.synchronization_snapshot[i, 13]
        state.team.teleport_rotation[i] = state.derived.synchronization_snapshot[i, 14]
        state.team.component_world_position[i, 0] = state.derived.synchronization_snapshot[i,
                15]
        state.team.component_world_position[i, 1] = state.derived.synchronization_snapshot[i,
                16]
        state.team.component_world_position[i, 2] = state.derived.synchronization_snapshot[i,
                17]
        state.team.component_world_rotation[i, 0] = state.derived.synchronization_snapshot[i,
                18]
        state.team.component_world_rotation[i, 1] = state.derived.synchronization_snapshot[i,
                19]
        state.team.component_world_rotation[i, 2] = state.derived.synchronization_snapshot[i,
                20]
        state.team.component_world_rotation[i, 3] = state.derived.synchronization_snapshot[i,
                21]


@wp.kernel
def apply_team_synchronization_clock(state: ClothState, substep: int, level: int,
        iteration: int):
    apply_team_synchronization_clock_element(state, wp.tid(), substep, level, iteration)


@wp.func
def advance_team_frame_clock_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    i = thread_index
    fdt = state.frame_scalar.frame_float[SCAL_FRAME_DT]
    global_time_scale = state.frame_scalar.frame_float[SCAL_TIME_SCALE]
    max_sim_count = state.frame_scalar.frame_int[SCAL_MAX_SIM]
    sim_dt = state.frame_scalar.frame_float[SCAL_SIM_DT]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            i):
        kernels.do_advance(i, fdt, sim_dt, max_sim_count, global_time_scale, state.team.time_reset_pending,
                state.team.time, state.team.old_time, state.team.now_update_time, state.team.old_update_time,
                state.team.frame_update_time, state.team.frame_old_time, state.team.frame_delta_time,
                state.team.time_scale, state.team.now_time_scale, state.team.update_count,
                state.team.skip_count, state.team.running)


@wp.kernel
def advance_team_frame_clock(state: ClothState, substep: int, level: int, iteration: int):
    advance_team_frame_clock_element(state, wp.tid(), substep, level, iteration)


@wp.func
def skin_particle_pose_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    p = thread_index
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            state.particle.team[p]):
        kernels.do_base_pose(p, state.particle.team, state.particle.local_positions,
                state.particle.local_normals, state.particle.local_tangents, state.particle.skin_indices,
                state.particle.skin_weights, state.particle.positions, state.particle.rotations,
                state.transform.world, state.transform.bind_pose)


@wp.kernel
def skin_particle_pose(state: ClothState, substep: int, level: int, iteration: int):
    skin_particle_pose_element(state, wp.tid(), substep, level, iteration)


@wp.func
def resolve_team_negative_scale_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    i = thread_index
    if kernels.team_frame_mask(state.team.enabled, state.team.valid,
            state.team.component_world_scale, i):
        cpx = state.team.component_world_position[i, 0]
        cpy = state.team.component_world_position[i, 1]
        cpz = state.team.component_world_position[i, 2]
        crx = state.team.component_world_rotation[i, 0]
        cry = state.team.component_world_rotation[i, 1]
        crz = state.team.component_world_rotation[i, 2]
        crw = state.team.component_world_rotation[i, 3]
        reflection_sign = kernels.component_reflection_sign(state.team.component_world_reflected,
                i)
        csx = state.team.component_world_scale[i, 0] * reflection_sign
        csy = state.team.component_world_scale[i, 1] * reflection_sign
        csz = state.team.component_world_scale[i, 2] * reflection_sign
        reflected = state.team.component_world_reflected[i] != 0
        was_reflected = state.team.old_component_world_reflected[i] != 0
        teleport = reflected != was_reflected
        state.team.negative_scale_teleport[i] = 1 if teleport else 0
        if teleport:
            component_matrix = dmath.trs_build_f64(cpx, cpy, cpz, crx, cry, crz, crw, csx, csy,
                    csz)
            old_reflection_sign = kernels.component_reflection_sign(state.team.old_component_world_reflected,
                    i)
            ocpx = state.team.old_component_world_position[i, 0]
            ocpy = state.team.old_component_world_position[i, 1]
            ocpz = state.team.old_component_world_position[i, 2]
            ocrx = state.team.old_component_world_rotation[i, 0]
            ocry = state.team.old_component_world_rotation[i, 1]
            ocrz = state.team.old_component_world_rotation[i, 2]
            ocrw = state.team.old_component_world_rotation[i, 3]
            ocsx = state.team.old_component_world_scale[i, 0] * old_reflection_sign
            ocsy = state.team.old_component_world_scale[i, 1] * old_reflection_sign
            ocsz = state.team.old_component_world_scale[i, 2] * old_reflection_sign
            old_component_inverse = dmath.trs_inverse_f64(ocpx, ocpy, ocpz, ocrx, ocry, ocrz,
                    ocrw, ocsx, ocsy, ocsz)
            component_delta = dmath.mat4_mul_f64(component_matrix, old_component_inverse)
            (nx, ny, nz) = dmath.transform_point(component_delta, ocpx, ocpy, ocpz)
            state.team.old_component_world_position[i, 0] = nx
            state.team.old_component_world_position[i, 1] = ny
            state.team.old_component_world_position[i, 2] = nz
            state.team.old_component_world_scale[i, 0] = state.team.component_world_scale[i,
                    0]
            state.team.old_component_world_scale[i, 1] = state.team.component_world_scale[i,
                    1]
            state.team.old_component_world_scale[i, 2] = state.team.component_world_scale[i,
                    2]
            state.team.old_component_world_reflected[i] = 1 if reflected else 0
            oax = state.team.old_anchor_position[i, 0]
            oay = state.team.old_anchor_position[i, 1]
            oaz = state.team.old_anchor_position[i, 2]
            (tax, tay, taz) = dmath.transform_point(component_delta, oax, oay, oaz)
            state.team.old_anchor_position[i, 0] = tax
            state.team.old_anchor_position[i, 1] = tay
            state.team.old_anchor_position[i, 2] = taz
            (tsvx, tsvy, tsvz) = dmath.transform_vector(component_delta, state.team.smoothing_velocity[i,
                    0], state.team.smoothing_velocity[i, 1], state.team.smoothing_velocity[i,
                    2])
            state.team.smoothing_velocity[i, 0] = tsvx
            state.team.smoothing_velocity[i, 1] = tsvy
            state.team.smoothing_velocity[i, 2] = tsvz


@wp.kernel
def resolve_team_negative_scale(state: ClothState, substep: int, level: int, iteration: int):
    resolve_team_negative_scale_element(state, wp.tid(), substep, level, iteration)


@wp.func
def resolve_team_frame_pose_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    i = thread_index
    if kernels.team_frame_mask(state.team.enabled, state.team.valid,
            state.team.component_world_scale, i):
        cpx = state.team.component_world_position[i, 0]
        cpy = state.team.component_world_position[i, 1]
        cpz = state.team.component_world_position[i, 2]
        crx = state.team.component_world_rotation[i, 0]
        cry = state.team.component_world_rotation[i, 1]
        crz = state.team.component_world_rotation[i, 2]
        crw = state.team.component_world_rotation[i, 3]
        csx = state.team.component_world_scale[i, 0]
        csy = state.team.component_world_scale[i, 1]
        csz = state.team.component_world_scale[i, 2]
        reflected = state.team.component_world_reflected[i] != 0
        reflection_sign = kernels.component_reflection_sign(state.team.component_world_reflected,
                i)
        teleport = state.team.negative_scale_teleport[i] != 0
        cwpx = cpx
        cwpy = cpy
        cwpz = cpz
        cwrx = crx
        cwry = cry
        cwrz = crz
        cwrw = crw
        nor_sx = wp.float64(0.0)
        nor_sy = wp.float64(0.0)
        nor_sz = wp.float64(0.0)
        tan_sx = wp.float64(0.0)
        tan_sy = wp.float64(0.0)
        tan_sz = wp.float64(0.0)
        pos_sx = wp.float64(0.0)
        pos_sy = wp.float64(0.0)
        pos_sz = wp.float64(0.0)
        fcount = int(0)
        seg0 = state.derived.center_fixed_csr_offsets[i]
        seg1 = state.derived.center_fixed_csr_offsets[i + 1]
        for e in range(seg0, seg1):
            fp = state.center_fixed.particle[state.derived.center_fixed_csr_order[e]]
            rx = state.particle.rotations[fp, 0]
            ry = state.particle.rotations[fp, 1]
            rz = state.particle.rotations[fp, 2]
            rw = state.particle.rotations[fp, 3]
            if reflected:
                (nnx, nny, nnz) = dmath.quat_to_normal(rx, ry, rz, rw)
                (ttx, tty, ttz) = dmath.quat_to_tangent(rx, ry, rz, rw)
                (rx, ry, rz, rw) = dmath.to_rotation(dmath.negate(nnx), dmath.negate(nny),
                        dmath.negate(nnz), dmath.negate(ttx), dmath.negate(tty),
                        dmath.negate(ttz))
            (rx, ry, rz, rw) = dmath.quat_mul(rx, ry, rz, rw, state.particle.vertex_bind_pose_rotations[fp,
                    0], state.particle.vertex_bind_pose_rotations[fp, 1],
                    state.particle.vertex_bind_pose_rotations[fp, 2], state.particle.vertex_bind_pose_rotations[fp,
                    3])
            (norx, nory, norz) = dmath.quat_to_normal(rx, ry, rz, rw)
            (tanx, tany, tanz) = dmath.quat_to_tangent(rx, ry, rz, rw)
            nor_sx += wp.float64(norx * reflection_sign)
            nor_sy += wp.float64(nory * reflection_sign)
            nor_sz += wp.float64(norz * reflection_sign)
            tan_sx += wp.float64(tanx * reflection_sign)
            tan_sy += wp.float64(tany * reflection_sign)
            tan_sz += wp.float64(tanz * reflection_sign)
            pos_sx += wp.float64(state.particle.positions[fp, 0])
            pos_sy += wp.float64(state.particle.positions[fp, 1])
            pos_sz += wp.float64(state.particle.positions[fp, 2])
            fcount += 1
        if fcount > 0:
            nl = wp.sqrt(nor_sx * nor_sx + nor_sy * nor_sy + nor_sz * nor_sz)
            tl = wp.sqrt(tan_sx * tan_sx + tan_sy * tan_sy + tan_sz * tan_sz)
            if nl > wp.float64(1e-30) and tl > wp.float64(1e-30):
                cwpx = wp.float32(pos_sx / wp.float64(fcount))
                cwpy = wp.float32(pos_sy / wp.float64(fcount))
                cwpz = wp.float32(pos_sz / wp.float64(fcount))
                (cwrx, cwry, cwrz, cwrw) = dmath.to_rotation(wp.float32(nor_sx / nl),
                        wp.float32(nor_sy / nl), wp.float32(nor_sz / nl), wp.float32(tan_sx / tl),
                        wp.float32(tan_sy / tl), wp.float32(tan_sz / tl))
        if teleport:
            frame_matrix = dmath.trs_build_f64(cwpx, cwpy, cwpz, cwrx, cwry, cwrz, cwrw,
                    csx * reflection_sign, csy * reflection_sign, csz * reflection_sign)
            old_frame_sign = kernels.component_reflection_sign(state.team.old_frame_world_reflected,
                    i)
            old_frame_inverse = dmath.trs_inverse_f64(state.team.old_frame_world_position[i, 0],
                    state.team.old_frame_world_position[i, 1], state.team.old_frame_world_position[i,
                    2], state.team.old_frame_world_rotation[i, 0], state.team.old_frame_world_rotation[i,
                    1], state.team.old_frame_world_rotation[i, 2], state.team.old_frame_world_rotation[i,
                    3], state.team.old_frame_world_scale[i, 0] * old_frame_sign,
                    state.team.old_frame_world_scale[i, 1] * old_frame_sign,
                    state.team.old_frame_world_scale[i, 2] * old_frame_sign)
            state.team.negative_scale_matrix[i] = dmath.mat4_mul_f64(frame_matrix,
                    old_frame_inverse)
        state.team.frame_world_position[i, 0] = cwpx
        state.team.frame_world_position[i, 1] = cwpy
        state.team.frame_world_position[i, 2] = cwpz
        state.team.frame_world_rotation[i, 0] = cwrx
        state.team.frame_world_rotation[i, 1] = cwry
        state.team.frame_world_rotation[i, 2] = cwrz
        state.team.frame_world_rotation[i, 3] = cwrw
        state.team.frame_world_scale[i, 0] = csx
        state.team.frame_world_scale[i, 1] = csy
        state.team.frame_world_scale[i, 2] = csz


@wp.kernel
def resolve_team_frame_pose(state: ClothState, substep: int, level: int, iteration: int):
    resolve_team_frame_pose_element(state, wp.tid(), substep, level, iteration)


@wp.func
def advance_team_component_inertia_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    i = thread_index
    if kernels.team_frame_mask(state.team.enabled, state.team.valid,
            state.team.component_world_scale, i):
        cpx = state.team.component_world_position[i, 0]
        cpy = state.team.component_world_position[i, 1]
        cpz = state.team.component_world_position[i, 2]
        crx = state.team.component_world_rotation[i, 0]
        cry = state.team.component_world_rotation[i, 1]
        crz = state.team.component_world_rotation[i, 2]
        crw = state.team.component_world_rotation[i, 3]
        csx = state.team.component_world_scale[i, 0]
        csy = state.team.component_world_scale[i, 1]
        csz = state.team.component_world_scale[i, 2]
        init_scale_len = wp.float64(dmath.length3(state.team.init_scale[i, 0],
                state.team.init_scale[i, 1], state.team.init_scale[i, 2]))
        if init_scale_len < wp.float64(1e-30):
            init_scale_len = wp.float64(1e-30)
        csr_ratio = wp.float64(dmath.length3(csx, csy, csz)) / init_scale_len
        teleport = state.team.negative_scale_teleport[i] != 0
        cwpx = state.team.frame_world_position[i, 0]
        cwpy = state.team.frame_world_position[i, 1]
        cwpz = state.team.frame_world_position[i, 2]
        cwrx = state.team.frame_world_rotation[i, 0]
        cwry = state.team.frame_world_rotation[i, 1]
        cwrz = state.team.frame_world_rotation[i, 2]
        cwrw = state.team.frame_world_rotation[i, 3]
        ocp_x = state.team.old_component_world_position[i, 0]
        ocp_y = state.team.old_component_world_position[i, 1]
        ocp_z = state.team.old_component_world_position[i, 2]
        ocr_x = state.team.old_component_world_rotation[i, 0]
        ocr_y = state.team.old_component_world_rotation[i, 1]
        ocr_z = state.team.old_component_world_rotation[i, 2]
        ocr_w = state.team.old_component_world_rotation[i, 3]
        adv_x = float(0.0)
        adv_y = float(0.0)
        adv_z = float(0.0)
        adr_x = float(0.0)
        adr_y = float(0.0)
        adr_z = float(0.0)
        adr_w = float(1.0)
        has_anc = state.team.has_anchor[i] != 0
        had_anc = state.team.had_anchor[i] != 0
        anchor_reset = state.team.reset_pending[i] != 0
        if has_anc and (not had_anc):
            anchor_reset = wp.bool(True)
        if had_anc and (not has_anc):
            anchor_reset = wp.bool(True)
        state.team.had_anchor[i] = 1 if has_anc else 0
        if anchor_reset:
            (iqx, iqy, iqz, iqw) = dmath.quat_inverse(state.team.anchor_rotation[i, 0],
                    state.team.anchor_rotation[i, 1], state.team.anchor_rotation[i, 2],
                    state.team.anchor_rotation[i, 3])
            (alx, aly, alz) = dmath.quat_rotate(iqx, iqy, iqz, iqw, cpx - state.team.anchor_position[i,
                    0], cpy - state.team.anchor_position[i, 1], cpz - state.team.anchor_position[i,
                    2])
            state.team.old_anchor_position[i, 0] = state.team.anchor_position[i, 0]
            state.team.old_anchor_position[i, 1] = state.team.anchor_position[i, 1]
            state.team.old_anchor_position[i, 2] = state.team.anchor_position[i, 2]
            state.team.old_anchor_rotation[i, 0] = state.team.anchor_rotation[i, 0]
            state.team.old_anchor_rotation[i, 1] = state.team.anchor_rotation[i, 1]
            state.team.old_anchor_rotation[i, 2] = state.team.anchor_rotation[i, 2]
            state.team.old_anchor_rotation[i, 3] = state.team.anchor_rotation[i, 3]
            state.team.anchor_component_local_position[i, 0] = alx
            state.team.anchor_component_local_position[i, 1] = aly
            state.team.anchor_component_local_position[i, 2] = alz
        if has_anc:
            (rlx, rly, rlz) = dmath.quat_rotate(state.team.anchor_rotation[i, 0],
                    state.team.anchor_rotation[i, 1], state.team.anchor_rotation[i, 2],
                    state.team.anchor_rotation[i, 3], state.team.anchor_component_local_position[i,
                    0], state.team.anchor_component_local_position[i, 1],
                    state.team.anchor_component_local_position[i, 2])
            dvx = rlx + state.team.anchor_position[i, 0] - ocp_x
            dvy = rly + state.team.anchor_position[i, 1] - ocp_y
            dvz = rlz + state.team.anchor_position[i, 2] - ocp_z
            (ioax, ioay, ioaz, ioaw) = dmath.quat_inverse(state.team.old_anchor_rotation[i, 0],
                    state.team.old_anchor_rotation[i, 1], state.team.old_anchor_rotation[i, 2],
                    state.team.old_anchor_rotation[i, 3])
            (drx, dry, drz, drw) = dmath.quat_mul(state.team.anchor_rotation[i, 0],
                    state.team.anchor_rotation[i, 1], state.team.anchor_rotation[i, 2],
                    state.team.anchor_rotation[i, 3], ioax, ioay, ioaz, ioaw)
            a_ratio = 1.0 - state.team.anchor_inertia[i]
            adv_x = dvx * a_ratio
            adv_y = dvy * a_ratio
            adv_z = dvz * a_ratio
            (adr_x, adr_y, adr_z, adr_w) = dmath.quat_slerp(0.0, 0.0, 0.0, 1.0, drx, dry, drz,
                    drw, a_ratio)
            ocp_x = ocp_x + adv_x
            ocp_y = ocp_y + adv_y
            ocp_z = ocp_z + adv_z
            (ocr_x, ocr_y, ocr_z, ocr_w) = dmath.quat_mul(adr_x, adr_y, adr_z, adr_w, ocr_x, ocr_y,
                    ocr_z, ocr_w)
            state.team.inertia_shift[i] = 1
        fdvx = cpx - ocp_x
        fdvy = cpy - ocp_y
        fdvz = cpz - ocp_z
        fda = dmath.quat_angle(ocr_x, ocr_y, ocr_z, ocr_w, crx, cry, crz, crw)
        if state.team.teleport_mode[i] != 0 and state.team.reset_pending[i] == 0:
            far = wp.float64(dmath.length3(fdvx, fdvy,
                    fdvz)) >= wp.float64(state.team.teleport_distance[i]) * csr_ratio
            spun = fda * RAD2DEG >= state.team.teleport_rotation[i]
            if far or spun:
                if state.team.teleport_mode[i] == TELEPORT_RESET:
                    state.team.reset_pending[i] = 1
                else:
                    state.team.keep_teleport_pending[i] = 1
        reset = state.team.reset_pending[i] != 0
        keep = state.team.keep_teleport_pending[i] != 0
        sdv_x = float(0.0)
        sdv_y = float(0.0)
        sdv_z = float(0.0)
        if state.team.movement_inertia_smoothing[i] >= 1e-06 and (not (keep or reset)):
            running = state.team.running[i] != 0
            fdt_i = state.team.frame_delta_time[i]
            if fdt_i > 0.0:
                dvx = fdvx / fdt_i
                dvy = fdvy / fdt_i
                dvz = fdvz / fdt_i
            else:
                dvx = float(0.0)
                dvy = float(0.0)
                dvz = float(0.0)
            limit = state.team.movement_speed_limit[i] * wp.float32(csr_ratio)
            mlim = limit if limit > 0.0 else float(0.0)
            (cvx, cvy, cvz) = dmath.clamp_vector(dvx, dvy, dvz, mlim)
            if limit >= 0.0:
                dvx = cvx
                dvy = cvy
                dvz = cvz
            mis = state.team.movement_inertia_smoothing[i]
            om = 1.0 - mis
            avg = dmath.saturate(om * om * om * 0.99 + 0.01)
            svx = state.team.smoothing_velocity[i, 0]
            svy = state.team.smoothing_velocity[i, 1]
            svz = state.team.smoothing_velocity[i, 2]
            smx = dmath.lerp(svx, dvx, avg)
            smy = dmath.lerp(svy, dvy, avg)
            smz = dmath.lerp(svz, dvz, avg)
            if running:
                state.team.smoothing_velocity[i, 0] = smx
                state.team.smoothing_velocity[i, 1] = smy
                state.team.smoothing_velocity[i, 2] = smz
                svx = smx
                svy = smy
                svz = smz
            spx = cpx - svx * fdt_i
            spy = cpy - svy * fdt_i
            spz = cpz - svz * fdt_i
            sdv_x = spx - ocp_x
            sdv_y = spy - ocp_y
            sdv_z = spz - ocp_z
            ocp_x = spx
            ocp_y = spy
            ocp_z = spz
            state.team.inertia_shift[i] = 1
        if reset:
            state.team.old_component_world_position[i, 0] = cpx
            state.team.old_component_world_position[i, 1] = cpy
            state.team.old_component_world_position[i, 2] = cpz
            state.team.old_component_world_rotation[i, 0] = crx
            state.team.old_component_world_rotation[i, 1] = cry
            state.team.old_component_world_rotation[i, 2] = crz
            state.team.old_component_world_rotation[i, 3] = crw
            state.team.old_component_world_scale[i, 0] = csx
            state.team.old_component_world_scale[i, 1] = csy
            state.team.old_component_world_scale[i, 2] = csz
            state.team.old_component_world_reflected[i] = state.team.component_world_reflected[i]
            ocp_x = cpx
            ocp_y = cpy
            ocp_z = cpz
            ocr_x = crx
            ocr_y = cry
            ocr_z = crz
            ocr_w = crw
        if reset or (teleport and (not reset)):
            state.team.old_frame_world_position[i, 0] = cwpx
            state.team.old_frame_world_position[i, 1] = cwpy
            state.team.old_frame_world_position[i, 2] = cwpz
            state.team.old_frame_world_rotation[i, 0] = cwrx
            state.team.old_frame_world_rotation[i, 1] = cwry
            state.team.old_frame_world_rotation[i, 2] = cwrz
            state.team.old_frame_world_rotation[i, 3] = cwrw
            state.team.old_frame_world_scale[i, 0] = csx
            state.team.old_frame_world_scale[i, 1] = csy
            state.team.old_frame_world_scale[i, 2] = csz
            state.team.old_frame_world_reflected[i] = state.team.component_world_reflected[i]
            state.team.now_world_position[i, 0] = cwpx
            state.team.now_world_position[i, 1] = cwpy
            state.team.now_world_position[i, 2] = cwpz
            state.team.now_world_rotation[i, 0] = cwrx
            state.team.now_world_rotation[i, 1] = cwry
            state.team.now_world_rotation[i, 2] = cwrz
            state.team.now_world_rotation[i, 3] = cwrw
            state.team.old_world_position[i, 0] = cwpx
            state.team.old_world_position[i, 1] = cwpy
            state.team.old_world_position[i, 2] = cwpz
            state.team.old_world_rotation[i, 0] = cwrx
            state.team.old_world_rotation[i, 1] = cwry
            state.team.old_world_rotation[i, 2] = cwrz
            state.team.old_world_rotation[i, 3] = cwrw
        state.derived.frame_transform_carry[i, CARRY_OLD_COMPONENT_POSITION + 0] = ocp_x
        state.derived.frame_transform_carry[i, CARRY_OLD_COMPONENT_POSITION + 1] = ocp_y
        state.derived.frame_transform_carry[i, CARRY_OLD_COMPONENT_POSITION + 2] = ocp_z
        state.derived.frame_transform_carry[i, CARRY_OLD_COMPONENT_ROTATION + 0] = ocr_x
        state.derived.frame_transform_carry[i, CARRY_OLD_COMPONENT_ROTATION + 1] = ocr_y
        state.derived.frame_transform_carry[i, CARRY_OLD_COMPONENT_ROTATION + 2] = ocr_z
        state.derived.frame_transform_carry[i, CARRY_OLD_COMPONENT_ROTATION + 3] = ocr_w
        state.derived.frame_transform_carry[i, CARRY_ANCHOR_SHIFT_VECTOR + 0] = adv_x
        state.derived.frame_transform_carry[i, CARRY_ANCHOR_SHIFT_VECTOR + 1] = adv_y
        state.derived.frame_transform_carry[i, CARRY_ANCHOR_SHIFT_VECTOR + 2] = adv_z
        state.derived.frame_transform_carry[i, CARRY_ANCHOR_SHIFT_ROTATION + 0] = adr_x
        state.derived.frame_transform_carry[i, CARRY_ANCHOR_SHIFT_ROTATION + 1] = adr_y
        state.derived.frame_transform_carry[i, CARRY_ANCHOR_SHIFT_ROTATION + 2] = adr_z
        state.derived.frame_transform_carry[i, CARRY_ANCHOR_SHIFT_ROTATION + 3] = adr_w
        state.derived.frame_transform_carry[i, CARRY_SMOOTHING_SHIFT_VECTOR + 0] = sdv_x
        state.derived.frame_transform_carry[i, CARRY_SMOOTHING_SHIFT_VECTOR + 1] = sdv_y
        state.derived.frame_transform_carry[i, CARRY_SMOOTHING_SHIFT_VECTOR + 2] = sdv_z


@wp.kernel
def advance_team_component_inertia(state: ClothState, substep: int, level: int, iteration: int):
    advance_team_component_inertia_element(state, wp.tid(), substep, level, iteration)


@wp.func
def resolve_team_world_inertia_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    i = thread_index
    sim_dt = state.frame_scalar.frame_float[SCAL_SIM_DT]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid,
            state.team.component_world_scale, i):
        cpx = state.team.component_world_position[i, 0]
        cpy = state.team.component_world_position[i, 1]
        cpz = state.team.component_world_position[i, 2]
        crx = state.team.component_world_rotation[i, 0]
        cry = state.team.component_world_rotation[i, 1]
        crz = state.team.component_world_rotation[i, 2]
        crw = state.team.component_world_rotation[i, 3]
        csx = state.team.component_world_scale[i, 0]
        csy = state.team.component_world_scale[i, 1]
        csz = state.team.component_world_scale[i, 2]
        init_scale_len = wp.float64(dmath.length3(state.team.init_scale[i, 0],
                state.team.init_scale[i, 1], state.team.init_scale[i, 2]))
        if init_scale_len < wp.float64(1e-30):
            init_scale_len = wp.float64(1e-30)
        csr_ratio = wp.float64(dmath.length3(csx, csy, csz)) / init_scale_len
        reset = state.team.reset_pending[i] != 0
        keep = state.team.keep_teleport_pending[i] != 0
        ocp_x = state.derived.frame_transform_carry[i, CARRY_OLD_COMPONENT_POSITION + 0]
        ocp_y = state.derived.frame_transform_carry[i, CARRY_OLD_COMPONENT_POSITION + 1]
        ocp_z = state.derived.frame_transform_carry[i, CARRY_OLD_COMPONENT_POSITION + 2]
        ocr_x = state.derived.frame_transform_carry[i, CARRY_OLD_COMPONENT_ROTATION + 0]
        ocr_y = state.derived.frame_transform_carry[i, CARRY_OLD_COMPONENT_ROTATION + 1]
        ocr_z = state.derived.frame_transform_carry[i, CARRY_OLD_COMPONENT_ROTATION + 2]
        ocr_w = state.derived.frame_transform_carry[i, CARRY_OLD_COMPONENT_ROTATION + 3]
        adv_x = state.derived.frame_transform_carry[i, CARRY_ANCHOR_SHIFT_VECTOR + 0]
        adv_y = state.derived.frame_transform_carry[i, CARRY_ANCHOR_SHIFT_VECTOR + 1]
        adv_z = state.derived.frame_transform_carry[i, CARRY_ANCHOR_SHIFT_VECTOR + 2]
        adr_x = state.derived.frame_transform_carry[i, CARRY_ANCHOR_SHIFT_ROTATION + 0]
        adr_y = state.derived.frame_transform_carry[i, CARRY_ANCHOR_SHIFT_ROTATION + 1]
        adr_z = state.derived.frame_transform_carry[i, CARRY_ANCHOR_SHIFT_ROTATION + 2]
        adr_w = state.derived.frame_transform_carry[i, CARRY_ANCHOR_SHIFT_ROTATION + 3]
        sdv_x = state.derived.frame_transform_carry[i, CARRY_SMOOTHING_SHIFT_VECTOR + 0]
        sdv_y = state.derived.frame_transform_carry[i, CARRY_SMOOTHING_SHIFT_VECTOR + 1]
        sdv_z = state.derived.frame_transform_carry[i, CARRY_SMOOTHING_SHIFT_VECTOR + 2]
        wpx = ocp_x
        wpy = ocp_y
        wpz = ocp_z
        wrx = ocr_x
        wry = ocr_y
        wrz = ocr_z
        wrw = ocr_w
        shv_x = float(0.0)
        shv_y = float(0.0)
        shv_z = float(0.0)
        shr_x = float(0.0)
        shr_y = float(0.0)
        shr_z = float(0.0)
        shr_w = float(1.0)
        if reset:
            state.team.smoothing_velocity[i, 0] = 0.0
            state.team.smoothing_velocity[i, 1] = 0.0
            state.team.smoothing_velocity[i, 2] = 0.0
            sdv_x = float(0.0)
            sdv_y = float(0.0)
            sdv_z = float(0.0)
        if not reset:
            shv_x = cpx - ocp_x
            shv_y = cpy - ocp_y
            shv_z = cpz - ocp_z
            (iox, ioy, ioz, iow) = dmath.quat_inverse(ocr_x, ocr_y, ocr_z, ocr_w)
            (shr_x, shr_y, shr_z, shr_w) = dmath.quat_mul(crx, cry, crz, crw, iox, ioy, ioz,
                    iow)
            msr = float(0.0)
            rsr = float(0.0)
            keep_now = keep or state.team.culling_invisible[i] != 0
            if keep_now:
                movement_shift = float(1.0)
            else:
                movement_shift = 1.0 - state.team.world_inertia[i]
            rotation_shift = movement_shift
            if movement_shift > EPSILON:
                state.team.inertia_shift[i] = 1
                msr = movement_shift
                rsr = rotation_shift
                wpx = dmath.lerp(wpx, cpx, movement_shift)
                wpy = dmath.lerp(wpy, cpy, movement_shift)
                wpz = dmath.lerp(wpz, cpz, movement_shift)
                (wrx, wry, wrz, wrw) = dmath.quat_slerp(wrx, wry, wrz, wrw, crx, cry, crz, crw,
                        rotation_shift)
            movement_limit = wp.float64(state.team.movement_speed_limit[i]) * csr_ratio
            rotation_limit = state.team.rotation_speed_limit[i]
            dvx = cpx - wpx
            dvy = cpy - wpy
            dvz = cpz - wpz
            dang = dmath.quat_angle(wrx, wry, wrz, wrw, crx, cry, crz, crw)
            fdt_l = state.team.frame_delta_time[i]
            if fdt_l > 0.0:
                frame_speed = dmath.length3(dvx, dvy, dvz) / fdt_l
                frame_rot_speed = dang * RAD2DEG / fdt_l
            else:
                frame_speed = float(0.0)
                frame_rot_speed = float(0.0)
            over_move = wp.float64(frame_speed) > movement_limit and state.team.movement_speed_limit[
                    i] >= 0.0
            if over_move:
                state.team.inertia_shift[i] = 1
                denom_fs = frame_speed if frame_speed > 0.0 else float(1.0)
                mlr = dmath.saturate((frame_speed - wp.float32(movement_limit)) / denom_fs)
            else:
                mlr = float(0.0)
            msr = msr + (1.0 - msr) * mlr
            if over_move:
                wpx = dmath.lerp(wpx, cpx, mlr)
                wpy = dmath.lerp(wpy, cpy, mlr)
                wpz = dmath.lerp(wpz, cpz, mlr)
            over_rot = frame_rot_speed > rotation_limit and rotation_limit >= 0.0
            if over_rot:
                state.team.inertia_shift[i] = 1
                denom_frs = frame_rot_speed if frame_rot_speed > 0.0 else float(1.0)
                rlr = dmath.saturate((frame_rot_speed - rotation_limit) / denom_frs)
            else:
                rlr = float(0.0)
            rsr = rsr + (1.0 - rsr) * rlr
            if over_rot:
                (wrx, wry, wrz, wrw) = dmath.quat_slerp(wrx, wry, wrz, wrw, crx, cry, crz, crw,
                        rlr)
            osr = wp.float64(0.0)
            skip = state.team.skip_count[i]
            scaled_dt = fdt_l * state.team.now_time_scale[i]
            if skip > 0 and scaled_dt > 0.0:
                sr = wp.float64(skip) * wp.float64(sim_dt) / wp.float64(scaled_dt)
                if sr < wp.float64(0.0):
                    sr = wp.float64(0.0)
                elif sr > wp.float64(1.0):
                    sr = wp.float64(1.0)
                osr = osr + (wp.float64(1.0) - osr) * sr
            vw = state.team.velocity_weight[i]
            if vw < 1.0:
                osr = osr + (wp.float64(1.0) - osr) * (wp.float64(1.0) - wp.float64(vw))
            nts = state.team.now_time_scale[i]
            if nts < 1.0:
                osr = osr + (wp.float64(1.0) - osr) * (wp.float64(1.0) - wp.float64(nts))
            msr_final = wp.float64(msr)
            rsr_final = wp.float64(rsr)
            if osr > wp.float64(0.0):
                state.team.inertia_shift[i] = 1
                msr_final = msr_final + (wp.float64(1.0) - msr_final) * osr
                osr_f32 = wp.float32(osr)
                wpx = dmath.lerp(wpx, cpx, osr_f32)
                wpy = dmath.lerp(wpy, cpy, osr_f32)
                wpz = dmath.lerp(wpz, cpz, osr_f32)
                rsr_final = rsr_final + (wp.float64(1.0) - rsr_final) * osr
                (wrx, wry, wrz, wrw) = dmath.quat_slerp(wrx, wry, wrz, wrw, crx, cry, crz, crw,
                        osr_f32)
            if state.team.inertia_shift[i] != 0:
                vecx = wp.float64(shv_x) * msr_final + wp.float64(adv_x) + wp.float64(sdv_x)
                vecy = wp.float64(shv_y) * msr_final + wp.float64(adv_y) + wp.float64(sdv_y)
                vecz = wp.float64(shv_z) * msr_final + wp.float64(adv_z) + wp.float64(sdv_z)
                (rqx, rqy, rqz, rqw) = dmath.quat_slerp(0.0, 0.0, 0.0, 1.0, shr_x, shr_y, shr_z,
                        shr_w, wp.float32(rsr_final))
                (rqx, rqy, rqz, rqw) = dmath.quat_mul(adr_x, adr_y, adr_z, adr_w, rqx, rqy, rqz,
                        rqw)
                state.team.frame_component_shift_vector[i, 0] = wp.float32(vecx)
                state.team.frame_component_shift_vector[i, 1] = wp.float32(vecy)
                state.team.frame_component_shift_vector[i, 2] = wp.float32(vecz)
                state.team.frame_component_shift_rotation[i, 0] = rqx
                state.team.frame_component_shift_rotation[i, 1] = rqy
                state.team.frame_component_shift_rotation[i, 2] = rqz
                state.team.frame_component_shift_rotation[i, 3] = rqw
                oc_x = state.team.old_component_world_position[i, 0]
                oc_y = state.team.old_component_world_position[i, 1]
                oc_z = state.team.old_component_world_position[i, 2]
                (rlx1, rly1, rlz1) = dmath.quat_rotate(rqx, rqy, rqz, rqw, state.team.old_frame_world_position[i,
                        0] - oc_x, state.team.old_frame_world_position[i, 1] - oc_y,
                        state.team.old_frame_world_position[i, 2] - oc_z)
                state.team.old_frame_world_position[i,
                        0] = wp.float32(wp.float64(rlx1 + oc_x) + vecx)
                state.team.old_frame_world_position[i,
                        1] = wp.float32(wp.float64(rly1 + oc_y) + vecy)
                state.team.old_frame_world_position[i,
                        2] = wp.float32(wp.float64(rlz1 + oc_z) + vecz)
                (oqx, oqy, oqz, oqw) = dmath.quat_mul(rqx, rqy, rqz, rqw, state.team.old_frame_world_rotation[i,
                        0], state.team.old_frame_world_rotation[i, 1], state.team.old_frame_world_rotation[i,
                        2], state.team.old_frame_world_rotation[i, 3])
                state.team.old_frame_world_rotation[i, 0] = oqx
                state.team.old_frame_world_rotation[i, 1] = oqy
                state.team.old_frame_world_rotation[i, 2] = oqz
                state.team.old_frame_world_rotation[i, 3] = oqw
                (rlx2, rly2, rlz2) = dmath.quat_rotate(rqx, rqy, rqz, rqw, state.team.now_world_position[i,
                        0] - oc_x, state.team.now_world_position[i, 1] - oc_y, state.team.now_world_position[i,
                        2] - oc_z)
                state.team.now_world_position[i, 0] = wp.float32(wp.float64(rlx2 + oc_x) + vecx)
                state.team.now_world_position[i, 1] = wp.float32(wp.float64(rly2 + oc_y) + vecy)
                state.team.now_world_position[i, 2] = wp.float32(wp.float64(rlz2 + oc_z) + vecz)
                (nqx, nqy, nqz, nqw) = dmath.quat_mul(rqx, rqy, rqz, rqw, state.team.now_world_rotation[i,
                        0], state.team.now_world_rotation[i, 1], state.team.now_world_rotation[i,
                        2], state.team.now_world_rotation[i, 3])
                state.team.now_world_rotation[i, 0] = nqx
                state.team.now_world_rotation[i, 1] = nqy
                state.team.now_world_rotation[i, 2] = nqz
                state.team.now_world_rotation[i, 3] = nqw
            else:
                state.team.frame_component_shift_vector[i, 0] = shv_x
                state.team.frame_component_shift_vector[i, 1] = shv_y
                state.team.frame_component_shift_vector[i, 2] = shv_z
                state.team.frame_component_shift_rotation[i, 0] = shr_x
                state.team.frame_component_shift_rotation[i, 1] = shr_y
                state.team.frame_component_shift_rotation[i, 2] = shr_z
                state.team.frame_component_shift_rotation[i, 3] = shr_w
        if reset:
            state.team.frame_component_shift_vector[i, 0] = 0.0
            state.team.frame_component_shift_vector[i, 1] = 0.0
            state.team.frame_component_shift_vector[i, 2] = 0.0
            state.team.frame_component_shift_rotation[i, 0] = 0.0
            state.team.frame_component_shift_rotation[i, 1] = 0.0
            state.team.frame_component_shift_rotation[i, 2] = 0.0
            state.team.frame_component_shift_rotation[i, 3] = 1.0
        mvx = cpx - wpx
        mvy = cpy - wpy
        mvz = cpz - wpz
        mlen = dmath.length3(mvx, mvy, mvz)
        fdt_m = state.team.frame_delta_time[i]
        if fdt_m > 0.0:
            speed = mlen / fdt_m
        else:
            speed = float(0.0)
        nts_m = state.team.now_time_scale[i]
        if nts_m > 1e-06:
            speed = speed * (1.0 / nts_m)
        else:
            speed = float(0.0)
        state.team.frame_moving_speed[i] = speed
        if mlen > 1e-06:
            state.team.frame_moving_direction[i, 0] = mvx / mlen
            state.team.frame_moving_direction[i, 1] = mvy / mlen
            state.team.frame_moving_direction[i, 2] = mvz / mlen
        else:
            state.team.frame_moving_direction[i, 0] = 0.0
            state.team.frame_moving_direction[i, 1] = 0.0
            state.team.frame_moving_direction[i, 2] = 0.0
        if state.team.reset_pending[i] != 0 or state.team.time_reset_pending[i] != 0:
            if state.team.stablization_time[i] > 1e-06:
                wgt = float(0.0)
            else:
                wgt = float(1.0)
            state.team.velocity_weight[i] = wgt
            state.team.blend_weight[i] = wgt


@wp.kernel
def resolve_team_world_inertia(state: ClothState, substep: int, level: int, iteration: int):
    resolve_team_world_inertia_element(state, wp.tid(), substep, level, iteration)


@wp.func
def resolve_particle_wind_zones_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    mt = state.particle.team[p]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt):
        n_zones = state.frame_scalar.frame_int[SCAL_N_ZONES]
        res_zone_id = ZONE_RESULT_INDICES()
        res_main = ZONE_RESULT_VALUES()
        res_dx = ZONE_RESULT_VALUES()
        res_dy = ZONE_RESULT_VALUES()
        res_dz = ZONE_RESULT_VALUES()
        res_turb = ZONE_RESULT_VALUES()
        count = int(0)
        if n_zones > 0 and state.team.wind_influence[mt] > EPSILON:
            cx64 = wp.float64(state.particle.positions[p, 0])
            cy64 = wp.float64(state.particle.positions[p, 1])
            cz64 = wp.float64(state.particle.positions[p, 2])
            min_volume = float(wp.inf)
            addition_count = int(0)
            latest_valid = wp.bool(False)
            latest_id = int(0)
            for zi in range(n_zones):
                is_add = state.zone.is_addition[zi] != 0
                if is_add and addition_count >= 3:
                    continue
                mode = state.zone.mode[zi]
                zvol = state.zone.zone_volume[zi]
                wm = state.zone.world_to_local[zi]
                lxx = wm[0, 0] * cx64 + wm[0, 1] * cy64 + wm[0, 2] * cz64 + wm[0, 3]
                lyy = wm[1, 0] * cx64 + wm[1, 1] * cy64 + wm[1, 2] * cz64 + wm[1, 3]
                lzz = wm[2, 0] * cx64 + wm[2, 1] * cy64 + wm[2, 2] * cz64 + wm[2, 3]
                llen = wp.sqrt(lxx * lxx + lyy * lyy + lzz * lzz)
                skip_zone = wp.bool(False)
                if mode == ZONE_BOX:
                    if wp.abs(lxx) * wp.float64(2.0) > wp.float64(state.zone.size[zi,
                            0]) or wp.abs(lyy) * wp.float64(2.0) > wp.float64(state.zone.size[zi,
                            1]) or wp.abs(lzz) * wp.float64(2.0) > wp.float64(state.zone.size[zi,
                            2]):
                        skip_zone = wp.bool(True)
                elif mode == ZONE_SPHERE_DIR or mode == ZONE_SPHERE_RADIAL:
                    if llen > wp.float64(state.zone.size[zi, 0]):
                        skip_zone = wp.bool(True)
                if skip_zone:
                    continue
                if not is_add and zvol > min_volume:
                    continue
                dirx = state.zone.world_direction[zi, 0]
                diry = state.zone.world_direction[zi, 1]
                dirz = state.zone.world_direction[zi, 2]
                zmain = state.zone.main[zi]
                if mode == ZONE_SPHERE_RADIAL:
                    if llen <= wp.float64(1e-06):
                        continue
                    vx64 = cx64 - wp.float64(state.zone.world_position[zi, 0])
                    vy64 = cy64 - wp.float64(state.zone.world_position[zi, 1])
                    vz64 = cz64 - wp.float64(state.zone.world_position[zi, 2])
                    vlen = wp.sqrt(vx64 * vx64 + vy64 * vy64 + vz64 * vz64)
                    dirx = wp.float32(vx64 / vlen)
                    diry = wp.float32(vy64 / vlen)
                    dirz = wp.float32(vz64 / vlen)
                    depth = llen / wp.float64(state.zone.size[zi, 0])
                    if depth < wp.float64(0.0):
                        depth = wp.float64(0.0)
                    elif depth > wp.float64(1.0):
                        depth = wp.float64(1.0)
                    zmain = zmain * dmath.evaluate_team_lut_clamp01(state.zone.attenuation_lut,
                            zi, wp.float32(depth))
                zid = state.zone.zone_id[zi]
                zturb = state.zone.turbulence[zi]
                registrable = zmain > WIND_ZONE_MIN_MAIN
                if is_add:
                    if registrable:
                        res_zone_id[count] = zid
                        res_main[count] = zmain
                        res_dx[count] = dirx
                        res_dy[count] = diry
                        res_dz[count] = dirz
                        res_turb[count] = zturb
                        count += 1
                    addition_count += 1
                else:
                    if latest_valid:
                        w = int(0)
                        for r in range(count):
                            if res_zone_id[r] != latest_id:
                                res_zone_id[w] = res_zone_id[r]
                                res_main[w] = res_main[r]
                                res_dx[w] = res_dx[r]
                                res_dy[w] = res_dy[r]
                                res_dz[w] = res_dz[r]
                                res_turb[w] = res_turb[r]
                                w += 1
                        count = w
                    if registrable:
                        res_zone_id[count] = zid
                        res_main[count] = zmain
                        res_dx[count] = dirx
                        res_dy[count] = diry
                        res_dz[count] = dirz
                        res_turb[count] = zturb
                        count += 1
                    min_volume = zvol
                    latest_id = zid
                    latest_valid = wp.bool(True)
        final = count if count < WIND_ZONE_SLOTS else WIND_ZONE_SLOTS
        state.particle.wind_count[p] = final
        for s in range(final):
            state.particle.wind_zone_id[p, s] = res_zone_id[s]
            state.particle.wind_main[p, s] = res_main[s]
            state.particle.wind_zone_turbulence[p, s] = res_turb[s]
            (dqx, dqy, dqz, dqw) = dmath.axis_quaternion(res_dx[s], res_dy[s], res_dz[s])
            state.particle.wind_dirq[p, s, 0] = dqx
            state.particle.wind_dirq[p, s, 1] = dqy
            state.particle.wind_dirq[p, s, 2] = dqz
            state.particle.wind_dirq[p, s, 3] = dqw


@wp.kernel
def resolve_particle_wind_zones(state: ClothState, substep: int, level: int, iteration: int):
    resolve_particle_wind_zones_element(state, wp.tid(), substep, level, iteration)


@wp.func
def sample_team_wind_zones_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    i = thread_index
    state.derived.wind_zone_overflow[i] = 0
    state.derived.wind_zone_demand[i] = 0
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            i):
        old_count = state.team.wind_count[i]
        old_zid = WIND_SLOT_INDICES()
        old_wt = WIND_SLOT_VALUES()
        for oc in range(WIND_ZONE_SLOTS):
            if oc < old_count:
                old_zid[oc] = state.team.wind_zone_id[i, oc]
                old_wt[oc] = state.team.wind_time[i, oc]
        union_zid = ZONE_RESULT_INDICES()
        union_main = ZONE_RESULT_VALUES()
        fill = int(0)
        demand = int(0)
        p0 = state.team.p_start[i]
        p1 = p0 + state.team.p_count[i]
        for p in range(p0, p1):
            wc = state.particle.wind_count[p]
            for s in range(WIND_ZONE_SLOTS):
                if s < wc:
                    zid = state.particle.wind_zone_id[p, s]
                    idx = int(-1)
                    for u in range(WIND_ZONE_RESULT_SLOTS):
                        if u < fill and union_zid[u] == zid:
                            idx = u
                    if idx < 0:
                        if fill < WIND_ZONE_RESULT_SLOTS:
                            union_zid[fill] = zid
                            union_main[fill] = state.particle.wind_main[p, s]
                            idx = fill
                            fill += 1
                        demand += 1
                    slot = int(0)
                    if idx >= 0 and idx < WIND_ZONE_SLOTS:
                        slot = idx
                    state.particle.wind_phase_slot[p, s] = slot
        assigned = fill if fill < WIND_ZONE_SLOTS else WIND_ZONE_SLOTS
        state.team.wind_count[i] = assigned
        for u in range(WIND_ZONE_SLOTS):
            if u < assigned:
                zid = union_zid[u]
                t_prev = dmath.negate(WIND_MAX_TIME)
                for oi in range(WIND_ZONE_SLOTS):
                    if oi < old_count and old_zid[oi] == zid:
                        t_prev = old_wt[oi]
                state.team.wind_zone_id[i, u] = zid
                state.team.wind_time[i, u] = t_prev
                state.team.wind_main[i, u] = union_main[u]
        if demand > WIND_ZONE_SLOTS:
            state.derived.wind_zone_overflow[i] = 1
            state.derived.wind_zone_demand[i] = demand


@wp.kernel
def sample_team_wind_zones(state: ClothState, substep: int, level: int, iteration: int):
    sample_team_wind_zones_element(state, wp.tid(), substep, level, iteration)


@wp.func
def reset_particle_frame_state_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            state.particle.team[p]):
        kernels.do_particles_frame_pre(p, state.particle.team, state.particle.positions,
                state.particle.rotations, state.particle.next_positions, state.particle.old_positions,
                state.particle.old_rotations, state.particle.base_positions, state.particle.base_rotations,
                state.particle.old_anim_positions, state.particle.old_anim_rotations,
                state.particle.velocity_positions, state.particle.display_positions,
                state.particle.velocities, state.particle.real_velocities, state.particle.friction,
                state.particle.static_friction, state.particle.collision_normals, state.team.reset_pending,
                state.team.negative_scale_teleport, state.team.negative_scale_matrix,
                state.team.inertia_shift, state.team.frame_component_shift_vector,
                state.team.frame_component_shift_rotation,
                state.team.old_component_world_position)


@wp.kernel
def reset_particle_frame_state(state: ClothState, substep: int, level: int, iteration: int):
    reset_particle_frame_state_element(state, wp.tid(), substep, level, iteration)


@wp.func
def prepare_collider_frame_pose_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    ci = thread_index
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            state.collider.team[ci]):
        kernels.do_collider_frame_pre(ci, state.collider.team, state.collider.enabled,
                state.collider.enabled_prev, state.collider.active, state.collider.input_positions,
                state.collider.input_rotations, state.collider.input_tips, state.collider.input_radii,
                state.collider.frame_positions, state.collider.frame_rotations, state.collider.frame_tips,
                state.collider.frame_radii, state.collider.old_frame_positions,
                state.collider.old_frame_rotations, state.collider.old_frame_tips,
                state.collider.now_positions, state.collider.now_rotations, state.collider.now_tips,
                state.collider.old_positions, state.collider.old_rotations, state.collider.old_tips,
                state.team.reset_pending, state.team.negative_scale_teleport, state.team.negative_scale_matrix,
                state.team.inertia_shift, state.team.frame_component_shift_vector,
                state.team.frame_component_shift_rotation,
                state.team.old_component_world_position)


@wp.kernel
def prepare_collider_frame_pose(state: ClothState, substep: int, level: int, iteration: int):
    prepare_collider_frame_pose_element(state, wp.tid(), substep, level, iteration)


@wp.func
def update_collider_face_primitives_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    f = thread_index
    kernels.do_collider_face_primitive(f, state.collider_faces.team,
            state.collider_faces.vertex, state.collider_faces.edge_ring_face,
            state.collider_faces.edge_ring_corner, state.collider_faces.aabb_min,
            state.collider_faces.aabb_max, state.collider_faces.normal,
            state.collider_faces.edge_normal,
            state.collider_vertices.local_position, state.team.enabled, state.team.valid,
            state.team.component_world_scale)


@wp.kernel
def update_collider_face_primitives(state: ClothState, substep: int, level: int,
        iteration: int):
    update_collider_face_primitives_element(state, wp.tid(), substep, level, iteration)


@wp.func
def update_collider_vertex_pseudo_normals_element(state: ClothState, thread_index: int,
        substep: int, level: int, iteration: int):
    v = thread_index
    kernels.do_collider_vertex_pseudo_normal(v, state.collider_vertices.team,
            state.collider_vertices.fan_face, state.collider_vertices.fan_corner,
            state.collider_vertices.local_position, state.collider_vertices.pseudo_normal,
            state.collider_faces.vertex, state.collider_faces.fan_next_face,
            state.collider_faces.fan_next_corner, state.collider_faces.normal,
            state.team.enabled, state.team.valid, state.team.component_world_scale)


@wp.kernel
def update_collider_vertex_pseudo_normals(state: ClothState, substep: int, level: int,
        iteration: int):
    update_collider_vertex_pseudo_normals_element(state, wp.tid(), substep, level, iteration)


@wp.func
def clear_intersect_pair_counter_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    tid = thread_index
    if tid == 0:
        state.derived.self_counters[SCL_IP_COUNT] = 0


@wp.kernel
def clear_intersect_pair_counter(state: ClothState, substep: int, level: int, iteration: int):
    clear_intersect_pair_counter_element(state, wp.tid(), substep, level, iteration)


@wp.func
def gather_intersect_pairs_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    g = thread_index
    num_it_slots = state.derived.self_intersect_task_query_offsets.shape[0] - 1
    total_query = state.derived.self_intersect_task_query_offsets[num_it_slots]
    if g == 0:
        state.derived.self_counters[SCL_IP_COUNT] = state.derived.self_intersect_task_slot_offsets[
                num_it_slots]
    frame_index = state.derived.self_counters[SCL_FRAME_INDEX]
    if g < total_query:
        lo = int(0)
        hi = num_it_slots
        while lo < hi:
            mid = lo + hi >> 1
            if state.derived.self_intersect_task_query_offsets[mid + 1] <= g:
                lo = mid + 1
            else:
                hi = mid
        task = lo
        i = g - state.derived.self_intersect_task_query_offsets[task]
        my_edge = state.derived.self_intersect_task_edge_start[task] + i
        my_team = state.self_edges.team[my_edge]
        capacity = state.team.self_contact_slots[my_team]
        base = state.derived.self_intersect_task_slot_offsets[task] + i * capacity
        for slot in range(capacity):
            state.derived.self_intersect_pair_edge[base + slot] = -1
            state.derived.self_intersect_pair_triangle[base + slot] = -1
            state.derived.self_intersect_pair_gap_key[base + slot] = 0
        tgt_team = state.derived.self_intersect_task_triangle_team[task]
        if state.team.self_grid_size[tgt_team] > EPSILON and state.team.self_max_primitive_size[
                tgt_team] > EPSILON and (state.self_edges.ignore[my_edge] == 0) and (i %
                SELF_COLLISION_INTERSECT_DIV == frame_index):
            same = state.derived.self_intersect_task_same_team[task]
            fill = int(0)
            worst = int(0)
            accepted = int(0)
            root = wp.bvh_get_group_root(state.self_triangles_index, tgt_team)
            query = wp.bvh_query_aabb(state.self_triangles_index, wp.vec3(state.self_edges.aabb_min[my_edge,
                    0], state.self_edges.aabb_min[my_edge, 1], state.self_edges.aabb_min[my_edge,
                    2]), wp.vec3(state.self_edges.aabb_max[my_edge, 0], state.self_edges.aabb_max[my_edge,
                    1], state.self_edges.aabb_max[my_edge, 2]), root)
            tgt_tri = int(0)
            while wp.bvh_query_next(query, tgt_tri):
                if state.self_triangles.use[tgt_tri] != 0 and state.self_triangles.ignore[tgt_tri] == 0 and kernels.self_aabb_overlap(state.self_edges.aabb_min,
                        state.self_edges.aabb_max, my_edge, state.self_triangles.aabb_min,
                        state.self_triangles.aabb_max,
                        tgt_tri) and (not (state.self_edges.all_fix[my_edge] != 0 and state.
                        self_triangles.all_fix[tgt_tri] != 0)):
                    if same == 0 or not kernels.self_connection_shared(state.self_edges.particles,
                            my_edge, state.self_triangles.particles, tgt_tri):
                        accepted = accepted + 1
                        key = kernels.gap_order_key(kernels.self_box_gap(state.self_edges.aabb_min,
                                state.self_edges.aabb_max, my_edge, state.self_triangles.aabb_min,
                                state.self_triangles.aabb_max, tgt_tri))
                        if fill < capacity:
                            state.derived.self_intersect_pair_edge[base + fill] = my_edge
                            state.derived.self_intersect_pair_triangle[base + fill] = tgt_tri
                            state.derived.self_intersect_pair_gap_key[base + fill] = key
                            fill = fill + 1
                            if fill == capacity:
                                worst = kernels.self_worst_slot(state.derived.self_intersect_pair_gap_key,
                                        state.derived.self_intersect_pair_triangle, base,
                                        capacity)
                        elif kernels.self_ranks_before(key, tgt_tri,
                                state.derived.self_intersect_pair_gap_key[base + worst],
                                state.derived.self_intersect_pair_triangle[base + worst]):
                            state.derived.self_intersect_pair_edge[base + worst] = my_edge
                            state.derived.self_intersect_pair_triangle[base + worst] = tgt_tri
                            state.derived.self_intersect_pair_gap_key[base + worst] = key
                            worst = kernels.self_worst_slot(state.derived.self_intersect_pair_gap_key,
                                    state.derived.self_intersect_pair_triangle, base, capacity)
            wp.atomic_max(state.derived.self_intersect_demand, my_team, accepted)
            if accepted > capacity:
                wp.atomic_add(state.derived.self_intersect_overflow, my_team,
                        accepted - capacity)
                state.derived.self_counters[SCL_ERROR] = 1


@wp.kernel
def gather_intersect_pairs(state: ClothState, substep: int, level: int, iteration: int):
    gather_intersect_pairs_element(state, wp.tid(), substep, level, iteration)


@wp.func
def solve_tether_constraint_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    e = thread_index
    tm = state.tether.team[e]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            tm) and state.team.update_count[tm] > substep:
        kernels.do_tether(e, state.tether.particle, state.particle.team, state.particle.next_positions,
                state.particle.velocity_positions, state.particle.step_basic_positions,
                state.particle.vertex_root, state.team.tether_compression)


@wp.kernel
def solve_tether_constraint(state: ClothState, substep: int, level: int, iteration: int):
    solve_tether_constraint_element(state, wp.tid(), substep, level, iteration)


@wp.func
def gather_distance_correction_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    power1 = state.frame_scalar.frame_float[SCAL_POWER1]
    mt = state.particle.team[p]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt) and state.team.update_count[mt] > substep:
        kernels.do_distance_gather(p, state.particle.team, state.particle.next_positions,
                state.particle.base_positions, state.particle.depth, state.particle.friction,
                state.particle.attr_move, state.team.is_spring, state.team.animation_pose_ratio,
                state.team.init_scale, state.team.scale_ratio, state.team.distance_lut, power1,
                state.derived.distance_csr_offsets, state.derived.distance_csr_order,
                state.distance.target, state.distance.rest, state.derived.distance_correction)


@wp.kernel
def gather_distance_correction(state: ClothState, substep: int, level: int, iteration: int):
    gather_distance_correction_element(state, wp.tid(), substep, level, iteration)


@wp.func
def advance_team_substep_motion_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    i = thread_index
    sim_dt = state.frame_scalar.frame_float[SCAL_SIM_DT]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            i) and state.team.update_count[i] > substep:
        kernels.do_step_update(i, sim_dt, state.team.now_update_time, state.team.time,
                state.team.frame_old_time, state.team.frame_interpolation, state.team.now_world_position,
                state.team.now_world_rotation, state.team.old_world_position, state.team.old_world_rotation,
                state.team.old_frame_world_position, state.team.old_frame_world_rotation,
                state.team.old_frame_world_scale, state.team.frame_world_position,
                state.team.frame_world_rotation, state.team.frame_world_scale, state.team.step_vector,
                state.team.step_rotation, state.team.step_move_inertia_ratio,
                state.team.step_rotation_inertia_ratio, state.team.local_inertia,
                state.team.local_movement_speed_limit, state.team.local_rotation_speed_limit,
                state.team.inertia_vector, state.team.inertia_rotation, state.team.angular_velocity,
                state.team.rotation_axis, state.team.init_scale, state.team.scale_ratio,
                state.team.gravity_direction, state.team.gravity_dot, state.team.init_local_gravity_direction,
                state.team.component_world_reflected, state.team.gravity, state.team.gravity_falloff,
                state.team.gravity_ratio, state.team.velocity_weight, state.team.stablization_time,
                state.team.blend_weight, state.team.blend_weight_param, state.team.distance_weight,
                state.team.wind_moving, state.team.frame_moving_speed, state.team.moving_wind_main,
                state.team.frame_moving_direction, state.team.moving_wind_direction,
                state.team.moving_wind_dirq, state.team.wind_main, state.team.wind_frequency,
                state.team.wind_count, state.team.wind_time, state.team.moving_wind_time)


@wp.kernel
def advance_team_substep_motion(state: ClothState, substep: int, level: int, iteration: int):
    advance_team_substep_motion_element(state, wp.tid(), substep, level, iteration)


@wp.func
def interpolate_collider_substep_pose_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    ci = thread_index
    cm = state.collider.team[ci]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            cm) and state.team.update_count[cm] > substep and (state.collider.active[ci] != 0):
        kernels.do_collider_start_step(ci, state.collider.team, state.collider.kind,
                state.collider.frame_positions, state.collider.frame_rotations, state.collider.frame_tips,
                state.collider.frame_radii, state.collider.old_frame_positions,
                state.collider.old_frame_rotations, state.collider.old_frame_tips,
                state.collider.now_positions, state.collider.now_rotations, state.collider.now_tips,
                state.collider.old_positions, state.collider.old_rotations, state.collider.old_tips,
                state.collider.work_rot, state.collider.work_inv_old_rot,
                state.collider.work_inv_rot, state.collider.work_radius,
                state.collider.work_old_pos, state.collider.work_next_pos, state.collider.work_aabb_min,
                state.collider.work_aabb_max, state.collider.mesh_local_bound_min,
                state.collider.mesh_local_bound_max,
                state.team.frame_interpolation, state.team.step_move_inertia_ratio,
                state.team.step_rotation_inertia_ratio)


@wp.kernel
def interpolate_collider_substep_pose(state: ClothState, substep: int, level: int,
        iteration: int):
    interpolate_collider_substep_pose_element(state, wp.tid(), substep, level, iteration)


@wp.func
def animate_particle_base_pose_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    mt = state.particle.team[p]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt) and state.team.update_count[mt] > substep:
        t = state.team.frame_interpolation[mt]
        bx = dmath.lerp(state.particle.old_anim_positions[p, 0], state.particle.positions[p, 0],
                t)
        by = dmath.lerp(state.particle.old_anim_positions[p, 1], state.particle.positions[p, 1],
                t)
        bz = dmath.lerp(state.particle.old_anim_positions[p, 2], state.particle.positions[p, 2],
                t)
        (qx, qy, qz, qw) = dmath.quat_slerp(state.particle.old_anim_rotations[p, 0],
                state.particle.old_anim_rotations[p, 1], state.particle.old_anim_rotations[p, 2],
                state.particle.old_anim_rotations[p, 3], state.particle.rotations[p, 0],
                state.particle.rotations[p, 1], state.particle.rotations[p, 2], state.particle.rotations[p,
                3], t)
        state.particle.base_positions[p, 0] = bx
        state.particle.base_positions[p, 1] = by
        state.particle.base_positions[p, 2] = bz
        state.particle.step_basic_positions[p, 0] = bx
        state.particle.step_basic_positions[p, 1] = by
        state.particle.step_basic_positions[p, 2] = bz
        state.particle.base_rotations[p, 0] = qx
        state.particle.base_rotations[p, 1] = qy
        state.particle.base_rotations[p, 2] = qz
        state.particle.base_rotations[p, 3] = qw
        state.particle.step_basic_rotations[p, 0] = qx
        state.particle.step_basic_rotations[p, 1] = qy
        state.particle.step_basic_rotations[p, 2] = qz
        state.particle.step_basic_rotations[p, 3] = qw


@wp.kernel
def animate_particle_base_pose(state: ClothState, substep: int, level: int, iteration: int):
    animate_particle_base_pose_element(state, wp.tid(), substep, level, iteration)


@wp.func
def integrate_particle_motion_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    e = thread_index
    power2 = state.frame_scalar.frame_float[SCAL_POWER2]
    sim_dt = state.frame_scalar.frame_float[SCAL_SIM_DT]
    mt = state.update_move.team[e]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt) and state.team.update_count[mt] > substep:
        pmi = state.update_move.particle[e]
        depth = state.particle.depth[pmi]
        ox = state.particle.old_positions[pmi, 0]
        oy = state.particle.old_positions[pmi, 1]
        oz = state.particle.old_positions[pmi, 2]
        inertia_depth = state.team.depth_inertia[mt] * (1.0 - depth * depth)
        ivx = dmath.lerp(state.team.inertia_vector[mt, 0], state.team.step_vector[mt, 0],
                inertia_depth)
        ivy = dmath.lerp(state.team.inertia_vector[mt, 1], state.team.step_vector[mt, 1],
                inertia_depth)
        ivz = dmath.lerp(state.team.inertia_vector[mt, 2], state.team.step_vector[mt, 2],
                inertia_depth)
        (irx, iry, irz, irw) = dmath.quat_slerp(state.team.inertia_rotation[mt, 0],
                state.team.inertia_rotation[mt, 1], state.team.inertia_rotation[mt, 2],
                state.team.inertia_rotation[mt, 3], state.team.step_rotation[mt, 0],
                state.team.step_rotation[mt, 1], state.team.step_rotation[mt, 2], state.team.step_rotation[mt,
                3], inertia_depth)
        owx = state.team.old_world_position[mt, 0]
        owy = state.team.old_world_position[mt, 1]
        owz = state.team.old_world_position[mt, 2]
        lx = ox - owx
        ly = oy - owy
        lz = oz - owz
        (rlx, rly, rlz) = dmath.quat_rotate(irx, iry, irz, irw, lx, ly, lz)
        lx = rlx + ivx
        ly = rly + ivy
        lz = rlz + ivz
        wx = owx + lx
        wy = owy + ly
        wz = owz + lz
        nextx = wx
        nexty = wy
        nextz = wz
        velposx = ox + (wx - ox)
        velposy = oy + (wy - oy)
        velposz = oz + (wz - oz)
        (vx, vy, vz) = dmath.quat_rotate(irx, iry, irz, irw, state.particle.velocities[pmi, 0],
                state.particle.velocities[pmi, 1], state.particle.velocities[pmi, 2])
        vw = state.team.velocity_weight[mt]
        vx = vx * vw
        vy = vy * vw
        vz = vz * vw
        damping = dmath.evaluate_team_lut_clamp01(state.team.damping_lut, mt, depth)
        damp = dmath.saturate(1.0 - damping * power2)
        vx = vx * damp
        vy = vy * damp
        vz = vz * damp
        fm = state.team.force_mode[mt]
        change = fm == FORCE_VELOCITY_CHANGE or fm == FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH
        if change:
            vx = 0.0
            vy = 0.0
            vz = 0.0
        g = state.team.gravity[mt] * state.team.gravity_ratio[mt]
        fx = state.team.gravity_direction[mt, 0] * g
        fy = state.team.gravity_direction[mt, 1] * g
        fz = state.team.gravity_direction[mt, 2] * g
        mass = dmath.calc_mass(depth)
        with_depth = fm == FORCE_VELOCITY_ADD or fm == FORCE_VELOCITY_CHANGE
        without_depth = fm == FORCE_VELOCITY_ADD_WITHOUT_DEPTH or fm == FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH
        if with_depth:
            fx = fx + state.team.impact_force[mt, 0] / mass
            fy = fy + state.team.impact_force[mt, 1] / mass
            fz = fz + state.team.impact_force[mt, 2] / mass
        if without_depth:
            fx = fx + state.team.impact_force[mt, 0]
            fy = fy + state.team.impact_force[mt, 1]
            fz = fz + state.team.impact_force[mt, 2]
        root = float(state.particle.vertex_root_local[pmi])
        seed = float(state.team.wind_seed[mt])
        sync = state.team.wind_synchronization[mt]
        wind_position = (seed + 1.0) * 4.19230645 + root * 0.0023963 * (1.0 - sync) * 100.0
        blend = state.team.wind_blend[mt]
        turbulence_param = state.team.wind_turbulence[mt]
        wfx = float(0.0)
        wfy = float(0.0)
        wfz = float(0.0)
        wc = state.particle.wind_count[pmi]
        for s in range(WIND_ZONE_SLOTS):
            if s < wc:
                ps = state.particle.wind_phase_slot[pmi, s]
                (cx, cy, cz) = kernels.do_wind_blend(state.particle.wind_main[pmi, s],
                        state.team.wind_time[mt, ps], state.particle.wind_dirq[pmi, s, 0],
                        state.particle.wind_dirq[pmi, s, 1], state.particle.wind_dirq[pmi, s, 2],
                        state.particle.wind_dirq[pmi, s, 3], state.particle.wind_zone_turbulence[pmi, s],
                        blend, turbulence_param, wind_position)
                wfx = wfx + cx
                wfy = wfy + cy
                wfz = wfz + cz
        moving_on = state.team.wind_moving[mt] > WIND_MIN_SPEED
        if moving_on:
            (mcx, mcy, mcz) = kernels.do_wind_blend(state.team.moving_wind_main[mt],
                    state.team.moving_wind_time[mt], state.team.moving_wind_dirq[mt, 0],
                    state.team.moving_wind_dirq[mt, 1], state.team.moving_wind_dirq[mt, 2],
                    state.team.moving_wind_dirq[mt, 3], 1.0, blend, turbulence_param,
                    wind_position)
            wfx = wfx + mcx
            wfy = wfy + mcy
            wfz = wfz + mcz
        influence = state.team.wind_influence[mt] * (1.0 - state.particle.friction[pmi])
        depth_scale = depth * depth
        influence = influence * dmath.lerp(1.0, depth_scale, state.team.wind_depth_weight[mt])
        fx = fx + wfx * influence
        fy = fy + wfy * influence
        fz = fz + wfz * influence
        sr = state.team.scale_ratio[mt]
        fx = fx * sr
        fy = fy * sr
        fz = fz * sr
        vx = vx + fx * sim_dt
        vy = vy + fy * sim_dt
        vz = vz + fz * sim_dt
        nextx = nextx + vx * sim_dt
        nexty = nexty + vy * sim_dt
        nextz = nextz + vz * sim_dt
        state.particle.velocities[pmi, 0] = vx
        state.particle.velocities[pmi, 1] = vy
        state.particle.velocities[pmi, 2] = vz
        state.particle.next_positions[pmi, 0] = nextx
        state.particle.next_positions[pmi, 1] = nexty
        state.particle.next_positions[pmi, 2] = nextz
        state.particle.velocity_positions[pmi, 0] = velposx
        state.particle.velocity_positions[pmi, 1] = velposy
        state.particle.velocity_positions[pmi, 2] = velposz


@wp.kernel
def integrate_particle_motion(state: ClothState, substep: int, level: int, iteration: int):
    integrate_particle_motion_element(state, wp.tid(), substep, level, iteration)


@wp.func
def pin_fixed_particles_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    e = thread_index
    ft = state.update_fixed.team[e]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            ft) and state.team.update_count[ft] > substep:
        pfi = state.update_fixed.particle[e]
        state.particle.next_positions[pfi, 0] = state.particle.base_positions[pfi, 0]
        state.particle.next_positions[pfi, 1] = state.particle.base_positions[pfi, 1]
        state.particle.next_positions[pfi, 2] = state.particle.base_positions[pfi, 2]
        state.particle.velocity_positions[pfi, 0] = state.particle.base_positions[pfi, 0]
        state.particle.velocity_positions[pfi, 1] = state.particle.base_positions[pfi, 1]
        state.particle.velocity_positions[pfi, 2] = state.particle.base_positions[pfi, 2]


@wp.kernel
def pin_fixed_particles(state: ClothState, substep: int, level: int, iteration: int):
    pin_fixed_particles_element(state, wp.tid(), substep, level, iteration)


@wp.func
def pin_spring_particles_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    e = thread_index
    st = state.spring.team[e]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            st) and state.team.update_count[st] > substep and (state.team.spring_power[st] > 0.0
            ):
        psi = state.spring.particle[e]
        bpx = state.particle.base_positions[psi, 0]
        bpy = state.particle.base_positions[psi, 1]
        bpz = state.particle.base_positions[psi, 2]
        n0 = state.particle.next_positions[psi, 0]
        n1 = state.particle.next_positions[psi, 1]
        n2 = state.particle.next_positions[psi, 2]
        vx = n0 - bpx
        vy = n1 - bpy
        vz = n2 - bpz
        (dx, dy, dz) = dmath.quat_rotate(state.particle.base_rotations[psi, 0],
                state.particle.base_rotations[psi, 1], state.particle.base_rotations[psi, 2],
                state.particle.base_rotations[psi, 3], state.team.normal_axis_vector[st, 0],
                state.team.normal_axis_vector[st, 1], state.team.normal_axis_vector[st, 2])
        limit = state.team.spring_limit_distance[st] * state.team.scale_ratio[st]
        clampable = limit > 1e-08
        l = dmath.length3(vx, vy, vz)
        over = clampable and l > limit
        if over and l > 1e-30:
            scale = limit / l
            vx = vx * scale
            vy = vy * scale
            vz = vz * scale
        ratio = state.team.spring_normal_limit_ratio[st]
        elliptic = clampable and ratio < 1.0
        ylen = dmath.dot3(dx, dy, dz, vx, vy, vz)
        vpx = vx - dx * ylen
        vpy = vy - dy * ylen
        vpz = vz - dz * ylen
        xlen = dmath.length3(vpx, vpy, vpz)
        safe_limit = limit if limit > 1e-30 else 1.0
        tval = dmath.saturate(xlen / safe_limit)
        y = wp.cos(wp.asin(dmath.clamp1(tval))) * (limit * ratio)
        exceed = elliptic and wp.abs(ylen) > y
        if exceed:
            adjust = (wp.abs(ylen) - y) * dmath.fsign(ylen)
            vx = vx - adjust * dx
            vy = vy - adjust * dy
            vz = vz - adjust * dz
        if not clampable:
            vx = 0.0
            vy = 0.0
            vz = 0.0
        power = state.team.spring_power[st]
        noise_param = state.team.spring_noise[st]
        if noise_param > 0.0:
            noise_time = (state.team.time[st] + float(psi) * 49.6198) * 2.4512 + (n0 + n1 + n2)
            noise = wp.sin(noise_time) * (noise_param * 0.6)
            power = power + power * noise
            if power < 0.0:
                power = 0.0
        vx = vx - vx * power
        vy = vy - vy * power
        vz = vz - vz * power
        state.particle.next_positions[psi, 0] = bpx + vx
        state.particle.next_positions[psi, 1] = bpy + vy
        state.particle.next_positions[psi, 2] = bpz + vz


@wp.kernel
def pin_spring_particles(state: ClothState, substep: int, level: int, iteration: int):
    pin_spring_particles_element(state, wp.tid(), substep, level, iteration)


@wp.func
def propagate_animated_chain_pose_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    group = thread_index
    if group + 1 < state.derived.fk_yes_root_offsets.shape[0]:
        start = state.derived.fk_yes_root_offsets[group]
        stop = state.derived.fk_yes_root_offsets[group + 1]
        for slot in range(start, stop):
            i = state.derived.fk_yes_root_entries[slot]
            v = state.derived.fk_yes[i]
            vt = state.particle.team[v]
            if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
                    vt) and state.team.update_count[vt] > substep and (state.team.animation_pose_ratio
                    [vt] <= 0.99):
                par = state.derived.fk_yes_parent[i]
                sr = state.team.scale_ratio[vt]
                reflection_sign = kernels.component_reflection_sign(state.team.component_world_reflected,
                        vt)
                scx = state.team.init_scale[vt, 0] * sr * reflection_sign
                scy = state.team.init_scale[vt, 1] * sr * reflection_sign
                scz = state.team.init_scale[vt, 2] * sr * reflection_sign
                lsx = state.particle.vertex_local_positions[v, 0] * scx
                lsy = state.particle.vertex_local_positions[v, 1] * scy
                lsz = state.particle.vertex_local_positions[v, 2] * scz
                prx = state.particle.step_basic_rotations[par, 0]
                pry = state.particle.step_basic_rotations[par, 1]
                prz = state.particle.step_basic_rotations[par, 2]
                prw = state.particle.step_basic_rotations[par, 3]
                (rx, ry, rz) = dmath.quat_rotate(prx, pry, prz, prw, lsx, lsy, lsz)
                state.particle.step_basic_positions[v, 0] = rx + state.particle.step_basic_positions[par,
                        0]
                state.particle.step_basic_positions[v, 1] = ry + state.particle.step_basic_positions[par,
                        1]
                state.particle.step_basic_positions[v, 2] = rz + state.particle.step_basic_positions[par,
                        2]
                lrx = state.particle.vertex_local_rotations[v, 0]
                lry = state.particle.vertex_local_rotations[v, 1]
                lrz = state.particle.vertex_local_rotations[v, 2]
                lrw = state.particle.vertex_local_rotations[v, 3]
                (qx, qy, qz, qw) = dmath.quat_mul(prx, pry, prz, prw, lrx, lry, lrz, lrw)
                state.particle.step_basic_rotations[v, 0] = qx
                state.particle.step_basic_rotations[v, 1] = qy
                state.particle.step_basic_rotations[v, 2] = qz
                state.particle.step_basic_rotations[v, 3] = qw


@wp.kernel
def propagate_animated_chain_pose(state: ClothState, substep: int, level: int, iteration: int):
    propagate_animated_chain_pose_element(state, wp.tid(), substep, level, iteration)


@wp.func
def propagate_static_chain_pose_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    i = thread_index
    if i < state.derived.fk_no.shape[0]:
        v = state.derived.fk_no[i]
        vt = state.particle.team[v]
        if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
                vt) and state.team.update_count[vt] > substep and (state.team.animation_pose_ratio
                [vt] <= 0.99) and (state.team.component_world_reflected[vt] != 0):
            rox = state.particle.step_basic_rotations[v, 0]
            roy = state.particle.step_basic_rotations[v, 1]
            roz = state.particle.step_basic_rotations[v, 2]
            row = state.particle.step_basic_rotations[v, 3]
            reflection_sign = kernels.component_reflection_sign(state.team.component_world_reflected,
                    vt)
            (nx, ny, nz) = dmath.quat_to_normal(rox, roy, roz, row)
            nnx = nx * reflection_sign
            nny = ny * reflection_sign
            nnz = nz * reflection_sign
            (tx, ty, tz) = dmath.quat_to_tangent(rox, roy, roz, row)
            ttx = tx * reflection_sign
            tty = ty * reflection_sign
            ttz = tz * reflection_sign
            (lqx, lqy, lqz, lqw) = dmath.look_rotation(ttx, tty, ttz, nnx, nny, nnz)
            state.particle.step_basic_rotations[v, 0] = lqx
            state.particle.step_basic_rotations[v, 1] = lqy
            state.particle.step_basic_rotations[v, 2] = lqz
            state.particle.step_basic_rotations[v, 3] = lqw


@wp.kernel
def propagate_static_chain_pose(state: ClothState, substep: int, level: int, iteration: int):
    propagate_static_chain_pose_element(state, wp.tid(), substep, level, iteration)


@wp.func
def blend_baseline_chain_pose_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    i = thread_index
    v = state.derived.baseline_entries[i]
    vt = state.particle.team[v]
    apr = state.team.animation_pose_ratio[vt]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            vt) and state.team.update_count[vt] > substep and (apr <= 0.99) and (apr > EPSILON):
        state.particle.step_basic_positions[v, 0] = dmath.lerp(state.particle.step_basic_positions[v,
                0], state.particle.base_positions[v, 0], apr)
        state.particle.step_basic_positions[v, 1] = dmath.lerp(state.particle.step_basic_positions[v,
                1], state.particle.base_positions[v, 1], apr)
        state.particle.step_basic_positions[v, 2] = dmath.lerp(state.particle.step_basic_positions[v,
                2], state.particle.base_positions[v, 2], apr)
        (qx, qy, qz, qw) = dmath.quat_slerp(state.particle.step_basic_rotations[v, 0],
                state.particle.step_basic_rotations[v, 1], state.particle.step_basic_rotations[v,
                2], state.particle.step_basic_rotations[v, 3], state.particle.base_rotations[v,
                0], state.particle.base_rotations[v, 1], state.particle.base_rotations[v, 2],
                state.particle.base_rotations[v, 3], apr)
        state.particle.step_basic_rotations[v, 0] = qx
        state.particle.step_basic_rotations[v, 1] = qy
        state.particle.step_basic_rotations[v, 2] = qz
        state.particle.step_basic_rotations[v, 3] = qw


@wp.kernel
def blend_baseline_chain_pose(state: ClothState, substep: int, level: int, iteration: int):
    blend_baseline_chain_pose_element(state, wp.tid(), substep, level, iteration)


@wp.func
def apply_distance_correction_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    p = thread_index
    mt = state.particle.team[p]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt) and state.team.update_count[mt] > substep:
        state.particle.next_positions[p, 0] = state.particle.next_positions[p,
                0] + state.derived.distance_correction[p, 0]
        state.particle.next_positions[p, 1] = state.particle.next_positions[p,
                1] + state.derived.distance_correction[p, 1]
        state.particle.next_positions[p, 2] = state.particle.next_positions[p,
                2] + state.derived.distance_correction[p, 2]
        state.particle.velocity_positions[p, 0] = state.particle.velocity_positions[p,
                0] + state.derived.distance_correction[p, 0] * DISTANCE_VELOCITY_ATTENUATION
        state.particle.velocity_positions[p, 1] = state.particle.velocity_positions[p,
                1] + state.derived.distance_correction[p, 1] * DISTANCE_VELOCITY_ATTENUATION
        state.particle.velocity_positions[p, 2] = state.particle.velocity_positions[p,
                2] + state.derived.distance_correction[p, 2] * DISTANCE_VELOCITY_ATTENUATION


@wp.kernel
def apply_distance_correction(state: ClothState, substep: int, level: int, iteration: int):
    apply_distance_correction_element(state, wp.tid(), substep, level, iteration)


@wp.func
def buffer_baseline_angle_state_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    i = thread_index
    v = state.derived.baseline_entries[i]
    vt = state.particle.team[v]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            vt) and state.team.update_count[vt] > substep and (state.team.angle_use_limit[vt] !=
            0 or state.team.angle_use_restoration[vt] != 0):
        state.particle.albuf_rotation[v, 0] = state.particle.step_basic_rotations[v, 0]
        state.particle.albuf_rotation[v, 1] = state.particle.step_basic_rotations[v, 1]
        state.particle.albuf_rotation[v, 2] = state.particle.step_basic_rotations[v, 2]
        state.particle.albuf_rotation[v, 3] = state.particle.step_basic_rotations[v, 3]


@wp.kernel
def buffer_baseline_angle_state(state: ClothState, substep: int, level: int, iteration: int):
    buffer_baseline_angle_state_element(state, wp.tid(), substep, level, iteration)


@wp.func
def buffer_carried_angle_state_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    i = thread_index
    v = state.angle_buffered.particle[i]
    vt = state.particle.team[v]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            vt) and state.team.update_count[vt] > substep and (state.team.angle_use_limit[vt] !=
            0 or state.team.angle_use_restoration[vt] != 0):
        par = state.particle.vertex_parent[v]
        bvx = state.particle.step_basic_positions[v, 0] - state.particle.step_basic_positions[par,
                0]
        bvy = state.particle.step_basic_positions[v, 1] - state.particle.step_basic_positions[par,
                1]
        bvz = state.particle.step_basic_positions[v, 2] - state.particle.step_basic_positions[par,
                2]
        if state.team.angle_use_limit[vt] != 0:
            dvx = state.particle.next_positions[v, 0] - state.particle.next_positions[par, 0]
            dvy = state.particle.next_positions[v, 1] - state.particle.next_positions[par, 1]
            dvz = state.particle.next_positions[v, 2] - state.particle.next_positions[par, 2]
            avlen = dmath.length3(dvx, dvy, dvz)
            bvlen = dmath.length3(bvx, bvy, bvz)
            if avlen < EPSILON or bvlen < EPSILON:
                state.particle.albuf_length[v] = 0.0
                state.particle.albuf_local_pos[v, 0] = 0.0
                state.particle.albuf_local_pos[v, 1] = 0.0
                state.particle.albuf_local_pos[v, 2] = 0.0
                state.particle.albuf_local_rot[v, 0] = 0.0
                state.particle.albuf_local_rot[v, 1] = 0.0
                state.particle.albuf_local_rot[v, 2] = 0.0
                state.particle.albuf_local_rot[v, 3] = 1.0
            else:
                safe_bv = bvlen if bvlen > 1e-30 else 1.0
                dirx = bvx / safe_bv
                diry = bvy / safe_bv
                dirz = bvz / safe_bv
                (ipx, ipy, ipz, ipw) = dmath.quat_inverse(state.particle.step_basic_rotations[par,
                        0], state.particle.step_basic_rotations[par, 1], state.particle.step_basic_rotations[par,
                        2], state.particle.step_basic_rotations[par, 3])
                (lpx, lpy, lpz) = dmath.quat_rotate(ipx, ipy, ipz, ipw, dirx, diry, dirz)
                (lrx, lry, lrz, lrw) = dmath.quat_mul(ipx, ipy, ipz, ipw, state.particle.step_basic_rotations[v,
                        0], state.particle.step_basic_rotations[v, 1], state.particle.step_basic_rotations[v,
                        2], state.particle.step_basic_rotations[v, 3])
                state.particle.albuf_length[v] = avlen
                state.particle.albuf_local_pos[v, 0] = lpx
                state.particle.albuf_local_pos[v, 1] = lpy
                state.particle.albuf_local_pos[v, 2] = lpz
                state.particle.albuf_local_rot[v, 0] = lrx
                state.particle.albuf_local_rot[v, 1] = lry
                state.particle.albuf_local_rot[v, 2] = lrz
                state.particle.albuf_local_rot[v, 3] = lrw
        if state.team.angle_use_restoration[vt] != 0:
            state.particle.albuf_restore[v, 0] = bvx
            state.particle.albuf_restore[v, 1] = bvy
            state.particle.albuf_restore[v, 2] = bvz


@wp.kernel
def buffer_carried_angle_state(state: ClothState, substep: int, level: int, iteration: int):
    buffer_carried_angle_state_element(state, wp.tid(), substep, level, iteration)


@wp.func
def solve_angle_constraint_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    power3 = state.frame_scalar.frame_float[SCAL_POWER3]
    group = thread_index
    if group + 1 < state.derived.angle_root_offsets.shape[0]:
        start = state.derived.angle_root_offsets[group]
        stop = state.derived.angle_root_offsets[group + 1]
        for pass_index in range(ANGLE_LIMIT_ITERATION):
            angle_rot_ratio = 0.1 + (0.5 - 0.1) * (float(pass_index) / 2.0)
            for slot in range(start, stop):
                e = state.derived.angle_root_entries[slot]
                v = state.derived.angle_pass_vertices[e]
                p = state.derived.angle_pass_parents[e]
                vt = state.particle.team[v]
                if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
                        vt) and state.team.update_count[vt] > substep:
                    ul = state.team.angle_use_limit[vt] != 0
                    ur = state.team.angle_use_restoration[vt] != 0
                    if ul or ur:
                        c_inv = 1.0 / (1.0 + state.particle.friction[v] * FRICTION_MASS)
                        p_inv = 1.0 / (1.0 + state.particle.friction[p] * FRICTION_MASS)
                        p_mv = state.particle.attr_move[p] != 0
                        if ul:
                            kernels.do_angle_limit(v, p, vt, c_inv, p_inv, p_mv, state.particle.next_positions,
                                    state.particle.velocity_positions, state.particle.albuf_rotation,
                                    state.particle.albuf_local_pos, state.particle.albuf_local_rot,
                                    state.particle.albuf_length, state.particle.depth, state.team.angle_limit_lut,
                                    state.team.angle_limit_stiffness)
                        if ur:
                            kernels.do_angle_restoration(v, p, vt, c_inv, p_inv, p_mv, angle_rot_ratio,
                                    power3, state.particle.next_positions, state.particle.velocity_positions,
                                    state.particle.albuf_restore, state.particle.depth,
                                    state.team.angle_restoration_lut, state.team.angle_restoration_attenuation,
                                    state.team.angle_restoration_gravity_falloff,
                                    state.team.gravity_dot)


@wp.kernel
def solve_angle_constraint(state: ClothState, substep: int, level: int, iteration: int):
    solve_angle_constraint_element(state, wp.tid(), substep, level, iteration)


@wp.func
def clear_distance_accumulator_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    state.derived.distance_correction_fixed[p, 0] = wp.int64(0)
    state.derived.distance_correction_fixed[p, 1] = wp.int64(0)
    state.derived.distance_correction_fixed[p, 2] = wp.int64(0)
    state.derived.distance_count[p] = wp.int64(0)


@wp.kernel
def clear_distance_accumulator(state: ClothState, substep: int, level: int, iteration: int):
    clear_distance_accumulator_element(state, wp.tid(), substep, level, iteration)


@wp.func
def solve_bending_constraint_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    power1 = state.frame_scalar.frame_float[SCAL_POWER1]
    e = thread_index
    team = state.bending.team[e]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            team) and state.team.update_count[team] > substep and (state.team.bending_stiffness[
            team] >= 1e-06):
        stiffness = dmath.saturate(state.team.bending_stiffness[team] * power1)
        pp0 = state.bending.pair[e, 0]
        pp1 = state.bending.pair[e, 1]
        pp2 = state.bending.pair[e, 2]
        pp3 = state.bending.pair[e, 3]
        rest = state.bending.rest[e]
        sgn = state.bending.sign[e]
        a0x = state.particle.next_positions[pp0, 0]
        a0y = state.particle.next_positions[pp0, 1]
        a0z = state.particle.next_positions[pp0, 2]
        a1x = state.particle.next_positions[pp1, 0]
        a1y = state.particle.next_positions[pp1, 1]
        a1z = state.particle.next_positions[pp1, 2]
        a2x = state.particle.next_positions[pp2, 0]
        a2y = state.particle.next_positions[pp2, 1]
        a2z = state.particle.next_positions[pp2, 2]
        a3x = state.particle.next_positions[pp3, 0]
        a3y = state.particle.next_positions[pp3, 1]
        a3z = state.particle.next_positions[pp3, 2]
        if state.particle.attr_move[pp0] == 0:
            inv0 = BENDING_FIXED_INVERSE_MASS
        else:
            inv0 = dmath.calc_inverse_mass(state.particle.friction[pp0],
                    state.particle.depth[pp0])
        if state.particle.attr_move[pp1] == 0:
            inv1 = BENDING_FIXED_INVERSE_MASS
        else:
            inv1 = dmath.calc_inverse_mass(state.particle.friction[pp1],
                    state.particle.depth[pp1])
        if state.particle.attr_move[pp2] == 0:
            inv2 = BENDING_FIXED_INVERSE_MASS
        else:
            inv2 = dmath.calc_inverse_mass(state.particle.friction[pp2],
                    state.particle.depth[pp2])
        if state.particle.attr_move[pp3] == 0:
            inv3 = BENDING_FIXED_INVERSE_MASS
        else:
            inv3 = dmath.calc_inverse_mass(state.particle.friction[pp3],
                    state.particle.depth[pp3])
        scale_ratio = state.team.scale_ratio[team]
        negative_sign = kernels.component_reflection_sign(state.team.component_world_reflected,
                team)
        result = wp.bool(False)
        a0dx = float(0.0)
        a0dy = float(0.0)
        a0dz = float(0.0)
        a1dx = float(0.0)
        a1dy = float(0.0)
        a1dz = float(0.0)
        a2dx = float(0.0)
        a2dy = float(0.0)
        a2dz = float(0.0)
        a3dx = float(0.0)
        a3dy = float(0.0)
        a3dz = float(0.0)
        if sgn == VOLUME_SIGN:
            volume_rest = rest * scale_ratio * negative_sign
            (cx, cy, cz) = dmath.cross3(a1x - a0x, a1y - a0y, a1z - a0z, a2x - a0x, a2y - a0y,
                    a2z - a0z)
            volume = ONE_SIXTH * (cx * (a3x - a0x) + cy * (a3y - a0y) + cz * (a3z -
                    a0z)) * VOLUME_SCALE
            (g0x, g0y, g0z) = dmath.cross3(a1x - a2x, a1y - a2y, a1z - a2z, a3x - a2x, a3y - a2y,
                    a3z - a2z)
            (g1x, g1y, g1z) = dmath.cross3(a2x - a0x, a2y - a0y, a2z - a0z, a3x - a0x, a3y - a0y,
                    a3z - a0z)
            (g2x, g2y, g2z) = dmath.cross3(a0x - a1x, a0y - a1y, a0z - a1z, a3x - a1x, a3y - a1y,
                    a3z - a1z)
            (g3x, g3y, g3z) = dmath.cross3(a1x - a0x, a1y - a0y, a1z - a0z, a2x - a0x, a2y - a0y,
                    a2z - a0z)
            lam = inv0 * (g0x * g0x + g0y * g0y + g0z * g0z) + inv1 * (g1x * g1x + g1y * g1y +
                    g1z * g1z) + inv2 * (g2x * g2x + g2y * g2y + g2z * g2z) + inv3 * (g3x * g3x +
                    g3y * g3y + g3z * g3z)
            lam = lam * VOLUME_SCALE
            if wp.abs(lam) >= 1e-06:
                lam = stiffness * (volume_rest - volume) / lam
                a0dx = lam * inv0 * g0x
                a0dy = lam * inv0 * g0y
                a0dz = lam * inv0 * g0z
                a1dx = lam * inv1 * g1x
                a1dy = lam * inv1 * g1y
                a1dz = lam * inv1 * g1z
                a2dx = lam * inv2 * g2x
                a2dy = lam * inv2 * g2y
                a2dz = lam * inv2 * g2z
                a3dx = lam * inv3 * g3x
                a3dy = lam * inv3 * g3y
                a3dz = lam * inv3 * g3z
                result = True
        else:
            rest_angle = rest * float(sgn) * negative_sign
            ex = a3x - a2x
            ey = a3y - a2y
            ez = a3z - a2z
            elen = dmath.length3(ex, ey, ez)
            ok = elen >= 1e-08
            safe_elen = elen if elen > 1e-30 else 1.0
            inv_elen = 1.0 / safe_elen
            (nn1x, nn1y, nn1z) = dmath.cross3(a2x - a0x, a2y - a0y, a2z - a0z, a3x - a0x, a3y - a0y,
                    a3z - a0z)
            (nn2x, nn2y, nn2z) = dmath.cross3(a3x - a1x, a3y - a1y, a3z - a1z, a2x - a1x, a2y - a1y,
                    a2z - a1z)
            sq1 = nn1x * nn1x + nn1y * nn1y + nn1z * nn1z
            sq2 = nn2x * nn2x + nn2y * nn2y + nn2z * nn2z
            ok = ok and sq1 != 0.0 and (sq2 != 0.0)
            safe_sq1 = sq1 if sq1 > 1e-30 else 1.0
            safe_sq2 = sq2 if sq2 > 1e-30 else 1.0
            nn1x = nn1x / safe_sq1
            nn1y = nn1y / safe_sq1
            nn1z = nn1z / safe_sq1
            nn2x = nn2x / safe_sq2
            nn2y = nn2y / safe_sq2
            nn2z = nn2z / safe_sq2
            d0x = nn1x * elen
            d0y = nn1y * elen
            d0z = nn1z * elen
            d1x = nn2x * elen
            d1y = nn2y * elen
            d1z = nn2z * elen
            dot03 = (a0x - a3x) * ex + (a0y - a3y) * ey + (a0z - a3z) * ez
            dot13 = (a1x - a3x) * ex + (a1y - a3y) * ey + (a1z - a3z) * ez
            d2x = dot03 * inv_elen * nn1x + dot13 * inv_elen * nn2x
            d2y = dot03 * inv_elen * nn1y + dot13 * inv_elen * nn2y
            d2z = dot03 * inv_elen * nn1z + dot13 * inv_elen * nn2z
            dot20 = (a2x - a0x) * ex + (a2y - a0y) * ey + (a2z - a0z) * ez
            dot21 = (a2x - a1x) * ex + (a2y - a1y) * ey + (a2z - a1z) * ez
            d3x = dot20 * inv_elen * nn1x + dot21 * inv_elen * nn2x
            d3y = dot20 * inv_elen * nn1y + dot21 * inv_elen * nn2y
            d3z = dot20 * inv_elen * nn1z + dot21 * inv_elen * nn2z
            (un1x, un1y, un1z) = dmath.normalize3(nn1x, nn1y, nn1z)
            (un2x, un2y, un2z) = dmath.normalize3(nn2x, nn2y, nn2z)
            dotu = dmath.clamp1(un1x * un2x + un1y * un2y + un1z * un2z)
            phi = wp.acos(dotu)
            lam = inv0 * (d0x * d0x + d0y * d0y + d0z * d0z) + inv1 * (d1x * d1x + d1y * d1y +
                    d1z * d1z) + inv2 * (d2x * d2x + d2y * d2y + d2z * d2z) + inv3 * (d3x * d3x +
                    d3y * d3y + d3z * d3z)
            ok = ok and lam != 0.0
            (crx, cry, crz) = dmath.cross3(un1x, un1y, un1z, un2x, un2y, un2z)
            dir_sign = dmath.fsign(crx * ex + cry * ey + crz * ez)
            phi = phi * dir_sign
            if ok:
                lam = (rest_angle - phi) / lam * stiffness
                a0dx = dmath.negate(inv0) * lam * d0x
                a0dy = dmath.negate(inv0) * lam * d0y
                a0dz = dmath.negate(inv0) * lam * d0z
                a1dx = dmath.negate(inv1) * lam * d1x
                a1dy = dmath.negate(inv1) * lam * d1y
                a1dz = dmath.negate(inv1) * lam * d1z
                a2dx = dmath.negate(inv2) * lam * d2x
                a2dy = dmath.negate(inv2) * lam * d2y
                a2dz = dmath.negate(inv2) * lam * d2z
                a3dx = dmath.negate(inv3) * lam * d3x
                a3dy = dmath.negate(inv3) * lam * d3y
                a3dz = dmath.negate(inv3) * lam * d3z
                result = True
        if result:
            wp.atomic_add(state.derived.distance_correction_fixed, pp0, 0, wp.int64(a0dx * TO_FIXED))
            wp.atomic_add(state.derived.distance_correction_fixed, pp0, 1, wp.int64(a0dy * TO_FIXED))
            wp.atomic_add(state.derived.distance_correction_fixed, pp0, 2, wp.int64(a0dz * TO_FIXED))
            wp.atomic_add(state.derived.distance_count, pp0, wp.int64(1))
            wp.atomic_add(state.derived.distance_correction_fixed, pp1, 0, wp.int64(a1dx * TO_FIXED))
            wp.atomic_add(state.derived.distance_correction_fixed, pp1, 1, wp.int64(a1dy * TO_FIXED))
            wp.atomic_add(state.derived.distance_correction_fixed, pp1, 2, wp.int64(a1dz * TO_FIXED))
            wp.atomic_add(state.derived.distance_count, pp1, wp.int64(1))
            wp.atomic_add(state.derived.distance_correction_fixed, pp2, 0, wp.int64(a2dx * TO_FIXED))
            wp.atomic_add(state.derived.distance_correction_fixed, pp2, 1, wp.int64(a2dy * TO_FIXED))
            wp.atomic_add(state.derived.distance_correction_fixed, pp2, 2, wp.int64(a2dz * TO_FIXED))
            wp.atomic_add(state.derived.distance_count, pp2, wp.int64(1))
            wp.atomic_add(state.derived.distance_correction_fixed, pp3, 0, wp.int64(a3dx * TO_FIXED))
            wp.atomic_add(state.derived.distance_correction_fixed, pp3, 1, wp.int64(a3dy * TO_FIXED))
            wp.atomic_add(state.derived.distance_correction_fixed, pp3, 2, wp.int64(a3dz * TO_FIXED))
            wp.atomic_add(state.derived.distance_count, pp3, wp.int64(1))


@wp.kernel
def solve_bending_constraint(state: ClothState, substep: int, level: int, iteration: int):
    solve_bending_constraint_element(state, wp.tid(), substep, level, iteration)


@wp.func
def apply_distance_accumulator_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    mt = state.particle.team[p]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt) and state.team.update_count[mt] > substep and (state.particle.attr_move[p] != 0
            ) and (state.derived.distance_count[p] > 0):
        inv_c = 1.0 / float(state.derived.distance_count[p])
        state.particle.next_positions[p, 0] = state.particle.next_positions[p,
                0] + float(state.derived.distance_correction_fixed[p, 0]) / TO_FIXED * inv_c
        state.particle.next_positions[p, 1] = state.particle.next_positions[p,
                1] + float(state.derived.distance_correction_fixed[p, 1]) / TO_FIXED * inv_c
        state.particle.next_positions[p, 2] = state.particle.next_positions[p,
                2] + float(state.derived.distance_correction_fixed[p, 2]) / TO_FIXED * inv_c


@wp.kernel
def apply_distance_accumulator(state: ClothState, substep: int, level: int, iteration: int):
    apply_distance_accumulator_element(state, wp.tid(), substep, level, iteration)


@wp.func
def measure_collider_edge_feet_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    pair = thread_index
    ee = state.edge_pairs.edge[pair]
    e0 = state.collision_edges.edge[ee, 0]
    et = state.particle.team[e0]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            et) and state.team.update_count[et] > substep and (state.team.collision_mode[et] ==
            COLLISION_EDGE):
        state.derived.collider_edge_foot_ratio[pair] = kernels.do_measure_edge_foot(e0,
                state.collision_edges.edge[ee, 1], state.edge_pairs.collider[pair], et,
                state.particle.next_positions, state.particle.old_positions,
                state.particle.depth,
                state.team.radius_lut, state.team.scale_ratio,
                state.collider.active, state.collider.kind,
                state.collider.work_old_pos,
                state.collider.work_next_pos, state.collider.work_radius,
                state.collider.work_inv_old_rot,
                state.collider.work_rot, state.collider.work_inv_rot,
                state.collider_faces_index, state.collider_faces.vertex,
                state.collider_faces.edge_normal, state.collider_faces.normal,
                state.collider_vertices.local_position,
                state.collider_vertices.pseudo_normal,
                state.collider.work_aabb_min, state.collider.work_aabb_max,
                state.derived.contact_path_incidence_gate_cos[CONTACT_PATH_COLLIDER])


@wp.kernel
def measure_collider_edge_feet(state: ClothState, substep: int, level: int,
        iteration: int):
    measure_collider_edge_feet_element(state, wp.tid(), substep, level, iteration)


@wp.func
def apply_collider_spring_response_element(state: ClothState, thread_index: int,
        substep: int, level: int, iteration: int):
    pair = thread_index
    p = state.point_pairs.particle[pair]
    mt = state.particle.team[p]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt) and state.team.update_count[mt] > substep and (state.team.collision_mode[mt] ==
            COLLISION_POINT) and (state.team.is_spring[mt] != 0) and (
            state.derived.collider_point_contact[pair, 0] != 0.0):
        (dist, cx, cy, cz) = kernels.do_spring_response(p, mt,
                state.derived.collider_point_contact[pair, 1],
                state.derived.collider_point_contact[pair, 2],
                state.derived.collider_point_contact[pair, 3],
                state.derived.collider_point_contact[pair, 4],
                state.particle.next_positions, state.particle.base_positions,
                state.particle.depth, state.team.radius_lut, state.team.scale_ratio,
                state.team.limit_distance_lut)
        state.derived.collider_point_contact[pair, 1] = dist
        state.derived.collider_point_contact[pair, 2] = cx
        state.derived.collider_point_contact[pair, 3] = cy
        state.derived.collider_point_contact[pair, 4] = cz


@wp.kernel
def apply_collider_spring_response(state: ClothState, substep: int, level: int,
        iteration: int):
    apply_collider_spring_response_element(state, wp.tid(), substep, level, iteration)


@wp.func
def measure_collider_point_contacts_element(state: ClothState, thread_index: int,
        substep: int, level: int, iteration: int):
    pair = thread_index
    p = state.point_pairs.particle[pair]
    mt = state.particle.team[p]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt) and state.team.update_count[mt] > substep and (state.team.collision_mode[mt] ==
            COLLISION_POINT):
        (live, dist, cx, cy, cz, nx, ny, nz) = kernels.do_measure_point_contact(p,
                state.point_pairs.collider[pair], mt, state.particle.next_positions,
                state.particle.old_positions,
                state.particle.depth, state.team.radius_lut, state.team.scale_ratio,
                state.collider.kind,
                state.collider.active, state.collider.work_old_pos, state.collider.work_next_pos,
                state.collider.work_radius, state.collider.work_inv_old_rot,
                state.collider.work_rot, state.collider.work_inv_rot,
                state.collider_faces_index, state.collider_faces.vertex,
                state.collider_faces.edge_normal, state.collider_faces.normal,
                state.collider_vertices.local_position,
                state.collider_vertices.pseudo_normal,
                state.collider.work_aabb_min,
                state.collider.work_aabb_max,
                state.derived.contact_path_incidence_gate_cos[CONTACT_PATH_COLLIDER])
        state.derived.collider_point_contact[pair, 0] = live
        state.derived.collider_point_contact[pair, 1] = dist
        state.derived.collider_point_contact[pair, 2] = cx
        state.derived.collider_point_contact[pair, 3] = cy
        state.derived.collider_point_contact[pair, 4] = cz
        state.derived.collider_point_contact[pair, 5] = nx
        state.derived.collider_point_contact[pair, 6] = ny
        state.derived.collider_point_contact[pair, 7] = nz


@wp.kernel
def measure_collider_point_contacts(state: ClothState, substep: int, level: int,
        iteration: int):
    measure_collider_point_contacts_element(state, wp.tid(), substep, level, iteration)


@wp.func
def measure_collider_edge_contacts_element(state: ClothState, thread_index: int,
        substep: int, level: int, iteration: int):
    pair = thread_index
    ee = state.edge_pairs.edge[pair]
    e0 = state.collision_edges.edge[ee, 0]
    et = state.particle.team[e0]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            et) and state.team.update_count[et] > substep and (state.team.collision_mode[et] ==
            COLLISION_EDGE):
        (live, dist, a0x, a0y, a0z, a1x, a1y, a1z, nx, ny, nz) = \
                kernels.do_measure_edge_contact(e0, state.collision_edges.edge[ee, 1],
                state.edge_pairs.collider[pair], et,
                state.derived.collider_edge_foot_ratio[pair], state.particle.next_positions,
                state.particle.old_positions, state.particle.depth, state.team.radius_lut,
                state.team.scale_ratio, state.collider.kind, state.collider.active,
                state.collider.work_old_pos, state.collider.work_next_pos,
                state.collider.work_radius, state.collider.work_inv_old_rot,
                state.collider.work_rot, state.collider.work_inv_rot,
                state.collider_faces_index, state.collider_faces.vertex,
                state.collider_faces.edge_normal, state.collider_faces.normal,
                state.collider_vertices.local_position,
                state.collider_vertices.pseudo_normal,
                state.collider.work_aabb_min,
                state.collider.work_aabb_max,
                state.derived.contact_path_incidence_gate_cos[CONTACT_PATH_COLLIDER])
        state.derived.collider_edge_contact[pair, 0] = live
        state.derived.collider_edge_contact[pair, 1] = dist
        state.derived.collider_edge_contact[pair, 2] = a0x
        state.derived.collider_edge_contact[pair, 3] = a0y
        state.derived.collider_edge_contact[pair, 4] = a0z
        state.derived.collider_edge_contact[pair, 5] = a1x
        state.derived.collider_edge_contact[pair, 6] = a1y
        state.derived.collider_edge_contact[pair, 7] = a1z
        state.derived.collider_edge_contact[pair, 8] = nx
        state.derived.collider_edge_contact[pair, 9] = ny
        state.derived.collider_edge_contact[pair, 10] = nz


@wp.kernel
def measure_collider_edge_contacts(state: ClothState, substep: int, level: int,
        iteration: int):
    measure_collider_edge_contacts_element(state, wp.tid(), substep, level, iteration)


@wp.func
def gather_collider_point_contacts_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    mt = state.particle.team[p]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt) and state.team.update_count[mt] > substep:
        (active, count, near_count, min_dist, psumx, psumy, psumz, nsumx, nsumy, nsumz, nnx, nny,
                nnz) = kernels.do_solve_point_gather(p, state.particle.team,
                state.particle.depth, state.team.collision_mode, state.team.radius_lut,
                state.team.scale_ratio, state.derived.point_pair_csr_offsets,
                state.derived.point_pair_csr_order, state.derived.collider_point_contact)
        state.derived.solve_point_active[p] = active
        state.derived.solve_point_contact_count[p] = count
        state.derived.solve_point_near_count[p] = near_count
        state.derived.solve_point_minimum_distance[p] = min_dist
        state.derived.solve_point_push_sum[p, 0] = psumx
        state.derived.solve_point_push_sum[p, 1] = psumy
        state.derived.solve_point_push_sum[p, 2] = psumz
        state.derived.solve_point_normal_sum[p, 0] = nsumx
        state.derived.solve_point_normal_sum[p, 1] = nsumy
        state.derived.solve_point_normal_sum[p, 2] = nsumz
        state.derived.solve_point_near_normal_sum[p, 0] = nnx
        state.derived.solve_point_near_normal_sum[p, 1] = nny
        state.derived.solve_point_near_normal_sum[p, 2] = nnz


@wp.kernel
def gather_collider_point_contacts(state: ClothState, substep: int, level: int, iteration: int):
    gather_collider_point_contacts_element(state, wp.tid(), substep, level, iteration)


@wp.func
def resolve_collider_point_contacts_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    mt = state.particle.team[p]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt) and state.team.update_count[mt] > substep:
        kernels.do_solve_point_resolve(p, state.derived.solve_point_active[p],
                state.derived.solve_point_contact_count[p], state.derived.solve_point_near_count[p],
                state.derived.solve_point_minimum_distance[p], state.derived.solve_point_push_sum[p,
                0], state.derived.solve_point_push_sum[p, 1], state.derived.solve_point_push_sum[p,
                2], state.derived.solve_point_normal_sum[p, 0], state.derived.solve_point_normal_sum[p,
                1], state.derived.solve_point_normal_sum[p, 2], state.derived.solve_point_near_normal_sum[p,
                0], state.derived.solve_point_near_normal_sum[p, 1], state.derived.solve_point_near_normal_sum[p,
                2], state.particle.team, state.particle.next_positions, state.particle.depth,
                state.particle.friction, state.particle.collision_normals, state.particle.velocity_positions,
                state.team.radius_lut, state.team.scale_ratio, state.team.is_spring)


@wp.kernel
def resolve_collider_point_contacts(state: ClothState, substep: int, level: int,
        iteration: int):
    resolve_collider_point_contacts_element(state, wp.tid(), substep, level, iteration)


@wp.func
def clear_collision_accumulator_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    state.derived.distance_correction_fixed[p, 0] = wp.int64(0)
    state.derived.distance_correction_fixed[p, 1] = wp.int64(0)
    state.derived.distance_correction_fixed[p, 2] = wp.int64(0)
    state.derived.distance_count[p] = wp.int64(0)
    state.derived.collision_friction_fixed[p] = wp.int64(0)
    state.derived.collision_normal_fixed[p, 0] = wp.int64(0)
    state.derived.collision_normal_fixed[p, 1] = wp.int64(0)
    state.derived.collision_normal_fixed[p, 2] = wp.int64(0)


@wp.kernel
def clear_collision_accumulator(state: ClothState, substep: int, level: int, iteration: int):
    clear_collision_accumulator_element(state, wp.tid(), substep, level, iteration)


@wp.func
def solve_collider_edge_contacts_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    ee = thread_index
    et = state.particle.team[state.collision_edges.edge[ee, 0]]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            et) and state.team.update_count[et] > substep and (state.team.collision_mode[et] ==
            COLLISION_EDGE):
        kernels.do_solve_edge(ee, state.particle.team, state.particle.depth,
                state.particle.attr_move, state.team.radius_lut, state.team.scale_ratio,
                state.derived.edge_pair_csr_offsets, state.derived.edge_pair_csr_order,
                state.collision_edges.edge, state.derived.collider_edge_contact,
                state.derived.distance_correction_fixed, state.derived.distance_count,
                state.derived.collision_friction_fixed, state.derived.collision_normal_fixed)


@wp.kernel
def solve_collider_edge_contacts(state: ClothState, substep: int, level: int, iteration: int):
    solve_collider_edge_contacts_element(state, wp.tid(), substep, level, iteration)


@wp.func
def apply_collision_accumulator_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    mt = state.particle.team[p]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt) and state.team.update_count[mt] > substep and (state.team.collision_mode[mt] ==
            COLLISION_EDGE):
        cnt = state.derived.distance_count[p]
        if cnt > 0:
            inv_e = 1.0 / float(cnt)
            state.particle.next_positions[p, 0] = state.particle.next_positions[p,
                    0] + float(state.derived.distance_correction_fixed[p, 0]) / TO_FIXED * inv_e
            state.particle.next_positions[p, 1] = state.particle.next_positions[p,
                    1] + float(state.derived.distance_correction_fixed[p, 1]) / TO_FIXED * inv_e
            state.particle.next_positions[p, 2] = state.particle.next_positions[p,
                    2] + float(state.derived.distance_correction_fixed[p, 2]) / TO_FIXED * inv_e
        ef = float(state.derived.collision_friction_fixed[p]) / TO_FIXED
        if ef > state.particle.friction[p]:
            state.particle.friction[p] = ef
        enx = float(state.derived.collision_normal_fixed[p, 0]) / TO_FIXED
        eny = float(state.derived.collision_normal_fixed[p, 1]) / TO_FIXED
        enz = float(state.derived.collision_normal_fixed[p, 2]) / TO_FIXED
        if enx * enx + eny * eny + enz * enz > 0.0:
            (onx, ony, onz) = dmath.normalize3(enx, eny, enz)
            state.particle.collision_normals[p, 0] = onx
            state.particle.collision_normals[p, 1] = ony
            state.particle.collision_normals[p, 2] = onz


@wp.kernel
def apply_collision_accumulator(state: ClothState, substep: int, level: int, iteration: int):
    apply_collision_accumulator_element(state, wp.tid(), substep, level, iteration)


@wp.func
def solve_motion_constraint_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    e = thread_index
    mt = state.motion.team[e]
    use_max = state.team.motion_use_max_distance[mt] != 0
    use_backstop = state.team.motion_use_backstop[mt] != 0
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt) and state.team.update_count[mt] > substep and (use_max or use_backstop):
        index = state.motion.particle[e]
        stiffness = state.team.motion_stiffness[mt]
        backstop_radius = state.team.motion_backstop_radius[mt]
        o0 = state.particle.next_positions[index, 0]
        o1 = state.particle.next_positions[index, 1]
        o2 = state.particle.next_positions[index, 2]
        n0 = float(o0)
        n1 = float(o1)
        n2 = float(o2)
        b0 = state.particle.base_positions[index, 0]
        b1 = state.particle.base_positions[index, 1]
        b2 = state.particle.base_positions[index, 2]
        depth = state.particle.depth[index]
        radius = dmath.evaluate_team_lut(state.team.radius_lut, mt, depth)
        if radius < 0.0001:
            radius = 0.0001
        cfr = radius
        depth2 = depth * depth
        (dirx, diry, dirz) = dmath.quat_rotate(state.particle.base_rotations[index, 0],
                state.particle.base_rotations[index, 1], state.particle.base_rotations[index, 2],
                state.particle.base_rotations[index, 3], state.team.normal_axis_vector[mt, 0],
                state.team.normal_axis_vector[mt, 1], state.team.normal_axis_vector[mt, 2])
        if use_max:
            max_distance = dmath.evaluate_team_lut(state.team.motion_max_distance_lut, mt,
                    depth2)
            (cvx, cvy, cvz) = dmath.clamp_vector(n0 - b0, n1 - b1, n2 - b2, max_distance)
            n0 = b0 + cvx
            n1 = b1 + cvy
            n2 = b2 + cvz
        if use_backstop and backstop_radius > 0.0:
            backstop_distance = dmath.evaluate_team_lut(state.team.motion_backstop_lut, mt,
                    depth2)
            backstop_offset = backstop_distance + backstop_radius
            cx = b0 - dirx * backstop_offset
            cy = b1 - diry * backstop_offset
            cz = b2 - dirz * backstop_offset
            vx = n0 - cx
            vy = n1 - cy
            vz = n2 - cz
            center_distance = dmath.length3(vx, vy, vz)
            near = center_distance > EPSILON and center_distance < backstop_radius + cfr
            if near and center_distance < backstop_radius:
                safe_distance = center_distance if center_distance > 1e-30 else 1.0
                n0 = cx + vx / safe_distance * backstop_radius
                n1 = cy + vy / safe_distance * backstop_radius
                n2 = cz + vz / safe_distance * backstop_radius
        n0 = dmath.lerp(o0, n0, stiffness)
        n1 = dmath.lerp(o1, n1, stiffness)
        n2 = dmath.lerp(o2, n2, stiffness)
        state.particle.next_positions[index, 0] = n0
        state.particle.next_positions[index, 1] = n1
        state.particle.next_positions[index, 2] = n2
        state.particle.velocity_positions[index, 0] = state.particle.velocity_positions[index,
                0] + (n0 - o0) * 0.95
        state.particle.velocity_positions[index, 1] = state.particle.velocity_positions[index,
                1] + (n1 - o1) * 0.95
        state.particle.velocity_positions[index, 2] = state.particle.velocity_positions[index,
                2] + (n2 - o2) * 0.95


@wp.kernel
def solve_motion_constraint(state: ClothState, substep: int, level: int, iteration: int):
    solve_motion_constraint_element(state, wp.tid(), substep, level, iteration)


@wp.func
def clear_self_primitive_size_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    i = thread_index
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            i) and state.team.update_count[i] > substep:
        state.derived.self_max_fixed_size[i] = wp.int64(0)


@wp.kernel
def clear_self_primitive_size(state: ClothState, substep: int, level: int, iteration: int):
    clear_self_primitive_size_element(state, wp.tid(), substep, level, iteration)


@wp.func
def update_self_point_primitives_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    q = thread_index
    kernels.do_self_update_primitive(q, 1, state.self_points.team, state.self_points.particles,
            state.self_points.fix, state.self_points.ignore, state.self_points.prim_depth,
            state.self_points.inv_mass, state.self_points.thickness, state.self_points.aabb_min,
            state.self_points.aabb_max, state.self_points.intersect, state.self_points.use,
            state.team.use_point, state.team.self_thickness_lut, state.team.self_cloth_mass,
            state.team.scale_ratio, state.team.enabled, state.team.valid, state.team.component_world_scale,
            state.team.update_count, state.particle.next_positions, state.particle.old_positions,
            state.particle.friction, state.particle.intersect_flag, state.derived.self_counters,
            state.derived.self_max_fixed_size, substep)


@wp.kernel
def update_self_point_primitives(state: ClothState, substep: int, level: int, iteration: int):
    update_self_point_primitives_element(state, wp.tid(), substep, level, iteration)


@wp.func
def update_self_edge_primitives_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    q = thread_index
    kernels.do_self_update_primitive(q, 2, state.self_edges.team, state.self_edges.particles,
            state.self_edges.fix, state.self_edges.ignore, state.self_edges.prim_depth,
            state.self_edges.inv_mass, state.self_edges.thickness, state.self_edges.aabb_min,
            state.self_edges.aabb_max, state.self_edges.intersect, state.self_edges.use,
            state.team.use_edge, state.team.self_thickness_lut, state.team.self_cloth_mass,
            state.team.scale_ratio, state.team.enabled, state.team.valid, state.team.component_world_scale,
            state.team.update_count, state.particle.next_positions, state.particle.old_positions,
            state.particle.friction, state.particle.intersect_flag, state.derived.self_counters,
            state.derived.self_max_fixed_size, substep)


@wp.kernel
def update_self_edge_primitives(state: ClothState, substep: int, level: int, iteration: int):
    update_self_edge_primitives_element(state, wp.tid(), substep, level, iteration)


@wp.func
def update_self_triangle_primitives_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    q = thread_index
    kernels.do_self_update_primitive(q, 3, state.self_triangles.team, state.self_triangles.particles,
            state.self_triangles.fix, state.self_triangles.ignore, state.self_triangles.prim_depth,
            state.self_triangles.inv_mass, state.self_triangles.thickness, state.self_triangles.aabb_min,
            state.self_triangles.aabb_max, state.self_triangles.intersect, state.self_triangles.use,
            state.team.use_triangle, state.team.self_thickness_lut, state.team.self_cloth_mass,
            state.team.scale_ratio, state.team.enabled, state.team.valid, state.team.component_world_scale,
            state.team.update_count, state.particle.next_positions, state.particle.old_positions,
            state.particle.friction, state.particle.intersect_flag, state.derived.self_counters,
            state.derived.self_max_fixed_size, substep)


@wp.kernel
def update_self_triangle_primitives(state: ClothState, substep: int, level: int,
        iteration: int):
    update_self_triangle_primitives_element(state, wp.tid(), substep, level, iteration)


@wp.func
def publish_self_primitive_size_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    i = thread_index
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            i) and state.team.update_count[i] > substep:
        ms = float(state.derived.self_max_fixed_size[i]) / TO_FIXED
        state.team.self_max_primitive_size[i] = ms
        state.team.self_grid_size[i] = ms * SELF_COLLISION_UNIFORM_GRID_SCALE


@wp.kernel
def publish_self_primitive_size(state: ClothState, substep: int, level: int, iteration: int):
    publish_self_primitive_size_element(state, wp.tid(), substep, level, iteration)


@wp.func
def clear_self_contact_counters_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    tid = thread_index
    if tid == 0:
        state.derived.self_counters[SCL_EE_COUNT] = 0
        state.derived.self_counters[SCL_PT_COUNT] = 0


@wp.kernel
def clear_self_contact_counters(state: ClothState, substep: int, level: int, iteration: int):
    clear_self_contact_counters_element(state, wp.tid(), substep, level, iteration)


@wp.func
def query_self_edge_contacts_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    g = thread_index
    num_ct_slots = state.derived.self_contact_task_query_offsets.shape[0] - 1
    total_query = state.derived.self_contact_task_query_offsets[num_ct_slots]
    if g == 0:
        state.derived.self_counters[SCL_EE_COUNT] = state.derived.self_contact_task_edge_slot_offsets[
                num_ct_slots]
    if g < total_query:
        lo = int(0)
        hi = num_ct_slots
        while lo < hi:
            mid = lo + hi >> 1
            if state.derived.self_contact_task_query_offsets[mid + 1] <= g:
                lo = mid + 1
            else:
                hi = mid
        task = lo
        if state.derived.self_contact_task_kind[task] == 0:
            tgt_team = state.derived.self_contact_task_target_team[task]
            same = state.derived.self_contact_task_same_team[task]
            i = g - state.derived.self_contact_task_query_offsets[task]
            my_edge = state.derived.self_contact_task_source_start[task] + i
            my_team = state.self_edges.team[my_edge]
            capacity = state.team.self_contact_slots[my_team]
            base = state.derived.self_contact_task_edge_slot_offsets[task] + i * capacity
            for slot in range(capacity):
                state.derived.self_edge_contact_source[base + slot] = -1
                state.derived.self_edge_contact_target[base + slot] = -1
                state.derived.self_edge_contact_enabled[base + slot] = 0
                state.derived.self_edge_contact_gap_key[base + slot] = 0
                state.derived.self_edge_contact_thickness[base + slot] = 0.0
                state.derived.self_edge_contact_source_parameter[base + slot] = 0.0
                state.derived.self_edge_contact_target_parameter[base + slot] = 0.0
                state.derived.self_edge_contact_normal[base + slot, 0] = 0.0
                state.derived.self_edge_contact_normal[base + slot, 1] = 0.0
                state.derived.self_edge_contact_normal[base + slot, 2] = 0.0
            fill = int(0)
            if state.team.self_grid_size[tgt_team] > EPSILON and state.self_edges.use[my_edge
                    ] != 0 and (state.self_edges.ignore[my_edge] == 0):
                worst = int(0)
                accepted = int(0)
                query = wp.bvh_query_aabb(state.self_edges_index, wp.vec3(state.self_edges.aabb_min[my_edge,
                        0], state.self_edges.aabb_min[my_edge, 1], state.self_edges.aabb_min[my_edge,
                        2]), wp.vec3(state.self_edges.aabb_max[my_edge, 0], state.self_edges.aabb_max[my_edge,
                        1], state.self_edges.aabb_max[my_edge, 2]), wp.bvh_get_group_root(state.self_edges_index,
                        tgt_team))
                tgt_edge = int(0)
                while wp.bvh_query_next(query, tgt_edge):
                    if state.self_edges.use[tgt_edge] != 0 and state.self_edges.ignore[tgt_edge] == 0 and (same == 0 or my_edge < tgt_edge) and kernels.self_aabb_overlap(state.self_edges.aabb_min,
                            state.self_edges.aabb_max, my_edge, state.self_edges.aabb_min,
                            state.self_edges.aabb_max,
                            tgt_edge) and (not (state.self_edges.all_fix[my_edge] != 0 and state
                            .self_edges.all_fix[tgt_edge] != 0)):
                        if same == 0 or not kernels.self_connection_shared(state.self_edges.particles,
                                my_edge, state.self_edges.particles, tgt_edge):
                            accepted = accepted + 1
                            key = kernels.gap_order_key(kernels.self_box_gap(state.self_edges.aabb_min,
                                    state.self_edges.aabb_max, my_edge, state.self_edges.aabb_min,
                                    state.self_edges.aabb_max, tgt_edge))
                            chosen = int(-1)
                            if fill < capacity:
                                chosen = fill
                            elif kernels.self_ranks_before(key, tgt_edge,
                                    state.derived.self_edge_contact_gap_key[base + worst],
                                    state.derived.self_edge_contact_target[base + worst]):
                                chosen = worst
                            if chosen >= 0:
                                state.derived.self_edge_contact_source[base + chosen] = my_edge
                                state.derived.self_edge_contact_target[base + chosen] = tgt_edge
                                state.derived.self_edge_contact_gap_key[base + chosen] = key
                                if fill < capacity:
                                    fill = fill + 1
                                if fill == capacity:
                                    worst = kernels.self_worst_slot(state.derived.self_edge_contact_gap_key,
                                            state.derived.self_edge_contact_target, base,
                                            capacity)
                wp.atomic_max(state.derived.self_contact_demand, my_team, accepted)
                if accepted > capacity:
                    wp.atomic_add(state.derived.self_contact_overflow, my_team,
                            accepted - capacity)
                    state.derived.self_counters[SCL_ERROR] = 1
            for slot in range(fill):
                tgt = state.derived.self_edge_contact_target[base + slot]
                thickness = state.self_edges.thickness[my_edge] + state.self_edges.thickness[tgt
                        ]
                (s, t, nx, ny, nz, enable) = kernels.self_ee_geometry(my_edge, tgt, thickness,
                        state.self_edges.particles, state.particle.next_positions,
                        state.particle.old_positions)
                state.derived.self_edge_contact_thickness[base + slot] = thickness
                state.derived.self_edge_contact_source_parameter[base + slot] = s
                state.derived.self_edge_contact_target_parameter[base + slot] = t
                state.derived.self_edge_contact_normal[base + slot, 0] = nx
                state.derived.self_edge_contact_normal[base + slot, 1] = ny
                state.derived.self_edge_contact_normal[base + slot, 2] = nz
                state.derived.self_edge_contact_enabled[base + slot] = 1 if enable else 0


@wp.kernel
def query_self_edge_contacts(state: ClothState, substep: int, level: int, iteration: int):
    query_self_edge_contacts_element(state, wp.tid(), substep, level, iteration)


@wp.func
def query_self_point_contacts_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    g = thread_index
    num_ct_slots = state.derived.self_contact_task_query_offsets.shape[0] - 1
    total_query = state.derived.self_contact_task_query_offsets[num_ct_slots]
    if g == 0:
        state.derived.self_counters[SCL_PT_COUNT] = state.derived.self_contact_task_point_slot_offsets[
                num_ct_slots]
    if g < total_query:
        lo = int(0)
        hi = num_ct_slots
        while lo < hi:
            mid = lo + hi >> 1
            if state.derived.self_contact_task_query_offsets[mid + 1] <= g:
                lo = mid + 1
            else:
                hi = mid
        task = lo
        if state.derived.self_contact_task_kind[task] == 1:
            tgt_team = state.derived.self_contact_task_target_team[task]
            same = state.derived.self_contact_task_same_team[task]
            i = g - state.derived.self_contact_task_query_offsets[task]
            my_point = state.derived.self_contact_task_source_start[task] + i
            my_team = state.self_points.team[my_point]
            capacity = state.team.self_contact_slots[my_team]
            base = state.derived.self_contact_task_point_slot_offsets[task] + i * capacity
            for slot in range(capacity):
                state.derived.self_point_contact_source[base + slot] = -1
                state.derived.self_point_contact_target[base + slot] = -1
                state.derived.self_point_contact_enabled[base + slot] = 0
                state.derived.self_point_contact_gap_key[base + slot] = 0
                state.derived.self_point_contact_thickness[base + slot] = 0.0
                state.derived.self_point_contact_weights[base + slot, 0] = 0.0
                state.derived.self_point_contact_weights[base + slot, 1] = 0.0
                state.derived.self_point_contact_weights[base + slot, 2] = 0.0
                state.derived.self_point_contact_normal[base + slot, 0] = 0.0
                state.derived.self_point_contact_normal[base + slot, 1] = 0.0
                state.derived.self_point_contact_normal[base + slot, 2] = 0.0
            fill = int(0)
            if state.team.self_grid_size[tgt_team] > EPSILON and state.self_points.use[my_point
                    ] != 0 and (state.self_points.ignore[my_point] == 0):
                worst = int(0)
                accepted = int(0)
                query = wp.bvh_query_aabb(state.self_triangles_index, wp.vec3(state.self_points.aabb_min[my_point,
                        0], state.self_points.aabb_min[my_point, 1], state.self_points.aabb_min[my_point,
                        2]), wp.vec3(state.self_points.aabb_max[my_point, 0], state.self_points.aabb_max[my_point,
                        1], state.self_points.aabb_max[my_point, 2]), wp.bvh_get_group_root(state.self_triangles_index,
                        tgt_team))
                tgt_triangle = int(0)
                while wp.bvh_query_next(query, tgt_triangle):
                    if state.self_triangles.use[tgt_triangle] != 0 and state.self_triangles.ignore[tgt_triangle] == 0 and kernels.self_aabb_overlap(state.self_points.aabb_min,
                            state.self_points.aabb_max, my_point, state.self_triangles.aabb_min,
                            state.self_triangles.aabb_max,
                            tgt_triangle) and (not (state.self_points.all_fix[my_point] != 0 and
                            state.self_triangles.all_fix[tgt_triangle] != 0)):
                        if same == 0 or not kernels.self_connection_shared(state.self_points.particles,
                                my_point, state.self_triangles.particles, tgt_triangle):
                            accepted = accepted + 1
                            key = kernels.gap_order_key(kernels.self_box_gap(state.self_points.aabb_min,
                                    state.self_points.aabb_max, my_point, state.self_triangles.aabb_min,
                                    state.self_triangles.aabb_max, tgt_triangle))
                            chosen = int(-1)
                            if fill < capacity:
                                chosen = fill
                            elif kernels.self_ranks_before(key, tgt_triangle,
                                    state.derived.self_point_contact_gap_key[base + worst],
                                    state.derived.self_point_contact_target[base + worst]):
                                chosen = worst
                            if chosen >= 0:
                                state.derived.self_point_contact_source[base +
                                        chosen] = my_point
                                state.derived.self_point_contact_target[base +
                                        chosen] = tgt_triangle
                                state.derived.self_point_contact_gap_key[base + chosen] = key
                                if fill < capacity:
                                    fill = fill + 1
                                if fill == capacity:
                                    worst = kernels.self_worst_slot(state.derived.self_point_contact_gap_key,
                                            state.derived.self_point_contact_target, base,
                                            capacity)
                wp.atomic_max(state.derived.self_contact_demand, my_team, accepted)
                if accepted > capacity:
                    wp.atomic_add(state.derived.self_contact_overflow, my_team,
                            accepted - capacity)
                    state.derived.self_counters[SCL_ERROR] = 1
            for slot in range(fill):
                tgt = state.derived.self_point_contact_target[base + slot]
                thickness = state.self_points.thickness[my_point] + state.self_triangles.thickness[
                        tgt]
                (enable, nx, ny, nz, u, v, w) = kernels.self_pt_geometry(my_point, tgt,
                        thickness, state.self_points.particles, state.self_triangles.particles,
                        state.particle.next_positions, state.particle.old_positions)
                state.derived.self_point_contact_thickness[base + slot] = thickness
                state.derived.self_point_contact_weights[base + slot, 0] = u
                state.derived.self_point_contact_weights[base + slot, 1] = v
                state.derived.self_point_contact_weights[base + slot, 2] = w
                state.derived.self_point_contact_normal[base + slot, 0] = nx
                state.derived.self_point_contact_normal[base + slot, 1] = ny
                state.derived.self_point_contact_normal[base + slot, 2] = nz
                state.derived.self_point_contact_enabled[base + slot] = 1 if enable else 0


@wp.kernel
def query_self_point_contacts(state: ClothState, substep: int, level: int, iteration: int):
    query_self_point_contacts_element(state, wp.tid(), substep, level, iteration)


@wp.func
def measure_self_edge_contacts_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    e = thread_index
    ee_count = state.derived.self_counters[SCL_EE_COUNT]
    ee_lim = ee_count if ee_count < state.derived.self_edge_contact_source.shape[0
            ] else state.derived.self_edge_contact_source.shape[0]
    if e < ee_lim and state.derived.self_edge_contact_source[e] >= 0:
        (s, t, rawx, rawy, rawz, enable) = kernels.self_ee_geometry(
                state.derived.self_edge_contact_source[e],
                state.derived.self_edge_contact_target[e],
                state.derived.self_edge_contact_thickness[e],
                state.self_edges.particles, state.particle.next_positions,
                state.particle.old_positions)
        (nx, ny, nz) = kernels.self_contact_side_keep(rawx, rawy, rawz,
                state.derived.self_edge_contact_normal[e, 0],
                state.derived.self_edge_contact_normal[e, 1],
                state.derived.self_edge_contact_normal[e, 2])
        state.derived.self_edge_contact_source_parameter[e] = s
        state.derived.self_edge_contact_target_parameter[e] = t
        state.derived.self_edge_contact_normal[e, 0] = nx
        state.derived.self_edge_contact_normal[e, 1] = ny
        state.derived.self_edge_contact_normal[e, 2] = nz
        state.derived.self_edge_contact_enabled[e] = 1 if enable else 0


@wp.kernel
def measure_self_edge_contacts(state: ClothState, substep: int, level: int, iteration: int):
    measure_self_edge_contacts_element(state, wp.tid(), substep, level, iteration)


@wp.func
def measure_self_point_contacts_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    e = thread_index
    pt_count = state.derived.self_counters[SCL_PT_COUNT]
    pt_lim = pt_count if pt_count < state.derived.self_point_contact_source.shape[0
            ] else state.derived.self_point_contact_source.shape[0]
    if e < pt_lim and state.derived.self_point_contact_source[e] >= 0:
        (enable, rawx, rawy, rawz, u, v, w) = kernels.self_pt_geometry(
                state.derived.self_point_contact_source[e],
                state.derived.self_point_contact_target[e],
                state.derived.self_point_contact_thickness[e],
                state.self_points.particles, state.self_triangles.particles,
                state.particle.next_positions, state.particle.old_positions)
        (nx, ny, nz) = kernels.self_contact_side_keep(rawx, rawy, rawz,
                state.derived.self_point_contact_normal[e, 0],
                state.derived.self_point_contact_normal[e, 1],
                state.derived.self_point_contact_normal[e, 2])
        state.derived.self_point_contact_weights[e, 0] = u
        state.derived.self_point_contact_weights[e, 1] = v
        state.derived.self_point_contact_weights[e, 2] = w
        state.derived.self_point_contact_normal[e, 0] = nx
        state.derived.self_point_contact_normal[e, 1] = ny
        state.derived.self_point_contact_normal[e, 2] = nz
        state.derived.self_point_contact_enabled[e] = 1 if enable else 0


@wp.kernel
def measure_self_point_contacts(state: ClothState, substep: int, level: int, iteration: int):
    measure_self_point_contacts_element(state, wp.tid(), substep, level, iteration)


@wp.func
def clear_self_contact_accumulator_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    state.derived.distance_correction_fixed[p, 0] = wp.int64(0)
    state.derived.distance_correction_fixed[p, 1] = wp.int64(0)
    state.derived.distance_correction_fixed[p, 2] = wp.int64(0)
    state.derived.distance_count[p] = wp.int64(0)


@wp.kernel
def clear_self_contact_accumulator(state: ClothState, substep: int, level: int, iteration: int):
    clear_self_contact_accumulator_element(state, wp.tid(), substep, level, iteration)


@wp.func
def accumulate_self_edge_contacts_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    e = thread_index
    ee_count2 = state.derived.self_counters[SCL_EE_COUNT]
    ee_lim2 = ee_count2 if ee_count2 < state.derived.self_edge_contact_source.shape[0
            ] else state.derived.self_edge_contact_source.shape[0]
    if e < ee_lim2:
        if state.derived.self_edge_contact_enabled[e] != 0:
            my = state.derived.self_edge_contact_source[e]
            tgt = state.derived.self_edge_contact_target[e]
            s = state.derived.self_edge_contact_source_parameter[e]
            t = state.derived.self_edge_contact_target_parameter[e]
            nx = state.derived.self_edge_contact_normal[e, 0]
            ny = state.derived.self_edge_contact_normal[e, 1]
            nz = state.derived.self_edge_contact_normal[e, 2]
            thk = state.derived.self_edge_contact_thickness[e]
            a0 = state.self_edges.particles[my, 0]
            a1 = state.self_edges.particles[my, 1]
            b0 = state.self_edges.particles[tgt, 0]
            b1 = state.self_edges.particles[tgt, 1]
            l = kernels.do_self_edge_contact_gap(a0, a1, b0, b1, s, t, nx, ny, nz,
                    state.particle.next_positions)
            c = thk - l
            bb0 = 1.0 - s
            bb1 = s
            bb2 = 1.0 - t
            bb3 = t
            im0 = state.self_edges.inv_mass[my, 0]
            im1 = state.self_edges.inv_mass[my, 1]
            im20 = state.self_edges.inv_mass[tgt, 0]
            im21 = state.self_edges.inv_mass[tgt, 1]
            denom = im0 * bb0 * bb0 + im1 * bb1 * bb1 + im20 * bb2 * bb2 + im21 * bb3 * bb3
            if l <= thk and denom != 0.0:
                scale = c / denom
                s0 = scale * im0 * bb0
                s1 = scale * im1 * bb1
                s2 = scale * im20 * bb2
                s3 = scale * im21 * bb3
                honour = state.derived.contact_path_honor_intersect_freeze[CONTACT_PATH_SELF_COLLISION]
                fm = state.self_edges.fix[my]
                imk = state.self_edges.intersect[my] * honour
                fmt = state.self_edges.fix[tgt]
                imt = state.self_edges.intersect[tgt] * honour
                if fm >> 0 & 1 == 0 and imk >> 0 & 1 == 0:
                    wp.atomic_add(state.derived.distance_correction_fixed, a0, 0,
                            wp.int64(nx * s0 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, a0, 1,
                            wp.int64(ny * s0 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, a0, 2,
                            wp.int64(nz * s0 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_count, a0, wp.int64(1))
                if fm >> 1 & 1 == 0 and imk >> 1 & 1 == 0:
                    wp.atomic_add(state.derived.distance_correction_fixed, a1, 0,
                            wp.int64(nx * s1 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, a1, 1,
                            wp.int64(ny * s1 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, a1, 2,
                            wp.int64(nz * s1 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_count, a1, wp.int64(1))
                if fmt >> 0 & 1 == 0 and imt >> 0 & 1 == 0:
                    wp.atomic_add(state.derived.distance_correction_fixed, b0, 0,
                            wp.int64(dmath.negate(nx) * s2 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, b0, 1,
                            wp.int64(dmath.negate(ny) * s2 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, b0, 2,
                            wp.int64(dmath.negate(nz) * s2 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_count, b0, wp.int64(1))
                if fmt >> 1 & 1 == 0 and imt >> 1 & 1 == 0:
                    wp.atomic_add(state.derived.distance_correction_fixed, b1, 0,
                            wp.int64(dmath.negate(nx) * s3 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, b1, 1,
                            wp.int64(dmath.negate(ny) * s3 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, b1, 2,
                            wp.int64(dmath.negate(nz) * s3 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_count, b1, wp.int64(1))


@wp.kernel
def accumulate_self_edge_contacts(state: ClothState, substep: int, level: int, iteration: int):
    accumulate_self_edge_contacts_element(state, wp.tid(), substep, level, iteration)


@wp.func
def accumulate_self_point_contacts_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    e = thread_index
    pt_count2 = state.derived.self_counters[SCL_PT_COUNT]
    pt_lim2 = pt_count2 if pt_count2 < state.derived.self_point_contact_source.shape[0
            ] else state.derived.self_point_contact_source.shape[0]
    if e < pt_lim2:
        if state.derived.self_point_contact_enabled[e] != 0:
            my = state.derived.self_point_contact_source[e]
            tgt = state.derived.self_point_contact_target[e]
            thk = state.derived.self_point_contact_thickness[e]
            u = state.derived.self_point_contact_weights[e, 0]
            v = state.derived.self_point_contact_weights[e, 1]
            w = state.derived.self_point_contact_weights[e, 2]
            nx = state.derived.self_point_contact_normal[e, 0]
            ny = state.derived.self_point_contact_normal[e, 1]
            nz = state.derived.self_point_contact_normal[e, 2]
            pp = state.self_points.particles[my, 0]
            t0 = state.self_triangles.particles[tgt, 0]
            t1 = state.self_triangles.particles[tgt, 1]
            t2 = state.self_triangles.particles[tgt, 2]
            dist = kernels.do_self_point_contact_gap(pp, t0, t1, t2, u, v, w, nx, ny, nz,
                    state.particle.next_positions)
            c = dist - thk
            imp = state.self_points.inv_mass[my, 0]
            imt0 = state.self_triangles.inv_mass[tgt, 0]
            imt1 = state.self_triangles.inv_mass[tgt, 1]
            imt2 = state.self_triangles.inv_mass[tgt, 2]
            denom = imp + imt0 * u * u + imt1 * v * v + imt2 * w * w
            if dist < thk and denom != 0.0:
                scale = c / denom
                sp = scale * imp
                st0 = scale * imt0 * u
                st1 = scale * imt1 * v
                st2 = scale * imt2 * w
                honour = state.derived.contact_path_honor_intersect_freeze[CONTACT_PATH_SELF_COLLISION]
                fp = state.self_points.fix[my]
                ipk = state.self_points.intersect[my] * honour
                ft = state.self_triangles.fix[tgt]
                itk = state.self_triangles.intersect[tgt] * honour
                if fp >> 0 & 1 == 0 and ipk >> 0 & 1 == 0:
                    wp.atomic_add(state.derived.distance_correction_fixed, pp, 0,
                            wp.int64(dmath.negate(nx) * sp * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, pp, 1,
                            wp.int64(dmath.negate(ny) * sp * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, pp, 2,
                            wp.int64(dmath.negate(nz) * sp * TO_FIXED))
                    wp.atomic_add(state.derived.distance_count, pp, wp.int64(1))
                if ft >> 0 & 1 == 0 and itk >> 0 & 1 == 0:
                    wp.atomic_add(state.derived.distance_correction_fixed, t0, 0,
                            wp.int64(nx * st0 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, t0, 1,
                            wp.int64(ny * st0 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, t0, 2,
                            wp.int64(nz * st0 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_count, t0, wp.int64(1))
                if ft >> 1 & 1 == 0 and itk >> 1 & 1 == 0:
                    wp.atomic_add(state.derived.distance_correction_fixed, t1, 0,
                            wp.int64(nx * st1 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, t1, 1,
                            wp.int64(ny * st1 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, t1, 2,
                            wp.int64(nz * st1 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_count, t1, wp.int64(1))
                if ft >> 2 & 1 == 0 and itk >> 2 & 1 == 0:
                    wp.atomic_add(state.derived.distance_correction_fixed, t2, 0,
                            wp.int64(nx * st2 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, t2, 1,
                            wp.int64(ny * st2 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_correction_fixed, t2, 2,
                            wp.int64(nz * st2 * TO_FIXED))
                    wp.atomic_add(state.derived.distance_count, t2, wp.int64(1))


@wp.kernel
def accumulate_self_point_contacts(state: ClothState, substep: int, level: int, iteration: int):
    accumulate_self_point_contacts_element(state, wp.tid(), substep, level, iteration)


@wp.func
def apply_self_contact_accumulator_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    cnt = state.derived.distance_count[p]
    if cnt > 0:
        mt = state.particle.team[p]
        if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
                mt) and state.team.update_count[mt] > substep:
            inv = 1.0 / float(cnt)
            state.particle.next_positions[p, 0] = state.particle.next_positions[p,
                    0] + float(state.derived.distance_correction_fixed[p, 0]) / TO_FIXED * inv
            state.particle.next_positions[p, 1] = state.particle.next_positions[p,
                    1] + float(state.derived.distance_correction_fixed[p, 1]) / TO_FIXED * inv
            state.particle.next_positions[p, 2] = state.particle.next_positions[p,
                    2] + float(state.derived.distance_correction_fixed[p, 2]) / TO_FIXED * inv
    state.derived.distance_correction_fixed[p, 0] = wp.int64(0)
    state.derived.distance_correction_fixed[p, 1] = wp.int64(0)
    state.derived.distance_correction_fixed[p, 2] = wp.int64(0)
    state.derived.distance_count[p] = wp.int64(0)


@wp.kernel
def apply_self_contact_accumulator(state: ClothState, substep: int, level: int, iteration: int):
    apply_self_contact_accumulator_element(state, wp.tid(), substep, level, iteration)


@wp.func
def apply_particle_friction_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    e = thread_index
    sim_dt = state.frame_scalar.frame_float[SCAL_SIM_DT]
    mt = state.update_move.team[e]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt) and state.team.update_count[mt] > substep:
        pmi = state.update_move.particle[e]
        n0 = state.particle.next_positions[pmi, 0]
        n1 = state.particle.next_positions[pmi, 1]
        n2 = state.particle.next_positions[pmi, 2]
        o0 = state.particle.old_positions[pmi, 0]
        o1 = state.particle.old_positions[pmi, 1]
        o2 = state.particle.old_positions[pmi, 2]
        vo0 = state.particle.velocity_positions[pmi, 0]
        vo1 = state.particle.velocity_positions[pmi, 1]
        vo2 = state.particle.velocity_positions[pmi, 2]
        depth = state.particle.depth[pmi]
        friction = state.particle.friction[pmi]
        cn0 = state.particle.collision_normals[pmi, 0]
        cn1 = state.particle.collision_normals[pmi, 1]
        cn2 = state.particle.collision_normals[pmi, 2]
        cn_len2 = cn0 * cn0 + cn1 * cn1 + cn2 * cn2
        is_collision = cn_len2 > EPSILON and friction > EPSILON
        static_param = state.team.static_friction[mt] * state.team.scale_ratio[mt]
        dynamic_param = state.team.dynamic_friction[mt]
        sfp = state.particle.static_friction[pmi]
        static_on = static_param > 0.0
        vx = n0 - o0
        vy = n1 - o1
        vz = n2 - o2
        vdotcn = vx * cn0 + vy * cn1 + vz * cn2
        tgx = vx - vdotcn * cn0
        tgy = vy - vdotcn * cn1
        tgz = vz - vdotcn * cn2
        tangent_velocity = dmath.length3(tgx, tgy, tgz) / sim_dt
        increase = dmath.saturate(sfp + 0.04)
        dec_amount = (tangent_velocity - static_param) / 0.2
        if dec_amount < 0.05:
            dec_amount = 0.05
        decrease = dmath.saturate(sfp - dec_amount)
        new_static = increase if tangent_velocity < static_param else decrease
        decayed = dmath.saturate(sfp - 0.05)
        updated_sf = new_static if is_collision else decayed
        sfp_new = updated_sf if static_on else decayed
        rbx = float(0.0)
        rby = float(0.0)
        rbz = float(0.0)
        if static_on and is_collision:
            rbx = tgx * sfp_new
            rby = tgy * sfp_new
            rbz = tgz * sfp_new
        n0 = n0 - rbx
        n1 = n1 - rby
        n2 = n2 - rbz
        vo0 = vo0 - rbx
        vo1 = vo1 - rby
        vo2 = vo2 - rbz
        state.particle.static_friction[pmi] = sfp_new
        velx = (n0 - vo0) / sim_dt
        vely = (n1 - vo1) / sim_dt
        velz = (n2 - vo2) / sim_dt
        sq_velocity = velx * velx + vely * vely + velz * velz
        (nvx, nvy, nvz) = dmath.normalize3(velx, vely, velz)
        if not sq_velocity > EPSILON:
            nvx = 0.0
            nvy = 0.0
            nvz = 0.0
        dynamic_on = dynamic_param > 0.0
        dd = cn0 * nvx + cn1 * nvy + cn2 * nvz
        dd = 0.5 + 0.5 * dd
        dd = dd * dd
        dd = 1.0 - dd
        damp = dd * dmath.saturate(friction * dynamic_param)
        if dynamic_on and is_collision and (sq_velocity >= EPSILON):
            velx = velx - velx * damp
            vely = vely - vely * damp
            velz = velz - velz * damp
        state.particle.friction[pmi] = friction * 0.6
        speed_limit = state.team.particle_speed_limit[mt]
        max_len = speed_limit * state.team.scale_ratio[mt]
        if max_len < 0.0:
            max_len = 0.0
        if speed_limit >= 0.0:
            (velx, vely, velz) = dmath.clamp_vector(velx, vely, velz, max_len)
        angular = state.team.angular_velocity[mt]
        centrifugal = state.team.centrifugal_acceleration[mt]
        if angular > EPSILON and centrifugal > EPSILON:
            axx = state.team.rotation_axis[mt, 0]
            axy = state.team.rotation_axis[mt, 1]
            axz = state.team.rotation_axis[mt, 2]
            lpx = n0 - state.team.now_world_position[mt, 0]
            lpy = n1 - state.team.now_world_position[mt, 1]
            lpz = n2 - state.team.now_world_position[mt, 2]
            lp_dot = lpx * axx + lpy * axy + lpz * axz
            v2x = lpx - lp_dot * axx
            v2y = lpy - lp_dot * axy
            v2z = lpz - lp_dot * axz
            rr = dmath.length3(v2x, v2y, v2z)
            if rr > EPSILON and sq_velocity >= EPSILON:
                (nx2, ny2, nz2) = dmath.normalize3(v2x, v2y, v2z)
                mm = 1.0 + (1.0 - depth)
                ff = mm * angular * angular * rr
                (ucx, ucy, ucz) = dmath.cross3(axx, axy, axz, nx2, ny2, nz2)
                (uux, uuy, uuz) = dmath.normalize3(ucx, ucy, ucz)
                ff = ff * dmath.saturate(nvx * uux + nvy * uuy + nvz * uuz)
                addc = ff * centrifugal * 0.02
                velx = velx + nx2 * addc
                vely = vely + ny2 * addc
                velz = velz + nz2 * addc
        vw = state.team.velocity_weight[mt]
        state.particle.velocities[pmi, 0] = velx * vw
        state.particle.velocities[pmi, 1] = vely * vw
        state.particle.velocities[pmi, 2] = velz * vw
        state.particle.next_positions[pmi, 0] = n0
        state.particle.next_positions[pmi, 1] = n1
        state.particle.next_positions[pmi, 2] = n2


@wp.kernel
def apply_particle_friction(state: ClothState, substep: int, level: int, iteration: int):
    apply_particle_friction_element(state, wp.tid(), substep, level, iteration)


@wp.func
def commit_particle_velocity_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    p = thread_index
    sim_dt = state.frame_scalar.frame_float[SCAL_SIM_DT]
    mt = state.particle.team[p]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt) and state.team.update_count[mt] > substep:
        state.particle.real_velocities[p, 0] = (state.particle.next_positions[p,
                0] - state.particle.old_positions[p, 0]) / sim_dt
        state.particle.real_velocities[p, 1] = (state.particle.next_positions[p,
                1] - state.particle.old_positions[p, 1]) / sim_dt
        state.particle.real_velocities[p, 2] = (state.particle.next_positions[p,
                2] - state.particle.old_positions[p, 2]) / sim_dt
        state.particle.old_positions[p, 0] = state.particle.next_positions[p, 0]
        state.particle.old_positions[p, 1] = state.particle.next_positions[p, 1]
        state.particle.old_positions[p, 2] = state.particle.next_positions[p, 2]


@wp.kernel
def commit_particle_velocity(state: ClothState, substep: int, level: int, iteration: int):
    commit_particle_velocity_element(state, wp.tid(), substep, level, iteration)


@wp.func
def commit_collider_substep_pose_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    ci = thread_index
    cm = state.collider.team[ci]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            cm) and state.team.update_count[cm] > substep and (state.collider.active[ci] != 0):
        kernels.do_collider_end_step(ci, state.collider.now_positions, state.collider.now_rotations,
                state.collider.now_tips, state.collider.old_positions, state.collider.old_rotations,
                state.collider.old_tips)


@wp.kernel
def commit_collider_substep_pose(state: ClothState, substep: int, level: int, iteration: int):
    commit_collider_substep_pose_element(state, wp.tid(), substep, level, iteration)


@wp.func
def clear_particle_intersect_flag_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            state.particle.team[p]):
        state.particle.intersect_flag[p] = 0


@wp.kernel
def clear_particle_intersect_flag(state: ClothState, substep: int, level: int, iteration: int):
    clear_particle_intersect_flag_element(state, wp.tid(), substep, level, iteration)


@wp.func
def mark_particle_intersect_flag_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    e = thread_index
    ip_count = state.derived.self_counters[SCL_IP_COUNT]
    ip_lim = ip_count if ip_count < state.derived.self_intersect_pair_edge.shape[0
            ] else state.derived.self_intersect_pair_edge.shape[0]
    if e < ip_lim and state.derived.self_intersect_pair_edge[e] >= 0:
        edge_prim = state.derived.self_intersect_pair_edge[e]
        tri_prim = state.derived.self_intersect_pair_triangle[e]
        ep0 = state.self_edges.particles[edge_prim, 0]
        ep1 = state.self_edges.particles[edge_prim, 1]
        ta = state.self_triangles.particles[tri_prim, 0]
        tb = state.self_triangles.particles[tri_prim, 1]
        tc = state.self_triangles.particles[tri_prim, 2]
        px = state.particle.next_positions[ep0, 0]
        py = state.particle.next_positions[ep0, 1]
        pz = state.particle.next_positions[ep0, 2]
        qx = state.particle.next_positions[ep1, 0]
        qy = state.particle.next_positions[ep1, 1]
        qz = state.particle.next_positions[ep1, 2]
        ax = state.particle.next_positions[ta, 0]
        ay = state.particle.next_positions[ta, 1]
        az = state.particle.next_positions[ta, 2]
        bx = state.particle.next_positions[tb, 0]
        by = state.particle.next_positions[tb, 1]
        bz = state.particle.next_positions[tb, 2]
        cx = state.particle.next_positions[tc, 0]
        cy = state.particle.next_positions[tc, 1]
        cz = state.particle.next_positions[tc, 2]
        qpx = px - qx
        qpy = py - qy
        qpz = pz - qz
        acx = cx - ax
        acy = cy - ay
        acz = cz - az
        abx = bx - ax
        aby = by - ay
        abz = bz - az
        (nx, ny, nz) = dmath.cross3(abx, aby, abz, acx, acy, acz)
        d = qpx * nx + qpy * ny + qpz * nz
        ok = wp.abs(d) >= EPSILON
        p2x = px
        p2y = py
        p2z = pz
        qp2x = qpx
        qp2y = qpy
        qp2z = qpz
        if d < 0.0:
            p2x = qx
            p2y = qy
            p2z = qz
            qp2x = dmath.negate(qpx)
            qp2y = dmath.negate(qpy)
            qp2z = dmath.negate(qpz)
        d2 = wp.abs(d)
        apx = p2x - ax
        apy = p2y - ay
        apz = p2z - az
        tparam = apx * nx + apy * ny + apz * nz
        ok = ok and tparam >= 0.0 and (tparam <= d2)
        (ecx, ecy, ecz) = dmath.cross3(qp2x, qp2y, qp2z, apx, apy, apz)
        vparam = acx * ecx + acy * ecy + acz * ecz
        ok = ok and vparam >= 0.0 and (vparam <= d2)
        wparam = dmath.negate(abx * ecx + aby * ecy + abz * ecz)
        ok = ok and wparam >= 0.0 and (vparam + wparam <= d2)
        if ok:
            state.particle.intersect_flag[ep0] = 1
            state.particle.intersect_flag[ep1] = 1


@wp.kernel
def mark_particle_intersect_flag(state: ClothState, substep: int, level: int, iteration: int):
    mark_particle_intersect_flag_element(state, wp.tid(), substep, level, iteration)


@wp.func
def blend_particle_display_pose_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    sim_dt = state.frame_scalar.frame_float[SCAL_SIM_DT]
    mt = state.particle.team[p]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt):
        kernels.do_display_particle(p, mt, sim_dt, state.particle.positions, state.particle.rotations,
                state.particle.old_positions, state.particle.real_velocities, state.particle.display_positions,
                state.particle.vertex_root, state.particle.old_anim_positions, state.particle.old_anim_rotations,
                state.particle.temp_base_positions, state.particle.temp_base_rotations,
                state.derived.display_update_move_mask, state.team.now_update_time, state.team.old_time,
                state.team.time, state.team.blend_weight, state.team.running,
                state.team.component_world_reflected)


@wp.kernel
def blend_particle_display_pose(state: ClothState, substep: int, level: int, iteration: int):
    blend_particle_display_pose_element(state, wp.tid(), substep, level, iteration)


@wp.func
def resolve_collider_frame_pose_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    ci = thread_index
    cm = state.collider.team[ci]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            cm) and state.team.update_count[cm] > 0 and (state.collider.active[ci] != 0):
        kernels.do_collider_frame_pose(ci, state.collider.kind, state.collider.frame_positions,
                state.collider.frame_rotations, state.collider.frame_tips,
                state.collider.frame_radii, state.collider.work_rot,
                state.collider.work_inv_old_rot, state.collider.work_inv_rot,
                state.collider.work_radius, state.collider.work_old_pos,
                state.collider.work_next_pos, state.collider.work_aabb_min,
                state.collider.work_aabb_max, state.collider.mesh_local_bound_min,
                state.collider.mesh_local_bound_max)


@wp.kernel
def resolve_collider_frame_pose(state: ClothState, substep: int, level: int,
        iteration: int):
    resolve_collider_frame_pose_element(state, wp.tid(), substep, level, iteration)


@wp.func
def project_particle_out_of_colliders_element(state: ClothState, thread_index: int,
        substep: int, level: int, iteration: int):
    p = thread_index
    mt = state.particle.team[p]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt) and state.team.update_count[mt] > 0 and (state.derived.display_update_move_mask[p]
            != 0):
        (ox, oy, oz) = kernels.do_collider_exit(p, mt, state.particle.positions[p, 0],
                state.particle.positions[p, 1], state.particle.positions[p, 2],
                state.team.collision_mode, state.collider.team, state.collider.kind,
                state.collider.active, state.collider.work_next_pos,
                state.collider.work_radius, state.collider.work_rot,
                state.collider.work_inv_rot,
                state.collider_faces_index, state.collider_faces.vertex,
                state.collider_faces.edge_normal, state.collider_faces.normal,
                state.collider_vertices.local_position,
                state.collider_vertices.pseudo_normal,
                state.collider.work_aabb_min, state.collider.work_aabb_max,
                state.derived.point_pair_csr_offsets,
                state.derived.point_pair_csr_order, state.point_pairs.collider,
                state.particle.intersect_flag,
                state.derived.contact_path_incidence_gate_cos[CONTACT_PATH_COLLIDER],
                state.derived.contact_path_honor_intersect_freeze[CONTACT_PATH_COLLIDER])
        state.particle.positions[p, 0] = ox
        state.particle.positions[p, 1] = oy
        state.particle.positions[p, 2] = oz


@wp.kernel
def project_particle_out_of_colliders(state: ClothState, substep: int, level: int,
        iteration: int):
    project_particle_out_of_colliders_element(state, wp.tid(), substep, level, iteration)


@wp.func
def propagate_postline_rotation_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    group = thread_index
    if group + 1 < state.derived.postline_root_offsets.shape[0]:
        start = state.derived.postline_root_offsets[group]
        stop = state.derived.postline_root_offsets[group + 1]
        for slot in range(start, stop):
            i = state.derived.postline_root_entries[slot]
            entry = state.derived.postline_entry_vertices[i]
            et = state.particle.team[entry]
            if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
                    et):
                kernels.do_postline_entry(entry, et, state.derived.postline_child_offsets[i],
                        state.derived.postline_child_offsets[i + 1], state.derived.postline_child_vertices,
                        state.particle.positions, state.particle.rotations, state.particle.temp_base_positions,
                        state.particle.temp_base_rotations, state.particle.vertex_local_positions,
                        state.particle.vertex_local_rotations, state.particle.attr_invalid,
                        state.particle.attr_zero_distance, state.particle.attr_move, state.particle.team,
                        state.team.rotational_interpolation, state.team.root_rotation, state.team.blend_weight,
                        state.team.animation_pose_ratio, state.team.component_world_reflected)


@wp.kernel
def propagate_postline_rotation(state: ClothState, substep: int, level: int, iteration: int):
    propagate_postline_rotation_element(state, wp.tid(), substep, level, iteration)


@wp.func
def accumulate_triangle_basis_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    tri_idx = thread_index
    tt_team = state.triangles.team[tri_idx]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            tt_team):
        kernels.do_triangle_normal_tangent(tri_idx, tt_team, state.triangles.triangle,
                state.particle.positions, state.particle.uv, state.team.component_world_reflected,
                state.derived.triangle_normal_double, state.derived.triangle_tangent_double)


@wp.kernel
def accumulate_triangle_basis(state: ClothState, substep: int, level: int, iteration: int):
    accumulate_triangle_basis_element(state, wp.tid(), substep, level, iteration)


@wp.func
def orient_particle_from_triangles_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    seg0 = state.derived.v2t_csr_offsets[p]
    seg1 = state.derived.v2t_csr_offsets[p + 1]
    if seg0 < seg1 and kernels.team_frame_mask(state.team.enabled, state.team.valid,
            state.team.component_world_scale, state.particle.team[p]):
        kernels.do_v2t_owner(p, seg0, seg1, state.derived.v2t_csr_order,
                state.v2t.triangle, state.v2t.flip_normal, state.v2t.flip_tangent,
                state.derived.triangle_normal_double, state.derived.triangle_tangent_double,
                state.particle.rotations, state.particle.normal_adjustment_rotations)


@wp.kernel
def orient_particle_from_triangles(state: ClothState, substep: int, level: int, iteration: int):
    orient_particle_from_triangles_element(state, wp.tid(), substep, level, iteration)


@wp.func
def emit_particle_output_rotation_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    mt = state.particle.team[p]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            mt):
        kernels.do_output_particle(p, state.particle.rotations, state.particle.vertex_to_transform_rotations,
                state.particle.out_rotations)


@wp.kernel
def emit_particle_output_rotation(state: ClothState, substep: int, level: int, iteration: int):
    emit_particle_output_rotation_element(state, wp.tid(), substep, level, iteration)


@wp.func
def publish_bone_transform_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    p = thread_index
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            state.particle.team[p]):
        kernels.do_publish_bone_transform(p, state.particle.publish_transform,
                state.particle.bone_row, state.particle.publish_position,
                state.particle.aim_child, state.particle.aim_rest_reach,
                state.particle.positions, state.particle.rotations,
                state.particle.vertex_to_transform_rotations, state.transform.world,
                state.transform.solved)


@wp.kernel
def publish_bone_transform(state: ClothState, substep: int, level: int, iteration: int):
    publish_bone_transform_element(state, wp.tid(), substep, level, iteration)


@wp.func
def commit_collider_frame_pose_element(state: ClothState, thread_index: int, substep: int,
        level: int, iteration: int):
    ci = thread_index
    cm = state.collider.team[ci]
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            cm) and state.team.running[cm] != 0 and (state.collider.active[ci] != 0):
        kernels.do_collider_frame_post(ci, state.collider.frame_positions, state.collider.frame_rotations,
                state.collider.frame_tips, state.collider.old_frame_positions,
                state.collider.old_frame_rotations, state.collider.old_frame_tips)


@wp.kernel
def commit_collider_frame_pose(state: ClothState, substep: int, level: int, iteration: int):
    commit_collider_frame_pose_element(state, wp.tid(), substep, level, iteration)


@wp.func
def close_team_frame_element(state: ClothState, thread_index: int, substep: int, level: int,
        iteration: int):
    i = thread_index
    if kernels.team_frame_mask(state.team.enabled, state.team.valid, state.team.component_world_scale,
            i):
        run = state.team.running[i] != 0
        state.team.old_component_world_position[i, 0] = state.team.component_world_position[i,
                0]
        state.team.old_component_world_position[i, 1] = state.team.component_world_position[i,
                1]
        state.team.old_component_world_position[i, 2] = state.team.component_world_position[i,
                2]
        state.team.old_component_world_rotation[i, 0] = state.team.component_world_rotation[i,
                0]
        state.team.old_component_world_rotation[i, 1] = state.team.component_world_rotation[i,
                1]
        state.team.old_component_world_rotation[i, 2] = state.team.component_world_rotation[i,
                2]
        state.team.old_component_world_rotation[i, 3] = state.team.component_world_rotation[i,
                3]
        state.team.old_component_world_scale[i, 0] = state.team.component_world_scale[i, 0]
        state.team.old_component_world_scale[i, 1] = state.team.component_world_scale[i, 1]
        state.team.old_component_world_scale[i, 2] = state.team.component_world_scale[i, 2]
        state.team.old_component_world_reflected[i] = state.team.component_world_reflected[i]
        if run:
            state.team.old_frame_world_position[i, 0] = state.team.frame_world_position[i, 0]
            state.team.old_frame_world_position[i, 1] = state.team.frame_world_position[i, 1]
            state.team.old_frame_world_position[i, 2] = state.team.frame_world_position[i, 2]
            state.team.old_frame_world_rotation[i, 0] = state.team.frame_world_rotation[i, 0]
            state.team.old_frame_world_rotation[i, 1] = state.team.frame_world_rotation[i, 1]
            state.team.old_frame_world_rotation[i, 2] = state.team.frame_world_rotation[i, 2]
            state.team.old_frame_world_rotation[i, 3] = state.team.frame_world_rotation[i, 3]
            state.team.old_frame_world_scale[i, 0] = state.team.frame_world_scale[i, 0]
            state.team.old_frame_world_scale[i, 1] = state.team.frame_world_scale[i, 1]
            state.team.old_frame_world_scale[i, 2] = state.team.frame_world_scale[i, 2]
            state.team.old_frame_world_reflected[i] = state.team.component_world_reflected[i]
            state.team.skip_count[i] = 0
            state.team.force_mode[i] = 0
            state.team.impact_force[i, 0] = 0.0
            state.team.impact_force[i, 1] = 0.0
            state.team.impact_force[i, 2] = 0.0
        state.team.old_anchor_position[i, 0] = state.team.anchor_position[i, 0]
        state.team.old_anchor_position[i, 1] = state.team.anchor_position[i, 1]
        state.team.old_anchor_position[i, 2] = state.team.anchor_position[i, 2]
        state.team.old_anchor_rotation[i, 0] = state.team.anchor_rotation[i, 0]
        state.team.old_anchor_rotation[i, 1] = state.team.anchor_rotation[i, 1]
        state.team.old_anchor_rotation[i, 2] = state.team.anchor_rotation[i, 2]
        state.team.old_anchor_rotation[i, 3] = state.team.anchor_rotation[i, 3]
        (qix, qiy, qiz, qiw) = dmath.quat_inverse(state.team.anchor_rotation[i, 0],
                state.team.anchor_rotation[i, 1], state.team.anchor_rotation[i, 2],
                state.team.anchor_rotation[i, 3])
        dpx = state.team.component_world_position[i, 0] - state.team.anchor_position[i, 0]
        dpy = state.team.component_world_position[i, 1] - state.team.anchor_position[i, 1]
        dpz = state.team.component_world_position[i, 2] - state.team.anchor_position[i, 2]
        (alx, aly, alz) = dmath.quat_rotate(qix, qiy, qiz, qiw, dpx, dpy, dpz)
        state.team.anchor_component_local_position[i, 0] = alx
        state.team.anchor_component_local_position[i, 1] = aly
        state.team.anchor_component_local_position[i, 2] = alz
        state.team.reset_pending[i] = 0
        state.team.time_reset_pending[i] = 0
        state.team.running[i] = 0
        state.team.keep_teleport_pending[i] = 0
        state.team.inertia_shift[i] = 0
        state.team.negative_scale_teleport[i] = 0
        if state.team.time[i] > 7200.0:
            state.team.time[i] = state.team.time[i] - 3600.0
            state.team.old_time[i] = state.team.old_time[i] - 3600.0
            state.team.now_update_time[i] = state.team.now_update_time[i] - 3600.0
            state.team.old_update_time[i] = state.team.old_update_time[i] - 3600.0
            state.team.frame_update_time[i] = state.team.frame_update_time[i] - 3600.0
            state.team.frame_old_time[i] = state.team.frame_old_time[i] - 3600.0


@wp.kernel
def close_team_frame(state: ClothState, substep: int, level: int, iteration: int):
    close_team_frame_element(state, wp.tid(), substep, level, iteration)
