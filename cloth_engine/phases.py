import warp as wp

from ..cloth_kernel import defs as _defs
from . import dmath
from . import kernels
from . import policy
from .kernels import EPSILON
from .kernels import FORCE_VELOCITY_ADD
from .kernels import FORCE_VELOCITY_ADD_WITHOUT_DEPTH
from .kernels import FORCE_VELOCITY_CHANGE
from .kernels import FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH
from .kernels import RAD2DEG
from .kernels import SCAL_FRAME_DT
from .kernels import SCAL_MAX_SIM
from .kernels import SCAL_N_ZONES
from .kernels import SCAL_POWER1
from .kernels import SCAL_POWER2
from .kernels import SCAL_SIM_DT
from .kernels import SCAL_TIME_SCALE
from .kernels import TELEPORT_RESET
from .kernels import WIND_MAX_TIME
from .kernels import WIND_MIN_SPEED
from .kernels import WIND_ZONE_MIN_MAIN
from .kernels import WIND_ZONE_SLOTS
from .kernels import ZONE_BOX
from .kernels import ZONE_SPHERE_DIR
from .kernels import ZONE_SPHERE_RADIAL

wp.set_module_options(policy.MODULE_OPTIONS)

ZONE_RESULT_INDICES = wp.types.vector(length=_defs.WIND_ZONE_RESULT_SLOTS, dtype=wp.int32)
ZONE_RESULT_VALUES = wp.types.vector(length=_defs.WIND_ZONE_RESULT_SLOTS, dtype=wp.float32)
WIND_SLOT_INDICES = wp.types.vector(length=_defs.WIND_ZONE_SLOTS, dtype=wp.int32)
WIND_SLOT_VALUES = wp.types.vector(length=_defs.WIND_ZONE_SLOTS, dtype=wp.float32)


@wp.kernel
def phase_00_resolve_top(sc_sync: wp.array2d(dtype=float),
                         t_anchor_inertia: wp.array(dtype=float),
                         t_component_world_position: wp.array2d(dtype=float),
                         t_component_world_rotation: wp.array2d(dtype=float),
                         t_enabled: wp.array(dtype=int),
                         t_frame_old: wp.array(dtype=float),
                         t_frame_update: wp.array(dtype=float),
                         t_movement_inertia_smoothing: wp.array(dtype=float),
                         t_movement_speed_limit: wp.array(dtype=float),
                         t_now_update: wp.array(dtype=float),
                         t_old_time: wp.array(dtype=float),
                         t_old_update: wp.array(dtype=float),
                         t_rotation_speed_limit: wp.array(dtype=float),
                         t_sync_target: wp.array(dtype=int),
                         t_sync_top: wp.array(dtype=int),
                         t_teleport_distance: wp.array(dtype=float),
                         t_teleport_mode: wp.array(dtype=int),
                         t_teleport_rotation: wp.array(dtype=float),
                         t_time: wp.array(dtype=float),
                         t_time_scale: wp.array(dtype=float),
                         t_valid: wp.array(dtype=int),
                         t_world_inertia: wp.array(dtype=float)):
    i = wp.tid()
    if t_enabled[i] != 0:
        target = t_sync_target[i]
        if target <= 0 or t_valid[target] == 0 or t_enabled[target] == 0:
            t_sync_top[i] = 0
        else:
            top = target
            for _hop in range(8):
                upper = t_sync_target[top]
                if upper <= 0 or upper == i or t_valid[upper] == 0 or t_enabled[upper] == 0:
                    break
                top = upper
            t_sync_top[i] = top


@wp.kernel
def phase_00_snapshot(sc_sync: wp.array2d(dtype=float),
                      t_anchor_inertia: wp.array(dtype=float),
                      t_component_world_position: wp.array2d(dtype=float),
                      t_component_world_rotation: wp.array2d(dtype=float),
                      t_enabled: wp.array(dtype=int),
                      t_frame_old: wp.array(dtype=float),
                      t_frame_update: wp.array(dtype=float),
                      t_movement_inertia_smoothing: wp.array(dtype=float),
                      t_movement_speed_limit: wp.array(dtype=float),
                      t_now_update: wp.array(dtype=float),
                      t_old_time: wp.array(dtype=float),
                      t_old_update: wp.array(dtype=float),
                      t_rotation_speed_limit: wp.array(dtype=float),
                      t_sync_target: wp.array(dtype=int),
                      t_sync_top: wp.array(dtype=int),
                      t_teleport_distance: wp.array(dtype=float),
                      t_teleport_mode: wp.array(dtype=int),
                      t_teleport_rotation: wp.array(dtype=float),
                      t_time: wp.array(dtype=float),
                      t_time_scale: wp.array(dtype=float),
                      t_valid: wp.array(dtype=int),
                      t_world_inertia: wp.array(dtype=float)):
    i = wp.tid()
    if t_enabled[i] != 0 and t_sync_top[i] > 0:
        top = t_sync_top[i]
        sc_sync[i, 0] = t_time[top]
        sc_sync[i, 1] = t_old_time[top]
        sc_sync[i, 2] = t_now_update[top]
        sc_sync[i, 3] = t_old_update[top]
        sc_sync[i, 4] = t_frame_update[top]
        sc_sync[i, 5] = t_frame_old[top]
        sc_sync[i, 6] = t_time_scale[top]
        sc_sync[i, 7] = t_anchor_inertia[top]
        sc_sync[i, 8] = t_world_inertia[top]
        sc_sync[i, 9] = t_movement_inertia_smoothing[top]
        sc_sync[i, 10] = t_movement_speed_limit[top]
        sc_sync[i, 11] = t_rotation_speed_limit[top]
        sc_sync[i, 12] = float(t_teleport_mode[top])
        sc_sync[i, 13] = t_teleport_distance[top]
        sc_sync[i, 14] = t_teleport_rotation[top]
        sc_sync[i, 15] = t_component_world_position[top, 0]
        sc_sync[i, 16] = t_component_world_position[top, 1]
        sc_sync[i, 17] = t_component_world_position[top, 2]
        sc_sync[i, 18] = t_component_world_rotation[top, 0]
        sc_sync[i, 19] = t_component_world_rotation[top, 1]
        sc_sync[i, 20] = t_component_world_rotation[top, 2]
        sc_sync[i, 21] = t_component_world_rotation[top, 3]


@wp.kernel
def phase_00_apply(sc_sync: wp.array2d(dtype=float),
                   t_anchor_inertia: wp.array(dtype=float),
                   t_component_world_position: wp.array2d(dtype=float),
                   t_component_world_rotation: wp.array2d(dtype=float),
                   t_enabled: wp.array(dtype=int),
                   t_frame_old: wp.array(dtype=float),
                   t_frame_update: wp.array(dtype=float),
                   t_movement_inertia_smoothing: wp.array(dtype=float),
                   t_movement_speed_limit: wp.array(dtype=float),
                   t_now_update: wp.array(dtype=float),
                   t_old_time: wp.array(dtype=float),
                   t_old_update: wp.array(dtype=float),
                   t_rotation_speed_limit: wp.array(dtype=float),
                   t_sync_target: wp.array(dtype=int),
                   t_sync_top: wp.array(dtype=int),
                   t_teleport_distance: wp.array(dtype=float),
                   t_teleport_mode: wp.array(dtype=int),
                   t_teleport_rotation: wp.array(dtype=float),
                   t_time: wp.array(dtype=float),
                   t_time_scale: wp.array(dtype=float),
                   t_valid: wp.array(dtype=int),
                   t_world_inertia: wp.array(dtype=float)):
    i = wp.tid()
    if t_enabled[i] != 0 and t_sync_top[i] > 0:
        t_time[i] = sc_sync[i, 0]
        t_old_time[i] = sc_sync[i, 1]
        t_now_update[i] = sc_sync[i, 2]
        t_old_update[i] = sc_sync[i, 3]
        t_frame_update[i] = sc_sync[i, 4]
        t_frame_old[i] = sc_sync[i, 5]
        t_time_scale[i] = sc_sync[i, 6]
        t_anchor_inertia[i] = sc_sync[i, 7]
        t_world_inertia[i] = sc_sync[i, 8]
        t_movement_inertia_smoothing[i] = sc_sync[i, 9]
        t_movement_speed_limit[i] = sc_sync[i, 10]
        t_rotation_speed_limit[i] = sc_sync[i, 11]
        t_teleport_mode[i] = int(sc_sync[i, 12])
        t_teleport_distance[i] = sc_sync[i, 13]
        t_teleport_rotation[i] = sc_sync[i, 14]
        t_component_world_position[i, 0] = sc_sync[i, 15]
        t_component_world_position[i, 1] = sc_sync[i, 16]
        t_component_world_position[i, 2] = sc_sync[i, 17]
        t_component_world_rotation[i, 0] = sc_sync[i, 18]
        t_component_world_rotation[i, 1] = sc_sync[i, 19]
        t_component_world_rotation[i, 2] = sc_sync[i, 20]
        t_component_world_rotation[i, 3] = sc_sync[i, 21]


@wp.kernel
def phase_01(scal_f: wp.array(dtype=float),
             scal_i: wp.array(dtype=int),
             t_cws: wp.array2d(dtype=float),
             t_enabled: wp.array(dtype=int),
             t_frame_dt: wp.array(dtype=float),
             t_frame_old: wp.array(dtype=float),
             t_frame_update: wp.array(dtype=float),
             t_now_time_scale: wp.array(dtype=float),
             t_now_update: wp.array(dtype=float),
             t_old_time: wp.array(dtype=float),
             t_old_update: wp.array(dtype=float),
             t_running: wp.array(dtype=int),
             t_skip_count: wp.array(dtype=int),
             t_time: wp.array(dtype=float),
             t_time_reset: wp.array(dtype=int),
             t_time_scale: wp.array(dtype=float),
             t_update_count: wp.array(dtype=int),
             t_valid: wp.array(dtype=int)):
    i = wp.tid()
    fdt = scal_f[SCAL_FRAME_DT]
    global_time_scale = scal_f[SCAL_TIME_SCALE]
    max_sim_count = scal_i[SCAL_MAX_SIM]
    sim_dt = scal_f[SCAL_SIM_DT]
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, i):
        kernels.do_advance(i, fdt, sim_dt, max_sim_count, global_time_scale,
                           t_time_reset, t_time, t_old_time, t_now_update, t_old_update,
                           t_frame_update, t_frame_old, t_frame_dt, t_time_scale,
                           t_now_time_scale, t_update_count, t_skip_count, t_running)


@wp.kernel
def phase_02(p_local_normals: wp.array2d(dtype=float),
             p_local_positions: wp.array2d(dtype=float),
             p_local_tangents: wp.array2d(dtype=float),
             p_positions: wp.array2d(dtype=float),
             p_rotations: wp.array2d(dtype=float),
             p_skin_indices: wp.array2d(dtype=int),
             p_skin_weights: wp.array2d(dtype=float),
             p_team: wp.array(dtype=int),
             t_cws: wp.array2d(dtype=float),
             t_enabled: wp.array(dtype=int),
             t_valid: wp.array(dtype=int),
             x_bind: wp.array3d(dtype=float),
             x_world: wp.array3d(dtype=float)):
    p = wp.tid()
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, p_team[p]):
        kernels.do_base_pose(p, p_team, p_local_positions, p_local_normals, p_local_tangents,
                             p_skin_indices, p_skin_weights, p_positions, p_rotations,
                             x_world, x_bind)


@wp.kernel
def phase_03(csr_center_fixed_offsets: wp.array(dtype=int),
             csr_center_fixed_order: wp.array(dtype=int),
             p_positions: wp.array2d(dtype=float),
             p_rotations: wp.array2d(dtype=float),
             p_vertex_bind_pose_rotations: wp.array2d(dtype=float),
             scal_f: wp.array(dtype=float),
             st_center_fixed_particle: wp.array(dtype=int),
             t_anchor_component_local_position: wp.array2d(dtype=float),
             t_anchor_inertia: wp.array(dtype=float),
             t_anchor_position: wp.array2d(dtype=float),
             t_anchor_rotation: wp.array2d(dtype=float),
             t_blend_weight: wp.array(dtype=float),
             t_component_world_position: wp.array2d(dtype=float),
             t_component_world_rotation: wp.array2d(dtype=float),
             t_culling_invisible: wp.array(dtype=int),
             t_cws: wp.array2d(dtype=float),
             t_enabled: wp.array(dtype=int),
             t_frame_component_shift_rotation: wp.array2d(dtype=float),
             t_frame_component_shift_vector: wp.array2d(dtype=float),
             t_frame_dt: wp.array(dtype=float),
             t_frame_moving_direction: wp.array2d(dtype=float),
             t_frame_moving_speed: wp.array(dtype=float),
             t_frame_world_position: wp.array2d(dtype=float),
             t_frame_world_rotation: wp.array2d(dtype=float),
             t_frame_world_scale: wp.array2d(dtype=float),
             t_had_anchor: wp.array(dtype=int),
             t_has_anchor: wp.array(dtype=int),
             t_inertia_shift: wp.array(dtype=int),
             t_init_scale: wp.array2d(dtype=float),
             t_is_negative_scale: wp.array(dtype=int),
             t_keep_teleport_pending: wp.array(dtype=int),
             t_movement_inertia_smoothing: wp.array(dtype=float),
             t_movement_speed_limit: wp.array(dtype=float),
             t_negative_scale_change: wp.array2d(dtype=float),
             t_negative_scale_direction: wp.array2d(dtype=float),
             t_negative_scale_matrix: wp.array(dtype=wp.mat44d),
             t_negative_scale_quaternion: wp.array2d(dtype=float),
             t_negative_scale_sign: wp.array(dtype=float),
             t_negative_scale_teleport: wp.array(dtype=int),
             t_negative_scale_triangle_sign: wp.array2d(dtype=float),
             t_now_time_scale: wp.array(dtype=float),
             t_now_world_position: wp.array2d(dtype=float),
             t_now_world_rotation: wp.array2d(dtype=float),
             t_old_anchor_position: wp.array2d(dtype=float),
             t_old_anchor_rotation: wp.array2d(dtype=float),
             t_old_component_world_position: wp.array2d(dtype=float),
             t_old_component_world_rotation: wp.array2d(dtype=float),
             t_old_component_world_scale: wp.array2d(dtype=float),
             t_old_frame_world_position: wp.array2d(dtype=float),
             t_old_frame_world_rotation: wp.array2d(dtype=float),
             t_old_frame_world_scale: wp.array2d(dtype=float),
             t_old_world_position: wp.array2d(dtype=float),
             t_old_world_rotation: wp.array2d(dtype=float),
             t_reset_pending: wp.array(dtype=int),
             t_rotation_speed_limit: wp.array(dtype=float),
             t_running: wp.array(dtype=int),
             t_skip_count: wp.array(dtype=int),
             t_smoothing_velocity: wp.array2d(dtype=float),
             t_stablization_time: wp.array(dtype=float),
             t_teleport_distance: wp.array(dtype=float),
             t_teleport_mode: wp.array(dtype=int),
             t_teleport_rotation: wp.array(dtype=float),
             t_time_reset: wp.array(dtype=int),
             t_valid: wp.array(dtype=int),
             t_velocity_weight: wp.array(dtype=float),
             t_world_inertia: wp.array(dtype=float)):
    i = wp.tid()
    sim_dt = scal_f[SCAL_SIM_DT]
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, i):
        cpx = t_component_world_position[i, 0]
        cpy = t_component_world_position[i, 1]
        cpz = t_component_world_position[i, 2]
        crx = t_component_world_rotation[i, 0]
        cry = t_component_world_rotation[i, 1]
        crz = t_component_world_rotation[i, 2]
        crw = t_component_world_rotation[i, 3]
        csx = t_cws[i, 0]
        csy = t_cws[i, 1]
        csz = t_cws[i, 2]

        init_scale_len = wp.float64(dmath.length3(
            t_init_scale[i, 0], t_init_scale[i, 1], t_init_scale[i, 2]))
        if init_scale_len < wp.float64(1e-30):
            init_scale_len = wp.float64(1e-30)
        csr_ratio = wp.float64(dmath.length3(csx, csy, csz)) / init_scale_len

        old_dx = t_negative_scale_direction[i, 0]
        old_dy = t_negative_scale_direction[i, 1]
        old_dz = t_negative_scale_direction[i, 2]
        sxv = 1.0 if csx == 0.0 else csx
        syv = 1.0 if csy == 0.0 else csy
        szv = 1.0 if csz == 0.0 else csz
        dir_x = dmath.fsign(sxv)
        dir_y = dmath.fsign(syv)
        dir_z = dmath.fsign(szv)
        t_negative_scale_direction[i, 0] = dir_x
        t_negative_scale_direction[i, 1] = dir_y
        t_negative_scale_direction[i, 2] = dir_z
        t_negative_scale_change[i, 0] = old_dx * dir_x
        t_negative_scale_change[i, 1] = old_dy * dir_y
        t_negative_scale_change[i, 2] = old_dz * dir_z
        is_negative = (csx < 0.0) or (csy < 0.0) or (csz < 0.0)
        t_is_negative_scale[i] = 1 if is_negative else 0
        t_negative_scale_sign[i] = -1.0 if is_negative else 1.0
        if is_negative:
            t_negative_scale_quaternion[i, 0] = dmath.negate(dir_x)
            t_negative_scale_quaternion[i, 1] = dmath.negate(dir_y)
            t_negative_scale_quaternion[i, 2] = dmath.negate(dir_z)
            t_negative_scale_quaternion[i, 3] = 1.0
            ts0 = -1.0 if (csx < 0.0 or csz < 0.0) else 1.0
            ts1 = -1.0 if (csx < 0.0) else 1.0
            t_negative_scale_triangle_sign[i, 0] = ts0
            t_negative_scale_triangle_sign[i, 1] = ts1
        else:
            t_negative_scale_quaternion[i, 0] = 1.0
            t_negative_scale_quaternion[i, 1] = 1.0
            t_negative_scale_quaternion[i, 2] = 1.0
            t_negative_scale_quaternion[i, 3] = 1.0
            t_negative_scale_triangle_sign[i, 0] = 1.0
            t_negative_scale_triangle_sign[i, 1] = 1.0
        teleport = (old_dx != dir_x) or (old_dy != dir_y) or (old_dz != dir_z)
        t_negative_scale_teleport[i] = 1 if teleport else 0

        if teleport:
            component_matrix = dmath.trs_build_f64(cpx, cpy, cpz, crx, cry, crz, crw,
                                                   csx, csy, csz)
            ocpx = t_old_component_world_position[i, 0]
            ocpy = t_old_component_world_position[i, 1]
            ocpz = t_old_component_world_position[i, 2]
            ocrx = t_old_component_world_rotation[i, 0]
            ocry = t_old_component_world_rotation[i, 1]
            ocrz = t_old_component_world_rotation[i, 2]
            ocrw = t_old_component_world_rotation[i, 3]
            ocsx = t_old_component_world_scale[i, 0]
            ocsy = t_old_component_world_scale[i, 1]
            ocsz = t_old_component_world_scale[i, 2]
            old_component_inverse = dmath.trs_inverse_f64(ocpx, ocpy, ocpz, ocrx, ocry, ocrz,
                                                          ocrw, ocsx, ocsy, ocsz)
            component_delta = dmath.mat4_mul_f64(component_matrix, old_component_inverse)
            nx, ny, nz = dmath.transform_point(component_delta, ocpx, ocpy, ocpz)
            t_old_component_world_position[i, 0] = nx
            t_old_component_world_position[i, 1] = ny
            t_old_component_world_position[i, 2] = nz
            t_old_component_world_scale[i, 0] = csx
            t_old_component_world_scale[i, 1] = csy
            t_old_component_world_scale[i, 2] = csz
            oax = t_old_anchor_position[i, 0]
            oay = t_old_anchor_position[i, 1]
            oaz = t_old_anchor_position[i, 2]
            tax, tay, taz = dmath.transform_point(component_delta, oax, oay, oaz)
            t_old_anchor_position[i, 0] = tax
            t_old_anchor_position[i, 1] = tay
            t_old_anchor_position[i, 2] = taz
            tsvx, tsvy, tsvz = dmath.transform_vector(
                component_delta, t_smoothing_velocity[i, 0], t_smoothing_velocity[i, 1],
                t_smoothing_velocity[i, 2])
            t_smoothing_velocity[i, 0] = tsvx
            t_smoothing_velocity[i, 1] = tsvy
            t_smoothing_velocity[i, 2] = tsvz

        ocp_x = t_old_component_world_position[i, 0]
        ocp_y = t_old_component_world_position[i, 1]
        ocp_z = t_old_component_world_position[i, 2]
        ocr_x = t_old_component_world_rotation[i, 0]
        ocr_y = t_old_component_world_rotation[i, 1]
        ocr_z = t_old_component_world_rotation[i, 2]
        ocr_w = t_old_component_world_rotation[i, 3]

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
        seg0 = csr_center_fixed_offsets[i]
        seg1 = csr_center_fixed_offsets[i + 1]
        for e in range(seg0, seg1):
            fp = st_center_fixed_particle[csr_center_fixed_order[e]]
            rx = p_rotations[fp, 0]
            ry = p_rotations[fp, 1]
            rz = p_rotations[fp, 2]
            rw = p_rotations[fp, 3]
            if is_negative:
                nnx, nny, nnz = dmath.quat_to_normal(rx, ry, rz, rw)
                ttx, tty, ttz = dmath.quat_to_tangent(rx, ry, rz, rw)
                rx, ry, rz, rw = dmath.to_rotation(
                    dmath.negate(nnx), dmath.negate(nny), dmath.negate(nnz),
                    dmath.negate(ttx), dmath.negate(tty), dmath.negate(ttz))
            rx, ry, rz, rw = dmath.quat_mul(
                rx, ry, rz, rw, p_vertex_bind_pose_rotations[fp, 0],
                p_vertex_bind_pose_rotations[fp, 1], p_vertex_bind_pose_rotations[fp, 2],
                p_vertex_bind_pose_rotations[fp, 3])
            norx, nory, norz = dmath.quat_to_normal(rx, ry, rz, rw)
            tanx, tany, tanz = dmath.quat_to_tangent(rx, ry, rz, rw)
            nflip = -1.0 if (dir_x < 0.0 or dir_z < 0.0) else 1.0
            tflip = -1.0 if (dir_x < 0.0 or dir_y < 0.0) else 1.0
            nor_sx += wp.float64(norx * nflip)
            nor_sy += wp.float64(nory * nflip)
            nor_sz += wp.float64(norz * nflip)
            tan_sx += wp.float64(tanx * tflip)
            tan_sy += wp.float64(tany * tflip)
            tan_sz += wp.float64(tanz * tflip)
            pos_sx += wp.float64(p_positions[fp, 0])
            pos_sy += wp.float64(p_positions[fp, 1])
            pos_sz += wp.float64(p_positions[fp, 2])
            fcount += 1
        if fcount > 0:
            nl = wp.sqrt(nor_sx * nor_sx + nor_sy * nor_sy + nor_sz * nor_sz)
            tl = wp.sqrt(tan_sx * tan_sx + tan_sy * tan_sy + tan_sz * tan_sz)
            if nl > wp.float64(1e-30) and tl > wp.float64(1e-30):
                cwpx = wp.float32(pos_sx / wp.float64(fcount))
                cwpy = wp.float32(pos_sy / wp.float64(fcount))
                cwpz = wp.float32(pos_sz / wp.float64(fcount))
                cwrx, cwry, cwrz, cwrw = dmath.to_rotation(
                    wp.float32(nor_sx / nl), wp.float32(nor_sy / nl), wp.float32(nor_sz / nl),
                    wp.float32(tan_sx / tl), wp.float32(tan_sy / tl), wp.float32(tan_sz / tl))

        if teleport:
            frame_matrix = dmath.trs_build_f64(cwpx, cwpy, cwpz, cwrx, cwry, cwrz, cwrw,
                                               csx, csy, csz)
            old_frame_inverse = dmath.trs_inverse_f64(
                t_old_frame_world_position[i, 0], t_old_frame_world_position[i, 1],
                t_old_frame_world_position[i, 2], t_old_frame_world_rotation[i, 0],
                t_old_frame_world_rotation[i, 1], t_old_frame_world_rotation[i, 2],
                t_old_frame_world_rotation[i, 3], t_old_frame_world_scale[i, 0],
                t_old_frame_world_scale[i, 1], t_old_frame_world_scale[i, 2])
            t_negative_scale_matrix[i] = dmath.mat4_mul_f64(frame_matrix, old_frame_inverse)

        adv_x = float(0.0)
        adv_y = float(0.0)
        adv_z = float(0.0)
        adr_x = float(0.0)
        adr_y = float(0.0)
        adr_z = float(0.0)
        adr_w = float(1.0)
        has_anc = t_has_anchor[i] != 0
        had_anc = t_had_anchor[i] != 0
        anchor_reset = t_reset_pending[i] != 0
        if has_anc and (not had_anc):
            anchor_reset = wp.bool(True)
        if had_anc and (not has_anc):
            anchor_reset = wp.bool(True)
        t_had_anchor[i] = 1 if has_anc else 0
        if anchor_reset:
            iqx, iqy, iqz, iqw = dmath.quat_inverse(
                t_anchor_rotation[i, 0], t_anchor_rotation[i, 1],
                t_anchor_rotation[i, 2], t_anchor_rotation[i, 3])
            alx, aly, alz = dmath.quat_rotate(
                iqx, iqy, iqz, iqw, cpx - t_anchor_position[i, 0],
                cpy - t_anchor_position[i, 1], cpz - t_anchor_position[i, 2])
            t_old_anchor_position[i, 0] = t_anchor_position[i, 0]
            t_old_anchor_position[i, 1] = t_anchor_position[i, 1]
            t_old_anchor_position[i, 2] = t_anchor_position[i, 2]
            t_old_anchor_rotation[i, 0] = t_anchor_rotation[i, 0]
            t_old_anchor_rotation[i, 1] = t_anchor_rotation[i, 1]
            t_old_anchor_rotation[i, 2] = t_anchor_rotation[i, 2]
            t_old_anchor_rotation[i, 3] = t_anchor_rotation[i, 3]
            t_anchor_component_local_position[i, 0] = alx
            t_anchor_component_local_position[i, 1] = aly
            t_anchor_component_local_position[i, 2] = alz
        if has_anc:
            rlx, rly, rlz = dmath.quat_rotate(
                t_anchor_rotation[i, 0], t_anchor_rotation[i, 1], t_anchor_rotation[i, 2],
                t_anchor_rotation[i, 3], t_anchor_component_local_position[i, 0],
                t_anchor_component_local_position[i, 1],
                t_anchor_component_local_position[i, 2])
            dvx = (rlx + t_anchor_position[i, 0]) - ocp_x
            dvy = (rly + t_anchor_position[i, 1]) - ocp_y
            dvz = (rlz + t_anchor_position[i, 2]) - ocp_z
            ioax, ioay, ioaz, ioaw = dmath.quat_inverse(
                t_old_anchor_rotation[i, 0], t_old_anchor_rotation[i, 1],
                t_old_anchor_rotation[i, 2], t_old_anchor_rotation[i, 3])
            drx, dry, drz, drw = dmath.quat_mul(
                t_anchor_rotation[i, 0], t_anchor_rotation[i, 1], t_anchor_rotation[i, 2],
                t_anchor_rotation[i, 3], ioax, ioay, ioaz, ioaw)
            a_ratio = 1.0 - t_anchor_inertia[i]
            adv_x = dvx * a_ratio
            adv_y = dvy * a_ratio
            adv_z = dvz * a_ratio
            adr_x, adr_y, adr_z, adr_w = dmath.quat_slerp(
                0.0, 0.0, 0.0, 1.0, drx, dry, drz, drw, a_ratio)
            ocp_x = ocp_x + adv_x
            ocp_y = ocp_y + adv_y
            ocp_z = ocp_z + adv_z
            ocr_x, ocr_y, ocr_z, ocr_w = dmath.quat_mul(
                adr_x, adr_y, adr_z, adr_w, ocr_x, ocr_y, ocr_z, ocr_w)
            t_inertia_shift[i] = 1

        fdvx = cpx - ocp_x
        fdvy = cpy - ocp_y
        fdvz = cpz - ocp_z
        fda = dmath.quat_angle(ocr_x, ocr_y, ocr_z, ocr_w, crx, cry, crz, crw)
        if (t_teleport_mode[i] != 0) and (t_reset_pending[i] == 0):
            far = wp.float64(dmath.length3(fdvx, fdvy, fdvz)) >= \
                wp.float64(t_teleport_distance[i]) * csr_ratio
            spun = (fda * RAD2DEG) >= t_teleport_rotation[i]
            if far or spun:
                if t_teleport_mode[i] == TELEPORT_RESET:
                    t_reset_pending[i] = 1
                else:
                    t_keep_teleport_pending[i] = 1

        reset = t_reset_pending[i] != 0
        keep = t_keep_teleport_pending[i] != 0

        sdv_x = float(0.0)
        sdv_y = float(0.0)
        sdv_z = float(0.0)
        if (t_movement_inertia_smoothing[i] >= 1.0e-6) and (not (keep or reset)):
            running = t_running[i] != 0
            fdt_i = t_frame_dt[i]
            if fdt_i > 0.0:
                dvx = fdvx / fdt_i
                dvy = fdvy / fdt_i
                dvz = fdvz / fdt_i
            else:
                dvx = float(0.0)
                dvy = float(0.0)
                dvz = float(0.0)
            limit = t_movement_speed_limit[i] * wp.float32(csr_ratio)
            mlim = limit if limit > 0.0 else float(0.0)
            cvx, cvy, cvz = dmath.clamp_vector(dvx, dvy, dvz, mlim)
            if limit >= 0.0:
                dvx = cvx
                dvy = cvy
                dvz = cvz
            mis = t_movement_inertia_smoothing[i]
            om = 1.0 - mis
            avg = dmath.saturate(om * om * om * 0.99 + 0.01)
            svx = t_smoothing_velocity[i, 0]
            svy = t_smoothing_velocity[i, 1]
            svz = t_smoothing_velocity[i, 2]
            smx = dmath.lerp(svx, dvx, avg)
            smy = dmath.lerp(svy, dvy, avg)
            smz = dmath.lerp(svz, dvz, avg)
            if running:
                t_smoothing_velocity[i, 0] = smx
                t_smoothing_velocity[i, 1] = smy
                t_smoothing_velocity[i, 2] = smz
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
            t_inertia_shift[i] = 1

        t_frame_world_position[i, 0] = cwpx
        t_frame_world_position[i, 1] = cwpy
        t_frame_world_position[i, 2] = cwpz
        t_frame_world_rotation[i, 0] = cwrx
        t_frame_world_rotation[i, 1] = cwry
        t_frame_world_rotation[i, 2] = cwrz
        t_frame_world_rotation[i, 3] = cwrw
        t_frame_world_scale[i, 0] = csx
        t_frame_world_scale[i, 1] = csy
        t_frame_world_scale[i, 2] = csz
        if reset:
            t_old_component_world_position[i, 0] = cpx
            t_old_component_world_position[i, 1] = cpy
            t_old_component_world_position[i, 2] = cpz
            t_old_component_world_rotation[i, 0] = crx
            t_old_component_world_rotation[i, 1] = cry
            t_old_component_world_rotation[i, 2] = crz
            t_old_component_world_rotation[i, 3] = crw
            t_old_component_world_scale[i, 0] = csx
            t_old_component_world_scale[i, 1] = csy
            t_old_component_world_scale[i, 2] = csz
            ocp_x = cpx
            ocp_y = cpy
            ocp_z = cpz
            ocr_x = crx
            ocr_y = cry
            ocr_z = crz
            ocr_w = crw
        if reset or (teleport and (not reset)):
            t_old_frame_world_position[i, 0] = cwpx
            t_old_frame_world_position[i, 1] = cwpy
            t_old_frame_world_position[i, 2] = cwpz
            t_old_frame_world_rotation[i, 0] = cwrx
            t_old_frame_world_rotation[i, 1] = cwry
            t_old_frame_world_rotation[i, 2] = cwrz
            t_old_frame_world_rotation[i, 3] = cwrw
            t_old_frame_world_scale[i, 0] = csx
            t_old_frame_world_scale[i, 1] = csy
            t_old_frame_world_scale[i, 2] = csz
            t_now_world_position[i, 0] = cwpx
            t_now_world_position[i, 1] = cwpy
            t_now_world_position[i, 2] = cwpz
            t_now_world_rotation[i, 0] = cwrx
            t_now_world_rotation[i, 1] = cwry
            t_now_world_rotation[i, 2] = cwrz
            t_now_world_rotation[i, 3] = cwrw
            t_old_world_position[i, 0] = cwpx
            t_old_world_position[i, 1] = cwpy
            t_old_world_position[i, 2] = cwpz
            t_old_world_rotation[i, 0] = cwrx
            t_old_world_rotation[i, 1] = cwry
            t_old_world_rotation[i, 2] = cwrz
            t_old_world_rotation[i, 3] = cwrw

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
            t_smoothing_velocity[i, 0] = 0.0
            t_smoothing_velocity[i, 1] = 0.0
            t_smoothing_velocity[i, 2] = 0.0
            sdv_x = float(0.0)
            sdv_y = float(0.0)
            sdv_z = float(0.0)

        if not reset:
            shv_x = cpx - ocp_x
            shv_y = cpy - ocp_y
            shv_z = cpz - ocp_z
            iox, ioy, ioz, iow = dmath.quat_inverse(ocr_x, ocr_y, ocr_z, ocr_w)
            shr_x, shr_y, shr_z, shr_w = dmath.quat_mul(crx, cry, crz, crw, iox, ioy, ioz, iow)
            msr = float(0.0)
            rsr = float(0.0)
            keep_now = keep or (t_culling_invisible[i] != 0)
            if keep_now:
                movement_shift = float(1.0)
            else:
                movement_shift = 1.0 - t_world_inertia[i]
            rotation_shift = movement_shift
            if movement_shift > EPSILON:
                t_inertia_shift[i] = 1
                msr = movement_shift
                rsr = rotation_shift
                wpx = dmath.lerp(wpx, cpx, movement_shift)
                wpy = dmath.lerp(wpy, cpy, movement_shift)
                wpz = dmath.lerp(wpz, cpz, movement_shift)
                wrx, wry, wrz, wrw = dmath.quat_slerp(
                    wrx, wry, wrz, wrw, crx, cry, crz, crw, rotation_shift)
            movement_limit = wp.float64(t_movement_speed_limit[i]) * csr_ratio
            rotation_limit = t_rotation_speed_limit[i]
            dvx = cpx - wpx
            dvy = cpy - wpy
            dvz = cpz - wpz
            dang = dmath.quat_angle(wrx, wry, wrz, wrw, crx, cry, crz, crw)
            fdt_l = t_frame_dt[i]
            if fdt_l > 0.0:
                frame_speed = dmath.length3(dvx, dvy, dvz) / fdt_l
                frame_rot_speed = (dang * RAD2DEG) / fdt_l
            else:
                frame_speed = float(0.0)
                frame_rot_speed = float(0.0)
            over_move = (wp.float64(frame_speed) > movement_limit) and \
                (t_movement_speed_limit[i] >= 0.0)
            if over_move:
                t_inertia_shift[i] = 1
                denom_fs = frame_speed if frame_speed > 0.0 else float(1.0)
                mlr = dmath.saturate((frame_speed - wp.float32(movement_limit)) / denom_fs)
            else:
                mlr = float(0.0)
            msr = msr + (1.0 - msr) * mlr
            if over_move:
                wpx = dmath.lerp(wpx, cpx, mlr)
                wpy = dmath.lerp(wpy, cpy, mlr)
                wpz = dmath.lerp(wpz, cpz, mlr)
            over_rot = (frame_rot_speed > rotation_limit) and (rotation_limit >= 0.0)
            if over_rot:
                t_inertia_shift[i] = 1
                denom_frs = frame_rot_speed if frame_rot_speed > 0.0 else float(1.0)
                rlr = dmath.saturate((frame_rot_speed - rotation_limit) / denom_frs)
            else:
                rlr = float(0.0)
            rsr = rsr + (1.0 - rsr) * rlr
            if over_rot:
                wrx, wry, wrz, wrw = dmath.quat_slerp(
                    wrx, wry, wrz, wrw, crx, cry, crz, crw, rlr)
            osr = wp.float64(0.0)
            skip = t_skip_count[i]
            scaled_dt = fdt_l * t_now_time_scale[i]
            if (skip > 0) and (scaled_dt > 0.0):
                sr = wp.float64(skip) * wp.float64(sim_dt) / wp.float64(scaled_dt)
                if sr < wp.float64(0.0):
                    sr = wp.float64(0.0)
                elif sr > wp.float64(1.0):
                    sr = wp.float64(1.0)
                osr = osr + (wp.float64(1.0) - osr) * sr
            vw = t_velocity_weight[i]
            if vw < 1.0:
                osr = osr + (wp.float64(1.0) - osr) * (wp.float64(1.0) - wp.float64(vw))
            nts = t_now_time_scale[i]
            if nts < 1.0:
                osr = osr + (wp.float64(1.0) - osr) * (wp.float64(1.0) - wp.float64(nts))
            msr_final = wp.float64(msr)
            rsr_final = wp.float64(rsr)
            if osr > wp.float64(0.0):
                t_inertia_shift[i] = 1
                msr_final = msr_final + (wp.float64(1.0) - msr_final) * osr
                osr_f32 = wp.float32(osr)
                wpx = dmath.lerp(wpx, cpx, osr_f32)
                wpy = dmath.lerp(wpy, cpy, osr_f32)
                wpz = dmath.lerp(wpz, cpz, osr_f32)
                rsr_final = rsr_final + (wp.float64(1.0) - rsr_final) * osr
                wrx, wry, wrz, wrw = dmath.quat_slerp(
                    wrx, wry, wrz, wrw, crx, cry, crz, crw, osr_f32)
            if t_inertia_shift[i] != 0:
                vecx = wp.float64(shv_x) * msr_final + wp.float64(adv_x) + wp.float64(sdv_x)
                vecy = wp.float64(shv_y) * msr_final + wp.float64(adv_y) + wp.float64(sdv_y)
                vecz = wp.float64(shv_z) * msr_final + wp.float64(adv_z) + wp.float64(sdv_z)
                rqx, rqy, rqz, rqw = dmath.quat_slerp(
                    0.0, 0.0, 0.0, 1.0,
                    shr_x, shr_y, shr_z, shr_w, wp.float32(rsr_final))
                rqx, rqy, rqz, rqw = dmath.quat_mul(adr_x, adr_y, adr_z, adr_w,
                                                    rqx, rqy, rqz, rqw)
                t_frame_component_shift_vector[i, 0] = wp.float32(vecx)
                t_frame_component_shift_vector[i, 1] = wp.float32(vecy)
                t_frame_component_shift_vector[i, 2] = wp.float32(vecz)
                t_frame_component_shift_rotation[i, 0] = rqx
                t_frame_component_shift_rotation[i, 1] = rqy
                t_frame_component_shift_rotation[i, 2] = rqz
                t_frame_component_shift_rotation[i, 3] = rqw
                oc_x = t_old_component_world_position[i, 0]
                oc_y = t_old_component_world_position[i, 1]
                oc_z = t_old_component_world_position[i, 2]
                rlx1, rly1, rlz1 = dmath.quat_rotate(
                    rqx, rqy, rqz, rqw, t_old_frame_world_position[i, 0] - oc_x,
                    t_old_frame_world_position[i, 1] - oc_y,
                    t_old_frame_world_position[i, 2] - oc_z)
                t_old_frame_world_position[i, 0] = wp.float32(wp.float64(rlx1 + oc_x) + vecx)
                t_old_frame_world_position[i, 1] = wp.float32(wp.float64(rly1 + oc_y) + vecy)
                t_old_frame_world_position[i, 2] = wp.float32(wp.float64(rlz1 + oc_z) + vecz)
                oqx, oqy, oqz, oqw = dmath.quat_mul(
                    rqx, rqy, rqz, rqw, t_old_frame_world_rotation[i, 0],
                    t_old_frame_world_rotation[i, 1], t_old_frame_world_rotation[i, 2],
                    t_old_frame_world_rotation[i, 3])
                t_old_frame_world_rotation[i, 0] = oqx
                t_old_frame_world_rotation[i, 1] = oqy
                t_old_frame_world_rotation[i, 2] = oqz
                t_old_frame_world_rotation[i, 3] = oqw
                rlx2, rly2, rlz2 = dmath.quat_rotate(
                    rqx, rqy, rqz, rqw, t_now_world_position[i, 0] - oc_x,
                    t_now_world_position[i, 1] - oc_y, t_now_world_position[i, 2] - oc_z)
                t_now_world_position[i, 0] = wp.float32(wp.float64(rlx2 + oc_x) + vecx)
                t_now_world_position[i, 1] = wp.float32(wp.float64(rly2 + oc_y) + vecy)
                t_now_world_position[i, 2] = wp.float32(wp.float64(rlz2 + oc_z) + vecz)
                nqx, nqy, nqz, nqw = dmath.quat_mul(
                    rqx, rqy, rqz, rqw, t_now_world_rotation[i, 0],
                    t_now_world_rotation[i, 1], t_now_world_rotation[i, 2],
                    t_now_world_rotation[i, 3])
                t_now_world_rotation[i, 0] = nqx
                t_now_world_rotation[i, 1] = nqy
                t_now_world_rotation[i, 2] = nqz
                t_now_world_rotation[i, 3] = nqw
            else:
                t_frame_component_shift_vector[i, 0] = shv_x
                t_frame_component_shift_vector[i, 1] = shv_y
                t_frame_component_shift_vector[i, 2] = shv_z
                t_frame_component_shift_rotation[i, 0] = shr_x
                t_frame_component_shift_rotation[i, 1] = shr_y
                t_frame_component_shift_rotation[i, 2] = shr_z
                t_frame_component_shift_rotation[i, 3] = shr_w
        if reset:
            t_frame_component_shift_vector[i, 0] = 0.0
            t_frame_component_shift_vector[i, 1] = 0.0
            t_frame_component_shift_vector[i, 2] = 0.0
            t_frame_component_shift_rotation[i, 0] = 0.0
            t_frame_component_shift_rotation[i, 1] = 0.0
            t_frame_component_shift_rotation[i, 2] = 0.0
            t_frame_component_shift_rotation[i, 3] = 1.0

        mvx = cpx - wpx
        mvy = cpy - wpy
        mvz = cpz - wpz
        mlen = dmath.length3(mvx, mvy, mvz)
        fdt_m = t_frame_dt[i]
        if fdt_m > 0.0:
            speed = mlen / fdt_m
        else:
            speed = float(0.0)
        nts_m = t_now_time_scale[i]
        if nts_m > 1.0e-6:
            speed = speed * (1.0 / nts_m)
        else:
            speed = float(0.0)
        t_frame_moving_speed[i] = speed
        if mlen > 1.0e-6:
            t_frame_moving_direction[i, 0] = mvx / mlen
            t_frame_moving_direction[i, 1] = mvy / mlen
            t_frame_moving_direction[i, 2] = mvz / mlen
        else:
            t_frame_moving_direction[i, 0] = 0.0
            t_frame_moving_direction[i, 1] = 0.0
            t_frame_moving_direction[i, 2] = 0.0

        if (t_reset_pending[i] != 0) or (t_time_reset[i] != 0):
            if t_stablization_time[i] > 1.0e-6:
                wgt = float(0.0)
            else:
                wgt = float(1.0)
            t_velocity_weight[i] = wgt
            t_blend_weight[i] = wgt


@wp.kernel
def phase_03b(scal_i: wp.array(dtype=int),
              t_cws: wp.array2d(dtype=float),
              t_enabled: wp.array(dtype=int),
              t_frame_world_position: wp.array2d(dtype=float),
              t_valid: wp.array(dtype=int),
              t_wind_count: wp.array(dtype=int),
              t_wind_direction: wp.array3d(dtype=float),
              t_wind_dirq: wp.array3d(dtype=float),
              t_wind_influence: wp.array(dtype=float),
              t_wind_main: wp.array2d(dtype=float),
              t_wind_time: wp.array2d(dtype=float),
              t_wind_zone_id: wp.array2d(dtype=int),
              t_wind_zone_turbulence: wp.array2d(dtype=float),
              z_attenuation_lut: wp.array2d(dtype=float),
              z_is_addition: wp.array(dtype=int),
              z_main: wp.array(dtype=float),
              z_mode: wp.array(dtype=int),
              z_size: wp.array2d(dtype=float),
              z_turbulence: wp.array(dtype=float),
              z_world_direction: wp.array2d(dtype=float),
              z_world_position: wp.array2d(dtype=float),
              z_world_to_local: wp.array(dtype=wp.mat44d),
              z_zone_id: wp.array(dtype=int),
              z_zone_volume: wp.array(dtype=float)):
    i = wp.tid()
    n_zones = scal_i[SCAL_N_ZONES]
    res_zone_id = ZONE_RESULT_INDICES()
    res_time = ZONE_RESULT_VALUES()
    res_main = ZONE_RESULT_VALUES()
    res_dx = ZONE_RESULT_VALUES()
    res_dy = ZONE_RESULT_VALUES()
    res_dz = ZONE_RESULT_VALUES()
    res_turb = ZONE_RESULT_VALUES()
    old_zid = WIND_SLOT_INDICES()
    old_wt = WIND_SLOT_VALUES()
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, i):
        old_count = t_wind_count[i]
        for oc in range(WIND_ZONE_SLOTS):
            if oc < old_count:
                old_zid[oc] = t_wind_zone_id[i, oc]
                old_wt[oc] = t_wind_time[i, oc]
        count = int(0)
        if n_zones > 0 and t_wind_influence[i] > EPSILON:
            cx64 = wp.float64(t_frame_world_position[i, 0])
            cy64 = wp.float64(t_frame_world_position[i, 1])
            cz64 = wp.float64(t_frame_world_position[i, 2])
            min_volume = float(wp.inf)
            addition_count = int(0)
            latest_valid = wp.bool(False)
            latest_id = int(0)
            for zi in range(n_zones):
                is_add = z_is_addition[zi] != 0
                if is_add and addition_count >= 3:
                    continue
                mode = z_mode[zi]
                zvol = z_zone_volume[zi]
                wm = z_world_to_local[zi]
                lxx = wm[0, 0] * cx64 + wm[0, 1] * cy64 + wm[0, 2] * cz64 + wm[0, 3]
                lyy = wm[1, 0] * cx64 + wm[1, 1] * cy64 + wm[1, 2] * cz64 + wm[1, 3]
                lzz = wm[2, 0] * cx64 + wm[2, 1] * cy64 + wm[2, 2] * cz64 + wm[2, 3]
                llen = wp.sqrt(lxx * lxx + lyy * lyy + lzz * lzz)
                skip_zone = wp.bool(False)
                if mode == ZONE_BOX:
                    if wp.abs(lxx) * wp.float64(2.0) > wp.float64(z_size[zi, 0]) or \
                            wp.abs(lyy) * wp.float64(2.0) > wp.float64(z_size[zi, 1]) or \
                            wp.abs(lzz) * wp.float64(2.0) > wp.float64(z_size[zi, 2]):
                        skip_zone = wp.bool(True)
                elif mode == ZONE_SPHERE_DIR or mode == ZONE_SPHERE_RADIAL:
                    if llen > wp.float64(z_size[zi, 0]):
                        skip_zone = wp.bool(True)
                if skip_zone:
                    continue
                if (not is_add) and (zvol > min_volume):
                    continue
                dirx = z_world_direction[zi, 0]
                diry = z_world_direction[zi, 1]
                dirz = z_world_direction[zi, 2]
                zmain = z_main[zi]
                if mode == ZONE_SPHERE_RADIAL:
                    if llen <= wp.float64(1e-6):
                        continue
                    vx64 = cx64 - wp.float64(z_world_position[zi, 0])
                    vy64 = cy64 - wp.float64(z_world_position[zi, 1])
                    vz64 = cz64 - wp.float64(z_world_position[zi, 2])
                    vlen = wp.sqrt(vx64 * vx64 + vy64 * vy64 + vz64 * vz64)
                    dirx = wp.float32(vx64 / vlen)
                    diry = wp.float32(vy64 / vlen)
                    dirz = wp.float32(vz64 / vlen)
                    depth = llen / wp.float64(z_size[zi, 0])
                    if depth < wp.float64(0.0):
                        depth = wp.float64(0.0)
                    elif depth > wp.float64(1.0):
                        depth = wp.float64(1.0)
                    zmain = zmain * dmath.evaluate_team_lut_clamp01(
                        z_attenuation_lut, zi, wp.float32(depth))
                zid = z_zone_id[zi]
                t_prev = dmath.negate(WIND_MAX_TIME)
                for oi in range(WIND_ZONE_SLOTS):
                    if oi < old_count and old_zid[oi] == zid:
                        t_prev = old_wt[oi]
                zturb = z_turbulence[zi]
                registrable = zmain > WIND_ZONE_MIN_MAIN
                if is_add:
                    if registrable:
                        res_zone_id[count] = zid
                        res_time[count] = t_prev
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
                                res_time[w] = res_time[r]
                                res_main[w] = res_main[r]
                                res_dx[w] = res_dx[r]
                                res_dy[w] = res_dy[r]
                                res_dz[w] = res_dz[r]
                                res_turb[w] = res_turb[r]
                                w += 1
                        count = w
                    if registrable:
                        res_zone_id[count] = zid
                        res_time[count] = t_prev
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
        t_wind_count[i] = final
        for s in range(final):
            t_wind_zone_id[i, s] = res_zone_id[s]
            t_wind_time[i, s] = res_time[s]
            t_wind_main[i, s] = res_main[s]
            t_wind_direction[i, s, 0] = res_dx[s]
            t_wind_direction[i, s, 1] = res_dy[s]
            t_wind_direction[i, s, 2] = res_dz[s]
            t_wind_zone_turbulence[i, s] = res_turb[s]
            dqx, dqy, dqz, dqw = dmath.axis_quaternion(res_dx[s], res_dy[s], res_dz[s])
            t_wind_dirq[i, s, 0] = dqx
            t_wind_dirq[i, s, 1] = dqy
            t_wind_dirq[i, s, 2] = dqz
            t_wind_dirq[i, s, 3] = dqw


@wp.kernel
def phase_04(p_base_positions: wp.array2d(dtype=float),
             p_base_rotations: wp.array2d(dtype=float),
             p_collision_normals: wp.array2d(dtype=float),
             p_display_positions: wp.array2d(dtype=float),
             p_friction: wp.array(dtype=float),
             p_next_positions: wp.array2d(dtype=float),
             p_old_anim_positions: wp.array2d(dtype=float),
             p_old_anim_rotations: wp.array2d(dtype=float),
             p_old_positions: wp.array2d(dtype=float),
             p_old_rotations: wp.array2d(dtype=float),
             p_positions: wp.array2d(dtype=float),
             p_real_velocities: wp.array2d(dtype=float),
             p_rotations: wp.array2d(dtype=float),
             p_static_friction: wp.array(dtype=float),
             p_team: wp.array(dtype=int),
             p_velocities: wp.array2d(dtype=float),
             p_velocity_positions: wp.array2d(dtype=float),
             t_cws: wp.array2d(dtype=float),
             t_enabled: wp.array(dtype=int),
             t_frame_component_shift_rotation: wp.array2d(dtype=float),
             t_frame_component_shift_vector: wp.array2d(dtype=float),
             t_inertia_shift: wp.array(dtype=int),
             t_negative_scale_matrix: wp.array(dtype=wp.mat44d),
             t_negative_scale_teleport: wp.array(dtype=int),
             t_old_component_world_position: wp.array2d(dtype=float),
             t_reset_pending: wp.array(dtype=int),
             t_valid: wp.array(dtype=int)):
    p = wp.tid()
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, p_team[p]):
        kernels.do_particles_frame_pre(p, p_team, p_positions, p_rotations, p_next_positions,
                                       p_old_positions, p_old_rotations, p_base_positions,
                                       p_base_rotations, p_old_anim_positions,
                                       p_old_anim_rotations, p_velocity_positions,
                                       p_display_positions, p_velocities,
                                       p_real_velocities, p_friction, p_static_friction,
                                       p_collision_normals, t_reset_pending,
                                       t_negative_scale_teleport, t_negative_scale_matrix,
                                       t_inertia_shift, t_frame_component_shift_vector,
                                       t_frame_component_shift_rotation,
                                       t_old_component_world_position)


@wp.kernel
def phase_05(c_active: wp.array(dtype=int),
             c_enabled: wp.array(dtype=int),
             c_enabled_prev: wp.array(dtype=int),
             c_frame_pos: wp.array2d(dtype=float),
             c_frame_radius: wp.array2d(dtype=float),
             c_frame_rot: wp.array2d(dtype=float),
             c_frame_tip: wp.array2d(dtype=float),
             c_input_positions: wp.array2d(dtype=float),
             c_input_radii: wp.array2d(dtype=float),
             c_input_rotations: wp.array2d(dtype=float),
             c_input_tips: wp.array2d(dtype=float),
             c_now_pos: wp.array2d(dtype=float),
             c_now_rot: wp.array2d(dtype=float),
             c_now_tip: wp.array2d(dtype=float),
             c_old_frame_pos: wp.array2d(dtype=float),
             c_old_frame_rot: wp.array2d(dtype=float),
             c_old_frame_tip: wp.array2d(dtype=float),
             c_old_pos: wp.array2d(dtype=float),
             c_old_rot: wp.array2d(dtype=float),
             c_old_tip: wp.array2d(dtype=float),
             c_team: wp.array(dtype=int),
             t_cws: wp.array2d(dtype=float),
             t_enabled: wp.array(dtype=int),
             t_frame_component_shift_rotation: wp.array2d(dtype=float),
             t_frame_component_shift_vector: wp.array2d(dtype=float),
             t_inertia_shift: wp.array(dtype=int),
             t_negative_scale_change: wp.array2d(dtype=float),
             t_negative_scale_matrix: wp.array(dtype=wp.mat44d),
             t_negative_scale_teleport: wp.array(dtype=int),
             t_old_component_world_position: wp.array2d(dtype=float),
             t_reset_pending: wp.array(dtype=int),
             t_valid: wp.array(dtype=int)):
    ci = wp.tid()
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, c_team[ci]):
        kernels.do_collider_frame_pre(ci, c_team, c_enabled, c_enabled_prev, c_active,
                                      c_input_positions, c_input_rotations, c_input_tips,
                                      c_input_radii, c_frame_pos, c_frame_rot, c_frame_tip,
                                      c_frame_radius, c_old_frame_pos, c_old_frame_rot,
                                      c_old_frame_tip, c_now_pos, c_now_rot, c_now_tip,
                                      c_old_pos, c_old_rot, c_old_tip,
                                      t_reset_pending, t_negative_scale_teleport,
                                      t_negative_scale_matrix, t_negative_scale_change,
                                      t_inertia_shift, t_frame_component_shift_vector,
                                      t_frame_component_shift_rotation,
                                      t_old_component_world_position)


@wp.kernel
def phase_16(k: int,
             p_next_positions: wp.array2d(dtype=float),
             p_step_basic_positions: wp.array2d(dtype=float),
             p_team: wp.array(dtype=int),
             p_velocity_positions: wp.array2d(dtype=float),
             p_vertex_root: wp.array(dtype=int),
             st_tether_particle: wp.array(dtype=int),
             st_tether_team: wp.array(dtype=int),
             t_cws: wp.array2d(dtype=float),
             t_enabled: wp.array(dtype=int),
             t_tether_compression: wp.array(dtype=float),
             t_update_count: wp.array(dtype=int),
             t_valid: wp.array(dtype=int)):
    e = wp.tid()
    tm = st_tether_team[e]
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, tm) and t_update_count[tm] > k:
        kernels.do_tether(e, st_tether_particle, p_team, p_next_positions,
                          p_velocity_positions, p_step_basic_positions, p_vertex_root,
                          t_tether_compression)


@wp.kernel
def phase_17(k: int,
             csr_distance_offsets: wp.array(dtype=int),
             csr_distance_order: wp.array(dtype=int),
             p_attr_move: wp.array(dtype=int),
             p_base_positions: wp.array2d(dtype=float),
             p_depth: wp.array(dtype=float),
             p_friction: wp.array(dtype=float),
             p_next_positions: wp.array2d(dtype=float),
             p_team: wp.array(dtype=int),
             sc_dcorr: wp.array2d(dtype=float),
             scal_f: wp.array(dtype=float),
             st_distance_rest: wp.array(dtype=float),
             st_distance_target: wp.array(dtype=int),
             t_animation_pose_ratio: wp.array(dtype=float),
             t_cws: wp.array2d(dtype=float),
             t_distance_lut: wp.array2d(dtype=float),
             t_enabled: wp.array(dtype=int),
             t_init_scale: wp.array2d(dtype=float),
             t_is_spring: wp.array(dtype=int),
             t_scale_ratio: wp.array(dtype=float),
             t_update_count: wp.array(dtype=int),
             t_valid: wp.array(dtype=int)):
    p = wp.tid()
    power1 = scal_f[SCAL_POWER1]
    mt = p_team[p]
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > k:
        kernels.do_distance_gather(p, p_team, p_next_positions, p_base_positions, p_depth,
                                   p_friction, p_attr_move, t_is_spring, t_animation_pose_ratio,
                                   t_init_scale, t_scale_ratio, t_distance_lut, power1,
                                   csr_distance_offsets, csr_distance_order,
                                   st_distance_target, st_distance_rest, sc_dcorr)


@wp.kernel
def phase_10(k: int,
             scal_f: wp.array(dtype=float),
             t_angular_velocity: wp.array(dtype=float),
             t_blend_weight: wp.array(dtype=float),
             t_blend_weight_param: wp.array(dtype=float),
             t_cws: wp.array2d(dtype=float),
             t_distance_weight: wp.array(dtype=float),
             t_enabled: wp.array(dtype=int),
             t_frame_interpolation: wp.array(dtype=float),
             t_frame_moving_direction: wp.array2d(dtype=float),
             t_frame_moving_speed: wp.array(dtype=float),
             t_frame_old: wp.array(dtype=float),
             t_frame_world_position: wp.array2d(dtype=float),
             t_frame_world_rotation: wp.array2d(dtype=float),
             t_frame_world_scale: wp.array2d(dtype=float),
             t_gravity: wp.array(dtype=float),
             t_gravity_direction: wp.array2d(dtype=float),
             t_gravity_dot: wp.array(dtype=float),
             t_gravity_falloff: wp.array(dtype=float),
             t_gravity_ratio: wp.array(dtype=float),
             t_inertia_rotation: wp.array2d(dtype=float),
             t_inertia_vector: wp.array2d(dtype=float),
             t_init_local_gravity_direction: wp.array2d(dtype=float),
             t_init_scale: wp.array2d(dtype=float),
             t_local_inertia: wp.array(dtype=float),
             t_local_movement_speed_limit: wp.array(dtype=float),
             t_local_rotation_speed_limit: wp.array(dtype=float),
             t_moving_wind_direction: wp.array2d(dtype=float),
             t_moving_wind_dirq: wp.array2d(dtype=float),
             t_moving_wind_main: wp.array(dtype=float),
             t_moving_wind_time: wp.array(dtype=float),
             t_negative_scale_direction: wp.array2d(dtype=float),
             t_now_update: wp.array(dtype=float),
             t_now_world_position: wp.array2d(dtype=float),
             t_now_world_rotation: wp.array2d(dtype=float),
             t_old_frame_world_position: wp.array2d(dtype=float),
             t_old_frame_world_rotation: wp.array2d(dtype=float),
             t_old_frame_world_scale: wp.array2d(dtype=float),
             t_old_world_position: wp.array2d(dtype=float),
             t_old_world_rotation: wp.array2d(dtype=float),
             t_rotation_axis: wp.array2d(dtype=float),
             t_scale_ratio: wp.array(dtype=float),
             t_stablization_time: wp.array(dtype=float),
             t_step_move_inertia_ratio: wp.array(dtype=float),
             t_step_rotation: wp.array2d(dtype=float),
             t_step_rotation_inertia_ratio: wp.array(dtype=float),
             t_step_vector: wp.array2d(dtype=float),
             t_time: wp.array(dtype=float),
             t_update_count: wp.array(dtype=int),
             t_valid: wp.array(dtype=int),
             t_velocity_weight: wp.array(dtype=float),
             t_wind_count: wp.array(dtype=int),
             t_wind_frequency: wp.array(dtype=float),
             t_wind_main: wp.array2d(dtype=float),
             t_wind_moving: wp.array(dtype=float),
             t_wind_time: wp.array2d(dtype=float)):
    i = wp.tid()
    sim_dt = scal_f[SCAL_SIM_DT]
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, i) and t_update_count[i] > k:
        kernels.do_step_update(i, sim_dt,
                               t_now_update, t_time, t_frame_old, t_frame_interpolation,
                               t_now_world_position, t_now_world_rotation,
                               t_old_world_position, t_old_world_rotation,
                               t_old_frame_world_position, t_old_frame_world_rotation,
                               t_old_frame_world_scale, t_frame_world_position,
                               t_frame_world_rotation, t_frame_world_scale,
                               t_step_vector, t_step_rotation,
                               t_step_move_inertia_ratio, t_step_rotation_inertia_ratio,
                               t_local_inertia, t_local_movement_speed_limit,
                               t_local_rotation_speed_limit,
                               t_inertia_vector, t_inertia_rotation,
                               t_angular_velocity, t_rotation_axis,
                               t_init_scale, t_scale_ratio,
                               t_gravity_direction, t_gravity_dot,
                               t_init_local_gravity_direction, t_negative_scale_direction,
                               t_gravity, t_gravity_falloff, t_gravity_ratio,
                               t_velocity_weight, t_stablization_time, t_blend_weight,
                               t_blend_weight_param, t_distance_weight,
                               t_wind_moving, t_frame_moving_speed, t_moving_wind_main,
                               t_frame_moving_direction, t_moving_wind_direction,
                               t_moving_wind_dirq,
                               t_wind_main, t_wind_frequency, t_wind_count, t_wind_time,
                               t_moving_wind_time)


@wp.kernel
def phase_11(k: int,
             c_active: wp.array(dtype=int),
             c_frame_pos: wp.array2d(dtype=float),
             c_frame_radius: wp.array2d(dtype=float),
             c_frame_rot: wp.array2d(dtype=float),
             c_frame_tip: wp.array2d(dtype=float),
             c_kind: wp.array(dtype=int),
             c_now_pos: wp.array2d(dtype=float),
             c_now_rot: wp.array2d(dtype=float),
             c_now_tip: wp.array2d(dtype=float),
             c_old_frame_pos: wp.array2d(dtype=float),
             c_old_frame_rot: wp.array2d(dtype=float),
             c_old_frame_tip: wp.array2d(dtype=float),
             c_old_pos: wp.array2d(dtype=float),
             c_old_rot: wp.array2d(dtype=float),
             c_old_tip: wp.array2d(dtype=float),
             c_team: wp.array(dtype=int),
             c_work_aabb_max: wp.array2d(dtype=float),
             c_work_aabb_min: wp.array2d(dtype=float),
             c_work_inv_old_rot: wp.array2d(dtype=float),
             c_work_next_pos: wp.array3d(dtype=float),
             c_work_old_pos: wp.array3d(dtype=float),
             c_work_radius: wp.array2d(dtype=float),
             c_work_rot: wp.array2d(dtype=float),
             t_cws: wp.array2d(dtype=float),
             t_enabled: wp.array(dtype=int),
             t_frame_interpolation: wp.array(dtype=float),
             t_step_move_inertia_ratio: wp.array(dtype=float),
             t_step_rotation_inertia_ratio: wp.array(dtype=float),
             t_update_count: wp.array(dtype=int),
             t_valid: wp.array(dtype=int)):
    ci = wp.tid()
    cm = c_team[ci]
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, cm) and t_update_count[cm] > k \
            and c_active[ci] != 0:
        kernels.do_collider_start_step(ci, c_team, c_kind,
                                       c_frame_pos, c_frame_rot, c_frame_tip, c_frame_radius,
                                       c_old_frame_pos, c_old_frame_rot, c_old_frame_tip,
                                       c_now_pos, c_now_rot, c_now_tip,
                                       c_old_pos, c_old_rot, c_old_tip,
                                       c_work_rot, c_work_inv_old_rot,
                                       c_work_radius, c_work_old_pos, c_work_next_pos,
                                       c_work_aabb_min, c_work_aabb_max,
                                       t_frame_interpolation, t_step_move_inertia_ratio,
                                       t_step_rotation_inertia_ratio)


@wp.kernel
def phase_12_animate(k: int,
                     p_base_positions: wp.array2d(dtype=float),
                     p_base_rotations: wp.array2d(dtype=float),
                     p_depth: wp.array(dtype=float),
                     p_friction: wp.array(dtype=float),
                     p_next_positions: wp.array2d(dtype=float),
                     p_old_anim_positions: wp.array2d(dtype=float),
                     p_old_anim_rotations: wp.array2d(dtype=float),
                     p_old_positions: wp.array2d(dtype=float),
                     p_positions: wp.array2d(dtype=float),
                     p_rotations: wp.array2d(dtype=float),
                     p_step_basic_positions: wp.array2d(dtype=float),
                     p_step_basic_rotations: wp.array2d(dtype=float),
                     p_team: wp.array(dtype=int),
                     p_velocities: wp.array2d(dtype=float),
                     p_velocity_positions: wp.array2d(dtype=float),
                     p_vertex_root_local: wp.array(dtype=int),
                     scal_f: wp.array(dtype=float),
                     st_move_particle: wp.array(dtype=int),
                     st_move_team: wp.array(dtype=int),
                     t_cws: wp.array2d(dtype=float),
                     t_damping_lut: wp.array2d(dtype=float),
                     t_depth_inertia: wp.array(dtype=float),
                     t_enabled: wp.array(dtype=int),
                     t_force_mode: wp.array(dtype=int),
                     t_frame_interpolation: wp.array(dtype=float),
                     t_gravity: wp.array(dtype=float),
                     t_gravity_direction: wp.array2d(dtype=float),
                     t_gravity_ratio: wp.array(dtype=float),
                     t_impact_force: wp.array2d(dtype=float),
                     t_inertia_rotation: wp.array2d(dtype=float),
                     t_inertia_vector: wp.array2d(dtype=float),
                     t_moving_wind_dirq: wp.array2d(dtype=float),
                     t_moving_wind_main: wp.array(dtype=float),
                     t_moving_wind_time: wp.array(dtype=float),
                     t_old_world_position: wp.array2d(dtype=float),
                     t_scale_ratio: wp.array(dtype=float),
                     t_step_rotation: wp.array2d(dtype=float),
                     t_step_vector: wp.array2d(dtype=float),
                     t_update_count: wp.array(dtype=int),
                     t_valid: wp.array(dtype=int),
                     t_velocity_weight: wp.array(dtype=float),
                     t_wind_blend: wp.array(dtype=float),
                     t_wind_count: wp.array(dtype=int),
                     t_wind_depth_weight: wp.array(dtype=float),
                     t_wind_dirq: wp.array3d(dtype=float),
                     t_wind_influence: wp.array(dtype=float),
                     t_wind_main: wp.array2d(dtype=float),
                     t_wind_moving: wp.array(dtype=float),
                     t_wind_seed: wp.array(dtype=int),
                     t_wind_synchronization: wp.array(dtype=float),
                     t_wind_time: wp.array2d(dtype=float),
                     t_wind_turbulence: wp.array(dtype=float),
                     t_wind_zone_turbulence: wp.array2d(dtype=float)):
    p = wp.tid()
    mt = p_team[p]
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > k:
        t = t_frame_interpolation[mt]
        bx = dmath.lerp(p_old_anim_positions[p, 0], p_positions[p, 0], t)
        by = dmath.lerp(p_old_anim_positions[p, 1], p_positions[p, 1], t)
        bz = dmath.lerp(p_old_anim_positions[p, 2], p_positions[p, 2], t)
        qx, qy, qz, qw = dmath.quat_slerp(
            p_old_anim_rotations[p, 0], p_old_anim_rotations[p, 1],
            p_old_anim_rotations[p, 2], p_old_anim_rotations[p, 3],
            p_rotations[p, 0], p_rotations[p, 1], p_rotations[p, 2], p_rotations[p, 3], t)
        p_base_positions[p, 0] = bx
        p_base_positions[p, 1] = by
        p_base_positions[p, 2] = bz
        p_step_basic_positions[p, 0] = bx
        p_step_basic_positions[p, 1] = by
        p_step_basic_positions[p, 2] = bz
        p_base_rotations[p, 0] = qx
        p_base_rotations[p, 1] = qy
        p_base_rotations[p, 2] = qz
        p_base_rotations[p, 3] = qw
        p_step_basic_rotations[p, 0] = qx
        p_step_basic_rotations[p, 1] = qy
        p_step_basic_rotations[p, 2] = qz
        p_step_basic_rotations[p, 3] = qw


@wp.kernel
def phase_12_force(k: int,
                   p_base_positions: wp.array2d(dtype=float),
                   p_base_rotations: wp.array2d(dtype=float),
                   p_depth: wp.array(dtype=float),
                   p_friction: wp.array(dtype=float),
                   p_next_positions: wp.array2d(dtype=float),
                   p_old_anim_positions: wp.array2d(dtype=float),
                   p_old_anim_rotations: wp.array2d(dtype=float),
                   p_old_positions: wp.array2d(dtype=float),
                   p_positions: wp.array2d(dtype=float),
                   p_rotations: wp.array2d(dtype=float),
                   p_step_basic_positions: wp.array2d(dtype=float),
                   p_step_basic_rotations: wp.array2d(dtype=float),
                   p_team: wp.array(dtype=int),
                   p_velocities: wp.array2d(dtype=float),
                   p_velocity_positions: wp.array2d(dtype=float),
                   p_vertex_root_local: wp.array(dtype=int),
                   scal_f: wp.array(dtype=float),
                   st_move_particle: wp.array(dtype=int),
                   st_move_team: wp.array(dtype=int),
                   t_cws: wp.array2d(dtype=float),
                   t_damping_lut: wp.array2d(dtype=float),
                   t_depth_inertia: wp.array(dtype=float),
                   t_enabled: wp.array(dtype=int),
                   t_force_mode: wp.array(dtype=int),
                   t_frame_interpolation: wp.array(dtype=float),
                   t_gravity: wp.array(dtype=float),
                   t_gravity_direction: wp.array2d(dtype=float),
                   t_gravity_ratio: wp.array(dtype=float),
                   t_impact_force: wp.array2d(dtype=float),
                   t_inertia_rotation: wp.array2d(dtype=float),
                   t_inertia_vector: wp.array2d(dtype=float),
                   t_moving_wind_dirq: wp.array2d(dtype=float),
                   t_moving_wind_main: wp.array(dtype=float),
                   t_moving_wind_time: wp.array(dtype=float),
                   t_old_world_position: wp.array2d(dtype=float),
                   t_scale_ratio: wp.array(dtype=float),
                   t_step_rotation: wp.array2d(dtype=float),
                   t_step_vector: wp.array2d(dtype=float),
                   t_update_count: wp.array(dtype=int),
                   t_valid: wp.array(dtype=int),
                   t_velocity_weight: wp.array(dtype=float),
                   t_wind_blend: wp.array(dtype=float),
                   t_wind_count: wp.array(dtype=int),
                   t_wind_depth_weight: wp.array(dtype=float),
                   t_wind_dirq: wp.array3d(dtype=float),
                   t_wind_influence: wp.array(dtype=float),
                   t_wind_main: wp.array2d(dtype=float),
                   t_wind_moving: wp.array(dtype=float),
                   t_wind_seed: wp.array(dtype=int),
                   t_wind_synchronization: wp.array(dtype=float),
                   t_wind_time: wp.array2d(dtype=float),
                   t_wind_turbulence: wp.array(dtype=float),
                   t_wind_zone_turbulence: wp.array2d(dtype=float)):
    e = wp.tid()
    power2 = scal_f[SCAL_POWER2]
    sim_dt = scal_f[SCAL_SIM_DT]
    mt = st_move_team[e]
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > k:
        pmi = st_move_particle[e]
        depth = p_depth[pmi]
        ox = p_old_positions[pmi, 0]
        oy = p_old_positions[pmi, 1]
        oz = p_old_positions[pmi, 2]

        inertia_depth = t_depth_inertia[mt] * (1.0 - depth * depth)
        ivx = dmath.lerp(t_inertia_vector[mt, 0], t_step_vector[mt, 0], inertia_depth)
        ivy = dmath.lerp(t_inertia_vector[mt, 1], t_step_vector[mt, 1], inertia_depth)
        ivz = dmath.lerp(t_inertia_vector[mt, 2], t_step_vector[mt, 2], inertia_depth)
        irx, iry, irz, irw = dmath.quat_slerp(
            t_inertia_rotation[mt, 0], t_inertia_rotation[mt, 1],
            t_inertia_rotation[mt, 2], t_inertia_rotation[mt, 3],
            t_step_rotation[mt, 0], t_step_rotation[mt, 1],
            t_step_rotation[mt, 2], t_step_rotation[mt, 3], inertia_depth)
        owx = t_old_world_position[mt, 0]
        owy = t_old_world_position[mt, 1]
        owz = t_old_world_position[mt, 2]
        lx = ox - owx
        ly = oy - owy
        lz = oz - owz
        rlx, rly, rlz = dmath.quat_rotate(irx, iry, irz, irw, lx, ly, lz)
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

        vx, vy, vz = dmath.quat_rotate(irx, iry, irz, irw,
                                       p_velocities[pmi, 0], p_velocities[pmi, 1],
                                       p_velocities[pmi, 2])
        vw = t_velocity_weight[mt]
        vx = vx * vw
        vy = vy * vw
        vz = vz * vw
        damping = dmath.evaluate_team_lut_clamp01(t_damping_lut, mt, depth)
        damp = dmath.saturate(1.0 - damping * power2)
        vx = vx * damp
        vy = vy * damp
        vz = vz * damp

        fm = t_force_mode[mt]
        change = (fm == FORCE_VELOCITY_CHANGE) or (fm == FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH)
        if change:
            vx = 0.0
            vy = 0.0
            vz = 0.0

        g = t_gravity[mt] * t_gravity_ratio[mt]
        fx = t_gravity_direction[mt, 0] * g
        fy = t_gravity_direction[mt, 1] * g
        fz = t_gravity_direction[mt, 2] * g
        mass = dmath.calc_mass(depth)
        with_depth = (fm == FORCE_VELOCITY_ADD) or (fm == FORCE_VELOCITY_CHANGE)
        without_depth = (fm == FORCE_VELOCITY_ADD_WITHOUT_DEPTH) \
            or (fm == FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH)
        if with_depth:
            fx = fx + t_impact_force[mt, 0] / mass
            fy = fy + t_impact_force[mt, 1] / mass
            fz = fz + t_impact_force[mt, 2] / mass
        if without_depth:
            fx = fx + t_impact_force[mt, 0]
            fy = fy + t_impact_force[mt, 1]
            fz = fz + t_impact_force[mt, 2]

        root = float(p_vertex_root_local[pmi])
        seed = float(t_wind_seed[mt])
        sync = t_wind_synchronization[mt]
        wind_position = (seed + 1.0) * 4.19230645 \
            + root * 0.0023963 * (1.0 - sync) * 100.0
        blend = t_wind_blend[mt]
        turbulence_param = t_wind_turbulence[mt]
        wfx = float(0.0)
        wfy = float(0.0)
        wfz = float(0.0)
        wc = t_wind_count[mt]
        for s in range(WIND_ZONE_SLOTS):
            if s < wc:
                cx, cy, cz = kernels.do_wind_blend(
                    t_wind_main[mt, s], t_wind_time[mt, s],
                    t_wind_dirq[mt, s, 0], t_wind_dirq[mt, s, 1],
                    t_wind_dirq[mt, s, 2], t_wind_dirq[mt, s, 3],
                    t_wind_zone_turbulence[mt, s], blend, turbulence_param, wind_position)
                wfx = wfx + cx
                wfy = wfy + cy
                wfz = wfz + cz
        moving_on = t_wind_moving[mt] > WIND_MIN_SPEED
        if moving_on:
            mcx, mcy, mcz = kernels.do_wind_blend(
                t_moving_wind_main[mt], t_moving_wind_time[mt],
                t_moving_wind_dirq[mt, 0], t_moving_wind_dirq[mt, 1],
                t_moving_wind_dirq[mt, 2], t_moving_wind_dirq[mt, 3],
                1.0, blend, turbulence_param, wind_position)
            wfx = wfx + mcx
            wfy = wfy + mcy
            wfz = wfz + mcz
        influence = t_wind_influence[mt] * (1.0 - p_friction[pmi])
        depth_scale = depth * depth
        influence = influence * dmath.lerp(1.0, depth_scale, t_wind_depth_weight[mt])
        fx = fx + wfx * influence
        fy = fy + wfy * influence
        fz = fz + wfz * influence

        sr = t_scale_ratio[mt]
        fx = fx * sr
        fy = fy * sr
        fz = fz * sr

        vx = vx + fx * sim_dt
        vy = vy + fy * sim_dt
        vz = vz + fz * sim_dt
        nextx = nextx + vx * sim_dt
        nexty = nexty + vy * sim_dt
        nextz = nextz + vz * sim_dt

        p_velocities[pmi, 0] = vx
        p_velocities[pmi, 1] = vy
        p_velocities[pmi, 2] = vz
        p_next_positions[pmi, 0] = nextx
        p_next_positions[pmi, 1] = nexty
        p_next_positions[pmi, 2] = nextz
        p_velocity_positions[pmi, 0] = velposx
        p_velocity_positions[pmi, 1] = velposy
        p_velocity_positions[pmi, 2] = velposz


@wp.kernel
def phase_13_fixed(k: int,
                   p_base_positions: wp.array2d(dtype=float),
                   p_base_rotations: wp.array2d(dtype=float),
                   p_next_positions: wp.array2d(dtype=float),
                   p_velocity_positions: wp.array2d(dtype=float),
                   st_fixed_particle: wp.array(dtype=int),
                   st_fixed_team: wp.array(dtype=int),
                   st_spring_particle: wp.array(dtype=int),
                   st_spring_team: wp.array(dtype=int),
                   t_cws: wp.array2d(dtype=float),
                   t_enabled: wp.array(dtype=int),
                   t_normal_axis_vector: wp.array2d(dtype=float),
                   t_scale_ratio: wp.array(dtype=float),
                   t_spring_limit_distance: wp.array(dtype=float),
                   t_spring_noise: wp.array(dtype=float),
                   t_spring_normal_limit_ratio: wp.array(dtype=float),
                   t_spring_power: wp.array(dtype=float),
                   t_time: wp.array(dtype=float),
                   t_update_count: wp.array(dtype=int),
                   t_valid: wp.array(dtype=int)):
    e = wp.tid()
    ft = st_fixed_team[e]
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, ft) and t_update_count[ft] > k:
        pfi = st_fixed_particle[e]
        p_next_positions[pfi, 0] = p_base_positions[pfi, 0]
        p_next_positions[pfi, 1] = p_base_positions[pfi, 1]
        p_next_positions[pfi, 2] = p_base_positions[pfi, 2]
        p_velocity_positions[pfi, 0] = p_base_positions[pfi, 0]
        p_velocity_positions[pfi, 1] = p_base_positions[pfi, 1]
        p_velocity_positions[pfi, 2] = p_base_positions[pfi, 2]


@wp.kernel
def phase_13_spring(k: int,
                    p_base_positions: wp.array2d(dtype=float),
                    p_base_rotations: wp.array2d(dtype=float),
                    p_next_positions: wp.array2d(dtype=float),
                    p_velocity_positions: wp.array2d(dtype=float),
                    st_fixed_particle: wp.array(dtype=int),
                    st_fixed_team: wp.array(dtype=int),
                    st_spring_particle: wp.array(dtype=int),
                    st_spring_team: wp.array(dtype=int),
                    t_cws: wp.array2d(dtype=float),
                    t_enabled: wp.array(dtype=int),
                    t_normal_axis_vector: wp.array2d(dtype=float),
                    t_scale_ratio: wp.array(dtype=float),
                    t_spring_limit_distance: wp.array(dtype=float),
                    t_spring_noise: wp.array(dtype=float),
                    t_spring_normal_limit_ratio: wp.array(dtype=float),
                    t_spring_power: wp.array(dtype=float),
                    t_time: wp.array(dtype=float),
                    t_update_count: wp.array(dtype=int),
                    t_valid: wp.array(dtype=int)):
    e = wp.tid()
    st = st_spring_team[e]
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, st) and t_update_count[st] > k \
            and t_spring_power[st] > 0.0:
        psi = st_spring_particle[e]
        bpx = p_base_positions[psi, 0]
        bpy = p_base_positions[psi, 1]
        bpz = p_base_positions[psi, 2]
        n0 = p_next_positions[psi, 0]
        n1 = p_next_positions[psi, 1]
        n2 = p_next_positions[psi, 2]
        vx = n0 - bpx
        vy = n1 - bpy
        vz = n2 - bpz
        dx, dy, dz = dmath.quat_rotate(
            p_base_rotations[psi, 0], p_base_rotations[psi, 1],
            p_base_rotations[psi, 2], p_base_rotations[psi, 3],
            t_normal_axis_vector[st, 0], t_normal_axis_vector[st, 1],
            t_normal_axis_vector[st, 2])
        limit = t_spring_limit_distance[st] * t_scale_ratio[st]
        clampable = limit > 1.0e-8
        l = dmath.length3(vx, vy, vz)
        over = clampable and (l > limit)
        if over and (l > 1.0e-30):
            scale = limit / l
            vx = vx * scale
            vy = vy * scale
            vz = vz * scale
        ratio = t_spring_normal_limit_ratio[st]
        elliptic = clampable and (ratio < 1.0)
        ylen = dmath.dot3(dx, dy, dz, vx, vy, vz)
        vpx = vx - dx * ylen
        vpy = vy - dy * ylen
        vpz = vz - dz * ylen
        xlen = dmath.length3(vpx, vpy, vpz)
        safe_limit = limit if limit > 1.0e-30 else 1.0
        tval = dmath.saturate(xlen / safe_limit)
        y = wp.cos(wp.asin(dmath.clamp1(tval))) * (limit * ratio)
        exceed = elliptic and (wp.abs(ylen) > y)
        if exceed:
            adjust = (wp.abs(ylen) - y) * dmath.fsign(ylen)
            vx = vx - adjust * dx
            vy = vy - adjust * dy
            vz = vz - adjust * dz
        if not clampable:
            vx = 0.0
            vy = 0.0
            vz = 0.0

        power = t_spring_power[st]
        noise_param = t_spring_noise[st]
        if noise_param > 0.0:
            noise_time = (t_time[st] + float(psi) * 49.6198) * 2.4512 \
                + (n0 + n1 + n2)
            noise = wp.sin(noise_time) * (noise_param * 0.6)
            power = power + power * noise
            if power < 0.0:
                power = 0.0
        vx = vx - vx * power
        vy = vy - vy * power
        vz = vz - vz * power
        p_next_positions[psi, 0] = bpx + vx
        p_next_positions[psi, 1] = bpy + vy
        p_next_positions[psi, 2] = bpz + vz


PHASE_TABLE = (
    ("phase_00", ((phase_00_resolve_top, "team"),
                  (phase_00_snapshot, "team"),
                  (phase_00_apply, "team"))),
    ("phase_01", ((phase_01, "team"),)),
    ("phase_02", ((phase_02, "particle"),)),
    ("phase_03", ((phase_03, "team"),)),
    ("phase_03b", ((phase_03b, "team"),)),
    ("phase_04", ((phase_04, "particle"),)),
    ("phase_05", ((phase_05, "collider"),)),
    ("phase_10", ((phase_10, "team"),)),
    ("phase_11", ((phase_11, "collider"),)),
    ("phase_12", ((phase_12_animate, "particle"),
                  (phase_12_force, "update_move"))),
    ("phase_13", ((phase_13_fixed, "update_fixed"),
                  (phase_13_spring, "spring"))),
    ("phase_16", ((phase_16, "tether"),)),
    ("phase_17", ((phase_17, "particle"),)),
)
