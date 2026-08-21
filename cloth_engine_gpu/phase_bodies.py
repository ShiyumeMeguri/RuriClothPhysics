def phase(grid_domain):
    def declare(function):
        function.phase_grid_domain = grid_domain
        return function
    return declare

@phase('single')
def phase_00(tid, bid, bdim, sc_sync, t_anchor_inertia, t_component_world_position,
             t_component_world_rotation, t_enabled, t_frame_old, t_frame_update,
             t_movement_inertia_smoothing, t_movement_speed_limit, t_now_update, t_old_time, t_old_update,
             t_rotation_speed_limit, t_sync_target, t_sync_top, t_teleport_distance, t_teleport_mode,
             t_teleport_rotation, t_time, t_time_scale, t_valid, t_world_inertia, num_teams):
    if bid == 0:
        i = tid
        while i < num_teams:
            if t_enabled[i] != 0:
                target = t_sync_target[i]
                if target <= 0 or t_valid[target] == 0 or t_enabled[target] == 0:
                    t_sync_top[i] = int32(0)
                else:
                    top = target
                    for _h in range(8):
                        upper = t_sync_target[top]
                        if upper <= 0 or upper == i or t_valid[upper] == 0 or t_enabled[upper] == 0:
                            break
                        top = upper
                    t_sync_top[i] = top
            i += bdim
    cuda.syncthreads()
    if bid == 0:
        i = tid
        while i < num_teams:
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
                sc_sync[i, 12] = float32(t_teleport_mode[top])
                sc_sync[i, 13] = t_teleport_distance[top]
                sc_sync[i, 14] = t_teleport_rotation[top]
                sc_sync[i, 15] = t_component_world_position[top, 0]
                sc_sync[i, 16] = t_component_world_position[top, 1]
                sc_sync[i, 17] = t_component_world_position[top, 2]
                sc_sync[i, 18] = t_component_world_rotation[top, 0]
                sc_sync[i, 19] = t_component_world_rotation[top, 1]
                sc_sync[i, 20] = t_component_world_rotation[top, 2]
                sc_sync[i, 21] = t_component_world_rotation[top, 3]
            i += bdim
    cuda.syncthreads()
    if bid == 0:
        i = tid
        while i < num_teams:
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
                t_teleport_mode[i] = int8(sc_sync[i, 12])
                t_teleport_distance[i] = sc_sync[i, 13]
                t_teleport_rotation[i] = sc_sync[i, 14]
                t_component_world_position[i, 0] = sc_sync[i, 15]
                t_component_world_position[i, 1] = sc_sync[i, 16]
                t_component_world_position[i, 2] = sc_sync[i, 17]
                t_component_world_rotation[i, 0] = sc_sync[i, 18]
                t_component_world_rotation[i, 1] = sc_sync[i, 19]
                t_component_world_rotation[i, 2] = sc_sync[i, 20]
                t_component_world_rotation[i, 3] = sc_sync[i, 21]
            i += bdim


@phase('teams')
def phase_01(tid, stride, fdt, global_time_scale, max_sim_count, sim_dt, t_cws, t_enabled, t_frame_dt,
             t_frame_old, t_frame_update, t_now_time_scale, t_now_update, t_old_time, t_old_update,
             t_running, t_skip_count, t_time, t_time_reset, t_time_scale, t_update_count, t_valid,
             num_teams):
    i = tid
    while i < num_teams:
        if team_frame_mask(t_enabled, t_valid, t_cws, i):
            do_advance(i, fdt, sim_dt, max_sim_count, global_time_scale,
                       t_time_reset, t_time, t_old_time, t_now_update, t_old_update,
                       t_frame_update, t_frame_old, t_frame_dt, t_time_scale,
                       t_now_time_scale, t_update_count, t_skip_count, t_running)
        i += stride


@phase('particles')
def phase_02(tid, stride, p_local_normals, p_local_positions, p_local_tangents, p_positions, p_rotations,
             p_skin_indices, p_skin_weights, p_team, t_cws, t_enabled, t_valid, x_bind, x_world,
             num_particles):
    p = tid
    while p < num_particles:
        if team_frame_mask(t_enabled, t_valid, t_cws, p_team[p]):
            do_base_pose(p, p_team, p_local_positions, p_local_normals, p_local_tangents,
                         p_skin_indices, p_skin_weights, p_positions, p_rotations,
                         x_world, x_bind)
        p += stride


@phase('teams')
def phase_03(tid, stride, sim_dt, csr_center_fixed_offsets, csr_center_fixed_order, p_positions,
             p_rotations, p_vertex_bind_pose_rotations, st_center_fixed_particle,
             t_anchor_component_local_position, t_anchor_inertia, t_anchor_position, t_anchor_rotation,
             t_blend_weight, t_component_world_position, t_component_world_rotation, t_culling_invisible,
             t_cws, t_enabled, t_frame_component_shift_rotation, t_frame_component_shift_vector,
             t_frame_dt, t_frame_moving_direction, t_frame_moving_speed, t_frame_world_position,
             t_frame_world_rotation, t_frame_world_scale, t_had_anchor, t_has_anchor, t_inertia_shift,
             t_init_scale, t_is_negative_scale, t_keep_teleport_pending, t_movement_inertia_smoothing,
             t_movement_speed_limit, t_negative_scale_change, t_negative_scale_direction,
             t_negative_scale_matrix, t_negative_scale_quaternion, t_negative_scale_sign,
             t_negative_scale_teleport, t_negative_scale_triangle_sign, t_now_time_scale,
             t_now_world_position, t_now_world_rotation, t_old_anchor_position, t_old_anchor_rotation,
             t_old_component_world_position, t_old_component_world_rotation, t_old_component_world_scale,
             t_old_frame_world_position, t_old_frame_world_rotation, t_old_frame_world_scale,
             t_old_world_position, t_old_world_rotation, t_reset_pending, t_rotation_speed_limit,
             t_running, t_skip_count, t_smoothing_velocity, t_stablization_time, t_teleport_distance,
             t_teleport_mode, t_teleport_rotation, t_time_reset, t_valid, t_velocity_weight,
             t_world_inertia, num_teams):
    mat_a = cuda.local.array((4, 4), float64)
    mat_b = cuda.local.array((4, 4), float64)
    mat_c = cuda.local.array((4, 4), float64)
    i = tid
    while i < num_teams:
        if team_frame_mask(t_enabled, t_valid, t_cws, i):
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

            init_scale_len = float64(dmath.length3(
                t_init_scale[i, 0], t_init_scale[i, 1], t_init_scale[i, 2]))
            if init_scale_len < float64(1e-30):
                init_scale_len = float64(1e-30)
            csr_ratio = float64(dmath.length3(csx, csy, csz)) / init_scale_len

            old_dx = t_negative_scale_direction[i, 0]
            old_dy = t_negative_scale_direction[i, 1]
            old_dz = t_negative_scale_direction[i, 2]
            sxv = float32(1.0) if csx == float32(0.0) else csx
            syv = float32(1.0) if csy == float32(0.0) else csy
            szv = float32(1.0) if csz == float32(0.0) else csz
            dir_x = dmath.fsign(sxv)
            dir_y = dmath.fsign(syv)
            dir_z = dmath.fsign(szv)
            t_negative_scale_direction[i, 0] = dir_x
            t_negative_scale_direction[i, 1] = dir_y
            t_negative_scale_direction[i, 2] = dir_z
            t_negative_scale_change[i, 0] = old_dx * dir_x
            t_negative_scale_change[i, 1] = old_dy * dir_y
            t_negative_scale_change[i, 2] = old_dz * dir_z
            is_negative = (csx < float32(0.0)) or (csy < float32(0.0)) or (csz < float32(0.0))
            t_is_negative_scale[i] = int32(1) if is_negative else int32(0)
            t_negative_scale_sign[i] = float32(-1.0) if is_negative else float32(1.0)
            if is_negative:
                t_negative_scale_quaternion[i, 0] = -dir_x
                t_negative_scale_quaternion[i, 1] = -dir_y
                t_negative_scale_quaternion[i, 2] = -dir_z
                t_negative_scale_quaternion[i, 3] = float32(1.0)
                ts0 = float32(-1.0) if (csx < float32(0.0) or csz < float32(0.0)) else float32(1.0)
                ts1 = float32(-1.0) if (csx < float32(0.0)) else float32(1.0)
                t_negative_scale_triangle_sign[i, 0] = ts0
                t_negative_scale_triangle_sign[i, 1] = ts1
            else:
                t_negative_scale_quaternion[i, 0] = float32(1.0)
                t_negative_scale_quaternion[i, 1] = float32(1.0)
                t_negative_scale_quaternion[i, 2] = float32(1.0)
                t_negative_scale_quaternion[i, 3] = float32(1.0)
                t_negative_scale_triangle_sign[i, 0] = float32(1.0)
                t_negative_scale_triangle_sign[i, 1] = float32(1.0)
            teleport = (old_dx != dir_x) or (old_dy != dir_y) or (old_dz != dir_z)
            t_negative_scale_teleport[i] = int32(1) if teleport else int32(0)

            if teleport:
                dmath.trs_build_f64(mat_a, cpx, cpy, cpz, crx, cry, crz, crw, csx, csy, csz)
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
                dmath.trs_inverse_f64(mat_b, ocpx, ocpy, ocpz, ocrx, ocry, ocrz, ocrw, ocsx, ocsy, ocsz)
                dmath.mat4_mul_f64(mat_c, mat_a, mat_b)
                nx, ny, nz = dmath.transform_point(mat_c, ocpx, ocpy, ocpz)
                t_old_component_world_position[i, 0] = nx
                t_old_component_world_position[i, 1] = ny
                t_old_component_world_position[i, 2] = nz
                t_old_component_world_scale[i, 0] = csx
                t_old_component_world_scale[i, 1] = csy
                t_old_component_world_scale[i, 2] = csz
                oax = t_old_anchor_position[i, 0]
                oay = t_old_anchor_position[i, 1]
                oaz = t_old_anchor_position[i, 2]
                tax, tay, taz = dmath.transform_point(mat_c, oax, oay, oaz)
                t_old_anchor_position[i, 0] = tax
                t_old_anchor_position[i, 1] = tay
                t_old_anchor_position[i, 2] = taz
                tsvx, tsvy, tsvz = dmath.transform_vector(
                    mat_c, t_smoothing_velocity[i, 0], t_smoothing_velocity[i, 1],
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
            nor_sx = float64(0.0)
            nor_sy = float64(0.0)
            nor_sz = float64(0.0)
            tan_sx = float64(0.0)
            tan_sy = float64(0.0)
            tan_sz = float64(0.0)
            pos_sx = float64(0.0)
            pos_sy = float64(0.0)
            pos_sz = float64(0.0)
            fcount = 0
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
                    rx, ry, rz, rw = dmath.to_rotation(-nnx, -nny, -nnz, -ttx, -tty, -ttz)
                rx, ry, rz, rw = dmath.quat_mul(
                    rx, ry, rz, rw, p_vertex_bind_pose_rotations[fp, 0],
                    p_vertex_bind_pose_rotations[fp, 1], p_vertex_bind_pose_rotations[fp, 2],
                    p_vertex_bind_pose_rotations[fp, 3])
                norx, nory, norz = dmath.quat_to_normal(rx, ry, rz, rw)
                tanx, tany, tanz = dmath.quat_to_tangent(rx, ry, rz, rw)
                nflip = float32(-1.0) if (dir_x < float32(0.0) or dir_z < float32(0.0)) else float32(1.0)
                tflip = float32(-1.0) if (dir_x < float32(0.0) or dir_y < float32(0.0)) else float32(1.0)
                nor_sx += float64(norx * nflip)
                nor_sy += float64(nory * nflip)
                nor_sz += float64(norz * nflip)
                tan_sx += float64(tanx * tflip)
                tan_sy += float64(tany * tflip)
                tan_sz += float64(tanz * tflip)
                pos_sx += float64(p_positions[fp, 0])
                pos_sy += float64(p_positions[fp, 1])
                pos_sz += float64(p_positions[fp, 2])
                fcount += 1
            if fcount > 0:
                nl = math.sqrt(nor_sx * nor_sx + nor_sy * nor_sy + nor_sz * nor_sz)
                tl = math.sqrt(tan_sx * tan_sx + tan_sy * tan_sy + tan_sz * tan_sz)
                if nl > float64(1e-30) and tl > float64(1e-30):
                    cwpx = float32(pos_sx / float64(fcount))
                    cwpy = float32(pos_sy / float64(fcount))
                    cwpz = float32(pos_sz / float64(fcount))
                    cwrx, cwry, cwrz, cwrw = dmath.to_rotation(
                        float32(nor_sx / nl), float32(nor_sy / nl), float32(nor_sz / nl),
                        float32(tan_sx / tl), float32(tan_sy / tl), float32(tan_sz / tl))

            if teleport:
                dmath.trs_build_f64(mat_a, cwpx, cwpy, cwpz, cwrx, cwry, cwrz, cwrw, csx, csy, csz)
                dmath.trs_inverse_f64(
                    mat_b, t_old_frame_world_position[i, 0], t_old_frame_world_position[i, 1],
                    t_old_frame_world_position[i, 2], t_old_frame_world_rotation[i, 0],
                    t_old_frame_world_rotation[i, 1], t_old_frame_world_rotation[i, 2],
                    t_old_frame_world_rotation[i, 3], t_old_frame_world_scale[i, 0],
                    t_old_frame_world_scale[i, 1], t_old_frame_world_scale[i, 2])
                dmath.mat4_mul_f64(t_negative_scale_matrix[i], mat_a, mat_b)

            adv_x = float32(0.0)
            adv_y = float32(0.0)
            adv_z = float32(0.0)
            adr_x = float32(0.0)
            adr_y = float32(0.0)
            adr_z = float32(0.0)
            adr_w = float32(1.0)
            has_anc = t_has_anchor[i] != 0
            anchor_reset = (has_anc != (t_had_anchor[i] != 0)) or (t_reset_pending[i] != 0)
            t_had_anchor[i] = int32(1) if has_anc else int32(0)
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
                    t_anchor_component_local_position[i, 1], t_anchor_component_local_position[i, 2])
                dvx = (rlx + t_anchor_position[i, 0]) - ocp_x
                dvy = (rly + t_anchor_position[i, 1]) - ocp_y
                dvz = (rlz + t_anchor_position[i, 2]) - ocp_z
                ioax, ioay, ioaz, ioaw = dmath.quat_inverse(
                    t_old_anchor_rotation[i, 0], t_old_anchor_rotation[i, 1],
                    t_old_anchor_rotation[i, 2], t_old_anchor_rotation[i, 3])
                drx, dry, drz, drw = dmath.quat_mul(
                    t_anchor_rotation[i, 0], t_anchor_rotation[i, 1], t_anchor_rotation[i, 2],
                    t_anchor_rotation[i, 3], ioax, ioay, ioaz, ioaw)
                a_ratio = float32(1.0) - t_anchor_inertia[i]
                adv_x = dvx * a_ratio
                adv_y = dvy * a_ratio
                adv_z = dvz * a_ratio
                adr_x, adr_y, adr_z, adr_w = dmath.quat_slerp(
                    float32(0.0), float32(0.0), float32(0.0), float32(1.0),
                    drx, dry, drz, drw, a_ratio)
                ocp_x = ocp_x + adv_x
                ocp_y = ocp_y + adv_y
                ocp_z = ocp_z + adv_z
                ocr_x, ocr_y, ocr_z, ocr_w = dmath.quat_mul(
                    adr_x, adr_y, adr_z, adr_w, ocr_x, ocr_y, ocr_z, ocr_w)
                t_inertia_shift[i] = int32(1)

            fdvx = cpx - ocp_x
            fdvy = cpy - ocp_y
            fdvz = cpz - ocp_z
            fda = dmath.quat_angle(ocr_x, ocr_y, ocr_z, ocr_w, crx, cry, crz, crw)
            if (t_teleport_mode[i] != 0) and (t_reset_pending[i] == 0):
                far = float64(dmath.length3(fdvx, fdvy, fdvz)) >= \
                    float64(t_teleport_distance[i]) * csr_ratio
                spun = (fda * RAD2DEG) >= t_teleport_rotation[i]
                if far or spun:
                    if t_teleport_mode[i] == TELEPORT_RESET:
                        t_reset_pending[i] = int32(1)
                    else:
                        t_keep_teleport_pending[i] = int32(1)

            reset = t_reset_pending[i] != 0
            keep = t_keep_teleport_pending[i] != 0

            sdv_x = float32(0.0)
            sdv_y = float32(0.0)
            sdv_z = float32(0.0)
            if (t_movement_inertia_smoothing[i] >= float32(1e-6)) and (not (keep or reset)):
                running = t_running[i] != 0
                fdt_i = t_frame_dt[i]
                if fdt_i > float32(0.0):
                    dvx = fdvx / fdt_i
                    dvy = fdvy / fdt_i
                    dvz = fdvz / fdt_i
                else:
                    dvx = float32(0.0)
                    dvy = float32(0.0)
                    dvz = float32(0.0)
                limit = t_movement_speed_limit[i] * float32(csr_ratio)
                mlim = limit if limit > float32(0.0) else float32(0.0)
                cvx, cvy, cvz = dmath.clamp_vector(dvx, dvy, dvz, mlim)
                if limit >= float32(0.0):
                    dvx = cvx
                    dvy = cvy
                    dvz = cvz
                mis = t_movement_inertia_smoothing[i]
                om = float32(1.0) - mis
                avg = dmath.saturate(om * om * om * float32(0.99) + float32(0.01))
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
                t_inertia_shift[i] = int32(1)

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
            shv_x = float32(0.0)
            shv_y = float32(0.0)
            shv_z = float32(0.0)
            shr_x = float32(0.0)
            shr_y = float32(0.0)
            shr_z = float32(0.0)
            shr_w = float32(1.0)
            if reset:
                t_smoothing_velocity[i, 0] = float32(0.0)
                t_smoothing_velocity[i, 1] = float32(0.0)
                t_smoothing_velocity[i, 2] = float32(0.0)
                sdv_x = float32(0.0)
                sdv_y = float32(0.0)
                sdv_z = float32(0.0)

            if not reset:
                shv_x = cpx - ocp_x
                shv_y = cpy - ocp_y
                shv_z = cpz - ocp_z
                iox, ioy, ioz, iow = dmath.quat_inverse(ocr_x, ocr_y, ocr_z, ocr_w)
                shr_x, shr_y, shr_z, shr_w = dmath.quat_mul(crx, cry, crz, crw, iox, ioy, ioz, iow)
                msr = float32(0.0)
                rsr = float32(0.0)
                keep_now = keep or (t_culling_invisible[i] != 0)
                if keep_now:
                    movement_shift = float32(1.0)
                else:
                    movement_shift = float32(1.0) - t_world_inertia[i]
                rotation_shift = movement_shift
                if movement_shift > EPSILON:
                    t_inertia_shift[i] = int32(1)
                    msr = movement_shift
                    rsr = rotation_shift
                    wpx = dmath.lerp(wpx, cpx, movement_shift)
                    wpy = dmath.lerp(wpy, cpy, movement_shift)
                    wpz = dmath.lerp(wpz, cpz, movement_shift)
                    wrx, wry, wrz, wrw = dmath.quat_slerp(
                        wrx, wry, wrz, wrw, crx, cry, crz, crw, rotation_shift)
                movement_limit = float64(t_movement_speed_limit[i]) * csr_ratio
                rotation_limit = t_rotation_speed_limit[i]
                dvx = cpx - wpx
                dvy = cpy - wpy
                dvz = cpz - wpz
                dang = dmath.quat_angle(wrx, wry, wrz, wrw, crx, cry, crz, crw)
                fdt_l = t_frame_dt[i]
                if fdt_l > float32(0.0):
                    frame_speed = dmath.length3(dvx, dvy, dvz) / fdt_l
                    frame_rot_speed = (dang * RAD2DEG) / fdt_l
                else:
                    frame_speed = float32(0.0)
                    frame_rot_speed = float32(0.0)
                over_move = (float64(frame_speed) > movement_limit) and \
                    (t_movement_speed_limit[i] >= float32(0.0))
                if over_move:
                    t_inertia_shift[i] = int32(1)
                    denom_fs = frame_speed if frame_speed > float32(0.0) else float32(1.0)
                    mlr = dmath.saturate((frame_speed - float32(movement_limit)) / denom_fs)
                else:
                    mlr = float32(0.0)
                msr = msr + (float32(1.0) - msr) * mlr
                if over_move:
                    wpx = dmath.lerp(wpx, cpx, mlr)
                    wpy = dmath.lerp(wpy, cpy, mlr)
                    wpz = dmath.lerp(wpz, cpz, mlr)
                over_rot = (frame_rot_speed > rotation_limit) and (rotation_limit >= float32(0.0))
                if over_rot:
                    t_inertia_shift[i] = int32(1)
                    denom_frs = frame_rot_speed if frame_rot_speed > float32(0.0) else float32(1.0)
                    rlr = dmath.saturate((frame_rot_speed - rotation_limit) / denom_frs)
                else:
                    rlr = float32(0.0)
                rsr = rsr + (float32(1.0) - rsr) * rlr
                if over_rot:
                    wrx, wry, wrz, wrw = dmath.quat_slerp(
                        wrx, wry, wrz, wrw, crx, cry, crz, crw, rlr)
                osr = float64(0.0)
                skip = t_skip_count[i]
                scaled_dt = fdt_l * t_now_time_scale[i]
                if (skip > 0) and (scaled_dt > float32(0.0)):
                    sr = float64(skip) * float64(sim_dt) / float64(scaled_dt)
                    if sr < float64(0.0):
                        sr = float64(0.0)
                    elif sr > float64(1.0):
                        sr = float64(1.0)
                    osr = osr + (float64(1.0) - osr) * sr
                vw = t_velocity_weight[i]
                if vw < float32(1.0):
                    osr = osr + (float64(1.0) - osr) * (float64(1.0) - float64(vw))
                nts = t_now_time_scale[i]
                if nts < float32(1.0):
                    osr = osr + (float64(1.0) - osr) * (float64(1.0) - float64(nts))
                msr_final = float64(msr)
                rsr_final = float64(rsr)
                if osr > float64(0.0):
                    t_inertia_shift[i] = int32(1)
                    msr_final = msr_final + (float64(1.0) - msr_final) * osr
                    osr_f32 = float32(osr)
                    wpx = dmath.lerp(wpx, cpx, osr_f32)
                    wpy = dmath.lerp(wpy, cpy, osr_f32)
                    wpz = dmath.lerp(wpz, cpz, osr_f32)
                    rsr_final = rsr_final + (float64(1.0) - rsr_final) * osr
                    wrx, wry, wrz, wrw = dmath.quat_slerp(
                        wrx, wry, wrz, wrw, crx, cry, crz, crw, osr_f32)
                if t_inertia_shift[i] != 0:
                    vecx = float64(shv_x) * msr_final + float64(adv_x) + float64(sdv_x)
                    vecy = float64(shv_y) * msr_final + float64(adv_y) + float64(sdv_y)
                    vecz = float64(shv_z) * msr_final + float64(adv_z) + float64(sdv_z)
                    rqx, rqy, rqz, rqw = dmath.quat_slerp(
                        float32(0.0), float32(0.0), float32(0.0), float32(1.0),
                        shr_x, shr_y, shr_z, shr_w, float32(rsr_final))
                    rqx, rqy, rqz, rqw = dmath.quat_mul(adr_x, adr_y, adr_z, adr_w, rqx, rqy, rqz, rqw)
                    t_frame_component_shift_vector[i, 0] = float32(vecx)
                    t_frame_component_shift_vector[i, 1] = float32(vecy)
                    t_frame_component_shift_vector[i, 2] = float32(vecz)
                    t_frame_component_shift_rotation[i, 0] = rqx
                    t_frame_component_shift_rotation[i, 1] = rqy
                    t_frame_component_shift_rotation[i, 2] = rqz
                    t_frame_component_shift_rotation[i, 3] = rqw
                    oc_x = t_old_component_world_position[i, 0]
                    oc_y = t_old_component_world_position[i, 1]
                    oc_z = t_old_component_world_position[i, 2]
                    rlx1, rly1, rlz1 = dmath.quat_rotate(
                        rqx, rqy, rqz, rqw, t_old_frame_world_position[i, 0] - oc_x,
                        t_old_frame_world_position[i, 1] - oc_y, t_old_frame_world_position[i, 2] - oc_z)
                    t_old_frame_world_position[i, 0] = float32(float64(rlx1 + oc_x) + vecx)
                    t_old_frame_world_position[i, 1] = float32(float64(rly1 + oc_y) + vecy)
                    t_old_frame_world_position[i, 2] = float32(float64(rlz1 + oc_z) + vecz)
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
                    t_now_world_position[i, 0] = float32(float64(rlx2 + oc_x) + vecx)
                    t_now_world_position[i, 1] = float32(float64(rly2 + oc_y) + vecy)
                    t_now_world_position[i, 2] = float32(float64(rlz2 + oc_z) + vecz)
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
                t_frame_component_shift_vector[i, 0] = float32(0.0)
                t_frame_component_shift_vector[i, 1] = float32(0.0)
                t_frame_component_shift_vector[i, 2] = float32(0.0)
                t_frame_component_shift_rotation[i, 0] = float32(0.0)
                t_frame_component_shift_rotation[i, 1] = float32(0.0)
                t_frame_component_shift_rotation[i, 2] = float32(0.0)
                t_frame_component_shift_rotation[i, 3] = float32(1.0)

            mvx = cpx - wpx
            mvy = cpy - wpy
            mvz = cpz - wpz
            mlen = dmath.length3(mvx, mvy, mvz)
            fdt_m = t_frame_dt[i]
            if fdt_m > float32(0.0):
                speed = mlen / fdt_m
            else:
                speed = float32(0.0)
            nts_m = t_now_time_scale[i]
            if nts_m > float32(1e-6):
                speed = speed * (float32(1.0) / nts_m)
            else:
                speed = float32(0.0)
            t_frame_moving_speed[i] = speed
            if mlen > float32(1e-6):
                t_frame_moving_direction[i, 0] = mvx / mlen
                t_frame_moving_direction[i, 1] = mvy / mlen
                t_frame_moving_direction[i, 2] = mvz / mlen
            else:
                t_frame_moving_direction[i, 0] = float32(0.0)
                t_frame_moving_direction[i, 1] = float32(0.0)
                t_frame_moving_direction[i, 2] = float32(0.0)

            if (t_reset_pending[i] != 0) or (t_time_reset[i] != 0):
                if t_stablization_time[i] > float32(1e-6):
                    wgt = float32(0.0)
                else:
                    wgt = float32(1.0)
                t_velocity_weight[i] = wgt
                t_blend_weight[i] = wgt
        i += stride


@phase('teams')
def phase_03b(tid, stride, n_zones, t_cws, t_enabled, t_frame_world_position, t_valid, t_wind_count,
              t_wind_direction, t_wind_dirq, t_wind_influence, t_wind_main, t_wind_time, t_wind_zone_id,
              t_wind_zone_turbulence, z_attenuation_lut, z_is_addition, z_main, z_mode, z_size,
              z_turbulence, z_world_direction, z_world_position, z_world_to_local, z_zone_id,
              z_zone_volume, num_teams):
    res_zone_id = cuda.local.array(8, int32)
    res_time = cuda.local.array(8, float32)
    res_main = cuda.local.array(8, float32)
    res_dx = cuda.local.array(8, float32)
    res_dy = cuda.local.array(8, float32)
    res_dz = cuda.local.array(8, float32)
    res_turb = cuda.local.array(8, float32)
    old_zid = cuda.local.array(4, int32)
    old_wt = cuda.local.array(4, float32)
    i = tid
    while i < num_teams:
        if team_frame_mask(t_enabled, t_valid, t_cws, i):
            old_count = t_wind_count[i]
            for oc in range(WIND_ZONE_SLOTS):
                if oc < old_count:
                    old_zid[oc] = t_wind_zone_id[i, oc]
                    old_wt[oc] = t_wind_time[i, oc]
            count = 0
            if n_zones > 0 and t_wind_influence[i] > EPSILON:
                cx64 = float64(t_frame_world_position[i, 0])
                cy64 = float64(t_frame_world_position[i, 1])
                cz64 = float64(t_frame_world_position[i, 2])
                min_volume = INF
                addition_count = 0
                latest_valid = False
                latest_id = int32(0)
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
                    llen = math.sqrt(lxx * lxx + lyy * lyy + lzz * lzz)
                    skip_zone = False
                    if mode == ZONE_BOX:
                        if abs(lxx) * float64(2.0) > float64(z_size[zi, 0]) or \
                                abs(lyy) * float64(2.0) > float64(z_size[zi, 1]) or \
                                abs(lzz) * float64(2.0) > float64(z_size[zi, 2]):
                            skip_zone = True
                    elif mode == ZONE_SPHERE_DIR or mode == ZONE_SPHERE_RADIAL:
                        if llen > float64(z_size[zi, 0]):
                            skip_zone = True
                    if skip_zone:
                        continue
                    if (not is_add) and (zvol > min_volume):
                        continue
                    dirx = z_world_direction[zi, 0]
                    diry = z_world_direction[zi, 1]
                    dirz = z_world_direction[zi, 2]
                    zmain = z_main[zi]
                    if mode == ZONE_SPHERE_RADIAL:
                        if llen <= float64(1e-6):
                            continue
                        vx64 = cx64 - float64(z_world_position[zi, 0])
                        vy64 = cy64 - float64(z_world_position[zi, 1])
                        vz64 = cz64 - float64(z_world_position[zi, 2])
                        vlen = math.sqrt(vx64 * vx64 + vy64 * vy64 + vz64 * vz64)
                        dirx = float32(vx64 / vlen)
                        diry = float32(vy64 / vlen)
                        dirz = float32(vz64 / vlen)
                        depth = llen / float64(z_size[zi, 0])
                        if depth < float64(0.0):
                            depth = float64(0.0)
                        elif depth > float64(1.0):
                            depth = float64(1.0)
                        zmain = zmain * dmath.evaluate_team_lut_clamp01(
                            z_attenuation_lut, zi, float32(depth))
                    zid = z_zone_id[zi]
                    t_prev = -WIND_MAX_TIME
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
                            w = 0
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
                        latest_valid = True
            final = count if count < WIND_ZONE_SLOTS else WIND_ZONE_SLOTS
            t_wind_count[i] = int8(final)
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
        i += stride


@phase('particles')
def phase_04(tid, stride, p_base_positions, p_base_rotations, p_collision_normals, p_display_positions,
             p_friction, p_next_positions, p_old_anim_positions, p_old_anim_rotations, p_old_positions,
             p_old_rotations, p_positions, p_real_velocities, p_rotations, p_static_friction, p_team,
             p_velocities, p_velocity_positions, t_cws, t_enabled, t_frame_component_shift_rotation,
             t_frame_component_shift_vector, t_inertia_shift, t_negative_scale_matrix,
             t_negative_scale_teleport, t_old_component_world_position, t_reset_pending, t_valid,
             num_particles):
    p = tid
    while p < num_particles:
        if team_frame_mask(t_enabled, t_valid, t_cws, p_team[p]):
            do_particles_frame_pre(p, p_team, p_positions, p_rotations, p_next_positions,
                                   p_old_positions, p_old_rotations, p_base_positions,
                                   p_base_rotations, p_old_anim_positions, p_old_anim_rotations,
                                   p_velocity_positions, p_display_positions, p_velocities,
                                   p_real_velocities, p_friction, p_static_friction,
                                   p_collision_normals, t_reset_pending,
                                   t_negative_scale_teleport, t_negative_scale_matrix,
                                   t_inertia_shift, t_frame_component_shift_vector,
                                   t_frame_component_shift_rotation,
                                   t_old_component_world_position)
        p += stride


@phase('colliders')
def phase_05(tid, stride, c_active, c_enabled, c_enabled_prev, c_frame_pos, c_frame_radius, c_frame_rot,
             c_frame_tip, c_input_positions, c_input_radii, c_input_rotations, c_input_tips, c_now_pos,
             c_now_rot, c_now_tip, c_old_frame_pos, c_old_frame_rot, c_old_frame_tip, c_old_pos, c_old_rot,
             c_old_tip, c_team, t_cws, t_enabled, t_frame_component_shift_rotation,
             t_frame_component_shift_vector, t_inertia_shift, t_negative_scale_change,
             t_negative_scale_matrix, t_negative_scale_teleport, t_old_component_world_position,
             t_reset_pending, t_valid, num_colliders):
    ci = tid
    while ci < num_colliders:
        if team_frame_mask(t_enabled, t_valid, t_cws, c_team[ci]):
            do_collider_frame_pre(ci, c_team, c_enabled, c_enabled_prev, c_active,
                                  c_input_positions, c_input_rotations, c_input_tips, c_input_radii,
                                  c_frame_pos, c_frame_rot, c_frame_tip, c_frame_radius,
                                  c_old_frame_pos, c_old_frame_rot, c_old_frame_tip,
                                  c_now_pos, c_now_rot, c_now_tip,
                                  c_old_pos, c_old_rot, c_old_tip,
                                  t_reset_pending, t_negative_scale_teleport,
                                  t_negative_scale_matrix, t_negative_scale_change,
                                  t_inertia_shift, t_frame_component_shift_vector,
                                  t_frame_component_shift_rotation,
                                  t_old_component_world_position)
        ci += stride


@phase('single')
def phase_07(tid, scl_counts):
    if tid == 0:
        scl_counts[SCL_IP_COUNT] = int32(0)


@phase('self_pairs')
def phase_08(tid, stride, ip_edge, ip_tri, it_edge_start, it_pair_off, it_same, it_tri_count, it_tri_start,
             it_tri_team, scl_counts, sfe_aabb_max, sfe_aabb_min, sfe_all_fix, sfe_ignore, sfe_particles,
             sft_aabb_max, sft_aabb_min, sft_all_fix, sft_ignore, sft_particles, sft_use, t_self_grid_size,
             t_self_max_primitive_size, ip_cap, num_it_slots):
    total_it = it_pair_off[num_it_slots]
    frame_index = scl_counts[SCL_FRAME_INDEX]
    g = tid
    while g < total_it:
        lo = int32(0)
        hi = num_it_slots
        while lo < hi:
            mid = (lo + hi) >> 1
            if it_pair_off[mid + 1] <= g:
                lo = mid + 1
            else:
                hi = mid
        task = lo
        tgt_team = it_tri_team[task]
        if t_self_grid_size[tgt_team] > EPSILON and t_self_max_primitive_size[tgt_team] > EPSILON:
            tri_count = it_tri_count[task]
            local = g - it_pair_off[task]
            i = local // tri_count
            j = local % tri_count
            my_edge = it_edge_start[task] + i
            tgt_tri = it_tri_start[task] + j
            same = it_same[task]
            if (sfe_ignore[my_edge] == 0 and (i % SELF_COLLISION_INTERSECT_DIV) == frame_index
                    and sft_use[tgt_tri] != 0 and sft_ignore[tgt_tri] == 0
                    and self_aabb_overlap(sfe_aabb_min, sfe_aabb_max, my_edge,
                                          sft_aabb_min, sft_aabb_max, tgt_tri)
                    and not (sfe_all_fix[my_edge] != 0 and sft_all_fix[tgt_tri] != 0)):
                conn = (same == 0) or (not self_connection_shared(
                    sfe_particles, my_edge, sft_particles, tgt_tri))
                if conn:
                    idx = cuda.atomic.add(scl_counts, SCL_IP_COUNT, 1)
                    if idx < ip_cap:
                        ip_edge[idx] = my_edge
                        ip_tri[idx] = tgt_tri
                    else:
                        scl_counts[SCL_ERROR] = int32(1)
        g += stride


@phase('teams')
def phase_10(tid, stride, _k, sim_dt, t_angular_velocity, t_blend_weight, t_blend_weight_param, t_cws,
             t_distance_weight, t_enabled, t_frame_interpolation, t_frame_moving_direction,
             t_frame_moving_speed, t_frame_old, t_frame_world_position, t_frame_world_rotation,
             t_frame_world_scale, t_gravity, t_gravity_direction, t_gravity_dot, t_gravity_falloff,
             t_gravity_ratio, t_inertia_rotation, t_inertia_vector, t_init_local_gravity_direction,
             t_init_scale, t_local_inertia, t_local_movement_speed_limit, t_local_rotation_speed_limit,
             t_moving_wind_direction, t_moving_wind_dirq, t_moving_wind_main, t_moving_wind_time,
             t_negative_scale_direction, t_now_update, t_now_world_position, t_now_world_rotation,
             t_old_frame_world_position, t_old_frame_world_rotation, t_old_frame_world_scale,
             t_old_world_position, t_old_world_rotation, t_rotation_axis, t_scale_ratio,
             t_stablization_time, t_step_move_inertia_ratio, t_step_rotation,
             t_step_rotation_inertia_ratio, t_step_vector, t_time, t_update_count, t_valid,
             t_velocity_weight, t_wind_count, t_wind_frequency, t_wind_main, t_wind_moving, t_wind_time,
             num_teams):
    i = tid
    while i < num_teams:
        if team_frame_mask(t_enabled, t_valid, t_cws, i) and t_update_count[i] > _k:
            do_step_update(i, sim_dt,
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
        i += stride


@phase('colliders')
def phase_11(tid, stride, _k, c_active, c_frame_pos, c_frame_radius, c_frame_rot, c_frame_tip, c_kind,
             c_now_pos, c_now_rot, c_now_tip, c_old_frame_pos, c_old_frame_rot, c_old_frame_tip, c_old_pos,
             c_old_rot, c_old_tip, c_team, c_work_aabb_max, c_work_aabb_min, c_work_inv_old_rot,
             c_work_next_pos, c_work_old_pos, c_work_radius, c_work_rot, t_cws, t_enabled,
             t_frame_interpolation, t_step_move_inertia_ratio, t_step_rotation_inertia_ratio,
             t_update_count, t_valid, num_colliders):
    ci = tid
    while ci < num_colliders:
        cm = c_team[ci]
        if team_frame_mask(t_enabled, t_valid, t_cws, cm) and t_update_count[cm] > _k \
                and c_active[ci] != 0:
            do_collider_start_step(ci, c_team, c_kind,
                                   c_frame_pos, c_frame_rot, c_frame_tip, c_frame_radius,
                                   c_old_frame_pos, c_old_frame_rot, c_old_frame_tip,
                                   c_now_pos, c_now_rot, c_now_tip,
                                   c_old_pos, c_old_rot, c_old_tip,
                                   c_work_rot, c_work_inv_old_rot,
                                   c_work_radius, c_work_old_pos, c_work_next_pos,
                                   c_work_aabb_min, c_work_aabb_max,
                                   t_frame_interpolation, t_step_move_inertia_ratio,
                                   t_step_rotation_inertia_ratio)
        ci += stride


@phase('particles')
def phase_12(tid, stride, _k, power2, sim_dt, p_base_positions, p_base_rotations, p_depth, p_friction,
             p_next_positions, p_old_anim_positions, p_old_anim_rotations, p_old_positions, p_positions,
             p_rotations, p_step_basic_positions, p_step_basic_rotations, p_team, p_velocities,
             p_velocity_positions, p_vertex_root_local, st_move_particle, st_move_team, t_cws,
             t_damping_lut, t_depth_inertia, t_enabled, t_force_mode, t_frame_interpolation, t_gravity,
             t_gravity_direction, t_gravity_ratio, t_impact_force, t_inertia_rotation, t_inertia_vector,
             t_moving_wind_dirq, t_moving_wind_main, t_moving_wind_time, t_old_world_position,
             t_scale_ratio, t_step_rotation, t_step_vector, t_update_count, t_valid, t_velocity_weight,
             t_wind_blend, t_wind_count, t_wind_depth_weight, t_wind_dirq, t_wind_influence, t_wind_main,
             t_wind_moving, t_wind_seed, t_wind_synchronization, t_wind_time, t_wind_turbulence,
             t_wind_zone_turbulence, n_move, num_particles):
    p = tid
    while p < num_particles:
        mt = p_team[p]
        if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
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
        p += stride

    e = tid
    while e < n_move:
        mt = st_move_team[e]
        if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
            pmi = st_move_particle[e]
            depth = p_depth[pmi]
            ox = p_old_positions[pmi, 0]
            oy = p_old_positions[pmi, 1]
            oz = p_old_positions[pmi, 2]

            inertia_depth = t_depth_inertia[mt] * (float32(1.0) - depth * depth)
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
            damp = dmath.saturate(float32(1.0) - damping * power2)
            vx = vx * damp
            vy = vy * damp
            vz = vz * damp

            fm = t_force_mode[mt]
            change = (fm == FORCE_VELOCITY_CHANGE) or (fm == FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH)
            if change:
                vx = float32(0.0)
                vy = float32(0.0)
                vz = float32(0.0)

            g = t_gravity[mt] * t_gravity_ratio[mt]
            fx = t_gravity_direction[mt, 0] * g
            fy = t_gravity_direction[mt, 1] * g
            fz = t_gravity_direction[mt, 2] * g
            mass = dmath.calc_mass(depth)
            with_depth = (fm == FORCE_VELOCITY_ADD) or (fm == FORCE_VELOCITY_CHANGE)
            without_depth = (fm == FORCE_VELOCITY_ADD_WITHOUT_DEPTH) or (fm == FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH)
            if with_depth:
                fx = fx + t_impact_force[mt, 0] / mass
                fy = fy + t_impact_force[mt, 1] / mass
                fz = fz + t_impact_force[mt, 2] / mass
            if without_depth:
                fx = fx + t_impact_force[mt, 0]
                fy = fy + t_impact_force[mt, 1]
                fz = fz + t_impact_force[mt, 2]

            root = float32(p_vertex_root_local[pmi])
            seed = float32(t_wind_seed[mt])
            sync = t_wind_synchronization[mt]
            wind_position = (seed + float32(1.0)) * float32(4.19230645) \
                + root * float32(0.0023963) * (float32(1.0) - sync) * float32(100.0)
            blend = t_wind_blend[mt]
            turbulence_param = t_wind_turbulence[mt]
            wfx = float32(0.0)
            wfy = float32(0.0)
            wfz = float32(0.0)
            wc = t_wind_count[mt]
            for s in range(4):
                if s < wc:
                    cx, cy, cz = do_wind_blend(
                        t_wind_main[mt, s], t_wind_time[mt, s],
                        t_wind_dirq[mt, s, 0], t_wind_dirq[mt, s, 1],
                        t_wind_dirq[mt, s, 2], t_wind_dirq[mt, s, 3],
                        t_wind_zone_turbulence[mt, s], blend, turbulence_param, wind_position)
                    wfx = wfx + cx
                    wfy = wfy + cy
                    wfz = wfz + cz
            moving_on = t_wind_moving[mt] > WIND_MIN_SPEED
            if moving_on:
                cx, cy, cz = do_wind_blend(
                    t_moving_wind_main[mt], t_moving_wind_time[mt],
                    t_moving_wind_dirq[mt, 0], t_moving_wind_dirq[mt, 1],
                    t_moving_wind_dirq[mt, 2], t_moving_wind_dirq[mt, 3],
                    float32(1.0), blend, turbulence_param, wind_position)
                wfx = wfx + cx
                wfy = wfy + cy
                wfz = wfz + cz
            influence = t_wind_influence[mt] * (float32(1.0) - p_friction[pmi])
            depth_scale = depth * depth
            influence = influence * dmath.lerp(float32(1.0), depth_scale, t_wind_depth_weight[mt])
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
        e += stride


@phase('particles')
def phase_13(tid, stride, _k, p_base_positions, p_base_rotations, p_next_positions, p_velocity_positions,
             st_fixed_particle, st_fixed_team, st_spring_particle, st_spring_team, t_cws, t_enabled,
             t_normal_axis_vector, t_scale_ratio, t_spring_limit_distance, t_spring_noise,
             t_spring_normal_limit_ratio, t_spring_power, t_time, t_update_count, t_valid, n_fixed,
             n_spring):
    e = tid
    while e < n_fixed:
        ft = st_fixed_team[e]
        if team_frame_mask(t_enabled, t_valid, t_cws, ft) and t_update_count[ft] > _k:
            pfi = st_fixed_particle[e]
            p_next_positions[pfi, 0] = p_base_positions[pfi, 0]
            p_next_positions[pfi, 1] = p_base_positions[pfi, 1]
            p_next_positions[pfi, 2] = p_base_positions[pfi, 2]
            p_velocity_positions[pfi, 0] = p_base_positions[pfi, 0]
            p_velocity_positions[pfi, 1] = p_base_positions[pfi, 1]
            p_velocity_positions[pfi, 2] = p_base_positions[pfi, 2]
        e += stride

    e = tid
    while e < n_spring:
        st = st_spring_team[e]
        if team_frame_mask(t_enabled, t_valid, t_cws, st) and t_update_count[st] > _k \
                and t_spring_power[st] > float32(0.0):
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
            clampable = limit > float32(1e-8)
            l = dmath.length3(vx, vy, vz)
            over = clampable and (l > limit)
            if over and (l > float32(1e-30)):
                scale = limit / l
                vx = vx * scale
                vy = vy * scale
                vz = vz * scale
            ratio = t_spring_normal_limit_ratio[st]
            elliptic = clampable and (ratio < float32(1.0))
            ylen = dmath.dot3(dx, dy, dz, vx, vy, vz)
            vpx = vx - dx * ylen
            vpy = vy - dy * ylen
            vpz = vz - dz * ylen
            xlen = dmath.length3(vpx, vpy, vpz)
            safe_limit = limit if limit > float32(1e-30) else float32(1.0)
            tval = dmath.saturate(xlen / safe_limit)
            y = libdevice.cosf(libdevice.asinf(dmath.clamp1(tval))) * (limit * ratio)
            exceed = elliptic and (libdevice.fabsf(ylen) > y)
            if exceed:
                adjust = (libdevice.fabsf(ylen) - y) * dmath.fsign(ylen)
                vx = vx - adjust * dx
                vy = vy - adjust * dy
                vz = vz - adjust * dz
            if not clampable:
                vx = float32(0.0)
                vy = float32(0.0)
                vz = float32(0.0)

            power = t_spring_power[st]
            noise_param = t_spring_noise[st]
            if noise_param > float32(0.0):
                noise_time = (t_time[st] + float32(psi) * float32(49.6198)) * float32(2.4512) \
                    + (n0 + n1 + n2)
                noise = libdevice.sinf(noise_time) * (noise_param * float32(0.6))
                power = power + power * noise
                if power < float32(0.0):
                    power = float32(0.0)
            vx = vx - vx * power
            vy = vy - vy * power
            vz = vz - vz * power
            p_next_positions[psi, 0] = bpx + vx
            p_next_positions[psi, 1] = bpy + vy
            p_next_positions[psi, 2] = bpz + vz
        e += stride


@phase('single')
def phase_14(tid, bid, bdim, _k, fk_no, fk_no_offsets, fk_yes, fk_yes_offsets, fk_yes_parent,
             p_step_basic_positions, p_step_basic_rotations, p_team, p_vertex_local_positions,
             p_vertex_local_rotations, t_animation_pose_ratio, t_cws, t_enabled, t_init_scale,
             t_is_negative_scale, t_negative_scale_direction, t_negative_scale_quaternion, t_scale_ratio,
             t_update_count, t_valid, num_fk_levels):
    for lvl in range(num_fk_levels):
        if bid == 0:
            ys = fk_yes_offsets[lvl]
            ye = fk_yes_offsets[lvl + 1]
            i = ys + tid
            while i < ye:
                v = fk_yes[i]
                vt = p_team[v]
                if team_frame_mask(t_enabled, t_valid, t_cws, vt) and t_update_count[vt] > _k \
                        and t_animation_pose_ratio[vt] <= float32(0.99):
                    par = fk_yes_parent[i]
                    sr = t_scale_ratio[vt]
                    scx = t_init_scale[vt, 0] * sr
                    scy = t_init_scale[vt, 1] * sr
                    scz = t_init_scale[vt, 2] * sr
                    lsx = (p_vertex_local_positions[v, 0] * t_negative_scale_direction[vt, 0]) * scx
                    lsy = (p_vertex_local_positions[v, 1] * t_negative_scale_direction[vt, 1]) * scy
                    lsz = (p_vertex_local_positions[v, 2] * t_negative_scale_direction[vt, 2]) * scz
                    prx = p_step_basic_rotations[par, 0]
                    pry = p_step_basic_rotations[par, 1]
                    prz = p_step_basic_rotations[par, 2]
                    prw = p_step_basic_rotations[par, 3]
                    rx, ry, rz = dmath.quat_rotate(prx, pry, prz, prw, lsx, lsy, lsz)
                    p_step_basic_positions[v, 0] = rx + p_step_basic_positions[par, 0]
                    p_step_basic_positions[v, 1] = ry + p_step_basic_positions[par, 1]
                    p_step_basic_positions[v, 2] = rz + p_step_basic_positions[par, 2]
                    lrx = p_vertex_local_rotations[v, 0] * t_negative_scale_quaternion[vt, 0]
                    lry = p_vertex_local_rotations[v, 1] * t_negative_scale_quaternion[vt, 1]
                    lrz = p_vertex_local_rotations[v, 2] * t_negative_scale_quaternion[vt, 2]
                    lrw = p_vertex_local_rotations[v, 3] * t_negative_scale_quaternion[vt, 3]
                    qx, qy, qz, qw = dmath.quat_mul(prx, pry, prz, prw, lrx, lry, lrz, lrw)
                    p_step_basic_rotations[v, 0] = qx
                    p_step_basic_rotations[v, 1] = qy
                    p_step_basic_rotations[v, 2] = qz
                    p_step_basic_rotations[v, 3] = qw
                i += bdim
        cuda.syncthreads()
        if bid == 0:
            ns = fk_no_offsets[lvl]
            ne = fk_no_offsets[lvl + 1]
            i = ns + tid
            while i < ne:
                v = fk_no[i]
                vt = p_team[v]
                if team_frame_mask(t_enabled, t_valid, t_cws, vt) and t_update_count[vt] > _k \
                        and t_animation_pose_ratio[vt] <= float32(0.99) \
                        and t_is_negative_scale[vt] != 0:
                    rox = p_step_basic_rotations[v, 0]
                    roy = p_step_basic_rotations[v, 1]
                    roz = p_step_basic_rotations[v, 2]
                    row = p_step_basic_rotations[v, 3]
                    nx, ny, nz = dmath.quat_to_normal(rox, roy, roz, row)
                    dy = t_negative_scale_direction[vt, 1]
                    dz = t_negative_scale_direction[vt, 2]
                    nnx = nx * dy
                    nny = ny * dy
                    nnz = nz * dy
                    tx, ty, tz = dmath.quat_to_tangent(rox, roy, roz, row)
                    ttx = tx * dz
                    tty = ty * dz
                    ttz = tz * dz
                    lqx, lqy, lqz, lqw = dmath.look_rotation(ttx, tty, ttz, nnx, nny, nnz)
                    p_step_basic_rotations[v, 0] = lqx
                    p_step_basic_rotations[v, 1] = lqy
                    p_step_basic_rotations[v, 2] = lqz
                    p_step_basic_rotations[v, 3] = lqw
                i += bdim
        cuda.syncthreads()


@phase('particles')
def phase_15(tid, stride, _k, baseline_entries, p_base_positions, p_base_rotations, p_step_basic_positions,
             p_step_basic_rotations, p_team, t_animation_pose_ratio, t_cws, t_enabled, t_update_count,
             t_valid, n_baseline):
    i = tid
    while i < n_baseline:
        v = baseline_entries[i]
        vt = p_team[v]
        apr = t_animation_pose_ratio[vt]
        if team_frame_mask(t_enabled, t_valid, t_cws, vt) and t_update_count[vt] > _k \
                and apr <= float32(0.99) and apr > EPSILON:
            p_step_basic_positions[v, 0] = dmath.lerp(p_step_basic_positions[v, 0],
                                                     p_base_positions[v, 0], apr)
            p_step_basic_positions[v, 1] = dmath.lerp(p_step_basic_positions[v, 1],
                                                     p_base_positions[v, 1], apr)
            p_step_basic_positions[v, 2] = dmath.lerp(p_step_basic_positions[v, 2],
                                                     p_base_positions[v, 2], apr)
            qx, qy, qz, qw = dmath.quat_slerp(
                p_step_basic_rotations[v, 0], p_step_basic_rotations[v, 1],
                p_step_basic_rotations[v, 2], p_step_basic_rotations[v, 3],
                p_base_rotations[v, 0], p_base_rotations[v, 1],
                p_base_rotations[v, 2], p_base_rotations[v, 3], apr)
            p_step_basic_rotations[v, 0] = qx
            p_step_basic_rotations[v, 1] = qy
            p_step_basic_rotations[v, 2] = qz
            p_step_basic_rotations[v, 3] = qw
        i += stride


@phase('particles')
def phase_16(tid, stride, _k, p_next_positions, p_step_basic_positions, p_team, p_velocity_positions,
             p_vertex_root, st_tether_particle, st_tether_team, t_cws, t_enabled, t_tether_compression,
             t_update_count, t_valid, n_tether):
    e = tid
    while e < n_tether:
        tm = st_tether_team[e]
        if team_frame_mask(t_enabled, t_valid, t_cws, tm) and t_update_count[tm] > _k:
            do_tether(e, st_tether_particle, p_team, p_next_positions,
                      p_velocity_positions, p_step_basic_positions, p_vertex_root,
                      t_tether_compression)
        e += stride


@phase('particles')
def phase_17(tid, stride, _k, power1, csr_distance_offsets, csr_distance_order, p_attr_move,
             p_base_positions, p_depth, p_friction, p_next_positions, p_team, sc_dcorr, st_distance_rest,
             st_distance_target, t_animation_pose_ratio, t_cws, t_distance_lut, t_enabled, t_init_scale,
             t_is_spring, t_scale_ratio, t_update_count, t_valid, num_particles):
    p = tid
    while p < num_particles:
        mt = p_team[p]
        if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
            do_distance_gather(p, p_team, p_next_positions, p_base_positions, p_depth,
                               p_friction, p_attr_move, t_is_spring, t_animation_pose_ratio,
                               t_init_scale, t_scale_ratio, t_distance_lut, power1,
                               csr_distance_offsets, csr_distance_order,
                               st_distance_target, st_distance_rest, sc_dcorr)
        p += stride


@phase('particles')
def phase_18(tid, stride, _k, p_next_positions, p_team, p_velocity_positions, sc_dcorr, t_cws, t_enabled,
             t_update_count, t_valid, num_particles):
    p = tid
    while p < num_particles:
        mt = p_team[p]
        if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
            p_next_positions[p, 0] += sc_dcorr[p, 0]
            p_next_positions[p, 1] += sc_dcorr[p, 1]
            p_next_positions[p, 2] += sc_dcorr[p, 2]
            p_velocity_positions[p, 0] += sc_dcorr[p, 0] * DISTANCE_VELOCITY_ATTENUATION
            p_velocity_positions[p, 1] += sc_dcorr[p, 1] * DISTANCE_VELOCITY_ATTENUATION
            p_velocity_positions[p, 2] += sc_dcorr[p, 2] * DISTANCE_VELOCITY_ATTENUATION
        p += stride


@phase('particles')
def phase_19(tid, stride, _k, baseline_entries, p_albuf_length, p_albuf_local_pos, p_albuf_local_rot,
             p_albuf_restore, p_albuf_rotation, p_next_positions, p_step_basic_positions,
             p_step_basic_rotations, p_team, p_vertex_parent, st_angle_buffered_particle,
             t_angle_use_limit, t_angle_use_restoration, t_cws, t_enabled, t_update_count, t_valid,
             n_angle_buffered, n_baseline):
    i = tid
    while i < n_baseline:
        v = baseline_entries[i]
        vt = p_team[v]
        if team_frame_mask(t_enabled, t_valid, t_cws, vt) and t_update_count[vt] > _k \
                and (t_angle_use_limit[vt] != 0 or t_angle_use_restoration[vt] != 0):
            p_albuf_rotation[v, 0] = p_step_basic_rotations[v, 0]
            p_albuf_rotation[v, 1] = p_step_basic_rotations[v, 1]
            p_albuf_rotation[v, 2] = p_step_basic_rotations[v, 2]
            p_albuf_rotation[v, 3] = p_step_basic_rotations[v, 3]
        i += stride
    i = tid
    while i < n_angle_buffered:
        v = st_angle_buffered_particle[i]
        vt = p_team[v]
        if team_frame_mask(t_enabled, t_valid, t_cws, vt) and t_update_count[vt] > _k \
                and (t_angle_use_limit[vt] != 0 or t_angle_use_restoration[vt] != 0):
            par = p_vertex_parent[v]
            bvx = p_step_basic_positions[v, 0] - p_step_basic_positions[par, 0]
            bvy = p_step_basic_positions[v, 1] - p_step_basic_positions[par, 1]
            bvz = p_step_basic_positions[v, 2] - p_step_basic_positions[par, 2]
            if t_angle_use_limit[vt] != 0:
                dvx = p_next_positions[v, 0] - p_next_positions[par, 0]
                dvy = p_next_positions[v, 1] - p_next_positions[par, 1]
                dvz = p_next_positions[v, 2] - p_next_positions[par, 2]
                avlen = dmath.length3(dvx, dvy, dvz)
                bvlen = dmath.length3(bvx, bvy, bvz)
                if avlen < EPSILON or bvlen < EPSILON:
                    p_albuf_length[v] = float32(0.0)
                    p_albuf_local_pos[v, 0] = float32(0.0)
                    p_albuf_local_pos[v, 1] = float32(0.0)
                    p_albuf_local_pos[v, 2] = float32(0.0)
                    p_albuf_local_rot[v, 0] = float32(0.0)
                    p_albuf_local_rot[v, 1] = float32(0.0)
                    p_albuf_local_rot[v, 2] = float32(0.0)
                    p_albuf_local_rot[v, 3] = float32(1.0)
                else:
                    safe_bv = bvlen if bvlen > float32(1e-30) else float32(1.0)
                    dirx = bvx / safe_bv
                    diry = bvy / safe_bv
                    dirz = bvz / safe_bv
                    ipx, ipy, ipz, ipw = dmath.quat_inverse(
                        p_step_basic_rotations[par, 0], p_step_basic_rotations[par, 1],
                        p_step_basic_rotations[par, 2], p_step_basic_rotations[par, 3])
                    lpx, lpy, lpz = dmath.quat_rotate(ipx, ipy, ipz, ipw, dirx, diry, dirz)
                    lrx, lry, lrz, lrw = dmath.quat_mul(
                        ipx, ipy, ipz, ipw,
                        p_step_basic_rotations[v, 0], p_step_basic_rotations[v, 1],
                        p_step_basic_rotations[v, 2], p_step_basic_rotations[v, 3])
                    p_albuf_length[v] = avlen
                    p_albuf_local_pos[v, 0] = lpx
                    p_albuf_local_pos[v, 1] = lpy
                    p_albuf_local_pos[v, 2] = lpz
                    p_albuf_local_rot[v, 0] = lrx
                    p_albuf_local_rot[v, 1] = lry
                    p_albuf_local_rot[v, 2] = lrz
                    p_albuf_local_rot[v, 3] = lrw
            if t_angle_use_restoration[vt] != 0:
                p_albuf_restore[v, 0] = bvx
                p_albuf_restore[v, 1] = bvy
                p_albuf_restore[v, 2] = bvz
        i += stride


@phase('single')
def phase_20(tid, bid, bdim, _k, power3, angle_pass_offsets, angle_pass_parents, angle_pass_vertices,
             p_albuf_length, p_albuf_local_pos, p_albuf_local_rot, p_albuf_restore, p_albuf_rotation,
             p_attr_move, p_depth, p_friction, p_next_positions, p_team, p_velocity_positions,
             t_angle_limit_lut, t_angle_limit_stiffness, t_angle_restoration_attenuation,
             t_angle_restoration_gravity_falloff, t_angle_restoration_lut, t_angle_use_limit,
             t_angle_use_restoration, t_cws, t_enabled, t_gravity_dot, t_update_count, t_valid,
             num_angle_passes):
    if bid == 0:
        for _ai in range(ANGLE_ITERATION):
            angle_rot_ratio = float32(0.1) + (float32(0.5) - float32(0.1)) \
                * (float32(_ai) / float32(2.0))
            for _ap in range(num_angle_passes):
                aps = angle_pass_offsets[_ap]
                ape = angle_pass_offsets[_ap + 1]
                e = aps + tid
                while e < ape:
                    v = angle_pass_vertices[e]
                    p = angle_pass_parents[e]
                    vt = p_team[v]
                    if team_frame_mask(t_enabled, t_valid, t_cws, vt) \
                            and t_update_count[vt] > _k:
                        ul = t_angle_use_limit[vt] != 0
                        ur = t_angle_use_restoration[vt] != 0
                        if ul or ur:
                            c_inv = float32(1.0) / (float32(1.0) + p_friction[v] * FRICTION_MASS)
                            p_inv = float32(1.0) / (float32(1.0) + p_friction[p] * FRICTION_MASS)
                            p_mv = p_attr_move[p] != 0
                            if ul:
                                do_angle_limit(v, p, vt, c_inv, p_inv, p_mv,
                                               p_next_positions, p_velocity_positions,
                                               p_albuf_rotation, p_albuf_local_pos,
                                               p_albuf_local_rot, p_albuf_length, p_depth,
                                               t_angle_limit_lut, t_angle_limit_stiffness)
                            if ur:
                                do_angle_restoration(v, p, vt, c_inv, p_inv, p_mv,
                                                     angle_rot_ratio, power3,
                                                     p_next_positions, p_velocity_positions,
                                                     p_albuf_restore, p_depth,
                                                     t_angle_restoration_lut,
                                                     t_angle_restoration_attenuation,
                                                     t_angle_restoration_gravity_falloff,
                                                     t_gravity_dot)
                    e += bdim
                cuda.syncthreads()


@phase('particles')
def phase_21(tid, stride, sc_dcorr_fixed, sc_dcount, num_particles):
    p = tid
    while p < num_particles:
        sc_dcorr_fixed[p, 0] = int32(0)
        sc_dcorr_fixed[p, 1] = int32(0)
        sc_dcorr_fixed[p, 2] = int32(0)
        sc_dcount[p] = int32(0)
        p += stride


@phase('bending')
def phase_22(tid, stride, _k, power1, p_attr_move, p_depth, p_friction, p_next_positions, sc_dcorr_fixed,
             sc_dcount, st_bending_pair, st_bending_rest, st_bending_sign, st_bending_team,
             t_bending_stiffness, t_cws, t_enabled, t_negative_scale_sign, t_scale_ratio, t_update_count,
             t_valid, n_bending):
    e = tid
    while e < n_bending:
        team = st_bending_team[e]
        if team_frame_mask(t_enabled, t_valid, t_cws, team) and t_update_count[team] > _k \
                and t_bending_stiffness[team] >= float32(1e-6):
            stiffness = dmath.saturate(t_bending_stiffness[team] * power1)
            pp0 = st_bending_pair[e, 0]
            pp1 = st_bending_pair[e, 1]
            pp2 = st_bending_pair[e, 2]
            pp3 = st_bending_pair[e, 3]
            rest = st_bending_rest[e]
            sgn = st_bending_sign[e]
            a0x = p_next_positions[pp0, 0]
            a0y = p_next_positions[pp0, 1]
            a0z = p_next_positions[pp0, 2]
            a1x = p_next_positions[pp1, 0]
            a1y = p_next_positions[pp1, 1]
            a1z = p_next_positions[pp1, 2]
            a2x = p_next_positions[pp2, 0]
            a2y = p_next_positions[pp2, 1]
            a2z = p_next_positions[pp2, 2]
            a3x = p_next_positions[pp3, 0]
            a3y = p_next_positions[pp3, 1]
            a3z = p_next_positions[pp3, 2]
            if p_attr_move[pp0] == 0:
                inv0 = BENDING_FIX_INV_MASS
            else:
                inv0 = dmath.calc_inverse_mass(p_friction[pp0], p_depth[pp0])
            if p_attr_move[pp1] == 0:
                inv1 = BENDING_FIX_INV_MASS
            else:
                inv1 = dmath.calc_inverse_mass(p_friction[pp1], p_depth[pp1])
            if p_attr_move[pp2] == 0:
                inv2 = BENDING_FIX_INV_MASS
            else:
                inv2 = dmath.calc_inverse_mass(p_friction[pp2], p_depth[pp2])
            if p_attr_move[pp3] == 0:
                inv3 = BENDING_FIX_INV_MASS
            else:
                inv3 = dmath.calc_inverse_mass(p_friction[pp3], p_depth[pp3])
            scale_ratio = t_scale_ratio[team]
            negative_sign = t_negative_scale_sign[team]
            result = False
            a0dx = float32(0.0)
            a0dy = float32(0.0)
            a0dz = float32(0.0)
            a1dx = float32(0.0)
            a1dy = float32(0.0)
            a1dz = float32(0.0)
            a2dx = float32(0.0)
            a2dy = float32(0.0)
            a2dz = float32(0.0)
            a3dx = float32(0.0)
            a3dy = float32(0.0)
            a3dz = float32(0.0)
            if sgn == VOLUME_SIGN:
                volume_rest = rest * scale_ratio * negative_sign
                cx, cy, cz = dmath.cross3(a1x - a0x, a1y - a0y, a1z - a0z,
                                          a2x - a0x, a2y - a0y, a2z - a0z)
                volume = ONE_SIXTH * (cx * (a3x - a0x) + cy * (a3y - a0y)
                                      + cz * (a3z - a0z)) * VOLUME_SCALE
                g0x, g0y, g0z = dmath.cross3(a1x - a2x, a1y - a2y, a1z - a2z,
                                             a3x - a2x, a3y - a2y, a3z - a2z)
                g1x, g1y, g1z = dmath.cross3(a2x - a0x, a2y - a0y, a2z - a0z,
                                             a3x - a0x, a3y - a0y, a3z - a0z)
                g2x, g2y, g2z = dmath.cross3(a0x - a1x, a0y - a1y, a0z - a1z,
                                             a3x - a1x, a3y - a1y, a3z - a1z)
                g3x, g3y, g3z = dmath.cross3(a1x - a0x, a1y - a0y, a1z - a0z,
                                             a2x - a0x, a2y - a0y, a2z - a0z)
                lam = (inv0 * (g0x * g0x + g0y * g0y + g0z * g0z)
                       + inv1 * (g1x * g1x + g1y * g1y + g1z * g1z)
                       + inv2 * (g2x * g2x + g2y * g2y + g2z * g2z)
                       + inv3 * (g3x * g3x + g3y * g3y + g3z * g3z))
                lam = lam * VOLUME_SCALE
                if libdevice.fabsf(lam) >= float32(1e-6):
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
                rest_angle = rest * float32(sgn) * negative_sign
                ex = a3x - a2x
                ey = a3y - a2y
                ez = a3z - a2z
                elen = dmath.length3(ex, ey, ez)
                ok = elen >= float32(1e-8)
                safe_elen = elen if elen > float32(1e-30) else float32(1.0)
                inv_elen = float32(1.0) / safe_elen
                nn1x, nn1y, nn1z = dmath.cross3(a2x - a0x, a2y - a0y, a2z - a0z,
                                                a3x - a0x, a3y - a0y, a3z - a0z)
                nn2x, nn2y, nn2z = dmath.cross3(a3x - a1x, a3y - a1y, a3z - a1z,
                                                a2x - a1x, a2y - a1y, a2z - a1z)
                sq1 = nn1x * nn1x + nn1y * nn1y + nn1z * nn1z
                sq2 = nn2x * nn2x + nn2y * nn2y + nn2z * nn2z
                ok = ok and (sq1 != float32(0.0)) and (sq2 != float32(0.0))
                safe_sq1 = sq1 if sq1 > float32(1e-30) else float32(1.0)
                safe_sq2 = sq2 if sq2 > float32(1e-30) else float32(1.0)
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
                un1x, un1y, un1z = dmath.normalize3(nn1x, nn1y, nn1z)
                un2x, un2y, un2z = dmath.normalize3(nn2x, nn2y, nn2z)
                dotu = dmath.clamp1(un1x * un2x + un1y * un2y + un1z * un2z)
                phi = libdevice.acosf(dotu)
                lam = (inv0 * (d0x * d0x + d0y * d0y + d0z * d0z)
                       + inv1 * (d1x * d1x + d1y * d1y + d1z * d1z)
                       + inv2 * (d2x * d2x + d2y * d2y + d2z * d2z)
                       + inv3 * (d3x * d3x + d3y * d3y + d3z * d3z))
                ok = ok and (lam != float32(0.0))
                crx, cry, crz = dmath.cross3(un1x, un1y, un1z, un2x, un2y, un2z)
                dir_sign = dmath.fsign(crx * ex + cry * ey + crz * ez)
                phi = phi * dir_sign
                if ok:
                    lam = (rest_angle - phi) / lam * stiffness
                    a0dx = -inv0 * lam * d0x
                    a0dy = -inv0 * lam * d0y
                    a0dz = -inv0 * lam * d0z
                    a1dx = -inv1 * lam * d1x
                    a1dy = -inv1 * lam * d1y
                    a1dz = -inv1 * lam * d1z
                    a2dx = -inv2 * lam * d2x
                    a2dy = -inv2 * lam * d2y
                    a2dz = -inv2 * lam * d2z
                    a3dx = -inv3 * lam * d3x
                    a3dy = -inv3 * lam * d3y
                    a3dz = -inv3 * lam * d3z
                    result = True
            if result:
                cuda.atomic.add(sc_dcorr_fixed, (pp0, 0), int32(a0dx * TO_FIXED))
                cuda.atomic.add(sc_dcorr_fixed, (pp0, 1), int32(a0dy * TO_FIXED))
                cuda.atomic.add(sc_dcorr_fixed, (pp0, 2), int32(a0dz * TO_FIXED))
                cuda.atomic.add(sc_dcount, pp0, int32(1))
                cuda.atomic.add(sc_dcorr_fixed, (pp1, 0), int32(a1dx * TO_FIXED))
                cuda.atomic.add(sc_dcorr_fixed, (pp1, 1), int32(a1dy * TO_FIXED))
                cuda.atomic.add(sc_dcorr_fixed, (pp1, 2), int32(a1dz * TO_FIXED))
                cuda.atomic.add(sc_dcount, pp1, int32(1))
                cuda.atomic.add(sc_dcorr_fixed, (pp2, 0), int32(a2dx * TO_FIXED))
                cuda.atomic.add(sc_dcorr_fixed, (pp2, 1), int32(a2dy * TO_FIXED))
                cuda.atomic.add(sc_dcorr_fixed, (pp2, 2), int32(a2dz * TO_FIXED))
                cuda.atomic.add(sc_dcount, pp2, int32(1))
                cuda.atomic.add(sc_dcorr_fixed, (pp3, 0), int32(a3dx * TO_FIXED))
                cuda.atomic.add(sc_dcorr_fixed, (pp3, 1), int32(a3dy * TO_FIXED))
                cuda.atomic.add(sc_dcorr_fixed, (pp3, 2), int32(a3dz * TO_FIXED))
                cuda.atomic.add(sc_dcount, pp3, int32(1))
        e += stride


@phase('particles')
def phase_23(tid, stride, _k, p_attr_move, p_next_positions, p_team, sc_dcorr_fixed, sc_dcount, t_cws,
             t_enabled, t_update_count, t_valid, num_particles):
    p = tid
    while p < num_particles:
        mt = p_team[p]
        if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k \
                and p_attr_move[p] != 0 and sc_dcount[p] > 0:
            inv_c = float32(1.0) / float32(sc_dcount[p])
            p_next_positions[p, 0] += float32(sc_dcorr_fixed[p, 0]) / TO_FIXED * inv_c
            p_next_positions[p, 1] += float32(sc_dcorr_fixed[p, 1]) / TO_FIXED * inv_c
            p_next_positions[p, 2] += float32(sc_dcorr_fixed[p, 2]) / TO_FIXED * inv_c
        p += stride


@phase('particles')
def phase_24(tid, stride, _k, c_active, c_kind, c_work_aabb_max, c_work_aabb_min, c_work_inv_old_rot,
             c_work_next_pos, c_work_old_pos, c_work_radius, c_work_rot, csr_point_pair_offsets,
             csr_point_pair_order, p_base_positions, p_collision_normals, p_depth, p_friction,
             p_next_positions, p_team, p_velocity_positions, st_point_pair_collider, t_collision_mode,
             t_cws, t_enabled, t_is_spring, t_limit_distance_lut, t_radius_lut, t_scale_ratio,
             t_update_count, t_valid, num_particles):
    p = tid
    while p < num_particles:
        mt = p_team[p]
        if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
            do_solve_point(p, p_team, p_next_positions, p_base_positions, p_depth,
                           p_friction, p_collision_normals, p_velocity_positions,
                           t_collision_mode, t_radius_lut, t_scale_ratio, t_is_spring,
                           t_limit_distance_lut, c_kind, c_active, c_work_old_pos,
                           c_work_next_pos, c_work_radius, c_work_inv_old_rot, c_work_rot,
                           c_work_aabb_min, c_work_aabb_max,
                           csr_point_pair_offsets, csr_point_pair_order,
                           st_point_pair_collider)
        p += stride


@phase('particles')
def phase_25(tid, stride, sc_col_friction_fixed, sc_col_normal_fixed, sc_dcorr_fixed, sc_dcount,
             num_particles):
    p = tid
    while p < num_particles:
        sc_dcorr_fixed[p, 0] = int32(0)
        sc_dcorr_fixed[p, 1] = int32(0)
        sc_dcorr_fixed[p, 2] = int32(0)
        sc_dcount[p] = int32(0)
        sc_col_friction_fixed[p] = int32(0)
        sc_col_normal_fixed[p, 0] = int32(0)
        sc_col_normal_fixed[p, 1] = int32(0)
        sc_col_normal_fixed[p, 2] = int32(0)
        p += stride


@phase('collision_edges')
def phase_26(tid, stride, _k, c_active, c_kind, c_work_aabb_max, c_work_aabb_min, c_work_next_pos,
             c_work_old_pos, c_work_radius, csr_edge_pair_offsets, csr_edge_pair_order, p_attr_move,
             p_depth, p_next_positions, p_team, sc_col_friction_fixed, sc_col_normal_fixed, sc_dcorr_fixed,
             sc_dcount, st_collision_edge, st_edge_pair_collider, t_collision_mode, t_cws, t_enabled,
             t_radius_lut, t_scale_ratio, t_update_count, t_valid):
    ee = tid
    while ee < st_collision_edge.shape[0]:
        et = p_team[st_collision_edge[ee, 0]]
        if team_frame_mask(t_enabled, t_valid, t_cws, et) and t_update_count[et] > _k \
                and t_collision_mode[et] == COLLISION_EDGE:
            do_solve_edge(ee, p_team, p_next_positions, p_depth, p_attr_move,
                          t_radius_lut, t_scale_ratio, c_kind, c_active, c_work_old_pos,
                          c_work_next_pos, c_work_radius, c_work_aabb_min, c_work_aabb_max,
                          csr_edge_pair_offsets, csr_edge_pair_order, st_edge_pair_collider,
                          st_collision_edge, sc_dcorr_fixed, sc_dcount,
                          sc_col_friction_fixed, sc_col_normal_fixed)
        ee += stride


@phase('particles')
def phase_27(tid, stride, _k, p_collision_normals, p_friction, p_next_positions, p_team,
             sc_col_friction_fixed, sc_col_normal_fixed, sc_dcorr_fixed, sc_dcount, t_collision_mode,
             t_cws, t_enabled, t_update_count, t_valid, num_particles):
    p = tid
    while p < num_particles:
        mt = p_team[p]
        if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k \
                and t_collision_mode[mt] == COLLISION_EDGE:
            cnt = sc_dcount[p]
            if cnt > int32(0):
                inv_e = float32(1.0) / float32(cnt)
                p_next_positions[p, 0] += float32(sc_dcorr_fixed[p, 0]) / TO_FIXED * inv_e
                p_next_positions[p, 1] += float32(sc_dcorr_fixed[p, 1]) / TO_FIXED * inv_e
                p_next_positions[p, 2] += float32(sc_dcorr_fixed[p, 2]) / TO_FIXED * inv_e
            ef = float32(sc_col_friction_fixed[p]) / TO_FIXED
            if ef > p_friction[p]:
                p_friction[p] = ef
            enx = float32(sc_col_normal_fixed[p, 0]) / TO_FIXED
            eny = float32(sc_col_normal_fixed[p, 1]) / TO_FIXED
            enz = float32(sc_col_normal_fixed[p, 2]) / TO_FIXED
            if (enx * enx + eny * eny + enz * enz) > float32(0.0):
                onx, ony, onz = dmath.normalize3(enx, eny, enz)
                p_collision_normals[p, 0] = onx
                p_collision_normals[p, 1] = ony
                p_collision_normals[p, 2] = onz
        p += stride


@phase('particles')
def phase_28(tid, stride, _k, power1, csr_distance_offsets, csr_distance_order, p_attr_move,
             p_base_positions, p_depth, p_friction, p_next_positions, p_team, sc_dcorr, st_distance_rest,
             st_distance_target, t_animation_pose_ratio, t_cws, t_distance_lut, t_enabled, t_init_scale,
             t_is_spring, t_scale_ratio, t_update_count, t_valid, num_particles):
    p = tid
    while p < num_particles:
        mt = p_team[p]
        if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
            do_distance_gather(p, p_team, p_next_positions, p_base_positions, p_depth,
                               p_friction, p_attr_move, t_is_spring, t_animation_pose_ratio,
                               t_init_scale, t_scale_ratio, t_distance_lut, power1,
                               csr_distance_offsets, csr_distance_order,
                               st_distance_target, st_distance_rest, sc_dcorr)
        p += stride


@phase('particles')
def phase_29(tid, stride, _k, p_next_positions, p_team, p_velocity_positions, sc_dcorr, t_cws, t_enabled,
             t_update_count, t_valid, num_particles):
    p = tid
    while p < num_particles:
        mt = p_team[p]
        if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
            p_next_positions[p, 0] += sc_dcorr[p, 0]
            p_next_positions[p, 1] += sc_dcorr[p, 1]
            p_next_positions[p, 2] += sc_dcorr[p, 2]
            p_velocity_positions[p, 0] += sc_dcorr[p, 0] * DISTANCE_VELOCITY_ATTENUATION
            p_velocity_positions[p, 1] += sc_dcorr[p, 1] * DISTANCE_VELOCITY_ATTENUATION
            p_velocity_positions[p, 2] += sc_dcorr[p, 2] * DISTANCE_VELOCITY_ATTENUATION
        p += stride


@phase('particles')
def phase_30(tid, stride, _k, p_base_positions, p_base_rotations, p_depth, p_next_positions,
             p_velocity_positions, st_motion_particle, st_motion_team, t_cws, t_enabled,
             t_motion_backstop_lut, t_motion_backstop_radius, t_motion_max_distance_lut,
             t_motion_stiffness, t_motion_use_backstop, t_motion_use_max_distance, t_normal_axis_vector,
             t_radius_lut, t_update_count, t_valid, n_motion):
    e = tid
    while e < n_motion:
        mt = st_motion_team[e]
        use_max = t_motion_use_max_distance[mt] != 0
        use_backstop = t_motion_use_backstop[mt] != 0
        if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k \
                and (use_max or use_backstop):
            index = st_motion_particle[e]
            stiffness = t_motion_stiffness[mt]
            backstop_radius = t_motion_backstop_radius[mt]
            o0 = p_next_positions[index, 0]
            o1 = p_next_positions[index, 1]
            o2 = p_next_positions[index, 2]
            n0 = o0
            n1 = o1
            n2 = o2
            b0 = p_base_positions[index, 0]
            b1 = p_base_positions[index, 1]
            b2 = p_base_positions[index, 2]
            depth = p_depth[index]
            radius = dmath.evaluate_team_lut(t_radius_lut, mt, depth)
            if radius < float32(0.0001):
                radius = float32(0.0001)
            cfr = radius
            depth2 = depth * depth
            dirx, diry, dirz = dmath.quat_rotate(
                p_base_rotations[index, 0], p_base_rotations[index, 1],
                p_base_rotations[index, 2], p_base_rotations[index, 3],
                t_normal_axis_vector[mt, 0], t_normal_axis_vector[mt, 1],
                t_normal_axis_vector[mt, 2])
            if use_max:
                max_distance = dmath.evaluate_team_lut(t_motion_max_distance_lut, mt, depth2)
                cvx, cvy, cvz = dmath.clamp_vector(n0 - b0, n1 - b1, n2 - b2, max_distance)
                n0 = b0 + cvx
                n1 = b1 + cvy
                n2 = b2 + cvz
            if use_backstop and backstop_radius > float32(0.0):
                backstop_distance = dmath.evaluate_team_lut(t_motion_backstop_lut, mt, depth2)
                off = backstop_distance + backstop_radius
                cx = b0 - dirx * off
                cy = b1 - diry * off
                cz = b2 - dirz * off
                vx = n0 - cx
                vy = n1 - cy
                vz = n2 - cz
                l = dmath.length3(vx, vy, vz)
                near = (l > EPSILON) and (l < backstop_radius + cfr)
                if near and (l < backstop_radius):
                    safe_l = l if l > float32(1e-30) else float32(1.0)
                    n0 = cx + vx / safe_l * backstop_radius
                    n1 = cy + vy / safe_l * backstop_radius
                    n2 = cz + vz / safe_l * backstop_radius
            n0 = dmath.lerp(o0, n0, stiffness)
            n1 = dmath.lerp(o1, n1, stiffness)
            n2 = dmath.lerp(o2, n2, stiffness)
            p_next_positions[index, 0] = n0
            p_next_positions[index, 1] = n1
            p_next_positions[index, 2] = n2
            p_velocity_positions[index, 0] += (n0 - o0) * float32(0.95)
            p_velocity_positions[index, 1] += (n1 - o1) * float32(0.95)
            p_velocity_positions[index, 2] += (n2 - o2) * float32(0.95)
        e += stride


@phase('teams')
def phase_33(tid, stride, _k, scl_max_fixed, t_cws, t_enabled, t_update_count, t_valid, num_teams):
    i = tid
    while i < num_teams:
        if team_frame_mask(t_enabled, t_valid, t_cws, i) and t_update_count[i] > _k:
            scl_max_fixed[i] = int32(0)
        i += stride


@phase('self_primitives')
def phase_34(tid, stride, _k, p_friction, p_intersect_flag, p_next_positions, p_old_positions, scl_counts,
             scl_max_fixed, sfe_aabb_max, sfe_aabb_min, sfe_fix, sfe_ignore, sfe_intersect, sfe_inv_mass,
             sfe_particles, sfe_prim_depth, sfe_team, sfe_thickness, sfe_use, sfp_aabb_max, sfp_aabb_min,
             sfp_fix, sfp_ignore, sfp_intersect, sfp_inv_mass, sfp_particles, sfp_prim_depth, sfp_team,
             sfp_thickness, sfp_use, sft_aabb_max, sft_aabb_min, sft_fix, sft_ignore, sft_intersect,
             sft_inv_mass, sft_particles, sft_prim_depth, sft_team, sft_thickness, sft_use, t_cws,
             t_enabled, t_scale_ratio, t_self_cloth_mass, t_self_thickness_lut, t_update_count, t_use_edge,
             t_use_point, t_use_triangle, t_valid, num_self_edges, num_self_points, num_self_triangles):
    q = tid
    while q < num_self_points:
        do_self_update_primitive(
            q, 1, sfp_team, sfp_particles, sfp_fix, sfp_ignore, sfp_prim_depth,
            sfp_inv_mass, sfp_thickness, sfp_aabb_min, sfp_aabb_max, sfp_intersect,
            sfp_use, t_use_point, t_self_thickness_lut, t_self_cloth_mass, t_scale_ratio,
            t_enabled, t_valid, t_cws, t_update_count, p_next_positions, p_old_positions,
            p_friction, p_intersect_flag, scl_counts, scl_max_fixed, _k)
        q += stride
    q = tid
    while q < num_self_edges:
        do_self_update_primitive(
            q, 2, sfe_team, sfe_particles, sfe_fix, sfe_ignore, sfe_prim_depth,
            sfe_inv_mass, sfe_thickness, sfe_aabb_min, sfe_aabb_max, sfe_intersect,
            sfe_use, t_use_edge, t_self_thickness_lut, t_self_cloth_mass, t_scale_ratio,
            t_enabled, t_valid, t_cws, t_update_count, p_next_positions, p_old_positions,
            p_friction, p_intersect_flag, scl_counts, scl_max_fixed, _k)
        q += stride
    q = tid
    while q < num_self_triangles:
        do_self_update_primitive(
            q, 3, sft_team, sft_particles, sft_fix, sft_ignore, sft_prim_depth,
            sft_inv_mass, sft_thickness, sft_aabb_min, sft_aabb_max, sft_intersect,
            sft_use, t_use_triangle, t_self_thickness_lut, t_self_cloth_mass, t_scale_ratio,
            t_enabled, t_valid, t_cws, t_update_count, p_next_positions, p_old_positions,
            p_friction, p_intersect_flag, scl_counts, scl_max_fixed, _k)
        q += stride


@phase('teams')
def phase_35(tid, stride, _k, scl_max_fixed, t_cws, t_enabled, t_self_grid_size, t_self_max_primitive_size,
             t_update_count, t_valid, num_teams):
    i = tid
    while i < num_teams:
        if team_frame_mask(t_enabled, t_valid, t_cws, i) and t_update_count[i] > _k:
            ms = float32(scl_max_fixed[i]) / TO_FIXED
            t_self_max_primitive_size[i] = ms
            t_self_grid_size[i] = ms * SELF_COLLISION_UNIFORM_GRID_SCALE
        i += stride


@phase('single')
def phase_36(tid, scl_counts):
    if tid == 0:
        scl_counts[SCL_EE_COUNT] = int32(0)
        scl_counts[SCL_PT_COUNT] = int32(0)


@phase('self_pairs')
def phase_37(tid, stride, ct_kind, ct_my_start, ct_pair_off, ct_same, ct_tgt_count, ct_tgt_start,
             ct_tgt_team, ee_enable, ee_my, ee_n, ee_s, ee_t, ee_target, ee_thickness, p_next_positions,
             p_old_positions, pt_enable, pt_my, pt_sign, pt_target, pt_thickness, scl_counts, sfe_aabb_max,
             sfe_aabb_min, sfe_all_fix, sfe_ignore, sfe_particles, sfe_thickness, sfe_use, sfp_aabb_max,
             sfp_aabb_min, sfp_all_fix, sfp_ignore, sfp_particles, sfp_thickness, sfp_use, sft_aabb_max,
             sft_aabb_min, sft_all_fix, sft_ignore, sft_particles, sft_thickness, sft_use,
             t_self_grid_size, ee_cap, num_ct_slots, pt_cap):
    total_ct = ct_pair_off[num_ct_slots]
    g = tid
    while g < total_ct:
        lo = int32(0)
        hi = num_ct_slots
        while lo < hi:
            mid = (lo + hi) >> 1
            if ct_pair_off[mid + 1] <= g:
                lo = mid + 1
            else:
                hi = mid
        task = lo
        tgt_team = ct_tgt_team[task]
        if t_self_grid_size[tgt_team] > EPSILON:
            tgt_count = ct_tgt_count[task]
            local = g - ct_pair_off[task]
            i = local // tgt_count
            j = local % tgt_count
            my_prim = ct_my_start[task] + i
            tgt_prim = ct_tgt_start[task] + j
            same = ct_same[task]
            if ct_kind[task] == 0:
                if (sfe_use[my_prim] != 0 and sfe_ignore[my_prim] == 0
                        and sfe_use[tgt_prim] != 0 and sfe_ignore[tgt_prim] == 0
                        and (same == 0 or my_prim < tgt_prim)
                        and self_aabb_overlap(sfe_aabb_min, sfe_aabb_max, my_prim,
                                              sfe_aabb_min, sfe_aabb_max, tgt_prim)
                        and not (sfe_all_fix[my_prim] != 0 and sfe_all_fix[tgt_prim] != 0)):
                    if (same == 0) or (not self_connection_shared(
                            sfe_particles, my_prim, sfe_particles, tgt_prim)):
                        thk = sfe_thickness[my_prim] + sfe_thickness[tgt_prim]
                        s, t, nx, ny, nz, enable = self_ee_geometry(
                            my_prim, tgt_prim, thk, sfe_particles,
                            p_next_positions, p_old_positions)
                        if enable:
                            idx = cuda.atomic.add(scl_counts, SCL_EE_COUNT, 1)
                            if idx < ee_cap:
                                ee_my[idx] = my_prim
                                ee_target[idx] = tgt_prim
                                ee_thickness[idx] = thk
                                ee_s[idx] = s
                                ee_t[idx] = t
                                ee_n[idx, 0] = nx
                                ee_n[idx, 1] = ny
                                ee_n[idx, 2] = nz
                                ee_enable[idx] = uint8(1)
                            else:
                                scl_counts[SCL_ERROR] = int32(1)
            else:
                if (sfp_use[my_prim] != 0 and sfp_ignore[my_prim] == 0
                        and sft_use[tgt_prim] != 0 and sft_ignore[tgt_prim] == 0
                        and self_aabb_overlap(sfp_aabb_min, sfp_aabb_max, my_prim,
                                              sft_aabb_min, sft_aabb_max, tgt_prim)
                        and not (sfp_all_fix[my_prim] != 0 and sft_all_fix[tgt_prim] != 0)):
                    if (same == 0) or (not self_connection_shared(
                            sfp_particles, my_prim, sft_particles, tgt_prim)):
                        thk = sfp_thickness[my_prim] + sft_thickness[tgt_prim]
                        enable, sign = self_pt_geometry(
                            my_prim, tgt_prim, thk, True, sfp_particles,
                            sft_particles, p_next_positions, p_old_positions)
                        if enable:
                            idx = cuda.atomic.add(scl_counts, SCL_PT_COUNT, 1)
                            if idx < pt_cap:
                                pt_my[idx] = my_prim
                                pt_target[idx] = tgt_prim
                                pt_thickness[idx] = thk
                                pt_sign[idx] = sign
                                pt_enable[idx] = uint8(1)
                            else:
                                scl_counts[SCL_ERROR] = int32(1)
        g += stride


@phase('self_contacts')
def phase_38(tid, stride, ee_enable, ee_my, ee_n, ee_s, ee_t, ee_target, ee_thickness, p_next_positions,
             p_old_positions, pt_enable, pt_my, pt_target, pt_thickness, scl_counts, sfe_particles,
             sfp_particles, sft_particles, ee_cap, pt_cap):
    ee_count = scl_counts[SCL_EE_COUNT]
    ee_lim = ee_count if ee_count < ee_cap else ee_cap
    e = tid
    while e < ee_lim:
        s, t, nx, ny, nz, enable = self_ee_geometry(
            ee_my[e], ee_target[e], ee_thickness[e], sfe_particles,
            p_next_positions, p_old_positions)
        ee_s[e] = s
        ee_t[e] = t
        ee_n[e, 0] = nx
        ee_n[e, 1] = ny
        ee_n[e, 2] = nz
        ee_enable[e] = uint8(1) if enable else uint8(0)
        e += stride
    pt_count = scl_counts[SCL_PT_COUNT]
    pt_lim = pt_count if pt_count < pt_cap else pt_cap
    e = tid
    while e < pt_lim:
        enable, _sign = self_pt_geometry(
            pt_my[e], pt_target[e], pt_thickness[e], False, sfp_particles,
            sft_particles, p_next_positions, p_old_positions)
        pt_enable[e] = uint8(1) if enable else uint8(0)
        e += stride


@phase('particles')
def phase_39(tid, stride, sc_dcorr_fixed, sc_dcount, scl_counts, ee_cap, num_particles, pt_cap):
    ee_count2 = scl_counts[SCL_EE_COUNT]
    ee_lim2 = ee_count2 if ee_count2 < ee_cap else ee_cap
    pt_count2 = scl_counts[SCL_PT_COUNT]
    pt_lim2 = pt_count2 if pt_count2 < pt_cap else pt_cap
    p = tid
    while p < num_particles:
        sc_dcorr_fixed[p, 0] = int32(0)
        sc_dcorr_fixed[p, 1] = int32(0)
        sc_dcorr_fixed[p, 2] = int32(0)
        sc_dcount[p] = int32(0)
        p += stride


@phase('self_contacts')
def phase_40(tid, stride, ee_enable, ee_my, ee_n, ee_s, ee_t, ee_target, ee_thickness, p_next_positions,
             pt_enable, pt_my, pt_sign, pt_target, pt_thickness, sc_dcorr_fixed, sc_dcount, scl_counts,
             sfe_fix, sfe_intersect, sfe_inv_mass, sfe_particles, sfp_fix, sfp_intersect, sfp_inv_mass,
             sfp_particles, sft_fix, sft_intersect, sft_inv_mass, sft_particles, ee_cap, pt_cap):
    ee_count2 = scl_counts[SCL_EE_COUNT]
    ee_lim2 = ee_count2 if ee_count2 < ee_cap else ee_cap
    pt_count2 = scl_counts[SCL_PT_COUNT]
    pt_lim2 = pt_count2 if pt_count2 < pt_cap else pt_cap
    e = tid
    while e < ee_lim2:
        if ee_enable[e] != 0:
            my = ee_my[e]
            tgt = ee_target[e]
            s = ee_s[e]
            t = ee_t[e]
            nx = ee_n[e, 0]
            ny = ee_n[e, 1]
            nz = ee_n[e, 2]
            thk = ee_thickness[e]
            a0 = sfe_particles[my, 0]
            a1 = sfe_particles[my, 1]
            b0 = sfe_particles[tgt, 0]
            b1 = sfe_particles[tgt, 1]
            ax = dmath.lerp(p_next_positions[a0, 0], p_next_positions[a1, 0], s)
            ay = dmath.lerp(p_next_positions[a0, 1], p_next_positions[a1, 1], s)
            az = dmath.lerp(p_next_positions[a0, 2], p_next_positions[a1, 2], s)
            bx = dmath.lerp(p_next_positions[b0, 0], p_next_positions[b1, 0], t)
            by = dmath.lerp(p_next_positions[b0, 1], p_next_positions[b1, 1], t)
            bz = dmath.lerp(p_next_positions[b0, 2], p_next_positions[b1, 2], t)
            l = nx * (ax - bx) + ny * (ay - by) + nz * (az - bz)
            c = thk - l
            bb0 = float32(1.0) - s
            bb1 = s
            bb2 = float32(1.0) - t
            bb3 = t
            im0 = sfe_inv_mass[my, 0]
            im1 = sfe_inv_mass[my, 1]
            im20 = sfe_inv_mass[tgt, 0]
            im21 = sfe_inv_mass[tgt, 1]
            denom = im0 * bb0 * bb0 + im1 * bb1 * bb1 + im20 * bb2 * bb2 + im21 * bb3 * bb3
            if l <= thk and denom != float32(0.0):
                scale = c / denom
                s0 = scale * im0 * bb0
                s1 = scale * im1 * bb1
                s2 = scale * im20 * bb2
                s3 = scale * im21 * bb3
                fm = sfe_fix[my]
                imk = sfe_intersect[my]
                fmt = sfe_fix[tgt]
                imt = sfe_intersect[tgt]
                if ((fm >> 0) & 1) == 0 and ((imk >> 0) & 1) == 0:
                    cuda.atomic.add(sc_dcorr_fixed, (a0, 0), int32(nx * s0 * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (a0, 1), int32(ny * s0 * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (a0, 2), int32(nz * s0 * TO_FIXED))
                    cuda.atomic.add(sc_dcount, a0, 1)
                if ((fm >> 1) & 1) == 0 and ((imk >> 1) & 1) == 0:
                    cuda.atomic.add(sc_dcorr_fixed, (a1, 0), int32(nx * s1 * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (a1, 1), int32(ny * s1 * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (a1, 2), int32(nz * s1 * TO_FIXED))
                    cuda.atomic.add(sc_dcount, a1, 1)
                if ((fmt >> 0) & 1) == 0 and ((imt >> 0) & 1) == 0:
                    cuda.atomic.add(sc_dcorr_fixed, (b0, 0), int32(-nx * s2 * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (b0, 1), int32(-ny * s2 * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (b0, 2), int32(-nz * s2 * TO_FIXED))
                    cuda.atomic.add(sc_dcount, b0, 1)
                if ((fmt >> 1) & 1) == 0 and ((imt >> 1) & 1) == 0:
                    cuda.atomic.add(sc_dcorr_fixed, (b1, 0), int32(-nx * s3 * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (b1, 1), int32(-ny * s3 * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (b1, 2), int32(-nz * s3 * TO_FIXED))
                    cuda.atomic.add(sc_dcount, b1, 1)
        e += stride
    e = tid
    while e < pt_lim2:
        if pt_enable[e] != 0:
            my = pt_my[e]
            tgt = pt_target[e]
            sign = pt_sign[e]
            thk = pt_thickness[e]
            pp = sfp_particles[my, 0]
            t0 = sft_particles[tgt, 0]
            t1 = sft_particles[tgt, 1]
            t2 = sft_particles[tgt, 2]
            npx = p_next_positions[pp, 0]
            npy = p_next_positions[pp, 1]
            npz = p_next_positions[pp, 2]
            t0x = p_next_positions[t0, 0]
            t0y = p_next_positions[t0, 1]
            t0z = p_next_positions[t0, 2]
            t1x = p_next_positions[t1, 0]
            t1y = p_next_positions[t1, 1]
            t1z = p_next_positions[t1, 2]
            t2x = p_next_positions[t2, 0]
            t2y = p_next_positions[t2, 1]
            t2z = p_next_positions[t2, 2]
            tnx, tny, tnz = dmath.triangle_normal(t0x, t0y, t0z, t1x, t1y, t1z,
                                                 t2x, t2y, t2z)
            nx = tnx * sign
            ny = tny * sign
            nz = tnz * sign
            dist = nx * (npx - t0x) + ny * (npy - t0y) + nz * (npz - t0z)
            _cx, _cy, _cz, u, v, w = dmath.closest_pt_point_triangle(
                npx, npy, npz, t0x, t0y, t0z, t1x, t1y, t1z, t2x, t2y, t2z)
            c = dist - thk
            imp = sfp_inv_mass[my, 0]
            imt0 = sft_inv_mass[tgt, 0]
            imt1 = sft_inv_mass[tgt, 1]
            imt2 = sft_inv_mass[tgt, 2]
            denom = imp + imt0 * u * u + imt1 * v * v + imt2 * w * w
            if dist < thk and denom != float32(0.0):
                scale = c / denom
                sp = scale * imp
                st0 = scale * imt0 * u
                st1 = scale * imt1 * v
                st2 = scale * imt2 * w
                fp = sfp_fix[my]
                ipk = sfp_intersect[my]
                ft = sft_fix[tgt]
                itk = sft_intersect[tgt]
                if ((fp >> 0) & 1) == 0 and ((ipk >> 0) & 1) == 0:
                    cuda.atomic.add(sc_dcorr_fixed, (pp, 0), int32(-nx * sp * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (pp, 1), int32(-ny * sp * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (pp, 2), int32(-nz * sp * TO_FIXED))
                    cuda.atomic.add(sc_dcount, pp, 1)
                if ((ft >> 0) & 1) == 0 and ((itk >> 0) & 1) == 0:
                    cuda.atomic.add(sc_dcorr_fixed, (t0, 0), int32(nx * st0 * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (t0, 1), int32(ny * st0 * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (t0, 2), int32(nz * st0 * TO_FIXED))
                    cuda.atomic.add(sc_dcount, t0, 1)
                if ((ft >> 1) & 1) == 0 and ((itk >> 1) & 1) == 0:
                    cuda.atomic.add(sc_dcorr_fixed, (t1, 0), int32(nx * st1 * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (t1, 1), int32(ny * st1 * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (t1, 2), int32(nz * st1 * TO_FIXED))
                    cuda.atomic.add(sc_dcount, t1, 1)
                if ((ft >> 2) & 1) == 0 and ((itk >> 2) & 1) == 0:
                    cuda.atomic.add(sc_dcorr_fixed, (t2, 0), int32(nx * st2 * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (t2, 1), int32(ny * st2 * TO_FIXED))
                    cuda.atomic.add(sc_dcorr_fixed, (t2, 2), int32(nz * st2 * TO_FIXED))
                    cuda.atomic.add(sc_dcount, t2, 1)
        e += stride


@phase('particles')
def phase_41(tid, stride, _k, p_next_positions, p_team, sc_dcorr_fixed, sc_dcount, t_cws, t_enabled,
             t_update_count, t_valid, num_particles):
    p = tid
    while p < num_particles:
        cnt = sc_dcount[p]
        if cnt > 0:
            mt = p_team[p]
            if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
                inv = float32(1.0) / float32(cnt)
                p_next_positions[p, 0] += float32(sc_dcorr_fixed[p, 0]) / TO_FIXED * inv
                p_next_positions[p, 1] += float32(sc_dcorr_fixed[p, 1]) / TO_FIXED * inv
                p_next_positions[p, 2] += float32(sc_dcorr_fixed[p, 2]) / TO_FIXED * inv
        sc_dcorr_fixed[p, 0] = int32(0)
        sc_dcorr_fixed[p, 1] = int32(0)
        sc_dcorr_fixed[p, 2] = int32(0)
        sc_dcount[p] = int32(0)
        p += stride


@phase('particles')
def phase_42(tid, stride, _k, sim_dt, p_collision_normals, p_depth, p_friction, p_next_positions,
             p_old_positions, p_static_friction, p_velocities, p_velocity_positions, st_move_particle,
             st_move_team, t_angular_velocity, t_centrifugal_acceleration, t_cws, t_dynamic_friction,
             t_enabled, t_now_world_position, t_particle_speed_limit, t_rotation_axis, t_scale_ratio,
             t_static_friction, t_update_count, t_valid, t_velocity_weight, n_move):
    e = tid
    while e < n_move:
        mt = st_move_team[e]
        if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
            pmi = st_move_particle[e]
            n0 = p_next_positions[pmi, 0]
            n1 = p_next_positions[pmi, 1]
            n2 = p_next_positions[pmi, 2]
            o0 = p_old_positions[pmi, 0]
            o1 = p_old_positions[pmi, 1]
            o2 = p_old_positions[pmi, 2]
            vo0 = p_velocity_positions[pmi, 0]
            vo1 = p_velocity_positions[pmi, 1]
            vo2 = p_velocity_positions[pmi, 2]
            depth = p_depth[pmi]
            friction = p_friction[pmi]
            cn0 = p_collision_normals[pmi, 0]
            cn1 = p_collision_normals[pmi, 1]
            cn2 = p_collision_normals[pmi, 2]
            cn_len2 = cn0 * cn0 + cn1 * cn1 + cn2 * cn2
            is_collision = (cn_len2 > EPSILON) and (friction > EPSILON)
            static_param = t_static_friction[mt] * t_scale_ratio[mt]
            dynamic_param = t_dynamic_friction[mt]

            sfp = p_static_friction[pmi]
            static_on = static_param > float32(0.0)
            vx = n0 - o0
            vy = n1 - o1
            vz = n2 - o2
            vdotcn = vx * cn0 + vy * cn1 + vz * cn2
            tgx = vx - vdotcn * cn0
            tgy = vy - vdotcn * cn1
            tgz = vz - vdotcn * cn2
            tangent_velocity = dmath.length3(tgx, tgy, tgz) / sim_dt
            increase = dmath.saturate(sfp + float32(0.04))
            dec_amount = (tangent_velocity - static_param) / float32(0.2)
            if dec_amount < float32(0.05):
                dec_amount = float32(0.05)
            decrease = dmath.saturate(sfp - dec_amount)
            new_static = increase if tangent_velocity < static_param else decrease
            decayed = dmath.saturate(sfp - float32(0.05))
            updated_sf = new_static if is_collision else decayed
            sfp_new = updated_sf if static_on else decayed
            if static_on and is_collision:
                rbx = tgx * sfp_new
                rby = tgy * sfp_new
                rbz = tgz * sfp_new
            else:
                rbx = float32(0.0)
                rby = float32(0.0)
                rbz = float32(0.0)
            n0 = n0 - rbx
            n1 = n1 - rby
            n2 = n2 - rbz
            vo0 = vo0 - rbx
            vo1 = vo1 - rby
            vo2 = vo2 - rbz
            p_static_friction[pmi] = sfp_new

            velx = (n0 - vo0) / sim_dt
            vely = (n1 - vo1) / sim_dt
            velz = (n2 - vo2) / sim_dt
            sq_velocity = velx * velx + vely * vely + velz * velz
            nvx, nvy, nvz = dmath.normalize3(velx, vely, velz)
            if not (sq_velocity > EPSILON):
                nvx = float32(0.0)
                nvy = float32(0.0)
                nvz = float32(0.0)
            dynamic_on = dynamic_param > float32(0.0)
            dd = cn0 * nvx + cn1 * nvy + cn2 * nvz
            dd = float32(0.5) + float32(0.5) * dd
            dd = dd * dd
            dd = float32(1.0) - dd
            damp = dd * dmath.saturate(friction * dynamic_param)
            if dynamic_on and is_collision and (sq_velocity >= EPSILON):
                velx = velx - velx * damp
                vely = vely - vely * damp
                velz = velz - velz * damp
            p_friction[pmi] = friction * float32(0.6)

            speed_limit = t_particle_speed_limit[mt]
            max_len = speed_limit * t_scale_ratio[mt]
            if max_len < float32(0.0):
                max_len = float32(0.0)
            if speed_limit >= float32(0.0):
                velx, vely, velz = dmath.clamp_vector(velx, vely, velz, max_len)

            angular = t_angular_velocity[mt]
            centrifugal = t_centrifugal_acceleration[mt]
            if (angular > EPSILON) and (centrifugal > EPSILON):
                axx = t_rotation_axis[mt, 0]
                axy = t_rotation_axis[mt, 1]
                axz = t_rotation_axis[mt, 2]
                lpx = n0 - t_now_world_position[mt, 0]
                lpy = n1 - t_now_world_position[mt, 1]
                lpz = n2 - t_now_world_position[mt, 2]
                lp_dot = lpx * axx + lpy * axy + lpz * axz
                v2x = lpx - lp_dot * axx
                v2y = lpy - lp_dot * axy
                v2z = lpz - lp_dot * axz
                rr = dmath.length3(v2x, v2y, v2z)
                if (rr > EPSILON) and (sq_velocity >= EPSILON):
                    nx2, ny2, nz2 = dmath.normalize3(v2x, v2y, v2z)
                    mm = float32(1.0) + (float32(1.0) - depth)
                    ff = mm * angular * angular * rr
                    ucx, ucy, ucz = dmath.cross3(axx, axy, axz, nx2, ny2, nz2)
                    uux, uuy, uuz = dmath.normalize3(ucx, ucy, ucz)
                    ff = ff * dmath.saturate(nvx * uux + nvy * uuy + nvz * uuz)
                    addc = ff * centrifugal * float32(0.02)
                    velx = velx + nx2 * addc
                    vely = vely + ny2 * addc
                    velz = velz + nz2 * addc

            vw = t_velocity_weight[mt]
            p_velocities[pmi, 0] = velx * vw
            p_velocities[pmi, 1] = vely * vw
            p_velocities[pmi, 2] = velz * vw
            p_next_positions[pmi, 0] = n0
            p_next_positions[pmi, 1] = n1
            p_next_positions[pmi, 2] = n2
        e += stride


@phase('particles')
def phase_43(tid, stride, _k, sim_dt, p_next_positions, p_old_positions, p_real_velocities, p_team, t_cws,
             t_enabled, t_update_count, t_valid, num_particles):
    p = tid
    while p < num_particles:
        mt = p_team[p]
        if team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > _k:
            p_real_velocities[p, 0] = (p_next_positions[p, 0] - p_old_positions[p, 0]) / sim_dt
            p_real_velocities[p, 1] = (p_next_positions[p, 1] - p_old_positions[p, 1]) / sim_dt
            p_real_velocities[p, 2] = (p_next_positions[p, 2] - p_old_positions[p, 2]) / sim_dt
            p_old_positions[p, 0] = p_next_positions[p, 0]
            p_old_positions[p, 1] = p_next_positions[p, 1]
            p_old_positions[p, 2] = p_next_positions[p, 2]
        p += stride


@phase('colliders')
def phase_44(tid, stride, _k, c_active, c_now_pos, c_now_rot, c_now_tip, c_old_pos, c_old_rot, c_old_tip,
             c_team, t_cws, t_enabled, t_update_count, t_valid, num_colliders):
    ci = tid
    while ci < num_colliders:
        cm = c_team[ci]
        if team_frame_mask(t_enabled, t_valid, t_cws, cm) and t_update_count[cm] > _k \
                and c_active[ci] != 0:
            do_collider_end_step(ci, c_now_pos, c_now_rot, c_now_tip,
                                 c_old_pos, c_old_rot, c_old_tip)
        ci += stride


@phase('particles')
def phase_45(tid, stride, p_intersect_flag, p_team, t_cws, t_enabled, t_valid, num_particles):
    p = tid
    while p < num_particles:
        if team_frame_mask(t_enabled, t_valid, t_cws, p_team[p]):
            p_intersect_flag[p] = uint8(0)
        p += stride


@phase('self_contacts')
def phase_46(tid, stride, ip_edge, ip_tri, p_intersect_flag, p_next_positions, scl_counts, sfe_particles,
             sft_particles):
    ip_count = scl_counts[SCL_IP_COUNT]
    ip_lim = ip_count if ip_count < ip_edge.shape[0] else ip_edge.shape[0]
    e = tid
    while e < ip_lim:
        edge_prim = ip_edge[e]
        tri_prim = ip_tri[e]
        ep0 = sfe_particles[edge_prim, 0]
        ep1 = sfe_particles[edge_prim, 1]
        ta = sft_particles[tri_prim, 0]
        tb = sft_particles[tri_prim, 1]
        tc = sft_particles[tri_prim, 2]
        px = p_next_positions[ep0, 0]; py = p_next_positions[ep0, 1]; pz = p_next_positions[ep0, 2]
        qx = p_next_positions[ep1, 0]; qy = p_next_positions[ep1, 1]; qz = p_next_positions[ep1, 2]
        ax = p_next_positions[ta, 0]; ay = p_next_positions[ta, 1]; az = p_next_positions[ta, 2]
        bx = p_next_positions[tb, 0]; by = p_next_positions[tb, 1]; bz = p_next_positions[tb, 2]
        cx = p_next_positions[tc, 0]; cy = p_next_positions[tc, 1]; cz = p_next_positions[tc, 2]
        qpx = px - qx; qpy = py - qy; qpz = pz - qz
        acx = cx - ax; acy = cy - ay; acz = cz - az
        abx = bx - ax; aby = by - ay; abz = bz - az
        nx, ny, nz = dmath.cross3(abx, aby, abz, acx, acy, acz)
        d = qpx * nx + qpy * ny + qpz * nz
        ok = libdevice.fabsf(d) >= EPSILON
        if d < float32(0.0):
            p2x = qx; p2y = qy; p2z = qz
            qp2x = -qpx; qp2y = -qpy; qp2z = -qpz
        else:
            p2x = px; p2y = py; p2z = pz
            qp2x = qpx; qp2y = qpy; qp2z = qpz
        d2 = libdevice.fabsf(d)
        apx = p2x - ax; apy = p2y - ay; apz = p2z - az
        tparam = apx * nx + apy * ny + apz * nz
        ok = ok and (tparam >= float32(0.0)) and (tparam <= d2)
        ecx, ecy, ecz = dmath.cross3(qp2x, qp2y, qp2z, apx, apy, apz)
        vparam = acx * ecx + acy * ecy + acz * ecz
        ok = ok and (vparam >= float32(0.0)) and (vparam <= d2)
        wparam = -(abx * ecx + aby * ecy + abz * ecz)
        ok = ok and (wparam >= float32(0.0)) and ((vparam + wparam) <= d2)
        if ok:
            p_intersect_flag[ep0] = uint8(1)
            p_intersect_flag[ep1] = uint8(1)
        e += stride


@phase('particles')
def phase_47(tid, stride, sim_dt, p_display_positions, p_old_anim_positions, p_old_anim_rotations,
             p_old_positions, p_positions, p_real_velocities, p_rotations, p_team, p_temp_base_positions,
             p_temp_base_rotations, p_vertex_root, st_display_update_move_mask, t_blend_weight, t_cws,
             t_enabled, t_is_negative_scale, t_negative_scale_direction, t_now_update, t_old_time,
             t_running, t_time, t_valid, num_particles):
    p = tid
    while p < num_particles:
        mt = p_team[p]
        if team_frame_mask(t_enabled, t_valid, t_cws, mt):
            do_display_particle(p, mt, sim_dt, p_positions, p_rotations, p_old_positions,
                                p_real_velocities, p_display_positions, p_vertex_root,
                                p_old_anim_positions, p_old_anim_rotations,
                                p_temp_base_positions, p_temp_base_rotations,
                                st_display_update_move_mask, t_now_update, t_old_time, t_time,
                                t_blend_weight, t_running, t_is_negative_scale,
                                t_negative_scale_direction)
        p += stride


@phase('single')
def phase_48(tid, bid, bdim, p_attr_invalid, p_attr_move, p_attr_zero_distance, p_positions, p_rotations,
             p_team, p_temp_base_positions, p_temp_base_rotations, p_vertex_local_positions,
             p_vertex_local_rotations, postline_child_offsets, postline_child_vertices,
             postline_entry_offsets, postline_entry_vertices, t_animation_pose_ratio, t_blend_weight,
             t_cws, t_enabled, t_negative_scale_direction, t_negative_scale_quaternion, t_root_rotation,
             t_rotational_interpolation, t_valid, num_postline_levels):
    for lvl in range(num_postline_levels):
        if bid == 0:
            pl_start = postline_entry_offsets[lvl]
            pl_end = postline_entry_offsets[lvl + 1]
            i = pl_start + tid
            while i < pl_end:
                entry = postline_entry_vertices[i]
                et = p_team[entry]
                if team_frame_mask(t_enabled, t_valid, t_cws, et):
                    do_postline_entry(entry, et, postline_child_offsets[i],
                                      postline_child_offsets[i + 1], postline_child_vertices,
                                      p_positions, p_rotations, p_temp_base_positions,
                                      p_temp_base_rotations, p_vertex_local_positions,
                                      p_vertex_local_rotations, p_attr_invalid,
                                      p_attr_zero_distance, p_attr_move, p_team,
                                      t_rotational_interpolation, t_root_rotation, t_blend_weight,
                                      t_animation_pose_ratio, t_negative_scale_direction,
                                      t_negative_scale_quaternion)
                i += bdim
        cuda.syncthreads()


@phase('triangles')
def phase_49(tid, stride, p_positions, p_uv, sc_tri_normal_f64, sc_tri_tangent_f64, st_triangle_particles,
             st_triangle_team, t_cws, t_enabled, t_negative_scale_triangle_sign, t_valid, num_triangles):
    tri_idx = tid
    while tri_idx < num_triangles:
        tt_team = st_triangle_team[tri_idx]
        if team_frame_mask(t_enabled, t_valid, t_cws, tt_team):
            do_triangle_normal_tangent(tri_idx, tt_team, st_triangle_particles, p_positions,
                                       p_uv, t_negative_scale_triangle_sign, sc_tri_normal_f64,
                                       sc_tri_tangent_f64)
        tri_idx += stride


@phase('particles')
def phase_50(tid, stride, csr_v2t_offsets, csr_v2t_order, p_normal_adjustment_rotations, p_rotations,
             p_team, sc_tri_normal_f64, sc_tri_tangent_f64, st_v2t_flip_normal, st_v2t_flip_tangent,
             st_v2t_triangle, t_cws, t_enabled, t_negative_scale_quaternion, t_valid, num_particles):
    p = tid
    while p < num_particles:
        seg0 = csr_v2t_offsets[p]
        seg1 = csr_v2t_offsets[p + 1]
        if seg0 < seg1 and team_frame_mask(t_enabled, t_valid, t_cws, p_team[p]):
            do_v2t_owner(p, p_team[p], seg0, seg1, csr_v2t_order, st_v2t_triangle,
                         st_v2t_flip_normal, st_v2t_flip_tangent, sc_tri_normal_f64,
                         sc_tri_tangent_f64, p_rotations, p_normal_adjustment_rotations,
                         t_negative_scale_quaternion)
        p += stride


@phase('particles')
def phase_51(tid, stride, p_out_rotations, p_rotations, p_team, p_vertex_to_transform_rotations, t_cws,
             t_enabled, t_negative_scale_quaternion, t_valid, num_particles):
    p = tid
    while p < num_particles:
        mt = p_team[p]
        if team_frame_mask(t_enabled, t_valid, t_cws, mt):
            do_output_particle(p, mt, p_rotations, p_vertex_to_transform_rotations,
                               t_negative_scale_quaternion, p_out_rotations)
        p += stride


@phase('colliders')
def phase_52(tid, stride, c_active, c_frame_pos, c_frame_rot, c_frame_tip, c_old_frame_pos,
             c_old_frame_rot, c_old_frame_tip, c_team, t_cws, t_enabled, t_running, t_valid,
             num_colliders):
    ci = tid
    while ci < num_colliders:
        cm = c_team[ci]
        if team_frame_mask(t_enabled, t_valid, t_cws, cm) and t_running[cm] != 0 \
                and c_active[ci] != 0:
            do_collider_frame_post(ci, c_frame_pos, c_frame_rot, c_frame_tip,
                                   c_old_frame_pos, c_old_frame_rot, c_old_frame_tip)
        ci += stride


@phase('teams')
def phase_53(tid, stride, t_anchor_component_local_position, t_anchor_position, t_anchor_rotation,
             t_component_world_position, t_component_world_rotation, t_cws, t_enabled, t_force_mode,
             t_frame_old, t_frame_update, t_frame_world_position, t_frame_world_rotation,
             t_frame_world_scale, t_impact_force, t_inertia_shift, t_keep_teleport_pending,
             t_negative_scale_teleport, t_now_update, t_old_anchor_position, t_old_anchor_rotation,
             t_old_component_world_position, t_old_component_world_rotation, t_old_component_world_scale,
             t_old_frame_world_position, t_old_frame_world_rotation, t_old_frame_world_scale, t_old_time,
             t_old_update, t_reset_pending, t_running, t_skip_count, t_time, t_time_reset, t_valid,
             num_teams):
    i = tid
    while i < num_teams:
        if team_frame_mask(t_enabled, t_valid, t_cws, i):
            run = t_running[i] != 0
            t_old_component_world_position[i, 0] = t_component_world_position[i, 0]
            t_old_component_world_position[i, 1] = t_component_world_position[i, 1]
            t_old_component_world_position[i, 2] = t_component_world_position[i, 2]
            t_old_component_world_rotation[i, 0] = t_component_world_rotation[i, 0]
            t_old_component_world_rotation[i, 1] = t_component_world_rotation[i, 1]
            t_old_component_world_rotation[i, 2] = t_component_world_rotation[i, 2]
            t_old_component_world_rotation[i, 3] = t_component_world_rotation[i, 3]
            t_old_component_world_scale[i, 0] = t_cws[i, 0]
            t_old_component_world_scale[i, 1] = t_cws[i, 1]
            t_old_component_world_scale[i, 2] = t_cws[i, 2]
            if run:
                t_old_frame_world_position[i, 0] = t_frame_world_position[i, 0]
                t_old_frame_world_position[i, 1] = t_frame_world_position[i, 1]
                t_old_frame_world_position[i, 2] = t_frame_world_position[i, 2]
                t_old_frame_world_rotation[i, 0] = t_frame_world_rotation[i, 0]
                t_old_frame_world_rotation[i, 1] = t_frame_world_rotation[i, 1]
                t_old_frame_world_rotation[i, 2] = t_frame_world_rotation[i, 2]
                t_old_frame_world_rotation[i, 3] = t_frame_world_rotation[i, 3]
                t_old_frame_world_scale[i, 0] = t_frame_world_scale[i, 0]
                t_old_frame_world_scale[i, 1] = t_frame_world_scale[i, 1]
                t_old_frame_world_scale[i, 2] = t_frame_world_scale[i, 2]
                t_skip_count[i] = int32(0)
                t_force_mode[i] = 0
                t_impact_force[i, 0] = float32(0.0)
                t_impact_force[i, 1] = float32(0.0)
                t_impact_force[i, 2] = float32(0.0)
            t_old_anchor_position[i, 0] = t_anchor_position[i, 0]
            t_old_anchor_position[i, 1] = t_anchor_position[i, 1]
            t_old_anchor_position[i, 2] = t_anchor_position[i, 2]
            t_old_anchor_rotation[i, 0] = t_anchor_rotation[i, 0]
            t_old_anchor_rotation[i, 1] = t_anchor_rotation[i, 1]
            t_old_anchor_rotation[i, 2] = t_anchor_rotation[i, 2]
            t_old_anchor_rotation[i, 3] = t_anchor_rotation[i, 3]
            qix, qiy, qiz, qiw = dmath.quat_inverse(
                t_anchor_rotation[i, 0], t_anchor_rotation[i, 1],
                t_anchor_rotation[i, 2], t_anchor_rotation[i, 3])
            dpx = t_component_world_position[i, 0] - t_anchor_position[i, 0]
            dpy = t_component_world_position[i, 1] - t_anchor_position[i, 1]
            dpz = t_component_world_position[i, 2] - t_anchor_position[i, 2]
            alx, aly, alz = dmath.quat_rotate(qix, qiy, qiz, qiw, dpx, dpy, dpz)
            t_anchor_component_local_position[i, 0] = alx
            t_anchor_component_local_position[i, 1] = aly
            t_anchor_component_local_position[i, 2] = alz
            t_reset_pending[i] = 0
            t_time_reset[i] = 0
            t_running[i] = 0
            t_keep_teleport_pending[i] = 0
            t_inertia_shift[i] = 0
            t_negative_scale_teleport[i] = 0
            if t_time[i] > float32(7200.0):
                t_time[i] = t_time[i] - float32(3600.0)
                t_old_time[i] = t_old_time[i] - float32(3600.0)
                t_now_update[i] = t_now_update[i] - float32(3600.0)
                t_old_update[i] = t_old_update[i] - float32(3600.0)
                t_frame_update[i] = t_frame_update[i] - float32(3600.0)
                t_frame_old[i] = t_frame_old[i] - float32(3600.0)
        i += stride
