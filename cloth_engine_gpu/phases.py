import math

from numba import cuda, float32, float64, int8, int32, uint8
from numba.cuda import libdevice

from . import dmath
from .kernels import (
    SCAL_FRAME_DT,
    SCAL_SIM_DT,
    SCAL_TIME_SCALE,
    SCAL_POWER0,
    SCAL_POWER1,
    SCAL_POWER2,
    SCAL_POWER3,
    SCAL_MAX_SIM,
    SCAL_N_ZONES,
    SCAL_SUB_END,
    SELF_COLLISION_SCR,
    SELF_COLLISION_SOLVER_ITERATION,
    SELF_COLLISION_INTERSECT_DIV,
    SELF_COLLISION_UNIFORM_GRID_SCALE,
    SELF_COLLISION_FIXED_MASS,
    SELF_COLLISION_FRICTION_MASS,
    SELF_COLLISION_CLOTH_MASS,
    SELF_COLLISION_POINT_TRIANGLE_ANGLE_COS,
    SCL_EE_COUNT,
    SCL_PT_COUNT,
    SCL_IP_COUNT,
    SCL_ERROR,
    SCL_USE_INTERSECT,
    SCL_FRAME_INDEX,
    MAX_SIM_COUNT,
    SCAL_F_LEN,
    SCAL_I_LEN,
    WIND_ZONE_SLOTS,
    ZONE_GLOBAL,
    ZONE_BOX,
    ZONE_SPHERE_DIR,
    ZONE_SPHERE_RADIAL,
    TELEPORT_RESET,
    TETHER_STRETCH_LIMIT,
    TETHER_STIFFNESS_WIDTH,
    TETHER_VELOCITY_ATTENUATION,
    EPSILON,
    WIND_BASE_SPEED,
    WIND_TURBULENCE_ANGLE,
    WIND_MAX_TIME,
    DEG2RAD,
    RAD2DEG,
    FORCE_VELOCITY_ADD,
    FORCE_VELOCITY_ADD_WITHOUT_DEPTH,
    FORCE_VELOCITY_CHANGE,
    FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH,
    BONE_SPRING_FIX_MASS,
    BONE_CLOTH_FIX_MASS,
    DISTANCE_HORIZONTAL_STIFFNESS,
    DISTANCE_VELOCITY_ATTENUATION,
    VOLUME_SIGN,
    VOLUME_SCALE,
    BENDING_FIX_INV_MASS,
    ONE_SIXTH,
    TO_FIXED,
    ANGLE_ITERATION,
    ANGLE_LIMIT_ROT_RATIO,
    ANGLE_LIMIT_ATTENUATION,
    FRICTION_MASS,
    COLLIDER_SPHERE,
    COLLIDER_CAPSULE,
    COLLIDER_PLANE,
    COLLISION_POINT,
    COLLISION_EDGE,
    INF,
    MAX_DISTANCE_RATIO_FUTURE_PREDICTION,
    TEAM_KERNEL_FIELDS,
    PARTICLE_KERNEL_FIELDS,
    TRANSFORM_KERNEL_FIELDS,
    COLLIDER_KERNEL_FIELDS,
    STATIC_KERNEL_FIELDS,
    STATIC_CSR_FIELDS,
    STATIC_DIRECT_FIELDS,
    PRIMITIVE_KERNEL_FIELDS,
    SELF_TEAM_KERNEL_FIELDS,
    SELF_PARTICLE_KERNEL_FIELDS,
    SELF_STATE_KERNEL_FIELDS,
    RESIDENT_BLOB_GROUPS,
    ZONE_BLOB_GROUPS,
    RESIDENT_BLOB_LAYOUT,
    ZONE_BLOB_LAYOUT,
    team_frame_mask,
    do_advance,
    _skin_row,
    do_base_pose,
    do_tether,
    do_wind_blend,
    do_distance_gather,
    do_step_update,
    _neg_transform_pose,
    _shift_pose,
    _shift_point,
    _premul_quat,
    _rotate_vec,
    do_collider_frame_pre,
    do_collider_start_step,
    do_collider_end_step,
    do_collider_frame_post,
    do_solve_point,
    _edge_sphere,
    _edge_capsule,
    do_solve_edge,
    do_particles_frame_pre,
    do_angle_limit,
    do_angle_restoration,
    do_display_particle,
    do_postline_entry,
    do_triangle_normal_tangent,
    do_v2t_owner,
    do_output_particle,
    do_self_update_primitive,
    self_aabb_overlap,
    self_connection_shared,
    self_ee_geometry,
    self_pt_geometry,
)


@cuda.jit(cache=True)
def phase_00(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    c_team = blob_i32_s[offs[192]:offs[192] + lens[192]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    sc_sync = blob_f32_v22[offs[275]:offs[275] + lens[275]]
    sfe_team = blob_i32_s[offs[290]:offs[290] + lens[290]]
    sfp_team = blob_i32_s[offs[278]:offs[278] + lens[278]]
    sft_team = blob_i32_s[offs[302]:offs[302] + lens[302]]
    t_anchor_inertia = blob_f32_s[offs[125]:offs[125] + lens[125]]
    t_component_world_position = blob_f32_v3[offs[75]:offs[75] + lens[75]]
    t_component_world_rotation = blob_f32_v4[offs[76]:offs[76] + lens[76]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_frame_old = blob_f32_s[offs[9]:offs[9] + lens[9]]
    t_frame_update = blob_f32_s[offs[8]:offs[8] + lens[8]]
    t_movement_inertia_smoothing = blob_f32_s[offs[127]:offs[127] + lens[127]]
    t_movement_speed_limit = blob_f32_s[offs[128]:offs[128] + lens[128]]
    t_now_update = blob_f32_s[offs[6]:offs[6] + lens[6]]
    t_old_time = blob_f32_s[offs[5]:offs[5] + lens[5]]
    t_old_update = blob_f32_s[offs[7]:offs[7] + lens[7]]
    t_rotation_speed_limit = blob_f32_s[offs[129]:offs[129] + lens[129]]
    t_sync_target = blob_i32_s[offs[119]:offs[119] + lens[119]]
    t_sync_top = blob_i32_s[offs[120]:offs[120] + lens[120]]
    t_teleport_distance = blob_f32_s[offs[131]:offs[131] + lens[131]]
    t_teleport_mode = blob_i8_s[offs[130]:offs[130] + lens[130]]
    t_teleport_rotation = blob_f32_s[offs[132]:offs[132] + lens[132]]
    t_time = blob_f32_s[offs[4]:offs[4] + lens[4]]
    t_time_scale = blob_f32_s[offs[11]:offs[11] + lens[11]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    t_world_inertia = blob_f32_s[offs[126]:offs[126] + lens[126]]
    num_teams = t_enabled.shape[0]
    num_particles = p_team.shape[0]
    num_colliders = c_team.shape[0]
    num_self_points = sfp_team.shape[0]
    num_self_edges = sfe_team.shape[0]
    num_self_triangles = sft_team.shape[0]
    num_teams = t_enabled.shape[0]

    num_particles = p_team.shape[0]
    num_colliders = c_team.shape[0]
    num_self_points = sfp_team.shape[0]
    num_self_edges = sfe_team.shape[0]
    num_self_triangles = sft_team.shape[0]

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


@cuda.jit(cache=True)
def phase_01(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    fdt = scal_f[SCAL_FRAME_DT]
    sim_dt = scal_f[SCAL_SIM_DT]
    global_time_scale = scal_f[SCAL_TIME_SCALE]
    max_sim_count = scal_i[SCAL_MAX_SIM]
    fdt = scal_f[SCAL_FRAME_DT]
    global_time_scale = scal_f[SCAL_TIME_SCALE]
    max_sim_count = scal_i[SCAL_MAX_SIM]
    sim_dt = scal_f[SCAL_SIM_DT]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_frame_dt = blob_f32_s[offs[10]:offs[10] + lens[10]]
    t_frame_old = blob_f32_s[offs[9]:offs[9] + lens[9]]
    t_frame_update = blob_f32_s[offs[8]:offs[8] + lens[8]]
    t_now_time_scale = blob_f32_s[offs[12]:offs[12] + lens[12]]
    t_now_update = blob_f32_s[offs[6]:offs[6] + lens[6]]
    t_old_time = blob_f32_s[offs[5]:offs[5] + lens[5]]
    t_old_update = blob_f32_s[offs[7]:offs[7] + lens[7]]
    t_running = blob_u8_s[offs[15]:offs[15] + lens[15]]
    t_skip_count = blob_i32_s[offs[14]:offs[14] + lens[14]]
    t_time = blob_f32_s[offs[4]:offs[4] + lens[4]]
    t_time_reset = blob_u8_s[offs[3]:offs[3] + lens[3]]
    t_time_scale = blob_f32_s[offs[11]:offs[11] + lens[11]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_teams = t_enabled.shape[0]
    i = tid
    while i < num_teams:
        if team_frame_mask(t_enabled, t_valid, t_cws, i):
            do_advance(i, fdt, sim_dt, max_sim_count, global_time_scale,
                       t_time_reset, t_time, t_old_time, t_now_update, t_old_update,
                       t_frame_update, t_frame_old, t_frame_dt, t_time_scale,
                       t_now_time_scale, t_update_count, t_skip_count, t_running)
        i += stride


@cuda.jit(cache=True)
def phase_02(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    p_local_normals = blob_f32_v3[offs[147]:offs[147] + lens[147]]
    p_local_positions = blob_f32_v3[offs[146]:offs[146] + lens[146]]
    p_local_tangents = blob_f32_v3[offs[148]:offs[148] + lens[148]]
    p_positions = blob_f32_v3[offs[151]:offs[151] + lens[151]]
    p_rotations = blob_f32_v4[offs[152]:offs[152] + lens[152]]
    p_skin_indices = blob_i32_v4[offs[149]:offs[149] + lens[149]]
    p_skin_weights = blob_f32_v4[offs[150]:offs[150] + lens[150]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    x_bind = blob_f32_m4x4[offs[191]:offs[191] + lens[191]]
    x_world = blob_f32_m4x4[offs[190]:offs[190] + lens[190]]
    num_particles = p_team.shape[0]
    p = tid
    while p < num_particles:
        if team_frame_mask(t_enabled, t_valid, t_cws, p_team[p]):
            do_base_pose(p, p_team, p_local_positions, p_local_normals, p_local_tangents,
                         p_skin_indices, p_skin_weights, p_positions, p_rotations,
                         x_world, x_bind)
        p += stride


@cuda.jit(cache=True)
def phase_03(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    sim_dt = scal_f[SCAL_SIM_DT]
    n_zones = scal_i[SCAL_N_ZONES]
    csr_center_fixed_offsets = blob_i32_s[offs[252]:offs[252] + lens[252]]
    csr_center_fixed_order = blob_i32_s[offs[253]:offs[253] + lens[253]]
    n_zones = scal_i[SCAL_N_ZONES]
    p_positions = blob_f32_v3[offs[151]:offs[151] + lens[151]]
    p_rotations = blob_f32_v4[offs[152]:offs[152] + lens[152]]
    p_vertex_bind_pose_rotations = blob_f32_v4[offs[175]:offs[175] + lens[175]]
    sim_dt = scal_f[SCAL_SIM_DT]
    st_center_fixed_particle = blob_i32_s[offs[239]:offs[239] + lens[239]]
    t_anchor_component_local_position = blob_f32_v3[offs[90]:offs[90] + lens[90]]
    t_anchor_inertia = blob_f32_s[offs[125]:offs[125] + lens[125]]
    t_anchor_position = blob_f32_v3[offs[86]:offs[86] + lens[86]]
    t_anchor_rotation = blob_f32_v4[offs[87]:offs[87] + lens[87]]
    t_blend_weight = blob_f32_s[offs[106]:offs[106] + lens[106]]
    t_component_world_position = blob_f32_v3[offs[75]:offs[75] + lens[75]]
    t_component_world_rotation = blob_f32_v4[offs[76]:offs[76] + lens[76]]
    t_culling_invisible = blob_u8_s[offs[133]:offs[133] + lens[133]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_frame_component_shift_rotation = blob_f32_v4[offs[118]:offs[118] + lens[118]]
    t_frame_component_shift_vector = blob_f32_v3[offs[117]:offs[117] + lens[117]]
    t_frame_dt = blob_f32_s[offs[10]:offs[10] + lens[10]]
    t_frame_moving_direction = blob_f32_v3[offs[110]:offs[110] + lens[110]]
    t_frame_moving_speed = blob_f32_s[offs[109]:offs[109] + lens[109]]
    t_frame_world_position = blob_f32_v3[offs[80]:offs[80] + lens[80]]
    t_frame_world_rotation = blob_f32_v4[offs[81]:offs[81] + lens[81]]
    t_frame_world_scale = blob_f32_v3[offs[82]:offs[82] + lens[82]]
    t_had_anchor = blob_u8_s[offs[124]:offs[124] + lens[124]]
    t_has_anchor = blob_u8_s[offs[123]:offs[123] + lens[123]]
    t_inertia_shift = blob_u8_s[offs[93]:offs[93] + lens[93]]
    t_init_scale = blob_f32_v3[offs[61]:offs[61] + lens[61]]
    t_is_negative_scale = blob_u8_s[offs[74]:offs[74] + lens[74]]
    t_keep_teleport_pending = blob_u8_s[offs[92]:offs[92] + lens[92]]
    t_movement_inertia_smoothing = blob_f32_s[offs[127]:offs[127] + lens[127]]
    t_movement_speed_limit = blob_f32_s[offs[128]:offs[128] + lens[128]]
    t_negative_scale_change = blob_f32_v3[offs[116]:offs[116] + lens[116]]
    t_negative_scale_direction = blob_f32_v3[offs[72]:offs[72] + lens[72]]
    t_negative_scale_matrix = blob_f64_m4x4[offs[115]:offs[115] + lens[115]]
    t_negative_scale_quaternion = blob_f32_v4[offs[73]:offs[73] + lens[73]]
    t_negative_scale_sign = blob_f32_s[offs[71]:offs[71] + lens[71]]
    t_negative_scale_teleport = blob_u8_s[offs[94]:offs[94] + lens[94]]
    t_negative_scale_triangle_sign = blob_f32_v2[offs[121]:offs[121] + lens[121]]
    t_now_time_scale = blob_f32_s[offs[12]:offs[12] + lens[12]]
    t_now_world_position = blob_f32_v3[offs[58]:offs[58] + lens[58]]
    t_now_world_rotation = blob_f32_v4[offs[95]:offs[95] + lens[95]]
    t_old_anchor_position = blob_f32_v3[offs[88]:offs[88] + lens[88]]
    t_old_anchor_rotation = blob_f32_v4[offs[89]:offs[89] + lens[89]]
    t_old_component_world_position = blob_f32_v3[offs[77]:offs[77] + lens[77]]
    t_old_component_world_rotation = blob_f32_v4[offs[78]:offs[78] + lens[78]]
    t_old_component_world_scale = blob_f32_v3[offs[79]:offs[79] + lens[79]]
    t_old_frame_world_position = blob_f32_v3[offs[83]:offs[83] + lens[83]]
    t_old_frame_world_rotation = blob_f32_v4[offs[84]:offs[84] + lens[84]]
    t_old_frame_world_scale = blob_f32_v3[offs[85]:offs[85] + lens[85]]
    t_old_world_position = blob_f32_v3[offs[23]:offs[23] + lens[23]]
    t_old_world_rotation = blob_f32_v4[offs[96]:offs[96] + lens[96]]
    t_reset_pending = blob_u8_s[offs[91]:offs[91] + lens[91]]
    t_rotation_speed_limit = blob_f32_s[offs[129]:offs[129] + lens[129]]
    t_running = blob_u8_s[offs[15]:offs[15] + lens[15]]
    t_skip_count = blob_i32_s[offs[14]:offs[14] + lens[14]]
    t_smoothing_velocity = blob_f32_v3[offs[122]:offs[122] + lens[122]]
    t_stablization_time = blob_f32_s[offs[105]:offs[105] + lens[105]]
    t_teleport_distance = blob_f32_s[offs[131]:offs[131] + lens[131]]
    t_teleport_mode = blob_i8_s[offs[130]:offs[130] + lens[130]]
    t_teleport_rotation = blob_f32_s[offs[132]:offs[132] + lens[132]]
    t_time_reset = blob_u8_s[offs[3]:offs[3] + lens[3]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    t_velocity_weight = blob_f32_s[offs[24]:offs[24] + lens[24]]
    t_wind_count = blob_i8_s[offs[41]:offs[41] + lens[41]]
    t_wind_direction = blob_f32_m4x3[offs[134]:offs[134] + lens[134]]
    t_wind_dirq = blob_f32_m4x4[offs[44]:offs[44] + lens[44]]
    t_wind_influence = blob_f32_s[offs[46]:offs[46] + lens[46]]
    t_wind_main = blob_f32_v4[offs[42]:offs[42] + lens[42]]
    t_wind_time = blob_f32_v4[offs[43]:offs[43] + lens[43]]
    t_wind_zone_id = blob_i32_v4[offs[135]:offs[135] + lens[135]]
    t_wind_zone_turbulence = blob_f32_v4[offs[45]:offs[45] + lens[45]]
    t_world_inertia = blob_f32_s[offs[126]:offs[126] + lens[126]]
    z_attenuation_lut = zone_f32_v16[zone_offs[10]:zone_offs[10] + zone_lens[10]]
    z_is_addition = zone_u8_s[zone_offs[2]:zone_offs[2] + zone_lens[2]]
    z_main = zone_f32_s[zone_offs[3]:zone_offs[3] + zone_lens[3]]
    z_mode = zone_i32_s[zone_offs[1]:zone_offs[1] + zone_lens[1]]
    z_size = zone_f32_v3[zone_offs[8]:zone_offs[8] + zone_lens[8]]
    z_turbulence = zone_f32_s[zone_offs[4]:zone_offs[4] + zone_lens[4]]
    z_world_direction = zone_f32_v3[zone_offs[6]:zone_offs[6] + zone_lens[6]]
    z_world_position = zone_f32_v3[zone_offs[5]:zone_offs[5] + zone_lens[5]]
    z_world_to_local = zone_f64_m4x4[zone_offs[7]:zone_offs[7] + zone_lens[7]]
    z_zone_id = zone_i32_s[zone_offs[0]:zone_offs[0] + zone_lens[0]]
    z_zone_volume = zone_f32_s[zone_offs[9]:zone_offs[9] + zone_lens[9]]
    num_teams = t_enabled.shape[0]
    mat_a = cuda.local.array((4, 4), float64)
    mat_b = cuda.local.array((4, 4), float64)
    mat_c = cuda.local.array((4, 4), float64)
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

            old_count = t_wind_count[i]
            for oc in range(WIND_ZONE_SLOTS):
                if oc < old_count:
                    old_zid[oc] = t_wind_zone_id[i, oc]
                    old_wt[oc] = t_wind_time[i, oc]
            count = 0
            if n_zones > 0 and t_wind_influence[i] > EPSILON:
                cx64 = float64(cwpx)
                cy64 = float64(cwpy)
                cz64 = float64(cwpz)
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
                    if is_add:
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


@cuda.jit(cache=True)
def phase_04(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    p_base_positions = blob_f32_v3[offs[159]:offs[159] + lens[159]]
    p_base_rotations = blob_f32_v4[offs[160]:offs[160] + lens[160]]
    p_collision_normals = blob_f32_v3[offs[167]:offs[167] + lens[167]]
    p_display_positions = blob_f32_v3[offs[174]:offs[174] + lens[174]]
    p_friction = blob_f32_s[offs[165]:offs[165] + lens[165]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_old_anim_positions = blob_f32_v3[offs[157]:offs[157] + lens[157]]
    p_old_anim_rotations = blob_f32_v4[offs[158]:offs[158] + lens[158]]
    p_old_positions = blob_f32_v3[offs[164]:offs[164] + lens[164]]
    p_old_rotations = blob_f32_v4[offs[173]:offs[173] + lens[173]]
    p_positions = blob_f32_v3[offs[151]:offs[151] + lens[151]]
    p_real_velocities = blob_f32_v3[offs[169]:offs[169] + lens[169]]
    p_rotations = blob_f32_v4[offs[152]:offs[152] + lens[152]]
    p_static_friction = blob_f32_s[offs[168]:offs[168] + lens[168]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    p_velocities = blob_f32_v3[offs[163]:offs[163] + lens[163]]
    p_velocity_positions = blob_f32_v3[offs[154]:offs[154] + lens[154]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_frame_component_shift_rotation = blob_f32_v4[offs[118]:offs[118] + lens[118]]
    t_frame_component_shift_vector = blob_f32_v3[offs[117]:offs[117] + lens[117]]
    t_inertia_shift = blob_u8_s[offs[93]:offs[93] + lens[93]]
    t_negative_scale_matrix = blob_f64_m4x4[offs[115]:offs[115] + lens[115]]
    t_negative_scale_teleport = blob_u8_s[offs[94]:offs[94] + lens[94]]
    t_old_component_world_position = blob_f32_v3[offs[77]:offs[77] + lens[77]]
    t_reset_pending = blob_u8_s[offs[91]:offs[91] + lens[91]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_particles = p_team.shape[0]
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


@cuda.jit(cache=True)
def phase_05(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    c_active = blob_u8_s[offs[200]:offs[200] + lens[200]]
    c_center = blob_f32_v3[offs[194]:offs[194] + lens[194]]
    c_enabled = blob_u8_s[offs[198]:offs[198] + lens[198]]
    c_enabled_prev = blob_u8_s[offs[199]:offs[199] + lens[199]]
    c_frame_pos = blob_f32_v3[offs[204]:offs[204] + lens[204]]
    c_frame_rot = blob_f32_v4[offs[205]:offs[205] + lens[205]]
    c_frame_scl = blob_f32_v3[offs[206]:offs[206] + lens[206]]
    c_input_positions = blob_f32_v3[offs[201]:offs[201] + lens[201]]
    c_input_rotations = blob_f32_v4[offs[202]:offs[202] + lens[202]]
    c_input_scales = blob_f32_v3[offs[203]:offs[203] + lens[203]]
    c_now_pos = blob_f32_v3[offs[209]:offs[209] + lens[209]]
    c_now_rot = blob_f32_v4[offs[210]:offs[210] + lens[210]]
    c_old_frame_pos = blob_f32_v3[offs[207]:offs[207] + lens[207]]
    c_old_frame_rot = blob_f32_v4[offs[208]:offs[208] + lens[208]]
    c_old_pos = blob_f32_v3[offs[211]:offs[211] + lens[211]]
    c_old_rot = blob_f32_v4[offs[212]:offs[212] + lens[212]]
    c_team = blob_i32_s[offs[192]:offs[192] + lens[192]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_frame_component_shift_rotation = blob_f32_v4[offs[118]:offs[118] + lens[118]]
    t_frame_component_shift_vector = blob_f32_v3[offs[117]:offs[117] + lens[117]]
    t_inertia_shift = blob_u8_s[offs[93]:offs[93] + lens[93]]
    t_negative_scale_change = blob_f32_v3[offs[116]:offs[116] + lens[116]]
    t_negative_scale_matrix = blob_f64_m4x4[offs[115]:offs[115] + lens[115]]
    t_negative_scale_teleport = blob_u8_s[offs[94]:offs[94] + lens[94]]
    t_old_component_world_position = blob_f32_v3[offs[77]:offs[77] + lens[77]]
    t_reset_pending = blob_u8_s[offs[91]:offs[91] + lens[91]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_colliders = c_team.shape[0]
    ci = tid
    while ci < num_colliders:
        if team_frame_mask(t_enabled, t_valid, t_cws, c_team[ci]):
            do_collider_frame_pre(ci, c_team, c_enabled, c_enabled_prev, c_active,
                                  c_input_scales, c_input_positions, c_input_rotations, c_center,
                                  c_frame_pos, c_frame_rot, c_frame_scl,
                                  c_old_frame_pos, c_old_frame_rot, c_now_pos, c_now_rot,
                                  c_old_pos, c_old_rot,
                                  t_reset_pending, t_negative_scale_teleport,
                                  t_negative_scale_matrix, t_negative_scale_change,
                                  t_inertia_shift, t_frame_component_shift_vector,
                                  t_frame_component_shift_rotation,
                                  t_old_component_world_position)
        ci += stride


@cuda.jit(cache=True)
def phase_06(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    it_pair_off = blob_i32_s[offs[359]:offs[359] + lens[359]]
    num_it_slots = it_pair_off.shape[0] - 1
    total_it = it_pair_off[num_it_slots]
    num_it_slots = it_pair_off.shape[0] - 1
    total_it = it_pair_off[num_it_slots]


@cuda.jit(cache=True)
def phase_07(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    scl_counts = blob_i32_s[offs[342]:offs[342] + lens[342]]
    if tid == 0:
        scl_counts[SCL_IP_COUNT] = int32(0)


@cuda.jit(cache=True)
def phase_08(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    ip_edge = blob_i32_s[offs[360]:offs[360] + lens[360]]
    ip_tri = blob_i32_s[offs[361]:offs[361] + lens[361]]
    it_edge_start = blob_i32_s[offs[353]:offs[353] + lens[353]]
    it_pair_off = blob_i32_s[offs[359]:offs[359] + lens[359]]
    it_same = blob_u8_s[offs[358]:offs[358] + lens[358]]
    it_tri_count = blob_i32_s[offs[357]:offs[357] + lens[357]]
    it_tri_start = blob_i32_s[offs[356]:offs[356] + lens[356]]
    it_tri_team = blob_i32_s[offs[355]:offs[355] + lens[355]]
    scl_counts = blob_i32_s[offs[342]:offs[342] + lens[342]]
    sfe_aabb_max = blob_f32_v3[offs[299]:offs[299] + lens[299]]
    sfe_aabb_min = blob_f32_v3[offs[298]:offs[298] + lens[298]]
    sfe_all_fix = blob_u8_s[offs[293]:offs[293] + lens[293]]
    sfe_ignore = blob_u8_s[offs[294]:offs[294] + lens[294]]
    sfe_particles = blob_i32_v3[offs[291]:offs[291] + lens[291]]
    sft_aabb_max = blob_f32_v3[offs[311]:offs[311] + lens[311]]
    sft_aabb_min = blob_f32_v3[offs[310]:offs[310] + lens[310]]
    sft_all_fix = blob_u8_s[offs[305]:offs[305] + lens[305]]
    sft_ignore = blob_u8_s[offs[306]:offs[306] + lens[306]]
    sft_particles = blob_i32_v3[offs[303]:offs[303] + lens[303]]
    sft_use = blob_u8_s[offs[313]:offs[313] + lens[313]]
    t_self_grid_size = blob_f32_s[offs[317]:offs[317] + lens[317]]
    t_self_max_primitive_size = blob_f32_s[offs[318]:offs[318] + lens[318]]
    num_it_slots = it_pair_off.shape[0] - 1
    total_it = it_pair_off[num_it_slots]
    frame_index = scl_counts[SCL_FRAME_INDEX]
    ip_cap = ip_edge.shape[0]
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


@cuda.jit(cache=True)
def phase_09(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    angle_pass_offsets = blob_i32_s[offs[262]:offs[262] + lens[262]]
    baseline_entries = blob_i32_s[offs[261]:offs[261] + lens[261]]
    fk_yes_offsets = blob_i32_s[offs[256]:offs[256] + lens[256]]
    st_angle_buffered_particle = blob_i32_s[offs[240]:offs[240] + lens[240]]
    st_bending_team = blob_i32_s[offs[232]:offs[232] + lens[232]]
    st_fixed_particle = blob_i32_s[offs[224]:offs[224] + lens[224]]
    st_motion_particle = blob_i32_s[offs[230]:offs[230] + lens[230]]
    st_move_particle = blob_i32_s[offs[222]:offs[222] + lens[222]]
    st_spring_particle = blob_i32_s[offs[226]:offs[226] + lens[226]]
    st_tether_particle = blob_i32_s[offs[220]:offs[220] + lens[220]]
    n_tether = st_tether_particle.shape[0]
    n_move = st_move_particle.shape[0]
    n_fixed = st_fixed_particle.shape[0]
    n_spring = st_spring_particle.shape[0]
    n_motion = st_motion_particle.shape[0]
    n_bending = st_bending_team.shape[0]
    n_baseline = baseline_entries.shape[0]
    num_fk_levels = fk_yes_offsets.shape[0] - 1
    n_angle_buffered = st_angle_buffered_particle.shape[0]
    num_angle_passes = angle_pass_offsets.shape[0] - 1
    n_tether = st_tether_particle.shape[0]
    n_move = st_move_particle.shape[0]
    n_fixed = st_fixed_particle.shape[0]
    n_spring = st_spring_particle.shape[0]
    n_motion = st_motion_particle.shape[0]
    n_bending = st_bending_team.shape[0]
    n_baseline = baseline_entries.shape[0]
    num_fk_levels = fk_yes_offsets.shape[0] - 1
    n_angle_buffered = st_angle_buffered_particle.shape[0]
    num_angle_passes = angle_pass_offsets.shape[0] - 1


@cuda.jit(cache=True)
def phase_10(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    sim_dt = scal_f[SCAL_SIM_DT]
    _k = k
    sim_dt = scal_f[SCAL_SIM_DT]
    t_angular_velocity = blob_f32_s[offs[55]:offs[55] + lens[55]]
    t_blend_weight = blob_f32_s[offs[106]:offs[106] + lens[106]]
    t_blend_weight_param = blob_f32_s[offs[107]:offs[107] + lens[107]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_distance_weight = blob_f32_s[offs[108]:offs[108] + lens[108]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_frame_interpolation = blob_f32_s[offs[17]:offs[17] + lens[17]]
    t_frame_moving_direction = blob_f32_v3[offs[110]:offs[110] + lens[110]]
    t_frame_moving_speed = blob_f32_s[offs[109]:offs[109] + lens[109]]
    t_frame_old = blob_f32_s[offs[9]:offs[9] + lens[9]]
    t_frame_world_position = blob_f32_v3[offs[80]:offs[80] + lens[80]]
    t_frame_world_rotation = blob_f32_v4[offs[81]:offs[81] + lens[81]]
    t_frame_world_scale = blob_f32_v3[offs[82]:offs[82] + lens[82]]
    t_gravity = blob_f32_s[offs[28]:offs[28] + lens[28]]
    t_gravity_direction = blob_f32_v3[offs[27]:offs[27] + lens[27]]
    t_gravity_dot = blob_f32_s[offs[102]:offs[102] + lens[102]]
    t_gravity_falloff = blob_f32_s[offs[104]:offs[104] + lens[104]]
    t_gravity_ratio = blob_f32_s[offs[29]:offs[29] + lens[29]]
    t_inertia_rotation = blob_f32_v4[offs[21]:offs[21] + lens[21]]
    t_inertia_vector = blob_f32_v3[offs[19]:offs[19] + lens[19]]
    t_init_local_gravity_direction = blob_f32_v3[offs[103]:offs[103] + lens[103]]
    t_init_scale = blob_f32_v3[offs[61]:offs[61] + lens[61]]
    t_local_inertia = blob_f32_s[offs[99]:offs[99] + lens[99]]
    t_local_movement_speed_limit = blob_f32_s[offs[100]:offs[100] + lens[100]]
    t_local_rotation_speed_limit = blob_f32_s[offs[101]:offs[101] + lens[101]]
    t_moving_wind_direction = blob_f32_v3[offs[111]:offs[111] + lens[111]]
    t_moving_wind_dirq = blob_f32_v4[offs[51]:offs[51] + lens[51]]
    t_moving_wind_main = blob_f32_s[offs[48]:offs[48] + lens[48]]
    t_moving_wind_time = blob_f32_s[offs[50]:offs[50] + lens[50]]
    t_negative_scale_direction = blob_f32_v3[offs[72]:offs[72] + lens[72]]
    t_now_update = blob_f32_s[offs[6]:offs[6] + lens[6]]
    t_now_world_position = blob_f32_v3[offs[58]:offs[58] + lens[58]]
    t_now_world_rotation = blob_f32_v4[offs[95]:offs[95] + lens[95]]
    t_old_frame_world_position = blob_f32_v3[offs[83]:offs[83] + lens[83]]
    t_old_frame_world_rotation = blob_f32_v4[offs[84]:offs[84] + lens[84]]
    t_old_frame_world_scale = blob_f32_v3[offs[85]:offs[85] + lens[85]]
    t_old_world_position = blob_f32_v3[offs[23]:offs[23] + lens[23]]
    t_old_world_rotation = blob_f32_v4[offs[96]:offs[96] + lens[96]]
    t_rotation_axis = blob_f32_v3[offs[57]:offs[57] + lens[57]]
    t_scale_ratio = blob_f32_s[offs[31]:offs[31] + lens[31]]
    t_stablization_time = blob_f32_s[offs[105]:offs[105] + lens[105]]
    t_step_move_inertia_ratio = blob_f32_s[offs[97]:offs[97] + lens[97]]
    t_step_rotation = blob_f32_v4[offs[22]:offs[22] + lens[22]]
    t_step_rotation_inertia_ratio = blob_f32_s[offs[98]:offs[98] + lens[98]]
    t_step_vector = blob_f32_v3[offs[20]:offs[20] + lens[20]]
    t_time = blob_f32_s[offs[4]:offs[4] + lens[4]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    t_velocity_weight = blob_f32_s[offs[24]:offs[24] + lens[24]]
    t_wind_count = blob_i8_s[offs[41]:offs[41] + lens[41]]
    t_wind_frequency = blob_f32_s[offs[112]:offs[112] + lens[112]]
    t_wind_main = blob_f32_v4[offs[42]:offs[42] + lens[42]]
    t_wind_moving = blob_f32_s[offs[49]:offs[49] + lens[49]]
    t_wind_time = blob_f32_v4[offs[43]:offs[43] + lens[43]]
    num_teams = t_enabled.shape[0]
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


@cuda.jit(cache=True)
def phase_11(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    c_active = blob_u8_s[offs[200]:offs[200] + lens[200]]
    c_aligned = blob_u8_s[offs[197]:offs[197] + lens[197]]
    c_axis = blob_f32_v3[offs[196]:offs[196] + lens[196]]
    c_frame_pos = blob_f32_v3[offs[204]:offs[204] + lens[204]]
    c_frame_rot = blob_f32_v4[offs[205]:offs[205] + lens[205]]
    c_frame_scl = blob_f32_v3[offs[206]:offs[206] + lens[206]]
    c_kind = blob_i32_s[offs[193]:offs[193] + lens[193]]
    c_now_pos = blob_f32_v3[offs[209]:offs[209] + lens[209]]
    c_now_rot = blob_f32_v4[offs[210]:offs[210] + lens[210]]
    c_old_frame_pos = blob_f32_v3[offs[207]:offs[207] + lens[207]]
    c_old_frame_rot = blob_f32_v4[offs[208]:offs[208] + lens[208]]
    c_old_pos = blob_f32_v3[offs[211]:offs[211] + lens[211]]
    c_old_rot = blob_f32_v4[offs[212]:offs[212] + lens[212]]
    c_size = blob_f32_v3[offs[195]:offs[195] + lens[195]]
    c_team = blob_i32_s[offs[192]:offs[192] + lens[192]]
    c_work_aabb_max = blob_f32_v3[offs[219]:offs[219] + lens[219]]
    c_work_aabb_min = blob_f32_v3[offs[218]:offs[218] + lens[218]]
    c_work_inv_old_rot = blob_f32_v4[offs[217]:offs[217] + lens[217]]
    c_work_next_pos = blob_f32_m2x3[offs[215]:offs[215] + lens[215]]
    c_work_old_pos = blob_f32_m2x3[offs[214]:offs[214] + lens[214]]
    c_work_radius = blob_f32_v2[offs[213]:offs[213] + lens[213]]
    c_work_rot = blob_f32_v4[offs[216]:offs[216] + lens[216]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_frame_interpolation = blob_f32_s[offs[17]:offs[17] + lens[17]]
    t_step_move_inertia_ratio = blob_f32_s[offs[97]:offs[97] + lens[97]]
    t_step_rotation_inertia_ratio = blob_f32_s[offs[98]:offs[98] + lens[98]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_colliders = c_team.shape[0]
    ci = tid
    while ci < num_colliders:
        cm = c_team[ci]
        if team_frame_mask(t_enabled, t_valid, t_cws, cm) and t_update_count[cm] > _k \
                and c_active[ci] != 0:
            do_collider_start_step(ci, c_team, c_kind, c_size, c_axis, c_aligned,
                                   c_frame_pos, c_frame_rot, c_frame_scl,
                                   c_old_frame_pos, c_old_frame_rot, c_now_pos, c_now_rot,
                                   c_old_pos, c_old_rot, c_work_rot, c_work_inv_old_rot,
                                   c_work_radius, c_work_old_pos, c_work_next_pos,
                                   c_work_aabb_min, c_work_aabb_max,
                                   t_frame_interpolation, t_step_move_inertia_ratio,
                                   t_step_rotation_inertia_ratio)
        ci += stride


@cuda.jit(cache=True)
def phase_12(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    sim_dt = scal_f[SCAL_SIM_DT]
    power2 = scal_f[SCAL_POWER2]
    _k = k
    p_base_positions = blob_f32_v3[offs[159]:offs[159] + lens[159]]
    p_base_rotations = blob_f32_v4[offs[160]:offs[160] + lens[160]]
    p_depth = blob_f32_s[offs[162]:offs[162] + lens[162]]
    p_friction = blob_f32_s[offs[165]:offs[165] + lens[165]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_old_anim_positions = blob_f32_v3[offs[157]:offs[157] + lens[157]]
    p_old_anim_rotations = blob_f32_v4[offs[158]:offs[158] + lens[158]]
    p_old_positions = blob_f32_v3[offs[164]:offs[164] + lens[164]]
    p_positions = blob_f32_v3[offs[151]:offs[151] + lens[151]]
    p_rotations = blob_f32_v4[offs[152]:offs[152] + lens[152]]
    p_step_basic_positions = blob_f32_v3[offs[155]:offs[155] + lens[155]]
    p_step_basic_rotations = blob_f32_v4[offs[161]:offs[161] + lens[161]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    p_velocities = blob_f32_v3[offs[163]:offs[163] + lens[163]]
    p_velocity_positions = blob_f32_v3[offs[154]:offs[154] + lens[154]]
    p_vertex_root_local = blob_i32_s[offs[166]:offs[166] + lens[166]]
    power2 = scal_f[SCAL_POWER2]
    sim_dt = scal_f[SCAL_SIM_DT]
    st_move_particle = blob_i32_s[offs[222]:offs[222] + lens[222]]
    st_move_team = blob_i32_s[offs[223]:offs[223] + lens[223]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_damping_lut = blob_f32_v16[offs[25]:offs[25] + lens[25]]
    t_depth_inertia = blob_f32_s[offs[18]:offs[18] + lens[18]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_force_mode = blob_i8_s[offs[26]:offs[26] + lens[26]]
    t_frame_interpolation = blob_f32_s[offs[17]:offs[17] + lens[17]]
    t_gravity = blob_f32_s[offs[28]:offs[28] + lens[28]]
    t_gravity_direction = blob_f32_v3[offs[27]:offs[27] + lens[27]]
    t_gravity_ratio = blob_f32_s[offs[29]:offs[29] + lens[29]]
    t_impact_force = blob_f32_v3[offs[30]:offs[30] + lens[30]]
    t_inertia_rotation = blob_f32_v4[offs[21]:offs[21] + lens[21]]
    t_inertia_vector = blob_f32_v3[offs[19]:offs[19] + lens[19]]
    t_moving_wind_dirq = blob_f32_v4[offs[51]:offs[51] + lens[51]]
    t_moving_wind_main = blob_f32_s[offs[48]:offs[48] + lens[48]]
    t_moving_wind_time = blob_f32_s[offs[50]:offs[50] + lens[50]]
    t_old_world_position = blob_f32_v3[offs[23]:offs[23] + lens[23]]
    t_scale_ratio = blob_f32_s[offs[31]:offs[31] + lens[31]]
    t_step_rotation = blob_f32_v4[offs[22]:offs[22] + lens[22]]
    t_step_vector = blob_f32_v3[offs[20]:offs[20] + lens[20]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    t_velocity_weight = blob_f32_s[offs[24]:offs[24] + lens[24]]
    t_wind_blend = blob_f32_s[offs[39]:offs[39] + lens[39]]
    t_wind_count = blob_i8_s[offs[41]:offs[41] + lens[41]]
    t_wind_depth_weight = blob_f32_s[offs[47]:offs[47] + lens[47]]
    t_wind_dirq = blob_f32_m4x4[offs[44]:offs[44] + lens[44]]
    t_wind_influence = blob_f32_s[offs[46]:offs[46] + lens[46]]
    t_wind_main = blob_f32_v4[offs[42]:offs[42] + lens[42]]
    t_wind_moving = blob_f32_s[offs[49]:offs[49] + lens[49]]
    t_wind_seed = blob_i32_s[offs[37]:offs[37] + lens[37]]
    t_wind_synchronization = blob_f32_s[offs[38]:offs[38] + lens[38]]
    t_wind_time = blob_f32_v4[offs[43]:offs[43] + lens[43]]
    t_wind_turbulence = blob_f32_s[offs[40]:offs[40] + lens[40]]
    t_wind_zone_turbulence = blob_f32_v4[offs[45]:offs[45] + lens[45]]
    num_particles = p_team.shape[0]
    n_move = st_move_particle.shape[0]
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
            moving_on = (t_moving_wind_main[mt] > float32(0.01)) and (t_wind_moving[mt] > float32(0.01))
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


@cuda.jit(cache=True)
def phase_13(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    p_base_positions = blob_f32_v3[offs[159]:offs[159] + lens[159]]
    p_base_rotations = blob_f32_v4[offs[160]:offs[160] + lens[160]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_velocity_positions = blob_f32_v3[offs[154]:offs[154] + lens[154]]
    st_fixed_particle = blob_i32_s[offs[224]:offs[224] + lens[224]]
    st_fixed_team = blob_i32_s[offs[225]:offs[225] + lens[225]]
    st_spring_particle = blob_i32_s[offs[226]:offs[226] + lens[226]]
    st_spring_team = blob_i32_s[offs[227]:offs[227] + lens[227]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_normal_axis_vector = blob_f32_v3[offs[32]:offs[32] + lens[32]]
    t_scale_ratio = blob_f32_s[offs[31]:offs[31] + lens[31]]
    t_spring_limit_distance = blob_f32_s[offs[33]:offs[33] + lens[33]]
    t_spring_noise = blob_f32_s[offs[36]:offs[36] + lens[36]]
    t_spring_normal_limit_ratio = blob_f32_s[offs[34]:offs[34] + lens[34]]
    t_spring_power = blob_f32_s[offs[35]:offs[35] + lens[35]]
    t_time = blob_f32_s[offs[4]:offs[4] + lens[4]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    n_fixed = st_fixed_particle.shape[0]
    n_spring = st_spring_particle.shape[0]
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


@cuda.jit(cache=True)
def phase_14(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    _k = k
    fk_no = blob_i32_s[offs[260]:offs[260] + lens[260]]
    fk_no_offsets = blob_i32_s[offs[259]:offs[259] + lens[259]]
    fk_yes = blob_i32_s[offs[257]:offs[257] + lens[257]]
    fk_yes_offsets = blob_i32_s[offs[256]:offs[256] + lens[256]]
    fk_yes_parent = blob_i32_s[offs[258]:offs[258] + lens[258]]
    p_step_basic_positions = blob_f32_v3[offs[155]:offs[155] + lens[155]]
    p_step_basic_rotations = blob_f32_v4[offs[161]:offs[161] + lens[161]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    p_vertex_local_positions = blob_f32_v3[offs[171]:offs[171] + lens[171]]
    p_vertex_local_rotations = blob_f32_v4[offs[172]:offs[172] + lens[172]]
    t_animation_pose_ratio = blob_f32_s[offs[60]:offs[60] + lens[60]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_init_scale = blob_f32_v3[offs[61]:offs[61] + lens[61]]
    t_is_negative_scale = blob_u8_s[offs[74]:offs[74] + lens[74]]
    t_negative_scale_direction = blob_f32_v3[offs[72]:offs[72] + lens[72]]
    t_negative_scale_quaternion = blob_f32_v4[offs[73]:offs[73] + lens[73]]
    t_scale_ratio = blob_f32_s[offs[31]:offs[31] + lens[31]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_fk_levels = fk_yes_offsets.shape[0] - 1
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


@cuda.jit(cache=True)
def phase_15(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    baseline_entries = blob_i32_s[offs[261]:offs[261] + lens[261]]
    p_base_positions = blob_f32_v3[offs[159]:offs[159] + lens[159]]
    p_base_rotations = blob_f32_v4[offs[160]:offs[160] + lens[160]]
    p_step_basic_positions = blob_f32_v3[offs[155]:offs[155] + lens[155]]
    p_step_basic_rotations = blob_f32_v4[offs[161]:offs[161] + lens[161]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    t_animation_pose_ratio = blob_f32_s[offs[60]:offs[60] + lens[60]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    n_baseline = baseline_entries.shape[0]
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


@cuda.jit(cache=True)
def phase_16(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_step_basic_positions = blob_f32_v3[offs[155]:offs[155] + lens[155]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    p_velocity_positions = blob_f32_v3[offs[154]:offs[154] + lens[154]]
    p_vertex_root = blob_i32_s[offs[156]:offs[156] + lens[156]]
    st_tether_particle = blob_i32_s[offs[220]:offs[220] + lens[220]]
    st_tether_team = blob_i32_s[offs[221]:offs[221] + lens[221]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_tether_compression = blob_f32_s[offs[16]:offs[16] + lens[16]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    n_tether = st_tether_particle.shape[0]
    e = tid
    while e < n_tether:
        tm = st_tether_team[e]
        if team_frame_mask(t_enabled, t_valid, t_cws, tm) and t_update_count[tm] > _k:
            do_tether(e, st_tether_particle, p_team, p_next_positions,
                      p_velocity_positions, p_step_basic_positions, p_vertex_root,
                      t_tether_compression)
        e += stride


@cuda.jit(cache=True)
def phase_17(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    power1 = scal_f[SCAL_POWER1]
    _k = k
    csr_distance_offsets = blob_i32_s[offs[246]:offs[246] + lens[246]]
    csr_distance_order = blob_i32_s[offs[247]:offs[247] + lens[247]]
    p_attr_move = blob_u8_s[offs[170]:offs[170] + lens[170]]
    p_base_positions = blob_f32_v3[offs[159]:offs[159] + lens[159]]
    p_depth = blob_f32_s[offs[162]:offs[162] + lens[162]]
    p_friction = blob_f32_s[offs[165]:offs[165] + lens[165]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    power1 = scal_f[SCAL_POWER1]
    sc_dcorr = blob_f32_v3[offs[270]:offs[270] + lens[270]]
    st_distance_rest = blob_f32_s[offs[229]:offs[229] + lens[229]]
    st_distance_target = blob_i32_s[offs[228]:offs[228] + lens[228]]
    t_animation_pose_ratio = blob_f32_s[offs[60]:offs[60] + lens[60]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_distance_lut = blob_f32_v16[offs[62]:offs[62] + lens[62]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_init_scale = blob_f32_v3[offs[61]:offs[61] + lens[61]]
    t_is_spring = blob_u8_s[offs[59]:offs[59] + lens[59]]
    t_scale_ratio = blob_f32_s[offs[31]:offs[31] + lens[31]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_particles = p_team.shape[0]
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


@cuda.jit(cache=True)
def phase_18(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    p_velocity_positions = blob_f32_v3[offs[154]:offs[154] + lens[154]]
    sc_dcorr = blob_f32_v3[offs[270]:offs[270] + lens[270]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_particles = p_team.shape[0]
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


@cuda.jit(cache=True)
def phase_19(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    baseline_entries = blob_i32_s[offs[261]:offs[261] + lens[261]]
    p_albuf_length = blob_f32_s[offs[177]:offs[177] + lens[177]]
    p_albuf_local_pos = blob_f32_v3[offs[178]:offs[178] + lens[178]]
    p_albuf_local_rot = blob_f32_v4[offs[179]:offs[179] + lens[179]]
    p_albuf_restore = blob_f32_v3[offs[180]:offs[180] + lens[180]]
    p_albuf_rotation = blob_f32_v4[offs[181]:offs[181] + lens[181]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_step_basic_positions = blob_f32_v3[offs[155]:offs[155] + lens[155]]
    p_step_basic_rotations = blob_f32_v4[offs[161]:offs[161] + lens[161]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    p_vertex_parent = blob_i32_s[offs[176]:offs[176] + lens[176]]
    st_angle_buffered_particle = blob_i32_s[offs[240]:offs[240] + lens[240]]
    t_angle_use_limit = blob_u8_s[offs[136]:offs[136] + lens[136]]
    t_angle_use_restoration = blob_u8_s[offs[137]:offs[137] + lens[137]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    n_baseline = baseline_entries.shape[0]
    n_angle_buffered = st_angle_buffered_particle.shape[0]
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


@cuda.jit(cache=True)
def phase_20(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    power3 = scal_f[SCAL_POWER3]
    _k = k
    angle_pass_offsets = blob_i32_s[offs[262]:offs[262] + lens[262]]
    angle_pass_parents = blob_i32_s[offs[264]:offs[264] + lens[264]]
    angle_pass_vertices = blob_i32_s[offs[263]:offs[263] + lens[263]]
    p_albuf_length = blob_f32_s[offs[177]:offs[177] + lens[177]]
    p_albuf_local_pos = blob_f32_v3[offs[178]:offs[178] + lens[178]]
    p_albuf_local_rot = blob_f32_v4[offs[179]:offs[179] + lens[179]]
    p_albuf_restore = blob_f32_v3[offs[180]:offs[180] + lens[180]]
    p_albuf_rotation = blob_f32_v4[offs[181]:offs[181] + lens[181]]
    p_attr_move = blob_u8_s[offs[170]:offs[170] + lens[170]]
    p_depth = blob_f32_s[offs[162]:offs[162] + lens[162]]
    p_friction = blob_f32_s[offs[165]:offs[165] + lens[165]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    p_velocity_positions = blob_f32_v3[offs[154]:offs[154] + lens[154]]
    power3 = scal_f[SCAL_POWER3]
    t_angle_limit_lut = blob_f32_v16[offs[138]:offs[138] + lens[138]]
    t_angle_limit_stiffness = blob_f32_s[offs[139]:offs[139] + lens[139]]
    t_angle_restoration_attenuation = blob_f32_s[offs[141]:offs[141] + lens[141]]
    t_angle_restoration_gravity_falloff = blob_f32_s[offs[142]:offs[142] + lens[142]]
    t_angle_restoration_lut = blob_f32_v16[offs[140]:offs[140] + lens[140]]
    t_angle_use_limit = blob_u8_s[offs[136]:offs[136] + lens[136]]
    t_angle_use_restoration = blob_u8_s[offs[137]:offs[137] + lens[137]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_gravity_dot = blob_f32_s[offs[102]:offs[102] + lens[102]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_angle_passes = angle_pass_offsets.shape[0] - 1
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


@cuda.jit(cache=True)
def phase_21(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    sc_dcorr_fixed = blob_i32_v3[offs[271]:offs[271] + lens[271]]
    sc_dcount = blob_i32_s[offs[272]:offs[272] + lens[272]]
    num_particles = p_team.shape[0]
    p = tid
    while p < num_particles:
        sc_dcorr_fixed[p, 0] = int32(0)
        sc_dcorr_fixed[p, 1] = int32(0)
        sc_dcorr_fixed[p, 2] = int32(0)
        sc_dcount[p] = int32(0)
        p += stride


@cuda.jit(cache=True)
def phase_22(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    power1 = scal_f[SCAL_POWER1]
    _k = k
    p_attr_move = blob_u8_s[offs[170]:offs[170] + lens[170]]
    p_depth = blob_f32_s[offs[162]:offs[162] + lens[162]]
    p_friction = blob_f32_s[offs[165]:offs[165] + lens[165]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    power1 = scal_f[SCAL_POWER1]
    sc_dcorr_fixed = blob_i32_v3[offs[271]:offs[271] + lens[271]]
    sc_dcount = blob_i32_s[offs[272]:offs[272] + lens[272]]
    st_bending_pair = blob_i32_v4[offs[233]:offs[233] + lens[233]]
    st_bending_rest = blob_f32_s[offs[234]:offs[234] + lens[234]]
    st_bending_sign = blob_i8_s[offs[235]:offs[235] + lens[235]]
    st_bending_team = blob_i32_s[offs[232]:offs[232] + lens[232]]
    t_bending_stiffness = blob_f32_s[offs[70]:offs[70] + lens[70]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_negative_scale_sign = blob_f32_s[offs[71]:offs[71] + lens[71]]
    t_scale_ratio = blob_f32_s[offs[31]:offs[31] + lens[31]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    n_bending = st_bending_team.shape[0]
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


@cuda.jit(cache=True)
def phase_23(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    p_attr_move = blob_u8_s[offs[170]:offs[170] + lens[170]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    sc_dcorr_fixed = blob_i32_v3[offs[271]:offs[271] + lens[271]]
    sc_dcount = blob_i32_s[offs[272]:offs[272] + lens[272]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_particles = p_team.shape[0]
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


@cuda.jit(cache=True)
def phase_24(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    c_active = blob_u8_s[offs[200]:offs[200] + lens[200]]
    c_kind = blob_i32_s[offs[193]:offs[193] + lens[193]]
    c_work_aabb_max = blob_f32_v3[offs[219]:offs[219] + lens[219]]
    c_work_aabb_min = blob_f32_v3[offs[218]:offs[218] + lens[218]]
    c_work_inv_old_rot = blob_f32_v4[offs[217]:offs[217] + lens[217]]
    c_work_next_pos = blob_f32_m2x3[offs[215]:offs[215] + lens[215]]
    c_work_old_pos = blob_f32_m2x3[offs[214]:offs[214] + lens[214]]
    c_work_radius = blob_f32_v2[offs[213]:offs[213] + lens[213]]
    c_work_rot = blob_f32_v4[offs[216]:offs[216] + lens[216]]
    csr_point_pair_offsets = blob_i32_s[offs[248]:offs[248] + lens[248]]
    csr_point_pair_order = blob_i32_s[offs[249]:offs[249] + lens[249]]
    p_base_positions = blob_f32_v3[offs[159]:offs[159] + lens[159]]
    p_collision_normals = blob_f32_v3[offs[167]:offs[167] + lens[167]]
    p_depth = blob_f32_s[offs[162]:offs[162] + lens[162]]
    p_friction = blob_f32_s[offs[165]:offs[165] + lens[165]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    p_velocity_positions = blob_f32_v3[offs[154]:offs[154] + lens[154]]
    st_point_pair_collider = blob_i32_s[offs[236]:offs[236] + lens[236]]
    t_collision_mode = blob_i8_s[offs[113]:offs[113] + lens[113]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_is_spring = blob_u8_s[offs[59]:offs[59] + lens[59]]
    t_limit_distance_lut = blob_f32_v16[offs[114]:offs[114] + lens[114]]
    t_radius_lut = blob_f32_v16[offs[67]:offs[67] + lens[67]]
    t_scale_ratio = blob_f32_s[offs[31]:offs[31] + lens[31]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_particles = p_team.shape[0]
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


@cuda.jit(cache=True)
def phase_25(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    sc_col_friction_fixed = blob_i32_s[offs[273]:offs[273] + lens[273]]
    sc_col_normal_fixed = blob_i32_v3[offs[274]:offs[274] + lens[274]]
    sc_dcorr_fixed = blob_i32_v3[offs[271]:offs[271] + lens[271]]
    sc_dcount = blob_i32_s[offs[272]:offs[272] + lens[272]]
    num_particles = p_team.shape[0]
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


@cuda.jit(cache=True)
def phase_26(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    c_active = blob_u8_s[offs[200]:offs[200] + lens[200]]
    c_kind = blob_i32_s[offs[193]:offs[193] + lens[193]]
    c_work_aabb_max = blob_f32_v3[offs[219]:offs[219] + lens[219]]
    c_work_aabb_min = blob_f32_v3[offs[218]:offs[218] + lens[218]]
    c_work_next_pos = blob_f32_m2x3[offs[215]:offs[215] + lens[215]]
    c_work_old_pos = blob_f32_m2x3[offs[214]:offs[214] + lens[214]]
    c_work_radius = blob_f32_v2[offs[213]:offs[213] + lens[213]]
    csr_edge_pair_offsets = blob_i32_s[offs[250]:offs[250] + lens[250]]
    csr_edge_pair_order = blob_i32_s[offs[251]:offs[251] + lens[251]]
    p_attr_move = blob_u8_s[offs[170]:offs[170] + lens[170]]
    p_depth = blob_f32_s[offs[162]:offs[162] + lens[162]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    sc_col_friction_fixed = blob_i32_s[offs[273]:offs[273] + lens[273]]
    sc_col_normal_fixed = blob_i32_v3[offs[274]:offs[274] + lens[274]]
    sc_dcorr_fixed = blob_i32_v3[offs[271]:offs[271] + lens[271]]
    sc_dcount = blob_i32_s[offs[272]:offs[272] + lens[272]]
    st_collision_edge = blob_i32_v2[offs[238]:offs[238] + lens[238]]
    st_edge_pair_collider = blob_i32_s[offs[237]:offs[237] + lens[237]]
    t_collision_mode = blob_i8_s[offs[113]:offs[113] + lens[113]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_radius_lut = blob_f32_v16[offs[67]:offs[67] + lens[67]]
    t_scale_ratio = blob_f32_s[offs[31]:offs[31] + lens[31]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
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


@cuda.jit(cache=True)
def phase_27(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    p_collision_normals = blob_f32_v3[offs[167]:offs[167] + lens[167]]
    p_friction = blob_f32_s[offs[165]:offs[165] + lens[165]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    sc_col_friction_fixed = blob_i32_s[offs[273]:offs[273] + lens[273]]
    sc_col_normal_fixed = blob_i32_v3[offs[274]:offs[274] + lens[274]]
    sc_dcorr_fixed = blob_i32_v3[offs[271]:offs[271] + lens[271]]
    sc_dcount = blob_i32_s[offs[272]:offs[272] + lens[272]]
    t_collision_mode = blob_i8_s[offs[113]:offs[113] + lens[113]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_particles = p_team.shape[0]
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


@cuda.jit(cache=True)
def phase_28(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    power1 = scal_f[SCAL_POWER1]
    _k = k
    csr_distance_offsets = blob_i32_s[offs[246]:offs[246] + lens[246]]
    csr_distance_order = blob_i32_s[offs[247]:offs[247] + lens[247]]
    p_attr_move = blob_u8_s[offs[170]:offs[170] + lens[170]]
    p_base_positions = blob_f32_v3[offs[159]:offs[159] + lens[159]]
    p_depth = blob_f32_s[offs[162]:offs[162] + lens[162]]
    p_friction = blob_f32_s[offs[165]:offs[165] + lens[165]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    power1 = scal_f[SCAL_POWER1]
    sc_dcorr = blob_f32_v3[offs[270]:offs[270] + lens[270]]
    st_distance_rest = blob_f32_s[offs[229]:offs[229] + lens[229]]
    st_distance_target = blob_i32_s[offs[228]:offs[228] + lens[228]]
    t_animation_pose_ratio = blob_f32_s[offs[60]:offs[60] + lens[60]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_distance_lut = blob_f32_v16[offs[62]:offs[62] + lens[62]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_init_scale = blob_f32_v3[offs[61]:offs[61] + lens[61]]
    t_is_spring = blob_u8_s[offs[59]:offs[59] + lens[59]]
    t_scale_ratio = blob_f32_s[offs[31]:offs[31] + lens[31]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_particles = p_team.shape[0]
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


@cuda.jit(cache=True)
def phase_29(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    p_velocity_positions = blob_f32_v3[offs[154]:offs[154] + lens[154]]
    sc_dcorr = blob_f32_v3[offs[270]:offs[270] + lens[270]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_particles = p_team.shape[0]
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


@cuda.jit(cache=True)
def phase_30(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    p_base_positions = blob_f32_v3[offs[159]:offs[159] + lens[159]]
    p_base_rotations = blob_f32_v4[offs[160]:offs[160] + lens[160]]
    p_depth = blob_f32_s[offs[162]:offs[162] + lens[162]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_velocity_positions = blob_f32_v3[offs[154]:offs[154] + lens[154]]
    st_motion_particle = blob_i32_s[offs[230]:offs[230] + lens[230]]
    st_motion_team = blob_i32_s[offs[231]:offs[231] + lens[231]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_motion_backstop_lut = blob_f32_v16[offs[69]:offs[69] + lens[69]]
    t_motion_backstop_radius = blob_f32_s[offs[66]:offs[66] + lens[66]]
    t_motion_max_distance_lut = blob_f32_v16[offs[68]:offs[68] + lens[68]]
    t_motion_stiffness = blob_f32_s[offs[65]:offs[65] + lens[65]]
    t_motion_use_backstop = blob_u8_s[offs[64]:offs[64] + lens[64]]
    t_motion_use_max_distance = blob_u8_s[offs[63]:offs[63] + lens[63]]
    t_normal_axis_vector = blob_f32_v3[offs[32]:offs[32] + lens[32]]
    t_radius_lut = blob_f32_v16[offs[67]:offs[67] + lens[67]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    n_motion = st_motion_particle.shape[0]
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


@cuda.jit(cache=True)
def phase_31(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    ct_pair_off = blob_i32_s[offs[351]:offs[351] + lens[351]]
    num_ct_slots = ct_pair_off.shape[0] - 1
    total_ct = ct_pair_off[num_ct_slots]
    num_ct_slots = ct_pair_off.shape[0] - 1
    total_ct = ct_pair_off[num_ct_slots]


@cuda.jit(cache=True)
def phase_32(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    ee_my = blob_i32_s[offs[330]:offs[330] + lens[330]]
    pt_my = blob_i32_s[offs[337]:offs[337] + lens[337]]
    ee_cap = ee_my.shape[0]
    pt_cap = pt_my.shape[0]
    ee_cap = ee_my.shape[0]
    pt_cap = pt_my.shape[0]


@cuda.jit(cache=True)
def phase_33(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    scl_max_fixed = blob_i32_s[offs[362]:offs[362] + lens[362]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_teams = t_enabled.shape[0]
    i = tid
    while i < num_teams:
        if team_frame_mask(t_enabled, t_valid, t_cws, i) and t_update_count[i] > _k:
            scl_max_fixed[i] = int32(0)
        i += stride


@cuda.jit(cache=True)
def phase_34(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    p_friction = blob_f32_s[offs[165]:offs[165] + lens[165]]
    p_intersect_flag = blob_u8_s[offs[329]:offs[329] + lens[329]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_old_positions = blob_f32_v3[offs[164]:offs[164] + lens[164]]
    scl_counts = blob_i32_s[offs[342]:offs[342] + lens[342]]
    scl_max_fixed = blob_i32_s[offs[362]:offs[362] + lens[362]]
    sfe_aabb_max = blob_f32_v3[offs[299]:offs[299] + lens[299]]
    sfe_aabb_min = blob_f32_v3[offs[298]:offs[298] + lens[298]]
    sfe_fix = blob_u8_s[offs[292]:offs[292] + lens[292]]
    sfe_ignore = blob_u8_s[offs[294]:offs[294] + lens[294]]
    sfe_intersect = blob_u8_s[offs[300]:offs[300] + lens[300]]
    sfe_inv_mass = blob_f32_v3[offs[296]:offs[296] + lens[296]]
    sfe_particles = blob_i32_v3[offs[291]:offs[291] + lens[291]]
    sfe_prim_depth = blob_f32_s[offs[295]:offs[295] + lens[295]]
    sfe_team = blob_i32_s[offs[290]:offs[290] + lens[290]]
    sfe_thickness = blob_f32_s[offs[297]:offs[297] + lens[297]]
    sfe_use = blob_u8_s[offs[301]:offs[301] + lens[301]]
    sfp_aabb_max = blob_f32_v3[offs[287]:offs[287] + lens[287]]
    sfp_aabb_min = blob_f32_v3[offs[286]:offs[286] + lens[286]]
    sfp_fix = blob_u8_s[offs[280]:offs[280] + lens[280]]
    sfp_ignore = blob_u8_s[offs[282]:offs[282] + lens[282]]
    sfp_intersect = blob_u8_s[offs[288]:offs[288] + lens[288]]
    sfp_inv_mass = blob_f32_v3[offs[284]:offs[284] + lens[284]]
    sfp_particles = blob_i32_v3[offs[279]:offs[279] + lens[279]]
    sfp_prim_depth = blob_f32_s[offs[283]:offs[283] + lens[283]]
    sfp_team = blob_i32_s[offs[278]:offs[278] + lens[278]]
    sfp_thickness = blob_f32_s[offs[285]:offs[285] + lens[285]]
    sfp_use = blob_u8_s[offs[289]:offs[289] + lens[289]]
    sft_aabb_max = blob_f32_v3[offs[311]:offs[311] + lens[311]]
    sft_aabb_min = blob_f32_v3[offs[310]:offs[310] + lens[310]]
    sft_fix = blob_u8_s[offs[304]:offs[304] + lens[304]]
    sft_ignore = blob_u8_s[offs[306]:offs[306] + lens[306]]
    sft_intersect = blob_u8_s[offs[312]:offs[312] + lens[312]]
    sft_inv_mass = blob_f32_v3[offs[308]:offs[308] + lens[308]]
    sft_particles = blob_i32_v3[offs[303]:offs[303] + lens[303]]
    sft_prim_depth = blob_f32_s[offs[307]:offs[307] + lens[307]]
    sft_team = blob_i32_s[offs[302]:offs[302] + lens[302]]
    sft_thickness = blob_f32_s[offs[309]:offs[309] + lens[309]]
    sft_use = blob_u8_s[offs[313]:offs[313] + lens[313]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_scale_ratio = blob_f32_s[offs[31]:offs[31] + lens[31]]
    t_self_cloth_mass = blob_f32_s[offs[322]:offs[322] + lens[322]]
    t_self_thickness_lut = blob_f32_v16[offs[321]:offs[321] + lens[321]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_use_edge = blob_u8_s[offs[315]:offs[315] + lens[315]]
    t_use_point = blob_u8_s[offs[314]:offs[314] + lens[314]]
    t_use_triangle = blob_u8_s[offs[316]:offs[316] + lens[316]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_self_points = sfp_team.shape[0]
    num_self_edges = sfe_team.shape[0]
    num_self_triangles = sft_team.shape[0]
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


@cuda.jit(cache=True)
def phase_35(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    scl_max_fixed = blob_i32_s[offs[362]:offs[362] + lens[362]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_self_grid_size = blob_f32_s[offs[317]:offs[317] + lens[317]]
    t_self_max_primitive_size = blob_f32_s[offs[318]:offs[318] + lens[318]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_teams = t_enabled.shape[0]
    i = tid
    while i < num_teams:
        if team_frame_mask(t_enabled, t_valid, t_cws, i) and t_update_count[i] > _k:
            ms = float32(scl_max_fixed[i]) / TO_FIXED
            t_self_max_primitive_size[i] = ms
            t_self_grid_size[i] = ms * SELF_COLLISION_UNIFORM_GRID_SCALE
        i += stride


@cuda.jit(cache=True)
def phase_36(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    scl_counts = blob_i32_s[offs[342]:offs[342] + lens[342]]
    if tid == 0:
        scl_counts[SCL_EE_COUNT] = int32(0)
        scl_counts[SCL_PT_COUNT] = int32(0)


@cuda.jit(cache=True)
def phase_37(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    ct_kind = blob_i32_s[offs[343]:offs[343] + lens[343]]
    ct_my_start = blob_i32_s[offs[345]:offs[345] + lens[345]]
    ct_pair_off = blob_i32_s[offs[351]:offs[351] + lens[351]]
    ct_same = blob_u8_s[offs[350]:offs[350] + lens[350]]
    ct_tgt_count = blob_i32_s[offs[349]:offs[349] + lens[349]]
    ct_tgt_start = blob_i32_s[offs[348]:offs[348] + lens[348]]
    ct_tgt_team = blob_i32_s[offs[347]:offs[347] + lens[347]]
    ee_enable = blob_u8_s[offs[336]:offs[336] + lens[336]]
    ee_my = blob_i32_s[offs[330]:offs[330] + lens[330]]
    ee_n = blob_f32_v3[offs[335]:offs[335] + lens[335]]
    ee_s = blob_f32_s[offs[333]:offs[333] + lens[333]]
    ee_t = blob_f32_s[offs[334]:offs[334] + lens[334]]
    ee_target = blob_i32_s[offs[331]:offs[331] + lens[331]]
    ee_thickness = blob_f32_s[offs[332]:offs[332] + lens[332]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_old_positions = blob_f32_v3[offs[164]:offs[164] + lens[164]]
    pt_enable = blob_u8_s[offs[341]:offs[341] + lens[341]]
    pt_my = blob_i32_s[offs[337]:offs[337] + lens[337]]
    pt_sign = blob_f32_s[offs[340]:offs[340] + lens[340]]
    pt_target = blob_i32_s[offs[338]:offs[338] + lens[338]]
    pt_thickness = blob_f32_s[offs[339]:offs[339] + lens[339]]
    scl_counts = blob_i32_s[offs[342]:offs[342] + lens[342]]
    sfe_aabb_max = blob_f32_v3[offs[299]:offs[299] + lens[299]]
    sfe_aabb_min = blob_f32_v3[offs[298]:offs[298] + lens[298]]
    sfe_all_fix = blob_u8_s[offs[293]:offs[293] + lens[293]]
    sfe_ignore = blob_u8_s[offs[294]:offs[294] + lens[294]]
    sfe_particles = blob_i32_v3[offs[291]:offs[291] + lens[291]]
    sfe_thickness = blob_f32_s[offs[297]:offs[297] + lens[297]]
    sfe_use = blob_u8_s[offs[301]:offs[301] + lens[301]]
    sfp_aabb_max = blob_f32_v3[offs[287]:offs[287] + lens[287]]
    sfp_aabb_min = blob_f32_v3[offs[286]:offs[286] + lens[286]]
    sfp_all_fix = blob_u8_s[offs[281]:offs[281] + lens[281]]
    sfp_ignore = blob_u8_s[offs[282]:offs[282] + lens[282]]
    sfp_particles = blob_i32_v3[offs[279]:offs[279] + lens[279]]
    sfp_thickness = blob_f32_s[offs[285]:offs[285] + lens[285]]
    sfp_use = blob_u8_s[offs[289]:offs[289] + lens[289]]
    sft_aabb_max = blob_f32_v3[offs[311]:offs[311] + lens[311]]
    sft_aabb_min = blob_f32_v3[offs[310]:offs[310] + lens[310]]
    sft_all_fix = blob_u8_s[offs[305]:offs[305] + lens[305]]
    sft_ignore = blob_u8_s[offs[306]:offs[306] + lens[306]]
    sft_particles = blob_i32_v3[offs[303]:offs[303] + lens[303]]
    sft_thickness = blob_f32_s[offs[309]:offs[309] + lens[309]]
    sft_use = blob_u8_s[offs[313]:offs[313] + lens[313]]
    t_self_grid_size = blob_f32_s[offs[317]:offs[317] + lens[317]]
    num_ct_slots = ct_pair_off.shape[0] - 1
    total_ct = ct_pair_off[num_ct_slots]
    ee_cap = ee_my.shape[0]
    pt_cap = pt_my.shape[0]
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


@cuda.jit(cache=True)
def phase_38(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    ee_enable = blob_u8_s[offs[336]:offs[336] + lens[336]]
    ee_my = blob_i32_s[offs[330]:offs[330] + lens[330]]
    ee_n = blob_f32_v3[offs[335]:offs[335] + lens[335]]
    ee_s = blob_f32_s[offs[333]:offs[333] + lens[333]]
    ee_t = blob_f32_s[offs[334]:offs[334] + lens[334]]
    ee_target = blob_i32_s[offs[331]:offs[331] + lens[331]]
    ee_thickness = blob_f32_s[offs[332]:offs[332] + lens[332]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_old_positions = blob_f32_v3[offs[164]:offs[164] + lens[164]]
    pt_enable = blob_u8_s[offs[341]:offs[341] + lens[341]]
    pt_my = blob_i32_s[offs[337]:offs[337] + lens[337]]
    pt_target = blob_i32_s[offs[338]:offs[338] + lens[338]]
    pt_thickness = blob_f32_s[offs[339]:offs[339] + lens[339]]
    scl_counts = blob_i32_s[offs[342]:offs[342] + lens[342]]
    sfe_particles = blob_i32_v3[offs[291]:offs[291] + lens[291]]
    sfp_particles = blob_i32_v3[offs[279]:offs[279] + lens[279]]
    sft_particles = blob_i32_v3[offs[303]:offs[303] + lens[303]]
    ee_cap = ee_my.shape[0]
    pt_cap = pt_my.shape[0]
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


@cuda.jit(cache=True)
def phase_39(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    ee_my = blob_i32_s[offs[330]:offs[330] + lens[330]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    pt_my = blob_i32_s[offs[337]:offs[337] + lens[337]]
    sc_dcorr_fixed = blob_i32_v3[offs[271]:offs[271] + lens[271]]
    sc_dcount = blob_i32_s[offs[272]:offs[272] + lens[272]]
    scl_counts = blob_i32_s[offs[342]:offs[342] + lens[342]]
    num_particles = p_team.shape[0]
    ee_cap = ee_my.shape[0]
    pt_cap = pt_my.shape[0]
    ee_count2 = scl_counts[SCL_EE_COUNT]
    ee_lim2 = ee_count2 if ee_count2 < ee_cap else ee_cap
    pt_count2 = scl_counts[SCL_PT_COUNT]
    pt_lim2 = pt_count2 if pt_count2 < pt_cap else pt_cap
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


@cuda.jit(cache=True)
def phase_40(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    ee_enable = blob_u8_s[offs[336]:offs[336] + lens[336]]
    ee_my = blob_i32_s[offs[330]:offs[330] + lens[330]]
    ee_n = blob_f32_v3[offs[335]:offs[335] + lens[335]]
    ee_s = blob_f32_s[offs[333]:offs[333] + lens[333]]
    ee_t = blob_f32_s[offs[334]:offs[334] + lens[334]]
    ee_target = blob_i32_s[offs[331]:offs[331] + lens[331]]
    ee_thickness = blob_f32_s[offs[332]:offs[332] + lens[332]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    pt_enable = blob_u8_s[offs[341]:offs[341] + lens[341]]
    pt_my = blob_i32_s[offs[337]:offs[337] + lens[337]]
    pt_sign = blob_f32_s[offs[340]:offs[340] + lens[340]]
    pt_target = blob_i32_s[offs[338]:offs[338] + lens[338]]
    pt_thickness = blob_f32_s[offs[339]:offs[339] + lens[339]]
    sc_dcorr_fixed = blob_i32_v3[offs[271]:offs[271] + lens[271]]
    sc_dcount = blob_i32_s[offs[272]:offs[272] + lens[272]]
    scl_counts = blob_i32_s[offs[342]:offs[342] + lens[342]]
    sfe_fix = blob_u8_s[offs[292]:offs[292] + lens[292]]
    sfe_intersect = blob_u8_s[offs[300]:offs[300] + lens[300]]
    sfe_inv_mass = blob_f32_v3[offs[296]:offs[296] + lens[296]]
    sfe_particles = blob_i32_v3[offs[291]:offs[291] + lens[291]]
    sfp_fix = blob_u8_s[offs[280]:offs[280] + lens[280]]
    sfp_intersect = blob_u8_s[offs[288]:offs[288] + lens[288]]
    sfp_inv_mass = blob_f32_v3[offs[284]:offs[284] + lens[284]]
    sfp_particles = blob_i32_v3[offs[279]:offs[279] + lens[279]]
    sft_fix = blob_u8_s[offs[304]:offs[304] + lens[304]]
    sft_intersect = blob_u8_s[offs[312]:offs[312] + lens[312]]
    sft_inv_mass = blob_f32_v3[offs[308]:offs[308] + lens[308]]
    sft_particles = blob_i32_v3[offs[303]:offs[303] + lens[303]]
    ee_cap = ee_my.shape[0]
    pt_cap = pt_my.shape[0]
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


@cuda.jit(cache=True)
def phase_41(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    sc_dcorr_fixed = blob_i32_v3[offs[271]:offs[271] + lens[271]]
    sc_dcount = blob_i32_s[offs[272]:offs[272] + lens[272]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_particles = p_team.shape[0]
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


@cuda.jit(cache=True)
def phase_42(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    sim_dt = scal_f[SCAL_SIM_DT]
    _k = k
    p_collision_normals = blob_f32_v3[offs[167]:offs[167] + lens[167]]
    p_depth = blob_f32_s[offs[162]:offs[162] + lens[162]]
    p_friction = blob_f32_s[offs[165]:offs[165] + lens[165]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_old_positions = blob_f32_v3[offs[164]:offs[164] + lens[164]]
    p_static_friction = blob_f32_s[offs[168]:offs[168] + lens[168]]
    p_velocities = blob_f32_v3[offs[163]:offs[163] + lens[163]]
    p_velocity_positions = blob_f32_v3[offs[154]:offs[154] + lens[154]]
    sim_dt = scal_f[SCAL_SIM_DT]
    st_move_particle = blob_i32_s[offs[222]:offs[222] + lens[222]]
    st_move_team = blob_i32_s[offs[223]:offs[223] + lens[223]]
    t_angular_velocity = blob_f32_s[offs[55]:offs[55] + lens[55]]
    t_centrifugal_acceleration = blob_f32_s[offs[56]:offs[56] + lens[56]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_dynamic_friction = blob_f32_s[offs[53]:offs[53] + lens[53]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_now_world_position = blob_f32_v3[offs[58]:offs[58] + lens[58]]
    t_particle_speed_limit = blob_f32_s[offs[54]:offs[54] + lens[54]]
    t_rotation_axis = blob_f32_v3[offs[57]:offs[57] + lens[57]]
    t_scale_ratio = blob_f32_s[offs[31]:offs[31] + lens[31]]
    t_static_friction = blob_f32_s[offs[52]:offs[52] + lens[52]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    t_velocity_weight = blob_f32_s[offs[24]:offs[24] + lens[24]]
    n_move = st_move_particle.shape[0]
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


@cuda.jit(cache=True)
def phase_43(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    sim_dt = scal_f[SCAL_SIM_DT]
    _k = k
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    p_old_positions = blob_f32_v3[offs[164]:offs[164] + lens[164]]
    p_real_velocities = blob_f32_v3[offs[169]:offs[169] + lens[169]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    sim_dt = scal_f[SCAL_SIM_DT]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_particles = p_team.shape[0]
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


@cuda.jit(cache=True)
def phase_44(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    _k = k
    c_active = blob_u8_s[offs[200]:offs[200] + lens[200]]
    c_now_pos = blob_f32_v3[offs[209]:offs[209] + lens[209]]
    c_now_rot = blob_f32_v4[offs[210]:offs[210] + lens[210]]
    c_old_pos = blob_f32_v3[offs[211]:offs[211] + lens[211]]
    c_old_rot = blob_f32_v4[offs[212]:offs[212] + lens[212]]
    c_team = blob_i32_s[offs[192]:offs[192] + lens[192]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_update_count = blob_i32_s[offs[13]:offs[13] + lens[13]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_colliders = c_team.shape[0]
    ci = tid
    while ci < num_colliders:
        cm = c_team[ci]
        if team_frame_mask(t_enabled, t_valid, t_cws, cm) and t_update_count[cm] > _k \
                and c_active[ci] != 0:
            do_collider_end_step(ci, c_now_pos, c_now_rot, c_old_pos, c_old_rot)
        ci += stride


@cuda.jit(cache=True)
def phase_45(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    p_intersect_flag = blob_u8_s[offs[329]:offs[329] + lens[329]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_particles = p_team.shape[0]
    p = tid
    while p < num_particles:
        if team_frame_mask(t_enabled, t_valid, t_cws, p_team[p]):
            p_intersect_flag[p] = uint8(0)
        p += stride


@cuda.jit(cache=True)
def phase_46(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    ip_edge = blob_i32_s[offs[360]:offs[360] + lens[360]]
    ip_tri = blob_i32_s[offs[361]:offs[361] + lens[361]]
    p_intersect_flag = blob_u8_s[offs[329]:offs[329] + lens[329]]
    p_next_positions = blob_f32_v3[offs[153]:offs[153] + lens[153]]
    scl_counts = blob_i32_s[offs[342]:offs[342] + lens[342]]
    sfe_particles = blob_i32_v3[offs[291]:offs[291] + lens[291]]
    sft_particles = blob_i32_v3[offs[303]:offs[303] + lens[303]]
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


@cuda.jit(cache=True)
def phase_47(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    sim_dt = scal_f[SCAL_SIM_DT]
    p_display_positions = blob_f32_v3[offs[174]:offs[174] + lens[174]]
    p_old_anim_positions = blob_f32_v3[offs[157]:offs[157] + lens[157]]
    p_old_anim_rotations = blob_f32_v4[offs[158]:offs[158] + lens[158]]
    p_old_positions = blob_f32_v3[offs[164]:offs[164] + lens[164]]
    p_positions = blob_f32_v3[offs[151]:offs[151] + lens[151]]
    p_real_velocities = blob_f32_v3[offs[169]:offs[169] + lens[169]]
    p_rotations = blob_f32_v4[offs[152]:offs[152] + lens[152]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    p_temp_base_positions = blob_f32_v3[offs[185]:offs[185] + lens[185]]
    p_temp_base_rotations = blob_f32_v4[offs[186]:offs[186] + lens[186]]
    p_vertex_root = blob_i32_s[offs[156]:offs[156] + lens[156]]
    postline_entry_offsets = blob_i32_s[offs[265]:offs[265] + lens[265]]
    sim_dt = scal_f[SCAL_SIM_DT]
    st_display_update_move_mask = blob_u8_s[offs[269]:offs[269] + lens[269]]
    st_triangle_team = blob_i32_s[offs[241]:offs[241] + lens[241]]
    t_blend_weight = blob_f32_s[offs[106]:offs[106] + lens[106]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_is_negative_scale = blob_u8_s[offs[74]:offs[74] + lens[74]]
    t_negative_scale_direction = blob_f32_v3[offs[72]:offs[72] + lens[72]]
    t_now_update = blob_f32_s[offs[6]:offs[6] + lens[6]]
    t_old_time = blob_f32_s[offs[5]:offs[5] + lens[5]]
    t_running = blob_u8_s[offs[15]:offs[15] + lens[15]]
    t_time = blob_f32_s[offs[4]:offs[4] + lens[4]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_particles = p_team.shape[0]
    num_triangles = st_triangle_team.shape[0]
    num_postline_levels = postline_entry_offsets.shape[0] - 1
    num_postline_levels = postline_entry_offsets.shape[0] - 1
    num_triangles = st_triangle_team.shape[0]
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


@cuda.jit(cache=True)
def phase_48(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    p_attr_invalid = blob_u8_s[offs[184]:offs[184] + lens[184]]
    p_attr_move = blob_u8_s[offs[170]:offs[170] + lens[170]]
    p_attr_zero_distance = blob_u8_s[offs[183]:offs[183] + lens[183]]
    p_positions = blob_f32_v3[offs[151]:offs[151] + lens[151]]
    p_rotations = blob_f32_v4[offs[152]:offs[152] + lens[152]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    p_temp_base_positions = blob_f32_v3[offs[185]:offs[185] + lens[185]]
    p_temp_base_rotations = blob_f32_v4[offs[186]:offs[186] + lens[186]]
    p_vertex_local_positions = blob_f32_v3[offs[171]:offs[171] + lens[171]]
    p_vertex_local_rotations = blob_f32_v4[offs[172]:offs[172] + lens[172]]
    postline_child_offsets = blob_i32_s[offs[267]:offs[267] + lens[267]]
    postline_child_vertices = blob_i32_s[offs[268]:offs[268] + lens[268]]
    postline_entry_offsets = blob_i32_s[offs[265]:offs[265] + lens[265]]
    postline_entry_vertices = blob_i32_s[offs[266]:offs[266] + lens[266]]
    t_animation_pose_ratio = blob_f32_s[offs[60]:offs[60] + lens[60]]
    t_blend_weight = blob_f32_s[offs[106]:offs[106] + lens[106]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_negative_scale_direction = blob_f32_v3[offs[72]:offs[72] + lens[72]]
    t_negative_scale_quaternion = blob_f32_v4[offs[73]:offs[73] + lens[73]]
    t_root_rotation = blob_f32_s[offs[144]:offs[144] + lens[144]]
    t_rotational_interpolation = blob_f32_s[offs[143]:offs[143] + lens[143]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_postline_levels = postline_entry_offsets.shape[0] - 1
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


@cuda.jit(cache=True)
def phase_49(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    p_positions = blob_f32_v3[offs[151]:offs[151] + lens[151]]
    p_uv = blob_f32_v2[offs[182]:offs[182] + lens[182]]
    sc_tri_normal_f64 = blob_f64_v3[offs[276]:offs[276] + lens[276]]
    sc_tri_tangent_f64 = blob_f64_v3[offs[277]:offs[277] + lens[277]]
    st_triangle_particles = blob_i32_v3[offs[242]:offs[242] + lens[242]]
    st_triangle_team = blob_i32_s[offs[241]:offs[241] + lens[241]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_negative_scale_triangle_sign = blob_f32_v2[offs[121]:offs[121] + lens[121]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_triangles = st_triangle_team.shape[0]
    tri_idx = tid
    while tri_idx < num_triangles:
        tt_team = st_triangle_team[tri_idx]
        if team_frame_mask(t_enabled, t_valid, t_cws, tt_team):
            do_triangle_normal_tangent(tri_idx, tt_team, st_triangle_particles, p_positions,
                                       p_uv, t_negative_scale_triangle_sign, sc_tri_normal_f64,
                                       sc_tri_tangent_f64)
        tri_idx += stride


@cuda.jit(cache=True)
def phase_50(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    csr_v2t_offsets = blob_i32_s[offs[254]:offs[254] + lens[254]]
    csr_v2t_order = blob_i32_s[offs[255]:offs[255] + lens[255]]
    p_normal_adjustment_rotations = blob_f32_v4[offs[187]:offs[187] + lens[187]]
    p_rotations = blob_f32_v4[offs[152]:offs[152] + lens[152]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    sc_tri_normal_f64 = blob_f64_v3[offs[276]:offs[276] + lens[276]]
    sc_tri_tangent_f64 = blob_f64_v3[offs[277]:offs[277] + lens[277]]
    st_v2t_flip_normal = blob_f32_s[offs[244]:offs[244] + lens[244]]
    st_v2t_flip_tangent = blob_f32_s[offs[245]:offs[245] + lens[245]]
    st_v2t_triangle = blob_i32_s[offs[243]:offs[243] + lens[243]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_negative_scale_quaternion = blob_f32_v4[offs[73]:offs[73] + lens[73]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_particles = p_team.shape[0]
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


@cuda.jit(cache=True)
def phase_51(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    p_out_rotations = blob_f32_v4[offs[189]:offs[189] + lens[189]]
    p_rotations = blob_f32_v4[offs[152]:offs[152] + lens[152]]
    p_team = blob_i32_s[offs[145]:offs[145] + lens[145]]
    p_vertex_to_transform_rotations = blob_f32_v4[offs[188]:offs[188] + lens[188]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_negative_scale_quaternion = blob_f32_v4[offs[73]:offs[73] + lens[73]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_particles = p_team.shape[0]
    p = tid
    while p < num_particles:
        mt = p_team[p]
        if team_frame_mask(t_enabled, t_valid, t_cws, mt):
            do_output_particle(p, mt, p_rotations, p_vertex_to_transform_rotations,
                               t_negative_scale_quaternion, p_out_rotations)
        p += stride


@cuda.jit(cache=True)
def phase_52(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    c_active = blob_u8_s[offs[200]:offs[200] + lens[200]]
    c_frame_pos = blob_f32_v3[offs[204]:offs[204] + lens[204]]
    c_frame_rot = blob_f32_v4[offs[205]:offs[205] + lens[205]]
    c_old_frame_pos = blob_f32_v3[offs[207]:offs[207] + lens[207]]
    c_old_frame_rot = blob_f32_v4[offs[208]:offs[208] + lens[208]]
    c_team = blob_i32_s[offs[192]:offs[192] + lens[192]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_running = blob_u8_s[offs[15]:offs[15] + lens[15]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_colliders = c_team.shape[0]
    ci = tid
    while ci < num_colliders:
        cm = c_team[ci]
        if team_frame_mask(t_enabled, t_valid, t_cws, cm) and t_running[cm] != 0 \
                and c_active[ci] != 0:
            do_collider_frame_post(ci, c_frame_pos, c_frame_rot, c_old_frame_pos, c_old_frame_rot)
        ci += stride


@cuda.jit(cache=True)
def phase_53(scal_f, scal_i, blob_u8_s, blob_f32_v3, blob_f32_s, blob_i32_s, blob_f32_v4, blob_f32_v16, blob_i8_s, blob_f32_m4x4, blob_f64_m4x4, blob_f32_v2, blob_f32_m4x3, blob_i32_v4, blob_f32_m2x3, blob_i32_v2, blob_i32_v3, blob_f32_v22, blob_f64_v3, offs, lens, zone_i32_s, zone_u8_s, zone_f32_s, zone_f32_v3, zone_f64_m4x4, zone_f32_v16, zone_offs, zone_lens, k, sit):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    t_anchor_component_local_position = blob_f32_v3[offs[90]:offs[90] + lens[90]]
    t_anchor_position = blob_f32_v3[offs[86]:offs[86] + lens[86]]
    t_anchor_rotation = blob_f32_v4[offs[87]:offs[87] + lens[87]]
    t_component_world_position = blob_f32_v3[offs[75]:offs[75] + lens[75]]
    t_component_world_rotation = blob_f32_v4[offs[76]:offs[76] + lens[76]]
    t_cws = blob_f32_v3[offs[2]:offs[2] + lens[2]]
    t_enabled = blob_u8_s[offs[0]:offs[0] + lens[0]]
    t_force_mode = blob_i8_s[offs[26]:offs[26] + lens[26]]
    t_frame_old = blob_f32_s[offs[9]:offs[9] + lens[9]]
    t_frame_update = blob_f32_s[offs[8]:offs[8] + lens[8]]
    t_frame_world_position = blob_f32_v3[offs[80]:offs[80] + lens[80]]
    t_frame_world_rotation = blob_f32_v4[offs[81]:offs[81] + lens[81]]
    t_frame_world_scale = blob_f32_v3[offs[82]:offs[82] + lens[82]]
    t_impact_force = blob_f32_v3[offs[30]:offs[30] + lens[30]]
    t_inertia_shift = blob_u8_s[offs[93]:offs[93] + lens[93]]
    t_keep_teleport_pending = blob_u8_s[offs[92]:offs[92] + lens[92]]
    t_negative_scale_teleport = blob_u8_s[offs[94]:offs[94] + lens[94]]
    t_now_update = blob_f32_s[offs[6]:offs[6] + lens[6]]
    t_old_anchor_position = blob_f32_v3[offs[88]:offs[88] + lens[88]]
    t_old_anchor_rotation = blob_f32_v4[offs[89]:offs[89] + lens[89]]
    t_old_component_world_position = blob_f32_v3[offs[77]:offs[77] + lens[77]]
    t_old_component_world_rotation = blob_f32_v4[offs[78]:offs[78] + lens[78]]
    t_old_component_world_scale = blob_f32_v3[offs[79]:offs[79] + lens[79]]
    t_old_frame_world_position = blob_f32_v3[offs[83]:offs[83] + lens[83]]
    t_old_frame_world_rotation = blob_f32_v4[offs[84]:offs[84] + lens[84]]
    t_old_frame_world_scale = blob_f32_v3[offs[85]:offs[85] + lens[85]]
    t_old_time = blob_f32_s[offs[5]:offs[5] + lens[5]]
    t_old_update = blob_f32_s[offs[7]:offs[7] + lens[7]]
    t_reset_pending = blob_u8_s[offs[91]:offs[91] + lens[91]]
    t_running = blob_u8_s[offs[15]:offs[15] + lens[15]]
    t_skip_count = blob_i32_s[offs[14]:offs[14] + lens[14]]
    t_time = blob_f32_s[offs[4]:offs[4] + lens[4]]
    t_time_reset = blob_u8_s[offs[3]:offs[3] + lens[3]]
    t_valid = blob_u8_s[offs[1]:offs[1] + lens[1]]
    num_teams = t_enabled.shape[0]
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


PHASE_TABLE = (
    ('phase_00', (), 27),
    ('phase_01', (), 20),
    ('phase_02', (), 13),
    ('phase_03', (), 86),
    ('phase_04', (), 27),
    ('phase_05', (), 28),
    ('phase_06', (), 1),
    ('phase_07', (('if', 'total_it > 0'),), 1),
    ('phase_08', (('if', 'total_it > 0'),), 22),
    ('phase_09', (), 10),
    ('phase_10', (('loop', '_k'),), 55),
    ('phase_11', (('loop', '_k'),), 29),
    ('phase_12', (('loop', '_k'),), 54),
    ('phase_13', (('loop', '_k'),), 19),
    ('phase_14', (('loop', '_k'),), 20),
    ('phase_15', (('loop', '_k'),), 11),
    ('phase_16', (('loop', '_k'),), 12),
    ('phase_17', (('loop', '_k'),), 21),
    ('phase_18', (('loop', '_k'),), 8),
    ('phase_19', (('loop', '_k'),), 18),
    ('phase_20', (('loop', '_k'),), 27),
    ('phase_21', (('loop', '_k'),), 3),
    ('phase_22', (('loop', '_k'),), 18),
    ('phase_23', (('loop', '_k'),), 9),
    ('phase_24', (('loop', '_k'),), 28),
    ('phase_25', (('loop', '_k'),), 5),
    ('phase_26', (('loop', '_k'),), 26),
    ('phase_27', (('loop', '_k'),), 13),
    ('phase_28', (('loop', '_k'),), 21),
    ('phase_29', (('loop', '_k'),), 8),
    ('phase_30', (('loop', '_k'),), 19),
    ('phase_31', (('loop', '_k'),), 1),
    ('phase_32', (('loop', '_k'), ('if', 'total_ct > 0')), 2),
    ('phase_33', (('loop', '_k'), ('if', 'total_ct > 0'), ('if', '_k == 0')), 5),
    ('phase_34', (('loop', '_k'), ('if', 'total_ct > 0'), ('if', '_k == 0')), 49),
    ('phase_35', (('loop', '_k'), ('if', 'total_ct > 0'), ('if', '_k == 0')), 7),
    ('phase_36', (('loop', '_k'), ('if', 'total_ct > 0'), ('if', '_k == 0')), 1),
    ('phase_37', (('loop', '_k'), ('if', 'total_ct > 0'), ('if', '_k == 0')), 44),
    ('phase_38', (('loop', '_k'), ('if', 'total_ct > 0'), ('else', '_k == 0')), 17),
    ('phase_39', (('loop', '_k'), ('if', 'total_ct > 0')), 6),
    ('phase_40', (('loop', '_k'), ('if', 'total_ct > 0'), ('loop', '_sit')), 28),
    ('phase_41', (('loop', '_k'), ('if', 'total_ct > 0'), ('loop', '_sit')), 8),
    ('phase_42', (('loop', '_k'),), 24),
    ('phase_43', (('loop', '_k'),), 9),
    ('phase_44', (('loop', '_k'),), 10),
    ('phase_45', (('if', 'total_it > 0'),), 5),
    ('phase_46', (('if', 'total_it > 0'),), 7),
    ('phase_47', (), 25),
    ('phase_48', (), 23),
    ('phase_49', (), 10),
    ('phase_50', (), 14),
    ('phase_51', (), 8),
    ('phase_52', (), 10),
    ('phase_53', (), 34),
)
