import warp as wp

from . import kernels
from . import policy

wp.set_module_options(policy.MODULE_OPTIONS)


@wp.kernel
def phase_01(fdt: float, global_time_scale: float, max_sim_count: int, sim_dt: float,
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
def phase_17(k: int, power1: float,
             csr_distance_offsets: wp.array(dtype=int),
             csr_distance_order: wp.array(dtype=int),
             p_attr_move: wp.array(dtype=int),
             p_base_positions: wp.array2d(dtype=float),
             p_depth: wp.array(dtype=float),
             p_friction: wp.array(dtype=float),
             p_next_positions: wp.array2d(dtype=float),
             p_team: wp.array(dtype=int),
             sc_dcorr: wp.array2d(dtype=float),
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
    mt = p_team[p]
    if kernels.team_frame_mask(t_enabled, t_valid, t_cws, mt) and t_update_count[mt] > k:
        kernels.do_distance_gather(p, p_team, p_next_positions, p_base_positions, p_depth,
                                   p_friction, p_attr_move, t_is_spring, t_animation_pose_ratio,
                                   t_init_scale, t_scale_ratio, t_distance_lut, power1,
                                   csr_distance_offsets, csr_distance_order,
                                   st_distance_target, st_distance_rest, sc_dcorr)


PHASE_TABLE = (
    ("phase_01", phase_01, "team"),
    ("phase_02", phase_02, "particle"),
    ("phase_16", phase_16, "tether"),
    ("phase_17", phase_17, "particle"),
)
