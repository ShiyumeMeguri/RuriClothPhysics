import numpy as np

from .. import defs
from .. import math as pm


def run(world, ctx):
    pa = world.particles
    tt = world.team
    need = ctx.step_team_mask & (tt["animation_pose_ratio"] <= 0.99)
    if not np.any(need):
        return

    for yes, yes_parent, no in world.fk_levels:
        if len(yes):
            active = need[pa["team"][yes]]
            if np.any(active):
                v = yes[active]
                p = yes_parent[active]
                vt = pa["team"][v]
                neg_dir = tt["negative_scale_direction"][vt]
                neg_quat = tt["negative_scale_quaternion"][vt]
                scale = tt["init_scale"][vt] * tt["scale_ratio"][vt][:, None]
                lpos = pa["vertex_local_positions"][v] * neg_dir
                lrot = pa["vertex_local_rotations"][v] * neg_quat
                prot = pa["step_basic_rotations"][p]
                ppos = pa["step_basic_positions"][p]
                pa["step_basic_positions"][v] = pm.quat_rotate(prot, lpos * scale) + ppos
                pa["step_basic_rotations"][v] = pm.quat_mul(prot, lrot)
        if len(no):
            nt = pa["team"][no]
            active = need[nt] & tt["is_negative_scale"][nt]
            if np.any(active):
                v = no[active]
                vt = pa["team"][v]
                neg_dir = tt["negative_scale_direction"][vt]
                rot = pa["step_basic_rotations"][v]
                normals = pm.quat_to_normal(rot) * neg_dir[:, 1:2]
                tangents = pm.quat_to_tangent(rot) * neg_dir[:, 2:3]
                pa["step_basic_rotations"][v] = pm.look_rotation(tangents, normals)

    blend_need = need & (tt["animation_pose_ratio"] > defs.EPSILON)
    if np.any(blend_need):
        entries = world.baseline_entries
        if len(entries):
            active = blend_need[pa["team"][entries]]
            if np.any(active):
                v = entries[active]
                vt = pa["team"][v]
                ratio = tt["animation_pose_ratio"][vt]
                pa["step_basic_positions"][v] = pm.lerp(pa["step_basic_positions"][v],
                                                        pa["base_positions"][v], ratio[:, None])
                pa["step_basic_rotations"][v] = pm.quat_slerp(pa["step_basic_rotations"][v],
                                                              pa["base_rotations"][v], ratio)
