import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from RuriClothPhysics.cloth_kernel import compile as kernel_compile
from RuriClothPhysics.cloth_kernel import defs
from RuriClothPhysics.cloth_kernel import io as kernel_io
from RuriClothPhysics.cloth_engine_cpu import pipeline
from RuriClothPhysics.cloth_kernel import world as kernel_world


def make_chain_snapshot(name, chain_count, link_count, spacing=0.1, offset=(0.0, 0.0, 0.0),
                        is_spring=False):
    snapshot = kernel_compile.TopologySnapshot()
    snapshot.token = None
    snapshot.is_spring = is_spring
    snapshot.spring_active = is_spring
    snapshot.connection_mode = 'SEQUENTIAL_NON_LOOP_MESH' if chain_count > 1 else 'LINE'
    snapshot.wind_seed = 7

    bone_names = []
    parent_index = []
    rest_world = []
    rest_relative = []
    root_names = []
    for chain in range(chain_count):
        for link in range(link_count):
            bone_names.append("%s_%d_%d" % (name, chain, link))
            if link == 0:
                parent_index.append(-1)
                root_names.append(bone_names[-1])
            else:
                parent_index.append(len(bone_names) - 2)
            world = np.eye(4)
            world[0, 3] = offset[0] + chain * spacing
            world[1, 3] = offset[1]
            world[2, 3] = offset[2] - link * spacing
            rest_world.append(world)
            relative = np.eye(4)
            if link > 0:
                relative[2, 3] = -spacing
            else:
                relative = world.copy()
            rest_relative.append(relative)

    snapshot.bone_names = bone_names
    snapshot.parent_index = np.array(parent_index, dtype=np.int32)
    snapshot.external_parent = [None] * len(bone_names)
    snapshot.rest_world = np.array(rest_world)
    snapshot.rest_relative = np.array(rest_relative)
    snapshot.matrix_world = np.eye(4)
    snapshot.root_names = root_names
    return snapshot


def default_params(is_spring=False):
    flat = lambda value: np.full(16, value, dtype=np.float32)
    ramp = lambda start, end: np.linspace(start, end, 16, dtype=np.float32)
    return {
        "gravity": 0.0 if is_spring else 5.0,
        "gravity_direction": (0.0, 0.0, -1.0),
        "gravity_falloff": 0.0,
        "stablization_time": 0.1,
        "blend_weight_param": 1.0,
        "damping_lut": flat(0.05) * 0.2,
        "radius_lut": flat(0.02),
        "normal_axis_vector": (0.0, 1.0, 0.0),
        "rotational_interpolation": 0.5,
        "root_rotation": 0.5,
        "animation_pose_ratio": 0.0,
        "time_scale": 1.0,
        "tether_compression": defs.BONE_SPRING_TETHER_COMPRESSION_LIMIT if is_spring else 0.4,
        "distance_lut": flat(defs.BONE_SPRING_DISTANCE_STIFFNESS) if is_spring else flat(1.0),
        "bending_stiffness": 0.0 if is_spring else 1.0,
        "angle_use_restoration": True,
        "angle_restoration_lut": ramp(0.2, 0.04) * 0.2,
        "angle_restoration_attenuation": 0.8,
        "angle_restoration_gravity_falloff": 0.0,
        "angle_use_limit": False,
        "angle_limit_lut": flat(60.0),
        "angle_limit_stiffness": 1.0,
        "motion_use_max_distance": False,
        "motion_max_distance_lut": flat(0.3),
        "motion_use_backstop": False,
        "motion_backstop_radius": 10.0,
        "motion_backstop_lut": flat(0.0),
        "motion_stiffness": 1.0,
        "collision_mode": defs.COLLISION_POINT,
        "dynamic_friction": defs.BONE_SPRING_COLLISION_FRICTION if is_spring else 0.05,
        "static_friction": defs.BONE_SPRING_COLLISION_FRICTION if is_spring else 0.05,
        "limit_distance_lut": flat(0.05),
        "self_mode": defs.SELF_MODE_FULL_MESH,
        "sync_mode": defs.SELF_MODE_NONE,
        "self_thickness_lut": flat(0.005),
        "self_cloth_mass": 0.0,
        "anchor_inertia": 0.0,
        "world_inertia": 1.0,
        "movement_inertia_smoothing": 0.4,
        "movement_speed_limit": 5.0,
        "rotation_speed_limit": 720.0,
        "local_inertia": 1.0,
        "local_movement_speed_limit": -1.0,
        "local_rotation_speed_limit": -1.0,
        "depth_inertia": 0.0,
        "centrifugal_acceleration": 0.0,
        "particle_speed_limit": 4.0,
        "teleport_mode": defs.TELEPORT_NONE,
        "teleport_distance": 0.5,
        "teleport_rotation": 90.0,
        "wind_influence": 1.0,
        "wind_frequency": 1.0,
        "wind_turbulence": 1.0,
        "wind_blend": 0.7,
        "wind_synchronization": 0.7,
        "wind_depth_weight": 0.0,
        "wind_moving": 0.0,
        "spring_power": 0.04 if is_spring else 0.0,
        "spring_limit_distance": 0.1,
        "spring_normal_limit_ratio": 1.0,
        "spring_noise": 0.0,
    }


class EmptyBinding:
    def __init__(self):
        self.count = 0
        self.kinds = np.zeros(0, dtype=np.int32)
        self.objects = []


def run_golden(frame_count=60):
    world = kernel_world.World()
    snapshot_a = make_chain_snapshot("cloth", 4, 6, offset=(0.0, 0.0, 0.0))
    snapshot_b = make_chain_snapshot("spring", 1, 3, offset=(2.0, 0.0, 0.0), is_spring=True)
    setup_a = kernel_compile.build_setup(snapshot_a)
    setup_b = kernel_compile.build_setup(snapshot_b)
    assert setup_a.valid, setup_a.error
    assert setup_b.valid, setup_b.error
    team_a = world.register_team(setup_a, default_params(False), EmptyBinding())
    team_b = world.register_team(setup_b, default_params(True), EmptyBinding())

    globals_input = kernel_io.FrameGlobals()
    globals_input.frame_delta_time = 1.0 / 24.0
    globals_input.simulation_frequency = 90
    globals_input.max_simulation_count = 3
    globals_input.global_time_scale = 1.0
    globals_input.zones = []

    checksum = np.float64(0.0)
    for frame in range(frame_count):
        globals_input.frame_index = frame
        phase = frame * 0.1
        for team, snapshot in ((team_a, snapshot_a), (team_b, snapshot_b)):
            offset = np.array([np.sin(phase) * 0.2, 0.0, 0.0], dtype=np.float32)
            kernel_io.set_team_frame_input(world, team,
                                           offset, np.array([0, 0, 0, 1], dtype=np.float32),
                                           np.ones(3, dtype=np.float32), None, False, 1.0, 0)
            worlds = np.array(snapshot.rest_world, dtype=np.float32)
            worlds[:, :3, 3] += offset
            kernel_io.set_team_transform_worlds(world, team, worlds)
        pipeline.run_frame(world, globals_input)
        positions_a, rotations_a = kernel_io.team_output(world, team_a)
        positions_b, rotations_b = kernel_io.team_output(world, team_b)
        checksum += np.float64(positions_a.astype(np.float64).sum()
                               + rotations_a.astype(np.float64).sum()
                               + positions_b.astype(np.float64).sum()
                               + rotations_b.astype(np.float64).sum())
        assert np.isfinite(positions_a).all()
        assert np.isfinite(rotations_a).all()
        assert np.isfinite(positions_b).all()
        assert np.isfinite(rotations_b).all()
    return checksum


if __name__ == "__main__":
    value = run_golden()
    print("golden checksum: %.10f" % value)
