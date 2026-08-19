import numpy as np


class WindZoneInput:
    __slots__ = ("zone_id", "mode", "main", "turbulence", "is_addition", "world_position",
                 "world_direction", "world_to_local", "size", "zone_volume",
                 "attenuation_lut")

    def __init__(self):
        self.zone_id = 0
        self.mode = 'GLOBAL_DIRECTION'
        self.main = 0.0
        self.turbulence = 1.0
        self.is_addition = False
        self.world_position = np.zeros(3, dtype=np.float32)
        self.world_direction = np.array([0, 1, 0], dtype=np.float32)
        self.world_to_local = np.eye(4)
        self.size = np.zeros(3, dtype=np.float32)
        self.zone_volume = 0.0
        self.attenuation_lut = None


class FrameGlobals:
    __slots__ = ("frame_delta_time", "simulation_frequency", "max_simulation_count",
                 "global_time_scale", "frame_index", "zones")

    def __init__(self):
        self.frame_delta_time = 1.0 / 24.0
        self.simulation_frequency = 90
        self.max_simulation_count = 3
        self.global_time_scale = 1.0
        self.frame_index = 0
        self.zones = []


def set_team_frame_input(world, slot, component_position, component_rotation,
                         component_scale_signed, anchor, culling_invisible, distance_weight,
                         sync_target):
    row = world.team[slot]
    row["enabled"] = True
    row["component_world_position"] = component_position
    row["component_world_rotation"] = component_rotation
    row["component_world_scale"] = component_scale_signed
    row["culling_invisible"] = culling_invisible
    row["distance_weight"] = distance_weight
    row["sync_target"] = sync_target if sync_target is not None else 0
    if anchor is not None:
        row["has_anchor"] = True
        row["anchor_position"] = anchor[0]
        row["anchor_rotation"] = anchor[1]
    else:
        row["has_anchor"] = False


def set_team_transform_worlds(world, slot, transform_worlds):
    s = world.transform_slice(slot)
    world.transforms["world"][s] = transform_worlds.astype(np.float32)


def set_team_collider_input(world, slot, positions, rotations, scales, enabled):
    s = world.collider_slice(slot)
    ca = world.colliders
    ca["input_positions"][s] = positions
    ca["input_rotations"][s] = rotations
    ca["input_scales"][s] = scales
    ca["enabled"][s] = enabled


def team_output(world, slot):
    s = world.particle_slice(slot)
    pa = world.particles
    return pa["positions"][s], pa["out_rotations"][s]


def end_frame(world):
    world.team["enabled"][:] = False
