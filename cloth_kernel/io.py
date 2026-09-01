import numpy as np

from . import world as _world


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


ZONE_FIELDS = {
    "zone_id": (np.int32, ()),
    "mode": (np.int32, ()),
    "is_addition": (np.uint8, ()),
    "main": (np.float32, ()),
    "turbulence": (np.float32, ()),
    "world_position": (np.float32, (3,)),
    "world_direction": (np.float32, (3,)),
    "world_to_local": (np.float64, (4, 4)),
    "size": (np.float32, (3,)),
    "zone_volume": (np.float32, ()),
    "attenuation_lut": (np.float32, (16,)),
}

assert set(ZONE_FIELDS) == set(WindZoneInput.__slots__), \
    "the wind zone field table and the wind zone input record must describe the same zone, " \
    "the table declares %r and the record declares %r" \
    % (sorted(ZONE_FIELDS), sorted(WindZoneInput.__slots__))


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
                         component_scale, component_reflected, anchor, culling_invisible,
                         distance_weight, sync_target):
    assert (np.asarray(component_scale) >= 0.0).all(), \
        "team %d was handed the component world scale %r, and this field carries the length " \
        "of each component basis axis, which is never negative; the reflection travels in " \
        "component_world_reflected because the axis a mirror sits on is not recoverable " \
        "from the matrix" % (slot, np.asarray(component_scale).tolist())
    row = world.team[slot]
    row["enabled"] = True
    row["component_world_position"] = component_position
    row["component_world_rotation"] = component_rotation
    row["component_world_scale"] = component_scale
    row["component_world_reflected"] = bool(component_reflected)
    row["culling_invisible"] = culling_invisible
    row["distance_weight"] = distance_weight
    row["sync_target"] = sync_target if sync_target is not None else 0
    if anchor is not None:
        row["has_anchor"] = True
        row["anchor_position"] = anchor[0]
        row["anchor_rotation"] = anchor[1]
    else:
        row["has_anchor"] = False


def set_team_contact_links(world, slot, targets):
    ordered = tuple(int(target) for target in targets)
    assert len(set(ordered)) == len(ordered), \
        "team %d declares the contact link targets %r and a link is declared once" \
        % (slot, ordered)
    if world.contact_links.get(int(slot)) != ordered:
        world.contact_links[int(slot)] = ordered
        world.note_contact_links_written()


def set_team_transform_worlds(world, slot, transform_worlds):
    s = world.transform_slice(slot)
    world.transforms["world"][s] = transform_worlds.astype(np.float32)


def set_team_collider_input(world, slot, positions, rotations, tips, radii, enabled):
    s = world.collider_slice(slot)
    ca = world.colliders
    ca["input_positions"][s] = positions
    ca["input_rotations"][s] = rotations
    ca["input_tips"][s] = tips
    ca["input_radii"][s] = radii
    ca["enabled"][s] = enabled


COLLIDER_MESH_VERTEX_INPUT_REASON = (
    "the triangles of a mesh collider live in the collider's own frame, so a body that "
    "bends, is skinned, carries a shape key or is edited changes its vertices and nothing "
    "else, and the pose the frame already carries says nothing about any of it; the "
    "vertices are therefore a frame input like the pose is, written on every frame into "
    "the block the binding reserved, while what stays a binding is the topology, which is "
    "the face corners, the half edge pairing and how many rows the block holds")


def set_team_collider_mesh_vertices(world, slot, vertices_per_collider):
    span = world.collider_slice(slot)
    colliders = world.colliders.arrays
    stored = world.collider_vertices.arrays["local_position"]
    for offset, values in enumerate(vertices_per_collider):
        if values is None:
            continue
        row = span.start + offset
        start = int(colliders["mesh_vertex_start"][row])
        count = int(colliders["mesh_vertex_count"][row])
        assert values.shape == (count, 3), \
            "%s\ncollider row %d holds a block of %d vertices and the host handed in %r" \
            % (COLLIDER_MESH_VERTEX_INPUT_REASON, row, count, (values.shape,))
        stored[start:start + count] = values
        _world.store_collider_mesh_bound(colliders, row, values)


DISPLAY_READ_REASON = (
    "a plane the solver writes is on the host only if the frame downloaded it, and what the "
    "frame downloads is what the viewport layers declared they read; reading one nobody "
    "declared hands back whatever the mirror was left holding, which is the value the team "
    "was registered with or the one some earlier rebuild happened to bring home, and it "
    "draws as a perfectly ordinary line at the wrong place and says nothing, so the read is "
    "refused here instead of answered wrongly")


def _assert_fresh(world, storage_name, field_name):
    assert (storage_name, field_name) not in world.stale_planes, \
        "%s\n%s.%s is written by the solver and no viewport layer declared it, so the frame " \
        "left it on the device" % (DISPLAY_READ_REASON, storage_name, field_name)


def team_plane(world, field_name):
    _assert_fresh(world, "team", field_name)
    return world.team[field_name]


def team_display(world, slot, field_name):
    return team_plane(world, field_name)[slot]


def particle_display(world, slot, field_name):
    _assert_fresh(world, "particle", field_name)
    return world.particles[field_name][world.particle_slice(slot)]


def team_output(world, slot):
    s = world.particle_slice(slot)
    pa = world.particles
    return pa["positions"][s], pa["out_rotations"][s]


def end_frame(world):
    world.team["enabled"][:] = False
