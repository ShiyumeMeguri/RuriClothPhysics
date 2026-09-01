import warp as wp

from ..cloth_kernel import defs as _defs
from ..cloth_kernel import fixed_point as _fixed_point
from . import dmath
from . import policy

wp.set_module_options(policy.MODULE_OPTIONS)

SELF_CONTACT_DETECTION_MARGIN = wp.constant(float(_defs.SELF_CONTACT_DETECTION_MARGIN))
GAP_ORDER_SCALE = wp.constant(float(_fixed_point.GAP_ORDER_SCALE))
GAP_ORDER_LIMIT_VALUE = wp.constant(float(_fixed_point.GAP_ORDER_LIMIT))
GAP_ORDER_LIMIT_KEY = wp.constant(int(_fixed_point.GAP_ORDER_LIMIT))
EPSILON = wp.constant(float(_defs.EPSILON))
COMPONENT_SCALE_EPSILON = wp.constant(float(_defs.COMPONENT_SCALE_EPSILON))
TETHER_STRETCH_LIMIT = wp.constant(float(_defs.TETHER_STRETCH_LIMIT))
TETHER_STIFFNESS_WIDTH = wp.constant(float(_defs.TETHER_STIFFNESS_WIDTH))
TETHER_COMPRESSION_STIFFNESS = wp.constant(float(_defs.TETHER_COMPRESSION_STIFFNESS))
TETHER_STRETCH_STIFFNESS = wp.constant(float(_defs.TETHER_STRETCH_STIFFNESS))
TETHER_COMPRESSION_VELOCITY_ATTENUATION = wp.constant(
    float(_defs.TETHER_COMPRESSION_VELOCITY_ATTENUATION))
TETHER_STRETCH_VELOCITY_ATTENUATION = wp.constant(
    float(_defs.TETHER_STRETCH_VELOCITY_ATTENUATION))
BONE_SPRING_FIX_MASS = wp.constant(float(_defs.BONE_SPRING_FIX_MASS))
BONE_CLOTH_FIX_MASS = wp.constant(float(_defs.BONE_CLOTH_FIX_MASS))
DISTANCE_HORIZONTAL_STIFFNESS = wp.constant(float(_defs.DISTANCE_HORIZONTAL_STIFFNESS))
DISTANCE_VELOCITY_ATTENUATION = wp.constant(float(_defs.DISTANCE_VELOCITY_ATTENUATION))
WIND_BASE_SPEED = wp.constant(float(_defs.WIND_BASE_SPEED))
WIND_TURBULENCE_ANGLE = wp.constant(float(_defs.WIND_TURBULENCE_ANGLE))
WIND_MIN_SPEED = wp.constant(float(_defs.WIND_MIN_SPEED))
WIND_MAX_TIME = wp.constant(float(_defs.WIND_MAX_TIME))
SCL_USE_INTERSECT = wp.constant(int(_defs.SCL_USE_INTERSECT))
SCL_EE_COUNT = wp.constant(int(_defs.SCL_EE_COUNT))
SCL_PT_COUNT = wp.constant(int(_defs.SCL_PT_COUNT))
SCL_IP_COUNT = wp.constant(int(_defs.SCL_IP_COUNT))
SCL_ERROR = wp.constant(int(_defs.SCL_ERROR))
SCL_FRAME_INDEX = wp.constant(int(_defs.SCL_FRAME_INDEX))
SELF_COLLISION_INTERSECT_DIV = wp.constant(int(_defs.SELF_COLLISION_INTERSECT_DIV))
SELF_COLLISION_UNIFORM_GRID_SCALE = wp.constant(
    float(_defs.SELF_COLLISION_UNIFORM_GRID_SCALE))
DEG2RAD = wp.constant(float(_defs.DEG2RAD))
RAD2DEG = wp.constant(float(_defs.RAD2DEG))
ANGLE_LIMIT_ROT_RATIO = wp.constant(float(_defs.ANGLE_LIMIT_ROTATION_RATIO))
ANGLE_LIMIT_ATTENUATION = wp.constant(float(_defs.ANGLE_LIMIT_ATTENUATION))
ANGLE_LIMIT_ITERATION = wp.constant(int(_defs.ANGLE_LIMIT_ITERATION))
VOLUME_SIGN = wp.constant(int(_defs.VOLUME_SIGN))
VOLUME_SCALE = wp.constant(float(_defs.VOLUME_SCALE))
BENDING_FIXED_INVERSE_MASS = wp.constant(float(_defs.BENDING_FIXED_INVERSE_MASS))
ONE_SIXTH = wp.constant(float(_defs.ONE_SIXTH))
FORCE_VELOCITY_ADD = wp.constant(int(_defs.FORCE_VELOCITY_ADD))
FORCE_VELOCITY_ADD_WITHOUT_DEPTH = wp.constant(int(_defs.FORCE_VELOCITY_ADD_WITHOUT_DEPTH))
FORCE_VELOCITY_CHANGE = wp.constant(int(_defs.FORCE_VELOCITY_CHANGE))
FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH = wp.constant(
    int(_defs.FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH))
COLLIDER_SPHERE = wp.constant(int(_defs.COLLIDER_SPHERE))
COLLIDER_CAPSULE = wp.constant(int(_defs.COLLIDER_CAPSULE))
COLLIDER_PLANE = wp.constant(int(_defs.COLLIDER_PLANE))
COLLIDER_MESH = wp.constant(int(_defs.COLLIDER_MESH))
CONTACT_PATH_SELF_COLLISION = wp.constant(int(_defs.CONTACT_PATH_SELF_COLLISION))
CONTACT_PATH_COLLIDER = wp.constant(int(_defs.CONTACT_PATH_COLLIDER))
COLLISION_POINT = wp.constant(int(_defs.COLLISION_POINT))
COLLISION_EDGE = wp.constant(int(_defs.COLLISION_EDGE))
TO_FIXED = wp.constant(float(_defs.TO_FIXED))
MAX_DISTANCE_RATIO_FUTURE_PREDICTION = wp.constant(
    float(_defs.MAX_DISTANCE_RATIO_FUTURE_PREDICTION))
COLLIDER_EXIT_BISECTION_STEPS = wp.constant(int(_defs.COLLIDER_EXIT_BISECTION_STEPS))
COLLIDER_EXIT_TOLERANCE = wp.constant(float(_defs.COLLIDER_EXIT_TOLERANCE))
TELEPORT_RESET = wp.constant(int(_defs.TELEPORT_RESET))
ZONE_BOX = wp.constant(int(_defs.ZONE_BOX))
ZONE_SPHERE_DIR = wp.constant(int(_defs.ZONE_SPHERE_DIR))
ZONE_SPHERE_RADIAL = wp.constant(int(_defs.ZONE_SPHERE_RADIAL))
WIND_ZONE_SLOTS = wp.constant(int(_defs.WIND_ZONE_SLOTS))
WIND_ZONE_RESULT_SLOTS = wp.constant(int(_defs.WIND_ZONE_RESULT_SLOTS))
WIND_ZONE_MIN_MAIN = wp.constant(float(_defs.WIND_ZONE_MIN_MAIN))
SCAL_FRAME_DT = wp.constant(int(_defs.SCAL_FRAME_DT))
SCAL_SIM_DT = wp.constant(int(_defs.SCAL_SIM_DT))
SCAL_TIME_SCALE = wp.constant(int(_defs.SCAL_TIME_SCALE))
SCAL_POWER1 = wp.constant(int(_defs.SCAL_POWER1))
SCAL_POWER2 = wp.constant(int(_defs.SCAL_POWER2))
SCAL_POWER3 = wp.constant(int(_defs.SCAL_POWER3))
SCAL_MAX_SIM = wp.constant(int(_defs.SCAL_MAX_SIM))
SCAL_N_ZONES = wp.constant(int(_defs.SCAL_N_ZONES))


@wp.func
def do_tether(e: int,
              tether_particle: wp.array(dtype=int),
              p_team: wp.array(dtype=int),
              next_positions: wp.array2d(dtype=float),
              velocity_positions: wp.array2d(dtype=float),
              step_basic_positions: wp.array2d(dtype=float),
              vertex_root: wp.array(dtype=int),
              t_tether_compression: wp.array(dtype=float)):
    idx = tether_particle[e]
    team = p_team[idx]
    root = vertex_root[idx]
    compression_limit = 1.0 - t_tether_compression[team]
    stretch_limit = 1.0 + TETHER_STRETCH_LIMIT

    vx = next_positions[root, 0] - next_positions[idx, 0]
    vy = next_positions[root, 1] - next_positions[idx, 1]
    vz = next_positions[root, 2] - next_positions[idx, 2]
    distance = dmath.length3(vx, vy, vz)
    cvx = step_basic_positions[idx, 0] - step_basic_positions[root, 0]
    cvy = step_basic_positions[idx, 1] - step_basic_positions[root, 1]
    cvz = step_basic_positions[idx, 2] - step_basic_positions[root, 2]
    calc_distance = dmath.length3(cvx, cvy, cvz)

    valid = (distance >= EPSILON) and (calc_distance != 0.0)
    ratio = distance / (calc_distance if calc_distance != 0.0 else 1.0)
    compress = valid and (ratio < compression_limit)
    stretch = valid and (ratio > stretch_limit)
    if not (compress or stretch):
        return
    if compress:
        dist = distance - compression_limit * calc_distance
        stiffness = dmath.saturate((compression_limit - ratio) / TETHER_STIFFNESS_WIDTH) \
            * TETHER_COMPRESSION_STIFFNESS
        attenuation = TETHER_COMPRESSION_VELOCITY_ATTENUATION
    else:
        dist = distance - stretch_limit * calc_distance
        stiffness = dmath.saturate((ratio - stretch_limit) / TETHER_STIFFNESS_WIDTH) \
            * TETHER_STRETCH_STIFFNESS
        attenuation = TETHER_STRETCH_VELOCITY_ATTENUATION
    inv = 1.0 / (distance if distance > 1.0e-30 else 1.0)
    scale = dist * stiffness * inv
    ax = vx * scale
    ay = vy * scale
    az = vz * scale
    next_positions[idx, 0] = next_positions[idx, 0] + ax
    next_positions[idx, 1] = next_positions[idx, 1] + ay
    next_positions[idx, 2] = next_positions[idx, 2] + az
    velocity_positions[idx, 0] = velocity_positions[idx, 0] + ax * attenuation
    velocity_positions[idx, 1] = velocity_positions[idx, 1] + ay * attenuation
    velocity_positions[idx, 2] = velocity_positions[idx, 2] + az * attenuation


@wp.func
def do_wind_blend(wind_main: float, time: float, dqx: float, dqy: float, dqz: float, dqw: float,
                  zone_turbulence: float,
                  blend: float, turbulence_param: float, wind_position: float):
    active = wind_main >= WIND_MIN_SPEED
    main_ratio = wind_main / WIND_BASE_SPEED

    sin_pos = wind_position + time * 10.0
    sin_wave = wp.sin(sin_pos)

    noise_pos = wind_position + time * 2.3132
    noise_wave = dmath.cnoise2(noise_pos, noise_pos) * 2.3

    wave_x = sin_wave + (noise_wave - sin_wave) * blend
    wave_y = wave_x

    turbulence = zone_turbulence * turbulence_param

    angle_x = (wave_x * WIND_TURBULENCE_ANGLE) * DEG2RAD
    angle_y = (wave_y * WIND_TURBULENCE_ANGLE) * DEG2RAD
    angle_y = angle_y * (0.1 + (0.5 - 0.1) * blend)
    angle_x = angle_x * turbulence
    angle_y = angle_y * turbulence

    rqx, rqy, rqz, rqw = dmath.euler_yx(angle_x, angle_y)
    cqx, cqy, cqz, cqw = dmath.quat_mul(dqx, dqy, dqz, dqw, rqx, rqy, rqz, rqw)
    wdx, wdy, wdz = dmath.quat_to_tangent(cqx, cqy, cqz, cqw)

    main_scale = dmath.saturate(1.0 - main_ratio)
    main_wave = (wave_x + 1.0) * 0.5
    main_wave = main_wave * main_scale * turbulence
    strength = wind_main - wind_main * main_wave
    if not active:
        strength = 0.0
    return (wdx * strength, wdy * strength, wdz * strength)


COLLIDER_MESH_FOOT_REASON = (
    "a closest point query hands back the point on the surface and the point that was "
    "asked about, and the direction between them is the direction the field grows in, "
    "except when the two are the same point, which is exactly what happens every time "
    "something lands on the surface on purpose; the difference is then float32 rounding "
    "noise of the coordinates it was subtracted from, and normalising noise gives a "
    "direction that has nothing to do with the surface, measured at minus nine tenths "
    "against the face it was standing on; below the resolution of those coordinates the "
    "direction is therefore taken from the face the query landed on, which is the same "
    "answer the incidence reading on this path already says it is, to the last float32 "
    "place, everywhere the foot is inside the triangle it reports")

COLLIDER_FIELD_REASON = (
    "every judgement a collider takes part in is a question about one scalar field, the "
    "signed distance from a point to that body at the pose the step ended on, and the "
    "direction that field grows in; the exit projection roots it, the bracket that root "
    "search needs marches on it, the narrow phase compares it against a thickness and "
    "pushes along its gradient, and a continuous test advances against it, so the four "
    "shapes answer that one question here and nothing downstream of this function knows "
    "how many shapes there are; the arrangement this replaced asked each shape the same "
    "geometric question in five places and derived the answer five times, which is how a "
    "bracket that disagreed with the field it bracketed survived long enough to push a "
    "particle fifty six million metres, and adding a fifth shape meant editing all five")


@wp.func
def _collider_field_sphere(c: int, px: float, py: float, pz: float,
                           c_work_next_pos: wp.array3d(dtype=float),
                           c_work_radius: wp.array2d(dtype=float)):
    vx = px - c_work_next_pos[c, 0, 0]
    vy = py - c_work_next_pos[c, 0, 1]
    vz = pz - c_work_next_pos[c, 0, 2]
    nx, ny, nz = dmath.normalize3_fb(vx, vy, vz, 0.0, 0.0, 1.0)
    return dmath.length3(vx, vy, vz) - c_work_radius[c, 0], nx, ny, nz


@wp.func
def _collider_field_capsule(c: int, px: float, py: float, pz: float,
                            c_work_next_pos: wp.array3d(dtype=float),
                            c_work_radius: wp.array2d(dtype=float)):
    sx = c_work_next_pos[c, 0, 0]
    sy = c_work_next_pos[c, 0, 1]
    sz = c_work_next_pos[c, 0, 2]
    ex = c_work_next_pos[c, 1, 0]
    ey = c_work_next_pos[c, 1, 1]
    ez = c_work_next_pos[c, 1, 2]
    start_radius = c_work_radius[c, 0]
    ratio = dmath.closest_pt_point_segment_ratio(px, py, pz, sx, sy, sz, ex, ey, ez)
    blended = start_radius + (c_work_radius[c, 1] - start_radius) * ratio
    vx = px - (sx + (ex - sx) * ratio)
    vy = py - (sy + (ey - sy) * ratio)
    vz = pz - (sz + (ez - sz) * ratio)
    nx, ny, nz = dmath.normalize3_fb(vx, vy, vz, 0.0, 0.0, 1.0)
    return dmath.length3(vx, vy, vz) - blended, nx, ny, nz


COLLIDER_PLANE_POSE_REASON = (
    "a plane is a pose like every other body, so the slot that holds where a body was at "
    "the start of the step holds where the plane was, and the direction it faces is read "
    "off the pose rotation where it is needed; that slot used to hold the plane's facing "
    "instead, which is the one thing that made a plane not answer the questions the other "
    "shapes answer, and it meant the step motion of a plane existed nowhere, so nothing "
    "could carry a point through it")


@wp.func
def _collider_field_plane(c: int, px: float, py: float, pz: float,
                          c_work_next_pos: wp.array3d(dtype=float),
                          c_work_rot: wp.array2d(dtype=float)):
    nx, ny, nz = dmath.quat_rotate(c_work_rot[c, 0], c_work_rot[c, 1], c_work_rot[c, 2],
                                   c_work_rot[c, 3], 0.0, 0.0, 1.0)
    ax = px - c_work_next_pos[c, 0, 0]
    ay = py - c_work_next_pos[c, 0, 1]
    az = pz - c_work_next_pos[c, 0, 2]
    return ax * nx + ay * ny + az * nz, nx, ny, nz


COLLIDER_MESH_PSEUDO_NORMAL_REASON = (
    "the sign of the distance to a surface is the side the point is on, and the direction "
    "that answers it is the angle weighted pseudo normal of Baerentzen and Aanaes, Signed "
    "Distance Computation Using the Angle Weighted Pseudonormal, IEEE Transactions on "
    "Visualization and Computer Graphics 11 number 3: the face normal where the closest "
    "point is strictly inside a triangle, the sum of the two adjoining face normals where "
    "it sits on an edge, and the sum over the fan of a corner of each face normal weighted "
    "by the angle that face opens at that corner where it sits on a corner; it is not the "
    "face normal in all three cases because a closest point on an edge stands at a right "
    "angle to both faces that meet there, so the offset it is dotted against carries no "
    "sign at all and a last place difference decides which side the point is on, which is "
    "a coin standing on its rim and not a small error; the self collision path in this "
    "project was measured on exactly that coin, where the incidence reading it stood on "
    "moved a translation residual from 0.0576 to 0.00355 once the barycentric interior was "
    "asked instead, and the collider path does not get to stand on it a second time")

COLLIDER_MESH_QUERY_REASON = (
    "the hierarchy answers which faces meet a box and nothing else, so a closest point is a "
    "box wide enough to hold the answer: a face whose closest point stands at distance d "
    "from the query point has a point of its own inside the cube of half width d around it, "
    "so a traversal at half width r either returns every face within r, and with them the "
    "closest one, or proves there is none within r at all; the search therefore opens at a "
    "sixty fourth of the body and doubles until it holds its own answer, which is one "
    "traversal for a point on the surface where every contact that matters is, and it stops "
    "at the distance its caller named whether it holds an answer or not, which is the whole "
    "of this change: what it used to stop at was the body's own diagonal, so a particle half "
    "a metre off a game LOD ended up testing every triangle of it, measured by lodperf.py on "
    "a twenty eight thousand triangle LOD at 184.29 ms a frame in the contact stage alone "
    "against 5.19 ms once the caller's distance became the stop, with the whole per frame "
    "cost the collider itself asks for at 0.29 ms; the caller can name that distance because "
    "every judgement a collider takes part in compares the field against a thickness and "
    "stops caring past it, and what comes back above it is that distance itself, which is a "
    "lower bound on the field and the safe direction for every caller: a continuous test "
    "told the surface is further off than it is advances by less than it could have and "
    "cannot step through anything, and a contact told the same reports a gap this step "
    "cannot close; the doubling stays because it is how a nearest point is found on a "
    "hierarchy that only answers boxes, and it is what makes a small answer cost a small "
    "traversal whatever the caller's distance is: opening at the caller's distance instead "
    "was measured, and it puts the exit projection, whose distance is legitimately the "
    "body's diagonal, at 974.82 ms a frame against 31.22 ms, because a box of that width "
    "holds the whole body every time")


COLLIDER_MESH_QUERY_WIDEN_REASON = (
    "a box that has already found a face is widened to that face's distance rather than "
    "doubled, because that distance is the smallest width that can still hold the answer: "
    "a face nearer than the one already held stands within that distance of the query "
    "point, so a point of its own stands within that distance on every axis, so its box "
    "overlaps a query box of that half width and the traversal returns it; the doubling "
    "overshoots instead, and the overshoot is what costs, because the faces a box returns "
    "grow with its width against a surface and the widths in question are a tenth of the "
    "body and up, measured by stage4_fieldshape.py on the frozen asset: of the hundred and "
    "sixty eight particles draped over a twenty eight thousand triangle game LOD, sixty six "
    "reach the fifth doubling and seven the sixth, where a box half the body wide holds "
    "nearly every triangle it has, and the exit projection over them reads 35.32 ms a frame "
    "doubling against 11.94 ms widening to the answer, with the same answers, since both "
    "loops leave through the same test and that test is a proof that the face held is the "
    "nearest one within the reach; the widening is also always the last turn of the loop, "
    "because the query it asks for returns the face whose distance it was given, so the "
    "test at the top of the next turn holds and the loop leaves, which is why the expansion "
    "budget the doubling needs is the budget this needs")


@wp.func
def collider_mesh_pseudo_normal(f: int, u: float, v: float, w: float,
                                cf_vertex: wp.array2d(dtype=int),
                                cf_edge_normal: wp.array3d(dtype=float),
                                cf_normal: wp.array2d(dtype=float),
                                cv_pseudo_normal: wp.array2d(dtype=float)):
    corner = int(-1)
    edge = int(-1)
    if v == 0.0 and w == 0.0:
        corner = 0
    elif w == 0.0 and u == 0.0:
        corner = 1
    elif u == 0.0 and v == 0.0:
        corner = 2
    elif w == 0.0:
        edge = 0
    elif u == 0.0:
        edge = 1
    elif v == 0.0:
        edge = 2
    if corner >= 0:
        row = cf_vertex[f, corner]
        return (cv_pseudo_normal[row, 0], cv_pseudo_normal[row, 1], cv_pseudo_normal[row, 2])
    if edge >= 0:
        return (cf_edge_normal[f, edge, 0], cf_edge_normal[f, edge, 1],
                cf_edge_normal[f, edge, 2])
    return (cf_normal[f, 0], cf_normal[f, 1], cf_normal[f, 2])


@wp.func
def _collider_field_mesh(c: int, px: float, py: float, pz: float, reach: float,
                         incidence_gate_cos: float,
                         c_work_next_pos: wp.array3d(dtype=float),
                         c_work_radius: wp.array2d(dtype=float),
                         c_work_rot: wp.array2d(dtype=float),
                         c_work_inv_rot: wp.array2d(dtype=float),
                         face_index: wp.uint64,
                         cf_vertex: wp.array2d(dtype=int),
                         cf_edge_normal: wp.array3d(dtype=float),
                         cf_normal: wp.array2d(dtype=float),
                         cv_local_position: wp.array2d(dtype=float),
                         cv_pseudo_normal: wp.array2d(dtype=float)):
    lx, ly, lz = dmath.quat_rotate(c_work_inv_rot[c, 0], c_work_inv_rot[c, 1],
                                   c_work_inv_rot[c, 2], c_work_inv_rot[c, 3],
                                   px - c_work_next_pos[c, 0, 0],
                                   py - c_work_next_pos[c, 0, 1],
                                   pz - c_work_next_pos[c, 0, 2])
    root = wp.bvh_get_group_root(face_index, c)
    box = c_work_radius[c, 0] * COLLIDER_MESH_QUERY_SEED_FRACTION
    span = float(wp.inf)
    best = int(-1)
    best_u = float(0.0)
    best_v = float(0.0)
    best_w = float(0.0)
    footx = float(0.0)
    footy = float(0.0)
    footz = float(0.0)
    for expansion in range(COLLIDER_MESH_QUERY_EXPANSIONS):
        if box > reach:
            box = reach
        query = wp.bvh_query_aabb(face_index, wp.vec3(lx - box, ly - box, lz - box),
                                  wp.vec3(lx + box, ly + box, lz + box), root)
        face = int(0)
        while wp.bvh_query_next(query, face):
            first = cf_vertex[face, 0]
            second = cf_vertex[face, 1]
            third = cf_vertex[face, 2]
            ptx, pty, ptz, u, v, w = dmath.closest_pt_point_triangle(
                lx, ly, lz,
                cv_local_position[first, 0], cv_local_position[first, 1],
                cv_local_position[first, 2],
                cv_local_position[second, 0], cv_local_position[second, 1],
                cv_local_position[second, 2],
                cv_local_position[third, 0], cv_local_position[third, 1],
                cv_local_position[third, 2])
            offx = lx - ptx
            offy = ly - pty
            offz = lz - ptz
            reading = dmath.length3(offx, offy, offz)
            if reading < span:
                span = reading
                best = face
                best_u = u
                best_v = v
                best_w = w
                footx = offx
                footy = offy
                footz = offz
        if best >= 0 and span <= box:
            break
        if box >= reach:
            break
        if best >= 0:
            box = span
        else:
            box = box + box
    if best < 0 or span > reach:
        return reach, 0.0, 0.0, 1.0
    sx, sy, sz = collider_mesh_pseudo_normal(best, best_u, best_v, best_w, cf_vertex,
                                             cf_edge_normal, cf_normal, cv_pseudo_normal)
    ox, oy, oz = dmath.normalize3_fb(sx, sy, sz, 0.0, 0.0, 1.0)
    away = float(1.0)
    if footx * sx + footy * sy + footz * sz < 0.0:
        away = -1.0
    if span <= COLLIDER_EXIT_TOLERANCE * (wp.abs(lx) + wp.abs(ly) + wp.abs(lz)
                                          + c_work_radius[c, 0]):
        ux = ox
        uy = oy
        uz = oz
    else:
        ux = footx * (away / span)
        uy = footy * (away / span)
        uz = footz * (away / span)
    if wp.abs(ux * ox + uy * oy + uz * oz) < incidence_gate_cos:
        return reach, 0.0, 0.0, 1.0
    nx, ny, nz = dmath.quat_rotate(c_work_rot[c, 0], c_work_rot[c, 1], c_work_rot[c, 2],
                                   c_work_rot[c, 3], ux, uy, uz)
    return span * away, nx, ny, nz


COLLIDER_BOUND_MISS_REASON = (
    "before any stage asks the field it asks whether the thing it is asking about can "
    "reach this body at all, which is the query point or the query segment grown by "
    "everything that stage cares about against the bound the body swept over the step; a "
    "miss there is a proof and not a filter, because the body is inside that bound at "
    "every instant of the step, so the stage answers far without opening a single "
    "traversal, and the case it answers is the one a game LOD makes common, a body the "
    "cloth is nowhere near that the cloth is still paired with; the test is written here "
    "once because it is the same test for the narrow phase and for the exit projection and "
    "the two used to be two copies of it, one of which did not exist")


@wp.func
def collider_bound_misses_point(c: int, px: float, py: float, pz: float, margin: float,
                                c_work_aabb_min: wp.array2d(dtype=float),
                                c_work_aabb_max: wp.array2d(dtype=float)):
    return (px - margin > c_work_aabb_max[c, 0]) or (px + margin < c_work_aabb_min[c, 0]) \
        or (py - margin > c_work_aabb_max[c, 1]) or (py + margin < c_work_aabb_min[c, 1]) \
        or (pz - margin > c_work_aabb_max[c, 2]) or (pz + margin < c_work_aabb_min[c, 2])


@wp.func
def collider_bound_misses_segment(c: int, p0x: float, p0y: float, p0z: float, r0: float,
                                  p1x: float, p1y: float, p1z: float, r1: float,
                                  margin: float,
                                  c_work_aabb_min: wp.array2d(dtype=float),
                                  c_work_aabb_max: wp.array2d(dtype=float)):
    return (dmath.fmin2(p0x - r0, p1x - r1) - margin > c_work_aabb_max[c, 0]) \
        or (dmath.fmax2(p0x + r0, p1x + r1) + margin < c_work_aabb_min[c, 0]) \
        or (dmath.fmin2(p0y - r0, p1y - r1) - margin > c_work_aabb_max[c, 1]) \
        or (dmath.fmax2(p0y + r0, p1y + r1) + margin < c_work_aabb_min[c, 1]) \
        or (dmath.fmin2(p0z - r0, p1z - r1) - margin > c_work_aabb_max[c, 2]) \
        or (dmath.fmax2(p0z + r0, p1z + r1) + margin < c_work_aabb_min[c, 2])


COLLIDER_BOUND_SWEEP_REASON = (
    "what a contact stage asks about is where a primitive went over the step and not where "
    "it ended, so the bound test in front of it is asked about that whole travel; testing "
    "the end alone refuses a particle that started on one side of a body, ended past the "
    "margin on the other and crossed it in between, which is the one case the continuous "
    "walk behind this test exists for, and it refused it before the walk began, so nothing "
    "downstream could see it and no reading of the contact could go red; tunnelcheck.py "
    "could not see it either, because it drives the body and leaves the cloth still, and a "
    "body's own swept bound always holds the still cloth's one position, so the case only "
    "opens when the cloth is the thing that moves, which is the ladder that file now "
    "carries; the box is taken in the frame the body ended the step in, which is the frame "
    "collider_transport names the start point in and the frame the field is asked in, so "
    "the body is one shape and not two")


COLLIDER_BOUND_INWARD_REASON = (
    "a bounded query proves there is no surface within the distance it searched and it "
    "cannot tell which side of the surface it is on when it finds none, so a point standing "
    "deeper inside a body than the distance its caller named comes back as far when it is "
    "as close to the surface as a body can put anything; what bounds that depth is the "
    "body's own bound, because a ray from a point inside the body to the nearest face of "
    "that bound leaves the body at or before it leaves the bound, so the surface is no "
    "further away than that face is, and the contact stage adds that distance to what it "
    "asks for, which is the wider of that depth and what the step itself reaches and not "
    "their sum, because either one alone is enough for the reading to be true; a point "
    "outside the bound adds nothing, which is the case a body the cloth "
    "is nowhere near is in, and a surface that does not close is bounded by none of this, "
    "because the side of such a surface the pseudo normal calls in reaches wherever the "
    "surface does not, which is why the exit projection and not the contact stage is the "
    "one that carries the unbounded case and names the reach the body declares")


@wp.func
def collider_bound_inward(c: int, px: float, py: float, pz: float,
                          c_work_aabb_min: wp.array2d(dtype=float),
                          c_work_aabb_max: wp.array2d(dtype=float)):
    inward = dmath.fmin2(px - c_work_aabb_min[c, 0], c_work_aabb_max[c, 0] - px)
    inward = dmath.fmin2(inward, dmath.fmin2(py - c_work_aabb_min[c, 1],
                                             c_work_aabb_max[c, 1] - py))
    inward = dmath.fmin2(inward, dmath.fmin2(pz - c_work_aabb_min[c, 2],
                                             c_work_aabb_max[c, 2] - pz))
    return dmath.fmax2(inward, 0.0)


@wp.func
def collider_bound_misses_edge_sweep(c: int, e0: int, e1: int, r0: float, r1: float,
                                     margin: float,
                                     p_next_positions: wp.array2d(dtype=float),
                                     p_old_positions: wp.array2d(dtype=float),
                                     c_work_old_pos: wp.array3d(dtype=float),
                                     c_work_next_pos: wp.array3d(dtype=float),
                                     c_work_rot: wp.array2d(dtype=float),
                                     c_work_inv_old_rot: wp.array2d(dtype=float),
                                     c_work_aabb_min: wp.array2d(dtype=float),
                                     c_work_aabb_max: wp.array2d(dtype=float)):
    s0x, s0y, s0z = collider_transport(c, p_old_positions[e0, 0], p_old_positions[e0, 1],
                                       p_old_positions[e0, 2], c_work_old_pos,
                                       c_work_next_pos, c_work_rot, c_work_inv_old_rot)
    s1x, s1y, s1z = collider_transport(c, p_old_positions[e1, 0], p_old_positions[e1, 1],
                                       p_old_positions[e1, 2], c_work_old_pos,
                                       c_work_next_pos, c_work_rot, c_work_inv_old_rot)
    lowx = dmath.fmin2(dmath.fmin2(p_next_positions[e0, 0] - r0,
                                   p_next_positions[e1, 0] - r1),
                       dmath.fmin2(s0x - r0, s1x - r1))
    lowy = dmath.fmin2(dmath.fmin2(p_next_positions[e0, 1] - r0,
                                   p_next_positions[e1, 1] - r1),
                       dmath.fmin2(s0y - r0, s1y - r1))
    lowz = dmath.fmin2(dmath.fmin2(p_next_positions[e0, 2] - r0,
                                   p_next_positions[e1, 2] - r1),
                       dmath.fmin2(s0z - r0, s1z - r1))
    highx = dmath.fmax2(dmath.fmax2(p_next_positions[e0, 0] + r0,
                                    p_next_positions[e1, 0] + r1),
                        dmath.fmax2(s0x + r0, s1x + r1))
    highy = dmath.fmax2(dmath.fmax2(p_next_positions[e0, 1] + r0,
                                    p_next_positions[e1, 1] + r1),
                        dmath.fmax2(s0y + r0, s1y + r1))
    highz = dmath.fmax2(dmath.fmax2(p_next_positions[e0, 2] + r0,
                                    p_next_positions[e1, 2] + r1),
                        dmath.fmax2(s0z + r0, s1z + r1))
    return collider_bound_misses_segment(c, lowx, lowy, lowz, 0.0, highx, highy, highz,
                                         0.0, margin, c_work_aabb_min, c_work_aabb_max)


COLLIDER_FACE_EMPTY_LOW = wp.constant(1.0e30)

COLLIDER_FACE_EMPTY_HIGH = wp.constant(-1.0e30)

COLLIDER_FACE_EMPTY_BOUND_REASON = (
    "the face slab is shared by every collider of every team and a row nobody owns still "
    "sits in it, carrying the corner indices a released block left behind, which name a "
    "triangle standing on one vertex somewhere inside another body; such a row is given a "
    "bound whose lower corner is above its upper one, which no query box can overlap and "
    "which raises no parent bound of the hierarchy either, because a parent takes the "
    "smallest lower and the largest upper of its children and an inverted box loses both "
    "of those comparisons; that is what lets a dead row be dropped without a test in the "
    "narrow phase and without a live flag beside it")


COLLIDER_MESH_EDGE_RING_LIMIT = wp.constant(int(_defs.COLLIDER_MESH_EDGE_RING_LIMIT))

COLLIDER_MESH_EDGE_SUM_REASON = (
    "the pseudo normal of an edge is the sum of the normals of the faces that meet on it, "
    "and the narrow phase reads that sum rather than building it, because the narrow phase "
    "is the one place in the frame whose register count decides how many warps the device "
    "can keep in flight; the sum is therefore built here, where every face is already "
    "being visited to have its own normal written, by walking the ring of half edges that "
    "share the edge and adding a normal for each one, which is one step for a rim, two for "
    "a surface a person models and however many are really there where a surface branches; "
    "a face on the ring has its normal worked out again rather than read back, because the "
    "pass that writes them is this one and a read would be a read of whatever the "
    "neighbouring thread had reached")


@wp.func
def collider_face_normal(f: int,
                         cf_vertex: wp.array2d(dtype=int),
                         cv_local_position: wp.array2d(dtype=float)):
    first = cf_vertex[f, 0]
    second = cf_vertex[f, 1]
    third = cf_vertex[f, 2]
    ax = cv_local_position[second, 0] - cv_local_position[first, 0]
    ay = cv_local_position[second, 1] - cv_local_position[first, 1]
    az = cv_local_position[second, 2] - cv_local_position[first, 2]
    bx = cv_local_position[third, 0] - cv_local_position[first, 0]
    by = cv_local_position[third, 1] - cv_local_position[first, 1]
    bz = cv_local_position[third, 2] - cv_local_position[first, 2]
    return dmath.normalize3_fb(ay * bz - az * by, az * bx - ax * bz,
                               ax * by - ay * bx, 0.0, 0.0, 1.0)


@wp.func
def do_collider_face_primitive(f: int,
                               cf_team: wp.array(dtype=int),
                               cf_vertex: wp.array2d(dtype=int),
                               cf_edge_ring_face: wp.array2d(dtype=int),
                               cf_edge_ring_corner: wp.array2d(dtype=int),
                               cf_aabb_min: wp.array2d(dtype=float),
                               cf_aabb_max: wp.array2d(dtype=float),
                               cf_normal: wp.array2d(dtype=float),
                               cf_edge_normal: wp.array3d(dtype=float),
                               cv_local_position: wp.array2d(dtype=float),
                               t_enabled: wp.array(dtype=int),
                               t_valid: wp.array(dtype=int),
                               t_cws: wp.array2d(dtype=float)):
    if not team_frame_mask(t_enabled, t_valid, t_cws, cf_team[f]):
        for axis in range(3):
            cf_aabb_min[f, axis] = COLLIDER_FACE_EMPTY_LOW
            cf_aabb_max[f, axis] = COLLIDER_FACE_EMPTY_HIGH
        return
    first = cf_vertex[f, 0]
    second = cf_vertex[f, 1]
    third = cf_vertex[f, 2]
    for axis in range(3):
        low = cv_local_position[first, axis]
        high = low
        held = cv_local_position[second, axis]
        if held < low:
            low = held
        if held > high:
            high = held
        held = cv_local_position[third, axis]
        if held < low:
            low = held
        if held > high:
            high = held
        cf_aabb_min[f, axis] = low
        cf_aabb_max[f, axis] = high
    nx, ny, nz = collider_face_normal(f, cf_vertex, cv_local_position)
    cf_normal[f, 0] = nx
    cf_normal[f, 1] = ny
    cf_normal[f, 2] = nz
    for edge in range(3):
        sumx = nx
        sumy = ny
        sumz = nz
        face = cf_edge_ring_face[f, edge]
        corner = cf_edge_ring_corner[f, edge]
        for _step in range(COLLIDER_MESH_EDGE_RING_LIMIT):
            if face == f and corner == edge:
                break
            ringx, ringy, ringz = collider_face_normal(face, cf_vertex, cv_local_position)
            sumx = sumx + ringx
            sumy = sumy + ringy
            sumz = sumz + ringz
            ahead_face = cf_edge_ring_face[face, corner]
            corner = cf_edge_ring_corner[face, corner]
            face = ahead_face
        cf_edge_normal[f, edge, 0] = sumx
        cf_edge_normal[f, edge, 1] = sumy
        cf_edge_normal[f, edge, 2] = sumz


COLLIDER_MESH_VERTEX_FAN_LIMIT = wp.constant(int(_defs.COLLIDER_MESH_VERTEX_FAN_LIMIT))


@wp.func
def do_collider_vertex_pseudo_normal(v: int,
                                     cv_team: wp.array(dtype=int),
                                     cv_fan_face: wp.array(dtype=int),
                                     cv_fan_corner: wp.array(dtype=int),
                                     cv_local_position: wp.array2d(dtype=float),
                                     cv_pseudo_normal: wp.array2d(dtype=float),
                                     cf_vertex: wp.array2d(dtype=int),
                                     cf_fan_next_face: wp.array2d(dtype=int),
                                     cf_fan_next_corner: wp.array2d(dtype=int),
                                     cf_normal: wp.array2d(dtype=float),
                                     t_enabled: wp.array(dtype=int),
                                     t_valid: wp.array(dtype=int),
                                     t_cws: wp.array2d(dtype=float)):
    sumx = float(0.0)
    sumy = float(0.0)
    sumz = float(0.0)
    seed_face = cv_fan_face[v]
    if seed_face >= 0 and team_frame_mask(t_enabled, t_valid, t_cws, cv_team[v]):
        seed_corner = cv_fan_corner[v]
        face = seed_face
        corner = seed_corner
        for _step in range(COLLIDER_MESH_VERTEX_FAN_LIMIT):
            ahead_corner = corner + 1
            if ahead_corner == 3:
                ahead_corner = 0
            behind_corner = ahead_corner + 1
            if behind_corner == 3:
                behind_corner = 0
            origin = cf_vertex[face, corner]
            ahead = cf_vertex[face, ahead_corner]
            behind = cf_vertex[face, behind_corner]
            ax = cv_local_position[ahead, 0] - cv_local_position[origin, 0]
            ay = cv_local_position[ahead, 1] - cv_local_position[origin, 1]
            az = cv_local_position[ahead, 2] - cv_local_position[origin, 2]
            bx = cv_local_position[behind, 0] - cv_local_position[origin, 0]
            by = cv_local_position[behind, 1] - cv_local_position[origin, 1]
            bz = cv_local_position[behind, 2] - cv_local_position[origin, 2]
            opening = wp.atan2(dmath.length3(ay * bz - az * by, az * bx - ax * bz,
                                             ax * by - ay * bx),
                               ax * bx + ay * by + az * bz)
            sumx = sumx + opening * cf_normal[face, 0]
            sumy = sumy + opening * cf_normal[face, 1]
            sumz = sumz + opening * cf_normal[face, 2]
            ahead_face = cf_fan_next_face[face, corner]
            corner = cf_fan_next_corner[face, corner]
            face = ahead_face
            if face == seed_face and corner == seed_corner:
                break
    cv_pseudo_normal[v, 0] = sumx
    cv_pseudo_normal[v, 1] = sumy
    cv_pseudo_normal[v, 2] = sumz


COLLIDER_MESH_QUERY_SEED_FRACTION = wp.constant(float(_defs.COLLIDER_MESH_QUERY_SEED_FRACTION))

COLLIDER_MESH_QUERY_EXPANSIONS = wp.constant(int(_defs.COLLIDER_MESH_QUERY_EXPANSIONS))

COLLIDER_FIELD_REACH_REASON = (
    "the field takes the distance the caller still cares about as an argument and not as a "
    "property of the shape, because the distance that matters is a property of the "
    "judgement being made and never of the body it is made against: the continuous test "
    "cares out to the offset it is testing plus the travel that is left in the step, the "
    "foot search along a cloth edge cares out to that plus the length of the edge, and the "
    "exit projection cares out to the deepest a point could be inside the body; a shape "
    "that can answer everywhere ignores it, a shape that answers by searching a box uses "
    "it as the half width of that box, and what comes back is the true field where the "
    "true field is below the argument and the argument itself where it is not, which is a "
    "lower bound and never an over estimate")


@wp.func
def collider_field(c: int, px: float, py: float, pz: float, reach: float,
                   incidence_gate_cos: float,
                   c_kind: wp.array(dtype=int),
                   c_work_next_pos: wp.array3d(dtype=float),
                   c_work_radius: wp.array2d(dtype=float),
                   c_work_rot: wp.array2d(dtype=float),
                   c_work_inv_rot: wp.array2d(dtype=float),
                   face_index: wp.uint64,
                   cf_vertex: wp.array2d(dtype=int),
                   cf_edge_normal: wp.array3d(dtype=float),
                   cf_normal: wp.array2d(dtype=float),
                   cv_local_position: wp.array2d(dtype=float),
                   cv_pseudo_normal: wp.array2d(dtype=float)):
    kind = c_kind[c]
    if kind == COLLIDER_SPHERE:
        return _collider_field_sphere(c, px, py, pz, c_work_next_pos, c_work_radius)
    if kind == COLLIDER_CAPSULE:
        return _collider_field_capsule(c, px, py, pz, c_work_next_pos, c_work_radius)
    if kind == COLLIDER_MESH:
        return _collider_field_mesh(c, px, py, pz, reach, incidence_gate_cos,
                                    c_work_next_pos,
                                    c_work_radius, c_work_rot, c_work_inv_rot,
                                    face_index, cf_vertex, cf_edge_normal,
                                    cf_normal, cv_local_position, cv_pseudo_normal)
    return _collider_field_plane(c, px, py, pz, c_work_next_pos, c_work_rot)


COLLIDER_TRANSPORT_REASON = (
    "a body that moved during the step carries the space around it along, so the place a "
    "particle started from has to be named in the frame the step ended in before the two "
    "can be compared; that is the pose the step ended at applied to the inverse of the "
    "pose it started at, which is one rigid map and knows nothing about which shape it is "
    "carrying, and it is what turns a moving body into a still one for everything "
    "downstream, so the field only ever has to answer about the pose the step ended on")


@wp.func
def collider_transport(c: int, px: float, py: float, pz: float,
                       c_work_old_pos: wp.array3d(dtype=float),
                       c_work_next_pos: wp.array3d(dtype=float),
                       c_work_rot: wp.array2d(dtype=float),
                       c_work_inv_old_rot: wp.array2d(dtype=float)):
    lx, ly, lz = dmath.quat_rotate(c_work_inv_old_rot[c, 0], c_work_inv_old_rot[c, 1],
                                   c_work_inv_old_rot[c, 2], c_work_inv_old_rot[c, 3],
                                   px - c_work_old_pos[c, 0, 0],
                                   py - c_work_old_pos[c, 0, 1],
                                   pz - c_work_old_pos[c, 0, 2])
    wx, wy, wz = dmath.quat_rotate(c_work_rot[c, 0], c_work_rot[c, 1], c_work_rot[c, 2],
                                   c_work_rot[c, 3], lx, ly, lz)
    return (c_work_next_pos[c, 0, 0] + wx, c_work_next_pos[c, 0, 1] + wy,
            c_work_next_pos[c, 0, 2] + wz)


ACCD_REASON = (
    "the contact search is additive continuous collision detection, from Li, Kaufman and "
    "Jiang, Codimensional Incremental Potential Contact, ACM Transactions on Graphics 40 "
    "number 4 article 170, section 5.4 and Algorithm 1; it takes the distance between the "
    "two primitives at the start of the step and divides it by a bound on how fast that "
    "distance can shrink, which is a time by which nothing can have touched, advances "
    "everything by that much and repeats, so the time of impact is approached from below "
    "by a sum of steps that are each certainly safe and no polynomial is ever solved; the "
    "paper needs that bound because its primitives deform and each of the four nodes "
    "carries its own displacement, while here the second primitive is a rigid body, so "
    "collider_transport names the start point in the frame the body ended the step in and "
    "the body stops moving, which leaves one moving point and makes the bound the length "
    "of that point's own relative travel; Algorithm 1 works in squared distances and "
    "spells its distance to the offset surface as the squared distance less the squared "
    "offset over the root of the squared distance plus the offset, which is a form that "
    "cancels well in floating point, and that expression is the plain difference of the "
    "distance and the offset, which is what the field here already returns, so it is "
    "written as the difference; the one deliberate departure is that the advanced point is "
    "recomputed from the start point and the accumulated time rather than accumulated onto "
    "itself as in lines 13 and 14, which is the same value in exact arithmetic and keeps "
    "the point and the time it stands for from drifting apart in float32, and drifting "
    "apart is exactly what would break the guarantee")

ACCD_SEPARATION_SCALE = wp.constant(float(_defs.ACCD_SEPARATION_SCALE))

ACCD_ADVANCE_SCALE = wp.constant(float(_defs.ACCD_ADVANCE_SCALE))

ACCD_STEP_LIMIT = wp.constant(int(_defs.ACCD_STEP_LIMIT))


ACCD_SINGLE_SITE_REASON = (
    "warp offers no control over inlining and no way to pass a flag through to the "
    "compiler, so every place this file names the field is a whole copy of the four shape "
    "join and, inside it, a whole copy of the mesh closest point query, which measures at "
    "seventy seven registers on its own against thirty one for an analytic shape; naming "
    "the field twice in one function therefore costs about eighty registers on top of the "
    "first, measured entry by entry with kernelregs.py, which is why the walk names it once "
    "and hands back the field and the gradient it read along with the time it proved safe, "
    "rather than being asked about them a second time by whoever called it; the same "
    "measurement is why the walk carries the resting case, where the point already sits "
    "inside the offset surface when the step begins, as further turns of its own loop "
    "instead of as a second reading")


ACCD_RESTING_MARCH_REASON = (
    "a point that begins the step already inside the offset surface cannot be asked the "
    "question the additive test asks, because the distance it would divide by is not "
    "positive; what that point needs proving is a different thing, that the step does not "
    "carry it through the body and out the far side, and the surface it must not cross to "
    "do that is the body itself and not the offset around it: the field is the true "
    "distance to that surface with a sign on it, a signed distance changes by at most one "
    "metre per metre, so a point standing that distance from the surface can travel that "
    "far and still be on the side it started on, whatever the shape does in between; that "
    "is the same inequality the additive test uses outside and it is the one sphere "
    "tracing is built on, so the resting case is the same loop advancing by the same "
    "reading, and the walk names the field once either way; the bound has to be the "
    "distance to the surface and not the distance to the offset surface, because a point "
    "resting on a body sits on the offset surface exactly, so a bound measured to the "
    "offset surface is nought there and would refuse a draped particle every millimetre "
    "of its travel along the body it is draped on, while the distance to the body is the "
    "whole offset there and carries the whole step in one turn, which is what leaves every "
    "resting contact the engine already had reading exactly as it did; the walk stops "
    "early on the same separation scale the additive test stops on, read against the "
    "distance to the surface the step opened at, and when it stops short the time it hands "
    "back is the largest it proved, so the point is held at the surface it reached and the "
    "contact reports the travel it was refused, which is what carries a body thin enough "
    "for a step to cross it, measured by the third table of tunnelcheck.py")


ACCD_REACH_REASON = (
    "what the walk asks the field for at each of its samples is bounded by the walk itself: "
    "a signed distance changes by at most one metre per metre, so a point whose field reads "
    "at least the offset, plus the band the caller still cares about past the offset, plus "
    "the travel it has left in this step, cannot come within that offset or that band "
    "before the step ends, whatever the shape does in between; that is a proof and not a "
    "guess, so on such a reading the walk is over at once, the whole step is safe, and the "
    "field it hands back is the far answer, which is the reading a body the cloth is "
    "nowhere near gives at the first sample and is where the two hundred milliseconds a "
    "game LOD used to cost went; the band is passed in rather than assumed to be the offset "
    "because the two are the same number on the point path and are not on the edge path, "
    "where the offset is the radius interpolated to the foot and the band is the mean of "
    "the two ends, and a band read too small would silently drop the friction reading of a "
    "contact that is inside it")


@wp.func
def collider_walk(c: int, ax: float, ay: float, az: float,
                  vx: float, vy: float, vz: float, travel: float, thickness: float,
                  band: float, incidence_gate_cos: float,
                  c_kind: wp.array(dtype=int),
                  c_work_next_pos: wp.array3d(dtype=float),
                  c_work_radius: wp.array2d(dtype=float),
                  c_work_rot: wp.array2d(dtype=float),
                  c_work_inv_rot: wp.array2d(dtype=float),
                  c_work_aabb_min: wp.array2d(dtype=float),
                  c_work_aabb_max: wp.array2d(dtype=float),
                  face_index: wp.uint64,
                  cf_vertex: wp.array2d(dtype=int),
                  cf_edge_normal: wp.array3d(dtype=float),
                  cf_normal: wp.array2d(dtype=float),
                  cv_local_position: wp.array2d(dtype=float),
                  cv_pseudo_normal: wp.array2d(dtype=float)):
    moment = float(0.0)
    ahead = float(0.0)
    target = float(0.0)
    resting = wp.bool(False)
    field = float(0.0)
    nx = float(0.0)
    ny = float(0.0)
    nz = float(1.0)
    reach = travel + dmath.fmax2(thickness + band,
                                 collider_bound_inward(c, ax, ay, az, c_work_aabb_min,
                                                       c_work_aabb_max))
    for walk in range(ACCD_STEP_LIMIT):
        field, nx, ny, nz = collider_field(c, ax + vx * ahead, ay + vy * ahead,
                                           az + vz * ahead, reach, incidence_gate_cos,
                                           c_kind,
                                           c_work_next_pos, c_work_radius, c_work_rot,
                                           c_work_inv_rot, face_index, cf_vertex,
                                           cf_edge_normal, cf_normal,
                                           cv_local_position, cv_pseudo_normal)
        if field >= reach:
            moment = 1.0
            field = float(wp.inf)
            break
        gap = field - thickness
        if walk == 0:
            if travel <= 0.0:
                resting = True
                moment = 1.0
                ahead = 1.0
            elif gap <= 0.0:
                resting = True
                target = ACCD_SEPARATION_SCALE * wp.abs(field)
                ahead = wp.abs(field) / travel
                if ahead > 1.0:
                    ahead = 1.0
            else:
                target = ACCD_SEPARATION_SCALE * gap
                ahead = (1.0 - ACCD_SEPARATION_SCALE) * gap / travel
                if ahead > 1.0:
                    ahead = 1.0
        elif resting:
            moment = ahead
            if ahead >= 1.0:
                break
            if wp.abs(field) <= target:
                break
            ahead = moment + wp.abs(field) / travel
            if ahead > 1.0:
                ahead = 1.0
        elif moment > 0.0 and gap < target:
            break
        else:
            moment = ahead
            if ahead >= 1.0:
                break
            ahead = moment + ACCD_ADVANCE_SCALE * gap / travel
            if ahead > 1.0:
                ahead = 1.0
    return moment, field, nx, ny, nz


COLLIDER_CONTACT_REASON = (
    "one contact against one body, in the two numbers everything downstream of it needs: "
    "where the point is allowed to be, and how far that place is from the offset surface; "
    "the point is carried to the time of impact rather than pushed out of wherever it "
    "landed, so a body that swept past it stops it on the way in instead of picking a side "
    "afterwards, and the gap is the field at the far end of the walk less the offset less "
    "the travel the point was refused, which is a lower bound on the gap it would have had "
    "if it had finished the travel because a signed distance field changes by at most one "
    "metre per metre; that makes one expression cover a resting contact the whole step "
    "belongs to, where nothing is refused and the walk ends at the end of the step so the "
    "gap is exact, a resting contact the walk stopped short on, where the point is left "
    "standing on the body it was about to be carried through and the travel it was refused "
    "is the depth reported, and a swept one, where the whole depth the point wanted is "
    "what the contact reports; none of it knows which shape it is standing on")


@wp.func
def collider_contact(c: int, ax: float, ay: float, az: float,
                     bx: float, by: float, bz: float, thickness: float, band: float,
                     incidence_gate_cos: float,
                     c_kind: wp.array(dtype=int),
                     c_work_next_pos: wp.array3d(dtype=float),
                     c_work_radius: wp.array2d(dtype=float),
                     c_work_rot: wp.array2d(dtype=float),
                     c_work_inv_rot: wp.array2d(dtype=float),
                     c_work_aabb_min: wp.array2d(dtype=float),
                     c_work_aabb_max: wp.array2d(dtype=float),
                     face_index: wp.uint64,
                     cf_vertex: wp.array2d(dtype=int),
                     cf_edge_normal: wp.array3d(dtype=float),
                     cf_normal: wp.array2d(dtype=float),
                     cv_local_position: wp.array2d(dtype=float),
                     cv_pseudo_normal: wp.array2d(dtype=float)):
    vx = bx - ax
    vy = by - ay
    vz = bz - az
    travel = dmath.length3(vx, vy, vz)
    moment, field, nx, ny, nz = collider_walk(
        c, ax, ay, az, vx, vy, vz, travel, thickness, band, incidence_gate_cos, c_kind,
        c_work_next_pos, c_work_radius, c_work_rot, c_work_inv_rot, c_work_aabb_min,
        c_work_aabb_max, face_index,
        cf_vertex, cf_edge_normal, cf_normal, cv_local_position, cv_pseudo_normal)
    push = dmath.fmin2(field - thickness, 0.0)
    return (field - thickness - travel * (1.0 - moment),
            ax + vx * moment - nx * push, ay + vy * moment - ny * push,
            az + vz * moment - nz * push, nx, ny, nz)


COLLIDER_EDGE_FOOT_REASON = (
    "an edge touches a body wherever the field is smallest along it, and that place is "
    "found on the field itself rather than by a closest point formula written once per "
    "shape; the field restricted to the segment has the direction of the segment dotted "
    "into the gradient as its slope, so the sign of that one number says which half of the "
    "span the smallest value is in, and halving on it walks to the foot with no geometry "
    "beyond the field; the search is what keeps a body thinner than the spacing between "
    "two particles from passing between them, which is the whole reason the edge path "
    "exists, and it is the part the previous arrangement wrote out three times and then "
    "did not write at all for a mesh, where it fell back to the two ends; it is a search "
    "and not one reading because one reading of the field names the nearest point on the "
    "surface and that point dropped onto the edge is not where the field is smallest: on "
    "a plane it lands in the middle of the edge every time while the smallest value is "
    "always at an end, measured by edgefoot.py at half an edge of error, and on a ball it "
    "is out by a fifth of an edge; and it opens with a scan because halving on the slope "
    "walks to a stationary point rather than to the smallest one, and a capsule gives an "
    "edge two of them, one at each end cap, so the halving would settle on whichever it "
    "started next to, measured by edgefoot.py at very nearly a whole edge of error on the "
    "tail and at thirty seven millimetres of field value, which is worse than the exact "
    "segment to segment closest point this replaced; the scan is coarse and the halving "
    "refines inside the one cell the scan picked, so the two together are one naming of "
    "the field in one loop and the register cost is the loop and not the samples; what is "
    "returned is the place with the smallest field value of every place the loop looked at "
    "rather than the place the halving converged on, because the halving converges on a "
    "stationary point and a capsule can put a second one inside the cell the scan picked, "
    "which left a measured worst ratio error of 0.73 of an edge at 1.9 millimetres of field "
    "value; every one of those samples was already read and its value already computed, so "
    "keeping the smallest costs no further reading of the field and the answer can never be "
    "worse than the best sample taken")

COLLIDER_EDGE_SCAN_STEPS = wp.constant(int(_defs.COLLIDER_EDGE_SCAN_STEPS))

COLLIDER_EDGE_SEARCH_STEPS = wp.constant(int(_defs.COLLIDER_EDGE_SEARCH_STEPS))


COLLIDER_EDGE_FOOT_REACH_REASON = (
    "the foot only has to be right on an edge that can touch, and an edge can touch when "
    "the smallest field along it is at most the offset plus the band; the field changes by "
    "at most one metre per metre and no two places on the edge are further apart than its "
    "length, so on such an edge every sample the search takes reads at most that sum plus "
    "that length, and asking for exactly that sum plus that length makes every sample of "
    "every edge that can touch read truly, while an edge whose samples all come back at "
    "the bound is an edge that cannot touch and whose foot therefore decides nothing, the "
    "contact stage measured at that foot refusing it on its own reading")


@wp.func
def collider_edge_foot(c: int, a0x: float, a0y: float, a0z: float,
                       a1x: float, a1y: float, a1z: float, reach: float,
                       incidence_gate_cos: float,
                       c_kind: wp.array(dtype=int),
                       c_work_next_pos: wp.array3d(dtype=float),
                       c_work_radius: wp.array2d(dtype=float),
                       c_work_rot: wp.array2d(dtype=float),
                       c_work_inv_rot: wp.array2d(dtype=float),
                       face_index: wp.uint64,
                       cf_vertex: wp.array2d(dtype=int),
                       cf_edge_normal: wp.array3d(dtype=float),
                       cf_normal: wp.array2d(dtype=float),
                       cv_local_position: wp.array2d(dtype=float),
                       cv_pseudo_normal: wp.array2d(dtype=float)):
    ex = a1x - a0x
    ey = a1y - a0y
    ez = a1z - a0z
    cell = 1.0 / float(COLLIDER_EDGE_SCAN_STEPS - 1)
    lower = float(0.0)
    upper = float(1.0)
    best = float(0.0)
    smallest = float(wp.inf)
    for walk in range(COLLIDER_EDGE_SCAN_STEPS + COLLIDER_EDGE_SEARCH_STEPS):
        if walk < COLLIDER_EDGE_SCAN_STEPS:
            middle = float(walk) * cell
        else:
            middle = 0.5 * (lower + upper)
        field, nx, ny, nz = collider_field(c, a0x + ex * middle, a0y + ey * middle,
                                           a0z + ez * middle, reach, incidence_gate_cos,
                                           c_kind,
                                           c_work_next_pos, c_work_radius, c_work_rot,
                                           c_work_inv_rot, face_index, cf_vertex,
                                           cf_edge_normal, cf_normal,
                                           cv_local_position, cv_pseudo_normal)
        if field < smallest:
            smallest = field
            best = middle
        if walk < COLLIDER_EDGE_SCAN_STEPS:
            if walk == COLLIDER_EDGE_SCAN_STEPS - 1:
                lower = dmath.fmax2(best - cell, 0.0)
                upper = dmath.fmin2(best + cell, 1.0)
        elif nx * ex + ny * ey + nz * ez > 0.0:
            upper = middle
        else:
            lower = middle
    return best


REFLECTION_TELEPORT_FLIP = wp.constant(-1.0)

REFLECTION_TELEPORT_FLIP_REASON = (
    "the teleport matrix is the new component basis times the inverse of the old one, so the "
    "sign of its determinant is the product of the two reflection bits, and the branch that "
    "applies it runs only on the frame those two bits differ, so that product is minus one "
    "there and the branch is never entered when it is plus one; an orientation reversing map "
    "sends an orthonormal frame to a left handed one, and the proper rotation that stands "
    "for it flips the normal and the tangent together and leaves the binormal, which is why "
    "one number covers both and why it is the same number for a particle frame and for a "
    "collider frame, the two having been given different flips before this was derived")


@wp.func
def component_reflection_sign(reflected: wp.array(dtype=int), i: int):
    if reflected[i] != 0:
        return -1.0
    return 1.0


@wp.func
def team_frame_mask(enabled: wp.array(dtype=int), valid: wp.array(dtype=int),
                    cws: wp.array2d(dtype=float), i: int):
    if enabled[i] == 0 or valid[i] == 0:
        return False
    ax = cws[i, 0]
    ay = cws[i, 1]
    az = cws[i, 2]
    lo = ax
    if ay < lo:
        lo = ay
    if az < lo:
        lo = az
    return lo >= COMPONENT_SCALE_EPSILON


@wp.func
def _skin_row(world: wp.array3d(dtype=float), bind: wp.array3d(dtype=float),
              t: int, r: int, c: int):
    return (world[t, r, 0] * bind[t, 0, c] + world[t, r, 1] * bind[t, 1, c]
            + world[t, r, 2] * bind[t, 2, c] + world[t, r, 3] * bind[t, 3, c])


@wp.func
def do_base_pose(p: int, p_team: wp.array(dtype=int),
                 local_positions: wp.array2d(dtype=float),
                 local_normals: wp.array2d(dtype=float),
                 local_tangents: wp.array2d(dtype=float),
                 skin_indices: wp.array2d(dtype=int),
                 skin_weights: wp.array2d(dtype=float),
                 positions: wp.array2d(dtype=float),
                 rotations: wp.array2d(dtype=float),
                 world: wp.array3d(dtype=float),
                 bind: wp.array3d(dtype=float)):
    world_position = wp.vec3()
    world_normal = wp.vec3()
    world_tangent = wp.vec3()
    for r in range(3):
        world_position[r] = 0.0
        world_normal[r] = 0.0
        world_tangent[r] = 0.0
    lx = local_positions[p, 0]
    ly = local_positions[p, 1]
    lz = local_positions[p, 2]
    lnx = local_normals[p, 0]
    lny = local_normals[p, 1]
    lnz = local_normals[p, 2]
    ltx = local_tangents[p, 0]
    lty = local_tangents[p, 1]
    ltz = local_tangents[p, 2]
    for j in range(4):
        w = skin_weights[p, j]
        t = skin_indices[p, j]
        for r in range(3):
            s0 = _skin_row(world, bind, t, r, 0)
            s1 = _skin_row(world, bind, t, r, 1)
            s2 = _skin_row(world, bind, t, r, 2)
            s3 = _skin_row(world, bind, t, r, 3)
            world_position[r] = world_position[r] + w * (s0 * lx + s1 * ly + s2 * lz + s3)
            world_normal[r] = world_normal[r] + w * (s0 * lnx + s1 * lny + s2 * lnz)
            world_tangent[r] = world_tangent[r] + w * (s0 * ltx + s1 * lty + s2 * ltz)
    positions[p, 0] = world_position[0]
    positions[p, 1] = world_position[1]
    positions[p, 2] = world_position[2]
    nx, ny, nz = dmath.normalize3_fb(world_normal[0], world_normal[1], world_normal[2],
                                     0.0, 1.0, 0.0)
    tx, ty, tz = dmath.normalize3_fb(world_tangent[0], world_tangent[1], world_tangent[2],
                                     0.0, 0.0, 1.0)
    qx, qy, qz, qw = dmath.to_rotation(nx, ny, nz, tx, ty, tz)
    rotations[p, 0] = qx
    rotations[p, 1] = qy
    rotations[p, 2] = qz
    rotations[p, 3] = qw


@wp.func
def do_distance_gather(p: int, p_team: wp.array(dtype=int),
                       next_positions: wp.array2d(dtype=float),
                       base_positions: wp.array2d(dtype=float),
                       depth: wp.array(dtype=float),
                       friction: wp.array(dtype=float),
                       attr_move: wp.array(dtype=int),
                       t_is_spring: wp.array(dtype=int),
                       t_animation_pose_ratio: wp.array(dtype=float),
                       t_init_scale: wp.array2d(dtype=float),
                       t_scale_ratio: wp.array(dtype=float),
                       t_distance_lut: wp.array2d(dtype=float),
                       power1: float,
                       csr_offsets: wp.array(dtype=int),
                       csr_order: wp.array(dtype=int),
                       distance_target: wp.array(dtype=int),
                       distance_rest: wp.array(dtype=float),
                       sc_dcorr: wp.array2d(dtype=float)):
    mt = p_team[p]
    fix_mass = BONE_SPRING_FIX_MASS if t_is_spring[mt] != 0 else BONE_CLOTH_FIX_MASS
    anime_ratio = t_animation_pose_ratio[mt]
    scale = t_init_scale[mt, 0] * t_scale_ratio[mt]
    depth_p = depth[p]
    fixed_p = attr_move[p] == 0
    inv_mass_p = dmath.calc_inverse_mass_fixed(friction[p], depth_p, fixed_p, fix_mass)
    stiffness = dmath.evaluate_team_lut_clamp01(t_distance_lut, mt, depth_p) * power1
    npx = next_positions[p, 0]
    npy = next_positions[p, 1]
    npz = next_positions[p, 2]
    bpx = base_positions[p, 0]
    bpy = base_positions[p, 1]
    bpz = base_positions[p, 2]

    sumx = wp.float64(0.0)
    sumy = wp.float64(0.0)
    sumz = wp.float64(0.0)
    count_ok = int(0)
    start = csr_offsets[p]
    stop = csr_offsets[p + 1]
    for k in range(start, stop):
        r = csr_order[k]
        tgt = distance_target[r]
        rest = distance_rest[r]
        if rest >= 0.0:
            final_stiffness = dmath.saturate(stiffness)
        else:
            final_stiffness = dmath.saturate(stiffness * DISTANCE_HORIZONTAL_STIFFNESS)
        fixed_t = attr_move[tgt] == 0
        inv_mass_t = dmath.calc_inverse_mass_fixed(friction[tgt], depth[tgt], fixed_t, fix_mass)
        vx = next_positions[tgt, 0] - npx
        vy = next_positions[tgt, 1] - npy
        vz = next_positions[tgt, 2] - npz
        zero_rest = rest == 0.0
        distance = dmath.length3(vx, vy, vz)
        ok = zero_rest or (distance >= EPSILON)
        base_len = dmath.length3(bpx - base_positions[tgt, 0],
                                 bpy - base_positions[tgt, 1],
                                 bpz - base_positions[tgt, 2])
        rest_length = dmath.lerp(wp.abs(rest) * scale, base_len, anime_ratio)
        safe_d = distance if distance > 1e-30 else 1.0
        nx = vx / safe_d
        ny = vy / safe_d
        nz = vz / safe_d
        a = (distance - rest_length) * final_stiffness
        denom = inv_mass_p + inv_mass_t
        cxr = a * nx / denom * inv_mass_p
        cyr = a * ny / denom * inv_mass_p
        czr = a * nz / denom * inv_mass_p
        if zero_rest:
            cxr = vx * 0.5
            cyr = vy * 0.5
            czr = vz * 0.5
        if ok:
            count_ok += 1
            sumx += wp.float64(cxr)
            sumy += wp.float64(cyr)
            sumz += wp.float64(czr)
    if count_ok > 0:
        sc_dcorr[p, 0] = wp.float32(sumx / wp.float64(count_ok))
        sc_dcorr[p, 1] = wp.float32(sumy / wp.float64(count_ok))
        sc_dcorr[p, 2] = wp.float32(sumz / wp.float64(count_ok))
    else:
        sc_dcorr[p, 0] = 0.0
        sc_dcorr[p, 1] = 0.0
        sc_dcorr[p, 2] = 0.0


@wp.func
def do_step_update(i: int, sim_dt: float,
                   t_now_update: wp.array(dtype=float),
                   t_time: wp.array(dtype=float),
                   t_frame_old: wp.array(dtype=float),
                   t_frame_interp: wp.array(dtype=float),
                   t_now_wp: wp.array2d(dtype=float),
                   t_now_wr: wp.array2d(dtype=float),
                   t_old_wp: wp.array2d(dtype=float),
                   t_old_wr: wp.array2d(dtype=float),
                   t_ofwp: wp.array2d(dtype=float),
                   t_ofwr: wp.array2d(dtype=float),
                   t_ofws: wp.array2d(dtype=float),
                   t_fwp: wp.array2d(dtype=float),
                   t_fwr: wp.array2d(dtype=float),
                   t_fws: wp.array2d(dtype=float),
                   t_step_vector: wp.array2d(dtype=float),
                   t_step_rotation: wp.array2d(dtype=float),
                   t_step_mir: wp.array(dtype=float),
                   t_step_rir: wp.array(dtype=float),
                   t_local_inertia: wp.array(dtype=float),
                   t_lmsl: wp.array(dtype=float),
                   t_lrsl: wp.array(dtype=float),
                   t_inertia_vector: wp.array2d(dtype=float),
                   t_inertia_rotation: wp.array2d(dtype=float),
                   t_angular_velocity: wp.array(dtype=float),
                   t_rotation_axis: wp.array2d(dtype=float),
                   t_init_scale: wp.array2d(dtype=float),
                   t_scale_ratio: wp.array(dtype=float),
                   t_gravity_direction: wp.array2d(dtype=float),
                   t_gravity_dot: wp.array(dtype=float),
                   t_ilgd: wp.array2d(dtype=float),
                   t_reflected: wp.array(dtype=int),
                   t_gravity: wp.array(dtype=float),
                   t_gravity_falloff: wp.array(dtype=float),
                   t_gravity_ratio: wp.array(dtype=float),
                   t_velocity_weight: wp.array(dtype=float),
                   t_stab_time: wp.array(dtype=float),
                   t_blend_weight: wp.array(dtype=float),
                   t_bwp: wp.array(dtype=float),
                   t_distance_weight: wp.array(dtype=float),
                   t_wind_moving: wp.array(dtype=float),
                   t_frame_moving_speed: wp.array(dtype=float),
                   t_moving_wind_main: wp.array(dtype=float),
                   t_frame_moving_dir: wp.array2d(dtype=float),
                   t_moving_wind_dir: wp.array2d(dtype=float),
                   t_moving_wind_dirq: wp.array2d(dtype=float),
                   t_wind_main: wp.array2d(dtype=float),
                   t_wind_frequency: wp.array(dtype=float),
                   t_wind_count: wp.array(dtype=int),
                   t_wind_time: wp.array2d(dtype=float),
                   t_moving_wind_time: wp.array(dtype=float)):
    nu = t_now_update[i] + sim_dt
    t_now_update[i] = nu
    span = t_time[i] - t_frame_old[i]
    if span > 0.0:
        interp = dmath.saturate((nu - t_frame_old[i]) / span)
    else:
        interp = 1.0
    t_frame_interp[i] = interp

    owpx = t_now_wp[i, 0]
    owpy = t_now_wp[i, 1]
    owpz = t_now_wp[i, 2]
    owrx = t_now_wr[i, 0]
    owry = t_now_wr[i, 1]
    owrz = t_now_wr[i, 2]
    owrw = t_now_wr[i, 3]
    t_old_wp[i, 0] = owpx
    t_old_wp[i, 1] = owpy
    t_old_wp[i, 2] = owpz
    t_old_wr[i, 0] = owrx
    t_old_wr[i, 1] = owry
    t_old_wr[i, 2] = owrz
    t_old_wr[i, 3] = owrw

    t = interp
    nwpx = dmath.lerp(t_ofwp[i, 0], t_fwp[i, 0], t)
    nwpy = dmath.lerp(t_ofwp[i, 1], t_fwp[i, 1], t)
    nwpz = dmath.lerp(t_ofwp[i, 2], t_fwp[i, 2], t)
    nwrx, nwry, nwrz, nwrw = dmath.quat_slerp(
        t_ofwr[i, 0], t_ofwr[i, 1], t_ofwr[i, 2], t_ofwr[i, 3],
        t_fwr[i, 0], t_fwr[i, 1], t_fwr[i, 2], t_fwr[i, 3], t)
    wsx = dmath.lerp(t_ofws[i, 0], t_fws[i, 0], t)
    wsy = dmath.lerp(t_ofws[i, 1], t_fws[i, 1], t)
    wsz = dmath.lerp(t_ofws[i, 2], t_fws[i, 2], t)
    t_now_wp[i, 0] = nwpx
    t_now_wp[i, 1] = nwpy
    t_now_wp[i, 2] = nwpz
    t_now_wr[i, 0] = nwrx
    t_now_wr[i, 1] = nwry
    t_now_wr[i, 2] = nwrz
    t_now_wr[i, 3] = nwrw

    svx = nwpx - owpx
    svy = nwpy - owpy
    svz = nwpz - owpz
    iowrx, iowry, iowrz, iowrw = dmath.quat_inverse(owrx, owry, owrz, owrw)
    srx, sry, srz, srw = dmath.quat_mul(nwrx, nwry, nwrz, nwrw,
                                        iowrx, iowry, iowrz, iowrw)
    step_angle = dmath.quat_angle(owrx, owry, owrz, owrw, nwrx, nwry, nwrz, nwrw)
    t_step_vector[i, 0] = svx
    t_step_vector[i, 1] = svy
    t_step_vector[i, 2] = svz
    t_step_rotation[i, 0] = srx
    t_step_rotation[i, 1] = sry
    t_step_rotation[i, 2] = srz
    t_step_rotation[i, 3] = srw

    li = t_local_inertia[i]
    lmi = 1.0 - li
    lri = 1.0 - li
    lvx = svx * (1.0 - lmi)
    lvy = svy * (1.0 - lmi)
    lvz = svz * (1.0 - lmi)
    local_speed = dmath.length3(lvx, lvy, lvz) / sim_dt
    limit = t_lmsl[i]
    if (local_speed > limit) and (limit >= 0.0):
        denom = local_speed if local_speed > 0.0 else 1.0
        ratio = limit / denom
        lmi = 1.0 + (lmi - 1.0) * ratio
    local_angle = step_angle * (1.0 - lri)
    local_angle_speed = (local_angle / sim_dt) * RAD2DEG
    limit = t_lrsl[i]
    if (local_angle_speed > limit) and (limit >= 0.0):
        denom = local_angle_speed if local_angle_speed > 0.0 else 1.0
        ratio = limit / denom
        lri = 1.0 + (lri - 1.0) * ratio
    t_step_mir[i] = lmi
    t_step_rir[i] = lri

    t_inertia_vector[i, 0] = svx * lmi
    t_inertia_vector[i, 1] = svy * lmi
    t_inertia_vector[i, 2] = svz * lmi
    irx, iry, irz, irw = dmath.quat_slerp(0.0, 0.0, 0.0, 1.0,
                                          srx, sry, srz, srw, lri)
    t_inertia_rotation[i, 0] = irx
    t_inertia_rotation[i, 1] = iry
    t_inertia_rotation[i, 2] = irz
    t_inertia_rotation[i, 3] = irw

    angular_velocity = step_angle / sim_dt
    t_angular_velocity[i] = angular_velocity
    _ang, axx, axy, axz = dmath.quat_to_angle_axis(srx, sry, srz, srw)
    if angular_velocity > EPSILON:
        t_rotation_axis[i, 0] = axx
        t_rotation_axis[i, 1] = axy
        t_rotation_axis[i, 2] = axz
    else:
        t_rotation_axis[i, 0] = 0.0
        t_rotation_axis[i, 1] = 0.0
        t_rotation_axis[i, 2] = 0.0

    isl = dmath.length3(t_init_scale[i, 0], t_init_scale[i, 1], t_init_scale[i, 2])
    if isl < 1.0e-30:
        isl = 1.0e-30
    wsl = dmath.length3(wsx, wsy, wsz)
    sr = wsl / isl
    if sr < 1.0e-6:
        sr = 1.0e-6
    t_scale_ratio[i] = sr

    gdx = t_gravity_direction[i, 0]
    gdy = t_gravity_direction[i, 1]
    gdz = t_gravity_direction[i, 2]
    gravity_dot = float(1.0)
    if (gdx * gdx + gdy * gdy + gdz * gdz) > EPSILON:
        reflection_sign = component_reflection_sign(t_reflected, i)
        ilx = t_ilgd[i, 0] * reflection_sign
        ily = t_ilgd[i, 1] * reflection_sign
        ilz = t_ilgd[i, 2] * reflection_sign
        wfx, wfy, wfz = dmath.quat_rotate(nwrx, nwry, nwrz, nwrw, ilx, ily, ilz)
        gdot = wfx * gdx + wfy * gdy + wfz * gdz
        gravity_dot = dmath.saturate(gdot * 0.5 + 0.5)
    t_gravity_dot[i] = gravity_dot

    gravity_ratio = float(1.0)
    if (t_gravity[i] > 1.0e-6) and (t_gravity_falloff[i] > 1.0e-6):
        low = dmath.saturate(1.0 - t_gravity_falloff[i])
        gravity_ratio = low + (1.0 - low) * dmath.saturate(1.0 - gravity_dot)
    t_gravity_ratio[i] = gravity_ratio

    vw = t_velocity_weight[i]
    if vw < 1.0:
        stab = t_stab_time[i]
        if stab > 1.0e-6:
            add = sim_dt / stab
        else:
            add = 1.0
        vw = vw + add
        if vw > 1.0:
            vw = 1.0
        t_velocity_weight[i] = vw
    t_blend_weight[i] = dmath.saturate(vw * t_bwp[i] * t_distance_weight[i])

    moving_active = t_wind_moving[i] > 0.01
    if moving_active:
        denom = sr if sr > 0.0 else 1.0
        mwm = (t_frame_moving_speed[i] * t_wind_moving[i]) / denom
    else:
        mwm = 0.0
    t_moving_wind_main[i] = mwm
    if moving_active:
        mdx = dmath.negate(t_frame_moving_dir[i, 0])
        mdy = dmath.negate(t_frame_moving_dir[i, 1])
        mdz = dmath.negate(t_frame_moving_dir[i, 2])
        t_moving_wind_dir[i, 0] = mdx
        t_moving_wind_dir[i, 1] = mdy
        t_moving_wind_dir[i, 2] = mdz
        mqx, mqy, mqz, mqw = dmath.axis_quaternion(mdx, mdy, mdz)
        t_moving_wind_dirq[i, 0] = mqx
        t_moving_wind_dirq[i, 1] = mqy
        t_moving_wind_dirq[i, 2] = mqz
        t_moving_wind_dirq[i, 3] = mqw

    wf = t_wind_frequency[i]
    wc = t_wind_count[i]
    for s in range(4):
        main_ratio = t_wind_main[i, s] / WIND_BASE_SPEED
        frequency = (0.2 + main_ratio * 0.5) * wf
        if frequency > 1.5:
            frequency = 1.5
        frequency = frequency * sim_dt
        if s < wc:
            nt = t_wind_time[i, s] + frequency
            if nt > WIND_MAX_TIME:
                nt = nt - WIND_MAX_TIME * 2.0
            t_wind_time[i, s] = nt
    move_ratio = mwm / WIND_BASE_SPEED
    mf = (0.2 + move_ratio * 0.5) * wf
    if mf > 1.5:
        mf = 1.5
    mf = mf * sim_dt
    if moving_active:
        mt2 = t_moving_wind_time[i] + mf
        if mt2 > WIND_MAX_TIME:
            mt2 = mt2 - WIND_MAX_TIME * 2.0
        t_moving_wind_time[i] = mt2


@wp.func
def _neg_transform_pose(arr_pos: wp.array2d(dtype=float), arr_rot: wp.array2d(dtype=float),
                        ci: int, m: wp.mat44d):
    px, py, pz = dmath.transform_point(m, arr_pos[ci, 0], arr_pos[ci, 1], arr_pos[ci, 2])
    arr_pos[ci, 0] = px
    arr_pos[ci, 1] = py
    arr_pos[ci, 2] = pz
    qx, qy, qz, qw = dmath.transform_rotation(m, arr_rot[ci, 0], arr_rot[ci, 1],
                                              arr_rot[ci, 2], arr_rot[ci, 3],
                                              REFLECTION_TELEPORT_FLIP)
    arr_rot[ci, 0] = qx
    arr_rot[ci, 1] = qy
    arr_rot[ci, 2] = qz
    arr_rot[ci, 3] = qw


@wp.func
def _neg_transform_point(arr_pos: wp.array2d(dtype=float), ci: int, m: wp.mat44d):
    px, py, pz = dmath.transform_point(m, arr_pos[ci, 0], arr_pos[ci, 1], arr_pos[ci, 2])
    arr_pos[ci, 0] = px
    arr_pos[ci, 1] = py
    arr_pos[ci, 2] = pz


@wp.func
def _shift_pose(arr_pos: wp.array2d(dtype=float), arr_rot: wp.array2d(dtype=float), ci: int,
                cpx: float, cpy: float, cpz: float, svx: float, svy: float, svz: float,
                srx: float, sry: float, srz: float, srw: float):
    lx = arr_pos[ci, 0] - cpx
    ly = arr_pos[ci, 1] - cpy
    lz = arr_pos[ci, 2] - cpz
    rx, ry, rz = dmath.quat_rotate(srx, sry, srz, srw, lx, ly, lz)
    arr_pos[ci, 0] = rx + cpx + svx
    arr_pos[ci, 1] = ry + cpy + svy
    arr_pos[ci, 2] = rz + cpz + svz
    qx, qy, qz, qw = dmath.quat_mul(srx, sry, srz, srw, arr_rot[ci, 0], arr_rot[ci, 1],
                                    arr_rot[ci, 2], arr_rot[ci, 3])
    arr_rot[ci, 0] = qx
    arr_rot[ci, 1] = qy
    arr_rot[ci, 2] = qz
    arr_rot[ci, 3] = qw


@wp.func
def _shift_point(arr: wp.array2d(dtype=float), p: int,
                 cpx: float, cpy: float, cpz: float, svx: float, svy: float, svz: float,
                 srx: float, sry: float, srz: float, srw: float):
    rx, ry, rz = dmath.quat_rotate(srx, sry, srz, srw,
                                   arr[p, 0] - cpx, arr[p, 1] - cpy, arr[p, 2] - cpz)
    arr[p, 0] = rx + cpx + svx
    arr[p, 1] = ry + cpy + svy
    arr[p, 2] = rz + cpz + svz


@wp.func
def _premul_quat(arr: wp.array2d(dtype=float), p: int,
                 srx: float, sry: float, srz: float, srw: float):
    qx, qy, qz, qw = dmath.quat_mul(srx, sry, srz, srw,
                                    arr[p, 0], arr[p, 1], arr[p, 2], arr[p, 3])
    arr[p, 0] = qx
    arr[p, 1] = qy
    arr[p, 2] = qz
    arr[p, 3] = qw


@wp.func
def _rotate_vec(arr: wp.array2d(dtype=float), p: int,
                srx: float, sry: float, srz: float, srw: float):
    vx, vy, vz = dmath.quat_rotate(srx, sry, srz, srw, arr[p, 0], arr[p, 1], arr[p, 2])
    arr[p, 0] = vx
    arr[p, 1] = vy
    arr[p, 2] = vz


@wp.func
def do_advance(i: int, fdt: float, sim_dt: float, max_sim_count: int, global_time_scale: float,
               time_reset: wp.array(dtype=int),
               time: wp.array(dtype=float),
               old_time: wp.array(dtype=float),
               now_update: wp.array(dtype=float),
               old_update: wp.array(dtype=float),
               frame_update: wp.array(dtype=float),
               frame_old: wp.array(dtype=float),
               frame_dt: wp.array(dtype=float),
               time_scale: wp.array(dtype=float),
               now_time_scale: wp.array(dtype=float),
               update_count: wp.array(dtype=int),
               skip_count: wp.array(dtype=int),
               running: wp.array(dtype=int)):
    reset = time_reset[i] != 0
    t_time = 0.0 if reset else time[i]
    t_now_update = 0.0 if reset else now_update[i]
    t_old_update = 0.0 if reset else old_update[i]
    t_frame_update = 0.0 if reset else frame_update[i]
    t_frame_old = 0.0 if reset else frame_old[i]

    frame_dt[i] = fdt
    ts = time_scale[i] * global_time_scale
    now_time_scale[i] = ts
    add_time = fdt * ts
    new_time = t_time + add_time
    interval = new_time - t_now_update
    uc = int(interval / sim_dt)
    clamped = uc if uc < max_sim_count else max_sim_count
    skip = uc - clamped
    if skip > 0:
        new_time = new_time - sim_dt * float(skip)
    guard = (clamped > 0) and (add_time == 0.0)
    if guard:
        clamped = int(0)
        skip = int(0)
        new_now_update = new_time - sim_dt + 0.0001
    else:
        new_now_update = t_now_update
    updated = clamped > 0

    old_time[i] = t_time
    time[i] = new_time
    now_update[i] = new_now_update
    if updated:
        frame_old[i] = t_frame_update
        frame_update[i] = new_time
        old_update[i] = new_now_update
    else:
        frame_old[i] = t_frame_old
        frame_update[i] = t_frame_update
        old_update[i] = t_old_update
    update_count[i] = clamped
    skip_count[i] = skip
    running[i] = 1 if updated else 0


@wp.func
def do_particles_frame_pre(p: int, p_team: wp.array(dtype=int),
                           p_positions: wp.array2d(dtype=float),
                           p_rotations: wp.array2d(dtype=float),
                           p_next_positions: wp.array2d(dtype=float),
                           p_old_positions: wp.array2d(dtype=float),
                           p_old_rotations: wp.array2d(dtype=float),
                           p_base_positions: wp.array2d(dtype=float),
                           p_base_rotations: wp.array2d(dtype=float),
                           p_old_anim_positions: wp.array2d(dtype=float),
                           p_old_anim_rotations: wp.array2d(dtype=float),
                           p_velocity_positions: wp.array2d(dtype=float),
                           p_display_positions: wp.array2d(dtype=float),
                           p_velocities: wp.array2d(dtype=float),
                           p_real_velocities: wp.array2d(dtype=float),
                           p_friction: wp.array(dtype=float),
                           p_static_friction: wp.array(dtype=float),
                           p_collision_normals: wp.array2d(dtype=float),
                           t_reset_pending: wp.array(dtype=int),
                           t_neg_teleport: wp.array(dtype=int),
                           t_neg_matrix: wp.array(dtype=wp.mat44d),
                           t_inertia_shift: wp.array(dtype=int),
                           t_shift_vec: wp.array2d(dtype=float),
                           t_shift_rot: wp.array2d(dtype=float),
                           t_old_cwp: wp.array2d(dtype=float)):
    team = p_team[p]
    if t_reset_pending[team] != 0:
        for j in range(3):
            pv = p_positions[p, j]
            p_next_positions[p, j] = pv
            p_old_positions[p, j] = pv
            p_base_positions[p, j] = pv
            p_old_anim_positions[p, j] = pv
            p_velocity_positions[p, j] = pv
            p_display_positions[p, j] = pv
            p_velocities[p, j] = 0.0
            p_real_velocities[p, j] = 0.0
            p_collision_normals[p, j] = 0.0
        for j in range(4):
            rv = p_rotations[p, j]
            p_old_rotations[p, j] = rv
            p_base_rotations[p, j] = rv
            p_old_anim_rotations[p, j] = rv
        p_friction[p] = 0.0
        p_static_friction[p] = 0.0
        return
    neg = t_neg_teleport[team] != 0
    shift = t_inertia_shift[team] != 0
    if not (neg or shift):
        return
    if neg:
        m = t_neg_matrix[team]
        _neg_transform_pose(p_old_positions, p_old_rotations, p, m)
        _neg_transform_pose(p_old_anim_positions, p_old_anim_rotations, p, m)
        dpx, dpy, dpz = dmath.transform_point(m, p_display_positions[p, 0],
                                              p_display_positions[p, 1],
                                              p_display_positions[p, 2])
        p_display_positions[p, 0] = dpx
        p_display_positions[p, 1] = dpy
        p_display_positions[p, 2] = dpz
        vx, vy, vz = dmath.transform_vector(m, p_velocities[p, 0], p_velocities[p, 1],
                                            p_velocities[p, 2])
        p_velocities[p, 0] = vx
        p_velocities[p, 1] = vy
        p_velocities[p, 2] = vz
        rvx, rvy, rvz = dmath.transform_vector(m, p_real_velocities[p, 0],
                                               p_real_velocities[p, 1],
                                               p_real_velocities[p, 2])
        p_real_velocities[p, 0] = rvx
        p_real_velocities[p, 1] = rvy
        p_real_velocities[p, 2] = rvz
    if shift:
        cpx = t_old_cwp[team, 0]
        cpy = t_old_cwp[team, 1]
        cpz = t_old_cwp[team, 2]
        svx = t_shift_vec[team, 0]
        svy = t_shift_vec[team, 1]
        svz = t_shift_vec[team, 2]
        srx = t_shift_rot[team, 0]
        sry = t_shift_rot[team, 1]
        srz = t_shift_rot[team, 2]
        srw = t_shift_rot[team, 3]
        _shift_point(p_old_positions, p, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_point(p_old_anim_positions, p, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_point(p_display_positions, p, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _premul_quat(p_old_rotations, p, srx, sry, srz, srw)
        _premul_quat(p_old_anim_rotations, p, srx, sry, srz, srw)
        _rotate_vec(p_velocities, p, srx, sry, srz, srw)
        _rotate_vec(p_real_velocities, p, srx, sry, srz, srw)


@wp.func
def do_collider_frame_pre(ci: int, c_team: wp.array(dtype=int),
                          c_enabled: wp.array(dtype=int),
                          c_enabled_prev: wp.array(dtype=int),
                          c_active: wp.array(dtype=int),
                          c_input_positions: wp.array2d(dtype=float),
                          c_input_rotations: wp.array2d(dtype=float),
                          c_input_tips: wp.array2d(dtype=float),
                          c_input_radii: wp.array2d(dtype=float),
                          c_frame_pos: wp.array2d(dtype=float),
                          c_frame_rot: wp.array2d(dtype=float),
                          c_frame_tip: wp.array2d(dtype=float),
                          c_frame_radius: wp.array2d(dtype=float),
                          c_old_frame_pos: wp.array2d(dtype=float),
                          c_old_frame_rot: wp.array2d(dtype=float),
                          c_old_frame_tip: wp.array2d(dtype=float),
                          c_now_pos: wp.array2d(dtype=float),
                          c_now_rot: wp.array2d(dtype=float),
                          c_now_tip: wp.array2d(dtype=float),
                          c_old_pos: wp.array2d(dtype=float),
                          c_old_rot: wp.array2d(dtype=float),
                          c_old_tip: wp.array2d(dtype=float),
                          t_reset_pending: wp.array(dtype=int),
                          t_neg_teleport: wp.array(dtype=int),
                          t_neg_matrix: wp.array(dtype=wp.mat44d),
                          t_inertia_shift: wp.array(dtype=int),
                          t_shift_vec: wp.array2d(dtype=float),
                          t_shift_rot: wp.array2d(dtype=float),
                          t_old_cwp: wp.array2d(dtype=float)):
    enabled_now = c_enabled[ci] != 0
    rising = enabled_now and (c_enabled_prev[ci] == 0)
    c_active[ci] = 1 if enabled_now else 0
    c_enabled_prev[ci] = 1 if enabled_now else 0
    if not enabled_now:
        return
    team = c_team[ci]
    for j in range(3):
        c_frame_pos[ci, j] = c_input_positions[ci, j]
        c_frame_tip[ci, j] = c_input_tips[ci, j]
    for j in range(4):
        c_frame_rot[ci, j] = c_input_rotations[ci, j]
    c_frame_radius[ci, 0] = c_input_radii[ci, 0]
    c_frame_radius[ci, 1] = c_input_radii[ci, 1]
    reset = (t_reset_pending[team] != 0) or rising
    if reset:
        for j in range(3):
            fp = c_frame_pos[ci, j]
            c_old_frame_pos[ci, j] = fp
            c_now_pos[ci, j] = fp
            c_old_pos[ci, j] = fp
            ft = c_frame_tip[ci, j]
            c_old_frame_tip[ci, j] = ft
            c_now_tip[ci, j] = ft
            c_old_tip[ci, j] = ft
        for j in range(4):
            fr = c_frame_rot[ci, j]
            c_old_frame_rot[ci, j] = fr
            c_now_rot[ci, j] = fr
            c_old_rot[ci, j] = fr
        return
    if t_neg_teleport[team] != 0:
        m = t_neg_matrix[team]
        _neg_transform_pose(c_old_frame_pos, c_old_frame_rot, ci, m)
        _neg_transform_pose(c_now_pos, c_now_rot, ci, m)
        _neg_transform_pose(c_old_pos, c_old_rot, ci, m)
        _neg_transform_point(c_old_frame_tip, ci, m)
        _neg_transform_point(c_now_tip, ci, m)
        _neg_transform_point(c_old_tip, ci, m)
    if t_inertia_shift[team] != 0:
        cpx = t_old_cwp[team, 0]
        cpy = t_old_cwp[team, 1]
        cpz = t_old_cwp[team, 2]
        svx = t_shift_vec[team, 0]
        svy = t_shift_vec[team, 1]
        svz = t_shift_vec[team, 2]
        srx = t_shift_rot[team, 0]
        sry = t_shift_rot[team, 1]
        srz = t_shift_rot[team, 2]
        srw = t_shift_rot[team, 3]
        _shift_pose(c_old_frame_pos, c_old_frame_rot, ci, cpx, cpy, cpz, svx, svy, svz,
                    srx, sry, srz, srw)
        _shift_pose(c_now_pos, c_now_rot, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_pose(c_old_pos, c_old_rot, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_point(c_old_frame_tip, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_point(c_now_tip, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)
        _shift_point(c_old_tip, ci, cpx, cpy, cpz, svx, svy, svz, srx, sry, srz, srw)


@wp.func
def rotated_box_half_extent(qx: float, qy: float, qz: float, qw: float,
                            hx: float, hy: float, hz: float):
    axx, axy, axz = dmath.quat_rotate(qx, qy, qz, qw, hx, 0.0, 0.0)
    ayx, ayy, ayz = dmath.quat_rotate(qx, qy, qz, qw, 0.0, hy, 0.0)
    azx, azy, azz = dmath.quat_rotate(qx, qy, qz, qw, 0.0, 0.0, hz)
    return (wp.abs(axx) + wp.abs(ayx) + wp.abs(azx),
            wp.abs(axy) + wp.abs(ayy) + wp.abs(azy),
            wp.abs(axz) + wp.abs(ayz) + wp.abs(azz))


COLLIDER_SWEEP_SOURCE_REASON = (
    "where a body was at the start of the step is a fact about the body, and it is the "
    "only thing the contact search has to work out that the body moved at all; it used to "
    "be handed the movement inertia blend instead, which is the cloth's answer to how much "
    "of the rig's travel it gets dragged along by, so one team property decided both how "
    "the cloth responds and whether the collider swept anything: at a local inertia of "
    "zero the blend lands exactly on the new pose, the start and the end of the step "
    "become the same pose, and every continuous test goes blind, silently, on a setting a "
    "user is entitled to choose; measured by tunnelcheck.py, a plate a quarter as thick as "
    "its own substep travel walked through all hundred and thirty two particles in its "
    "path at that setting and through none of them once the step motion was visible, which "
    "is a hundred percent of the contacts on that world turned off by a response "
    "parameter; the blend is still what the collider's own inertia state is carried "
    "forward by, because that is the job it was written for, and the sweep now reads the "
    "pose the previous substep actually committed, which commit_collider_substep_pose "
    "writes unblended at the end of every substep; this is the same shape as the "
    "component scale carrying an amplitude and a sign in Phase F and the fixed point scale "
    "carrying an accumulation scale and a sort key in Phase E, and it is taken apart the "
    "same way")


COLLIDER_WORK_POSE_REASON = (
    "which slots of the work planes a shape puts its pose into is shape knowledge, so it "
    "is written once and called from wherever a pose has to be laid out; there are two "
    "such places and they are two different instants, the substep the solver is inside and "
    "the frame the picture is drawn at, and before this was lifted out there was only the "
    "first one, which is why the output projection was pushing particles out of a body "
    "posed somewhere between seven tenths and nine tenths of the way through the frame")


@wp.func
def collider_write_work_pose(ci: int, kind: int,
                             opx: float, opy: float, opz: float,
                             otx: float, oty: float, otz: float,
                             orx: float, ory: float, orz: float, orw: float,
                             posx: float, posy: float, posz: float,
                             tipx: float, tipy: float, tipz: float,
                             rotx: float, roty: float, rotz: float, rotw: float,
                             c_frame_radius: wp.array2d(dtype=float),
                             c_work_rot: wp.array2d(dtype=float),
                             c_work_inv_old_rot: wp.array2d(dtype=float),
                             c_work_inv_rot: wp.array2d(dtype=float),
                             c_work_radius: wp.array2d(dtype=float),
                             c_work_old_pos: wp.array3d(dtype=float),
                             c_work_next_pos: wp.array3d(dtype=float),
                             c_work_aabb_min: wp.array2d(dtype=float),
                             c_work_aabb_max: wp.array2d(dtype=float),
                             c_mesh_bound_min: wp.array2d(dtype=float),
                             c_mesh_bound_max: wp.array2d(dtype=float)):
    c_work_rot[ci, 0] = rotx
    c_work_rot[ci, 1] = roty
    c_work_rot[ci, 2] = rotz
    c_work_rot[ci, 3] = rotw
    iox, ioy, ioz, iow = dmath.quat_inverse(orx, ory, orz, orw)
    c_work_inv_old_rot[ci, 0] = iox
    c_work_inv_old_rot[ci, 1] = ioy
    c_work_inv_old_rot[ci, 2] = ioz
    c_work_inv_old_rot[ci, 3] = iow
    inx, iny, inz, inw = dmath.quat_inverse(rotx, roty, rotz, rotw)
    c_work_inv_rot[ci, 0] = inx
    c_work_inv_rot[ci, 1] = iny
    c_work_inv_rot[ci, 2] = inz
    c_work_inv_rot[ci, 3] = inw
    if kind == COLLIDER_SPHERE:
        radius = c_frame_radius[ci, 0]
        c_work_radius[ci, 0] = radius
        c_work_radius[ci, 1] = radius
        c_work_old_pos[ci, 0, 0] = opx
        c_work_old_pos[ci, 0, 1] = opy
        c_work_old_pos[ci, 0, 2] = opz
        c_work_next_pos[ci, 0, 0] = posx
        c_work_next_pos[ci, 0, 1] = posy
        c_work_next_pos[ci, 0, 2] = posz
        c_work_aabb_min[ci, 0] = dmath.fmin2(opx, posx) - radius
        c_work_aabb_min[ci, 1] = dmath.fmin2(opy, posy) - radius
        c_work_aabb_min[ci, 2] = dmath.fmin2(opz, posz) - radius
        c_work_aabb_max[ci, 0] = dmath.fmax2(opx, posx) + radius
        c_work_aabb_max[ci, 1] = dmath.fmax2(opy, posy) + radius
        c_work_aabb_max[ci, 2] = dmath.fmax2(opz, posz) + radius
    elif kind == COLLIDER_CAPSULE:
        start_radius = c_frame_radius[ci, 0]
        end_radius = c_frame_radius[ci, 1]
        sox = opx
        soy = opy
        soz = opz
        eox = otx
        eoy = oty
        eoz = otz
        snx = posx
        sny = posy
        snz = posz
        enx = tipx
        eny = tipy
        enz = tipz
        c_work_radius[ci, 0] = start_radius
        c_work_radius[ci, 1] = end_radius
        c_work_old_pos[ci, 0, 0] = sox
        c_work_old_pos[ci, 0, 1] = soy
        c_work_old_pos[ci, 0, 2] = soz
        c_work_old_pos[ci, 1, 0] = eox
        c_work_old_pos[ci, 1, 1] = eoy
        c_work_old_pos[ci, 1, 2] = eoz
        c_work_next_pos[ci, 0, 0] = snx
        c_work_next_pos[ci, 0, 1] = sny
        c_work_next_pos[ci, 0, 2] = snz
        c_work_next_pos[ci, 1, 0] = enx
        c_work_next_pos[ci, 1, 1] = eny
        c_work_next_pos[ci, 1, 2] = enz
        c_work_aabb_min[ci, 0] = dmath.fmin2(dmath.fmin2(sox, snx) - start_radius,
                                             dmath.fmin2(eox, enx) - end_radius)
        c_work_aabb_min[ci, 1] = dmath.fmin2(dmath.fmin2(soy, sny) - start_radius,
                                             dmath.fmin2(eoy, eny) - end_radius)
        c_work_aabb_min[ci, 2] = dmath.fmin2(dmath.fmin2(soz, snz) - start_radius,
                                             dmath.fmin2(eoz, enz) - end_radius)
        c_work_aabb_max[ci, 0] = dmath.fmax2(dmath.fmax2(sox, snx) + start_radius,
                                             dmath.fmax2(eox, enx) + end_radius)
        c_work_aabb_max[ci, 1] = dmath.fmax2(dmath.fmax2(soy, sny) + start_radius,
                                             dmath.fmax2(eoy, eny) + end_radius)
        c_work_aabb_max[ci, 2] = dmath.fmax2(dmath.fmax2(soz, snz) + start_radius,
                                             dmath.fmax2(eoz, enz) + end_radius)
    elif kind == COLLIDER_MESH:
        lowx = c_mesh_bound_min[ci, 0]
        lowy = c_mesh_bound_min[ci, 1]
        lowz = c_mesh_bound_min[ci, 2]
        highx = c_mesh_bound_max[ci, 0]
        highy = c_mesh_bound_max[ci, 1]
        highz = c_mesh_bound_max[ci, 2]
        reach = dmath.length3(highx - lowx, highy - lowy, highz - lowz)
        c_work_radius[ci, 0] = reach
        c_work_radius[ci, 1] = reach
        c_work_old_pos[ci, 0, 0] = opx
        c_work_old_pos[ci, 0, 1] = opy
        c_work_old_pos[ci, 0, 2] = opz
        c_work_next_pos[ci, 0, 0] = posx
        c_work_next_pos[ci, 0, 1] = posy
        c_work_next_pos[ci, 0, 2] = posz
        centrex = 0.5 * (lowx + highx)
        centrey = 0.5 * (lowy + highy)
        centrez = 0.5 * (lowz + highz)
        halfx = 0.5 * (highx - lowx)
        halfy = 0.5 * (highy - lowy)
        halfz = 0.5 * (highz - lowz)
        ocx, ocy, ocz = dmath.quat_rotate(orx, ory, orz, orw, centrex, centrey, centrez)
        ncx, ncy, ncz = dmath.quat_rotate(rotx, roty, rotz, rotw, centrex, centrey, centrez)
        oex, oey, oez = rotated_box_half_extent(orx, ory, orz, orw, halfx, halfy, halfz)
        nex, ney, nez = rotated_box_half_extent(rotx, roty, rotz, rotw, halfx, halfy, halfz)
        c_work_aabb_min[ci, 0] = dmath.fmin2(opx + ocx - oex, posx + ncx - nex)
        c_work_aabb_min[ci, 1] = dmath.fmin2(opy + ocy - oey, posy + ncy - ney)
        c_work_aabb_min[ci, 2] = dmath.fmin2(opz + ocz - oez, posz + ncz - nez)
        c_work_aabb_max[ci, 0] = dmath.fmax2(opx + ocx + oex, posx + ncx + nex)
        c_work_aabb_max[ci, 1] = dmath.fmax2(opy + ocy + oey, posy + ncy + ney)
        c_work_aabb_max[ci, 2] = dmath.fmax2(opz + ocz + oez, posz + ncz + nez)
    else:
        c_work_old_pos[ci, 0, 0] = opx
        c_work_old_pos[ci, 0, 1] = opy
        c_work_old_pos[ci, 0, 2] = opz
        c_work_next_pos[ci, 0, 0] = posx
        c_work_next_pos[ci, 0, 1] = posy
        c_work_next_pos[ci, 0, 2] = posz
        c_work_aabb_min[ci, 0] = -wp.inf
        c_work_aabb_min[ci, 1] = -wp.inf
        c_work_aabb_min[ci, 2] = -wp.inf
        c_work_aabb_max[ci, 0] = wp.inf
        c_work_aabb_max[ci, 1] = wp.inf
        c_work_aabb_max[ci, 2] = wp.inf


@wp.func
def do_collider_start_step(ci: int, c_team: wp.array(dtype=int), c_kind: wp.array(dtype=int),
                           c_frame_pos: wp.array2d(dtype=float),
                           c_frame_rot: wp.array2d(dtype=float),
                           c_frame_tip: wp.array2d(dtype=float),
                           c_frame_radius: wp.array2d(dtype=float),
                           c_old_frame_pos: wp.array2d(dtype=float),
                           c_old_frame_rot: wp.array2d(dtype=float),
                           c_old_frame_tip: wp.array2d(dtype=float),
                           c_now_pos: wp.array2d(dtype=float),
                           c_now_rot: wp.array2d(dtype=float),
                           c_now_tip: wp.array2d(dtype=float),
                           c_old_pos: wp.array2d(dtype=float),
                           c_old_rot: wp.array2d(dtype=float),
                           c_old_tip: wp.array2d(dtype=float),
                           c_work_rot: wp.array2d(dtype=float),
                           c_work_inv_old_rot: wp.array2d(dtype=float),
                           c_work_inv_rot: wp.array2d(dtype=float),
                           c_work_radius: wp.array2d(dtype=float),
                           c_work_old_pos: wp.array3d(dtype=float),
                           c_work_next_pos: wp.array3d(dtype=float),
                           c_work_aabb_min: wp.array2d(dtype=float),
                           c_work_aabb_max: wp.array2d(dtype=float),
                           c_mesh_bound_min: wp.array2d(dtype=float),
                           c_mesh_bound_max: wp.array2d(dtype=float),
                           t_frame_interp: wp.array(dtype=float),
                           t_step_mir: wp.array(dtype=float),
                           t_step_rir: wp.array(dtype=float)):
    team = c_team[ci]
    t = t_frame_interp[team]
    posx = dmath.lerp(c_old_frame_pos[ci, 0], c_frame_pos[ci, 0], t)
    posy = dmath.lerp(c_old_frame_pos[ci, 1], c_frame_pos[ci, 1], t)
    posz = dmath.lerp(c_old_frame_pos[ci, 2], c_frame_pos[ci, 2], t)
    tipx = dmath.lerp(c_old_frame_tip[ci, 0], c_frame_tip[ci, 0], t)
    tipy = dmath.lerp(c_old_frame_tip[ci, 1], c_frame_tip[ci, 1], t)
    tipz = dmath.lerp(c_old_frame_tip[ci, 2], c_frame_tip[ci, 2], t)
    rotx, roty, rotz, rotw = dmath.quat_slerp(
        c_old_frame_rot[ci, 0], c_old_frame_rot[ci, 1], c_old_frame_rot[ci, 2],
        c_old_frame_rot[ci, 3],
        c_frame_rot[ci, 0], c_frame_rot[ci, 1], c_frame_rot[ci, 2], c_frame_rot[ci, 3], t)
    c_now_pos[ci, 0] = posx
    c_now_pos[ci, 1] = posy
    c_now_pos[ci, 2] = posz
    c_now_tip[ci, 0] = tipx
    c_now_tip[ci, 1] = tipy
    c_now_tip[ci, 2] = tipz
    c_now_rot[ci, 0] = rotx
    c_now_rot[ci, 1] = roty
    c_now_rot[ci, 2] = rotz
    c_now_rot[ci, 3] = rotw
    swept_px = c_old_pos[ci, 0]
    swept_py = c_old_pos[ci, 1]
    swept_pz = c_old_pos[ci, 2]
    swept_tx = c_old_tip[ci, 0]
    swept_ty = c_old_tip[ci, 1]
    swept_tz = c_old_tip[ci, 2]
    swept_rx = c_old_rot[ci, 0]
    swept_ry = c_old_rot[ci, 1]
    swept_rz = c_old_rot[ci, 2]
    swept_rw = c_old_rot[ci, 3]
    mir = t_step_mir[team]
    rir = t_step_rir[team]
    opx = dmath.lerp(c_old_pos[ci, 0], posx, mir)
    opy = dmath.lerp(c_old_pos[ci, 1], posy, mir)
    opz = dmath.lerp(c_old_pos[ci, 2], posz, mir)
    otx = dmath.lerp(c_old_tip[ci, 0], tipx, mir)
    oty = dmath.lerp(c_old_tip[ci, 1], tipy, mir)
    otz = dmath.lerp(c_old_tip[ci, 2], tipz, mir)
    orx, ory, orz, orw = dmath.quat_slerp(
        c_old_rot[ci, 0], c_old_rot[ci, 1], c_old_rot[ci, 2], c_old_rot[ci, 3],
        rotx, roty, rotz, rotw, rir)
    c_old_pos[ci, 0] = opx
    c_old_pos[ci, 1] = opy
    c_old_pos[ci, 2] = opz
    c_old_tip[ci, 0] = otx
    c_old_tip[ci, 1] = oty
    c_old_tip[ci, 2] = otz
    c_old_rot[ci, 0] = orx
    c_old_rot[ci, 1] = ory
    c_old_rot[ci, 2] = orz
    c_old_rot[ci, 3] = orw
    collider_write_work_pose(ci, c_kind[ci], swept_px, swept_py, swept_pz,
                             swept_tx, swept_ty, swept_tz, swept_rx, swept_ry, swept_rz,
                             swept_rw, posx, posy, posz, tipx, tipy, tipz, rotx, roty, rotz,
                             rotw, c_frame_radius, c_work_rot, c_work_inv_old_rot,
                             c_work_inv_rot, c_work_radius, c_work_old_pos,
                             c_work_next_pos, c_work_aabb_min, c_work_aabb_max,
                             c_mesh_bound_min, c_mesh_bound_max)


COLLIDER_FRAME_POSE_REASON = (
    "the exit projection is the step that makes the output non penetrating, and the output "
    "is written at the solver's own instant but looked at at the frame's; a fixed step "
    "solver running at a rate that does not divide the frame is right to end its last "
    "substep inside the frame, so the two instants are different by construction and the "
    "answer is not to move the solver but to project out of the body the picture will "
    "actually show; at ninety hertz against twenty four frames a second the last substep "
    "lands between seven tenths and nine tenths of the way through the frame, and measured "
    "against the body at the pose the frame handed in that left up to seven centimetres of "
    "the cloth inside it while the very same output measured against the body the solver "
    "posed read no penetration at all, so the whole of that reading was the two instants "
    "and none of it was a contact the search missed; the narrow phase is left alone, "
    "because inside a substep the substep pose is the correct one and it is the solver's "
    "own timeline that it advances along")


@wp.func
def do_collider_frame_pose(ci: int, c_kind: wp.array(dtype=int),
                           c_frame_pos: wp.array2d(dtype=float),
                           c_frame_rot: wp.array2d(dtype=float),
                           c_frame_tip: wp.array2d(dtype=float),
                           c_frame_radius: wp.array2d(dtype=float),
                           c_work_rot: wp.array2d(dtype=float),
                           c_work_inv_old_rot: wp.array2d(dtype=float),
                           c_work_inv_rot: wp.array2d(dtype=float),
                           c_work_radius: wp.array2d(dtype=float),
                           c_work_old_pos: wp.array3d(dtype=float),
                           c_work_next_pos: wp.array3d(dtype=float),
                           c_work_aabb_min: wp.array2d(dtype=float),
                           c_work_aabb_max: wp.array2d(dtype=float),
                           c_mesh_bound_min: wp.array2d(dtype=float),
                           c_mesh_bound_max: wp.array2d(dtype=float)):
    posx = c_frame_pos[ci, 0]
    posy = c_frame_pos[ci, 1]
    posz = c_frame_pos[ci, 2]
    tipx = c_frame_tip[ci, 0]
    tipy = c_frame_tip[ci, 1]
    tipz = c_frame_tip[ci, 2]
    rotx = c_frame_rot[ci, 0]
    roty = c_frame_rot[ci, 1]
    rotz = c_frame_rot[ci, 2]
    rotw = c_frame_rot[ci, 3]
    collider_write_work_pose(ci, c_kind[ci], posx, posy, posz, tipx, tipy, tipz, rotx,
                             roty, rotz, rotw, posx, posy, posz, tipx, tipy, tipz, rotx,
                             roty, rotz, rotw, c_frame_radius, c_work_rot,
                             c_work_inv_old_rot, c_work_inv_rot, c_work_radius,
                             c_work_old_pos, c_work_next_pos, c_work_aabb_min,
                             c_work_aabb_max, c_mesh_bound_min, c_mesh_bound_max)


@wp.func
def do_collider_end_step(ci: int, c_now_pos: wp.array2d(dtype=float),
                         c_now_rot: wp.array2d(dtype=float),
                         c_now_tip: wp.array2d(dtype=float),
                         c_old_pos: wp.array2d(dtype=float),
                         c_old_rot: wp.array2d(dtype=float),
                         c_old_tip: wp.array2d(dtype=float)):
    for j in range(3):
        c_old_pos[ci, j] = c_now_pos[ci, j]
        c_old_tip[ci, j] = c_now_tip[ci, j]
    for j in range(4):
        c_old_rot[ci, j] = c_now_rot[ci, j]


@wp.func
def do_collider_frame_post(ci: int, c_frame_pos: wp.array2d(dtype=float),
                           c_frame_rot: wp.array2d(dtype=float),
                           c_frame_tip: wp.array2d(dtype=float),
                           c_old_frame_pos: wp.array2d(dtype=float),
                           c_old_frame_rot: wp.array2d(dtype=float),
                           c_old_frame_tip: wp.array2d(dtype=float)):
    for j in range(3):
        c_old_frame_pos[ci, j] = c_frame_pos[ci, j]
        c_old_frame_tip[ci, j] = c_frame_tip[ci, j]
    for j in range(4):
        c_old_frame_rot[ci, j] = c_frame_rot[ci, j]


COLLIDER_SPRING_RELEASE = wp.constant(0.85)

COLLIDER_SPRING_GAP_SCALE = wp.constant(3.0)

COLLIDER_PAIR_PLANE_REASON = (
    "a contact against one body is work for one thread, and gathering the contacts of one "
    "particle is work for another; putting both in one kernel means the thread that walks "
    "the field also carries twelve float64 accumulators and a candidate loop, and warp "
    "inlines the field at every naming, so the two costs add on top of a mesh closest "
    "point query that is seventy seven registers by itself; the pair writes its answer "
    "into a row of its own and the particle reads the rows the compressed row table "
    "already points it at, which is the shape the self collision path has had since W3, "
    "where a query family fills slots and a later family collects them; the split also "
    "ends a second waste, which is that the edge family used to run for every team and "
    "return at once for the teams that are not in edge mode")


@wp.func
def do_measure_point_contact(p: int, c: int, team: int,
                             p_next_positions: wp.array2d(dtype=float),
                             p_old_positions: wp.array2d(dtype=float),
                             p_depth: wp.array(dtype=float),
                             t_radius_lut: wp.array2d(dtype=float),
                             t_scale_ratio: wp.array(dtype=float),
                             c_kind: wp.array(dtype=int),
                             c_active: wp.array(dtype=int),
                             c_work_old_pos: wp.array3d(dtype=float),
                             c_work_next_pos: wp.array3d(dtype=float),
                             c_work_radius: wp.array2d(dtype=float),
                             c_work_inv_old_rot: wp.array2d(dtype=float),
                             c_work_rot: wp.array2d(dtype=float),
                             c_work_inv_rot: wp.array2d(dtype=float),
                             face_index: wp.uint64,
                             cf_vertex: wp.array2d(dtype=int),
                             cf_edge_normal: wp.array3d(dtype=float),
                             cf_normal: wp.array2d(dtype=float),
                             cv_local_position: wp.array2d(dtype=float),
                             cv_pseudo_normal: wp.array2d(dtype=float),
                             c_work_aabb_min: wp.array2d(dtype=float),
                             c_work_aabb_max: wp.array2d(dtype=float),
                             incidence_gate_cos: float):
    if c_active[c] == 0:
        return 0.0, float(wp.inf), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    radius = dmath.evaluate_team_lut(t_radius_lut, team, p_depth[p])
    if radius < 0.0001:
        radius = 0.0001
    radius = radius * t_scale_ratio[team]
    npx = p_next_positions[p, 0]
    npy = p_next_positions[p, 1]
    npz = p_next_positions[p, 2]
    tsx, tsy, tsz = collider_transport(c, p_old_positions[p, 0], p_old_positions[p, 1],
                                       p_old_positions[p, 2], c_work_old_pos,
                                       c_work_next_pos, c_work_rot, c_work_inv_old_rot)
    if collider_bound_misses_segment(c, tsx, tsy, tsz, radius, npx, npy, npz, radius,
                                     radius, c_work_aabb_min, c_work_aabb_max):
        return 1.0, float(wp.inf), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dist, ox, oy, oz, ux, uy, uz = collider_contact(
        c, tsx, tsy, tsz, npx, npy, npz, radius, radius, incidence_gate_cos, c_kind,
        c_work_next_pos, c_work_radius, c_work_rot, c_work_inv_rot, c_work_aabb_min,
        c_work_aabb_max, face_index,
        cf_vertex, cf_edge_normal, cf_normal, cv_local_position, cv_pseudo_normal)
    return 1.0, dist, ox - npx, oy - npy, oz - npz, ux, uy, uz


@wp.func
def do_spring_response(p: int, team: int, dist: float, cx: float, cy: float, cz: float,
                       p_next_positions: wp.array2d(dtype=float),
                       p_base_positions: wp.array2d(dtype=float),
                       p_depth: wp.array(dtype=float),
                       t_radius_lut: wp.array2d(dtype=float),
                       t_scale_ratio: wp.array(dtype=float),
                       t_limit_distance_lut: wp.array2d(dtype=float)):
    depth = p_depth[p]
    radius = dmath.evaluate_team_lut(t_radius_lut, team, depth)
    if radius < 0.0001:
        radius = 0.0001
    radius = radius * t_scale_ratio[team]
    max_length = dmath.evaluate_team_lut(t_limit_distance_lut, team, depth)
    if max_length < 0.0001:
        max_length = 0.0001
    npx = p_next_positions[p, 0]
    npy = p_next_positions[p, 1]
    npz = p_next_positions[p, 2]
    bpx = p_base_positions[p, 0]
    bpy = p_base_positions[p, 1]
    bpz = p_base_positions[p, 2]
    clx, cly, clz = dmath.clamp_distance(bpx, bpy, bpz, npx + cx, npy + cy, npz + cz,
                                         max_length * t_scale_ratio[team])
    tspr = dmath.saturate(dmath.length3(clx - bpx, cly - bpy, clz - bpz) / radius) \
        * COLLIDER_SPRING_RELEASE
    return (dist * COLLIDER_SPRING_GAP_SCALE, dmath.lerp(clx, npx, tspr) - npx,
            dmath.lerp(cly, npy, tspr) - npy, dmath.lerp(clz, npz, tspr) - npz)


@wp.func
def do_solve_point_gather(p: int, p_team: wp.array(dtype=int),
                          p_depth: wp.array(dtype=float),
                          t_collision_mode: wp.array(dtype=int),
                          t_radius_lut: wp.array2d(dtype=float),
                          t_scale_ratio: wp.array(dtype=float),
                          csr_off: wp.array(dtype=int),
                          csr_ord: wp.array(dtype=int),
                          pair_contact: wp.array2d(dtype=float)):
    team = p_team[p]
    start = csr_off[p]
    stop = csr_off[p + 1]
    if t_collision_mode[team] != COLLISION_POINT or start == stop:
        return int(0), int(0), int(0), float(wp.inf), \
            wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), \
            wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), \
            wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    radius = dmath.evaluate_team_lut(t_radius_lut, team, p_depth[p])
    if radius < 0.0001:
        radius = 0.0001
    cfr = radius * t_scale_ratio[team]
    count = int(0)
    near_count = int(0)
    psumx = wp.float64(0.0)
    psumy = wp.float64(0.0)
    psumz = wp.float64(0.0)
    nsumx = wp.float64(0.0)
    nsumy = wp.float64(0.0)
    nsumz = wp.float64(0.0)
    nnx = wp.float64(0.0)
    nny = wp.float64(0.0)
    nnz = wp.float64(0.0)
    min_dist = float(wp.inf)
    any_active = wp.bool(False)
    for k in range(start, stop):
        row = csr_ord[k]
        if pair_contact[row, 0] == 0.0:
            continue
        any_active = True
        dist = pair_contact[row, 1]
        if dist <= 0.0:
            count += 1
            psumx += wp.float64(pair_contact[row, 2])
            psumy += wp.float64(pair_contact[row, 3])
            psumz += wp.float64(pair_contact[row, 4])
            nsumx += wp.float64(pair_contact[row, 5])
            nsumy += wp.float64(pair_contact[row, 6])
            nsumz += wp.float64(pair_contact[row, 7])
        if dist <= cfr:
            near_count += 1
            nnx += wp.float64(pair_contact[row, 5])
            nny += wp.float64(pair_contact[row, 6])
            nnz += wp.float64(pair_contact[row, 7])
            if dist < min_dist:
                min_dist = dist
    active = int(1) if any_active else int(0)
    return active, count, near_count, min_dist, \
        psumx, psumy, psumz, nsumx, nsumy, nsumz, nnx, nny, nnz


@wp.func
def do_solve_point_resolve(p: int, active: int, count: int, near_count: int,
                           min_dist: float,
                           psumx: wp.float64, psumy: wp.float64, psumz: wp.float64,
                           nsumx: wp.float64, nsumy: wp.float64, nsumz: wp.float64,
                           nnx: wp.float64, nny: wp.float64, nnz: wp.float64,
                           p_team: wp.array(dtype=int),
                           p_next_positions: wp.array2d(dtype=float),
                           p_depth: wp.array(dtype=float),
                           p_friction: wp.array(dtype=float),
                           p_collision_normals: wp.array2d(dtype=float),
                           p_velocity_positions: wp.array2d(dtype=float),
                           t_radius_lut: wp.array2d(dtype=float),
                           t_scale_ratio: wp.array(dtype=float),
                           t_is_spring: wp.array(dtype=int)):
    if active == 0:
        return
    team = p_team[p]
    depth = p_depth[p]
    radius = dmath.evaluate_team_lut(t_radius_lut, team, depth)
    if radius < 0.0001:
        radius = 0.0001
    radius = radius * t_scale_ratio[team]
    cfr = radius
    npx = p_next_positions[p, 0]
    npy = p_next_positions[p, 1]
    npz = p_next_positions[p, 2]
    is_spring = t_is_spring[team] != 0
    has_push = count > 0
    sc = float(count) if count > 0 else 1.0
    navx = wp.float32(nsumx) / sc
    navy = wp.float32(nsumy) / sc
    navz = wp.float32(nsumz) / sc
    normal_length = dmath.length3(navx, navy, navz)
    pavx = wp.float32(psumx) / sc
    pavy = wp.float32(psumy) / sc
    pavz = wp.float32(psumz) / sc
    tclamp = normal_length if normal_length < 1.0 else 1.0
    if has_push and (normal_length >= EPSILON):
        p_next_positions[p, 0] = npx + pavx * tclamp
        p_next_positions[p, 1] = npy + pavy * tclamp
        p_next_positions[p, 2] = npz + pavz * tclamp
    nsx = wp.float32(nnx)
    nsy = wp.float32(nny)
    nsz = wp.float32(nnz)
    near_len = dmath.length3(nsx, nsy, nsz)
    has_near = (near_count > 0) and (cfr > 0.0) and (near_len * near_len > 1e-6)
    md = min_dist if (min_dist < wp.inf) else 0.0
    denom_cfr = cfr if cfr > 0.0 else 1.0
    friction_val = 1.0 - dmath.saturate(md / denom_cfr)
    if has_near and (friction_val > p_friction[p]):
        p_friction[p] = friction_val
    if has_near:
        onx, ony, onz = dmath.normalize3(nsx, nsy, nsz)
        p_collision_normals[p, 0] = onx
        p_collision_normals[p, 1] = ony
        p_collision_normals[p, 2] = onz
    else:
        p_collision_normals[p, 0] = nsx
        p_collision_normals[p, 1] = nsy
        p_collision_normals[p, 2] = nsz
    if is_spring and has_push:
        p_velocity_positions[p, 0] = p_velocity_positions[p, 0] + pavx
        p_velocity_positions[p, 1] = p_velocity_positions[p, 1] + pavy
        p_velocity_positions[p, 2] = p_velocity_positions[p, 2] + pavz


COLLIDER_ONE_SITE_REASON = (
    "no kernel here names collider_field more than once, and that is a rule and not an "
    "accident: warp inlines every naming and a mesh closest point query is seventy seven "
    "registers with a hundred and twenty eight bytes of stack, so a second naming costs "
    "about eighty registers on top of the first, measured entry by entry with "
    "kernelregs.py; the edge path needs two answers from the field, where along the edge "
    "the body is closest and what the contact there is, so it asks them in two families "
    "with a row of a plane between them, the same shape the point path uses and the same "
    "shape the self collision path has had since W3")


@wp.func
def do_measure_edge_foot(e0: int, e1: int, c: int, team: int,
                         p_next_positions: wp.array2d(dtype=float),
                         p_old_positions: wp.array2d(dtype=float),
                         p_depth: wp.array(dtype=float),
                         t_radius_lut: wp.array2d(dtype=float),
                         t_scale_ratio: wp.array(dtype=float),
                         c_active: wp.array(dtype=int),
                         c_kind: wp.array(dtype=int),
                         c_work_old_pos: wp.array3d(dtype=float),
                         c_work_next_pos: wp.array3d(dtype=float),
                         c_work_radius: wp.array2d(dtype=float),
                         c_work_inv_old_rot: wp.array2d(dtype=float),
                         c_work_rot: wp.array2d(dtype=float),
                         c_work_inv_rot: wp.array2d(dtype=float),
                         face_index: wp.uint64,
                         cf_vertex: wp.array2d(dtype=int),
                         cf_edge_normal: wp.array3d(dtype=float),
                         cf_normal: wp.array2d(dtype=float),
                         cv_local_position: wp.array2d(dtype=float),
                         cv_pseudo_normal: wp.array2d(dtype=float),
                         c_work_aabb_min: wp.array2d(dtype=float),
                         c_work_aabb_max: wp.array2d(dtype=float),
                         incidence_gate_cos: float):
    if c_active[c] == 0:
        return 0.5
    p0x = p_next_positions[e0, 0]
    p0y = p_next_positions[e0, 1]
    p0z = p_next_positions[e0, 2]
    p1x = p_next_positions[e1, 0]
    p1y = p_next_positions[e1, 1]
    p1z = p_next_positions[e1, 2]
    r0 = dmath.evaluate_team_lut(t_radius_lut, team, p_depth[e0]) * t_scale_ratio[team]
    r1 = dmath.evaluate_team_lut(t_radius_lut, team, p_depth[e1]) * t_scale_ratio[team]
    widest = dmath.fmax2(r0, r1)
    if collider_bound_misses_edge_sweep(c, e0, e1, r0, r1, widest, p_next_positions,
                                        p_old_positions, c_work_old_pos, c_work_next_pos,
                                        c_work_rot, c_work_inv_old_rot, c_work_aabb_min,
                                        c_work_aabb_max):
        return 0.5
    length = dmath.length3(p1x - p0x, p1y - p0y, p1z - p0z)
    reach = length + dmath.fmax2(widest + widest,
                                 collider_bound_inward(c, p0x, p0y, p0z, c_work_aabb_min,
                                                       c_work_aabb_max))
    return collider_edge_foot(c, p0x, p0y, p0z, p1x, p1y, p1z, reach,
                              incidence_gate_cos, c_kind, c_work_next_pos, c_work_radius,
                              c_work_rot, c_work_inv_rot, face_index, cf_vertex,
                              cf_edge_normal, cf_normal, cv_local_position,
                              cv_pseudo_normal)


@wp.func
def do_measure_edge_contact(e0: int, e1: int, c: int, team: int, foot: float,
                            p_next_positions: wp.array2d(dtype=float),
                            p_old_positions: wp.array2d(dtype=float),
                            p_depth: wp.array(dtype=float),
                            t_radius_lut: wp.array2d(dtype=float),
                            t_scale_ratio: wp.array(dtype=float),
                            c_kind: wp.array(dtype=int),
                            c_active: wp.array(dtype=int),
                            c_work_old_pos: wp.array3d(dtype=float),
                            c_work_next_pos: wp.array3d(dtype=float),
                            c_work_radius: wp.array2d(dtype=float),
                            c_work_inv_old_rot: wp.array2d(dtype=float),
                            c_work_rot: wp.array2d(dtype=float),
                            c_work_inv_rot: wp.array2d(dtype=float),
                            face_index: wp.uint64,
                            cf_vertex: wp.array2d(dtype=int),
                            cf_edge_normal: wp.array3d(dtype=float),
                            cf_normal: wp.array2d(dtype=float),
                            cv_local_position: wp.array2d(dtype=float),
                            cv_pseudo_normal: wp.array2d(dtype=float),
                            c_work_aabb_min: wp.array2d(dtype=float),
                            c_work_aabb_max: wp.array2d(dtype=float),
                            incidence_gate_cos: float):
    if c_active[c] == 0:
        return (0.0, float(wp.inf), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    p0x = p_next_positions[e0, 0]
    p0y = p_next_positions[e0, 1]
    p0z = p_next_positions[e0, 2]
    p1x = p_next_positions[e1, 0]
    p1y = p_next_positions[e1, 1]
    p1z = p_next_positions[e1, 2]
    r0 = dmath.evaluate_team_lut(t_radius_lut, team, p_depth[e0]) * t_scale_ratio[team]
    r1 = dmath.evaluate_team_lut(t_radius_lut, team, p_depth[e1]) * t_scale_ratio[team]
    cfr = (r0 + r1) * 0.5
    if collider_bound_misses_edge_sweep(c, e0, e1, r0, r1, cfr, p_next_positions,
                                        p_old_positions, c_work_old_pos, c_work_next_pos,
                                        c_work_rot, c_work_inv_old_rot, c_work_aabb_min,
                                        c_work_aabb_max):
        return (1.0, float(wp.inf), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    tsx, tsy, tsz = collider_transport(
        c, p_old_positions[e0, 0] + (p_old_positions[e1, 0] - p_old_positions[e0, 0]) * foot,
        p_old_positions[e0, 1] + (p_old_positions[e1, 1] - p_old_positions[e0, 1]) * foot,
        p_old_positions[e0, 2] + (p_old_positions[e1, 2] - p_old_positions[e0, 2]) * foot,
        c_work_old_pos, c_work_next_pos, c_work_rot, c_work_inv_old_rot)
    bx = p0x + (p1x - p0x) * foot
    by = p0y + (p1y - p0y) * foot
    bz = p0z + (p1z - p0z) * foot
    d, ox, oy, oz, ux, uy, uz = collider_contact(
        c, tsx, tsy, tsz, bx, by, bz, r0 + (r1 - r0) * foot, cfr, incidence_gate_cos,
        c_kind,
        c_work_next_pos, c_work_radius, c_work_rot, c_work_inv_rot, c_work_aabb_min,
        c_work_aabb_max, face_index,
        cf_vertex, cf_edge_normal, cf_normal, cv_local_position, cv_pseudo_normal)
    b0 = 1.0 - foot
    denom = b0 * b0 + foot * foot
    share = 1.0 / (denom if denom > 0.0 else 1.0)
    w0 = b0 * share
    w1 = foot * share
    return (1.0, d, (ox - bx) * w0, (oy - by) * w0, (oz - bz) * w0,
            (ox - bx) * w1, (oy - by) * w1, (oz - bz) * w1, ux, uy, uz)


@wp.func
def do_solve_edge(ee: int, p_team: wp.array(dtype=int),
                  p_depth: wp.array(dtype=float),
                  p_attr_move: wp.array(dtype=int),
                  t_radius_lut: wp.array2d(dtype=float),
                  t_scale_ratio: wp.array(dtype=float),
                  csr_off: wp.array(dtype=int),
                  csr_ord: wp.array(dtype=int),
                  st_collision_edge: wp.array2d(dtype=int),
                  pair_contact: wp.array2d(dtype=float),
                  sc_dcorr_fixed: wp.array2d(dtype=wp.int64),
                  sc_dcount: wp.array(dtype=wp.int64),
                  sc_col_friction_fixed: wp.array(dtype=wp.int64),
                  sc_col_normal_fixed: wp.array2d(dtype=wp.int64)):
    e0 = st_collision_edge[ee, 0]
    e1 = st_collision_edge[ee, 1]
    team = p_team[e0]
    start = csr_off[ee]
    stop = csr_off[ee + 1]
    if start == stop:
        return
    r0 = dmath.evaluate_team_lut(t_radius_lut, team, p_depth[e0]) * t_scale_ratio[team]
    r1 = dmath.evaluate_team_lut(t_radius_lut, team, p_depth[e1]) * t_scale_ratio[team]
    cfr = (r0 + r1) * 0.5
    count = int(0)
    near_count = int(0)
    c0sx = wp.float64(0.0)
    c0sy = wp.float64(0.0)
    c0sz = wp.float64(0.0)
    c1sx = wp.float64(0.0)
    c1sy = wp.float64(0.0)
    c1sz = wp.float64(0.0)
    nsumx = wp.float64(0.0)
    nsumy = wp.float64(0.0)
    nsumz = wp.float64(0.0)
    nnx = wp.float64(0.0)
    nny = wp.float64(0.0)
    nnz = wp.float64(0.0)
    min_dist = float(wp.inf)
    any_active = wp.bool(False)
    for k in range(start, stop):
        row = csr_ord[k]
        if pair_contact[row, 0] == 0.0:
            continue
        any_active = True
        d = pair_contact[row, 1]
        if d <= 0.0:
            count += 1
            c0sx += wp.float64(pair_contact[row, 2])
            c0sy += wp.float64(pair_contact[row, 3])
            c0sz += wp.float64(pair_contact[row, 4])
            c1sx += wp.float64(pair_contact[row, 5])
            c1sy += wp.float64(pair_contact[row, 6])
            c1sz += wp.float64(pair_contact[row, 7])
            nsumx += wp.float64(pair_contact[row, 8])
            nsumy += wp.float64(pair_contact[row, 9])
            nsumz += wp.float64(pair_contact[row, 10])
        if d <= cfr:
            near_count += 1
            nnx += wp.float64(pair_contact[row, 8])
            nny += wp.float64(pair_contact[row, 9])
            nnz += wp.float64(pair_contact[row, 10])
            if d < min_dist:
                min_dist = d
    if not any_active:
        return
    has_push = count > 0
    sc = float(count) if count > 0 else 1.0
    navx = wp.float32(nsumx) / sc
    navy = wp.float32(nsumy) / sc
    navz = wp.float32(nsumz) / sc
    normal_length = dmath.length3(navx, navy, navz)
    tclamp = normal_length if normal_length < 1.0 else 1.0
    valid = has_push and (normal_length > EPSILON)
    scale = (tclamp if valid else 0.0) / sc
    d0x = wp.float32(c0sx) * scale
    d0y = wp.float32(c0sy) * scale
    d0z = wp.float32(c0sz) * scale
    d1x = wp.float32(c1sx) * scale
    d1y = wp.float32(c1sy) * scale
    d1z = wp.float32(c1sz) * scale
    nsx = wp.float32(nnx)
    nsy = wp.float32(nny)
    nsz = wp.float32(nnz)
    near_len = dmath.length3(nsx, nsy, nsz)
    has_near = (near_count > 0) and (cfr > 0.0) and (near_len * near_len > 1e-6)
    md = min_dist if (min_dist < wp.inf) else 0.0
    denom_cfr = cfr if cfr > 0.0 else 1.0
    friction = 1.0 - dmath.saturate(md / denom_cfr)
    if has_near:
        noutx, nouty, noutz = dmath.normalize3(nsx, nsy, nsz)
    else:
        noutx = 0.0
        nouty = 0.0
        noutz = 0.0
    move0 = p_attr_move[e0] != 0
    move1 = p_attr_move[e1] != 0
    mask0 = has_near and move0
    mask1 = has_near and move1
    vint = wp.int64(1 if valid else 0)
    wp.atomic_add(sc_dcorr_fixed, e0, 0, wp.int64(d0x * TO_FIXED))
    wp.atomic_add(sc_dcorr_fixed, e0, 1, wp.int64(d0y * TO_FIXED))
    wp.atomic_add(sc_dcorr_fixed, e0, 2, wp.int64(d0z * TO_FIXED))
    wp.atomic_add(sc_dcount, e0, vint)
    wp.atomic_add(sc_dcorr_fixed, e1, 0, wp.int64(d1x * TO_FIXED))
    wp.atomic_add(sc_dcorr_fixed, e1, 1, wp.int64(d1y * TO_FIXED))
    wp.atomic_add(sc_dcorr_fixed, e1, 2, wp.int64(d1z * TO_FIXED))
    wp.atomic_add(sc_dcount, e1, vint)
    f0 = friction if mask0 else 0.0
    f1 = friction if mask1 else 0.0
    wp.atomic_max(sc_col_friction_fixed, e0, wp.int64(f0 * TO_FIXED))
    wp.atomic_max(sc_col_friction_fixed, e1, wp.int64(f1 * TO_FIXED))
    if mask0:
        wp.atomic_add(sc_col_normal_fixed, e0, 0, wp.int64(noutx * TO_FIXED))
        wp.atomic_add(sc_col_normal_fixed, e0, 1, wp.int64(nouty * TO_FIXED))
        wp.atomic_add(sc_col_normal_fixed, e0, 2, wp.int64(noutz * TO_FIXED))
    if mask1:
        wp.atomic_add(sc_col_normal_fixed, e1, 0, wp.int64(noutx * TO_FIXED))
        wp.atomic_add(sc_col_normal_fixed, e1, 1, wp.int64(nouty * TO_FIXED))
        wp.atomic_add(sc_col_normal_fixed, e1, 2, wp.int64(noutz * TO_FIXED))


SELF_CONTACT_GAP_REASON = (
    "how deep a self collision contact still is is the one number every stage of the self "
    "collision response argues about, and a stage that measured it its own way would be "
    "arguing about a different contact than the stage before it; so it is written once "
    "here, as a function of the values the caller has already loaded, and every reader "
    "calls it")


@wp.func
def do_self_edge_contact_gap(a0: int, a1: int, b0: int, b1: int, s: float, t: float,
                             nx: float, ny: float, nz: float,
                             p_next: wp.array2d(dtype=float)):
    ax = dmath.lerp(p_next[a0, 0], p_next[a1, 0], s)
    ay = dmath.lerp(p_next[a0, 1], p_next[a1, 1], s)
    az = dmath.lerp(p_next[a0, 2], p_next[a1, 2], s)
    bx = dmath.lerp(p_next[b0, 0], p_next[b1, 0], t)
    by = dmath.lerp(p_next[b0, 1], p_next[b1, 1], t)
    bz = dmath.lerp(p_next[b0, 2], p_next[b1, 2], t)
    return nx * (ax - bx) + ny * (ay - by) + nz * (az - bz)


SELF_GAP_ORIGIN_REASON = (
    "the barycentric weights sum to one, so the contact point on the triangle is its first "
    "corner plus a combination of the two edges leaving that corner, and the gap is written "
    "that way rather than as a weighted sum of the three corners; the two forms are the "
    "same number in exact arithmetic and they are not the same number in float32, because "
    "the weighted sum rounds three products and two sums at the magnitude the garment sits "
    "at in the world while the gap it is subtracted from is millimetres, and the form below "
    "does every one of those steps at the magnitude of a triangle edge instead; that is not "
    "a tidiness argument, it is the difference between a gap that is the same number when "
    "the world is moved and one that is not, measured on the translation equivariance "
    "residual, and the whole point of the criterion is that a solver may not care where the "
    "world origin is")


@wp.func
def do_self_point_contact_gap(pp: int, t0: int, t1: int, t2: int,
                              u: float, v: float, w: float,
                              nx: float, ny: float, nz: float,
                              p_next: wp.array2d(dtype=float)):
    cornerx = p_next[t0, 0]
    cornery = p_next[t0, 1]
    cornerz = p_next[t0, 2]
    offx = (p_next[t1, 0] - cornerx) * v + (p_next[t2, 0] - cornerx) * w
    offy = (p_next[t1, 1] - cornery) * v + (p_next[t2, 1] - cornery) * w
    offz = (p_next[t1, 2] - cornerz) * v + (p_next[t2, 2] - cornerz) * w
    return (nx * (p_next[pp, 0] - cornerx - offx)
            + ny * (p_next[pp, 1] - cornery - offy)
            + nz * (p_next[pp, 2] - cornerz - offz))


@wp.func
def do_self_update_primitive(prim: int, axes: int,
                             a_team: wp.array(dtype=int),
                             a_particles: wp.array2d(dtype=int),
                             a_fix: wp.array(dtype=int),
                             a_ignore: wp.array(dtype=int),
                             a_prim_depth: wp.array(dtype=float),
                             a_inv_mass: wp.array2d(dtype=float),
                             a_thickness: wp.array(dtype=float),
                             a_aabb_min: wp.array2d(dtype=float),
                             a_aabb_max: wp.array2d(dtype=float),
                             a_intersect: wp.array(dtype=int),
                             a_use: wp.array(dtype=int),
                             t_use_flag: wp.array(dtype=int),
                             t_thickness_lut: wp.array2d(dtype=float),
                             t_cloth_mass: wp.array(dtype=float),
                             t_scale_ratio: wp.array(dtype=float),
                             t_enabled: wp.array(dtype=int),
                             t_valid: wp.array(dtype=int),
                             t_cws: wp.array2d(dtype=float),
                             t_update_count: wp.array(dtype=int),
                             p_next: wp.array2d(dtype=float),
                             p_old: wp.array2d(dtype=float),
                             p_friction: wp.array(dtype=float),
                             p_iflag: wp.array(dtype=int),
                             scl_counts: wp.array(dtype=int),
                             scl_max_fixed: wp.array(dtype=wp.int64),
                             k: int):
    team = a_team[prim]
    if not (team_frame_mask(t_enabled, t_valid, t_cws, team) and t_update_count[team] > k
            and t_use_flag[team] != 0):
        a_use[prim] = 0
        return
    a_use[prim] = 1
    fix_mask = a_fix[prim]
    thickness = dmath.evaluate_team_lut(t_thickness_lut, team,
                                        a_prim_depth[prim]) * t_scale_ratio[team]
    a_thickness[prim] = thickness
    cloth_mass = t_cloth_mass[team]
    use_intersect = scl_counts[SCL_USE_INTERSECT] != 0
    imask = int(0)
    lowx = float(1.0e30)
    lowy = float(1.0e30)
    lowz = float(1.0e30)
    highx = float(-1.0e30)
    highy = float(-1.0e30)
    highz = float(-1.0e30)
    for slot in range(axes):
        raw = a_particles[prim, slot]
        pp = raw if raw >= 0 else int(0)
        fixed = ((fix_mask >> slot) & int(1)) != 0
        a_inv_mass[prim, slot] = dmath.calc_self_collision_inverse_mass(
            p_friction[pp], fixed, cloth_mass)
        nx = p_next[pp, 0]
        ny = p_next[pp, 1]
        nz = p_next[pp, 2]
        ox = p_old[pp, 0]
        oy = p_old[pp, 1]
        oz = p_old[pp, 2]
        slx = nx if nx < ox else ox
        shx = nx if nx > ox else ox
        sly = ny if ny < oy else oy
        shy = ny if ny > oy else oy
        slz = nz if nz < oz else oz
        shz = nz if nz > oz else oz
        if slx < lowx:
            lowx = slx
        if shx > highx:
            highx = shx
        if sly < lowy:
            lowy = sly
        if shy > highy:
            highy = shy
        if slz < lowz:
            lowz = slz
        if shz > highz:
            highz = shz
        if use_intersect and p_iflag[pp] != 0:
            imask = imask | (int(1) << slot)
    a_intersect[prim] = imask
    a_aabb_min[prim, 0] = lowx - thickness
    a_aabb_min[prim, 1] = lowy - thickness
    a_aabb_min[prim, 2] = lowz - thickness
    a_aabb_max[prim, 0] = highx + thickness
    a_aabb_max[prim, 1] = highy + thickness
    a_aabb_max[prim, 2] = highz + thickness
    size = highx - lowx
    ey = highy - lowy
    ez = highz - lowz
    if ey > size:
        size = ey
    if ez > size:
        size = ez
    if a_ignore[prim] == 0:
        wp.atomic_max(scl_max_fixed, team, wp.int64(size * TO_FIXED))


@wp.func
def self_aabb_overlap(a_min: wp.array2d(dtype=float), a_max: wp.array2d(dtype=float), i: int,
                      b_min: wp.array2d(dtype=float), b_max: wp.array2d(dtype=float), j: int):
    return (a_min[i, 0] <= b_max[j, 0] and a_max[i, 0] >= b_min[j, 0]
            and a_min[i, 1] <= b_max[j, 1] and a_max[i, 1] >= b_min[j, 1]
            and a_min[i, 2] <= b_max[j, 2] and a_max[i, 2] >= b_min[j, 2])


@wp.func
def gap_order_key(gap: float):
    if gap != gap:
        return GAP_ORDER_LIMIT_KEY
    scaled = gap * GAP_ORDER_SCALE
    if scaled >= GAP_ORDER_LIMIT_VALUE:
        return GAP_ORDER_LIMIT_KEY
    if scaled <= -GAP_ORDER_LIMIT_VALUE:
        return -GAP_ORDER_LIMIT_KEY
    return int(scaled)


@wp.func
def self_ranks_before(left_key: int, left_target: int, right_key: int, right_target: int):
    if left_key != right_key:
        return left_key < right_key
    return left_target < right_target


@wp.func
def self_worst_slot(gap_key: wp.array(dtype=int), target: wp.array(dtype=int), base: int,
                    capacity: int):
    worst = int(0)
    for slot in range(1, capacity):
        if self_ranks_before(gap_key[base + worst], target[base + worst],
                             gap_key[base + slot], target[base + slot]):
            worst = slot
    return worst


@wp.func
def self_box_gap(a_min: wp.array2d(dtype=float), a_max: wp.array2d(dtype=float), i: int,
                 b_min: wp.array2d(dtype=float), b_max: wp.array2d(dtype=float), j: int):
    gap = a_min[i, 0] - b_max[j, 0]
    other = b_min[j, 0] - a_max[i, 0]
    if other > gap:
        gap = other
    for axis in range(1, 3):
        low = a_min[i, axis] - b_max[j, axis]
        if low > gap:
            gap = low
        high = b_min[j, axis] - a_max[i, axis]
        if high > gap:
            gap = high
    return gap


@wp.func
def self_connection_shared(a_particles: wp.array2d(dtype=int), i: int,
                           b_particles: wp.array2d(dtype=int), j: int):
    for x in range(3):
        pa = a_particles[i, x]
        if pa >= 0:
            for y in range(3):
                pb = b_particles[j, y]
                if pb >= 0 and pa == pb:
                    return True
    return False


SELF_ACCD_REASON = (
    "whether a pair of cloth primitives is a contact this substep used to be decided by "
    "linearising the distance over the substep and then asking whether the direction "
    "between the two primitives sat within sixty degrees of the triangle normal, which is a "
    "guess at the question a continuous test answers exactly; selfattrib.py measured that "
    "guess on the frozen asset and it refused 971 of the 1424 crossings the run produced, "
    "sixty eight per cent of them, while the broad phase, the slot budget and the task "
    "table refused none at all, so the guess was the whole of the upstream cause; the pair "
    "is therefore decided here by the same additive continuous collision detection the "
    "collider path already runs, Li, Kaufman and Jiang, Codimensional Incremental Potential "
    "Contact, ACM Transactions on Graphics 40 number 4 article 170, section 5.4 and "
    "Algorithm 1, in the general form where all four nodes carry their own displacement "
    "rather than the degenerate form where the second primitive is a rigid body; the "
    "collider walk is that degenerate form and this is the same algorithm, which is why the "
    "separation scale, the advance scale and the step limit are the constants it already "
    "declares and not a second set beside them")

SELF_ACCD_LOCAL_ORIGIN_REASON = (
    "the walk is run in coordinates measured from one node of the pair rather than from the "
    "world origin, because everything it decides is a comparison between a gap of "
    "millimetres and a thickness of millimetres while a garment on a character sits metres "
    "from the world origin, and the walk both interpolates positions and subtracts them, so "
    "at world magnitude the increment it advances by and the gap it reads are being resolved "
    "against a float32 step that is a fair fraction of them; that would be a slow leak in a "
    "straight line formula and it is not one here, because the walk turns the gap into "
    "branches, so a difference of one float32 place in the gap becomes a different number of "
    "steps and a different contact frame, which is a difference no equivariance any longer "
    "holds through; it was measured: the same kernel run against a world moved by two and "
    "three quarter metres put the translation residual p99 fifteen times further out than "
    "the engine it replaced, and the whole of that was this")

SELF_ACCD_MEAN_REASON = (
    "the distance between two primitives is unchanged when both are carried by one "
    "translation, so the bound the walk divides by has to bound the relative motion and not "
    "the motion; Algorithm 1 takes the mean displacement of the nodes out before forming "
    "the bound, and on a garment that matters more than anywhere else in this engine, "
    "because a skirt on a walking body carries a whole body of translation that no two of "
    "its own triangles are closing on each other with, and a bound that counted it would "
    "report every pair in the garment as a contact")

SELF_CONTACT_SIDE_REASON = (
    "which side of a contact each primitive is on is decided when the contact is created "
    "and is not decided again while it lives; the distance between a point and a triangle "
    "and the distance between two segments carry no sign, so a pair that crossed during an "
    "earlier substep reads as separated again on the far side, and a detector that took the "
    "new reading at face value would ratify the crossing and then hold the pair there, "
    "which is the one outcome the response exists to prevent; so the frame the walk "
    "measures is turned to agree with the frame the contact was opened with, and a pair "
    "that crossed therefore reads a negative gap and is pushed back the way it came; the "
    "engine did this before with a sign frozen at the first substep of the frame, and this "
    "keeps that memory while letting the closest feature itself move, which is what the "
    "frozen sign could not do")


@wp.func
def self_contact_side_keep(nx: float, ny: float, nz: float,
                           sx: float, sy: float, sz: float):
    if nx * sx + ny * sy + nz * sz < 0.0:
        return dmath.negate(nx), dmath.negate(ny), dmath.negate(nz)
    return nx, ny, nz


SELF_CONTACT_SLOT_KEPT_REASON = (
    "which pairs are tracked is decided once a frame by the broad phase and the ranking, "
    "and whether a tracked pair is a contact right now is decided every substep by the walk "
    "below; the query used to conflate the two by writing minus one over the source and the "
    "target of a pair its narrow phase refused, which threw the pair out of the frame "
    "entirely, and that was survivable only because the test it refused on carried three "
    "thicknesses of slack and therefore refused almost nothing that could matter later; an "
    "exact test has no slack to spare, so a pair it proves safe for the first substep would "
    "be discarded before the substep it actually collides in, which is the tunnel the exact "
    "test was brought in to close; the slot is therefore kept for the whole frame and the "
    "enabled flag alone carries the per substep answer")

SELF_ACCD_REACH_REASON = (
    "the distance between the two primitives cannot fall faster than the bound the walk "
    "divides by, so a pair whose gap already exceeds that bound cannot close inside the "
    "step at all and is answered by the first reading rather than by a second one taken at "
    "the end of the step; that is the common case by a wide margin, because the list holds "
    "every pair whose swept boxes touched and most of them are merely near, and it is the "
    "difference between naming the closest point function once per pair and naming it "
    "twice; it is also strictly the more accurate answer, because the walk's own stopping "
    "test fires when the gap has shrunk to a tenth of what it was, which a pair that "
    "closed by nine tenths of a gap it was never going to cross can satisfy without ever "
    "having been a contact")

SELF_PT_INTERIOR_REASON = (
    "a point triangle contact is one this triangle owns, and it owns the point only while "
    "the closest place on it is inside its own face; the moment that place slides onto an "
    "edge or a corner the point is beside this triangle rather than above it, the one next "
    "to it owns the contact, and the pair of edges that meet there is what the edge list is "
    "for; that is the exact form of the question the sixty degree incidence gate was asking "
    "with an angle, and asking it exactly matters for more than tidiness, because the side "
    "the response pushes to is the sign of the triangle normal against the direction to the "
    "point, and on an interior face that direction is the normal so the sign is plus or "
    "minus one, while on an edge it is perpendicular to the normal and the sign is a coin "
    "standing on its rim; a coin on its rim is not a rounding error that stays small, it is "
    "a response that reverses, and it was measured as such: with the angle gate removed and "
    "nothing exact in its place, the same kernel run against a world moved by two and three "
    "quarter metres put the translation equivariance residual seven times further out than "
    "the engine it replaced, on a criterion whose whole subject is that the solver may not "
    "care where the world origin is; the weights come out of the closest point routine's "
    "own branch, which sets a weight to exactly zero when it leaves the face, so this reads "
    "a decision that was already taken rather than taking a new one")

SELF_PT_NORMAL_REASON = (
    "the direction a point and a triangle are pushed apart along is the triangle's own "
    "normal, turned to the side the point is on, and not the direction from the closest "
    "point on the triangle to the point; Bridson section 5 registers a point triangle "
    "proximity by the barycentric weights of the point's projection and applies the impulse "
    "along the normal, and the two directions differ only where the closest point sits on "
    "an edge or a vertex, which is where the point is not above this triangle at all and is "
    "above the one next to it; taking the closest point direction there pushes a point "
    "sideways along the sheet it should be resting on, and it was measured, 1526 crossings "
    "on the closest point direction against 1424 on the normal, so the normal it is; a pair "
    "that really does meet edge on is the edge to edge list's to answer, and that list is "
    "built over the same frame")

SELF_ACCD_FRAME_REASON = (
    "the direction the response pushes along and the weights it splits the push by are read "
    "at the last pose the walk proved safe, which is the pose the two primitives were still "
    "separated in; that is the whole of the contact frame and it is well conditioned there, "
    "which is what the incidence gate was faking and what the sign carried over from the "
    "first substep was remembering, so both are gone; it has to be the last safe pose and "
    "not the pose the substep ends in, because the distance between a point and a triangle "
    "carries no sign, so a point that crossed during the substep reads as separated again "
    "on the far side and a direction taken there points the way it was already going; that "
    "was measured, not reasoned about, at 1424 crossings before and 1677 after when the "
    "frame was read at the end of the substep, and a resting pair, one already inside the "
    "thickness when the substep began, is exactly the pair that has no safe pose later than "
    "the one it started in, which is why its frame is the one at the start")


@wp.func
def self_ee_geometry(my_edge: int, tgt_edge: int, thickness: float,
                     sfe_particles: wp.array2d(dtype=int),
                     p_next: wp.array2d(dtype=float),
                     p_old: wp.array2d(dtype=float)):
    a0 = sfe_particles[my_edge, 0]
    a1 = sfe_particles[my_edge, 1]
    b0 = sfe_particles[tgt_edge, 0]
    b1 = sfe_particles[tgt_edge, 1]
    originx = p_old[a0, 0]
    originy = p_old[a0, 1]
    originz = p_old[a0, 2]
    a0x = 0.0
    a0y = 0.0
    a0z = 0.0
    a1x = p_old[a1, 0] - originx
    a1y = p_old[a1, 1] - originy
    a1z = p_old[a1, 2] - originz
    b0x = p_old[b0, 0] - originx
    b0y = p_old[b0, 1] - originy
    b0z = p_old[b0, 2] - originz
    b1x = p_old[b1, 0] - originx
    b1y = p_old[b1, 1] - originy
    b1z = p_old[b1, 2] - originz
    da0x = p_next[a0, 0] - originx
    da0y = p_next[a0, 1] - originy
    da0z = p_next[a0, 2] - originz
    da1x = p_next[a1, 0] - p_old[a1, 0]
    da1y = p_next[a1, 1] - p_old[a1, 1]
    da1z = p_next[a1, 2] - p_old[a1, 2]
    db0x = p_next[b0, 0] - p_old[b0, 0]
    db0y = p_next[b0, 1] - p_old[b0, 1]
    db0z = p_next[b0, 2] - p_old[b0, 2]
    db1x = p_next[b1, 0] - p_old[b1, 0]
    db1y = p_next[b1, 1] - p_old[b1, 1]
    db1z = p_next[b1, 2] - p_old[b1, 2]
    meanx = (da0x + da1x + db0x + db1x) * 0.25
    meany = (da0y + da1y + db0y + db1y) * 0.25
    meanz = (da0z + da1z + db0z + db1z) * 0.25
    da0x = da0x - meanx
    da0y = da0y - meany
    da0z = da0z - meanz
    da1x = da1x - meanx
    da1y = da1y - meany
    da1z = da1z - meanz
    db0x = db0x - meanx
    db0y = db0y - meany
    db0z = db0z - meanz
    db1x = db1x - meanx
    db1y = db1y - meany
    db1z = db1z - meanz
    offset = thickness * SELF_CONTACT_DETECTION_MARGIN
    mine = dmath.fmax2(dmath.length3(da0x, da0y, da0z), dmath.length3(da1x, da1y, da1z))
    theirs = dmath.fmax2(dmath.length3(db0x, db0y, db0z), dmath.length3(db1x, db1y, db1z))
    travel = mine + theirs
    fallx, fally, fallz = dmath.normalize3_fb(
        (a1y - a0y) * (b1z - b0z) - (a1z - a0z) * (b1y - b0y),
        (a1z - a0z) * (b1x - b0x) - (a1x - a0x) * (b1z - b0z),
        (a1x - a0x) * (b1y - b0y) - (a1y - a0y) * (b1x - b0x), 0.0, 0.0, 1.0)
    moment = float(0.0)
    ahead = float(0.0)
    target = float(0.0)
    resting = wp.bool(False)
    settled = wp.bool(False)
    s = float(0.0)
    t = float(0.0)
    framex = float(0.0)
    framey = float(0.0)
    framez = float(0.0)
    for walk in range(ACCD_STEP_LIMIT):
        step, spot, c1x, c1y, c1z, c2x, c2y, c2z = dmath.closest_pt_segment_segment(
            a0x + da0x * ahead, a0y + da0y * ahead, a0z + da0z * ahead,
            a1x + da1x * ahead, a1y + da1y * ahead, a1z + da1z * ahead,
            b0x + db0x * ahead, b0y + db0y * ahead, b0z + db0z * ahead,
            b1x + db1x * ahead, b1y + db1y * ahead, b1z + db1z * ahead)
        reach = dmath.length3(c1x - c2x, c1y - c2y, c1z - c2z) - offset
        if walk == 0:
            s = step
            t = spot
            framex = c1x - c2x
            framey = c1y - c2y
            framez = c1z - c2z
            if reach <= 0.0:
                resting = True
                break
            if reach >= travel:
                settled = True
                break
            target = ACCD_SEPARATION_SCALE * reach
            ahead = (1.0 - ACCD_SEPARATION_SCALE) * reach / travel
            if ahead > 1.0:
                ahead = 1.0
        elif moment > 0.0 and reach < target:
            break
        else:
            s = step
            t = spot
            framex = c1x - c2x
            framey = c1y - c2y
            framez = c1z - c2z
            moment = ahead
            if ahead >= 1.0:
                settled = True
                break
            ahead = moment + ACCD_ADVANCE_SCALE * reach / travel
            if ahead > 1.0:
                ahead = 1.0
    nx, ny, nz = dmath.normalize3_fb(framex, framey, framez, fallx, fally, fallz)
    return s, t, nx, ny, nz, resting or not settled


@wp.func
def self_pt_geometry(point_prim: int, tri_prim: int, thickness: float,
                     sfp_particles: wp.array2d(dtype=int),
                     sft_particles: wp.array2d(dtype=int),
                     p_next: wp.array2d(dtype=float),
                     p_old: wp.array2d(dtype=float)):
    pp = sfp_particles[point_prim, 0]
    t0 = sft_particles[tri_prim, 0]
    t1 = sft_particles[tri_prim, 1]
    t2 = sft_particles[tri_prim, 2]
    originx = p_old[pp, 0]
    originy = p_old[pp, 1]
    originz = p_old[pp, 2]
    b0x = p_old[t0, 0] - originx
    b0y = p_old[t0, 1] - originy
    b0z = p_old[t0, 2] - originz
    b1x = p_old[t1, 0] - originx
    b1y = p_old[t1, 1] - originy
    b1z = p_old[t1, 2] - originz
    b2x = p_old[t2, 0] - originx
    b2y = p_old[t2, 1] - originy
    b2z = p_old[t2, 2] - originz
    dax = p_next[pp, 0] - originx
    day = p_next[pp, 1] - originy
    daz = p_next[pp, 2] - originz
    db0x = p_next[t0, 0] - p_old[t0, 0]
    db0y = p_next[t0, 1] - p_old[t0, 1]
    db0z = p_next[t0, 2] - p_old[t0, 2]
    db1x = p_next[t1, 0] - p_old[t1, 0]
    db1y = p_next[t1, 1] - p_old[t1, 1]
    db1z = p_next[t1, 2] - p_old[t1, 2]
    db2x = p_next[t2, 0] - p_old[t2, 0]
    db2y = p_next[t2, 1] - p_old[t2, 1]
    db2z = p_next[t2, 2] - p_old[t2, 2]
    meanx = (dax + db0x + db1x + db2x) * 0.25
    meany = (day + db0y + db1y + db2y) * 0.25
    meanz = (daz + db0z + db1z + db2z) * 0.25
    dax = dax - meanx
    day = day - meany
    daz = daz - meanz
    db0x = db0x - meanx
    db0y = db0y - meany
    db0z = db0z - meanz
    db1x = db1x - meanx
    db1y = db1y - meany
    db1z = db1z - meanz
    db2x = db2x - meanx
    db2y = db2y - meany
    db2z = db2z - meanz
    offset = thickness * SELF_CONTACT_DETECTION_MARGIN
    theirs = dmath.fmax2(dmath.fmax2(dmath.length3(db0x, db0y, db0z),
                                     dmath.length3(db1x, db1y, db1z)),
                         dmath.length3(db2x, db2y, db2z))
    travel = dmath.length3(dax, day, daz) + theirs
    moment = float(0.0)
    ahead = float(0.0)
    target = float(0.0)
    safe = float(0.0)
    resting = wp.bool(False)
    settled = wp.bool(False)
    u = float(1.0)
    v = float(0.0)
    w = float(0.0)
    framex = float(0.0)
    framey = float(0.0)
    framez = float(0.0)
    for walk in range(ACCD_STEP_LIMIT):
        qx = dax * ahead
        qy = day * ahead
        qz = daz * ahead
        cpx, cpy, cpz, bary_u, bary_v, bary_w = dmath.closest_pt_point_triangle(
            qx, qy, qz,
            b0x + db0x * ahead, b0y + db0y * ahead, b0z + db0z * ahead,
            b1x + db1x * ahead, b1y + db1y * ahead, b1z + db1z * ahead,
            b2x + db2x * ahead, b2y + db2y * ahead, b2z + db2z * ahead)
        reach = dmath.length3(qx - cpx, qy - cpy, qz - cpz) - offset
        if walk == 0:
            u = bary_u
            v = bary_v
            w = bary_w
            framex = qx - cpx
            framey = qy - cpy
            framez = qz - cpz
            if reach <= 0.0:
                resting = True
                break
            if reach >= travel:
                settled = True
                break
            target = ACCD_SEPARATION_SCALE * reach
            ahead = (1.0 - ACCD_SEPARATION_SCALE) * reach / travel
            if ahead > 1.0:
                ahead = 1.0
        elif moment > 0.0 and reach < target:
            break
        else:
            u = bary_u
            v = bary_v
            w = bary_w
            framex = qx - cpx
            framey = qy - cpy
            framez = qz - cpz
            safe = ahead
            moment = ahead
            if ahead >= 1.0:
                settled = True
                break
            ahead = moment + ACCD_ADVANCE_SCALE * reach / travel
            if ahead > 1.0:
                ahead = 1.0
    tnx, tny, tnz = dmath.triangle_normal(
        b0x + db0x * safe, b0y + db0y * safe, b0z + db0z * safe,
        b1x + db1x * safe, b1y + db1y * safe, b1z + db1z * safe,
        b2x + db2x * safe, b2y + db2y * safe, b2z + db2z * safe)
    if framex * tnx + framey * tny + framez * tnz < 0.0:
        tnx = dmath.negate(tnx)
        tny = dmath.negate(tny)
        tnz = dmath.negate(tnz)
    interior = u > 0.0 and v > 0.0 and w > 0.0
    return (interior and (resting or not settled)), tnx, tny, tnz, u, v, w


@wp.func
def do_angle_limit(v: int, p: int, vt: int, c_inv: float, p_inv: float, p_move: bool,
                   p_next_positions: wp.array2d(dtype=float),
                   p_velocity_positions: wp.array2d(dtype=float),
                   p_albuf_rotation: wp.array2d(dtype=float),
                   p_albuf_local_pos: wp.array2d(dtype=float),
                   p_albuf_local_rot: wp.array2d(dtype=float),
                   p_albuf_length: wp.array(dtype=float),
                   p_depth: wp.array(dtype=float),
                   t_angle_limit_lut: wp.array2d(dtype=float),
                   t_angle_limit_stiffness: wp.array(dtype=float)):
    prx = p_albuf_rotation[p, 0]
    pry = p_albuf_rotation[p, 1]
    prz = p_albuf_rotation[p, 2]
    prw = p_albuf_rotation[p, 3]
    lpx = p_albuf_local_pos[v, 0]
    lpy = p_albuf_local_pos[v, 1]
    lpz = p_albuf_local_pos[v, 2]
    lrx = p_albuf_local_rot[v, 0]
    lry = p_albuf_local_rot[v, 1]
    lrz = p_albuf_local_rot[v, 2]
    lrw = p_albuf_local_rot[v, 3]
    cpx = p_next_positions[v, 0]
    cpy = p_next_positions[v, 1]
    cpz = p_next_positions[v, 2]
    ppx = p_next_positions[p, 0]
    ppy = p_next_positions[p, 1]
    ppz = p_next_positions[p, 2]
    vvx = cpx - ppx
    vvy = cpy - ppy
    vvz = cpz - ppz
    vlen = dmath.length3(vvx, vvy, vvz)
    skip1 = vlen < EPSILON
    tvx, tvy, tvz = dmath.quat_rotate(prx, pry, prz, prw, lpx, lpy, lpz)
    tvlen = dmath.length3(tvx, tvy, tvz)
    snap = (not skip1) and (tvlen < EPSILON)
    if snap:
        p_velocity_positions[v, 0] = p_velocity_positions[v, 0] + (ppx - cpx)
        p_velocity_positions[v, 1] = p_velocity_positions[v, 1] + (ppy - cpy)
        p_velocity_positions[v, 2] = p_velocity_positions[v, 2] + (ppz - cpz)
        p_next_positions[v, 0] = ppx
        p_next_positions[v, 1] = ppy
        p_next_positions[v, 2] = ppz
        sqx, sqy, sqz, sqw = dmath.quat_mul(prx, pry, prz, prw, lrx, lry, lrz, lrw)
        p_albuf_rotation[v, 0] = sqx
        p_albuf_rotation[v, 1] = sqy
        p_albuf_rotation[v, 2] = sqz
        p_albuf_rotation[v, 3] = sqw
        cpx = ppx
        cpy = ppy
        cpz = ppz
    work = (not skip1) and (not snap)
    safe_vlen = vlen if vlen > 1.0e-30 else 1.0
    uvx = vvx / safe_vlen
    uvy = vvy / safe_vlen
    uvz = vvz / safe_vlen
    safe_tvlen = tvlen if tvlen > 1.0e-30 else 1.0
    utvx = tvx / safe_tvlen
    utvy = tvy / safe_tvlen
    utvz = tvz / safe_tvlen
    blen = p_albuf_length[v]
    vlen2 = dmath.lerp(vlen, blen, 0.5)
    work = work and (blen >= EPSILON) and (vlen2 >= EPSILON)
    vsx = uvx * vlen2
    vsy = uvy * vlen2
    vsz = uvz * vlen2
    ang = dmath.angle_between(vsx, vsy, vsz, utvx, utvy, utvz)
    max_angle = DEG2RAD * dmath.evaluate_team_lut(t_angle_limit_lut, vt, p_depth[v])
    over = ang > max_angle
    recovery = dmath.lerp(ang, max_angle, t_angle_limit_stiffness[vt])
    clx, cly, clz = dmath.clamp_angle_vector(vsx, vsy, vsz, utvx, utvy, utvz, recovery)
    if over and work:
        rvx = clx
        rvy = cly
        rvz = clz
    else:
        rvx = vsx
        rvy = vsy
        rvz = vsz
    rpx = ppx + vsx * ANGLE_LIMIT_ROT_RATIO
    rpy = ppy + vsy * ANGLE_LIMIT_ROT_RATIO
    rpz = ppz + vsz * ANGLE_LIMIT_ROT_RATIO
    pfx = rpx - rvx * ANGLE_LIMIT_ROT_RATIO
    pfy = rpy - rvy * ANGLE_LIMIT_ROT_RATIO
    pfz = rpz - rvz * ANGLE_LIMIT_ROT_RATIO
    cfx = rpx + rvx * (1.0 - ANGLE_LIMIT_ROT_RATIO)
    cfy = rpy + rvy * (1.0 - ANGLE_LIMIT_ROT_RATIO)
    cfz = rpz + rvz * (1.0 - ANGLE_LIMIT_ROT_RATIO)
    if work:
        paddx = (pfx - ppx) * p_inv
        paddy = (pfy - ppy) * p_inv
        paddz = (pfz - ppz) * p_inv
        caddx = (cfx - cpx) * c_inv
        caddy = (cfy - cpy) * c_inv
        caddz = (cfz - cpz) * c_inv
    else:
        paddx = 0.0
        paddy = 0.0
        paddz = 0.0
        caddx = 0.0
        caddy = 0.0
        caddz = 0.0
    cpx = cpx + caddx
    cpy = cpy + caddy
    cpz = cpz + caddz
    p_next_positions[v, 0] = cpx
    p_next_positions[v, 1] = cpy
    p_next_positions[v, 2] = cpz
    p_velocity_positions[v, 0] = p_velocity_positions[v, 0] + caddx * ANGLE_LIMIT_ATTENUATION
    p_velocity_positions[v, 1] = p_velocity_positions[v, 1] + caddy * ANGLE_LIMIT_ATTENUATION
    p_velocity_positions[v, 2] = p_velocity_positions[v, 2] + caddz * ANGLE_LIMIT_ATTENUATION
    if work and p_move:
        ppx = ppx + paddx
        ppy = ppy + paddy
        ppz = ppz + paddz
        p_next_positions[p, 0] = ppx
        p_next_positions[p, 1] = ppy
        p_next_positions[p, 2] = ppz
        p_velocity_positions[p, 0] = p_velocity_positions[p, 0] \
            + paddx * ANGLE_LIMIT_ATTENUATION
        p_velocity_positions[p, 1] = p_velocity_positions[p, 1] \
            + paddy * ANGLE_LIMIT_ATTENUATION
        p_velocity_positions[p, 2] = p_velocity_positions[p, 2] \
            + paddz * ANGLE_LIMIT_ATTENUATION
    v3x = cpx - ppx
    v3y = cpy - ppy
    v3z = cpz - ppz
    vlen3 = dmath.length3(v3x, v3y, v3z)
    fix_ok = work and (vlen3 >= EPSILON)
    safe_v3 = vlen3 if vlen3 > 1.0e-30 else 1.0
    uv3x = v3x / safe_v3
    uv3y = v3y / safe_v3
    uv3z = v3z / safe_v3
    nrx, nry, nrz, nrw = dmath.quat_mul(prx, pry, prz, prw, lrx, lry, lrz, lrw)
    qx, qy, qz, qw = dmath.from_to_rotation(utvx, utvy, utvz, uv3x, uv3y, uv3z, 1.0, True)
    frx, fry, frz, frw = dmath.quat_mul(qx, qy, qz, qw, nrx, nry, nrz, nrw)
    if fix_ok:
        p_albuf_rotation[v, 0] = frx
        p_albuf_rotation[v, 1] = fry
        p_albuf_rotation[v, 2] = frz
        p_albuf_rotation[v, 3] = frw


@wp.func
def do_angle_restoration(v: int, p: int, vt: int, c_inv: float, p_inv: float, p_move: bool,
                         rot_ratio: float, power3: float,
                         p_next_positions: wp.array2d(dtype=float),
                         p_velocity_positions: wp.array2d(dtype=float),
                         p_albuf_restore: wp.array2d(dtype=float),
                         p_depth: wp.array(dtype=float),
                         t_angle_restoration_lut: wp.array2d(dtype=float),
                         t_angle_restoration_attenuation: wp.array(dtype=float),
                         t_angle_restoration_gravity_falloff: wp.array(dtype=float),
                         t_gravity_dot: wp.array(dtype=float)):
    stiff = dmath.evaluate_team_lut_clamp01(t_angle_restoration_lut, vt, p_depth[v])
    stiff = dmath.saturate(stiff * power3)
    gfo = dmath.lerp(1.0 - t_angle_restoration_gravity_falloff[vt], 1.0, t_gravity_dot[vt])
    stiff = stiff * gfo
    r_attn = t_angle_restoration_attenuation[vt]
    cpx = p_next_positions[v, 0]
    cpy = p_next_positions[v, 1]
    cpz = p_next_positions[v, 2]
    ppx = p_next_positions[p, 0]
    ppy = p_next_positions[p, 1]
    ppz = p_next_positions[p, 2]
    tvx = p_albuf_restore[v, 0]
    tvy = p_albuf_restore[v, 1]
    tvz = p_albuf_restore[v, 2]
    tvlen = dmath.length3(tvx, tvy, tvz)
    snap = tvlen < EPSILON
    if snap:
        p_velocity_positions[v, 0] = p_velocity_positions[v, 0] + (ppx - cpx)
        p_velocity_positions[v, 1] = p_velocity_positions[v, 1] + (ppy - cpy)
        p_velocity_positions[v, 2] = p_velocity_positions[v, 2] + (ppz - cpz)
        p_next_positions[v, 0] = ppx
        p_next_positions[v, 1] = ppy
        p_next_positions[v, 2] = ppz
        cpx = ppx
        cpy = ppy
        cpz = ppz
    vvx = cpx - ppx
    vvy = cpy - ppy
    vvz = cpz - ppz
    vlen = dmath.length3(vvx, vvy, vvz)
    work = (not snap) and (vlen >= EPSILON)
    safe_vlen = vlen if vlen > 1.0e-30 else 1.0
    uvx = vvx / safe_vlen
    uvy = vvy / safe_vlen
    uvz = vvz / safe_vlen
    safe_tvlen = tvlen if tvlen > 1.0e-30 else 1.0
    utvx = tvx / safe_tvlen
    utvy = tvy / safe_tvlen
    utvz = tvz / safe_tvlen
    rqx, rqy, rqz, rqw = dmath.from_to_rotation(uvx, uvy, uvz, utvx, utvy, utvz, stiff, True)
    rvx, rvy, rvz = dmath.quat_rotate(rqx, rqy, rqz, rqw, vvx, vvy, vvz)
    rpx = ppx + vvx * rot_ratio
    rpy = ppy + vvy * rot_ratio
    rpz = ppz + vvz * rot_ratio
    pfx = rpx - rvx * rot_ratio
    pfy = rpy - rvy * rot_ratio
    pfz = rpz - rvz * rot_ratio
    cfx = rpx + rvx * (1.0 - rot_ratio)
    cfy = rpy + rvy * (1.0 - rot_ratio)
    cfz = rpz + rvz * (1.0 - rot_ratio)
    if work:
        paddx = (pfx - ppx) * p_inv
        paddy = (pfy - ppy) * p_inv
        paddz = (pfz - ppz) * p_inv
        caddx = (cfx - cpx) * c_inv
        caddy = (cfy - cpy) * c_inv
        caddz = (cfz - cpz) * c_inv
    else:
        paddx = 0.0
        paddy = 0.0
        paddz = 0.0
        caddx = 0.0
        caddy = 0.0
        caddz = 0.0
    p_next_positions[v, 0] = cpx + caddx
    p_next_positions[v, 1] = cpy + caddy
    p_next_positions[v, 2] = cpz + caddz
    p_velocity_positions[v, 0] = p_velocity_positions[v, 0] + caddx * r_attn
    p_velocity_positions[v, 1] = p_velocity_positions[v, 1] + caddy * r_attn
    p_velocity_positions[v, 2] = p_velocity_positions[v, 2] + caddz * r_attn
    if work and p_move:
        p_next_positions[p, 0] = ppx + paddx
        p_next_positions[p, 1] = ppy + paddy
        p_next_positions[p, 2] = ppz + paddz
        p_velocity_positions[p, 0] = p_velocity_positions[p, 0] + paddx * r_attn
        p_velocity_positions[p, 1] = p_velocity_positions[p, 1] + paddy * r_attn
        p_velocity_positions[p, 2] = p_velocity_positions[p, 2] + paddz * r_attn


COLLIDER_BOUND_EXIT_REASON = (
    "the root search needs an end it is certain is outside the body, and while the sample "
    "is inside, the field's own gradient names the next place to look; when that gradient "
    "stops pointing along the push out direction the field has nothing left to say, and "
    "the statement that a ray which does not climb the field never leaves is true of a "
    "half space and false of everything bounded, which is what the first version of this "
    "search got wrong: it refused to push a body whose gradient happened to point aside, "
    "and on a rig of a hundred and ninety two overlapping bodies that left eighteen "
    "particles inside where two had been, and the worst output penetration went from a "
    "quarter of a micrometre to thirty eight millimetres, caught by the non penetration "
    "invariant of the total gate; the certain end therefore comes from the one bound every "
    "shape already publishes, where the ray leaves the body's axis aligned box, which is "
    "the smallest of the three axis crossings and not the largest, the largest being the "
    "arithmetic that measured a particle at fifty six million metres; a plane publishes an "
    "unbounded box, so it reports no exit, which is the same statement the plane branch "
    "used to make on its own and is now made by data")


@wp.func
def collider_ray_leaves_bound(c: int, px: float, py: float, pz: float,
                              dx: float, dy: float, dz: float,
                              c_work_aabb_min: wp.array2d(dtype=float),
                              c_work_aabb_max: wp.array2d(dtype=float)):
    limit = float(wp.inf)
    if dx != 0.0:
        reach = dmath.fmax2((c_work_aabb_min[c, 0] - px) / dx,
                            (c_work_aabb_max[c, 0] - px) / dx)
        if reach < limit:
            limit = reach
    if dy != 0.0:
        reach = dmath.fmax2((c_work_aabb_min[c, 1] - py) / dy,
                            (c_work_aabb_max[c, 1] - py) / dy)
        if reach < limit:
            limit = reach
    if dz != 0.0:
        reach = dmath.fmax2((c_work_aabb_min[c, 2] - pz) / dz,
                            (c_work_aabb_max[c, 2] - pz) / dz)
        if reach < limit:
            limit = reach
    return limit


COLLIDER_EXIT_SEARCH_REASON = (
    "the push out is a root of the field along one direction, and it is searched with one "
    "loop that names the field once, because warp inlines every naming of it and a mesh "
    "closest point query costs seventy seven registers a copy; while the sample is inside, "
    "the depth divided by how fast the field climbs along the direction names a distance "
    "that is at least as far as the surface, which is a step no smaller than the surface "
    "and so a walk that either lands outside or shows the ray never leaves, and a direction "
    "that does not climb the field cannot leave the body along it, which used to be written "
    "out for a plane alone and is the same statement for every shape; once a sample has "
    "read as outside the loop halves between the last inside sample and it, so the same "
    "iteration is a conservative walk while it needs one and a bisection afterwards; the "
    "walk may never take a step shorter than the resolution it measures the field with, "
    "because the step the gradient asks for goes to zero as the walk lands on the surface "
    "and the field there reads as a few float32 places of noise on either side of nothing, "
    "so a walk that honoured that step would stand on the surface spending its whole budget "
    "and then report that the body cannot be left at all, measured as a hundred points of "
    "eleven hundred inside a box left where they were; that floor is scaled by the "
    "distance walked so far as well as by the place being sampled, because it is added "
    "to the distance walked and a floor below that number's own last float32 place cannot "
    "move it: a plane through the origin sampled at a tenth of a metre out and three "
    "tenths deep gives a sample magnitude smaller than the walk, the floor lands under "
    "the walk's own resolution, the walk stands still for its whole budget and reports "
    "no exit, and thirty six points of fifteen hundred were left inside a half space "
    "by exactly that, measured by colliderfield.py; what this replaced derived the "
    "bracket a second time out of each shape's own geometry, and the second derivation "
    "disagreed with the field it was bracketing, taking the largest of the three axis "
    "crossings of a box instead of the smallest, so a push out that ran along an axis "
    "divided by a component near zero and handed the search a bracket the size of one over "
    "that component, and the answer came back at fifty six million metres")

COLLIDER_EXIT_SEARCH_STEPS = wp.constant(int(_defs.COLLIDER_EXIT_SEARCH_STEPS))

COLLIDER_EXIT_REACH_REASON = (
    "the exit projection is the one caller that needs the field from inside a body, so the "
    "distance it names is how deep a point is allowed to be, and the only bound on that is "
    "the diagonal of the body's own bound: the surface is inside that bound and so is any "
    "point that could be inside the body, and nothing inside a box is further from anything "
    "else inside it than that box's diagonal, which is the reach a mesh already declares; "
    "half the shortest side of the same bound looks tighter and was tried, and it is not a "
    "bound at all, because the argument for it, that a body holds no ball wider than the "
    "narrowest way through its own bound, is an argument about a solid and a game LOD is a "
    "surface: where the surface does not close, the side of it the pseudo normal calls in "
    "reaches past the narrow way; measured on the frozen asset with stage3_fieldprobe.py, "
    "particle 29 stands 116.1 mm from the nearest triangle of a body whose shortest side is "
    "228.9 mm, and an exit projection asking only 114.4 mm read it as outside and left it "
    "where it was on nine of the forty frames lodcollider.py watches, which is the whole of "
    "what that criterion is for; what is bounded here instead is which bodies get asked, "
    "because a point outside the bound a body swept over the step is outside that body, "
    "and that is a proof and not a guess")


@wp.func
def do_collider_exit_root(c: int, px: float, py: float, pz: float,
                          dx: float, dy: float, dz: float, low: float,
                          incidence_gate_cos: float,
                          c_kind: wp.array(dtype=int),
                          c_work_next_pos: wp.array3d(dtype=float),
                          c_work_radius: wp.array2d(dtype=float),
                          c_work_rot: wp.array2d(dtype=float),
                          c_work_inv_rot: wp.array2d(dtype=float),
                          face_index: wp.uint64,
                          cf_vertex: wp.array2d(dtype=int),
                          cf_edge_normal: wp.array3d(dtype=float),
                          cf_normal: wp.array2d(dtype=float),
                          cv_local_position: wp.array2d(dtype=float),
                          cv_pseudo_normal: wp.array2d(dtype=float),
                          c_work_aabb_min: wp.array2d(dtype=float),
                          c_work_aabb_max: wp.array2d(dtype=float)):
    cap = collider_ray_leaves_bound(c, px, py, pz, dx, dy, dz, c_work_aabb_min,
                                    c_work_aabb_max)
    reach = c_work_radius[c, 0]
    inside = float(low)
    outside = float(wp.inf)
    travel = float(low)
    found = wp.bool(False)
    for _step in range(COLLIDER_EXIT_SEARCH_STEPS):
        mx = px + dx * travel
        my = py + dy * travel
        mz = pz + dz * travel
        field, nx, ny, nz = collider_field(c, mx, my, mz, reach, incidence_gate_cos,
                                           c_kind,
                                           c_work_next_pos, c_work_radius, c_work_rot,
                                           c_work_inv_rot, face_index, cf_vertex,
                                           cf_edge_normal, cf_normal,
                                           cv_local_position, cv_pseudo_normal)
        resolution = COLLIDER_EXIT_TOLERANCE * (wp.abs(field) + wp.abs(mx) + wp.abs(my)
                                                + wp.abs(mz) + travel)
        if field >= resolution:
            outside = travel
            found = True
        else:
            inside = travel
        if found:
            travel = 0.5 * (inside + outside)
        else:
            along = dx * nx + dy * ny + dz * nz
            ahead = float(wp.inf)
            if along > 0.0:
                ahead = travel + dmath.fmax2(dmath.negate(field) / along, resolution)
            if ahead < cap:
                travel = ahead
            elif cap >= wp.inf:
                return float(wp.inf)
            else:
                outside = cap
                found = True
                travel = 0.5 * (inside + outside)
    return outside


@wp.func
def do_collider_exit(p: int, mt: int, px: float, py: float, pz: float,
                     t_collision_mode: wp.array(dtype=int),
                     c_team: wp.array(dtype=int),
                     c_kind: wp.array(dtype=int),
                     c_active: wp.array(dtype=int),
                     c_work_next_pos: wp.array3d(dtype=float),
                     c_work_radius: wp.array2d(dtype=float),
                     c_work_rot: wp.array2d(dtype=float),
                     c_work_inv_rot: wp.array2d(dtype=float),
                     face_index: wp.uint64,
                     cf_vertex: wp.array2d(dtype=int),
                     cf_edge_normal: wp.array3d(dtype=float),
                     cf_normal: wp.array2d(dtype=float),
                     cv_local_position: wp.array2d(dtype=float),
                     cv_pseudo_normal: wp.array2d(dtype=float),
                     c_work_aabb_min: wp.array2d(dtype=float),
                     c_work_aabb_max: wp.array2d(dtype=float),
                     csr_off: wp.array(dtype=int),
                     csr_ord: wp.array(dtype=int),
                     st_pp_collider: wp.array(dtype=int),
                     p_intersect_flag: wp.array(dtype=int),
                     incidence_gate_cos: float, honor_intersect_freeze: int):
    mode = t_collision_mode[mt]
    if mode != COLLISION_POINT and mode != COLLISION_EDGE:
        return px, py, pz
    if honor_intersect_freeze != 0 and p_intersect_flag[p] != 0:
        return px, py, pz
    point_mode = mode == COLLISION_POINT
    start = csr_off[p]
    span = c_team.shape[0]
    if point_mode:
        span = csr_off[p + 1] - start
    deepest = float(wp.inf)
    dx = float(0.0)
    dy = float(0.0)
    dz = float(0.0)
    bodies = int(0)
    for k in range(span):
        c = k
        if point_mode:
            c = st_pp_collider[csr_ord[start + k]]
        elif c_team[c] != mt:
            continue
        if c_active[c] == 0:
            continue
        bodies += 1
        if collider_bound_misses_point(c, px, py, pz, 0.0, c_work_aabb_min,
                                       c_work_aabb_max):
            continue
        reach = c_work_radius[c, 0]
        field, ux, uy, uz = collider_field(c, px, py, pz, reach, incidence_gate_cos,
                                           c_kind,
                                           c_work_next_pos, c_work_radius, c_work_rot,
                                           c_work_inv_rot, face_index, cf_vertex,
                                           cf_edge_normal, cf_normal,
                                           cv_local_position, cv_pseudo_normal)
        if field < deepest:
            deepest = field
            dx = ux
            dy = uy
            dz = uz
    if deepest >= 0.0:
        return px, py, pz
    travel = float(0.0)
    for _merge in range(bodies + 1):
        advance = float(travel)
        for k in range(span):
            c = k
            if point_mode:
                c = st_pp_collider[csr_ord[start + k]]
            elif c_team[c] != mt:
                continue
            if c_active[c] == 0:
                continue
            root = do_collider_exit_root(
                c, px, py, pz, dx, dy, dz, travel, incidence_gate_cos, c_kind,
                c_work_next_pos, c_work_radius, c_work_rot, c_work_inv_rot,
                face_index, cf_vertex, cf_edge_normal, cf_normal,
                cv_local_position, cv_pseudo_normal, c_work_aabb_min,
                c_work_aabb_max)
            if root < wp.inf and root > advance:
                advance = root
        if advance <= travel:
            break
        travel = advance
    return px + dx * travel, py + dy * travel, pz + dz * travel


@wp.func
def do_display_particle(p: int, mt: int, sim_dt: float,
                        p_positions: wp.array2d(dtype=float),
                        p_rotations: wp.array2d(dtype=float),
                        p_old_positions: wp.array2d(dtype=float),
                        p_real_velocities: wp.array2d(dtype=float),
                        p_display_positions: wp.array2d(dtype=float),
                        p_vertex_root: wp.array(dtype=int),
                        p_old_anim_positions: wp.array2d(dtype=float),
                        p_old_anim_rotations: wp.array2d(dtype=float),
                        p_temp_base_positions: wp.array2d(dtype=float),
                        p_temp_base_rotations: wp.array2d(dtype=float),
                        st_update_move_mask: wp.array(dtype=int),
                        t_now_update: wp.array(dtype=float),
                        t_old_time: wp.array(dtype=float),
                        t_time: wp.array(dtype=float),
                        t_blend_weight: wp.array(dtype=float),
                        t_running: wp.array(dtype=int),
                        t_reflected: wp.array(dtype=int)):
    snap_px = p_positions[p, 0]
    snap_py = p_positions[p, 1]
    snap_pz = p_positions[p, 2]
    snap_rx = p_rotations[p, 0]
    snap_ry = p_rotations[p, 1]
    snap_rz = p_rotations[p, 2]
    snap_rw = p_rotations[p, 3]
    fx = snap_px
    fy = snap_py
    fz = snap_pz
    if st_update_move_mask[p] != 0:
        sdt = sim_dt
        fposx = p_old_positions[p, 0] + p_real_velocities[p, 0] * sdt
        fposy = p_old_positions[p, 1] + p_real_velocities[p, 1] * sdt
        fposz = p_old_positions[p, 2] + p_real_velocities[p, 2] * sdt
        interval = (t_now_update[mt] + sdt) - t_old_time[mt]
        if interval > 0.0:
            tval = (t_time[mt] - t_old_time[mt]) / interval
        else:
            tval = 0.0
        fposx = dmath.lerp(p_display_positions[p, 0], fposx, tval)
        fposy = dmath.lerp(p_display_positions[p, 1], fposy, tval)
        fposz = dmath.lerp(p_display_positions[p, 2], fposz, tval)
        root = p_vertex_root[p]
        if root >= 0:
            rpx = p_positions[root, 0]
            rpy = p_positions[root, 1]
            rpz = p_positions[root, 2]
            original_dist = dmath.length3(rpx - snap_px, rpy - snap_py, rpz - snap_pz)
            clamp_dist = original_dist * MAX_DISTANCE_RATIO_FUTURE_PREDICTION
            vx, vy, vz = dmath.clamp_vector(fposx - rpx, fposy - rpy, fposz - rpz, clamp_dist)
            fposx = rpx + vx
            fposy = rpy + vy
            fposz = rpz + vz
        p_display_positions[p, 0] = fposx
        p_display_positions[p, 1] = fposy
        p_display_positions[p, 2] = fposz
        blend = t_blend_weight[mt]
        fx = dmath.lerp(snap_px, fposx, blend)
        fy = dmath.lerp(snap_py, fposy, blend)
        fz = dmath.lerp(snap_pz, fposz, blend)
        p_positions[p, 0] = fx
        p_positions[p, 1] = fy
        p_positions[p, 2] = fz
    else:
        p_display_positions[p, 0] = snap_px
        p_display_positions[p, 1] = snap_py
        p_display_positions[p, 2] = snap_pz
    if t_running[mt] != 0:
        p_old_anim_positions[p, 0] = snap_px
        p_old_anim_positions[p, 1] = snap_py
        p_old_anim_positions[p, 2] = snap_pz
        p_old_anim_rotations[p, 0] = snap_rx
        p_old_anim_rotations[p, 1] = snap_ry
        p_old_anim_rotations[p, 2] = snap_rz
        p_old_anim_rotations[p, 3] = snap_rw
    qx = snap_rx
    qy = snap_ry
    qz = snap_rz
    qw = snap_rw
    if t_reflected[mt] != 0:
        reflection_sign = component_reflection_sign(t_reflected, mt)
        nnx, nny, nnz = dmath.quat_to_normal(snap_rx, snap_ry, snap_rz, snap_rw)
        nnx = nnx * reflection_sign
        nny = nny * reflection_sign
        nnz = nnz * reflection_sign
        ttx, tty, ttz = dmath.quat_to_tangent(snap_rx, snap_ry, snap_rz, snap_rw)
        ttx = ttx * reflection_sign
        tty = tty * reflection_sign
        ttz = ttz * reflection_sign
        qx, qy, qz, qw = dmath.look_rotation(ttx, tty, ttz, nnx, nny, nnz)
        p_rotations[p, 0] = qx
        p_rotations[p, 1] = qy
        p_rotations[p, 2] = qz
        p_rotations[p, 3] = qw
    p_temp_base_positions[p, 0] = snap_px
    p_temp_base_positions[p, 1] = snap_py
    p_temp_base_positions[p, 2] = snap_pz
    p_temp_base_rotations[p, 0] = qx
    p_temp_base_rotations[p, 1] = qy
    p_temp_base_rotations[p, 2] = qz
    p_temp_base_rotations[p, 3] = qw


@wp.func
def do_postline_entry(entry: int, et: int, ch_start: int, ch_end: int,
                      postline_child_vertices: wp.array(dtype=int),
                      p_positions: wp.array2d(dtype=float),
                      p_rotations: wp.array2d(dtype=float),
                      p_temp_base_positions: wp.array2d(dtype=float),
                      p_temp_base_rotations: wp.array2d(dtype=float),
                      p_vertex_local_positions: wp.array2d(dtype=float),
                      p_vertex_local_rotations: wp.array2d(dtype=float),
                      p_attr_invalid: wp.array(dtype=int),
                      p_attr_zero_distance: wp.array(dtype=int),
                      p_attr_move: wp.array(dtype=int),
                      p_team: wp.array(dtype=int),
                      t_rotational_interpolation: wp.array(dtype=float),
                      t_root_rotation: wp.array(dtype=float),
                      t_blend_weight: wp.array(dtype=float),
                      t_animation_pose_ratio: wp.array(dtype=float),
                      t_reflected: wp.array(dtype=int)):
    rx = p_rotations[entry, 0]
    ry = p_rotations[entry, 1]
    rz = p_rotations[entry, 2]
    rw = p_rotations[entry, 3]
    posx = p_positions[entry, 0]
    posy = p_positions[entry, 1]
    posz = p_positions[entry, 2]
    bpx = p_temp_base_positions[entry, 0]
    bpy = p_temp_base_positions[entry, 1]
    bpz = p_temp_base_positions[entry, 2]
    brx = p_temp_base_rotations[entry, 0]
    bry = p_temp_base_rotations[entry, 1]
    brz = p_temp_base_rotations[entry, 2]
    brw = p_temp_base_rotations[entry, 3]
    bix, biy, biz, biw = dmath.quat_inverse(brx, bry, brz, brw)
    owner_valid = p_attr_invalid[entry] == 0
    ctvx = wp.float64(0.0)
    ctvy = wp.float64(0.0)
    ctvz = wp.float64(0.0)
    cvx = wp.float64(0.0)
    cvy = wp.float64(0.0)
    cvz = wp.float64(0.0)
    has_children = ch_end > ch_start
    for k in range(ch_start, ch_end):
        c = postline_child_vertices[k]
        ct = p_team[c]
        anime_ratio = t_animation_pose_ratio[ct]
        reflection_sign = component_reflection_sign(t_reflected, ct)
        clpx, clpy, clpz = dmath.quat_rotate(
            bix, biy, biz, biw, p_temp_base_positions[c, 0] - bpx,
            p_temp_base_positions[c, 1] - bpy, p_temp_base_positions[c, 2] - bpz)
        clrx, clry, clrz, clrw = dmath.quat_mul(
            bix, biy, biz, biw, p_temp_base_rotations[c, 0], p_temp_base_rotations[c, 1],
            p_temp_base_rotations[c, 2], p_temp_base_rotations[c, 3])
        lposx = dmath.lerp(p_vertex_local_positions[c, 0] * reflection_sign, clpx, anime_ratio)
        lposy = dmath.lerp(p_vertex_local_positions[c, 1] * reflection_sign, clpy, anime_ratio)
        lposz = dmath.lerp(p_vertex_local_positions[c, 2] * reflection_sign, clpz, anime_ratio)
        lrx, lry, lrz, lrw = dmath.quat_slerp(
            p_vertex_local_rotations[c, 0], p_vertex_local_rotations[c, 1],
            p_vertex_local_rotations[c, 2], p_vertex_local_rotations[c, 3],
            clrx, clry, clrz, clrw, anime_ratio)
        is_c0 = p_attr_zero_distance[c] != 0
        if is_c0:
            tvx = 0.0
            tvy = 0.0
            tvz = 0.0
        else:
            tvx, tvy, tvz = dmath.quat_rotate(rx, ry, rz, rw, lposx, lposy, lposz)
        c_move = p_attr_move[c] != 0
        vx = p_positions[c, 0] - posx
        vy = p_positions[c, 1] - posy
        vz = p_positions[c, 2] - posz
        if c_move:
            contx = vx
            conty = vy
            contz = vz
        else:
            contx = tvx
            conty = tvy
            contz = tvz
        if owner_valid:
            ctvx += wp.float64(tvx)
            ctvy += wp.float64(tvy)
            ctvz += wp.float64(tvz)
            cvx += wp.float64(contx)
            cvy += wp.float64(conty)
            cvz += wp.float64(contz)
            if c_move:
                crx, cry, crz, crw = dmath.quat_mul(rx, ry, rz, rw, lrx, lry, lrz, lrw)
                if not is_c0:
                    qfx, qfy, qfz, qfw = dmath.from_to_rotation(tvx, tvy, tvz, vx, vy, vz,
                                                                1.0, False)
                    crx, cry, crz, crw = dmath.quat_mul(qfx, qfy, qfz, qfw, crx, cry, crz, crw)
                p_rotations[c, 0] = crx
                p_rotations[c, 1] = cry
                p_rotations[c, 2] = crz
                p_rotations[c, 3] = crw
    if has_children and owner_valid:
        ctv32x = wp.float32(ctvx)
        ctv32y = wp.float32(ctvy)
        ctv32z = wp.float32(ctvz)
        cv32x = wp.float32(cvx)
        cv32y = wp.float32(cvy)
        cv32z = wp.float32(cvz)
        zero = (dmath.length3(ctv32x, ctv32y, ctv32z) < 1e-8) \
            or (dmath.length3(cv32x, cv32y, cv32z) < 1e-8)
        if not zero:
            if p_attr_move[entry] != 0:
                t_ratio = t_rotational_interpolation[et]
            else:
                t_ratio = t_root_rotation[et]
            cqx, cqy, cqz, cqw = dmath.from_to_rotation(ctv32x, ctv32y, ctv32z,
                                                        cv32x, cv32y, cv32z, t_ratio, False)
            rx, ry, rz, rw = dmath.quat_mul(cqx, cqy, cqz, cqw, rx, ry, rz, rw)
    rx, ry, rz, rw = dmath.quat_slerp(brx, bry, brz, brw, rx, ry, rz, rw, t_blend_weight[et])
    p_rotations[entry, 0] = rx
    p_rotations[entry, 1] = ry
    p_rotations[entry, 2] = rz
    p_rotations[entry, 3] = rw


@wp.func
def do_triangle_normal_tangent(tri: int, tt_team: int,
                               st_triangle_particles: wp.array2d(dtype=int),
                               p_positions: wp.array2d(dtype=float),
                               p_uv: wp.array2d(dtype=float),
                               t_reflected: wp.array(dtype=int),
                               tri_normal_f64: wp.array2d(dtype=wp.float64),
                               tri_tangent_f64: wp.array2d(dtype=wp.float64)):
    i0 = st_triangle_particles[tri, 0]
    i1 = st_triangle_particles[tri, 1]
    i2 = st_triangle_particles[tri, 2]
    p0x = p_positions[i0, 0]
    p0y = p_positions[i0, 1]
    p0z = p_positions[i0, 2]
    p1x = p_positions[i1, 0]
    p1y = p_positions[i1, 1]
    p1z = p_positions[i1, 2]
    p2x = p_positions[i2, 0]
    p2y = p_positions[i2, 1]
    p2z = p_positions[i2, 2]
    cx, cy, cz = dmath.cross3(p1x - p0x, p1y - p0y, p1z - p0z,
                              p2x - p0x, p2y - p0y, p2z - p0z)
    lc = dmath.length3(cx, cy, cz)
    if lc > EPSILON:
        nnx = cx / lc
        nny = cy / lc
        nnz = cz / lc
    else:
        nnx = cx
        nny = cy
        nnz = cz
    reflection_sign = component_reflection_sign(t_reflected, tt_team)
    tri_normal_f64[tri, 0] = wp.float64(nnx) * wp.float64(reflection_sign)
    tri_normal_f64[tri, 1] = wp.float64(nny) * wp.float64(reflection_sign)
    tri_normal_f64[tri, 2] = wp.float64(nnz) * wp.float64(reflection_sign)
    q0x = wp.float64(p0x)
    q0y = wp.float64(p0y)
    q0z = wp.float64(p0z)
    dbax = wp.float64(p1x) - q0x
    dbay = wp.float64(p1y) - q0y
    dbaz = wp.float64(p1z) - q0z
    dcax = wp.float64(p2x) - q0x
    dcay = wp.float64(p2y) - q0y
    dcaz = wp.float64(p2z) - q0z
    uv0x = wp.float64(p_uv[i0, 0])
    uv0y = wp.float64(p_uv[i0, 1])
    tbax = wp.float64(p_uv[i1, 0]) - uv0x
    tbay = wp.float64(p_uv[i1, 1]) - uv0y
    tcax = wp.float64(p_uv[i2, 0]) - uv0x
    tcay = wp.float64(p_uv[i2, 1]) - uv0y
    area = tbax * tcay - tbay * tcax
    if area == wp.float64(0.0):
        area = wp.float64(1.0)
    delta = wp.float64(-1.0) / area
    tanx = (dbax * tcay + dcax * dmath.negate(tbay)) * delta
    tany = (dbay * tcay + dcay * dmath.negate(tbay)) * delta
    tanz = (dbaz * tcay + dcaz * dmath.negate(tbay)) * delta
    ltan = wp.sqrt(tanx * tanx + tany * tany + tanz * tanz)
    if ltan > wp.float64(1e-30):
        tanx = tanx / ltan
        tany = tany / ltan
        tanz = tanz / ltan
    tri_tangent_f64[tri, 0] = tanx * wp.float64(reflection_sign)
    tri_tangent_f64[tri, 1] = tany * wp.float64(reflection_sign)
    tri_tangent_f64[tri, 2] = tanz * wp.float64(reflection_sign)


@wp.func
def do_v2t_owner(p: int, seg0: int, seg1: int,
                 csr_v2t_order: wp.array(dtype=int),
                 st_v2t_triangle: wp.array(dtype=int),
                 st_v2t_flip_normal: wp.array(dtype=float),
                 st_v2t_flip_tangent: wp.array(dtype=float),
                 tri_normal_f64: wp.array2d(dtype=wp.float64),
                 tri_tangent_f64: wp.array2d(dtype=wp.float64),
                 p_rotations: wp.array2d(dtype=float),
                 p_normal_adjustment_rotations: wp.array2d(dtype=float)):
    norx = wp.float64(0.0)
    nory = wp.float64(0.0)
    norz = wp.float64(0.0)
    tanx = wp.float64(0.0)
    tany = wp.float64(0.0)
    tanz = wp.float64(0.0)
    for k in range(seg0, seg1):
        row = csr_v2t_order[k]
        tri = st_v2t_triangle[row]
        fn = wp.float64(st_v2t_flip_normal[row])
        ft = wp.float64(st_v2t_flip_tangent[row])
        norx += tri_normal_f64[tri, 0] * fn
        nory += tri_normal_f64[tri, 1] * fn
        norz += tri_normal_f64[tri, 2] * fn
        tanx += tri_tangent_f64[tri, 0] * ft
        tany += tri_tangent_f64[tri, 1] * ft
        tanz += tri_tangent_f64[tri, 2] * ft
    ln = wp.sqrt(norx * norx + nory * nory + norz * norz)
    lt = wp.sqrt(tanx * tanx + tany * tany + tanz * tanz)
    ok = (ln > wp.float64(1e-6)) and (lt > wp.float64(1e-6))
    if ln > wp.float64(1e-30):
        nnx = norx / ln
        nny = nory / ln
        nnz = norz / ln
    else:
        nnx = norx
        nny = nory
        nnz = norz
    if lt > wp.float64(1e-30):
        ntx = tanx / lt
        nty = tany / lt
        ntz = tanz / lt
    else:
        ntx = tanx
        nty = tany
        ntz = tanz
    d = nnx * ntx + nny * nty + nnz * ntz
    if d == wp.float64(1.0) or d == wp.float64(-1.0):
        ok = False
    bx = nny * ntz - nnz * nty
    by = nnz * ntx - nnx * ntz
    bz = nnx * nty - nny * ntx
    bl = wp.sqrt(bx * bx + by * by + bz * bz)
    if bl > wp.float64(1e-30):
        bx = bx / bl
        by = by / bl
        bz = bz / bl
    rrx, rry, rrz, rrw = dmath.look_rotation(wp.float32(bx), wp.float32(by), wp.float32(bz),
                                             wp.float32(nnx), wp.float32(nny), wp.float32(nnz))
    frx, fry, frz, frw = dmath.quat_mul(rrx, rry, rrz, rrw,
                                        p_normal_adjustment_rotations[p, 0],
                                        p_normal_adjustment_rotations[p, 1],
                                        p_normal_adjustment_rotations[p, 2],
                                        p_normal_adjustment_rotations[p, 3])
    if ok:
        p_rotations[p, 0] = frx
        p_rotations[p, 1] = fry
        p_rotations[p, 2] = frz
        p_rotations[p, 3] = frw


@wp.func
def do_publish_bone_transform(p: int, p_publish_transform: wp.array(dtype=int),
                              p_bone_row: wp.array(dtype=int),
                              p_publish_position: wp.array(dtype=int),
                              p_aim_child: wp.array(dtype=int),
                              p_aim_rest_reach: wp.array(dtype=float),
                              p_positions: wp.array2d(dtype=float),
                              p_rotations: wp.array2d(dtype=float),
                              p_vertex_to_transform_rotations: wp.array2d(dtype=float),
                              world: wp.array3d(dtype=float),
                              solved: wp.array3d(dtype=float)):
    row = p_bone_row[p]
    if row < 0:
        return
    qx, qy, qz, qw = dmath.quat_mul(p_rotations[p, 0], p_rotations[p, 1], p_rotations[p, 2],
                                    p_rotations[p, 3],
                                    p_vertex_to_transform_rotations[p, 0],
                                    p_vertex_to_transform_rotations[p, 1],
                                    p_vertex_to_transform_rotations[p, 2],
                                    p_vertex_to_transform_rotations[p, 3])
    scale_x = wp.sqrt(world[row, 0, 0] * world[row, 0, 0] + world[row, 1, 0] * world[row, 1, 0]
                      + world[row, 2, 0] * world[row, 2, 0])
    scale_y = wp.sqrt(world[row, 0, 1] * world[row, 0, 1] + world[row, 1, 1] * world[row, 1, 1]
                      + world[row, 2, 1] * world[row, 2, 1])
    scale_z = wp.sqrt(world[row, 0, 2] * world[row, 0, 2] + world[row, 1, 2] * world[row, 1, 2]
                      + world[row, 2, 2] * world[row, 2, 2])

    child = p_aim_child[p]
    if child >= 0:
        tx = p_positions[child, 0] - p_positions[p, 0]
        ty = p_positions[child, 1] - p_positions[p, 1]
        tz = p_positions[child, 2] - p_positions[p, 2]
        reach = wp.sqrt(tx * tx + ty * ty + tz * tz)
        span = p_aim_rest_reach[p] * scale_y
        if reach > EPSILON and span > EPSILON:
            ax, ay, az = dmath.quat_rotate(qx, qy, qz, qw, 0.0, 1.0, 0.0)
            sx, sy, sz, sw = dmath.from_to_rotation(ax, ay, az, tx / reach, ty / reach,
                                                    tz / reach, 1.0, True)
            qx, qy, qz, qw = dmath.quat_mul(sx, sy, sz, sw, qx, qy, qz, qw)
            stretch = reach / span
            scale_x = scale_x * stretch
            scale_y = scale_y * stretch
            scale_z = scale_z * stretch

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = dmath.quat_to_matrix3_f32(qx, qy, qz, qw)
    px = p_positions[p, 0]
    py = p_positions[p, 1]
    pz = p_positions[p, 2]
    solved[row, 0, 0] = m00 * scale_x
    solved[row, 0, 1] = m01 * scale_y
    solved[row, 0, 2] = m02 * scale_z
    solved[row, 1, 0] = m10 * scale_x
    solved[row, 1, 1] = m11 * scale_y
    solved[row, 1, 2] = m12 * scale_z
    solved[row, 2, 0] = m20 * scale_x
    solved[row, 2, 1] = m21 * scale_y
    solved[row, 2, 2] = m22 * scale_z
    solved[row, 0, 3] = px
    solved[row, 1, 3] = py
    solved[row, 2, 3] = pz
    solved[row, 3, 0] = 0.0
    solved[row, 3, 1] = 0.0
    solved[row, 3, 2] = 0.0
    solved[row, 3, 3] = 1.0

    shared = p_publish_transform[p]
    if shared >= 0:
        for r in range(4):
            for c in range(4):
                world[shared, r, c] = solved[row, r, c]


@wp.func
def do_output_particle(p: int, p_rotations: wp.array2d(dtype=float),
                       p_vertex_to_transform_rotations: wp.array2d(dtype=float),
                       p_out_rotations: wp.array2d(dtype=float)):
    ox, oy, oz, ow = dmath.quat_mul(p_rotations[p, 0], p_rotations[p, 1], p_rotations[p, 2],
                                    p_rotations[p, 3],
                                    p_vertex_to_transform_rotations[p, 0],
                                    p_vertex_to_transform_rotations[p, 1],
                                    p_vertex_to_transform_rotations[p, 2],
                                    p_vertex_to_transform_rotations[p, 3])
    p_out_rotations[p, 0] = ox
    p_out_rotations[p, 1] = oy
    p_out_rotations[p, 2] = oz
    p_out_rotations[p, 3] = ow
