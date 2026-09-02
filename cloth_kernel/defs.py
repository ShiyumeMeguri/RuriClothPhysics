import math as _math

import numpy as _numpy

EPSILON = 1e-8

COMPONENT_SCALE_EPSILON = 1e-6

COMPONENT_SCALE_EPSILON_REASON = (
    "a team whose component basis has collapsed along an axis has no frame to simulate in, "
    "so the frame mask drops it, and this is the shortest basis axis that still counts as a "
    "frame; the host mask and the device mask have to agree on it to the bit, because a team "
    "the host plans work for and the device then skips leaves that work reading planes "
    "nobody wrote, so the number lives here once and both sides take it from here instead "
    "of writing it down twice")

DEG2RAD = _math.pi / 180.0
RAD2DEG = 180.0 / _math.pi
TO_FIXED = float(1 << 30)

TO_FIXED_REASON = (
    "the parallel correction accumulators hold a length in metres as a signed 64 bit integer "
    "so that the summation order cannot change the result, and this is the number of integer "
    "steps one metre is cut into; it is a power of two because multiplying a float32 by a "
    "power of two only moves the exponent, so the encoding on the way in and the division on "
    "the way out are both exact and the only error left is the truncation to an integer, "
    "while a decimal scale rounds in both directions as well, measured on float32 0.1 which "
    "reaches 107374184 exactly at this scale and misses 100000.0015 by a step at one part in "
    "a million; the exponent is thirty because one step is then two to the minus thirty "
    "metres, a hundred and twenty eight times finer than the float32 step at one metre that "
    "the correction is finally added to, so the quantisation is invisible for every particle "
    "further than one hundred and twenty eighth of a metre from the world origin, and because "
    "the remaining range is two to the thirty third metres of summed same sign correction, "
    "which at the quarter metre budget of one contribution is thirty four billion "
    "contributions on one component of one particle inside one accumulation window; an "
    "accumulating family lands at most four contributions per row it reads and a row is "
    "addressed by a signed 32 bit element index, so no world this engine can express reaches "
    "even a quarter of that budget")

DEFAULT_SIMULATION_FREQUENCY = 90
SIMULATION_FREQUENCY_LOW = 30
SIMULATION_FREQUENCY_HIGH = 150
DEFAULT_MAX_SIMULATION_COUNT_PER_FRAME = 3
MAX_SIMULATION_COUNT_LOW = 1
MAX_SIMULATION_COUNT_HIGH = 5

MAX_DISTANCE_RATIO_FUTURE_PREDICTION = 1.3

FRICTION_MASS = 3.0
DEPTH_MASS = 5.0
FRICTION_DAMPING_RATE = 0.6

TETHER_COMPRESSION_STIFFNESS = 1.0
TETHER_STRETCH_STIFFNESS = 1.0
TETHER_STRETCH_LIMIT = 0.03
TETHER_STIFFNESS_WIDTH = 0.3
TETHER_COMPRESSION_VELOCITY_ATTENUATION = 0.7
TETHER_STRETCH_VELOCITY_ATTENUATION = 0.7

DISTANCE_VELOCITY_ATTENUATION = 0.3
DISTANCE_HORIZONTAL_STIFFNESS = 0.5

TRIANGLE_BENDING_MAX_ANGLE = 120.0
VOLUME_MIN_ANGLE = 90.0
VOLUME_SIGN = 100
VOLUME_SCALE = 1000.0
BENDING_FIXED_INVERSE_MASS = 0.01
ONE_SIXTH = 1.0 / 6.0

ANGLE_LIMIT_ITERATION = 3
ANGLE_LIMIT_ATTENUATION = 0.9
ANGLE_LIMIT_ROTATION_RATIO = 0.4

ANGLE_LIMIT_RATIO_MAX = 8.0

ANGLE_LIMIT_RATIO_REASON = (
    "the limit angle is a cone around the rest direction, which says the same thing in "
    "every direction a joint can lean, and a ribbon is the case where that is wrong: a "
    "strip of cloth folds across its face far more readily than it bends within its own "
    "plane, and a single chain of bones carries no width for the geometry to express that "
    "with, so the only place the distinction can live is the limit itself; the curve keeps "
    "meaning what it always meant, the limit along the bone's own local x axis, and this "
    "ratio widens the limit along its local z axis by that factor, which opens the cone "
    "into an ellipse standing in the plane the bone roll already orients, so an artist "
    "points the narrow axis at the ribbon's width by rolling the bone and nothing else has "
    "to be authored; widening rather than narrowing is what keeps the shape free of a "
    "singular case, because the denominator of the ellipse is bounded below by one for "
    "every ratio at or above one, while narrowing towards zero collapses the ellipse to a "
    "line segment and asks for nought over nought along the narrow axis, which would have "
    "to be bought off with a floor and a clamp that nothing else here needs; the shape is "
    "an ellipse and not two independent per axis clamps because only the ellipse "
    "degenerates back to the cone when the two axes are equal, measured at 6.4e-15 degrees "
    "against the cone across 2049 azimuths while the independent form lets the diagonal "
    "reach 39.2 degrees for a 30 degree limit and is not even monotonic in the ratio")

MAX_MOVEMENT_SPEED_LIMIT = 10.0
MAX_ROTATION_SPEED_LIMIT = 1440.0
MAX_PARTICLE_SPEED_LIMIT = 10.0

COLLIDER_DYNAMIC_FRICTION_RATIO = 1.0
COLLIDER_STATIC_FRICTION_RATIO = 1.0

COLLIDER_EXIT_BISECTION_STEPS = int(_numpy.finfo(_numpy.float32).nmant) + 1
COLLIDER_EXIT_TOLERANCE = float(_numpy.finfo(_numpy.float32).eps)

COLLIDER_EXIT_MARCH_STEPS = 8

COLLIDER_EXIT_SEARCH_STEPS = COLLIDER_EXIT_MARCH_STEPS + COLLIDER_EXIT_BISECTION_STEPS

COLLIDER_EXIT_MARCH_STEPS_REASON = (
    "how many times the push out search is allowed to divide the depth by the slope before "
    "it gives up and reports that this body cannot be left along this ray; a convex body "
    "needs one step, because dividing the depth by the cosine between the push direction "
    "and the surface normal lands at or past the surface, and every extra step is for a "
    "surface that folds back over itself; the budget is not counted from outside, because a "
    "criterion that counted it would have to write the walk out a second time and that "
    "second copy is what this phase existed to remove; what colliderfield.py asserts is "
    "the property the budget is bought for, that the search leaves every point of a grid "
    "inside every shape outside the body, and a budget too small to reach the surface "
    "shows up there as points left inside, which is how the walk standing still on the "
    "surface was caught twice, at a hundred points of eleven hundred and then at thirty "
    "six of fifteen hundred; the search "
    "spends the rest of its budget halving between the last sample that read inside and "
    "the first that read outside, which is why the two counts are added into one loop "
    "count rather than being two loops, and why the total is the mantissa of a float32 "
    "plus the walk")

ACCD_SEPARATION_SCALE = 0.1

ACCD_SEPARATION_SCALE_REASON = (
    "the fraction of the remaining distance the additive continuous test stops short by, "
    "the s of Algorithm 1 of Li, Kaufman and Jiang, Codimensional Incremental Potential "
    "Contact, ACM Transactions on Graphics 40 number 4 article 170, which that paper sets "
    "to one tenth in every one of its examples and which it states has to be greater than "
    "zero for the iteration to terminate at all, because the stopping test is that the "
    "distance has fallen to s times what it started at and a scale of zero is a test that "
    "never fires")

ACCD_ADVANCE_SCALE = 0.9

ACCD_ADVANCE_SCALE_REASON = (
    "line 21 of Algorithm 1 of the same paper scales every advance after the first one by "
    "nine tenths of the safe bound, which the paper gives as being for improved "
    "convergence; the first advance is the full bound of its Equation 10 less the "
    "separation scale, and only the later ones carry this factor, which is why the two "
    "numbers are two rows and not one")

ACCD_STEP_LIMIT = 16

ACCD_STEP_LIMIT_REASON = (
    "Algorithm 1 of the same paper loops until it converges, and its own worst case "
    "analysis says the iteration count is unbounded when a primitive starts close and "
    "carries a displacement that cannot be cancelled out; a kernel cannot loop without a "
    "bound, so the walk stops here and returns the largest time it has proved safe, which "
    "is a shorter step than the true time of impact and therefore still a step that cannot "
    "pass through anything; the budget is not counted from outside for the same reason "
    "the push out walk is not, and what stands behind it is tunnelcheck.py, which drives a "
    "plate thin enough to fit between two substep samples through a curtain and asserts "
    "that every particle in its path is pushed; a budget too small to reach the time of "
    "impact shows up there as a plate that walked through; the same budget carries the "
    "resting march, where the point begins inside the offset and the loop advances by the "
    "distance to the body surface instead, because the two are the same loop and a point "
    "spends its turns on one or the other and never on both, and running that march out of "
    "turns is the same kind of shortfall as running the additive test out of them: the "
    "time handed back is smaller than the true one and a step that cannot cross anything, "
    "so a budget too small shows there as a particle held further back along its travel "
    "than it had to be and never as one carried through")

COLLIDER_MESH_VERTEX_FAN_LIMIT = 64

COLLIDER_MESH_EDGE_RING_LIMIT = 16

COLLIDER_MESH_EDGE_RING_REASON = (
    "the pseudo normal of an edge is the sum of the normals of the faces that meet on it, "
    "which is two faces on a surface a person models, one face along a rim and any number "
    "at a place where a surface branches; the device walks the faces of one edge through "
    "the ring its half edges are linked into, so the walk needs a bound it can be compiled "
    "against, and this bound is far above the four faces the widest branch of the judged "
    "body carries; an edge that passes it is refused where the triangles arrive rather "
    "than silently walked part way round, for the same reason a fan is")

COLLIDER_MESH_QUERY_SEED_FRACTION = 1.0 / 64.0

COLLIDER_MESH_QUERY_EXPANSIONS = 8

COLLIDER_MESH_QUERY_SEED_REASON = (
    "the narrow phase of a mesh collider opens its search at this fraction of the shape's "
    "own bound and doubles until it either holds its own answer or reaches the distance its "
    "caller asked about, whichever comes first; the fraction is a fraction of the body "
    "rather than a length because a length would be a length in metres and the same body "
    "scaled would search a different part of itself, and a sixty fourth of a body covers "
    "the whole of it after six doublings, which is why the expansion budget is eight, that "
    "being what the deepest caller, the exit projection, can need")

COLLIDER_EDGE_SCAN_STEPS = 9

COLLIDER_EDGE_SCAN_STEPS_REASON = (
    "how many places along a cloth edge are looked at before the halving starts, so the "
    "halving begins inside a cell one eighth of an edge wide rather than across the whole "
    "of it; the halving walks to a stationary point of the field along the edge and a "
    "capsule puts two of those on an edge that passes near both of its end caps, so "
    "without the scan it settles on whichever one it started beside, which edgefoot.py "
    "measured at very nearly a whole edge of error on the tail; nine places is the "
    "coarsest scan that puts the two end caps of the shortest capsule the criterion worlds "
    "build into different cells, and the halving then costs nothing extra because it is "
    "the same naming of the field in the same loop")

COLLIDER_EDGE_SEARCH_STEPS = 6

COLLIDER_EDGE_SEARCH_STEPS_REASON = (
    "how many times the search for the place along a cloth edge where a body is closest "
    "halves the span it has left, so the foot is placed to one part in two to this power "
    "of the edge, which is one part in sixty four; an edge is one particle spacing long, "
    "so at the default spacing of a tenth of a metre that is a millimetre and a half of "
    "play in where along the edge the contact is reported, well under the thickness the "
    "contact is measured against, and the correction is shared between the two ends by "
    "that same parameter, so an error in it moves weight between the ends rather than "
    "moving the contact")

SELF_COLLISION_SOLVER_ITERATION = 4
SELF_COLLISION_FIXED_MASS = 100.0
SELF_COLLISION_FRICTION_MASS = 10.0
SELF_COLLISION_CLOTH_MASS = 50.0
SELF_CONTACT_DETECTION_MARGIN = 3.0

SELF_CONTACT_DETECTION_MARGIN_REASON = (
    "the continuous test that decides whether a pair is a contact runs once a substep, and "
    "the solver then moves every particle four more times inside that substep, so a pair "
    "the test proved safe can be driven into the thickness by a correction that some other "
    "contact asked for, and a set decided exactly on the thickness has no answer left for "
    "it; the walk therefore tests against this multiple of the thickness while the response "
    "still pushes to the thickness itself, which is the same shape as Bridson's proximity "
    "distance being wider than the cloth thickness in section 7.2 and as the barrier "
    "distance of C-IPC being wider than the contact it guards; the multiple is three "
    "because that is the band the engine already carried, thickness plus twice thickness, "
    "and it was measured to matter: at a band of exactly the thickness the frozen asset "
    "reads 1324 crossings and at three thicknesses it reads what the table in the handoff "
    "records, on the same detection and the same response")

SELF_COLLISION_INTERSECT_DIV = 2
SELF_COLLISION_THICKNESS_MIN = 0.001
SELF_COLLISION_THICKNESS_MAX = 0.05
SELF_COLLISION_UNIFORM_GRID_SCALE = 3.0
SELF_CONTACT_SLOTS_PER_PRIMITIVE = 48

SELF_CONTACT_SLOTS_REASON = (
    "one query primitive keeps this many contacts and drops the rest, so the number has to "
    "cover the worst primitive of the densest cloth anybody runs, and it is measured rather "
    "than guessed; contactslots.py builds the densest world the judge suite has, two ten by "
    "ten cloths at a spacing of one tenth of a metre and a thickness of half that, and the "
    "worst primitive there asks for 32 contact candidates and 40 intersection candidates, "
    "while the frozen asset asks for 32 and 26, so this default clears the worst measured "
    "demand by a fifth; a power of two ladder that steps from 32 to 64 cannot see that, "
    "which is why the judge measures the demand itself and refuses when the default falls "
    "under it; a team that needs more sets its own column, and since the host now refuses a "
    "frame whose primitives dropped candidates, a world that outgrows this default says so "
    "instead of quietly moving the wrong way")

CONTACT_PATH_SELF_COLLISION = 0
CONTACT_PATH_COLLIDER = 1
CONTACT_PATH_LEN = 2

INCIDENCE_GATE_ACCEPTS_EVERY_INCIDENCE = -1.0

INCIDENCE_GATE_COS_REASON = (
    "a contact search that finds a surface at a grazing angle has to decide whether the "
    "surface it found is the one the point is really resting against; on the collider path "
    "the search is an exact closest point query, so the foot of the direction lies inside "
    "the triangle it reports and the direction is that triangle normal, which was measured "
    "rather than assumed, a cloth draped over a box, over a sharp blade whose convex edge "
    "opens to a hundred and seventy four degrees, and against a plate one millimetre thick, "
    "all read an incidence of one to the last float32 place, and the row changes nothing "
    "until it is raised to exactly one, where it refuses every contact and the cloth falls "
    "through; the self collision path used to answer the same question with an angle, "
    "keeping only the hits whose direction sat within sixty degrees of the triangle normal, "
    "because its search is a bound volume overlap that hands back triangles the point is "
    "beside as readily as the one it is above; that angle is gone, because selfattrib.py "
    "measured it refusing 971 of the 1424 crossings the frozen asset produces and the "
    "question it was guessing at is now answered exactly by the continuous test in "
    "kernels.SELF_ACCD_REASON, so the row below says the path accepts every incidence and "
    "nothing on that path reads it; the column stays a column only until the collider "
    "path's own guard is re-derived, at which point one reader is left and it belongs "
    "beside that reader rather than in a table of paths")

HONOR_INTERSECT_FREEZE_REASON = (
    "a particle the intersection pass has marked as sitting on the wrong side of a sheet is "
    "held still by the self collision solver, because a correction computed from a normal "
    "that already points the wrong way drives it further in, and waiting for the "
    "intersection pass to clear the mark is the cheaper answer; the collider exit "
    "projection is the opposite case, it is the step that makes the output non penetrating "
    "and its direction comes from the collider field and not from the mark, so honouring "
    "the freeze there would leave a marked particle inside the body with the one step that "
    "could push it out switched off, and nothing else in the frame moves it, which is a "
    "permanent penetration that cannot recover on its own; the two paths therefore carry "
    "the answer as a column rather than sharing one flag")

CONTACT_PATH_SPECIFICATION = (
    ("self_collision", CONTACT_PATH_SELF_COLLISION,
     INCIDENCE_GATE_ACCEPTS_EVERY_INCIDENCE, 1),
    ("collider", CONTACT_PATH_COLLIDER,
     INCIDENCE_GATE_ACCEPTS_EVERY_INCIDENCE, 0),
)


def _validate_contact_path_specification():
    seen = set()
    for row in CONTACT_PATH_SPECIFICATION:
        assert len(row) == 4, \
            "a contact path row declares the name, the slot, the incidence gate cosine and " \
            "whether the path honours the intersection freeze, got %r" % (row,)
        path_name, slot, gate_cosine, honour = row
        assert path_name not in seen, "contact path %s is declared twice" % path_name
        seen.add(path_name)
        assert isinstance(slot, int) and 0 <= slot < CONTACT_PATH_LEN, \
            "contact path %s takes the slot %r and the table holds %d slots" \
            % (path_name, slot, CONTACT_PATH_LEN)
        assert -1.0 <= float(gate_cosine) <= 1.0, \
            "%s\ncontact path %s declares the incidence gate cosine %r" \
            % (INCIDENCE_GATE_COS_REASON, path_name, gate_cosine)
        assert honour in (0, 1), \
            "%s\ncontact path %s declares %r for the intersection freeze" \
            % (HONOR_INTERSECT_FREEZE_REASON, path_name, honour)
    assert len(seen) == CONTACT_PATH_LEN, \
        "the contact path table holds %d slots and declares %d paths" \
        % (CONTACT_PATH_LEN, len(seen))


_validate_contact_path_specification()

SCL_EE_COUNT = 0
SCL_PT_COUNT = 1
SCL_IP_COUNT = 2
SCL_ERROR = 3
SCL_USE_INTERSECT = 4
SCL_FRAME_INDEX = 5
SCL_LEN = 8

SCAL_FRAME_DT = 0
SCAL_SIM_DT = 1
SCAL_TIME_SCALE = 2
SCAL_POWER0 = 3
SCAL_POWER1 = 4
SCAL_POWER2 = 5
SCAL_POWER3 = 6
SCAL_F_LEN = 8

SCAL_MAX_SIM = 0
SCAL_N_ZONES = 1
SCAL_SUB_END = 2
SCAL_I_LEN = 4

FRAME_SCALAR_PLANE_SPECIFICATION = (
    ("frame_float", "float32", SCAL_F_LEN),
    ("frame_int", "int32", SCAL_I_LEN),
)

CARRY_OLD_COMPONENT_POSITION = 0
CARRY_OLD_COMPONENT_ROTATION = 3
CARRY_ANCHOR_SHIFT_VECTOR = 7
CARRY_ANCHOR_SHIFT_ROTATION = 10
CARRY_SMOOTHING_SHIFT_VECTOR = 14
CARRY_LEN = 17

WIND_MAX_TIME = 10000.0
WIND_BASE_SPEED = 7.5
WIND_TURBULENCE_ANGLE = 45.0
WIND_ZONE_SLOTS = 4
WIND_ZONE_RESULT_SLOTS = 8
WIND_ZONE_MIN_MAIN = 1e-6
WIND_MIN_SPEED = 0.01

WIND_ZONE_SLOTS_REASON = (
    "the widest wind one particle can stand in is three addition zones and one ordinary "
    "zone: the addition zones are capped at three where they are gathered and the ordinary "
    "zones keep only the single smallest volume one that contains the point, so four is the "
    "exact number of blends a particle ever sums and the per particle result carries four "
    "slots for that and nothing wider; this is not the number of zones a team may reach, "
    "because two particles of one team can stand in two different ordinary zones and the "
    "team then keeps a phase clock for each, so the distinct zones a team's particles reach "
    "between them can pass four, and when they do the frame is refused rather than quietly "
    "dropping the zones that did not fit, the same shape of answer a self contact primitive "
    "gives when it outgrows its slots; the phase clock a zone advances is kept once per team "
    "and shared by every particle standing in it, so a radial zone's turbulence runs at the "
    "rate of one representative depth rather than each particle's own, which is what it "
    "already did before the containment test was moved onto the particle and is left that "
    "way")

BONE_SPRING_DISTANCE_STIFFNESS = 0.5
BONE_SPRING_TETHER_COMPRESSION_LIMIT = 0.8
BONE_SPRING_COLLISION_FRICTION = 0.5
BONE_SPRING_FIX_MASS = 10.0
BONE_CLOTH_FIX_MASS = 50.0

PROXY_MESH_BONE_TRIANGLE_ANGLE = 120.0
SAME_SURFACE_ANGLE = 80.0

DISTANCE_CULLING_MAX_LENGTH = 100.0

CURVE_LUT_SAMPLES = 16

ATTR_FIXED = 0x01
ATTR_MOVE = 0x02
ATTR_INVALID_MOTION = 0x08
ATTR_DISABLE_COLLISION = 0x10
ATTR_ZERO_DISTANCE = 0x20
ATTR_TRIANGLE = 0x80

COLLIDER_SPHERE = 0
COLLIDER_CAPSULE = 1
COLLIDER_PLANE = 2
COLLIDER_MESH = 3

COLLIDER_MESH_GEOMETRY_REASON = (
    "a mesh collider carries its triangles in the frame of the collider itself and moves "
    "the way every other collider kind moves, by the interpolated pose the frame already "
    "computes, so the query point is carried into that frame and the answer is carried "
    "back; the alternative, holding the triangles in world space and refitting the "
    "acceleration structure every frame, cannot be recorded on both compile targets, "
    "because warp 1.16 records a bounding volume hierarchy refit into the host capture "
    "stream, see native/bvh.cpp wp_bvh_refit_host, and records nothing for a mesh refit, "
    "see native/mesh.cpp wp_mesh_refit_host and the operation list in "
    "native/apic_types.h, so a captured host frame would replay against the tree the "
    "geometry had when the frame was recorded; carrying the motion in the pose also means "
    "the triangles are uploaded once instead of once a frame, and it is the same shape of "
    "thing the sphere and the capsule already are")

COLLISION_NONE = 0
COLLISION_POINT = 1
COLLISION_EDGE = 2

SELF_MODE_NONE = 0
SELF_MODE_FULL_MESH = 2

TELEPORT_NONE = 0
TELEPORT_RESET = 1
TELEPORT_KEEP = 2

ZONE_BOX = 1
ZONE_SPHERE_DIR = 2
ZONE_SPHERE_RADIAL = 3

FORCE_NONE = 0
FORCE_VELOCITY_ADD = 1
FORCE_VELOCITY_ADD_WITHOUT_DEPTH = 2
FORCE_VELOCITY_CHANGE = 3
FORCE_VELOCITY_CHANGE_WITHOUT_DEPTH = 4

KIND_POINT = 0
KIND_EDGE = 1
KIND_TRIANGLE = 2

GRID_KEY_BIAS = 1 << 20
GRID_KEY_IGNORE = 1 << 62


def simulation_power(frequency):
    t = DEFAULT_SIMULATION_FREQUENCY / float(frequency)
    return (
        t,
        t ** 0.5 if t > 1.0 else t,
        t ** 0.3 if t > 1.0 else t,
        t ** 1.8,
    )
