import numpy as np

from . import math as pm

SLOT_COUNT_DEFAULT = 16

SIGN_MODE_FROZEN_FIRST = 0
SIGN_MODE_PSEUDO_NORMAL = 1
SIGN_MODE_FIXED_PLUS = 2

NORMAL_WRITE_IF_VALID = 0
NORMAL_WRITE_ALWAYS = 1

MASS_MODEL_UNIT = 0
MASS_MODEL_SELF_COLLISION = 1

INCIDENCE_GATE_OFF = np.float32(-1.0)

SIGN_BITS = np.int32(-2147483648)
MAGNITUDE_BITS = np.int32(0x7FFFFFFF)


def float_to_ordered_int(value):
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.int32)
    return (bits ^ ((bits >> np.int32(31)) & MAGNITUDE_BITS)).astype(np.int32)


def contact_key(gap, target_row):
    return float_to_ordered_int(gap), np.ascontiguousarray(target_row, dtype=np.int32)


def radius_at(lut, team_index, depth, scale_ratio, radius_base):
    sampled = pm.evaluate_team_lut(lut, team_index, depth).astype(np.float32)
    return (sampled * np.float32(scale_ratio) * np.float32(radius_base)).astype(np.float32)


def interpolate_radius_segment(radius_0, radius_1, ratio):
    return radius_0 + (radius_1 - radius_0) * ratio


def interpolate_radius_triangle(radius_0, radius_1, radius_2, weight_1, weight_2):
    return radius_0 + (radius_1 - radius_0) * weight_1 + (radius_2 - radius_0) * weight_2
