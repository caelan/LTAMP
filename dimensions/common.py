import numpy as np
import os

from dimensions.bowls.dimensions import BOWL, BOWL_PROPERTIES
from dimensions.cups.dimensions import CUP, CUP_PROPERTIES
from perception_tools.common import get_models_path
from pybullet_tools.utils import create_obj, apply_alpha

# in_off = 0.0025 m

SCALE = 0.01 # Units are in cm

MODELS_TEMPLATE = '{}s/' # relative to models/
SUFFIX_TEMPLATE = '_{}'
OBJ_TEMPLATE = '{}.obj'
PCD_TEMPLATE = '{}.pcd'

# TODO: merge BOWL_PROPERTIES and CUP_PROPERTIES directly
OBJECT_PROPERTIES = {
    BOWL: BOWL_PROPERTIES,
    CUP: CUP_PROPERTIES,
}

def apply_suffix(names, ty):
    return sorted(name + SUFFIX_TEMPLATE.format(ty) for name in names)

CUPS = apply_suffix(CUP_PROPERTIES, CUP)
BOWLS = apply_suffix(BOWL_PROPERTIES, BOWL)
#SPOONS = []
# 8 bowls, 7 cups

def approximate_bowl(properties, d=2, n_pieces=5):
    # TODO: edge detection to automatically produce curvature
    # Upside-down right half
    # (0, h) to (top_d/2, 0)
    assert 1 <= d
    if d == 1:
        n_pieces = 2 # Not sure why 1 doesn't work
    assert 1 <= n_pieces
    bottom_d, top_d, h = properties[:3]
    assert bottom_d <= top_d
    delta_r = (top_d - bottom_d) / 2.
    x = [-delta_r, 0, +delta_r]
    z = [h, 0, h]
    coeff = np.polyfit(x[:d + 1], z[:d + 1], d)
    poly = np.poly1d(coeff)
    fn = lambda v: (h - abs(poly(v - bottom_d/2.)) ) #, 3)
    points = [(0, h)] + [(x, fn(x)) for x in np.linspace(
        bottom_d/2., top_d/2., n_pieces+1, endpoint=True)]
    return points[::-1]

#####################################

def get_prefix(name, suffix):
    if not name.endswith(suffix):
        return None
    return name[:-len(suffix)]

def normalize_rgb(color, **kwargs):
    return apply_alpha(np.array(color, dtype=float) / 255.0, **kwargs)

#####################################

def get_properties(bowl):
    for ty, properties in OBJECT_PROPERTIES.items():
        suffix = SUFFIX_TEMPLATE.format(ty)
        prefix = get_prefix(bowl, suffix)
        if prefix:
            return properties[prefix]
    return None

def load_cup_bowl_obj(bowl):
    for ty in OBJECT_PROPERTIES:
        if not bowl.endswith(SUFFIX_TEMPLATE.format(ty)):
            continue
        obj_path = os.path.join(get_models_path(),
                                MODELS_TEMPLATE.format(ty),
                                OBJ_TEMPLATE.format(bowl))
        properties = get_properties(bowl)
        assert properties is not None
        color = normalize_rgb(properties.color)
        return obj_path, color
    return None, None

def load_cup_bowl(bowl, **kwargs):
    obj_path, color = load_cup_bowl_obj(bowl)
    if obj_path is None:
        return None
    from plan_tools.common import MODEL_MASSES
    return create_obj(obj_path, mass=MODEL_MASSES[bowl], color=color, **kwargs)
