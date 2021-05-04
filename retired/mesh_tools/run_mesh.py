from __future__ import print_function

import sys
import os
import argparse

sys.path.extend([
    'ss-pybullet',
])

from perception_tools.common import get_models_path
from pybullet_tools.utils import ensure_dir, connect, create_obj, approximate_as_cylinder, wait_for_user, remove_body
from retired.mesh_tools.create_off_meshes import draw_curvature
from retired.mesh_tools.create_obj_meshes import make_revolute_chunks, write_obj, pcd_from_mesh
from dimensions.common import normalize_rgb, approximate_bowl, SCALE, OBJECT_PROPERTIES, MODELS_TEMPLATE, \
    SUFFIX_TEMPLATE, OBJ_TEMPLATE, PCD_TEMPLATE
from dimensions.bowls.dimensions import BOWL
from dimensions.cups.dimensions import CUP


def create_meshes(ty, draw=False, visualize=False):
    assert not (visualize and draw) # Incompatible?
    suffix = SUFFIX_TEMPLATE.format(ty)

    models_path = os.path.join(get_models_path(), MODELS_TEMPLATE.format(ty))
    ensure_dir(models_path)
    for prefix, properties in OBJECT_PROPERTIES[ty].items():
        color = normalize_rgb(properties.color)
        side = approximate_bowl(properties, d=2) # 1 doesn't seem to really work
        name = prefix + suffix
        print(name, color)
        print(side)
        if draw:
            draw_curvature(side, name=name)
        chunks = make_revolute_chunks(side, n_theta=60, n_chunks=10,
                                      in_off=properties.thickness/4.,
                                      scale=SCALE)
        obj_path = os.path.join(models_path, OBJ_TEMPLATE.format(name))
        write_obj(chunks, obj_path)
        if visualize:
            body = create_obj(obj_path, color=color)
            _, dims = approximate_as_cylinder(body)
            print(dims)
            wait_for_user()
            remove_body(body)
        pcd_path = os.path.join(models_path, PCD_TEMPLATE.format(name))
        pcd_from_mesh(obj_path, pcd_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--draw', action='store_true',
                        help='When enabled, draws cross sections.')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='When enabled, visualizes the meshes.')
    args = parser.parse_args()
    # Requires MeshLab: http://www.meshlab.net/

    if args.visualize:
        connect(use_gui=True)
    for ty in [CUP, BOWL]:
        create_meshes(ty, draw=args.draw, visualize=args.visualize)

if __name__ == '__main__':
    main()
