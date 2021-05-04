from __future__ import print_function

import numpy as np

# This is the "standard" transformations.py file
# https://www.lfd.uci.edu/~gohlke/code/transformations.py
# http://docs.ros.org/jade/api/tf/html/python/transformations.html
from pybullet_tools.transformations import rotation_matrix, translation_matrix
from pybullet_tools.utils import user_input, Mesh

# To convert off mesh file into PCD
# Import off into Meshlab,
# >Remeshing>Turn into pure-triangular mesh
# >Sampling>Montecarlo sampling (choose 10000 samples)
# >Export as> .xyz file
# Add standard pcd header, copy it from existing file, change number of points

# brown bowl: scale = 0.0155
bowl_side = [[4.4, 0.], [4.2, 0.8], [4.0, 2.0], [3.8, 2.4],
             [3.6, 3.0], [3.4, 3.3], [3.2, 3.6], [3.0, 3.8],
             [2.8, 4.1], [2.6, 4.3], [2.4, 4.5], [2.3, 4.6],
             [2.3, 4.8], [2.4, 5.0], [2.3, 5.2], [2.2, 5.33],
             [0.0, 5.33]] # flipped
# 60 outside faces, 10 inside faces

# blue cup: scale = 0.01 (cms)
cup_side = [[3.5, 0.], [2.4975, 10.4975], [2.5, 10.5], [0.0, 10.5]] # flipped
# 60 outside faces, 10 inside faces

#####################################

# Define an OFF mesh object that is rotationally symmetric, given one
# radial scan (points are x,z).  Origin is at x=0, z=0,.  The
# assumption is that this is a "bowl" or "cup" shaped object.  Assume
# object is upside down; the rim is on the x-y plane.  The radial
# scan of the outside surface starts at the rim and goes to to center
# of the "bottom", which is at (x=0, z=zmax).  The inner surface is
# offset (inward along x) by the value of in_off.  The inner surface
# has one fewer point than the outer surface.  The next-to-last point
# on the outer surface has no corresponding point on the inner
# surfaces

# side: is a list (x,z) pairs
# ntheta: is number of values of theta, e.g. 60
# filename: where to write the OFF file
# in_off: is the offset between the outer and inner surfaces
# scale: (turn x,z values into meters)
# flip: whether the final object should be flipped upside down

def draw_curvature(points, name=''):
    import matplotlib.pyplot as plt
    x, z = zip(*points)
    plt.plot(x, z)
    plt.title(name)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()
    return plt

def rotation(theta):
    return rotation_matrix(theta, (0, 0, 1))

def flip_points(points):
    points = np.dot(rotation_matrix(np.pi, (1, 0, 0)), points)
    mn = np.amin(points, axis=1)
    return np.dot(translation_matrix((0, 0, -mn[2])), points)

def make_revolute(side, n_theta, in_off=0.0025, scale=0.01, flip=True):
    th_step = 2*np.pi / n_theta
    outer = [np.array([scale*x, 0., scale*z, 1]) for [x,z] in side]
    # This offset is "special purpose"
    inner = [np.array([scale*x-in_off, 0., scale*z, 1]) for [x,z] in side[:-2]] + \
        [np.array([scale*side[-1][0], 0., scale*side[-1][1]-in_off, 1])]

    outer_scans = []
    inner_scans = []
    all_outer_verts = []
    all_inner_verts = []
    # The vertices
    for i in range(n_theta):
        mat = rotation(i * th_step)
        ov = [np.dot(mat, p) for p in outer]
        iv = [np.dot(mat, p) for p in inner]
        outer_scans.append(ov)
        inner_scans.append(iv)
        all_outer_verts.extend(ov)
        all_inner_verts.extend(iv)

    faces = []
    ov = np.vstack(all_outer_verts).T
    nos = len(outer)
    # The outer shell
    for s in range(1, n_theta):
        faces.extend(stitch_faces(outer_scans[s-1], outer_scans[s], (s-1)*nos, s*nos))
    faces.extend(stitch_faces(outer_scans[n_theta - 1], outer_scans[0], (n_theta - 1) * nos, 0))

    iv = np.vstack(all_inner_verts).T
    nis = len(inner)
    # The inner shell
    noff = n_theta * nos
    for s in range(1, n_theta):
        # Reverse faces for inside
        faces.extend([f[::-1] for f in stitch_faces(inner_scans[s-1], inner_scans[s],
                                                    noff+(s-1)*nis, noff+s*nis)])
    faces.extend([f[::-1] for f in stitch_faces(inner_scans[n_theta - 1], inner_scans[0],
                                                noff + (n_theta - 1) * nis, noff)])

    # The rim connecting the inner and outer shells
    for i in range(n_theta-1):
        faces.append((noff+i*nis, i*nos, (i+1)*nos, noff+(i+1)*nis)[::-1])
    faces.append((noff + (n_theta - 1) * nis, (n_theta - 1) * nos, 0, noff)[::-1])

    verts = np.hstack([ov, iv])
    if flip:
        # Flip the bowl
        verts = flip_points(verts)
    print(np.amin(verts, axis=1).tolist(),
          np.amax(verts, axis=1).tolist())
    verts = verts[:3,:].T.tolist()
    return Mesh(verts, faces)

def stitch_faces(scan1, scan2, off1, off2):
    # TODO(tlp): scan2 isn't used
    faces = []
    n = len(scan1)
    for i in range(n-1):
        faces.append((i+off1, i+off2, i+1+off2, i+1+off1))
    return faces

#####################################

def write_off(mesh, path, scale=1):
    verts, faces = mesh
    verts = np.array(verts).T
    nv = verts.shape[1]
    nf = len(faces)
    ne = 0 # nv + nf - 2 # Euler's formula...
    with open(path, 'w') as fl:
        fl.write('OFF\n')
        fl.write('%d %d %d\n'%(nv, nf, ne))
        for p in range(verts.shape[1]):
            fl.write('  %6.3f %6.3f %6.3f\n'%tuple([x*scale for x in verts[0:3,p]]))
        for f in range(len(faces)):
            face = faces[f]
            fl.write('  %d'%len(face))
            for k in range(len(face)):
                fl.write(' %d'%(face[k]))
            fl.write('\n')
    print('Saved', path)

#####################################

# http://pointclouds.org/documentation/tutorials/pcd_file_format.php
# https://github.mit.edu/Learning-and-Intelligent-Systems/cardblite/blob/master/data/bowl.pcd

PCD_TEMPLATE = """
# .PCD v.6 - Point Cloud Data file format
FIELDS x y z 
SIZE 4 4 4 
TYPE F F F 
COUNT 1 1 1 
WIDTH {0}
HEIGHT 1
VIEWPOINT 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.00000
POINTS {0}
DATA ascii
{1}
"""

def read_pcd(filename):
    with open(filename) as fl:
        for i in range(8):
            fl.readline()
        t, pts = fl.readline().split()
        assert t == 'POINTS'
        assert fl.readline().split()[0] == 'DATA'
        vl = []
        for i in range(int(pts)):
            vl.append(np.array([float(x) for x in fl.readline().split()[:3]]+[1.0]))
    verts = np.vstack(vl).T
    return verts

#####################################

# Simple tool to read and draw the pcd files; uses low-level HPN
# graphics, but all we need is to plot 3D points, Matplotlib could be
# used instead.

bowl = "/Users/tlp/git/cardblite/data/bowl.pcd"
greenblock = "/Users/tlp/git/cardblite/data/greenblock.pcd"
purpleblock = "/Users/tlp/git/cardblite/data/purpleblock.pcd"
bluecup = "/Users/tlp/git/cardblite/data/bluecup.pcd"
pcds = [
    (bowl, 'brown'),
    (greenblock, 'green'),
    (purpleblock, 'purple'),
    (bluecup, 'blue'),
]

old_bowl = "/Users/tlp/git/cardblite/data/bowl_old.pcd"
old_greenblock = "/Users/tlp/git/cardblite/data/greenblock_old.pcd"
old_purpleblock = "/Users/tlp/git/cardblite/data/purpleblock_old.pcd"
old_bluecup = "/Users/tlp/git/cardblite/data/bluecup_old.pcd"
old_pcds = [
    (old_bowl, 'brown'),
    (old_greenblock, 'green'),
    (old_purpleblock, 'purple'),
    (old_bluecup, 'blue'),
]

#####################################

# def show_pcds(files_and_colors):
#     from graphics.windowManager3D import makeWindow
#     w = makeWindow('W', 3*[-0.25, 0.25])
#     for fn, color in files_and_colors:
#         w.clear()
#         v = read_pcd(fn)
#         w.draw(v, color=color)
#         user_input('Next?')
