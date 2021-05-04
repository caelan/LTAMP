from __future__ import print_function

import os
import numpy as np
import subprocess
#import sys
#sys.path.extend([
#    '/Users/caelan/Programs/LIS/git/collaborations/amber',
#])

# Convex decomposition
# Needs this bit of BHPN/Amber - mostly a call to Scipy interface to QHull
# TODO: amber uses python2 only
from pybullet_tools.transformations import rotation_matrix, translation_matrix
from pybullet_tools.utils import Mesh, read, write, safe_remove
from retired.mesh_tools.create_off_meshes import rotation, PCD_TEMPLATE

from scipy.spatial import ConvexHull

#####################################

# https://stackoverflow.com/questions/42440281/meshlabserver-on-macos-wont-work
# https://github.com/cnr-isti-vclab/meshlab/blob/master/src/meshlabserver/meshlabserver.txt
MESHLAB = '/Applications/meshlab.app/Contents/MacOS/meshlabserver -i {} -s {} -o {}' #  -l x -m ...
MESHLAB_SCRIPT = './perception_tools/pcd_from_mesh.mlx'
MESHLAB_WD = '/Applications/meshlab.app/Contents/MacOS'

# TODO: meshlab bug: https://github.com/cnr-isti-vclab/meshlab/issues/64

def pcd_from_mesh(mesh_path, pcd_path, delete=True):
    filename, ext = os.path.splitext(pcd_path)
    assert ext == '.pcd'
    xyz_path = '{}.xyz'.format(filename)
    command = MESHLAB.format(mesh_path, MESHLAB_SCRIPT, xyz_path)
    print(command)
    result = subprocess.check_output(command, shell=True) #, cwd=MESHLAB_WD
    #print(result)
    xyz_s = read(xyz_path)
    if delete:
        safe_remove(xyz_path)
    lines = xyz_s.strip().splitlines()
    pcd_s = PCD_TEMPLATE.format(len(lines), xyz_s)
    write(pcd_path, pcd_s)
    return pcd_s

#####################################

def adjacent(s1, s2):
    # two faces share an edge
    return len(set(s1).intersection(set(s2))) == 2

def normSimplices(ch):
    """
    Sometimes the simplices don't go around CCW, this reverses them if
    they don't.  Stupid wasteful...
    """
    points = ch.points
    simplices = ch.simplices
    normSimp = []
    for f in range(simplices.shape[0]):
        normal = np.cross(points[simplices[f][1]] - points[simplices[f][0]],
                          points[simplices[f][2]] - points[simplices[f][1]])
        if np.dot(normal, ch.equations[f][:3]) > 0:
            normSimp.append(simplices[f].tolist())
        else:
            normSimp.append(simplices[f].tolist()[::-1]) # reverse
    return normSimp

def groupSimplices(ch):
    """
    Find all the simplices that share a face.  Only works for convex
    solids, that is, we assume those simplices are adjacent.
    """
    groups = []
    eqns = ch.equations
    ne = eqns.shape[0]
    done = set([]) # faces already done
    for e in range(ne): # loop over eqns
        if e in done:
            continue
        face = [e]
        done.add(e)
        for e1 in range(e+1, ne): # loop for remaining eqns
            if e1 in done:
                continue
            if np.all(np.abs(eqns[e] - eqns[e1]) < 1e-6):
                # close enough
                face.append(e1)
                done.add(e1)
        # remember this "face"
        groups.append(face)
    # all the faces.
    return groups

def mergedFace(face, simplex):
    """
    Insert the new vertex from the simplex into the growing face.
    """
    common = set(simplex).intersection(set(face))
    diff = set(simplex).difference(set(face))
    assert (len(diff) == 1) and (len(common) == 2), 'mergedFace - Inconsistent'
    newFace = face[:] # copy
    n = len(face)
    for i in range(n):
        if (newFace[i] in common) and (newFace[(i+1)%n] in common):
            newFace[i+1:] = [diff.pop()] + newFace[i+1:]
            break
    return newFace

def mergeFaces(ch):
    # This cleans up the faces, but it really slows things down a lot.
    # Normalize the faces, a list of lists
    normSimp = normSimplices(ch)
    # Group simplices on each face
    groups = groupSimplices(ch)
    # Merge the normed simplices; this is reduce with mergedFace
    mergedFaces = []
    for group in groups:
        face = normSimp[group[0]]
        remaining = group[1:]
        while remaining:
            found = False
            for fi in remaining:
                if adjacent(face, normSimp[fi]):
                    face = mergedFace(face, normSimp[fi])
                    remaining.remove(fi)
                    found = True
                    break
            if not found:
                raise Exception('Could not find adjacent face')
        mergedFaces.append(face)
    # The simplex specifies indices into the full point set.
    return mergedFaces

def chFaces(ch, indices, merge=True):
    """
    Unfortunately ConvexHull triangulates faces, so we have to undo it...
    """
    if merge:
        mergedFaces = mergeFaces(ch)
    else:
        mergedFaces = ch.simplices.tolist()
    mapping = dict((indices[i], i) for i in range(indices.shape[0]))
    faces = [np.array([mapping[fi] for fi in face]) for face in mergedFaces]
    return faces

def convexHull(verts):
    """
    Return a Prim that is the convex hull of the input verts.
    """
    ch = ConvexHull(verts[:3,:].T)
    # unique indices of ch vertices
    # indices = np.array(sorted(list(set(ch.simplices.flatten().tolist()))))
    indices = ch.vertices
    return Mesh(verts[:, indices], chFaces(ch, indices))

#####################################

def make_revolute_chunks(side, n_theta=60, n_chunks=10,
                         in_off=0.25, scale=0.01, flip=True):
    #import geometry.shapes
    th_step = 2*np.pi / n_theta
    outer_all = [np.array([scale*x, 0., scale*z, 1]) for [x,z] in side]
    # This offset is "special purpose"
    # TODO: apply to z as well?
    inner_all = [np.array([scale*(x-in_off), 0., scale*z, 1]) for [x,z] in side[:-2]] + \
        [np.array([scale*side[-1][0], 0., scale*(side[-1][1]-in_off), 1])]
    outer = outer_all[:-2]
    inner = inner_all[:-1]
    assert len(outer) == len(inner)

    mn = np.amax(np.vstack(outer).T, axis=1)
    def flip_points(verts):
        if flip:
            verts = np.dot(rotation_matrix(np.pi, (1,0,0)), verts)
            verts = np.dot(translation_matrix((0, 0, mn[2])), verts)
        return verts

    outer_scans = []
    inner_scans = []
    # The vertices
    for i in range(n_theta):
        mat = rotation(i*th_step)
        outer_scans.append([np.dot(mat, p) for p in outer])
        inner_scans.append([np.dot(mat, p) for p in inner])

    chunk_size = int(n_theta / float(n_chunks))
    chunks = []
    ind = 0
    n = len(outer_scans)
    #m = len(outer_scans[0])
    for j in range(n_chunks):
        vl = []
        for k in range(chunk_size+1):
            vl.extend([outer_scans[(ind+k)%n], inner_scans[(ind+k)%n]])
        ind += chunk_size
        verts = np.vstack(vl).T
        chunks.append(convexHull(flip_points(verts)))

    # The cap
    vl = []
    off = np.array([0., 0., -scale*in_off, 0.])
    for i in range(n_theta):
        vl.append(outer_scans[i][-1])
        vl.append(off + outer_scans[i][-1])

    verts = np.vstack(vl).T
    chunks.append(convexHull(flip_points(verts)))
    return chunks
    #write_obj(chunks, filename)

#####################################

def write_obj(chunks, path):
    template = 'prim%d'
    with open(path, 'w') as fl:
        fl.write('# OBJ file\n')
        fl.write('mtllib %s\n' % path)
        nv = 0
        for i, (verts, faces) in enumerate(chunks):
            name = template%i
            fl.write('o %s\n'%name)
            for p in range(verts.shape[1]):
                fl.write('v  %6.3f %6.3f %6.3f\n'%tuple(x for x in verts[0:3,p]))
            fl.write('s off\n')
            for f in range(len(faces)):
                face = faces[f]
                fl.write('f')
                for k in range(len(face)):
                    fl.write(' %d'%(nv+1+face[k]))
                fl.write('\n')
            nv += verts.shape[1]
    print('Saved', path)
