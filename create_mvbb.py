#!/usr/bin/env python

from klampt import *
import klampt.robotsim
from klampt import vis
from klampt.vis.glprogram import *
from klampt.vis.glprogram import GLNavigationProgram	#Per il
from klampt.sim import *
import importlib
import sys
import os
import random
from klampt.math import so3, se3
import string
import pydany_bb
import numpy as np
import math
from IPython import embed
from mvbb.graspvariation import PoseVariation
from mvbb.TakePoses import SimulationPoses
from mvbb.draw_bbox import draw_GL_frame, draw_bbox
from i16mc import make_object
import time

objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]

object_geom_file_patterns = {
	'ycb':['data/objects/ycb/%s/meshes/tsdf_mesh.stl','data/objects/ycb/%s/meshes/poisson_mesh.stl'],
	'apc2015':['data/objects/apc2015/%s/textured_meshes/optimized_tsdf_textured_mesh.ply']
}

class MVBBVisualizer(GLNavigationProgram):
    def __init__(self, poses, poses_variations, boxes, world, alt_trimesh = None):
        GLNavigationProgram.__init__(self, 'MVBB Visualizer')

        self.poses = poses
        self.poses_variations = poses_variations
        self.boxes = boxes
        self.obj = world.rigidObject(0)
        self.old_tm = self.obj.geometry().getTriangleMesh()
        self.new_tm = alt_trimesh
        self.robot = None
        self.world = world
        if world.numRobots() > 0:
            self.robot = world.robot(0)

        self.using_decimated_tm = False

    def invert_obj_color(self):
        color = self.obj.appearance().getColor()
        for i in range(3):
            color[i] = 1 - color[i]
        self.obj.appearance().setColor(*color)

    def display(self):
        if self.robot is not None:
            self.robot.drawGL()
        self.obj.drawGL()
        for pose in self.poses:
            T = se3.from_homogeneous(pose)
            draw_GL_frame(T)
        for box in self.boxes:
            draw_bbox(box.Isobox, box.T)

    def keyboardfunc(self, c, x, y):
        # Put your keyboard handler here
        # the current example toggles simulation / movie mode
        print c, "pressed"

        if c == 's' and self.new_tm is not None:
            self.using_decimated_tm = not self.using_decimated_tm
            print "Showing Decimated Trimesh", self.using_decimated_tm
            if self.using_decimated_tm:
                self.obj.geometry().setTriangleMesh(self.new_tm)
                self.invert_obj_color()
            else:
                self.obj.geometry().setTriangleMesh(self.old_tm)
                self.invert_obj_color()
        self.refresh()

    def idle(self):
        pass

def trimesh_to_numpy(klampt_TriangleMesh):
    tm = klampt_TriangleMesh
    n_vertices = tm.vertices.size() / 3
    n_faces = tm.indices.size() / 3
    vertices = np.zeros((n_vertices,3))
    faces = np.ndarray((n_faces, 3), dtype=np.intc)
    for i in range(n_vertices):
        vertices[i, :] = np.array([tm.vertices[3 * i], tm.vertices[3 * i + 1], tm.vertices[3 * i + 2]])
    for i in range(n_faces):
        faces[i, :] = np.array([tm.indices[3 * i], tm.indices[3 * i + 1], tm.indices[3 * i + 2]], dtype=np.intc)

    return vertices, faces

def numpy_to_trimesh(vertices, faces):
    tm = klampt.robotsim.TriangleMesh()
    for i in range(vertices.shape[0]):
        tm.vertices.append(vertices[i, 0])
        tm.vertices.append(vertices[i, 1])
        tm.vertices.append(vertices[i, 2])
    for i in range(faces.shape[0]):
        tm.indices.append(int(faces[i, 0]))
        tm.indices.append(int(faces[i, 1]))
        tm.indices.append(int(faces[i, 2]))
    return tm

def skip_decimate_or_return(object, min_vertices = 0, max_vertices = 2000):
    tm = object.geometry().getTriangleMesh()
    n_vertices = tm.vertices.size() / 3
    decimator = pydany_bb.MVBBDecimator()
    vertices_old, faces_old = trimesh_to_numpy(tm)
    if n_vertices <= min_vertices:
        return None, None
    if n_vertices > max_vertices:
        print "Object has", n_vertices, "vertices - decimating"
        decimator.decimateTriMesh(vertices_old, faces_old)
        vertices = decimator.getEigenVertices()
        faces = decimator.getEigenFaces()
        tm_decimated = numpy_to_trimesh(vertices, faces)
        print "Decimated to", vertices.shape[0], "vertices"
        return vertices, tm_decimated
    else:
        return object, None

def compute_poses(obj, new_method = False):
    if isinstance(obj, np.ndarray):
        vertices = obj
        n_vertices = vertices.shape[0]
        box = pydany_bb.Box(n_vertices)

        box.SetPoints(vertices)
    else:
        tm = obj.geometry().getTriangleMesh()
        n_vertices = tm.vertices.size() / 3
        box = pydany_bb.Box(n_vertices)

        for i in range(n_vertices):
            box.SetPoint(i, tm.vertices[3 * i], tm.vertices[3 * i + 1], tm.vertices[3 * i + 2])

    I = np.eye(4)
    print "doing PCA"
    box.doPCA(I)
    print box.T
    print "computing Bounding Box"
    bbox = pydany_bb.ComputeBoundingBox(box)
    p_0 = bbox.Isobox[0, :]
    p_1 = bbox.Isobox[1, :]
    long_side = np.max(np.abs(p_0 - p_1))
    print "Found Bounding Box:"
    print bbox.Isobox

    # rubbermaid_ice_guard_pitcher_blue
    #param_area = 0.95
    #param_volume = 4E-7
    param_area = 0.8
    param_volume = 5E-7

    print "extracting Boxes"
    boxes = pydany_bb.extractBoxes(bbox, param_area, param_volume)
    print "getting transforms"
    # new_method =  True
    poses = pydany_bb.getTransformsForHand(boxes, bbox, 0.005)

    poses_variations = []
    for pose in poses:
        poses_variations += PoseVariation(pose, long_side)
    for box in boxes:
        poses += pydany_bb.get_populated_TrasformsforHand(box, bbox, 2, .005)
    poses_total = poses + poses_variations
    # poses_sorted = sorted(poses_total, key=lambda pose:pose[2,3], reverse=True)
    # for posesss in poses_sorted:
    #     print 'pose', posesss 
    # poses_sorted = sorted(poses_total, key=lambda pose:(pose[2,3]-np.linalg.norm(pose[0:2,3])))
    poses_sorted = sorted(poses_total, key=lambda pose: pose[2,3]-np.linalg.norm(pose[0:2,3]), reverse=True)
    
    print "done. Found", len(poses_total), "poses,", len(boxes), "boxes"
    return poses_sorted, [], boxes

def launch_mvbb(object_set, objectname):
    """Launches a very simple program that spawns an object from one of the
    databases.
    It launches a visualization of the mvbb decomposition of the object, and corresponding generated poses.
    If use_box is True, then the test object is placed inside a box.
    """

    use_program = True

    world = WorldModel()
    world.loadElement("data/terrains/plane.env")
    object = make_object(object_set, objectname, world)
    R,t = object.getTransform()
    object.setTransform(R, [0, 0, 0])
    pattern = object_geom_file_patterns[object_set][0]
    tm = object.geometry().getTriangleMesh()
    n_vertices = tm.vertices.size() / 3
    decimator = pydany_bb.MVBBDecimator()
    vertices_old, faces_old = trimesh_to_numpy(tm)
    tm_decimated = None
    if n_vertices > 2000:
        print "Object has", n_vertices, "vertices - decimating"
        meshfile = pattern%(objectname,)
        decimator.decimateTriMesh(meshfile)
        #decimator.decimateTriMesh(vertices_old, faces_old)
        vertices = decimator.getEigenVertices()
        faces = decimator.getEigenFaces()
        tm_decimated = numpy_to_trimesh(vertices, faces)
        print "Decimated to", vertices.shape[0], "vertices"
        poses, poses_variations, boxes = compute_poses(vertices)
    else:
        poses, poses_variations, boxes = compute_poses(object)

    embed()
    # now the simulation is launched

    if use_program:
        program = MVBBVisualizer(poses, poses_variations, boxes, world, tm_decimated)
        vis.setPlugin(program)
        program.reshape(800, 600)
    else:
        vis.add("world", world)
    vis.show()

    # this code manually updates the visualization

    while vis.shown():
        if not use_program:
            vis.lock()
            # draw here
            time.sleep(0.01)
            vis.unlock()
        else:
            time.sleep(0.01)

    vis.kill()
    return

#************************************************Main******************************************

if __name__ == '__main__':
    import random
    try:
        dataset = sys.argv[1]
    except IndexError:
        dataset = random.choice(objects.keys())

    #just plan grasping
    try:
        index = int(sys.argv[2])
        objname = objects[dataset][index]
    except IndexError:
        index = random.randint(0,len(objects[dataset])-1)
        objname = objects[dataset][index]
    except ValueError:
        objname = sys.argv[2]

    print "loading object", index, " -", objname, "-from set", dataset

    launch_mvbb(dataset, objname)
