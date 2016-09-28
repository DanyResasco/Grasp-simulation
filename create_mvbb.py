#!/usr/bin/env python

from klampt import *
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

class MVBBVisualizer(GLNavigationProgram):
    def __init__(self, poses_variations, boxes, object):
        GLNavigationProgram.__init__(self, 'MVBB Visualizer')

        self.poses_variations = poses_variations
        self.boxes = boxes
        self.obj = object

    def display(self):
        #self.obj.drawGL()
        for pose in self.poses_variations:
            T = se3.from_homogeneous(pose)
            draw_GL_frame(T,10)
        for box in self.boxes:
            pass #draw_bbox(box.Isobox, box.T)

    def idle(self):
        pass

def compute_poses(obj):
    tm = obj.geometry().getTriangleMesh()
    n_vertices = tm.vertices.size() / 3
    box = pydany_bb.Box(n_vertices)

    for i in range(n_vertices):
        box.SetPoint(i, tm.vertices[3 * i], tm.vertices[3 * i + 1], tm.vertices[3 * i + 2])

    I = np.ones((4, 4))
    print "doing PCA"
    box.doPCA(I)
    print "computing Bounding Box"
    bbox = pydany_bb.ComputeBoundingBox(box)
    p_0 = bbox.Isobox[0, :]
    p_1 = bbox.Isobox[1, :]
    long_side = np.max(np.abs(p_0 - p_1))

    param_area = 0.98
    param_volume = 9E-6

    print "extracting Boxes"
    boxes = pydany_bb.extractBoxes(bbox, param_area, param_volume)
    print "getting transforms"
    poses = pydany_bb.getTrasformsforHand(boxes, bbox)

    poses_variations = []
    for pose in poses:
        poses_variations += PoseVariation(pose, long_side)

    print "done. Found", len(poses_variations), "poses"
    return poses, poses_variations, boxes

def launch_mvbb(object_set, objectname):
    """Launches a very simple program that spawns an object from one of the
    databases.
    It launches a visualization of the mvbb decomposition of the object, and corresponding generated poses.
    If use_box is True, then the test object is placed inside a box.
    """

    use_program = False

    world = WorldModel()
    world.loadElement("data/terrains/plane.env")
    object = make_object(object_set, objectname, world)

    poses, poses_variations, boxes = compute_poses(object)
    # now the simulation is launched

    if use_program:
        program = MVBBVisualizer(poses_variations, boxes, object)
        vis.setPlugin(program)
        program.reshape(800, 600)
    else:
        vis.add("world", world)
    vis.show()

    # this code manually updates the visualization

    while vis.shown():
        if not use_program:
            vis.lock()
            pose = poses_variations[0]
            T = se3.from_homogeneous(pose)
            draw_GL_frame(T, 1)
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

    launch_mvbb(dataset, objname)
