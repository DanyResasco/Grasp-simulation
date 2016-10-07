#!/usr/bin/env python

import pkg_resources
pkg_resources.require("klampt>=0.7.0")
from klampt import *
from klampt import vis 
from klampt.vis.glrobotprogram import *
from klampt.math import *
from klampt.model import collide
from klampt.io import resource
from klampt.sim import *
from moving_base_control import *
import importlib
import math
import os
import string
import sys
import time

from create_mvbb import MVBBVisualizer, trimesh_to_numpy, numpy_to_trimesh, compute_poses, skip_decimate_or_return
from klampt.math import so3, se3
import pydany_bb
import numpy as np
from IPython import embed
from mvbb.graspvariation import PoseVariation
from mvbb.TakePoses import SimulationPoses
from mvbb.db import MVBBLoader
from mvbb.draw_bbox import draw_GL_frame, draw_bbox
from i16mc import make_object, make_moving_base_robot
from mvbb.CollisionCheck import CheckCollision, CollisionTestInterpolate, CollisionTestPose


objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
robots = ['reflex_col', 'soft_hand', 'reflex']

object_geom_file_patterns = {
    'ycb':['data/objects/ycb/%s/meshes/tsdf_mesh.stl','data/objects/ycb/%s/meshes/poisson_mesh.stl'],
    'apc2015':['data/objects/apc2015/%s/textured_meshes/optimized_tsdf_textured_mesh.ply']
}

robot_files = {
    'reflex_col':'data/robots/reflex_col.rob',
    'soft_hand':'data/robots/soft_hand.urdf',
    'reflex':'data/robots/reflex.rob'
}

class FilteredMVBBVisualizer(MVBBVisualizer):
    def __init__(self, poses, poses_variations, boxes, world, xform, alt_trimesh = None):
        MVBBVisualizer.__init__(self, poses, poses_variations, boxes, world, alt_trimesh)
        self.xform = xform

    def display(self):
        T0 = get_moving_base_xform(self.robot)
        T0_invisible = T0
        T0_invisible[1][2] = -1
        set_moving_base_xform(self.robot,T0_invisible[0], T0_invisible[1])
        self.world.drawGL()
        set_moving_base_xform(self.robot, T0[0], T0[1])
        self.obj.drawGL()
        for pose in self.poses:
            T = se3.from_homogeneous(pose)
            T_coll = pose.dot(np.array(se3.homogeneous(self.xform)))
            if not CollisionTestPose(self.world, self.robot, self.obj, T_coll):
                draw_GL_frame(T)
                #set_moving_base_xform(self.robot, T[0], T[1])
                #self.robot.drawGL()
            else:
                "robot collides with object at", T_coll
                draw_GL_frame(T, color=(0.5,0.5,0.5))
        for box in self.boxes:
            draw_bbox(box.Isobox, box.T)
        set_moving_base_xform(self.robot, T0[0], T0[1])

def kdltonumpy3(R):
    npR = np.eye(3)
    for i in range(3):
        for j in range(3):
            npR[i,j] = R[i,j]
    return npR

def kdltonumpy4(F):
    npF = np.eye(4)
    npF[0:3,0:3] = kdltonumpy3(F.M)
    for i in range(3):
        npF[i,3] = F.p[i]
    return npF

def launch_mvbb_filtered(robotname, object_set, objectname):
    """Launches a very simple program that spawns an object from one of the
    databases.
    It launches a visualization of the mvbb decomposition of the object, and corresponding generated poses.
    It then spawns a hand and tries all different poses to check for collision
    """

    world = WorldModel()
    world.loadElement("data/terrains/plane.env")
    robot = make_moving_base_robot(robotname, world)
    object = make_object(object_set, objectname, world)

    R,t = object.getTransform()
    object.setTransform(R, [0, 0, 0])

    db = MVBBLoader()
    loaded_poses = db.get_poses(object.getName())
    loaded_poses = []

    if len(loaded_poses) > 0:
        tm_decimated = None
        poses, poses_variations, boxes = (loaded_poses, [], [])
    else:
        object_vertices_or_none, tm_decimated = skip_decimate_or_return(object)
        poses, poses_variations, boxes = compute_poses(object_vertices_or_none)

    # now the simulation is launched
    xform = resource.get("default_initial_%s.xform" % robotname, description="Initial hand transform",
                         default=se3.identity(), world=world, doedit=False)

    p_T_h = np.array(se3.homogeneous(xform))

    poses_h = []
    poses_variations_h = []

    for i in range(len(poses)):
        poses_h.append(poses[i].dot(p_T_h))
    for i in range(len(poses_variations)):
        poses_variations_h.append(poses_variations[i].dot(p_T_h))

    filtered_poses = []
    for i in range(len(poses)):
        if not CollisionTestPose(world, robot, object, poses_h[i]):
            filtered_poses.append(poses[i])
    filtered_poses_variations = []
    for i in range(len(poses_variations)):
        if not CollisionTestPose(world, robot, object, poses_variations_h[i]):
            filtered_poses_variations.append(poses_variations[i])
    print "Filtered from", len(poses + poses_variations), "to", len(filtered_poses + filtered_poses_variations)
    if len(filtered_poses + filtered_poses_variations) == 0:
        print "Filtering returned 0 feasible poses"

    program = FilteredMVBBVisualizer(poses, poses_variations, boxes, world, xform, tm_decimated)
    vis.setPlugin(program)
    program.reshape(800, 600)

    vis.show()

    # this code manually updates the visualization

    while vis.shown():
        time.sleep(0.01)

    vis.kill()
    return

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

    launch_mvbb_filtered("soft_hand", dataset, objname)
