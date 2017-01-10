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
import pickle

from create_mvbb import MVBBVisualizer, compute_poses, skip_decimate_or_return
from create_mvbb_filtered import FilteredMVBBVisualizer
from klampt.math import so3, se3
import pydany_bb
import numpy as np
from IPython import embed
from mvbb.graspvariation import PoseVariation
from mvbb.TakePoses import SimulationPoses
from mvbb.draw_bbox import draw_GL_frame, draw_bbox
from i16mc import make_object, make_moving_base_robot
from mvbb.CollisionCheck import CheckCollision, CollisionTestInterpolate, CollisionTestPose, CollisionCheckWordFinger
from mvbb.db import MVBBLoader
from mvbb.kindness import Differential,RelativePosition
from mvbb.GetForces import get_contact_forces_and_jacobians
from mvbb.ScalaReduce import DanyReduceScale


objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
# objects['newObjdany'] = [f for f in os.listdir('data/objects/newObjdany')]
objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]
robots = ['reflex_col', 'soft_hand', 'reflex']

class PoseVisualizer(GLNavigationProgram):
    def __init__(self, poses,world,obj):
        GLNavigationProgram.__init__(self, 'PoseVisualizer')

        self.poses = poses
        # self.poses_variations = poses_variations
        # self.boxes = boxes
        self.obj = obj
        # self.old_tm = self.obj.geometry().getTriangleMesh()
        # self.new_tm = alt_trimesh
        # self.robot = None
        # self.world = world
        # if world.numRobots() > 0:
        #     self.robot = world.robot(0)

    def display(self):
        # if self.robot is not None:
        #     self.robot.drawGL()
        self.obj.drawGL()
        # T = se3.from_homogeneous(self.poses)
        draw_GL_frame(self.poses)
        # for box in self.boxes:
        #     draw_bbox(box.Isobox, box.T)


    def idle(self):
        pass





def launch_test_mvbb_filtered(robotname, object_list, min_vertices = 0):
    """Launches a very simple program that spawns an object from one of the
    databases.
    It launches a visualization of the mvbb decomposition of the object, and corresponding generated poses.
    It then spawns a hand and tries all different poses to check for collision
    """

    world = WorldModel()
    world.loadElement("data/terrains/plane.env")
    # world.readFile('sensor_test.xml')
    # robot = make_moving_base_robot(robotname, world)
    # xform = resource.get("default_initial_%s.xform" % robotname, description="Initial hand transform",
    #                      default=se3.identity(), world=world, doedit=False)

    for object_name in object_list:
        obj = None
        for object_set, objects_in_set in objects.items():
            if object_name in objects_in_set:
                if world.numRigidObjects() > 0:
                    world.remove(world.rigidObject(0))
                if object_name in objects['princeton']:
                    print "*************Dentro princeton********************" #need to scale the obj size
                    objfilename = 'data/objects/template_obj_scale_princeton.obj'
                    print"objfilename", objfilename
                    obj = DanyReduceScale(object_name, world,objfilename,object_set)
                else:
                    obj = make_object(object_set, object_name, world)
        if obj is None:
            print "Could not find object", object_name
            continue


    #obj.setTransform(R, [t[0], t[1], t[2]]) #[0,0,0] or t?
    #     obj.setTransform(R, [0,0,0]) #[0,0,0] or t?
        #now the simulation is launched
    # program = GLSimulationProgram(world)
    GLRealtimeProgram(world)
    # sim = program.sim
    sim = SimpleSimulator(world)
    sim.simulate(0)
    obj_p = obj.getTransform()
    # camera = (sim.controller(0).sensor('rgbd_camera')).getMeasurements()
    vis.setPlugin(PoseVisualizer(obj.getTransform(),world,obj))
    
    embed()
    
    
    vis.add("world",world)
    # vis.show()
    t0 = time.time()
    while vis.show():
        vis.lock()
        draw_GL_frame(obj.getTransform())
        vis.unlock()
        sim.simulate(0.01)
        sim.updateWorld()
        # vis.unlock()
        t1 = time.time()
        time.sleep(max(0.01-(t1-t0),0.001))
        t0 = t1



if __name__ == '__main__':
    all_objects = []
    for dataset in objects.values():
        all_objects += dataset


    to_filter = []
    done = []
    to_check = []


    for obj_name in to_filter +  done + to_check:
        all_objects.pop(all_objects.index(obj_name))

    print "-------------"
    print all_objects
    print "-------------"

    try:
        objname = sys.argv[1]
        # launch_test_mvbb_filtered("soft_hand", [objname], 100)
        launch_test_mvbb_filtered("reflex_col", [objname], 100)
    except:
        # launch_test_mvbb_filtered("soft_hand", all_objects, 100)
        launch_test_mvbb_filtered("reflex_col", all_objects, 100)


