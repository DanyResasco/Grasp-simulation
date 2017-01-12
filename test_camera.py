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


objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
objects['newObjdany'] = [f for f in os.listdir('data/objects/newObjdany')]
objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]
robots = ['reflex_col', 'soft_hand', 'reflex']

def launch_test_mvbb_filtered(robotname, object_list, min_vertices = 0):
    """Launches a very simple program that spawns an object from one of the
    databases.
    It launches a visualization of the mvbb decomposition of the object, and corresponding generated poses.
    It then spawns a hand and tries all different poses to check for collision
    """

    world = WorldModel()
    world.readFile('sensortest.xml')
    # world.loadElement("data/terrains/plane.env")
    robot = make_moving_base_robot(robotname, world)
    xform = resource.get("default_initial_%s.xform" % robotname, description="Initial hand transform",
                         default=se3.identity(), world=world, doedit=False)

    for object_name in object_list:
        obj = None
        for object_set, objects_in_set in objects.items():
            if object_name in objects_in_set:
                if world.numRigidObjects() > 0:
                    world.remove(world.rigidObject(0))
                if object_name in objects['princeton']:
                    print "*************Dentro princeton********************" #need to scale the obj size
                    objfilename = 'data/objects/'+ object_set +'/'+ object_name + '/'+ object_name  +'.obj'
                    print"objfilename", objfilename
                    obj = DanyReduceScale(object_name, world,objfilename,object_set)
                else:    
                    obj = make_object(object_set, object_name, world)
        if obj is None:
            print "Could not find object", object_name
            continue


        R,t = obj.getTransform()
        # obj.setTransform(R, [t[0], t[1], t[2]]) #[0,0,0] or t?
        obj.setTransform(R, [0,0,0]) #[0,0,0] or t?

        #Try
        CameraSensor.get_measurement() 



if __name__ == '__main__':
    all_objects = []
    for dataset in objects.values():
        all_objects += dataset


    to_check = []
    done =[]
    to_filter=[]
    to_do=[]

    for obj_name in to_filter + to_do + done + to_check:
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