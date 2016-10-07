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
from mvbb.CollisionCheck import CheckCollision, CollisionTestInterpolate, CollisionTestPose


objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
robots = ['reflex_col', 'soft_hand', 'reflex']

class FilteredMVBBTesterVisualizer(GLRealtimeProgram):
    def __init__(self, poses, poses_variations, world, p_T_h, R, module):
        GLRealtimeProgram.__init__(self, "FilteredMVBBTEsterVisualizer")
        self.world = world
        self.p_T_h = p_T_h
        self.h_T_p = np.linalg.inv(self.p_T_h)
        self.poses = poses
        self.poses_variations = poses_variations
        self.R = R
        self.hand = None
        self.is_simulating = False
        self.curr_pose = None
        self.all_poses = self.poses + self.poses_variations
        self.robot = self.world.robot(0)
        self.q_0 = self.robot.getConfig()
        self.w_T_o = None
        self.obj = None
        self.t_0 = None
        self.object_com_z_0 = None
        self.object_fell = None
        self.sim = None
        self.module = module
        self.running = True

    def display(self):
        if self.running:
            self.world.drawGL()

            for pose in self.poses+self.poses_variations:
                T = se3.from_homogeneous(pose)
                draw_GL_frame(T, color=(0.5,0.5,0.5))
            if self.curr_pose is not None:
                T = se3.from_homogeneous(self.curr_pose)
                draw_GL_frame(T)

            hand_xform = get_moving_base_xform(self.robot)
            w_T_p_np = np.array(se3.homogeneous(hand_xform)).dot(self.h_T_p)
            w_T_p = se3.from_homogeneous(w_T_p_np)
            draw_GL_frame(w_T_p)

    def idle(self):
        if not self.running:
            return

        if self.world.numRigidObjects() > 0:
            self.obj = self.world.rigidObject(0)
        else:
            return

        if not self.is_simulating:
            if len(self.all_poses) > 0:
                self.curr_pose = self.all_poses.pop()
                print "Simulating Next Pose Grasp"
                print self.curr_pose
            else:
                print "Done testing all", len(self.poses+self.poses_variations), "poses for object", self.obj.getName()
                print "Quitting"
                self.running = False
                vis.show(hidden=True)
                return

            self.obj.setTransform(self.R, [0, 0, 0])
            self.w_T_o = np.array(se3.homogeneous(self.obj.getTransform()))
            pose_se3 = se3.from_homogeneous(self.w_T_o.dot(self.curr_pose).dot(self.p_T_h))
            self.robot.setConfig(self.q_0)
            set_moving_base_xform(self.robot, pose_se3[0], pose_se3[1])

            if self.sim is None:
                self.sim = SimpleSimulator(self.world)
                self.hand = self.module.HandEmulator(self.sim, 0, 6, 6)
                self.sim.addEmulator(0, self.hand)
                # the next line latches the current configuration in the PID controller...
                self.sim.controller(0).setPIDCommand(self.robot.getConfig(), self.robot.getVelocity())

            self.object_com_z_0 = getObjectGlobalCom(self.obj)[2]
            self.object_fell = False
            self.t_0 = self.sim.getTime()
            self.is_simulating = True

        if self.is_simulating:
            t_lift = 1.3 # when to lift
            d_lift = 1.0 # duration
            print "t:", self.sim.getTime() - self.t_0
            object_com_z = getObjectGlobalCom(self.obj)[2]
            if self.sim.getTime() - self.t_0 == 0:
                print "Closing hand"
                self.hand.setCommand([1.0])
            elif (self.sim.getTime() - self.t_0) >= t_lift and (self.sim.getTime() - self.t_0) <= t_lift+d_lift:
                print "Lifting"
                pose_se3 = se3.from_homogeneous(self.w_T_o.dot(self.curr_pose).dot(self.p_T_h))
                t_i = pose_se3[1]
                t_f = vectorops.add(t_i, (0,0,0.2))
                u = np.min((self.sim.getTime() - self.t_0 - t_lift, 1))
                send_moving_base_xform_PID(self.sim.controller(0), pose_se3[0], vectorops.interpolate(t_i, t_f ,u))


            if object_com_z < self.object_com_z_0 - 0.5:
                self.object_fell = True # TODO use grasp quality evaluator from Daniela

            self.sim.simulate(0.01)
            self.sim.updateWorld()

            if not vis.shown() or (self.sim.getTime() - self.t_0) >= 2.5 or self.object_fell:
                self.is_simulating = False
                self.sim = None

def getObjectGlobalCom(obj):
    return se3.apply(obj.getTransform(), obj.getMass().getCom())

def launch_test_mvbb_filtered(robotname, object_list, min_vertices = 0):
    """Launches a very simple program that spawns an object from one of the
    databases.
    It launches a visualization of the mvbb decomposition of the object, and corresponding generated poses.
    It then spawns a hand and tries all different poses to check for collision
    """

    world = WorldModel()
    world.loadElement("data/terrains/plane.env")
    robot = make_moving_base_robot(robotname, world)
    xform = resource.get("default_initial_%s.xform" % robotname, description="Initial hand transform",
                         default=se3.identity(), world=world, doedit=False)

    for object_name in object_list:
        obj = None
        for object_set, objects_in_set in objects.items():
            if object_name in objects_in_set:
                if world.numRigidObjects() > 0:
                    world.remove(world.rigidObject(0))
                obj = make_object(object_set, object_name, world)
        if obj is None:
            print "Could not find object", object_name
            continue


        R,t = obj.getTransform()
        obj.setTransform(R, [0, 0, 0])
        object_vertices_or_none, tm_decimated = skip_decimate_or_return(obj, min_vertices, 2000)
        if object_vertices_or_none is None:
            print "-------skipping object", obj.getName()
            continue
        object_or_vertices = object_vertices_or_none

        print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        print object_name
        print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

        print "------Computing poses for object:", object_name
        poses, poses_variations, boxes = compute_poses(object_or_vertices)

        w_T_o = np.array(se3.homogeneous((R,[0, 0, 0]))) # object is at origin

        p_T_h = np.array(se3.homogeneous(xform))

        poses_h = []
        poses_variations_h = []

        for i in range(len(poses)):
            poses_h.append(w_T_o.dot(poses[i]).dot(p_T_h))
        for i in range(len(poses_variations)):
            poses_variations_h.append(w_T_o.dot(poses_variations[i]).dot(p_T_h))

        print "-------Filtering poses:"
        filtered_poses = []
        for i in range(len(poses)):
            if not CollisionTestPose(world, robot, obj, poses_h[i]):
                filtered_poses.append(poses[i])
        filtered_poses_variations = []
        for i in range(len(poses_variations)):
            if not CollisionTestPose(world, robot, obj, poses_variations_h[i]):
                filtered_poses_variations.append(poses_variations[i])
        print "Filtered from", len(poses+poses_variations), "to", len(filtered_poses+filtered_poses_variations)
        if len(filtered_poses+filtered_poses_variations) == 0:
            print "Filtering returned 0 feasible poses"
            continue

        # create a hand emulator from the given robot name
        module = importlib.import_module('plugins.' + robotname)
        # emulator takes the robot index (0), start link index (6), and start driver index (6)

        program = FilteredMVBBTesterVisualizer(filtered_poses,
                                               filtered_poses_variations,
                                               world,
                                               p_T_h,
                                               R,
                                               module)

        vis.setPlugin(None)
        vis.setPlugin(program)
        program.reshape(800, 600)

        vis.show()
        # this code manually updates the visualization
        while vis.shown():
            time.sleep(0.1)
    return

if __name__ == '__main__':
    all_objects = []
    for dataset in objects.values():
        all_objects += dataset

    to_filter = ['stanley_13oz_hammer', # falls down
                 'red_wood_block_1inx1in', # too small TODO come up with a strategy for small objects
                 '2in_metal_washer', # too small TODO come up with a strategy for small objects
                 'blue_wood_block_1inx1in', # too small TODO come up with a strategy for small objects
                 'domino_sugar_1lb',  # cannot grasp box?
                 'expo_black_dry_erase_marker', # cannot grasp box?
                 'cheeze-it_388g', # cannot grasp box?
                 '1_and_a_half_in_metal_washer', # too small TODO come up with a strategy for small objects
                 'block_of_wood_6in', # canno grasp box, too big?
                 'sterilite_bin_12qt_cap', # what ?
                 'starkist_chunk_light_tuna',  # what?
                 'yellow_plastic_chain', # what?
                 'purple_wood_block_1inx1in', # TODO too small
                 'stainless_steel_spatula', # ?!?
                 ]
    to_do = [   'champion_sports_official_softball', # TODO grasp balls
                'penn_raquet_ball',                  # TODO grasp balls
                'wilson_100_tennis_ball',            # TODO grasp balls
                'wearever_cooking_pan_with_lid',     # TODO good handle, should be easy to grasp
                'rubbermaid_ice_guard_pitcher_blue', # TODO good handle, should be easy to grasp
                'jell-o_strawberry_gelatin_dessert',  # box, should be graspable
                ]
    done = [    'red_metal_bowl_white_speckles'] # effort_scaling = -0.5; sinergy_scaling = 11
    other =  [   'starkist_chunk_light_tuna']
    for obj_name in to_filter + to_do + done:
        all_objects.pop(all_objects.index(obj_name))

    print "-------------"
    print all_objects
    print "-------------"

    try:
        objname = sys.argv[1]
        launch_test_mvbb_filtered("soft_hand", [objname], 100)
    except:
        launch_test_mvbb_filtered("soft_hand", all_objects, 100)


