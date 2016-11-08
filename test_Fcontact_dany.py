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
from mvbb.CollisionCheck import CheckCollision, CollisionTestInterpolate, CollisionTestPose, CollisionCheckWordFinger,CompenetrateCheckFinger
from mvbb.db import MVBBLoader
from mvbb.kindness import Differential,RelativePosition
from mvbb.ScalaReduce import DanyReduceScale
from mvbb.DanyLogFile import DanyLog
from mvbb.GetForces import get_contact_forces_and_jacobians


objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
objects['newObjdany'] = [f for f in os.listdir('data/objects/newObjdany')]
objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]
robots = ['reflex_col', 'soft_hand', 'reflex']

class FilteredMVBBTesterVisualizer(GLRealtimeProgram):
    def __init__(self, poses, poses_variations, world, p_T_h, R,t, module,PoseDanyDiff):
        GLRealtimeProgram.__init__(self, "FilteredMVBBTEsterVisualizer")
        self.world = world
        self.p_T_h = p_T_h
        self.h_T_p = np.linalg.inv(self.p_T_h)
        self.poses = poses
        self.poses_variations = poses_variations
        self.R = R
        self.T = t
        self.hand = None
        self.is_simulating = False
        self.curr_pose = None
        self.all_poses = self.poses + self.poses_variations
        self.robot = self.world.robot(0)
        self.q_0 = self.robot.getConfig()
        self.PoseDany = PoseDanyDiff
        self.w_T_o = None
        self.obj = None
        self.t_0 = None
        self.t0dany = None
        self.object_com_z_0 = None
        self.object_fell = None
        self.sim = None
        self.module = module
        self.running = True
        self.HandClose = False
        self.db = MVBBLoader(suffix='reflex')
        # self.logFile = DanyLog(suffix='logFile')
        self.kindness = None
        self.f1_contact = []
        self.f2_contact = []
        self.f3_contact = []
        self.crashing_states = []
        try:
            state = open('state.dump','r')
            self.crashing_states = pickle.load(state)
        except:
            pass

    def display(self):
        """ Draw a desired pose and the end-effector pose """
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

            self.obj.setTransform(self.R, [0,0,0])
            self.w_T_o = np.array(se3.homogeneous(self.obj.getTransform()))
            pose_se3 = se3.from_homogeneous(self.w_T_o.dot(self.curr_pose).dot(self.p_T_h))
            self.robot.setConfig(self.q_0)
            set_moving_base_xform(self.robot, pose_se3[0], pose_se3[1])

            if self.sim is None:
                self.sim = SimpleSimulator(self.world)
                self.sim.beginLogging()
                self.hand = self.module.HandEmulator(self.sim,0,6,6)
                self.sim.addEmulator(0, self.hand)
                # the next line latches the current configuration in the PID controller...
                self.sim.controller(0).setPIDCommand(self.robot.getConfig(), self.robot.getVelocity())

                # setup the preshrink
                visPreshrink = False  # turn this to true if you want to see the "shrunken" models used for collision detection
                for l in range(self.robot.numLinks()):
                    self.sim.body(self.robot.link(l)).setCollisionPreshrink(visPreshrink)
                for l in range(self.world.numRigidObjects()):
                    self.sim.body(self.world.rigidObject(l)).setCollisionPreshrink(visPreshrink)

            self.object_com_z_0 = getObjectGlobalCom(self.obj)[2]
            self.object_fell = False
            self.t_0 = self.sim.getTime()
            self.t0dany = self.sim.getTime()
            self.is_simulating = True

        if self.is_simulating:
            t_lift = 1.3 # when to lift
            d_lift = 1.0 # duration
            object_com_z = getObjectGlobalCom(self.obj)[2]
            hand_curr_pose = get_moving_base_xform(self.robot)
            pose_se3 = se3.from_homogeneous(self.w_T_o.dot(self.curr_pose).dot(self.p_T_h))

            if self.sim.getTime() - self.t_0 == 0:
                print "Closing hand"
                # self.hand.setCommand([0.2,0.2,0.2,0]) #TODO chiudila incrementalmente e controlla le forze di contatto
                hand_close = np.array([0.1,0.1,0.1,0])
                hand_open = np.array([1.0,1.0,1.0,0])
                step_size = 0.01
                while(self.HandClose == False):
                    d = vectorops.distance(hand_open, hand_close)
                    # print"d",d
                    n_steps = int(math.ceil(d / step_size))
                    # print"n_steps",n_steps
                    if n_steps == 0:    # if arrived
                        self.hand.setCommand([0.2,0.2,0.2,0])
                        self.HandClose = True
                    for i in range(n_steps):
                        hand_temp = vectorops.interpolate(hand_open,hand_close,float(i+1)/n_steps)
                        self.hand.setCommand([hand_temp[0] ,hand_temp[1] ,hand_temp[2] ,0])
                        self.sim.simulate(0.01)
                        self.sim.updateWorld()
                        FC = get_contact_forces_and_jacobians(self.robot,self.world,self.sim)
                        if hand_temp[0] <= hand_close[0] and hand_temp[1] <= hand_close[1] and hand_temp[2] <= hand_close[2]:
                            print"qui"
                            self.HandClose = True
                            break

            elif (self.sim.getTime() - self.t_0) >= t_lift and (self.sim.getTime() - self.t_0) <= t_lift+d_lift:
                print "Lifting"
                pose_se3 = se3.from_homogeneous(self.w_T_o.dot(self.curr_pose).dot(self.p_T_h))
                t_i = pose_se3[1]
                t_f = vectorops.add(t_i, (0,0,0.2))
                u = np.min((self.sim.getTime() - self.t_0 - t_lift, 1))
                send_moving_base_xform_PID(self.sim.controller(0), pose_se3[0], vectorops.interpolate(t_i, t_f ,u))
                timeDany = self.sim.getTime() - self.t_0
                self.kindness = Differential(self.robot, self.obj, self.PoseDany, timeDany)
                self.PoseDany = RelativePosition(self.robot, self.obj)

            if (self.sim.getTime() - self.t_0) >= t_lift and (self.sim.getTime() - self.t_0) >= t_lift+d_lift:# wait for a lift before checking if object fell
                d_hand = hand_curr_pose[1][2] - pose_se3[1][2]
                d_com = object_com_z - self.object_com_z_0
                if (d_hand - d_com > 0.1) and (self.kindness > 1E-4):
                    self.object_fell = True # TODO use grasp quality evaluator from Daniela
                    print "!!!!!!!!!!!!!!!!!!"
                    print "Object fell"
                    print "!!!!!!!!!!!!!!!!!!"

            self.sim.simulate(0.01)
            self.sim.updateWorld()

            if not vis.shown() or (self.sim.getTime() - self.t_0) >= 2.5 or self.object_fell:
                if vis.shown(): # simulation stopped because it was succesfull
                    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    print "Saving grasp, object fall status:", "fallen" if self.object_fell else "grasped"
                    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    self.db.save_score(self.world.rigidObject(0).getName(), self.curr_pose, not self.object_fell,self.kindness)
                    # self.logFile.save_score(self.world.rigidObject(0).getName(), self.curr_pose, not self.object_fell,self.obj.getVelocity(), self.robot.getVelocity(), self.f1_contact,self.f2_contact,self.f3_contact)
                    if len(self.crashing_states) > 0:
                        self.crashing_states.pop()
                    state = open('state.dump','w')
                    pickle.dump(self.crashing_states, state)
                    state.close()
                self.is_simulating = False
                self.sim = None
                self.HandClose = False

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
        object_vertices_or_none, tm_decimated = skip_decimate_or_return(obj, min_vertices, 2000)
        if object_vertices_or_none is None:
            print "??????????????????????????????????????????????????"
            print "??????????????????????????????????????????????????"
            print "??????????????????????????????????????????????????"
            print "skipping object, too few vertices", obj.getName()
            print "??????????????????????????????????????????????????"
            print "??????????????????????????????????????????????????"
            print "??????????????????????????????????????????????????"
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

        # aa = so3.axis_angle(so3.identity())
        Ry = np.array(se3.homogeneous((so3.from_axis_angle(((0,1,0), 45.*np.pi/180.)),[0,0,0])))
        Rx = np.array(se3.homogeneous((so3.from_axis_angle(((1,0,0), 45.*np.pi/180.)),[0,0,0])))
        Rz = np.array(se3.homogeneous((so3.from_axis_angle(((0,0,1), 45.*np.pi/180.)),[0,0,0])))
        Tx = np.array(se3.homogeneous((so3.identity(), [-.0,0,0])))
        T = Tx.dot(Rz).dot(Rx).dot(Rx) # object is at origin)
        T = Rz;

        poses_new = []

        for pose in poses:
            poses_new.append(pose.dot(T));
        poses = poses_new


        # w_T_o = np.array(se3.homogeneous((R,[t[0], t[1], t[2]]))) # object is at origin
        w_T_o = np.array(se3.homogeneous((R,[0,0,0]))) # object is at origin

        p_T_h = np.array(se3.homogeneous(xform))

        poses_h = []
        poses_variations_h = []

        for i in range(len(poses)):
            poses_h.append((w_T_o.dot(poses[i]).dot(p_T_h)))
        for i in range(len(poses_variations)):
            poses_variations_h.append((w_T_o.dot(poses_variations[i]).dot(p_T_h)))

        print "-------Filtering poses:"
        filtered_poses = []
        for i in range(len(poses)):
            if not CollisionTestPose(world, robot, obj, poses_h[i]):
                # if not  CompenetrateCheckFinger(robot, obj,poses_h[i]):
                # print "No collision wit obj. check the finger. first check"
                if not CollisionCheckWordFinger(robot, poses_h[i]):
                    # print "no collision with finger. first check"
                    filtered_poses.append(poses[i])
        filtered_poses_variations = []
        for i in range(len(poses_variations)):
            if not CollisionTestPose(world, robot, obj, poses_variations_h[i]):
                # if not  CompenetrateCheckFinger(robot, obj,poses_variations_h[i]):
                # print "No collision wit obj. check the finger. second check"
                if not CollisionCheckWordFinger(robot, poses_variations_h[i]):
                # print "no collision with finger. second check"
                    filtered_poses_variations.append(poses_variations[i])
        print "Filtered from", len(poses+poses_variations), "to", len(filtered_poses+filtered_poses_variations)
        if len(filtered_poses+filtered_poses_variations) == 0:
            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print "Filtering returned 0 feasible poses"
            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            continue

        # create a hand emulator from the given robot name
        module = importlib.import_module('plugins.' + robotname)
        # emulator takes the robot index (0), start link index (6), and start driver index (6)
        PoseDanyDiff = RelativePosition(robot,obj)
        program = FilteredMVBBTesterVisualizer(filtered_poses,
                                               filtered_poses_variations,
                                               world,
                                               p_T_h,
                                               R,
                                               t,
                                               module,
                                               PoseDanyDiff)

        vis.setPlugin(None)
        vis.setPlugin(program)
        program.reshape(800, 600)

        vis.show()
        # this code manually updates the visualization
        t0= time.time()
        while vis.shown():
            # time.sleep(0.1)
            t1 = time.time()
            time.sleep(max(0.01-(t1-t0),0.001))
            t0 = t1
    return

if __name__ == '__main__':
    all_objects = []
    for dataset in objects.values():
        all_objects += dataset


    to_check = [
    #ycb and acp
    'soft_scrub_2lb_4oz', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140027378132736)>> Traceback (most recent call last)
    'black_and_decker_lithium_drill_driver_unboxed', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140577888667392)>>Closing ODE...
    '1_and_a_half_in_metal_washer', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'melissa_doug_farm_fresh_fruit_strawberry', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h].Aborted (core dumped)
    'yellow_plastic_chain', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h].Aborted (core dumped)
    'starkist_chunk_light_tuna', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h].Aborted (core dumped)
    'melissa_doug_farm_fresh_fruit_plum', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h].Aborted (core dumped)
    'play_go_rainbow_stakin_cups_9_red', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h].Aborted (core dumped)
    'stainless_steel_spoon_red_handle', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 139856490284800)>> Traceback (most recent call last):
    'comet_lemon_fresh_bleach', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140621720352512)>>Traceback (most recent call last):   File "/usr/lib/python2.7/threading.py", line 783, in __bootstrap Closing ODE...
    'stainless_steel_spatula', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h].Aborted (core dumped)
    'cheeze-it_388g',# Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140356403099392)>> Traceback (most recent call last): Closing ODE...
    'moutain_security_steel_shackle', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h].Aborted (core dumped) 
     'wescott_orange_grey_scissors', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 139634373859072)>>Traceback (most recent call last): Closing ODE...
    'purple_wood_block_1inx1in', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h].Aborted (core dumped) 
    'stainless_steel_fork_red_handle', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140021259945728)>>Traceback (most recent call last): Closing ODE...
    'play_go_rainbow_stakin_cups_10_blue' ,#ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h].Aborted (core dumped)
    'plastic_bolt_grey', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'play_go_rainbow_stakin_cups_3_red',# ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h]
    'sharpie_marker', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140103165626112)>>.Traceback (most recent call last):Closing ODE...
    'sterilite_bin_12qt_bottom', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140004478109440)>>Traceback (most recent call last):File "/usr/lib/python2.7/threading.py", line 783, in __bootstrap Closing ODE...
    'dark_red_foam_block_with_three_holes', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h]Aborted (core dumped)
    'melissa_doug_farm_fresh_fruit_lemon', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h].Aborted (core dumped) PID
    'master_chef_ground_coffee_297g', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140315041806080)>>Traceback (most recent call last):File "/usr/lib/python2.7/threading.py", line 783, in __bootstrap Closing ODE...
    'jell-o_chocolate_flavor_pudding', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140119563753216)>>Traceback (most recent call last):Closing ODE...
    'morton_salt_shaker', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 139735494244096)>>Traceback (most recent call last):  File "/usr/lib/python2.7/threading.py", line 783, in __bootstrap
    'wilson_golf_ball', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h]Aborted (core dumped) PID
    'white_rope', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h]. Aborted (core dumped)
    'red_metal_cup_white_speckles',#Exception in thread Thread-1 (most likely raised during interpreter shutdown):Closing ODE...
    'cheerios_14oz', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140367152363264)>>Traceback (most recent call last):  File "/usr/lib/python2.7/threading.py", line 783, in __bootstrap
    'blue_wood_block_1inx1in', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h]Aborted (core dumped)
    'jell-o_strawberry_gelatin_dessert', #ODECustomMesh: Triangles penetrate margin 0+0.0025: can't trust contact detector
    'champion_sports_official_softball', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140416335058688)>>Closing ODE...
    'clorox_disinfecting_wipes_35', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140221284226816)>>Closing ODE...
    'large_black_spring_clamp', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'melissa_doug_farm_fresh_fruit_apple', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140038930360064)>>Closing ODE...
    'frenchs_classic_yellow_mustard_14oz', # Exception in thread Thread-1 (most likely raised during interpreter shutdown): Closing ODE...
    'windex', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140105912764160)>>Traceback (most recent call last): Closing ODE...
    'stainless_steel_knife_red_handle', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140019860342528)>>Traceback (most recent call last): Closing ODE...
    '2in_metal_washer', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140326007203584)>>Closing ODE... BUT end-effecto fails
    'black_and_decker_lithium_drill_driver', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 139996229932800)>>Traceback (most recent call last): Closing ODE...
    'red_metal_bowl_white_speckles', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h]Aborted (core dumped)
    'wearever_cooking_pan_with_lid', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'block_of_wood_12in', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 139916506646272)>>Traceback (most recent call last): Closing ODE...
    'melissa_doug_play-time_produce_farm_fresh_fruit_unopened_box', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'medium_black_spring_clamp',#Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140373540939520)>> Traceback (most recent call last): File "/usr/lib/python2.7/threading.py", line 783, in __bootstrap Closing ODE...
    'red_metal_plate_white_speckles', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'moutain_security_steel_shackle_key', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'play_go_rainbow_stakin_cups_7_yellow', # Exception in thread Thread-1 (most likely raised during interpreter shutdown):Closing ODE...
    'rubbermaid_ice_guard_pitcher_blue', # python: ../nptl/pthread_mutex_lock.c:80: __pthread_mutex_lock: Assertion `mutex->__data.__owner == 0' failed. Aborted (core dumped)
    'melissa_doug_farm_fresh_fruit_pear', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140112034821888)>>Closing ODE...
    'domino_sugar_1lb', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140651650115328)>>Closing ODE...
    'yellow_wood_block_1inx1in', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'stanley_13oz_hammer', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140154012018432)>>Closing ODE...
    'penn_raquet_ball', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'melissa_doug_farm_fresh_fruit_peach', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'play_go_rainbow_stakin_cups_6_purple', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'morton_pepper_shaker', # segmentation fault
    'play_go_rainbow_stakin_cups_5_green', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'red_wood_block_1inx1in', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'melissa_doug_farm_fresh_fruit_banana', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'first_years_take_and_toss_straw_cups', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'kong_duck_dog_toy', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'oreo_mega_stuf', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 139803403032320)>> Traceback (most recent call last): File "/usr/lib/python2.7/threading.py", line 783, in __bootstrap self.__bootstrap_inner() Closing ODE...
    'crayola_64_ct', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'mommys_helper_outlet_plugs', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 139850326169344)>>Traceback (most recent call last):   File "/usr/lib/python2.7/threading.py", line 783, in __bootstrap Closing ODE...
    'stanley_66_052', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID 
    'feline_greenies_dental_treats', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140115112707840)>>Traceback (most recent call last): Closing ODE...
    'expo_dry_erase_board_eraser', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'cheezit_big_original', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 139825170249472)>>Traceback (most recent call last): Closing ODE...
    'genuine_joe_plastic_stir_sticks', # Exception in thread Thread-1 (most likely raised during interpreter shutdown):Closing ODE...
    'kong_sitting_frog_dog_toy', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h]Aborted (core dumped)
    'safety_works_safety_glasses', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h]Aborted (core dumped)
    'kong_air_dog_squeakair_tennis_ball', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'highland_6539_self_stick_notes', #Exception in thread Thread-1 (most likely raised during interpreter shutdown):Closing ODE...
    'mead_index_cards', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'mark_twain_huckleberry_finn', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140645453305600)>>Closing ODE...
    'rollodex_mesh_collection_jumbo_pencil_cup', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'laugh_out_loud_joke_book', # Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140682578249472)>> Traceback (most recent call last): Closing ODE...
    'munchkin_white_hot_duck_bath_toy', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'kygen_squeakin_eggs_plush_puppies', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'paper_mate_12_count_mirado_black_warrior', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    #newobjdany
    "juicerB",#ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'bowA', #ee compenetrate with object. ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h]Aborted (core dumped)
    'bowlB', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'containerA', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140681384974080)>> Traceback (most recent call last): File "/usr/lib/python2.7/threading.py", line 783, in __bootstrap self.__bootstrap_inner() Closing ODE...
    'containerC', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'kitchenUtensilD', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'kitchenUtensilA', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    # 'kitchenUtensilB ',# ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'kitchenUtensilC', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'kitchenUtensilE',#ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'kitchenUtensilF', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'kitchenUtensilG', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h]Aborted (core dumped)
    'mugD', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140436462683904)>>Traceback (most recent call last): Closing ODE...
    'panA', #obj compenetrate with table
    'pot', #
    #princeton
    'antenna', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 139743886427904)>> Traceback (most recent call last): File "/usr/lib/python2.7/threading.py", line 783, in __bootstrap Closing ODE...
    'fireplace', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'arch', # ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'cashmachine', #segmentation fault
    'cashmachinewhite', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'cashmachinebig', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140511671895808)>> Traceback (most recent call last):   File "/usr/lib/python2.7/threading.py", line 783, in __bootstrap Closing ODE...
    'bigparabola', #ee compenetrate
    'blackparabola', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'parabola', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 139625796851456)>>Traceback (most recent call last):File "/usr/lib/python2.7/threading.py", line 783, in __bootstrap     self.__bootstrap_inner() Closing ODE...
    'sink', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 139788023400192)>>Traceback (most recent call last): Closing ODE...
    'squaresink', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h]Aborted (core dumped)
    'whitecontainer', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 139695596787456)>> Traceback (most recent call last): Closing ODE...
    'chest', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140371207259904)>> Traceback (most recent call last): File "/usr/lib/python2.7/threading.py", line 783, in __bootstrap self.__bootstrap_inner() Closing ODE...
    'openchest', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'whitedoor', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'whitebigdoor', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped) PID
    'graybook', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140014212699904)>>Closing ODE...
    'ropebridge', #problem, ee disapear
    'chinesbridge', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'redandgreenbridge', #ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)
    'whitebigbridge', #Unhandled exception in thread started by <bound method Thread.__bootstrap of <Thread(Thread-1, stopped daemon 140408517834496)>> Traceback (most recent call last): Closing ODE...
    'longwhitebridge',#ODE INTERNAL ERROR 1: assertion "bNormalizationResult" failed in _dNormalize4() [../../include/ode/odemath.h] Aborted (core dumped)

    ]
    to_filter = [
    #ycb and acp
    'play_go_rainbow_stakin_cups_2_orange', #zero feasible poses found
    'spam_12oz', #zero poses found
    '1in_metal_washer', #zero feasible poses found
    'stanley_flathead_screwdriver', #zero feasible poses found
    'plastic_nut_grey', #too small
    'small_black_spring_clamp', #end-effector fails
    'campbells_condensed_tomato_soup', #zero feasible poses found
    'blank_hard_plastic_card', #zero feasible poses found
    'plastic_wine_cup', #zero feasible poses found
    'extra_small_black_spring_clamp',#zero feasible poses found
    'orange_wood_block_1inx1in',#zero feasible poses found   
    'play_go_rainbow_stakin_cups_1_yellow',#one poses and objct fall
    'expo_black_dry_erase_marker_fine',#zero feasible poses found
    'expo_black_dry_erase_marker', #zero feasible poses found
    'champion_copper_plus_spark_plug', # zero feasible poses found
    'sharpie_accent_tank_style_highlighters', # zero feasible poses found
    'dove_beauty_bar', #zero feasible poses found
    'one_with_nature_soap_dead_sea_mud', #zero feasible poses found
    'fireplace2', # too few vertices
    'brownchest', #too few vertices
    'brownandyellowchest', #too few vertices
    'openbrownchest', # zero pose found
    'door', #too few vertices
    'browndoor', #too few vertices
    'graydoor', ##too few vertices
    'blackdoor', #too few vertices
    'whitefireplace', #zero poses found
    'book', #too few verticies
    'redbridge', #zero poses found
    'doorwithwindow', #to few vertices
    'blackdoorwithwindow', #error linea 19 in moving_base_control


    ]
    to_do = []
    done = [
    #ycb and acp
    'pringles_original',
    'brine_mini_soccer_ball',
    'learning_resources_one-inch_color_cubes_box',
    'play_go_rainbow_stakin_cups_blue_4',
    'sponge_with_textured_cover',
    'play_go_rainbow_stakin_cups_box',#rfallo il pc si e' bloccato
    'melissa_doug_farm_fresh_fruit_orange',
    'play_go_rainbow_stakin_cups_8_orange',
    'wilson_100_tennis_ball',
    'thick_wood_block_6in', 
    'sterilite_bin_12qt_cap', #but ee fails
    'stanley_philips_screwdriver',
    'block_of_wood_6in',
    'elmers_washable_no_run_school_glue',
    'dr_browns_bottle_brush',
    #NewObjdany
    'containerB',
    'jug',
    'containerD',
    'wc',
    'brownfireplace',
    'blackfireplace', #only one poses and ee fail
    'swing',
    'whiteswing',
    'grayswing', #oneposes and ee fail
    'redswing',
    'yellowbook',
    'twocilinder'
    
    ]

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

