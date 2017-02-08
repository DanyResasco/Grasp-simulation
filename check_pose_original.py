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
import csv
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
from dany_make_rotate_voxel import make_objectRotate



'''Simulation of rotation mesh'''




objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
objects['newObjdany'] = [f for f in os.listdir('data/objects/newObjdany')]
objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]
robots = ['reflex_col']

done = []

class TesterGrab(GLRealtimeProgram):
    def __init__(self, poses, world,p_T_h,R,T, PoseDanyDiff,module):
        GLRealtimeProgram.__init__(self, "FilteredMVBBTEsterVisualizer")
        self.world = world
        self.poses = poses
        self.p_T_h = p_T_h
        self.h_T_p = np.linalg.inv(self.p_T_h)
        self.hand = None
        self.is_simulating = False
        self.curr_pose = None
        self.R = R
        self.t =T
        self.robot = self.world.robot(0)
        self.q_0 = self.robot.getConfig()
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
        self.db = MVBBLoader(suffix='test_pose')
        # self.logFile = DanyLog(suffix='logFile')
        self.kindness = None
        self.f1_contact = []
        self.f2_contact = []
        self.f3_contact = []
        self.crashing_states = []
        self.PoseDany = PoseDanyDiff
        self.danyK = []

        try:
            state = open('state.dump','r')
            self.crashing_states = pickle.load(state)
        except:
            pass

    def display(self):
        """ Draw a desired pose and the end-effector pose """
        if self.running:
            self.world.drawGL()

            # for i in range(len(self.poses)):
            for pose in self.poses:
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
        elif self.obj is None:
            return

        if not self.is_simulating:
            if len(self.poses) > 0:
                # self.curr_pose = np.array(se3.homogeneous(self.poses.pop(0)))
                self.curr_pose = self.poses.pop(0)
                # vis.show(hidden=False)
                # print "Simulating Next Pose Grasp"
                # print self.curr_pose
            else:
                # print "Done testing all", len(self.poses+self.poses_variations), "poses for object", self.obj.getName()
                # print "Quitting"
                self.running = False
                vis.show(hidden=True)
                return

            self.obj.setTransform(self.R, self.t)
            self.obj.setVelocity([0., 0., 0.],[0., 0., 0.])
            # self.obj.setVelocity([0,0,0,e0])
            self.w_T_o = np.array(se3.homogeneous(self.obj.getTransform()))
            # embed()
            pose_se3 = se3.from_homogeneous(self.w_T_o.dot(self.curr_pose).dot(self.p_T_h))
            self.robot.setConfig(self.q_0)
            set_moving_base_xform(self.robot, pose_se3[0], pose_se3[1])

            if self.sim is None:
                self.sim = SimpleSimulator(self.world)
                self.sim.enableContactFeedbackAll()
                ##uncomment to see the log file
                # n = len(self.poses)+len(self.poses_variations) - len(self.all_poses)
                # self.sim.log_state_fn="simulation_state_" + self.obj.getName() + "_%03d"%n + ".csv"
                # self.sim.log_contact_fn="simulation_contact_"+ self.obj.getName() + "_%03d"%n + ".csv"
                # self.sim.beginLogging()

                self.hand = self.module.HandEmulator(self.sim,0,6,6)
                self.sim.addEmulator(0, self.hand)
                # self.obj.setVelocity([0., 0., 0.],[0., 0., 0.])
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
                        n = len(self.poses)
                        # print FC
                        # +len(self.poses_variations) - len(self.all_poses)
                        # print"pose", n, "contact forces@t:", self.sim.getTime(), "-", FC
                        if hand_temp[0] <= hand_close[0] and hand_temp[1] <= hand_close[1] and hand_temp[2] <= hand_close[2]:
                            # print"qui"
                            self.HandClose = True
                            break

            elif (self.sim.getTime() - self.t_0) >= t_lift and (self.sim.getTime() - self.t_0) <= t_lift+d_lift:
                # print "Lifting"
                pose_se3 = se3.from_homogeneous(self.w_T_o.dot(self.curr_pose).dot(self.p_T_h))
                t_i = pose_se3[1]
                t_f = vectorops.add(t_i, (0,0,0.2))
                u = np.min((self.sim.getTime() - self.t_0 - t_lift, 1))
                send_moving_base_xform_PID(self.sim.controller(0), pose_se3[0], vectorops.interpolate(t_i, t_f ,u))
                timeDany = self.sim.getTime() - self.t_0
                self.kindness = Differential(self.robot, self.obj, self.PoseDany, timeDany)
                # print self.kindness
                self.danyK.append(self.kindness)
                self.PoseDany = RelativePosition(self.robot, self.obj)

            if (self.sim.getTime() - self.t_0) >= t_lift and (self.sim.getTime() - self.t_0) >= t_lift+d_lift:# wait for a lift before checking if object fell
                d_hand = hand_curr_pose[1][2] - pose_se3[1][2]
                d_com = object_com_z - self.object_com_z_0
                if (d_hand - d_com > 0.1) and (self.kindness >= 1E-4):
                    self.object_fell = True # TODO use grasp quality evaluator from Daniela
                    print "!!!!!!!!!!!!!!!!!!"
                    print "Object fell"
                    print "!!!!!!!!!!!!!!!!!!"
                    # Draw_Grasph(self.danyK)
                    # del self.danyK
                # else:
                #     Draw_Grasph(self.danyK)
                #     del self.danyK

            self.sim.simulate(0.01)
            self.sim.updateWorld()

            if not vis.shown() or (self.sim.getTime() - self.t_0) >= 2.5 or self.object_fell:
                if vis.shown(): # simulation stopped because it was succesfull
                    # print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                    # print "Saving grasp, object fall status:", "fallen" if self.object_fell else "grasped"
                    # print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

                    self.db.save_score(self.world.rigidObject(0).getName(), self.curr_pose, not self.object_fell,self.kindness)
                    # self.logFile.save_score(self.world.rigidObject(0).getName(), self.curr_pose, not self.object_fell,self.obj.getVelocity(), self.robot.getVelocity(), self.f1_contact,self.f2_contact,self.f3_contact)
                    if len(self.crashing_states) > 0:
                        self.crashing_states.pop()
                    state = open('state.dump','w')
                    pickle.dump(self.crashing_states, state)
                    state.close()
                # vis.show(hidden=True)
                self.is_simulating = False
                self.sim = None
                self.HandClose = False

def getObjectGlobalCom(obj):
    return se3.apply(obj.getTransform(), obj.getMass().getCom())


def Read_Poses(nome):

    obj_dataset = '3DCNN/NNSet/Pose/pose/%s.csv'%(nome)
    vector_set = []
    with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
        file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
        for row in file_reader:
            T = row[9:12]
            pp = row[:9]
            # embed()
            vector_set.append(np.array(se3.homogeneous((pp,T))))
            # break
    return vector_set


def launch_test_mvbb_filtered(robotname, object_list):
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
            # print objects_in_set
            if object_name in objects_in_set:
                    if object_name in objects['princeton']:
                        # print "*************Dentro princeton********************" #need to scale the obj size
                        objfilename = 'data/objects/template_obj_scale_princeton.obj'
                        # print"objfilename", objfilename
                        obj = DanyReduceScale(object_name, world,objfilename,object_set)
                    else:    
                        obj = make_object(object_set, object_name, world)
                    
                    # poses = [] #o_T_h
                    poses = Read_Poses(object_name)
                    if obj is None:
                        continue
                    R,t = obj.getTransform()
                    obj.setTransform(R, [0,0,0]) #[0,0,0] or t?
                    # obj.setTransform(R, [0,0,0]) #[0,0,0] or t?
                    w_T_o = np.array(se3.homogeneous((R,[0,0,0]))) # object is at origin

                    p_T_h = np.array(se3.homogeneous(xform))

                    poses_h = []
                    # # embed()
                    for j in range(len(poses)):
                        poses_h.append(w_T_o.dot(np.dot(poses[j],p_T_h))) #w_T_h
                        # poses_h.append((w_T_o.dot(poses[i]).dot(p_T_h)))

                    # embed()
                    # print "-------Filtering poses:"
                    filtered_poses = []
                    # embed()
                    for j in range(len(poses)):
                        if not CollisionTestPose(world, robot, obj, poses_h[j]):
                            if not CollisionCheckWordFinger(robot, poses_h[j]):
                                filtered_poses.append(poses[j])

                    if len(filtered_poses) == 0:
                        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        print "Filtering returned 0 feasible poses"
                        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        continue
                    # embed()
                    # create a hand emulator from the given robot name
                    module = importlib.import_module('plugins.' + robotname)
                    # R,t = obj.getTransform()
                    # emulator takes the robot index (0), start link index (6), and start driver index (6)
                    PoseDanyDiff = RelativePosition(robot,obj)
                    program = TesterGrab(filtered_poses,
                                                           world,
                                                           p_T_h,
                                                           R,
                                                           t,
                                                           PoseDanyDiff,
                                                           module)
                    vis.setPlugin(None)
                    vis.setPlugin(program)
                    program.reshape(800, 600)
                    # vis.lock()
                    vis.show()
                    # vis.unlock()
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



    try:
        objname = sys.argv[1]
        launch_test_mvbb_filtered("reflex_col", [objname])
    except:
        launch_test_mvbb_filtered("reflex_col", all_objects)