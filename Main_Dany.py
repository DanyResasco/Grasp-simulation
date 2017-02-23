#!/usr/bin/env python

import pkg_resources
pkg_resources.require("klampt>=0.7.0")
from klampt import *
from klampt import vis 
# from klampt.vis.glrobotprogram import *
# from klampt.vis.glrobotprogram import GLSimulationProgram
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
from klampt.vis.glprogram import GLRealtimeProgram

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
objects['newObjdany'] = [f for f in os.listdir('data/objects/newObjdany')]
objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]
robots = ['reflex_col', 'soft_hand', 'reflex']

def Draw_Grasph(kindness):
    import matplotlib.pyplot as plt
    print "disegno"
    plt.plot(kindness)
    plt.ylabel('kindness [cm]')
    plt.xlabel('times [s]')
    plt.show()

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
        self.db = MVBBLoader(suffix='blabla')
        # self.logFile = DanyLog(suffix='logFile')
        self.kindness = None
        self.f1_contact = []
        self.f2_contact = []
        self.f3_contact = []
        self.crashing_states = []
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
        elif self.obj is None:
            return

        if not self.is_simulating:
            if len(self.all_poses) > 0:
                self.curr_pose = self.all_poses.pop(0)
               
                # print "Simulating Next Pose Grasp"
                # print self.curr_pose
            else:
                # print "Done testing all", len(self.poses+self.poses_variations), "poses for object", self.obj.getName()
                # print "Quitting"
                self.running = False
                vis.show(hidden=True)
                return

            self.obj.setTransform(self.R, [0,0,0])
            self.obj.setVelocity([0., 0., 0.],[0., 0., 0.])
            # self.obj.setVelocity([0,0,0,0])
            self.w_T_o = np.array(se3.homogeneous(self.obj.getTransform()))
            pose_se3 = se3.from_homogeneous(self.w_T_o.dot(self.curr_pose).dot(self.p_T_h))
            self.robot.setConfig(self.q_0)
            set_moving_base_xform(self.robot, pose_se3[0], pose_se3[1])

            if self.sim is None:
                self.sim = SimpleSimulator(self.world)
                self.sim.enableContactFeedbackAll()
                print "******INizio: ", self.sim.getTime()
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
                        n = len(self.poses)+len(self.poses_variations) - len(self.all_poses)
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
                if (d_hand - d_com > 0.1) and (self.kindness > 1E-4):
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
                print "******fine: ", self.sim.getTime()
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
                    # print "*************Dentro princeton********************" #need to scale the obj size
                    objfilename = 'data/objects/template_obj_scale_princeton.obj'
                    # print"objfilename", objfilename
                    obj = DanyReduceScale(object_name, world,objfilename,object_set)
                else:    
                    obj = make_object(object_set, object_name, world)
        if obj is None:
            # print "Could not find object", object_name
            continue


        R,t = obj.getTransform()
        # obj.setTransform(R, [t[0], t[1], t[2]]) #[0,0,0] or t?
        obj.setTransform(R, [0,0,0]) #[0,0,0] or t?
        object_vertices_or_none, tm_decimated = skip_decimate_or_return(obj, min_vertices, 2000)
        if object_vertices_or_none is None:
            # print "??????????????????????????????????????????????????"
            # print "??????????????????????????????????????????????????"
            # print "??????????????????????????????????????????????????"
            # print "skipping object, too few vertices", obj.getName()
            # print "??????????????????????????????????????????????????"
            # print "??????????????????????????????????????????????????"
            # print "??????????????????????????????????????????????????"
            continue
        object_or_vertices = object_vertices_or_none

        # print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        # print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        # print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        # print object_name
        # print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        # print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        # print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

        # print "------Computing poses for object:", object_name
        poses, poses_variations, boxes = compute_poses(object_or_vertices)
        # print time.time()
        # embed()
        # aa = so3.axis_angle(so3.identity())
        Ry = np.array(se3.homogeneous((so3.from_axis_angle(((0,1,0), 45.*np.pi/180.)),[0,0,0])))
        Rx = np.array(se3.homogeneous((so3.from_axis_angle(((1,0,0), 45.*np.pi/180.)),[0,0,0])))
        Rz = np.array(se3.homogeneous((so3.from_axis_angle(((0,0,1), 45.*np.pi/180.)),[0,0,0])))
        Tx = np.array(se3.homogeneous((so3.identity(), [-.0,0,0])))
        T = Tx.dot(Rz).dot(Rx).dot(Rx) # object is at origin)
        T = Rz;

        poses_new = []

        for pose in poses:
            poses_new.append(pose.dot(T))
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

        # print "-------Filtering poses:"
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
        # print "Filtered from", len(poses+poses_variations), "to", len(filtered_poses+filtered_poses_variations)
        if len(filtered_poses+filtered_poses_variations) == 0:
            # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            # print "Filtering returned 0 feasible poses"
            # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
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
        vis.lock()
        vis.show()
        vis.unlock()
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
    'sterilite_bin_12qt_bottom', #Dont find
    'melissa_doug_play-time_produce_farm_fresh_fruit_unopened_box', #dont find
    #newobjdany
    "juicerB", #dont find
    'bowlB',
    'panA', #dont find
    'containerD', #don't find
    #princeton
    'orangeusa',#error math
    'colorchess', # error math
    'usa', # error math
    'gastruck', #error line 91 i16mc.py
    'stopsignace', #error line 91
    'openchest', #error math
    'yellowcart', #error math
    'whitemonstertruck', #si blocca il pc
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
    '3stair', #too few verticies
    'flipoverpc', #too few verticies
    'pcblue', #too few verticies
    'bigbridge', #zero feasible poses
    'oldgraypc', #zeros feasible poses
    'redumbrella', #too few verticies
    'knight2', #too few verticies
    'buildingposter',  #too few verticies
    'brownboxwithballs', #zeros poses found
    'longformula1', #zeros
    'trianglesignace', # zeros
    'whitebuildingposter', #too
    'stopsignace2', #too
    'whitesignace', #zeros
    'signace', #zeros
    'gravedark', #zeros
    'postbox', #zeros
    'kettle',
    'standingdoublestaircase', #zeros
    'longship', #zeros
    'colouramerica', #zeros
    'yellowsignace', #zeros
    'blackandgreenonewingsboat', #too few verticies
    ]
    done = [
    #ycb and acp
    'pringles_original',
    'dr_browns_bottle_brush',
    'block_of_wood_6in',
    'stanley_philips_screwdriver',
    'elmers_washable_no_run_school_glue',
    'sterilite_bin_12qt_cap',
    'thick_wood_block_6in',  # SI BLOCCA
    'wilson_100_tennis_ball',
    'melissa_doug_farm_fresh_fruit_orange',
    'play_go_rainbow_stakin_cups_8_orange',
    'play_go_rainbow_stakin_cups_box', #il pc si blocca sempre
    'sponge_with_textured_cover',
    'play_go_rainbow_stakin_cups_blue_4',
    'brine_mini_soccer_ball',
    'learning_resources_one-inch_color_cubes_box',
    'soft_scrub_2lb_4oz', 
    'black_and_decker_lithium_drill_driver_unboxed', 
    '1_and_a_half_in_metal_washer', 
    'melissa_doug_farm_fresh_fruit_strawberry',
    'yellow_plastic_chain', 
    'starkist_chunk_light_tuna',
    'melissa_doug_farm_fresh_fruit_plum',
    'play_go_rainbow_stakin_cups_9_red',
    'stainless_steel_spoon_red_handle',
    'comet_lemon_fresh_bleach',
    'stainless_steel_spatula',
    'cheeze-it_388g',
    'moutain_security_steel_shackle',
    'wescott_orange_grey_scissors',
    'purple_wood_block_1inx1in',
    'stainless_steel_fork_red_handle',
    'play_go_rainbow_stakin_cups_10_blue' ,
    'plastic_bolt_grey',
    'play_go_rainbow_stakin_cups_3_red',
    'sharpie_marker',
    'dark_red_foam_block_with_three_holes',
    'melissa_doug_farm_fresh_fruit_lemon',
    'master_chef_ground_coffee_297g',
    'jell-o_chocolate_flavor_pudding',
    'morton_salt_shaker',
    'wilson_golf_ball',
    'white_rope',
    'red_metal_cup_white_speckles',
    'cheerios_14oz',
    'blue_wood_block_1inx1in',
    'jell-o_strawberry_gelatin_dessert',
    'champion_sports_official_softball',
    'clorox_disinfecting_wipes_35',
    'large_black_spring_clamp',
    'melissa_doug_farm_fresh_fruit_apple',
    'frenchs_classic_yellow_mustard_14oz',
    'windex',
    'stainless_steel_knife_red_handle',
    '2in_metal_washer',
    'black_and_decker_lithium_drill_driver',
    'red_metal_bowl_white_speckles',
    'wearever_cooking_pan_with_lid',
    'block_of_wood_12in',
    'medium_black_spring_clamp',
    'red_metal_plate_white_speckles',
    'moutain_security_steel_shackle_key',
    'play_go_rainbow_stakin_cups_7_yellow',
    'rubbermaid_ice_guard_pitcher_blue',
    'melissa_doug_farm_fresh_fruit_pear',
    'domino_sugar_1lb',
    'yellow_wood_block_1inx1in',
    'stanley_13oz_hammer',
    'penn_raquet_ball',
    'melissa_doug_farm_fresh_fruit_peach',
    'play_go_rainbow_stakin_cups_6_purple',
    'morton_pepper_shaker',
    'play_go_rainbow_stakin_cups_5_green',
    'red_wood_block_1inx1in', 
    'melissa_doug_farm_fresh_fruit_banana',
    'first_years_take_and_toss_straw_cups',
    'kong_duck_dog_toy',
    'oreo_mega_stuf',
    'crayola_64_ct',
    'mommys_helper_outlet_plugs',
    'stanley_66_052',
    'feline_greenies_dental_treats',
    'expo_dry_erase_board_eraser',
    'cheezit_big_original',
    'genuine_joe_plastic_stir_sticks',
    'kong_sitting_frog_dog_toy',
    'safety_works_safety_glasses',
    'kong_air_dog_squeakair_tennis_ball',
    'highland_6539_self_stick_notes',
    'mead_index_cards',
    'mark_twain_huckleberry_finn',
    'rollodex_mesh_collection_jumbo_pencil_cup',
    'laugh_out_loud_joke_book',
    'munchkin_white_hot_duck_bath_toy',
    'kygen_squeakin_eggs_plush_puppies',
    'paper_mate_12_count_mirado_black_warrior',
    #newobjdany
    'bowA',
    'containerA',
    'containerC',
    'kitchenUtensilD',
    'kitchenUtensilA',
    'kitchenUtensilB',
    'kitchenUtensilC',
    'kitchenUtensilE',
    'kitchenUtensilF',
    'kitchenUtensilG',
    'mugD',
    'pot',
    'containerB',
    'jug',
    'wc',
    #princeton
    'longwhitebridge',
    'cashmachineblack',
    'antenna',
    'fireplace',
    'arch',
    'cashmachine',
    'cashmachinewhite',
    'cashmachinebig',
    'bigparabola',
    'blackparabola',
    'yellowbook',
    'parabola',
    'whiteswing',
    'sink',
    'squaresink',
    'whitecontainer',
    'chest',
    'whitedoor',
    'whitebigdoor',
    'graybook',
    'ropebridge',
    'chinesbridge',
    'redandgreenbridge',
    'whitebigbridge',
    'doublegraydoor',
    'brownfireplace',
    'blackfireplace',
    'swing',
    'grayswing',
    'redswing',
    'tank',
    'twocilinder',
    'longblackbridge',
    'longpinksatellite',
    'totalblackwithwhitewheeltank',
    'totalwhitetank',
    'blackbook',
    'biggraydoor',
    'bigredstair',
    'blackshoes',
    'bronwstair',
    'glassblackdoor',
    'longblackcontainership',
    'graydooropen',
    'grayshoes',
    'graystripesshoes',
    'standingxwing',
    'lyingblackshoes', #found a different mesh
    'newbalanceshoes',
    'oldpc',
    'pcmac',
    'boatshoes',
    'oldpcwhitkeyboard',
    'flipoverblacktank',
    'colorcar2',
    'flipoverblackship',
    'pinkstandingreceiver',
    'receiver', #il pc si e' bloccato 
    'redandblueumbrella',
    'yellowandbronwumbrella',
    'redandwoodstair',
    'redstair',
    'totalblackubot',
    'smallyellowumbrella',
    'snickers',
    'snowman',
    'snowman2',
    'flipoverwhitemoto',
    'snowmanparty',
    'snowmanwhithat',
    'snowmanwithtrianglehat',
    'stairwhite',
    'standingreceiver',
    'whitebigshoes',
    'whitereceiver',
    'yellowumbrella',
    '2blacksignace',
    'airsignace',
    'america',
    'bishop',
    'blackandstripeshat',
    'blackoldhat',
    'bluemovingballs',
    'blueskate',
    'bigblueskate',
    'flipoveryellowskate',
    'lyingstaircase',
    'captainhat',
    'yellowtrain',
    'notstraightlineoldcarriage',
    'graystair',
    'standingformula1',
    'yellowformula1', # chiede intervento user
    'miniflagcar',
    'redlocomotive',
    'greenskate',
    'blackandwhitebuildingposter',
    'backtruck',
    'glasstruck',
    'lyinggrayformula1',
    'pharmacyblock',
    'manyblock',
    'lyingpawn',
    'knight3',
    'behindbluecar',
    'degreehat',
    'whitechess',
    'flipoverwhitecar',
    'batmancar',
    'locomotive',
    'greendragon',
    'standingbluecar',
    'lyingoldcarriage',
    'greendragonwithwings',
    'yellowdragon',
    'oldcar',
    'train',
    'behindwhitecar',
    'vintagecar',
    'smallcity',
    'blacktruck',
    'pawn3',
    'queen3',
    'car',
    'oldcarriage',
    'notstandingstaircase',
    'rockamerica2',
    'yellowromanhat',
    'smallcity2',
    'behindtruck',
    'whitehat',
    'king',
    'monstertruck',
    'orangeskate',
    'queen',
    'lyingtrain',
    'oldwhitehat',
    'rockcone',
    'redubot',
    'queen2',
    'yellowandgreenlocomotive',
    'personalcomputerblack',
    'pawn4',
    'notstraightlinecar',
    'redsmallship',
    'ship',
    'blueandredbigcar',
    'standingshuttle',
    'redbigcar',
    'pickup',
    'bronwusa',
    'rockamerica',
    'verysmallboat',
    'standingcar',
    'snowmanwhithatandarm',
    'blueandyellowbike',
    'standingwhitetank',
    'smallwhitetank',
    'whitecontainership',
    'flipoverblackxwing',
    'containership',
    'totalblackship',
    'movingballs',
    'historyhat',
    'bluecar',
    'whiteubot',
    'oldcoach',
    'blueandredcar',
    'sideubot',
    'smallcity3',
    'orangeamerica',
    'bronwcowboyhat',
    'pawn2',
    'backoldcarriage',
    'tankblack',
    'highhatwhite',
    'whitelocomotive',
    'blackformula1',
    'graybuildingposter',
    'blackandgreenship',
    'icedragonwithwings',
    'pinkstandingstaircase',
    'vitangeglasscar',
    'orangetruck',
    'boxwithballs',
    'pterosauri',
    'chess',
    'romansailboat',
    'yellowtram',
    'king2',
    'pawn',
    'redxwing',
    'pilothat',
    'longpinkcar',
    'researchship',
    'whiteflagsignace',
    'whoodboat',
    'browndragonwithwings',
    'yellowlocomotive',
    'bluebike',
    'pole',
    'truck',
    'lyingformula1',
    'blackstopsignace',
    'allcolorcar',
    'pinklocomotive',
    'whiteresearchship',
    'tram',
    'yellowskate',
    'backmonstertruck',
    'vikingssailboat',
    'redformula1',
    'formula1',
    'brownchess',
    'manycontainer',
    'sidewhiteubot',
    'knight',
    'behindformula1',
    'longwhiteship',
    'smallcitycolor',
    'militaryhat',
    'pinktriumph',
    'texasyellowhat',
    'blackandwhitetank',
    'behindblueformula1',
    'moto',
    'redmanbike',
    'longwhitecontainership',
    'ship2',
    'messyboat',
    'bluetank',
    'opencar',
    'flipoveropencar',
    'whiteonewingsboat',
    'blueship',
    'longredcontainership',
    'verysmallonewings',
    'onewingsboat',
    'bigship',
    'graygip',
    'blueandgrayTie',
    'grayubot',
    'longbleandredcar',
    'militarygreencar',
    'yellowcar',
    'redbike',
    'militarytruck',
    'xwing',
    'blackcatamaran',
    'redandyellowbike',
    'azureship',
    'carwithflag',
    'purplexwing',
    'whiteandredlongship',
    'flipoverwhitecar2',
    'yacht',
    'longblacksatellite',
    'stanidngwhitebike',
    'pinksatellite',
    'militaryship',
    'whitebike',
    'etruscansailboat',
    'totalblacktank',
    'blacktank',
    'sailboat',
    'brownsmallship',
    'whitegip',
    'whitetank',
    'orangewingboat',
    'whitetank2',
    'smallboat',
    'yellowtank',
    'behindubot',
    'redmoto',
    'harley',
    'blacktie',
    'flipovergreentank',
    'motogp',
    'whitebigsailboat',
    'greentank',
    'bigsailboat',


    ]

    for obj_name in to_filter +  done + to_check:
        all_objects.pop(all_objects.index(obj_name))

    print "-------------"
    print all_objects
    print "-------------"

    # try:
    objname = sys.argv[1]
    # print time.time()
    launch_test_mvbb_filtered("reflex_col", [objname], 100)
    # except:
    #     launch_test_mvbb_filtered("reflex_col", all_objects, 100)