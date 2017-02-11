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
from klampt.math import so3, se3
# import pydany_bb
from mvbb.ScalaReduce import DanyReduceScale
import numpy as np
from IPython import embed
# from mvbb.TakePoses import SimulationPoses
from mvbb.draw_bbox import draw_GL_frame
from i16mc import make_object, make_moving_base_robot
# from mvbb.db import MVBBLoader
import csv
import scipy.misc
import random
from Add_variation_camera_pose import Add_variation,Make_camera_poses
from utils_camera import FromCamera2rgb, Find_axis_rotation


Pose = {}
Pose['pose'] = [f for f in os.listdir('3DCNN/NNSet/Pose/pose')]
objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
# objects['newObjdany'] = [f for f in os.listdir('data/objects/newObjdany')]
objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]
robotname = "reflex_col"
robot_files = {
	'reflex_col':'reflex_col.rob'
}



# class PoseVisualizer(GLNavigationProgram):
#     def __init__(self, obj,world,robot,o_T_p_r,o_T_p):
#         GLNavigationProgram.__init__(self, 'PoseVisualizer')
#         self.world = world
#         self.robot = robot
#         self.obj = obj
#         self.o_T_p_r = o_T_p_r
#         self.o_T_p = o_T_p

#     def display(self):
#         self.world.drawGL()
#         # if self.camera is None:
#         R,t = self.obj.getTransform()
#         bmin,bmax = self.obj.geometry().getBB()
#         centerX = 0.5 * ( bmax[0] - bmin[0] ) +t[0] 
#         centerY = 0.5 * ( bmax[1] - bmin[1] ) + t[1]
#         centerZ = 0.5 * ( bmax[2] - bmin[2] ) + t[2]
#         P = R,[centerX,centerY,centerZ]
#         # print P
#         draw_GL_frame(P)
#         draw_GL_frame(se3.from_homogeneous(self.o_T_p_r))
#         draw_GL_frame(se3.from_homogeneous(self.o_T_p),axis_length = 1)
#     def idle(self):
#         pass

class Camera_simulation(GLRealtimeProgram):
    def __init__(self, poses, world,obj,object_name):
        GLRealtimeProgram.__init__(self, "Camera_simulation")
        self.poses = poses
        self.world = world
        self.is_simulating = False
        self.curr_pose = None
        self.running = True
        self.obj = obj
        self.curr_pose = None
        self.sim = None
        self.step = 0
        self.camera = None
        self.nome_obj = object_name
        self.t_0 = None

    def display(self):
        """ Draw a desired pose and the end-effector pose """
        if self.running:
            self.world.drawGL()

            if len(self.poses)>0:
                R,t = self.obj.getTransform()
                bmin,bmax = self.obj.geometry().getBB()
                centerX = 0.5 * ( bmax[0] - bmin[0] ) +t[0] 
                centerY = 0.5 * ( bmax[1] - bmin[1] ) + t[1]
                centerZ = 0.5 * ( bmax[2] - bmin[2] ) + t[2]
                P = R,[centerX,centerY,centerZ]
                # print P
                draw_GL_frame(P)
                if self.curr_pose is not None:
                    draw_GL_frame(self.curr_pose)


    def idle(self):
        if not self.running:
            return

        if not self.is_simulating:
            if len(self.poses) > 0:
                self.curr_pose = self.poses.pop(0)
                print  self.curr_pose
                print len(self.poses)
            else:
                self.running = False
                vis.show(hidden=True)
                return

            if self.sim is None:
                self.sim = SimpleSimulator(self.world)
                self.camera = (self.sim.controller(0).sensor('rgbd_camera'))
            self.t_0 = self.sim.getTime()
            self.is_simulating = True

        if self.is_simulating:
            vis.add("sensor",camera)

            Tsensor = ''
            for s in self.curr_pose[0]+self.curr_pose[1]:
                Tsensor += str(s) + ' '

            self.camera.setSetting("Tsensor",Tsensor)
            self.sim.simulate(0.1)
            self.sim.updateWorld()
            # camera_measure = self.camera.getMeasurements()
            # print camera_measure
            # rgb = FromCamera2rgb(camera_measure)
            # scipy.misc.imsave('outfile_%s.jpg'%self.step, rgb)
            # res_dataset = '2DCNN/NNSet/Image/%s_rotate_%s.csv'% (self.nome_obj,self.step)
            # Write_image(camera_measure,res_dataset)
            # self.step +=1
            # print self.step
            if not vis.shown() or (self.sim.getTime() - self.t_0) >= 2.5:
                if  vis.shown():
                    camera_measure = self.camera.getMeasurements()
                    rgb = FromCamera2rgb(camera_measure)
                    scipy.misc.imsave('outfile_%s.jpg'%self.step, rgb)
                    res_dataset = '2DCNN/NNSet/Image/%s_rotate_%s.csv'% (self.nome_obj,self.step)
                    Write_image(camera_measure,res_dataset)
                    self.step +=1
                    self.is_simulating = False
                    self.sim = None




def Write_image(camera,dataset):
    '''Write the dataset'''
    # embed()
    import csv
    f = open(dataset, 'w')
    # embed()
    for i in camera:
        f.write(','.join([str(i)]))
        f.write('\n')
    f.close()



def Read_Poses(nome,vector_set):

    obj_dataset = '3DCNN/NNSet/Pose/pose/%s.csv'%(nome)
    with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
        file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
        for row in file_reader:
            T = row[9:12]
            pp = row[:9]
            # vector_set.append(np.array(se3.homogeneous((pp,T))))
            # embed()
            vector_set.append( np.array(se3.homogeneous((pp,T)))) 





def MainDany(object_list):
    world = WorldModel()
    world.loadElement("data/terrains/plane.env")
    robot = make_moving_base_robot(robotname, world)
    xform = resource.get("default_initial_%s.xform" % robotname, description="Initial hand transform",default=se3.identity(), world=world, doedit=False)
    # pose_se3= ([0.983972802704,-0.0441290216922,0.172771968165,0.177866245396,0.173919267423,-0.968563723855,0.0126933954459,0.983770663246,0.178980892412],[0.234435798004,0.0102866113634,0.0952616290142])
    set_moving_base_xform(robot,xform[0],xform[1])

    for object_name in object_list:
        obj = None
        for object_set, objects_in_set in objects.items():
            if object_name in objects_in_set:
                for i,t in Pose.items():
                    if (object_name +'.csv') in t:
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
            
        R,t = obj.getTransform()
        obj.setTransform(R, [0,0,0]) #[0,0,0] or t?
        o_T_p= []
        Read_Poses(object_name,o_T_p)
        print object_name

        o_T_p_r = Make_camera_poses(o_T_p,obj)
        print "n_poses: ", len(o_T_p_r)




                    #now the simulation is launched
                    # program = GLSimulationProgram(world)
                    # sim = program.sim
                    # camera = (sim.controller(0).sensor('rgbd_camera'))

                    
                    
                    # Tsensor = ''
                    # for s in o_T_p_r[0]+o_T_p_r[1]:
                    #     Tsensor += str(s) + ' '
                    
        program = Camera_simulation(o_T_p_r,world,obj,object_name)

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
                      

#Main
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

    # print "-------------"
    # print all_objects
    # print "-------------"

    try:
        objname = sys.argv[1]
        MainDany([objname])
    except:
        MainDany(all_objects)