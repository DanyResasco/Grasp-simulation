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



class PoseVisualizer(GLNavigationProgram):
    def __init__(self, obj,world,robot,o_T_p_r,o_T_p):
        GLNavigationProgram.__init__(self, 'PoseVisualizer')
        self.world = world
        self.robot = robot
        self.obj = obj
        self.o_T_p_r = o_T_p_r
        self.o_T_p = o_T_p

    def display(self):
        self.world.drawGL()
        # if self.camera is None:
        R,t = self.obj.getTransform()
        bmin,bmax = self.obj.geometry().getBB()
        centerX = 0.5 * ( bmax[0] - bmin[0] ) +t[0] 
        centerY = 0.5 * ( bmax[1] - bmin[1] ) + t[1]
        centerZ = 0.5 * ( bmax[2] - bmin[2] ) + t[2]
        P = R,[centerX,centerY,centerZ]
        # print P
        draw_GL_frame(P)
        draw_GL_frame(se3.from_homogeneous(self.o_T_p_r))
        draw_GL_frame(se3.from_homogeneous(self.o_T_p),axis_length = 1)
        # Pc = 
        # [-0.99, 0, -0.0015, 0.0 ,1, 0, 0.0015 ,0,-0.99 ],[ 0.0, 0, 0.5]
        # Pc = [0.99, 0, -0.0015, 0.0 ,1 ,0, 0.0015, 0, 0.99], [0.0 ,0 ,1.5]
        # draw_GL_frame(Pc)


        #camera -> link = 0
        # draw_GL_frame(self.robot.link(0).getTransform())


    def idle(self):
        pass


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

def Fin_min_angle(pose,axis):
    x = pose[:3,0]
    y = pose[:3,1]
    z = pose[:3,2]
    # embed()
    Dx = math.acos(np.dot(axis,x) / np.dot(np.sqrt(np.dot(x,x)),np.sqrt(np.dot(axis,axis))) )
    Dy = math.acos(np.dot(axis,y) / np.dot(np.sqrt(np.dot(y,y)),np.sqrt(np.dot(axis,axis))) )
    Dz = math.acos(np.dot(axis,z) / np.dot(np.sqrt(np.dot(z,z)),np.sqrt(np.dot(axis,axis))) )
    return [Dx,Dy,Dz]


def  Find_axis_rotation(o_T_p):

    mini2 = Fin_min_angle(o_T_p,[1,0,0])
    minx = mini2.index(min(mini2))
    print "minx min x:", minx
    mini = Fin_min_angle(o_T_p,[0,0,1])
    maxz = mini.index(max(mini))
    miniy = Fin_min_angle(o_T_p,[0,1,0])
    miny = miniy.index(max(miniy))
    print "distance with y",miny




    print "index max z", maxz
    if ((minx == 2) and (maxz == 1)) and miny !=2 or ((minx ==0) and (maxz ==2)) : #zy
        axis = [0,1,0]
        R = so3.from_axis_angle((axis,math.radians(90)))
    # elif ((minx == 2) and (maxz == 1)) and miny ==2:
    elif ((minx == 1) and (maxz == 2)and miny !=2):
        axis = [0,1,0]
        R = so3.from_axis_angle((axis,math.radians(180)))
    # elif ((minx == 1) and (maxz == 2)and miny ==0):
    #     axis = [0,1,0]
    #     R = so3.from_axis_angle((axis,math.radians(90)))
    elif  (minx ==2) and (maxz ==0):
        R = so3.from_axis_angle(([0,1,0],math.radians(180)))
    elif ((minx ==0) and (maxz ==0)) and miny == 2 or (((minx == 2) and (maxz == 1)) and miny ==2):
        R = so3.from_axis_angle(([0,1,0],math.radians(0)))
    elif ((minx ==0) and (maxz ==0)) and miny != 2:
        R = so3.from_axis_angle(([0,1,0],math.radians(90)))
    elif ((minx ==2) and (maxz ==2) and miny ==0):
        axis = [0,1,0]
        R = so3.from_axis_angle((axis,math.radians(90)))
    elif ((minx ==1) and (maxz ==2) and miny ==2):
        axis = [1,0,0]
        R = so3.from_axis_angle((axis,math.radians(90)))

    else:
        Normal = np.array([0,0,1]) #z axis
        mini = Fin_min_angle(o_T_p,Normal)
        index = mini.index(min(mini))
        print"**************min z*", index
        if index == 0: #x axis
            axis = [1,0,0]
        elif index == 1: #y axis
            axis = [0,1,0]
        else: #no z  
            if mini[0] < mini[1]:
                axis = [1,0,0]
            else:
                axis = [0,1,0]
        R = so3.from_axis_angle((axis,math.radians(180)))
    return R

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
    # set_moving_base_xform(robot, pose_se3[0], pose_se3[1])

    for object_name in object_list:
        obj = None
        for object_set, objects_in_set in objects.items():
            if object_name in objects_in_set:
                # embed()
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

                    # w_T_o = np.array(se3.homogeneous((R,t))) 
                    # p_T_h = np.array(se3.homogeneous(xform))

                    o_T_p= []
                    Read_Poses(object_name,o_T_p)
                    print object_name
                    #now the simulation is launched
                    program = GLSimulationProgram(world)
                    sim = program.sim
                    camera = (sim.controller(0).sensor('rgbd_camera'))
                    
                    
                    
                    for k in range(0,len(o_T_p)):
                        
                        R,t = obj.getTransform()
                        if k == 0:
                            bmin,bmax = obj.geometry().getBB()
                            centerX = 0.5 * ( bmax[0] - bmin[0] ) +t[0] 
                            centerY = 0.5 * ( bmax[1] - bmin[1] ) + t[1]
                            centerZ = 0.5 * ( bmax[2] - bmin[2] ) + t[2]
                            P = np.array(se3.homogeneous((R,[centerX,centerY,centerZ])))
                            R = np.array(se3.homogeneous((so3.from_axis_angle(([1,0,0],math.radians(90))),[0,0,0] )))
                            o_T_p_r = se3.from_homogeneous(np.dot(P,R))
                        else:
                            R = np.array(se3.homogeneous((Find_axis_rotation(o_T_p[k]), [0,0,0])))
                            o_T_p_r = se3.from_homogeneous(( np.dot(o_T_p[k],R )))
                        
                        o_T_p_r[1][1] = o_T_p_r[1][1] - 0.7 #move along object y frame
                        # o_T_p = [0.99 ,0, -0.0015, 0.0 ,1 ,0 ,0.0015, 0 ,0.99],[ 0.0, 0, 1]

                        Tsensor = ''
                        for s in o_T_p_r[0]+o_T_p_r[1]:
                            Tsensor += str(s) + ' '
                        
                        vis.setPlugin(program)
                        vis.setPlugin(PoseVisualizer(obj,world,robot,se3.homogeneous(o_T_p_r),o_T_p[k]))
                        camera.setSetting("Tsensor",Tsensor)
                        vis.add("sensor",camera)
                        sim.simulate(0.1)
                        sim.updateWorld()
                        camera_measure = camera.getMeasurements()

                        rgb = (np.array(camera_measure)[0:len(camera_measure)/2]).reshape(128,128)
                        scipy.misc.imsave('outfile_%s.jpg'%k, rgb)
                        res_dataset = '2DCNN/NNSet/Image/%s_rotate_%s.csv'% (object_name,k)
                        Write_image(camera_measure,res_dataset)
                        # embed()
                        #this code manually updates the visualization
                        
                        vis.show()
                        t0 = time.time()
                        while vis.shown():
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



