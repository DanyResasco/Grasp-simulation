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
import numpy as np
import math


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

def  FromCamera2rgb(camera_measure):
    abgr = (np.array(camera_measure)[0:len(camera_measure)/2]).reshape(128,128).astype(np.uint32)
    rgb = np.zeros((128,128,3),dtype=np.uint8)
    rgb[:,:,0] =                np.bitwise_and(abgr,0x000f)
    rgb[:,:,1] = np.right_shift(np.bitwise_and(abgr,0x00f0), 8)
    rgb[:,:,2] = np.right_shift(np.bitwise_and(abgr,0x0f00), 16)

    return rgb