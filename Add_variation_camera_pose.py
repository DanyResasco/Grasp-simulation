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
import random
from utils_camera import FromCamera2rgb, Find_axis_rotation, FInd_Plane
from IPython import embed

def Add_variation(o_T_p_r_original,o_T_p_r):


    theta_deg = random.randint(-5,5)
    if theta_deg == 0.0:
        theta_deg = random.randint(-5,5)
    theta = math.radians(theta_deg)


    ##Variation on x
    R = np.array(se3.homogeneous((so3.from_axis_angle(([1,0,0],theta)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)


    #  #variazione su y
    R = np.array(se3.homogeneous((so3.from_axis_angle(([0,1,0],theta)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)
    
    #  #variazione su z
    # R = np.array(se3.homogeneous((so3.from_axis_angle(([0,0,1],theta)),[0,0,0] )))
    # o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    # # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    # o_T_p_r.append(o_T_p_r_temp)


    # #variazione su x e muovo su z
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle(([1,0,0],theta)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)


    # #variazione su y e muovo su z
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle(([0,1,0],theta)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)


    # #variazione su y e muovo su z
    # move_z = random.randrange(-5,5)
    # R = np.array(se3.homogeneous((so3.from_axis_angle(([0,0,1],theta)),[0,0,0] )))
    # o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    # # o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - move_z*0.01
    # # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    # o_T_p_r.append(o_T_p_r_temp)

    # # muovo su - z
    o_T_p_r_temp = o_T_p_r_original
    move_z = random.randrange(-5,5)
    o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)

    # #muovo su + z
    o_T_p_r_temp = o_T_p_r_original
    move_z = random.randrange(-5,5)
    o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - move_z*0.011
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)

    #  #variazione su y e muovo su z
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle(([0,1,0],theta)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)

    #  #variazione su x e muovo su y
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle(([1,0,0],theta)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)


def Make_camera_poses(o_T_p,obj):
    o_T_p_r = []


    print "len: ", len(o_T_p)



    for k in range(0,len(o_T_p)):
        # print se3.from_homogeneous( o_T_p[k])[1]

        if k == 0:
            R_o,t = obj.getTransform()
            bmin,bmax = obj.geometry().getBB()
            centerX = 0.5 * ( bmax[0] - bmin[0] ) + t[0]
            centerY = 0.5 * ( bmax[1] - bmin[1] ) + t[1]
            centerZ = 0.5 * ( bmax[2] - bmin[2] ) + t[2]
            P = np.array(se3.homogeneous((R_o,[centerX,centerY-0.7,centerZ])))
            R = np.array(se3.homogeneous((so3.from_axis_angle(([1,0,0],math.radians(-90))),[0,0,0] )))
            o_T_c = so3.mul(so3.from_axis_angle(([1,0,0],math.radians(-90))),R_o)
            # embed()
            o_T_p_r.append(se3.from_homogeneous(np.dot(P,R)))
        else:
            # embed()
            x = np.array(t) - np.array(se3.from_homogeneous( o_T_p[k]))[1]
            angle =  math.acos(np.dot([0,0,1],OC) / np.dot(np.sqrt(np.dot(OC,OC)),np.sqrt(np.dot([0,0,1],[0,0,1]))) )
            # angle2  = math.radians(90)- angle
            embed()

            n = FInd_Plane([0,0,0],[0,0,1],x)
            y  = np.dot( np.array(se3.from_homogeneous( o_T_p[k])[1]) ,np.array(so3.from_axis_angle((n,angle))).reshape(3,3))


            z  = np.cross(x,y)
            embed()


            o_T_p_r.append( se3.from_homogeneous( o_T_p[k]))

        # Add_variation(o_T_p_r[k],o_T_p_r)


    return o_T_p_r