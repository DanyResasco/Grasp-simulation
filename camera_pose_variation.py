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
from utils_camera import FromCamera2rgb,Find_min_angle
from IPython import embed
import PyKDL as kdl
import pydany_bb
import math
from utils_camera import Find_long_side_and_axis,Find_closed_plane
from create_mvbb import MVBBVisualizer, compute_poses, skip_decimate_or_return




def Add_variation(o_T_p_r_vect,o_T_p_r):

    o_T_p_r_vect.append(o_T_p_r)

    # theta_deg = random.randint(-15,15)
    # if theta_deg == 0.0:
    #     theta_deg = random.randint(-15,15)
    # theta = math.radians(theta_deg)


    theta_deg = random.randint(-15,15)
    if theta_deg == 0.0:
        theta_deg = random.randint(-15,15)
    theta_piu = math.radians(theta_deg)
    theta_meno = -theta_piu

    # embed()
    # axis_x = o_T_p_r_original[0][0:3]
    # axis_y = o_T_p_r_original[0][3:6]
    # axis_z = o_T_p_r_original[0][6:9]

    axis_x = o_T_p_r[0][0:3]
    axis_y = o_T_p_r[0][3:6]
    axis_z = o_T_p_r[0][6:9]

    # embed()
    ##Variation on x
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    # embed()
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),R)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    #  #variazione su y
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),R)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)
    
    #  #variazione su z
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),R)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    # #variazione su x e muovo su z
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),R)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    # #variazione su y e muovo su z
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),R)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    # #variazione su z e muovo su z
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),R)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # # muovo su - z
    o_T_p_r_temp = o_T_p_r
    move_z = random.randrange(-5,5)
    o_T_p_r_temp[1][2] = o_T_p_r[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # #muovo su + z
    o_T_p_r_temp = o_T_p_r
    move_z = random.randrange(-5,5)
    o_T_p_r_temp[1][2] = o_T_p_r[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    #  #variazione su y e muovo su y
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),R)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


        #  #variazione su y e muovo su x
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),R)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    #  #variazione su x e muovo su x
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),R)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    #  #variazione su z e muovo su y
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),R)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


        #  #variazione su z e muovo su x
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),R)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

            #  #variazione su x e muovo su y
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),R)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    #             #  #variazione su x e muovo su x
    # move_z = random.randrange(-5,5)
    # R = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    # o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),R)))
    # # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    # o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    # o_T_p_r_vect.append(o_T_p_r_temp)

        # # muovo su - x
    o_T_p_r_temp = o_T_p_r
    move_z = random.randrange(-5,5)
    o_T_p_r_temp[1][0] = o_T_p_r[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # #muovo su + x
    o_T_p_r_temp = o_T_p_r
    move_z = random.randrange(-5,5)
    o_T_p_r_temp[1][0] = o_T_p_r[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)



    # # muovo su - y
    o_T_p_r_temp = o_T_p_r
    move_z = random.randrange(-5,5)
    o_T_p_r_temp[1][1] = o_T_p_r[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # #muovo su + y
    o_T_p_r_temp = o_T_p_r
    move_z = random.randrange(-5,5)
    o_T_p_r_temp[1][1] = o_T_p_r[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    # #variazione su xz
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rx,Rz))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    # o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # #variazione su xz move x
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rx,Rz))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    # #variazione su xz move y
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rx,Rz))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    # #variazione su xz move z
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rx,Rz))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    
    # #variazione su xz move x-y
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rx,Rz))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    # #variazione su xz move yz
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rx,Rz))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    # #variazione su xz move z
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rx,Rz))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)



        # #variazione su zx
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rz,Rx))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    # o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # #variazione su zx move x
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rz,Rx))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    # #variazione su zx move y
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rz,Rx))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    # #variazione su zx move z
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rz,Rx))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # #variazione su zx move x-y
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rz,Rx))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    # #variazione su zx move y-z
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rz,Rx))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    # #variazione su zx move z-x
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rz,Rx))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


    # #variazione su xy 
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rx,Ry))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    # o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


     # #variazione su xy  var y
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rx,Ry))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


         # #variazione su xy move z
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rx,Ry))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

         # #variazione su xy  move x
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rx,Ry))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


         # #variazione su xy  var y-x
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rx,Ry))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


         # #variazione su xy move zx
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rx,Ry))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

         # #variazione su xy  move z-y
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rx,Ry))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


        # #variazione su yx 
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Ry,Rx))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    # o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


     # #variazione su yx  var y
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Ry,Rx))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


         # #variazione su yx move z
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Ry,Rx))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

         # #variazione su yx  move x
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Ry,Rx))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)





         # #variazione su yx  var y-x
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Ry,Rx))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


         # #variazione su yx move z-y
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Ry,Rx))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

         # #variazione su yx  move x-z
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Ry,Rx))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)





    # #variazione su zy 
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rz,Ry))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    # o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

                 # #variazione su zy   mov x
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rz,Ry))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


     # #variazione su zy   mov y
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rz,Ry))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # #variazione su zy   mov z
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rz,Ry))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)



                     # #variazione su zy   mov x-y
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rz,Ry))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


     # #variazione su zy   mov y-z
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rz,Ry))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # #variazione su zy   mov z-x
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Rz,Ry))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


        # #variazione su yz 
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Ry,Rz))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    # o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

                 # #variazione su yz   mov x
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Ry,Rz))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


     # #variazione su yz   mov y
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Ry,Rz))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # #variazione su yz   mov z
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Ry,Rz))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)



    # #variazione su yz   mov y-x
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Ry,Rz))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


     # #variazione su yz   mov y
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Ry,Rz))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # #variazione su yz   mov z
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),np.dot(Ry,Rz))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)









        # #variazione su x   mov x-y
    move_z = random.randrange(-5,5)
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    # Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),Rx)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

            # #variazione su x   mov x-z
    move_z = random.randrange(-5,5)
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    # Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),Rx)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # #variazione su x   mov y-z
    move_z = random.randrange(-5,5)
    Rx = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta_piu)),[0,0,0] )))
    # Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),Rx)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)


        # #variazione su y  mov y-z
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    # Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),Ry)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

            # #variazione su y  mov y-x
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    # Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),Ry)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # #variazione su y  mov z-x
    move_z = random.randrange(-5,5)
    Ry = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta_piu)),[0,0,0] )))
    # Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),Ry)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # #variazione su z  mov y-x
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    # Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),Rz)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

        # #variazione su z  mov y-z
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    # Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),Rz)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_temp[1][1] - move_z*0.01
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)

    # #variazione su z  mov x-z
    move_z = random.randrange(-5,5)
    Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    # Rz = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta_piu)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(np.array(se3.homogeneous(o_T_p_r)),Rz)))
    # o_T_p_r_temp[1][0:3] = o_T_p_r[1][0:3]
    o_T_p_r_temp[1][0] = o_T_p_r_temp[1][0] - move_z*0.01
    o_T_p_r_temp[1][2] = o_T_p_r_temp[1][2] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r[1][1] - 0.7
    o_T_p_r_vect.append(o_T_p_r_temp)
