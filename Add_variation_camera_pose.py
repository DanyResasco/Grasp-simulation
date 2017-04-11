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
from camera_pose_variation import Add_variation



def create_camera_frame(z):
    """
    create_camera_frame returns a rotation matrix with z parallel to z
    """
    _R = np.eye(3)

    _z = np.array(z).reshape(3)
    _z = _z / np.linalg.norm(_z)
    _z_world = np.array([0,0,1])
    if np.abs(_z.dot(_z_world)) > (1 - 1e-9):  # collinearity check
        if _z.dot(_z_world) < 0.:
            _R = np.array([[1, 0, 0],
                           [0,-1, 0],
                           [0, 0,-1]])
    else:
        _x_cand = np.cross(_z_world, _z)
        _y = np.cross(_x_cand, -_z)
        _y = _y / np.linalg.norm(_y)
        _x = -np.cross(_z, _y)
        # _R = np.vstack((_x, _y, _z)).transpose()
        _R = np.vstack((_x, _y, _z)).transpose()
    return _R







def Camera_first_pose(axis,R_o,center):
    # print R_o
    if axis == 0 :
        print "axis 0"
        P = np.array(se3.homogeneous((so3.from_matrix(R_o),[center[0],center[1]-0.7,center[2]])))
        R = np.array(se3.homogeneous((so3.from_axis_angle(([1,0,0],math.radians(90))),[0,0,0] )))
        # embed()
        # vect_pose.append(se3.from_homogeneous(np.dot(P,R)))
        vect_pose=(se3.from_homogeneous(np.dot(P,R)))
    elif axis == 1:
        print "axis 1"
        xc = R_o[1]
        yc = R_o[2]
        zc = R_o[0]
        P = np.array(se3.homogeneous((so3.from_matrix((xc,yc,zc)),[center[0]+0.7,center[1],center[2]])))
        # embed()
        # vect_pose.append(se3.from_homogeneous(P))
        vect_pose =(se3.from_homogeneous(P))
    elif axis ==2:
        print "axis 2"
        xc = R_o[2]
        yc = R_o[0]
        zc = R_o[1]
        P = np.array(se3.homogeneous((so3.from_matrix((xc,yc,zc)),[center[0],center[1]-0.7,center[2]])))
        # embed()
        # vect_pose.append(se3.from_homogeneous(P))
        vect_pose = (se3.from_homogeneous(P))
        # embed()
    return vect_pose




def Make_camera_poses_no_work(o_T_p,obj):
    o_T_p_r = []


    # print "len: ", len(o_T_p)

    # long_side,index, standing, T_box = Compute_box(obj)

    R_o,t = obj.getTransform()
    # embed()
    bmin,bmax = obj.geometry().getBB()
    centerX = 0.5 * ( bmax[0] - bmin[0] ) + t[0]
    centerY = 0.5 * ( bmax[1] - bmin[1] ) + t[1]
    centerZ = 0.5 * ( bmax[2] - bmin[2] ) + t[2]

    T_o = [centerX,centerY,centerZ]

    print "len(o_T_p)", len(o_T_p)
    for k in range(0,len(o_T_p)):
        # print k
        if k == 0:
            # pass
            axis = Find_long_side_and_axis(bmin,bmax)
            # Camera_first_pose(axis,so3.matrix(R_o),[centerX,centerY,centerZ],o_T_p_r)
            o_T_p_temp = Camera_first_pose(axis,so3.matrix(R_o),[centerX,centerY,centerZ])
            # embed()
            
        else:
            # pass
            tx = se3.from_homogeneous( o_T_p[k])[1][0]
            ty = se3.from_homogeneous( o_T_p[k])[1][1]
            tz = se3.from_homogeneous( o_T_p[k])[1][2]

            z = -(np.array([tx,ty,tz]) - np.array([centerX,centerY,centerZ]) ) / np.linalg.norm((np.array([tx,ty,tz]) - np.array([centerX,centerY,centerZ])))

            if np.linalg.norm(np.cross(z,np.array([0,0,1])) ) is not 0:
                x = np.cross(z,np.array([0,0,1])) / np.linalg.norm(np.cross(z,np.array([0,0,1])) )
            else:
                x = [1,0,0]
            y = np.cross(z,x)

            # index_plane = Find_closed_plane([tx ,ty ,tz ])
            # print index_plane
            # if index_plane == 0: #xy
            #     T_camera  =[tx ,ty ,tz +0.5]
            # elif index_plane == 1: #xz
            #     T_camera  =[tx ,ty -0.5,tz]
            # elif index_plane == 2: #yz
            #     T_camera  =[tx +0.5,ty ,tz ]

            # n_ = []
            # n_.append( math.acos(np.dot([0,0,1],z) / np.dot(np.sqrt(np.dot(z,z)),np.sqrt(np.dot([0,0,1],[0,0,1]))) ))
            # n_.append( math.acos(np.dot([0,1,0],z) / np.dot(np.sqrt(np.dot(z,z)),np.sqrt(np.dot([0,1,0],[0,1,0]))) ))
            # n_.append( math.acos(np.dot([1,0,0],z) / np.dot(np.sqrt(np.dot(z,z)),np.sqrt(np.dot([1,0,0],[1,0,0]))) ))

            # index_plane = n_.index(min(n_))
            # if index_plane == 0: #xy
            #     T_camera  =[tx ,ty ,tz +0.5]
            # elif index_plane == 1: #xz
            #     T_camera  =[tx ,ty +0.5,tz+0.5]
            # elif index_plane == 2: #yz
            #     T_camera  =[tx ,ty ,tz ]

            # print index_plane
            # embed()


            print '****',se3.from_homogeneous( o_T_p[k])[1]
            T_camera = [tx,ty,tz]
            o_T_p_temp = so3.from_matrix(np.vstack((x,y,z)).transpose()),T_camera
            print np.linalg.det(np.vstack((x,y,z)).transpose())
            # embed()
        o_T_p_r.append(o_T_p_temp)

        # Add_variation(o_T_p_r,o_T_p_temp)
        # break

    # print o_T_p_r

    return o_T_p_r








def Make_camera_poses(o_T_p,obj):
    o_T_p_r = []


    # print "len: ", len(o_T_p)

    # long_side,index, standing, T_box = Compute_box(obj)

    R_o,t = obj.getTransform()
    # embed()
    bmin,bmax = obj.geometry().getBB()
    centerX = 0.5 * ( bmax[0] - bmin[0] ) + t[0]
    centerY = 0.5 * ( bmax[1] - bmin[1] ) + t[1]
    centerZ = 0.5 * ( bmax[2] - bmin[2] ) + t[2]

    T_o = [centerX,centerY,centerZ]

    print "len(o_T_p)", len(o_T_p)
    len_o_t_p = len(o_T_p)
    if len_o_t_p == 1:
                    # pass
            axis = Find_long_side_and_axis(bmin,bmax)
            # Camera_first_pose(axis,so3.matrix(R_o),[centerX,centerY,centerZ],o_T_p_r)
            o_T_p_temp = Camera_first_pose(axis,so3.matrix(R_o),[centerX,centerY,centerZ])
            # embed()
    # else: 

    for k in range(0,len(o_T_p)):
        # print k
        if k == 0: #xz
            x = [1,0,0]
            y = [0,0,1]
            z = [0,-1,0]
            o_T_p_temp = so3.from_matrix(np.vstack((x,y,z)).transpose()),[centerX, centerY-0.7,centerZ]
            
        elif k ==1: #zy
            z = [1,0,0]
            x = [0,0,1]
            y = [0,-1,0]
            o_T_p_temp = so3.from_matrix(np.vstack((x,y,z)).transpose()),[centerX +0.7, centerY,centerZ]
        elif k==2:#xz back
            x = [1,0,0]
            y = [0,0,-1]
            z = [0,1,0]
            o_T_p_temp = so3.from_matrix(np.vstack((x,y,z)).transpose()),[centerX, centerY+0.7,centerZ]
        elif k==3:#zy back
            z = [-1,0,0]
            x = [0,0,1]
            y = [0,1,0]
            # embed
            temp_r = np.array(se3.homogeneous((so3.from_matrix(np.vstack((x,y,z)).transpose()),[centerX-0.7,centerY,centerZ])))
            r_y = np.array(se3.homogeneous((so3.from_axis_angle(([0,1,0],math.radians(180))),[0,0,0] )))
            o_T_p_temp = (se3.from_homogeneous(np.dot(temp_r,r_y)))
        elif k==4:#xy
            x = [1,0,0]
            z = [0,0,1]
            y = [0,1,0]
            temp_r = np.array(se3.homogeneous((so3.from_matrix(np.vstack((x,y,z)).transpose()),[centerX,centerY,centerZ+0.7])))
            r_x = np.array(se3.homogeneous((so3.from_axis_angle(([1,0,0],math.radians(180))),[0,0,0] )))
            o_T_p_temp = (se3.from_homogeneous(np.dot(temp_r,r_x)))
            # o_T_p_temp = so3.from_matrix(np.vstack((x,y,z)).transpose()),[centerX, centerY,centerZ+0.7]
        else:
            break;

        Add_variation(o_T_p_r,o_T_p_temp)
        # break

    # print o_T_p_r

    return o_T_p_r
