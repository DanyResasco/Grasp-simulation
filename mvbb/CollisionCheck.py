#!/usr/bin/env python
from klampt import *
from klampt.vis.glrobotprogram import * #Per il simulatore
from klampt.model import collide
from moving_base_control import set_moving_base_xform, get_moving_base_xform
import numpy as np
from IPython import embed 
from klampt.math import se3,so3



def CheckCollision(world, robot, obj, collision = None):
    """
    :param world: world object
    :param robot: robot object
    :param obj: object to check collision against
    :collision: an optional WorldCollider
    :return: True if in collision
    """
    if collision is None:
        collision = collide.WorldCollider(world) #init
    r_o_coll = collision.robotObjectCollisions(robot,obj) #check collision robot-object. the output is generator
    list_r_o_coll = [] # make a list to know how many collisions we have been
    for coll in r_o_coll:
        list_r_o_coll.append(coll)
    r_w_coll = collision.robotTerrainCollisions(robot) # check collision robot-plane
    list_r_w_coll = [] # same as above. The output is generator so make a list to know how many collision we have been
    for coll in r_w_coll:
        list_r_w_coll.append(coll)
    if len(list_r_o_coll) > 0 or len(list_r_w_coll) >0: #if the length is greater that zero, we have collision
        # print "collision"
        return True
    else:
        return False

def CollisionTestInterpolate(world,robot,obj,o_P_h):
    q_old = robot.getConfig()
    o_T_h_curr = get_moving_base_xform(robot)

    if not isinstance(o_P_h, tuple):
        o_T_h = se3.from_homogeneous(o_P_h) #o_P_h is end-effector in object frame
    else:
        o_T_h = o_P_h

    t_des = o_T_h[1]
    t_curr = o_T_h_curr[1]

    set_moving_base_xform(robot, o_T_h[0], o_T_h[1])

    step_size = 0.01
    d = vectorops.distance(t_curr, t_des)

    n_steps = int(math.ceil(d / step_size))
    if n_steps == 0:    # if o_T_h_current == o_T_h_desired
        return CheckCollision(world, robot, obj)

    collider = collide.WorldCollider(world)

    for i in range(n_steps):
        t_int = vectorops.interpolate(t_curr,t_des,float(i+1)/n_steps)

        set_moving_base_xform(robot, o_T_h[0], t_int)
        if CheckCollision(world, robot, obj, collider):
            robot.setConfig(q_old)
            return True

    robot.setConfig(q_old)
    return False


def CollisionTestPose(world,robot,obj,w_P_h):
    q_old = robot.getConfig()
    if not isinstance(w_P_h, tuple):
        w_T_h = se3.from_homogeneous(w_P_h) #w_P_h is end-effector in object frame
    else:
        w_T_h = w_P_h

    set_moving_base_xform(robot, w_T_h[0], w_T_h[1])

    coll = CheckCollision(world, robot, obj)

    robot.setConfig(q_old)
    return coll


# def FromRPY(rpy):
#     roll,pitch,yaw = rpy
#     Rx,Ry,Rz = so3.from_axis_angle([(1,0,0),roll]),so3.from_axis_angle([(0,1,0),pitch]),so3.from_axis_angle([(0,0,1),yaw])
#     return so3.mul(Rz,so3.mul(Ry,Rx))

def CollisionCheckWordFinger(robot, w_P_h):
    """
    :param robot: robot object
    :param robot: pose from hand to word
    :return: True if the fingers are under the table
    Thanks Ale :)
    """

    q_old = robot.getConfig()
    if not isinstance(w_P_h, tuple):
        w_T_h = se3.from_homogeneous(w_P_h) #w_P_h is end-effector in object frame
    else:
        w_T_h = w_P_h

    set_moving_base_xform(robot, w_T_h[0], w_T_h[1]) #move the robot

    collFinger = 0

    for i in range(robot.numLinks()):

        w_T_l = robot.link(i).getTransform()
        g = robot.link(i).geometry()
        bb = None
        if not g.empty():
            bb = g.getBB() #take axis-aligned bb

        if w_T_l[1][2] < 0. or (bb is not None and (bb[0][2] < 0. or bb[1][2] < 0.)): #under the table
            print robot.link(i).getName(), "has negative z coord for pose", w_T_h
            collFinger += 1

    coll = collFinger > 0

    robot.setConfig(q_old)
    return coll