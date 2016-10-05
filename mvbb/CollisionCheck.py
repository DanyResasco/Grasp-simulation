#!/usr/bin/env python
from klampt import *
from klampt.vis.glrobotprogram import * #Per il simulatore
from klampt.model import collide



def CheckCollision(world,robot,obj):
    collision = collide.WorldCollider(world) #init
    R_O = collision.robotObjectCollisions(robot,obj) #check collision robot-object. the output is generator
    li = [] # make a list to know how many collisions we have been
    for i in R_O:
        li.append(R_O)
    R_w = collision.robotTerrainCollisions(robot) # check collision robot-plane
    li2 = [] # same as above. The output is generator so make a list to know how many collision we have been
    for j in R_w:
        li2.append(R_w)
    if(len(li)>0 or len(li2)>0): #is the lenght is greater that zero, so we have collision
        return True
    else:
        return False

def CollisionTestInterpolate(world,robot,obj,T):
    P_prev = robot.getConfig()
    SetRobotConfig(robot,T)
    xd = robot.getConfig()
    P_des = [xd[0],xd[1],xd[2]] #take only traslation
    arrived = False
    while arrived == False:
        xa = robot.getConfig() #robot pos actual
        P_actual = [xa[0],xa[1],xa[2]] #take only t
        X = vectorops.interpolate(P_actual,P_des,0.01) #linear interpolation 
        # print("X"), X
        Xi = vectorops.add(P_actual ,vectorops.mul(X,vectorops.sub(P_des,P_actual))) #xa + interpolate*(xd-xa)
        robot.setConfig(Xi)
        # print("collision "),CheckCollision(world,robot,obj)
        # print("P_des"),P_des
        # print("P_actual"),P_actual
        c = vectorops.sub(P_des,Xi) #check if robot has arrived
        C = [c[0],c[1],c[2]]
        # print("c"),C
        Threshold = [0.00005, 0.00005, 0.00005] #position threshold
        if(c < Threshold):
             arrived = True
             robot.setConfig(xa) #set the previously configuration


def  SetRobotConfig(robot,o_P_h):
    #****+ Simple function that takes robot and desired pose. From pose extract rpy
    # and make a robot configuration
    if not isinstance(o_P_h, tuple):
        o_T_h = se3.from_homogeneous(o_P_h) #o_P_h is end-effector in object frame
    import PyKDL
    # print("o_T_w"),o_T_w[0]
    R_kdl = PyKDL.Rotation(o_T_h[0][0],o_T_h[0][1],o_T_h[0][2],o_T_h[0][3],o_T_h[0][4],o_T_h[0][5],o_T_h[0][6],o_T_h[0][7],o_T_h[0][8])
    rpy = R_kdl.GetRPY()
    q = robot.getConfig()
    q[0] = o_T_h[1][0]
    q[1] = o_T_h[1][1]
    q[2] = o_T_h[1][2]
    q[3] = rpy[2] #yaw
    q[4] = rpy[1]#pitch
    q[5] = rpy[0] #roll
    robot.setConfig(q)



def CollisionTestPoseRobotObject(world,robot,obj,o_P_h):
    P_prev = robot.getConfig()
    SetRobotConfig(robot,o_P_h)
    collision = collide.WorldCollider(world) #init
    R_O = collision.robotObjectCollisions(robot,obj) #check collision robot-object. the output is generator
    li = [] # make a list to know how many collisions we have been
    for i in R_O:
        li.append(R_O)
    if(len(li)>0 ): #is the lenght is greater that zero, so we have collision
        robot.setConfig(P_prev)
        return True
    else:
        robot.setConfig(P_prev)
        return False

def CollisionTestPoseRobotTerrain(world,robot,w_P_h):
    P_prev = robot.getConfig()
    SetRobotConfig(robot,w_P_h)
    collision = collide.WorldCollider(world) #init
    R_w = collision.robotTerrainCollisions(robot) # check collision robot-plane
    li2 = [] # The output is generator so we make a list to know how many collision we have been
    for j in R_w:
        li2.append(R_w)
    if(len(li2)>0): #is the lenght is greater that zero, so we have collision
        # robot.setConfig(P_prev)
        return True
    else:
        robot.setConfig(P_prev)
        return False


