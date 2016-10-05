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

def CollisionTestTraj(world,robot,obj,xd):
    # xd = move_reflex(robot, 5) #desired pose
    P_des = [xd[0],xd[1],xd[2]] #take only traslation
    arrived = False
    while arrived == False:
        xa = robot.getConfig() #robot pos actual
        P_actual = [xa[0],xa[1],xa[2]] #take only t
        X = vectorops.interpolate(P_actual,P_des,0.01) #linear interpolation
        print("X"), X
        Xi = vectorops.add(P_actual ,vectorops.mul(X,vectorops.sub(P_des,P_actual))) #xa + interpolate*(xd-xa)
        robot.setConfig(Xi)
        print("collision "),CheckCollision(world,robot,obj)
        print("P_des"),P_des
        print("P_actual"),P_actual
        c = vectorops.sub(P_des,Xi) #check if robot has arrived
        C = [c[0],c[1],c[2]]
        print("c"),C
        Threshold = [0.00005, 0.00005, 0.00005] #position threshold
        if(c < Threshold):
             arrived = True
             robot.setConfig(xa) #set the previously configuration

 def CollisionTestPose(world, robot, obj, obj_T_hand):
     """

     :param world:
     :param robot:
     :param obj:
     :param obj_T_hand: desired transform from object to hand (klampt.se3)
     :return: True if the desired pose is in collision with the object or the environment
     """
     pass