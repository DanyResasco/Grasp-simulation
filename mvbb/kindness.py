#!/usr/bin/env python
from klampt import *
from klampt.vis.glrobotprogram import * #Per il simulatore
import math


#distance between object and gripper
def RelativePosition(robot,object):
    # print "dentro RelativePosition"
    robot_transform = robot.getConfig()
    Robot_position = [robot_transform[0], robot_transform[1],robot_transform[2]]
    object_transform = object.getTransform()
    Pos = vectorops.distance(Robot_position,object_transform[1])
    # print("Pos"), Pos
    return Pos
#make the differential 
def Differential(robot, object, Pos_prev, time):
    # print "dentro diff"
    Pos_actual =  RelativePosition(robot,object)
    Diff = (Pos_actual - Pos_prev) / time
    # print("Derivate"), Diff
    return Diff


#check if grasp is good or not and write the object name, hand position and kindness in file.txt
def GraspValuate(diff,kindness,posedict):
    if diff > 0:
        print("No good grasp")
    else:
        print("good grasp")
        f = open('grasp_valuation_template.rob','r')
        pattern_2 = ''.join(f.readlines())
        f.close()
        nameFile = 'grasp_valuation_' + objectName + '.txt'
        f2 = open(nameFile ,'w')
        pos = robot.getConfig()
        f2.write(pattern_2 % (objectName,pos,kindness,posedict['desired'],posedict['actual']))
        f2.close()

