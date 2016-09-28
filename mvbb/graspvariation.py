#!/usr/bin/env python
import numpy as np


def PoseVariation(pose, longSide):

    I = np.eye(4)

    # rotations
    import math
    
    degree = 10*math.pi/180;   # angle in rad
    R = np.eye(4)
    
    rot_x = np.array([[1, 0, 0],[0, math.cos(degree), -math.sin(degree)],[ 0, math.sin(degree), math.cos(degree)]])
    R[0:3,0:3] = rot_x

    dim = longSide / 5.0

    listVarPose = []
    listVarPose.append(pose)
    listVarPose.append(pose.dot(R)) 

    I[0:3,3] = np.array([[dim,0,0]])
    listVarPose.append(pose.dot(I))
    listVarPose.append(pose.dot(I).dot(R))

    I[0:3,3] = np.array([[2*dim,0,0]])
    listVarPose.append(pose.dot(I))
    listVarPose.append(pose.dot(I).dot(R))

    I[0:3,3] = np.array([[-dim,0,0]])
    listVarPose.append(pose.dot(I))
    listVarPose.append(pose.dot(I).dot(R))

    I[0:3,3] = np.array([[-2*dim,0,0]])
    listVarPose.append(pose.dot(I))
    listVarPose.append(pose.dot(I).dot(R))

    I[0:3,3] = np.array([[3*dim,0,0]])
    listVarPose.append(pose.dot(I))
    listVarPose.append(pose.dot(I).dot(R))

    return listVarPose
