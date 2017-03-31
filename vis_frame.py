from klampt import *
#Klampt v0.6.x
#from klampt import visualization as vis
# from klampt import resource
#from klampt import robotcollide as collide
#from klampt.simulation import *
#from klampt.glrobotprogram import *
#Klampt v0.7.x
from klampt import vis 
from klampt.vis.glrobotprogram import * #Per il simulatore
# from klampt.math import *
from klampt.model import collide
# from klampt.io import resource
from klampt.sim import *
# # from moving_base_control import *
import importlib
import os
import time
import math
import sys
# import grasp_chose
#import an object dataset
from klampt.math import so3,se3
import numpy as np
from IPython import embed
from mvbb.graspvariation import PoseVariation
from mvbb.TakePoses import SimulationPoses
from mvbb.draw_bbox import draw_GL_frame, draw_bbox
from i16mc import make_object, make_moving_base_robot
from mvbb.CollisionCheck import CheckCollision, CollisionTestInterpolate, CollisionTestPose, CollisionCheckWordFinger
from mvbb.db import MVBBLoader
from mvbb.kindness import Differential,RelativePosition
from mvbb.GetForces import get_contact_forces_and_jacobians
from mvbb.ScalaReduce import DanyReduceScale
from klampt.vis.glprogram import GLRealtimeProgram

from klampt.io import resource
from klampt.sim import *
from moving_base_control import *
#Declare all variables



moving_base_template_fn = 'moving_base_template.rob'
robotname = 'soft_hand'
# "reflex_col"
robot_files = {
    'reflex_col':'data/robots/reflex_col.rob',
    'soft_hand':'data/robots/soft_hand.urdf',
}



class PoseVisualizer(GLRealtimeProgram):
    def __init__(self, robot,poses,world):
        GLRealtimeProgram.__init__(self, 'PoseVisualizer')
        self.world = world
        self.poses = poses
        self.robot = robot
        self.sim = SimpleSimulator(world)
        
    def display(self):
        self.world.drawGL()
        R,t = self.poses
        ##To soft-hand
        # print R
        # print t
        # rot_z = so3.from_axis_angle(([0,0,1],math.radians(90)))
        # rot_x = so3.from_axis_angle(([1,0,0],math.radians(90)))
        # r = so3.mul(rot_x,so3.mul(R, rot_z))
        # t = [0,-0.05,0]
        # pose = r,t
                # bmin,bmax = self.robot.geometry().getBB()
        # centerX = 0.5 * ( bmax[0] - bmin[0] ) +t[0] 
        # centerY = 0.5 * ( bmax[1] - bmin[1] ) + t[1]
        # centerZ = 0.5 * ( bmax[2] - bmin[2] ) + t[2]
        # P = R,[centerX,centerY,centerZ]
        draw_GL_frame(pose)
        # draw_GL_frame(self.robot.getConfig())


    def idle(self):
        pass





def MainDany():
    world = WorldModel()
    # world.loadElement("data/terrains/plane.env")
    robot = make_moving_base_robot(robotname, world)
    xform = resource.get("default_initial_%s.xform" % robotname, description="Initial hand transform",default=se3.identity(), world=world, doedit=False)
    # pose_se3= ([0.983972802704,-0.0441290216922,0.172771968165,0.177866245396,0.173919267423,-0.968563723855,0.0126933954459,0.983770663246,0.178980892412],[0.234435798004,0.0102866113634,0.0952616290142])
    set_moving_base_xform(robot,xform[0],xform[1])
    # set_moving_base_xform(robot, pose_se3[0], pose_se3[1])

    # set_moving_base_xform(robot, pose_se3[0], pose_se3[1])
    #now the simulation is launched
    program = PoseVisualizer(robot,xform,world)
    vis.setPlugin(program)
    sim = program.sim
    #this code manually updates the visualization
    vis.add("world",world)
    vis.show()
    t0 = time.time()
    # Pos_ = RelativePosition(robot,object) 
    # kindness = 0 
    # Td_prev = 0
    while vis.shown():
        vis.lock()
        sim.simulate(0.01)
        sim.updateWorld()
        vis.unlock()
        t1 = time.time()
        time.sleep(max(0.01-(t1-t0),0.001))
        t0 = t1 


#Main
if __name__ == '__main__':
    MainDany()

