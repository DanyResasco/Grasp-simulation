from klampt import *
#Klampt v0.6.x
#from klampt import visualization as vis
# from klampt import resource
#from klampt import robotcollide as collide
#from klampt.simulation import *
#from klampt.glrobotprogram import *
#Klampt v0.7.x
from klampt import vis 
from klampt.vis.glrobotprogram import *	#Per il simulatore
# from klampt.math import *
from klampt.model import collide
# from klampt.io import resource
from klampt.sim import *
# # from moving_base_control import *
import importlib
import os
# import time
import sys
# import grasp_chose
#import an object dataset
from klampt.math import so3
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

from klampt.io import resource
from klampt.sim import *
from moving_base_control import *
#Declare all variables


# set di oggetti
object_template_fn = 'object_template.obj'	
objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
# objects['newObjdany'] = [f for f in os.listdir('data/objects/newObjdany')]
objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]
# robots = ['reflex_col', 'soft_hand', 'reflex']


moving_base_template_fn = 'moving_base_template.rob'
robotname = "reflex_col"
robot_files = {
	'reflex_col':'reflex_col.rob'
}


def move_reflex(robot):
	q = robot.getConfig()
	q[0] = 0.05
	q[1] = 0.01
	q[2] = 0.18
	q[3] = 0 #yaw
	q[4] = 0#pitch
	q[5] = math.radians(180) #roll
	robot.setConfig(q)
	print("******* Robot dove sono **********"), robot.getConfig()

class PoseVisualizer(GLNavigationProgram):
    def __init__(self, obj,world):
        GLNavigationProgram.__init__(self, 'PoseVisualizer')
        self.world = world
        # self.poses = poses
        # self.robot = robot
        self.obj = obj

    def display(self):
        self.world.drawGL()
        R,t = self.obj.getTransform()
        bmin,bmax = self.obj.geometry().getBB()
        centerX = 0.5 * ( bmax[0] - bmin[0] ) +t[0] 
        centerY = 0.5 * ( bmax[1] - bmin[1] ) + t[1]
        centerZ = 0.5 * ( bmax[2] - bmin[2] ) + t[2]
        P = R,[centerX,centerY,centerZ]
        draw_GL_frame(P)
        # draw_GL_frame(self.robot.getConfig())


    def idle(self):
        pass





def MainDany(object_list):
    world = WorldModel()
    world.loadElement("data/terrains/plane.env")
    robot = make_moving_base_robot(robotname, world)
    xform = resource.get("default_initial_%s.xform" % robotname, description="Initial hand transform",default=se3.identity(), world=world, doedit=False)
    # pose_se3= ([0.983972802704,-0.0441290216922,0.172771968165,0.177866245396,0.173919267423,-0.968563723855,0.0126933954459,0.983770663246,0.178980892412],[0.234435798004,0.0102866113634,0.0952616290142])
    set_moving_base_xform(robot,xform[0],xform[1])
    # set_moving_base_xform(robot, pose_se3[0], pose_se3[1])
    for object_name in object_list:
        obj = None
        for object_set, objects_in_set in objects.items():
            if object_name in objects_in_set:
                if world.numRigidObjects() > 0:
                    world.remove(world.rigidObject(0))
                if object_name in objects['princeton']:
                    # print "*************Dentro princeton********************" #need to scale the obj size
                    objfilename = 'data/objects/template_obj_scale_princeton.obj'
                    # print"objfilename", objfilename
                    obj = DanyReduceScale(object_name, world,objfilename,object_set)
                else:
                    obj = make_object(object_set, object_name, world)
        if obj is None:
            print "Could not find object", object_name
            continue
	#Simulation 
	# set_moving_base_xform(robot, pose_se3[0], pose_se3[1])
	#now the simulation is launched
    program = GLSimulationProgram(world)
    vis.setPlugin(PoseVisualizer(obj,world))
    sim = program.sim
    # sim = SimpleSimulator(world)
    sim.simulate(0)

    camera = (sim.controller(0).sensor('rgbd_camera')).getMeasurements()
    sim.updateWorld()
    embed()
    


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
    all_objects = []
    for dataset in objects.values():
        all_objects += dataset


    to_filter = []
    done = []
    to_check = []


    for obj_name in to_filter +  done + to_check:
        all_objects.pop(all_objects.index(obj_name))

    print "-------------"
    print all_objects
    print "-------------"

    try:
        objname = sys.argv[1]
        # launch_test_mvbb_filtered("soft_hand", [objname], 100)
        MainDany([objname])
    except:
        # launch_test_mvbb_filtered("soft_hand", all_objects, 100)
        MainDany(all_objects)


