import pkg_resources
pkg_resources.require("klampt>=0.7.0")
from klampt import vis
from klampt import *
from klampt.math import so3,se3,vectorops
from klampt.vis.glcommon import *
from klampt.io import resource
import time
from IPython import embed
from i16mc import make_object, make_moving_base_robot
import csv
import numpy as np
import math
import os
import string
import sys
from moving_base_control import *
# from Add_variation_camera_pose import Make_camera_poses
import scipy.misc
from Add_variation_camera_pose import Add_variation, Make_camera_poses
from utils_camera import FromCamera2rgb, Find_axis_rotation, Read_Poses, Write_image
from mvbb.ScalaReduce import DanyReduceScale


Pose = {}
Pose['pose'] = [f for f in os.listdir('3DCNN/NNSet/Pose/pose')]
objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]





def processDepthSensor(sensor):
    data = sensor.getMeasurements()
    # print data
    w = int(sensor.getSetting("xres"))
    h = int(sensor.getSetting("yres"))
    #first, RGB then depth
    mind,maxd = float('inf'),float('-inf')
    for i in range(h):
        for j in range(w):
            pixelofs = (j+i*w)
            rgb = int(data[pixelofs])
            depth = data[pixelofs+w*h]
            mind = min(depth,mind)
            maxd = max(depth,maxd)
    print "Depth range",mind,maxd


world = WorldModel()

# world.loadElement("data/terrains/plane.env")
robot = make_moving_base_robot('reflex_col', world)
xform = resource.get("default_initial_reflex_col.xform" , description="Initial hand transform",
                        default=se3.identity(), world=world, doedit=False)
set_moving_base_xform(robot,xform[0],[0,0,-1])

all_objects = []
object_list = []
for dataset in objects.values():
    all_objects += dataset

to_check = []
done =[]
to_filter=[]
to_do=[]

for obj_name in to_filter + to_do + done + to_check:
    all_objects.pop(all_objects.index(obj_name))

# print "-------------"
# print all_objects
# print "-------------"

try:
    object_list.append(sys.argv[1])
except:
    object_list = (all_objects)


for object_name in object_list:
    obj = None
    for object_set, objects_in_set in objects.items():
        if object_name in objects_in_set:
            for i,t in Pose.items():
                if (object_name +'.csv') in t:
                    if world.numRigidObjects() > 0:
                        world.remove(world.rigidObject(0))
                    if object_name in objects['princeton']:
                        print "*************Dentro princeton********************" #need to scale the obj size
                        objfilename = 'data/objects/template_obj_scale_princeton.obj'
                        print"objfilename", objfilename
                        obj = DanyReduceScale(object_name, world,objfilename,object_set)
                    else:
                        obj = make_object(object_set, object_name, world)
                if obj is None:
                    print "Could not find object", object_name
                    continue
                # embed()
                o_T_p= []

                Read_Poses(object_name,o_T_p)
                # embed()
                o_T_p_r = Make_camera_poses(o_T_p,obj)

                vis.add("world",world)

                sim = Simulator(world)
                sim.setGravity([0,0,0])
                sensor = sim.controller(0).sensor("rgbd_camera")
                print"LINK", sensor.getSetting("link")
                print "Tsensor", sensor.getSetting("Tsensor")

                #Note: GLEW sensor simulation only runs if it occurs in the visualization thread (e.g., the idle loop)
                class SensorTestWorld (GLPluginInterface):
                    def __init__(self,poses,world,object_name):
                        self.p = 0
                        self.poses = poses
                        self.world = world
                        self.is_simulating = False
                        self.curr_pose = None
                        self.running = True
                        self.obj = obj
                        self.curr_pose = None
                        self.step = 0
                        self.nome_obj = object_name
                        self.t_0 = None
                        self.simulation_ = None

                    def idle(self):
                        print "Idle..."
                        if not self.running:
                            return

                        if not self.is_simulating:
                            if len(self.poses) > 0:
                                self.curr_pose = self.poses.pop(0)
                                print len(self.poses)
                            else:
                                self.running = False
                                vis.show(hidden=True)
                                return

                            if self.simulation_ is None:
                                vis.add("world",self.world)
                            self.t_0 = sim.getTime()
                            self.is_simulating = True

                        if self.is_simulating:
                            obj.setVelocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                            sensor.setSetting("Tsensor",' '.join(str(v) for v in self.curr_pose[0]+self.curr_pose[1]))
                            vis.add("sensor",sensor)
                            
                            sim.simulate(0.1)
                            sim.updateWorld()

                            if not vis.shown() or (sim.getTime() - self.t_0) >= 2.5:
                                if  vis.shown():
                                    camera_measure = sensor.getMeasurements()
                                    image,rgb = FromCamera2rgb(camera_measure)
                                    scipy.misc.imsave('outfile_%s.jpg'%self.step, image)
                                    res_dataset = '2DCNN/NNSet/Image/%s_rotate_%s.csv'% (self.nome_obj,self.step)
                                    Write_image(rgb,res_dataset)
                                    self.step +=1
                                    self.is_simulating = False
                                    self.simulation_  = None


                vis.pushPlugin(SensorTestWorld(o_T_p_r,world,'xwing'))
                vis.show()
                while vis.shown():
                    time.sleep(0.5)
                vis.kill()