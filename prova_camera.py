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
from moving_base_control import *
# from Add_variation_camera_pose import Make_camera_poses
import scipy.misc
from Add_variation_camera_pose import Add_variation,Make_camera_poses
from utils_camera import FromCamera2rgb, Find_axis_rotation

def Read_Poses(nome,vector_set):

    obj_dataset = '3DCNN/NNSet/Pose/pose/%s.csv'%(nome)
    with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
        file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
        for row in file_reader:
            T = row[9:12]
            pp = row[:9]
            vector_set.append( np.array((pp,T))) 




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
world.readFile("data/robots/test_sensor.xml")
world.loadElement("data/terrains/plane.env")
# embed()
# robot = world.robot(0)
# robot = make_moving_base_robot("reflex_col", world)
# xform = resource.get("default_initial_reflex_col.xform" , description="Initial hand transform",default=se3.identity(), world=world, doedit=False)
# # pose_se3= ([0.983972802704,-0.0441290216922,0.172771968165,0.177866245396,0.173919267423,-0.968563723855,0.0126933954459,0.983770663246,0.178980892412],[0.234435798004,0.0102866113634,0.0952616290142])
# set_moving_base_xform(robot,so3.identity(),[0,0,-8])




obj = make_object('princeton', 'xwing', world)


o_T_p= []
 # = []
Read_Poses('xwing',o_T_p)
o_T_p_r = Make_camera_poses(o_T_p,obj)

vis.add("world",world)

sim = Simulator(world)
sensor = sim.controller(0).sensor("rgbd_camera")
print"LINK", sensor.getSetting("link")
print "Tsensor", sensor.getSetting("Tsensor")

def Write_image(camera,dataset):
    '''Write the dataset'''
    # embed()
    import csv
    f = open(dataset, 'w')
    # embed()
    for i in camera:
        f.write(','.join([str(i)]))
        f.write('\n')
    f.close()

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
        # self.sim = None
        self.step = 0
        # self.camera = None
        self.nome_obj = object_name
        self.t_0 = None
        self.simulation_ = None
        # robot.randomizeConfig()
        #sensor.kinematicSimulate(world,0.01)
        # sim.controller(0).setPIDCommand(robot.getConfig(),[0.0]*7)

    def idle(self):
        print "Idle..."
        #this uses the simulation
        # sim.simulate(0.1)
        # sim.updateWorld()
        # # time.sleep(0.1)
        # #this commented out line just uses the world and kinematic simulation
        # #sensor.kinematicSimulate(world,0.01)
        # processDepthSensor(sensor)
        # camera_measure = sensor.getMeasurements()
        # rgb = FromCamera2rgb(camera_measure)
        # scipy.misc.imsave('outfile_%s.jpg'%self.p, rgb)
        # res_dataset = '2DCNN/NNSet/Image/%s_rotate_%s.csv'% ('xwing',self.p)
        # Write_image(camera_measure,res_dataset)
        # self.p +=1
        #         # o_T_p.pop(0)
        # return True
        if not self.running:
            return

        if not self.is_simulating:
            if len(self.poses) > 0:
                self.curr_pose = self.poses.pop(0)

                # print  self.curr_pose
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
            sensor.setSetting("Tsensor",' '.join(str(v) for v in self.curr_pose[0]+self.curr_pose[1]))
            vis.add("sensor",sensor)
            

            # self.camera.setSetting("Tsensor",Tsensor)
            sim.simulate(0.1)
            sim.updateWorld()

            if not vis.shown() or (sim.getTime() - self.t_0) >= 2.5:
                if  vis.shown():
                    camera_measure = sensor.getMeasurements()
                    rgb = FromCamera2rgb(camera_measure)
                    scipy.misc.imsave('outfile_%s.jpg'%self.step, rgb)
                    res_dataset = '2DCNN/NNSet/Image/%s_rotate_%s.csv'% (self.nome_obj,self.step)
                    Write_image(camera_measure,res_dataset)
                    self.step +=1
                    self.is_simulating = False
                    self.simulation_  = None


vis.pushPlugin(SensorTestWorld(o_T_p_r,world,'xwing'))
vis.show()
while vis.shown():
    time.sleep(0.5)
vis.kill()
