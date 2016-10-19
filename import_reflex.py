from klampt.vis.glrobotprogram import * #Per il simulatore
# from klampt.math import *
from klampt.model import collide
# from klampt.io import resource
from klampt.sim import *
# # from moving_base_control import *
import importlib
# import os
# import time
import sys
from klampt import vis 
# import grasp_chose
#import an object dataset
from klampt.math import so3

world = WorldModel()

moving_base_template_fn = 'data/robots/moving_base_template.rob'
robots = ['reflex_col', 'soft_hand', 'reflex']


robotname = 'reflex_col'
robot_files = {
    'reflex_col':'data/robots/reflex_col.rob',
    'soft_hand':'data/robots/soft_hand.urdf',
    'reflex':'data/robots/reflex.rob'
}

world = WorldModel()
world.loadElement("data/terrains/plane.env")


def make_moving_base_robot(robotname,world):
    """Converts the given fixed-base robot into a moving base robot
    and loads it into the given world.
    """
    f = open(moving_base_template_fn,'r')
    pattern = ''.join(f.readlines())
    f.close()
    f2 = open("temp.rob",'w')
    f2.write(pattern
        % (robot_files[robotname],robotname))
    f2.close()
    world.loadElement("temp.rob")
    return world.robot(world.numRobots()-1)





robot = make_moving_base_robot(robotname,world)
q = robot.getConfig()
q[0] = 0.05
q[1] = 0.01
q[2] = 0.3
q[3] = 0 #yaw
q[4] = math.radians(45) #pitch
q[5] = math.radians(180) #roll
robot.setConfig(q)
n = robot.numLinks()

for i in range(0,n):
    print "n link",i ,"name", robot.link(i).getName()
    print "link pose", robot.link(i).getTransform()[1][0]
    for j in range(1,3):
        if robot.link(i).getName() == 'distal_pad_'+str(j):
            print "robot.link(i).getName()"



#now the simulation is launched
program = GLSimulationProgram(world)
sim = program.sim


    

#create a hand emulator from the given robot name
module = importlib.import_module('plugins.'+robotname)
#emulator takes the robot index (0), start link index (6), and start driver index (6)
hand = module.HandEmulator(sim,0,6,6)
sim.addEmulator(0,hand)
print("EMULATORE")
import simple_controller
sim.setController(robot,simple_controller.make(sim,hand,program.dt))    #serve per tenere la mano dove voglio io


vis.add("world",world)
vis.show()
t0 = time.time()
while vis.shown():
    vis.lock()
    sim.simulate(0.01)
    sim.updateWorld()
    vis.unlock()
    t1 = time.time()
    time.sleep(max(0.01-(t1-t0),0.001))
    t0 = t1




