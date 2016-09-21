
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
# import os
# import time
import sys
# import grasp_chose
#import an object dataset
from klampt.math import so3

world = WorldModel()

world.loadElement("terrains/plane.env")	#file che richiama la mesh del piano

# carico il robot
moving_base_template_fn = 'moving_base_template.rob'
robotname = "reflex_col"
robot_files = {
	'reflex_col':'reflex_col.rob'
}

f = open(moving_base_template_fn,'r')
pattern_2 = ''.join(f.readlines())
f.close()
f2 = open("temp.rob",'w')
f2.write(pattern_2 % (robot_files[robotname],robotname))
f2.close()
world.loadElement("temp.rob")
robot =  world.robot(world.numRobots()-1)
q = robot.getConfig()
q[0] = 0.05
q[1] = 0.01
q[2] = 0.3
q[3] = 0 #yaw
q[4] = math.radians(45) #pitch
q[5] = math.radians(180) #roll
robot.setConfig(q)


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
# R = so3.rotation([1,0,0],math.radians(180))
# T = [0.1,0,0.2]
# print("make "), robot.link(5).getTransform() #fornisce rotazione e traslazione come se fosse un unico vettore

# p = (R,T) #concateno la matrice di rotazione e la traslazione creo s03
# print("p"), p

# robot.link[5].setTransform(so3.rotation([1,0,0],180*180/3.14))
# sim.setController(sim.controller(0).model(),robot.link(5).getTransform() )

sim.setController(robot,simple_controller.make(sim,hand,program.dt))	#serve per tenere la mano dove voglio io

# q = robot.getCommandedConfig()
# for i in range(3):
# 	q[i] = T[i]
# roll,pitch,yaw = so3.rpy(R)
# q[3]=yaw
# q[4]=pitch
# q[5]=roll
# robot.setLinear(q,0.005)























# #now the simulation is launched
# program = GLSimulationProgram(world)
# sim = program.sim




#this code manually updates the visualization
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




