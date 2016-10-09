import pkg_resources
pkg_resources.require("klampt>=0.7.0")
from klampt import *
#Klampt v0.6.x
#from klampt import visualization as vis
#from klampt import resource
#from klampt import robotcollide as collide
#from klampt.simulation import *
#from klampt.glrobotprogram import *
#Klampt v0.7.x
from klampt import vis 
from klampt.vis.glrobotprogram import *
from klampt.math import *
from klampt.model import collide
from klampt.io import resource
from klampt.sim import *
from moving_base_control import *
import importlib
import os
import time
import sys

box_dims = (0.5,0.5,0.3)
shelf_dims = (0.4,0.4,0.3)
shelf_offset = 0.6
shelf_height = 0.7
moving_base_template_fn = 'data/robots/moving_base_template.rob'
object_template_fn = 'data/objects/object_template.obj'
objects = {}
objects['ycb'] = [f for f in sorted(os.listdir('data/objects/ycb'))]
objects['apc2015'] = [f for f in sorted(os.listdir('data/objects/apc2015'))]
robots = ['reflex_col', 'soft_hand', 'reflex']

object_geom_file_patterns = {
	'ycb':['data/objects/ycb/%s/meshes/tsdf_mesh.stl','data/objects/ycb/%s/meshes/poisson_mesh.stl'],
	'apc2015':['data/objects/apc2015/%s/textured_meshes/optimized_tsdf_textured_mesh.ply']
}
#default mass for objects whose masses are not specified, in kg
default_object_mass = 0.5
object_masses = {
	'ycb':dict(),
	'apc2015':dict(),
}
robot_files = {
	'reflex_col':'data/robots/reflex_col.rob',
	'soft_hand':'data/robots/soft_hand.urdf',
	'reflex':'data/robots/reflex.rob'
}


def mkdir_p(path):
	"""Quietly makes the directories in path"""
	import os, errno
	try:
		os.makedirs(path)
	except OSError as exc: # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else: raise

def make_object(object_set,objectname,world):
	"""Adds an object to the world using its geometry / mass properties
	and places it in a default location (x,y)=(0,0) and resting on plane."""
	for pattern in object_geom_file_patterns[object_set]:
		objfile = pattern%(objectname,)
		objmass = object_masses[object_set].get('mass',default_object_mass)
		f = open(object_template_fn,'r')
		pattern = ''.join(f.readlines())
		f.close()
		f2 = open("temp.obj",'w')
		f2.write(pattern % (objfile,objmass))
		f2.close()
		nobjs = world.numRigidObjects()
		if world.loadElement('temp.obj') < 0 :
			continue
		assert nobjs < world.numRigidObjects(),"Hmm... the object didn't load, but loadElement didn't return -1?"
		obj = world.rigidObject(world.numRigidObjects()-1)
		obj.setTransform(*se3.identity())
		bmin,bmax = obj.geometry().getBB()
		T = obj.getTransform()
		spacing = 0.006
		T = (T[0],vectorops.add(T[1],(-(bmin[0]+bmax[0])*0.5,-(bmin[1]+bmax[1])*0.5,-bmin[2]+spacing)))
		obj.setTransform(*T)
		obj.appearance().setColor(0.2,0.5,0.7,1.0)
		obj.setName(objectname)
		return obj
	raise RuntimeError("Unable to load object name %s from set %s"%(objectname,object_set))

def make_box(world,width,depth,height,wall_thickness=0.005,mass=float('inf')):
	"""Makes a new axis-aligned box centered at the origin with
	dimensions width x depth x height. Walls have thickness wall_thickness. 
	If mass=inf, then the box is a Terrain, otherwise it's a RigidObject
	with automatically determined inertia.
	"""
	left = Geometry3D()
	right = Geometry3D()
	front = Geometry3D()
	back = Geometry3D()
	bottom = Geometry3D()
	left.loadFile("data/objects/cube.tri")
	right.loadFile("data/objects/cube.tri")
	front.loadFile("data/objects/cube.tri")
	back.loadFile("data/objects/cube.tri")
	bottom.loadFile("data/objects/cube.tri")
	left.transform([wall_thickness,0,0,0,depth,0,0,0,height],[-width*0.5,-depth*0.5,0])
	right.transform([wall_thickness,0,0,0,depth,0,0,0,height],[width*0.5,-depth*0.5,0])
	front.transform([width,0,0,0,wall_thickness,0,0,0,height],[-width*0.5,-depth*0.5,0])
	back.transform([width,0,0,0,wall_thickness,0,0,0,height],[-width*0.5,depth*0.5,0])
	bottom.transform([width,0,0,0,depth,0,0,0,wall_thickness],[-width*0.5,-depth*0.5,0])
	#bottom.setAABB([-width*0.5,-depth*0.5,0],[width*0.5,depth*0.5,wall_thickness])
	#left.setAABB([-width*0.5,-depth*0.5,0],[-width*0.5+wall_thickness,depth*0.5,height])
	#right.setAABB([width*0.5-wall_thickness,-depth*0.5,0],[width*0.5,depth*0.5,height])
	#front.setAABB([-width*0.5,-depth*0.5,0],[width*0.5,-depth*0.5+wall_thickness,height])
	#back.setAABB([-width*0.5,depth*0.5-wall_thickness,0],[width*0.5,depth*0.5,height])
	boxgeom = Geometry3D()
	boxgeom.setGroup()
	for i,elem in enumerate([left,right,front,back,bottom]):
		g = Geometry3D(elem)
		boxgeom.setElement(i,g)
	if mass != float('inf'):
		print "Making a box a rigid object"
		bmass = Mass()
		bmass.setMass(mass)
		bmass.setCom([0,0,height*0.3])
		bmass.setInertia([width/12,depth/12,height/12])
		box = world.makeRigidObject("box")
		box.geometry().set(boxgeom)
		box.appearance().setColor(0.6,0.3,0.2,1.0)
		box.setMass(bmass)
		return box
	else:
		print "Making a box a terrain"
		box = world.makeTerrain("box")
		box.geometry().set(boxgeom)
		box.appearance().setColor(0.6,0.3,0.2,1.0)
		return box

def make_shelf(world,width,depth,height,wall_thickness=0.005):
	"""Makes a new axis-aligned "shelf" centered at the origin with
	dimensions width x depth x height. Walls have thickness wall_thickness. 
	If mass=inf, then the box is a Terrain, otherwise it's a RigidObject
	with automatically determined inertia.
	"""
	left = Geometry3D()
	right = Geometry3D()
	back = Geometry3D()
	bottom = Geometry3D()
	top = Geometry3D()
	left.loadFile("data/objects/cube.tri")
	right.loadFile("data/objects/cube.tri")
	back.loadFile("data/objects/cube.tri")
	bottom.loadFile("data/objects/cube.tri")
	top.loadFile("data/objects/cube.tri")
	left.transform([wall_thickness,0,0,0,depth,0,0,0,height],[-width*0.5,-depth*0.5,0])
	right.transform([wall_thickness,0,0,0,depth,0,0,0,height],[width*0.5,-depth*0.5,0])
	back.transform([width,0,0,0,wall_thickness,0,0,0,height],[-width*0.5,depth*0.5,0])
	bottom.transform([width,0,0,0,depth,0,0,0,wall_thickness],[-width*0.5,-depth*0.5,0])
	top.transform([width,0,0,0,depth,0,0,0,wall_thickness],[-width*0.5,-depth*0.5,height-wall_thickness])
	shelfgeom = Geometry3D()
	shelfgeom.setGroup()
	for i,elem in enumerate([left,right,back,bottom,top]):
		g = Geometry3D(elem)
		shelfgeom.setElement(i,g)
	shelf = world.makeTerrain("shelf")
	shelf.geometry().set(shelfgeom)
	shelf.appearance().setColor(0.2,0.6,0.3,1.0)
	return shelf

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



def launch_simple(robotname,object_set,objectname,use_box=False):
	"""Launches a very simple program that simulates a robot grasping an object from one of the
	databases. It first allows a user to position the robot's free-floating base in a GUI. 
	Then, it sets up a simulation with those initial conditions, and launches a visualization.
	The controller closes the hand, and then lifts the hand upward.  The output of the robot's
	tactile sensors are printed to the console.

	If use_box is True, then the test object is placed inside a box.
	"""
	world = WorldModel()
	world.loadElement("data/terrains/plane.env")
	robot = make_moving_base_robot(robotname,world)
	object = make_object(object_set,objectname,world)
	if use_box:
		box = make_box(world,*box_dims)
		object.setTransform(*se3.mul((so3.identity(),[0,0,0.01]),object.getTransform()))
	doedit = True
	xform = resource.get("%s/default_initial_%s.xform"%(object_set,robotname),description="Initial hand transform",default=robot.link(5).getTransform(),world=world)
	set_moving_base_xform(robot,xform[0],xform[1])
	xform = resource.get("%s/initial_%s_%s.xform"%(object_set,robotname,objectname),description="Initial hand transform",default=robot.link(5).getTransform(),world=world,doedit=False)
	if xform:
		set_moving_base_xform(robot,xform[0],xform[1])
	xform = resource.get("%s/initial_%s_%s.xform"%(object_set,robotname,objectname),description="Initial hand transform",default=robot.link(5).getTransform(),world=world,doedit=doedit)
	if not xform:
		print "User quit the program"
		return
	#this sets the initial condition for the simulation
	set_moving_base_xform(robot,xform[0],xform[1])

	#now the simulation is launched
	program = GLSimulationProgram(world)
	sim = program.sim

	#setup some simulation parameters
	visPreshrink = True   #turn this to true if you want to see the "shrunken" models used for collision detection
	for l in range(robot.numLinks()):
		sim.body(robot.link(l)).setCollisionPreshrink(visPreshrink)
	for l in range(world.numRigidObjects()):
		sim.body(world.rigidObject(l)).setCollisionPreshrink(visPreshrink)

	#create a hand emulator from the given robot name
	module = importlib.import_module('plugins.'+robotname)
	#emulator takes the robot index (0), start link index (6), and start driver index (6)
	hand = module.HandEmulator(sim,0,6,6)
	sim.addEmulator(0,hand)

	#the result of simple_controller.make() is now attached to control the robot
	import simple_controller
	sim.setController(robot,simple_controller.make(sim,hand,program.dt))

	#the next line latches the current configuration in the PID controller...
	sim.controller(0).setPIDCommand(robot.getConfig(),robot.getVelocity())
	
	#this code uses the GLSimulationProgram structure, which gives a little more control over the visualization
	"""
	program.simulate = True
	vis.setPlugin(program)
	vis.show()
	while vis.shown():
		time.sleep(0.1)
	return
	"""

	wait_for_setup = 7.5
	wait_after_lift = 2.5

	#this code manually updates the visualization
	vis.add("world",world)
	vis.show()

	print "waiting for the user to setup viewport"
	time.sleep(wait_for_setup)

	t0 = time.time()
	while vis.shown():
		if sim.getTime() <= wait_after_lift \
				and not program.saveScreenshots:
			program.saveScreenshots = True
			program.nextScreenshotTime = sim.getTime()
			print "Started recording"

		if sim.getTime() >= wait_after_lift and program.saveScreenshots:
			program.saveScreenshots = False
			print "Stopped recording"
		vis.lock()
		sim.simulate(0.01)
		sim.updateWorld()
		if program.saveScreenshots and sim.getTime() >= program.nextScreenshotTime:
			program.save_screen("image%04d.ppm" % (program.screenshotCount))
			program.screenshotCount += 1
			program.nextScreenshotTime += 1.0 / 30.0;
		vis.unlock()

		t1 = time.time()
		time.sleep(max(0.01-(t1-t0),0.001))
		t0 = t1
	return



def launch_balls(robotname,num_balls=10):
	"""Launches a very simple program that simulates a robot grasping an object from one of the
	databases. It first allows a user to position the robot's free-floating base in a GUI. 
	Then, it sets up a simulation with those initial conditions, and launches a visualization.
	The controller closes the hand, and then lifts the hand upward.  The output of the robot's
	tactile sensors are printed to the console.
	"""
	world = WorldModel()
	world.loadElement("data/terrains/plane.env")
	maxlayer = 16
	numlayers = int(math.ceil(float(num_balls)/float(maxlayer)))
	for layer in xrange(numlayers):
		if layer + 1 == numlayers:
			ballsperlayer = num_balls%maxlayer
		else:
			ballsperlayer = maxlayer
		w = int(math.ceil(math.sqrt(ballsperlayer)))
		h = int(math.ceil(float(ballsperlayer)/float(w)))
		for i in xrange(ballsperlayer):
			bid = world.loadElement("data/objects/sphere_10cm.obj")
			if bid < 0:
				raise RuntimeError("data/objects/sphere_10cm.obj couldn't be loaded")
			ball = world.rigidObject(world.numRigidObjects()-1)
			R = so3.identity()
			x = i % w
			y = i / w
			x = (x - (w-1)*0.5)*box_dims[0]*0.7/(w-1)
			y = (y - (h-1)*0.5)*box_dims[1]*0.7/(h-1)
			t = [x,y,0.08 + layer*0.11]
			t[0] += random.uniform(-0.005,0.005)
			t[1] += random.uniform(-0.005,0.005)
			ball.setTransform(R,t)
	robot = make_moving_base_robot(robotname,world)
	box = make_box(world,*box_dims)
	box2 = make_box(world,*box_dims)
	box2.geometry().translate((0.7,0,0))
	xform = resource.get("balls/default_initial_%s.xform"%(robotname,),description="Initial hand transform",default=robot.link(5).getTransform(),world=world,doedit=True)
	if not xform:
		print "User quit the program"
		return
	#this sets the initial condition for the simulation
	set_moving_base_xform(robot,xform[0],xform[1])

	#now the simulation is launched
	program = GLSimulationProgram(world)  
	sim = program.sim

	#setup some simulation parameters
	visPreshrink = True   #turn this to true if you want to see the "shrunken" models used for collision detection
	for l in range(robot.numLinks()):
		sim.body(robot.link(l)).setCollisionPreshrink(visPreshrink)
	for l in range(world.numRigidObjects()):
		sim.body(world.rigidObject(l)).setCollisionPreshrink(visPreshrink)

	#create a hand emulator from the given robot name
	module = importlib.import_module('plugins.'+robotname)
	#emulator takes the robot index (0), start link index (6), and start driver index (6)
	hand = module.HandEmulator(sim,0,6,6)
	sim.addEmulator(0,hand)

	#A StateMachineController instance is now attached to control the robot
	import balls_controller
	sim.setController(robot,balls_controller.make(sim,hand,program.dt))

	#the next line latches the current configuration in the PID controller...
	sim.controller(0).setPIDCommand(robot.getConfig(),robot.getVelocity())

	"""
	#this code uses the GLSimulationProgram structure, which gives a little more control over the visualization
	vis.setPlugin(program)
	vis.show()
	while vis.shown():
		time.sleep(0.1)
	return
	"""

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
	return

def xy_randomize(obj,bmin,bmax):
	R,t = obj.getTransform()
	obmin,obmax = obj.geometry().getBB()
	w = 0.5*(obmax[0]-obmin[0])
	h = 0.5*(obmax[1]-obmin[1])
	correction = max(w,h)
	R = so3.mul(so3.rotation([0,0,1],random.uniform(0,math.pi*2)),R)
	t[0] = random.uniform(bmin[0]+correction,bmax[0]-correction)
	t[1] = random.uniform(bmin[1]+correction,bmax[1]-correction)
	obj.setTransform(R,t)

def xy_jiggle(world,objects,fixed_objects,bmin,bmax,iters,randomize = True):
	"""Jiggles the objects' x-y positions within the range bmin - bmax, and randomizes orientation about the z
	axis until the objects are collision free.  A list of fixed objects (fixed_objects) may be given as well.

	Objects for which collision-free resolutions are not found after iters steps will be
	deleted from the world.
	"""
	if randomize:
		for obj in objects:
			xy_randomize(obj,bmin,bmax)
	inner_iters = 10
	while iters > 0:
		numConflicts = [0]*len(objects)
		for (i,j) in collide.self_collision_iter([o.geometry() for o in objects]):
			numConflicts[i] += 1
			numConflicts[j] += 1
		for (i,j) in collide.group_collision_iter([o.geometry() for o in objects],[o.geometry() for o in fixed_objects]):
			numConflicts[i] += 1
		
		amax = max((c,i) for (i,c) in enumerate(numConflicts))[1]
		cmax = numConflicts[amax]
		if cmax == 0:
			#conflict free
			return
		print cmax,"conflicts with object",objects[amax].getName()
		other_geoms = [o.geometry() for o in objects[:amax]+objects[amax+1:]+fixed_objects]
		for it in xrange(inner_iters):
			xy_randomize(objects[amax],bmin,bmax)
			nc = sum([1 for p in collide.group_collision_iter([objects[amax].geometry()],other_geoms)])
			if nc < cmax:
				break
			iters-=1
		print "Now",nc,"conflicts with object",objects[amax].getName()

	numConflicts = [0]*len(objects)
	for (i,j) in collide.self_collision_iter([o.geometry() for o in objects]):
		numConflicts[i] += 1
		numConflicts[j] += 1
	for (i,j) in collide.group_collision_iter([o.geometry() for o in objects],[o.geometry() for o in fixed_objects]):
		numConflicts[i] += 1
	removed = []
	while max(numConflicts) > 0:
		amax = max((c,i) for (i,c) in enumerate(numConflicts))[1]
		cmax = numConflicts[amax]
		print "Unable to find conflict-free configuration, removing object",objects[amax].getName(),"with",cmax,"conflicts"
		removed.append(amax)

		#revise # of conflicts -- this could be faster, but whatever...
		numConflicts = [0]*len(objects)
		for (i,j) in collide.self_collision_iter([o.geometry() for o in objects]):
			if i in removed or j in removed:
				continue
			numConflicts[i] += 1
			numConflicts[j] += 1
		for (i,j) in collide.group_collision_iter([o.geometry() for o in objects],[o.geometry() for o in fixed_objects]):
			if i in removed:
				continue
			numConflicts[i] += 1
	removeIDs = [objects[i].index for i in removed]
	for i in sorted(removeIDs)[::-1]:
		world.remove(world.rigidObject(i))
	raw_input("Press enter to continue")


def launch_shelf(robotname,objects):
	"""Launches the task 2 program that asks the robot to retrieve some set of objects
	packed within a shelf.
	"""
	world = WorldModel()
	world.loadElement("data/terrains/plane.env")
	robot = make_moving_base_robot(robotname,world)
	box = make_box(world,*box_dims)
	shelf = make_shelf(world,*shelf_dims)
	shelf.geometry().translate((0,shelf_offset,shelf_height))
	rigid_objects = []
	for objectset,objectname in objects:
		object = make_object(objectset,objectname,world)
		#TODO: pack in the shelf using x-y translations and z rotations
		object.setTransform(*se3.mul((so3.identity(),[0,shelf_offset,shelf_height + 0.01]),object.getTransform()))
		rigid_objects.append(object)
	xy_jiggle(world,rigid_objects,[shelf],[-0.5*shelf_dims[0],-0.5*shelf_dims[1]+shelf_offset],[0.5*shelf_dims[0],0.5*shelf_dims[1]+shelf_offset],100)

	doedit = True
	xform = resource.get("shelf/default_initial_%s.xform"%(robotname,),description="Initial hand transform",default=robot.link(5).getTransform(),world=world)
	if not xform:
		print "User quit the program"
		return
	set_moving_base_xform(robot,xform[0],xform[1])

	#now the simulation is launched
	program = GLSimulationProgram(world)
	sim = program.sim

	#setup some simulation parameters
	visPreshrink = True   #turn this to true if you want to see the "shrunken" models used for collision detection
	for l in range(robot.numLinks()):
		sim.body(robot.link(l)).setCollisionPreshrink(visPreshrink)
	for l in range(world.numRigidObjects()):
		sim.body(world.rigidObject(l)).setCollisionPreshrink(visPreshrink)

	#create a hand emulator from the given robot name
	module = importlib.import_module('plugins.'+robotname)
	#emulator takes the robot index (0), start link index (6), and start driver index (6)
	hand = module.HandEmulator(sim,0,6,6)
	sim.addEmulator(0,hand)

	#controlfunc is now attached to control the robot
	import shelf_controller
	sim.setController(robot,shelf_controller.make(sim,hand,program.dt))

	#the next line latches the current configuration in the PID controller...
	sim.controller(0).setPIDCommand(robot.getConfig(),robot.getVelocity())
	
	#this code uses the GLSimulationProgram structure, which gives a little more control over the visualization
	vis.setPlugin(program)
	program.reshape(800,600)
	vis.show()
	while vis.shown():
		time.sleep(0.1)
	return
	
	#this code manually updates the vis
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
	return

if __name__ == '__main__':
	import random
	try:
		dataset = sys.argv[1]
	except IndexError:
		dataset = random.choice(objects.keys())

	#choose the robot model here
	robot = "soft_hand"
	#choose the setup here
	if dataset == 'balls':
		try:
			numballs = int(sys.argv[2])
		except IndexError:
			numballs = 10
		launch_balls(robot,numballs)
	elif dataset == 'shelf':
		shelved = []
		if len(sys.argv[2:]) == 0:
			#default: pick 3 random objects
			for i in range(3):
				dataset = random.choice(objects.keys())
				index = random.randint(0,len(objects[dataset])-1)
				objname = objects[dataset][index]
				shelved.append((dataset,objname))
		elif len(sys.argv[2:]) == 1:
			try:
				nobjects = int(sys.argv[2])
				#default: pick n random objects
				for i in range(nobjects):
					dataset = random.choice(objects.keys())
					index = random.randint(0,len(objects[dataset])-1)
					objname = objects[dataset][index]
					shelved.append((dataset,objname))
			except ValueError:
				#format: dataset/object
				for o in sys.argv[2:]:
					dataset,objname = o.split('/',2)
					try:
						index = int(objname)
						objname = objects[dataset][index]
					except:
						pass
					shelved.append((dataset,objname))
		else:
			#format: dataset/object
			for o in sys.argv[2:]:
				dataset,objname = o.split('/',2)
				try:
					index = int(objname)
					objname = objects[dataset][index]
				except:
					pass
				shelved.append((dataset,objname))
		launch_shelf(robot,shelved)
	else:
		#just plan grasping
		try:
			index = int(sys.argv[2])
			objname = objects[dataset][index]
		except IndexError:
			index = random.randint(0,len(objects[dataset])-1)
			objname = objects[dataset][index]
		except ValueError:
			objname = sys.argv[2]
		launch_simple(robot,dataset,objname)
	vis.kill()
