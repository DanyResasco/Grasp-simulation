from klampt import *
from klampt.math import vectorops,so3,se3
from moving_base_control import *
from reflex_control import *

class StateMachineController(ReflexController):
	"""A more sophisticated controller that uses a state machine."""
	def __init__(self,sim,hand,dt):
		self.sim = sim
		self.hand = hand
		self.dt = dt
		self.sim.updateWorld()
		self.base_xform = get_moving_base_xform(self.sim.controller(0).model())
		self.state = 'idle'

	def __call__(self,controller):
		sim = self.sim
		xform = self.base_xform
		#turn this to false to turn off print statements
		ReflexController.verbose = True
		ReflexController.__call__(self,controller)

		#controller state machine
		#print "State:",state
		t_lower = 0.5
		t_grasp = 1.0
		t_lift = 2
		lift_traj_duration = 0.5
		if self.state == 'idle':
			if sim.getTime() > t_lower:
				self.state = 'lowering'
				print self.state
		elif self.state == 'lowering':
			if sim.getTime() < t_grasp:
				t_traj = min(1, max(0, (sim.getTime() - t_lower) / lift_traj_duration))
				desired = se3.mul((so3.identity(), [0, 0, -0.10 * t_traj]), xform)
				send_moving_base_xform_PID(controller, desired[0], desired[1])
			else:
				#this is needed to stop at the current position in case there's some residual velocity
				controller.setPIDCommand(controller.getCommandedConfig(),[0.0]*len(controller.getCommandedConfig()))
				#the controller sends a command to the hand: f1,f2,f3,preshape
				self.hand.setCommand([0.2,0.2,0.2,0])
				self.state = 'closing'
				print self.state
		elif self.state == 'closing':
			if sim.getTime() > t_lift:
				self.base_xform = get_moving_base_xform(self.sim.controller(0).model())
				# this is needed to stop at the current position in case there's some residual velocity
				controller.setPIDCommand(controller.getCommandedConfig(), [0.0] * len(controller.getCommandedConfig()))
				self.state = 'raising'
				print self.state
		elif self.state == 'raising':
			if sim.getTime() < (t_lift + lift_traj_duration):
				#the controller sends a command to the base after 1 s to lift the object
				t_traj = min(1, max(0, (sim.getTime() - t_lift) / lift_traj_duration))
				desired = se3.mul((so3.identity(), [0, 0, 0.10 * t_traj]), xform)
				send_moving_base_xform_PID(controller, desired[0], desired[1])
			else:
				controller.setPIDCommand(controller.getCommandedConfig(), [0.0] * len(controller.getCommandedConfig()))
				self.state = 'raised'
				print self.state

		
def make(sim,hand,dt):
	"""The make() function returns a 1-argument function that takes a SimRobotController and performs whatever
	processing is needed when it is invoked."""
	return StateMachineController(sim,hand,dt)
