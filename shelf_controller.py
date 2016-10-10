from klampt import *
from klampt.math import vectorops,so3,se3
from moving_base_control import *
from reflex_control import *
from klampt.io import resource
from i16mc import shelf_dims, shelf_offset, shelf_height
from IPython import embed
import numpy as np
from copy import deepcopy
from mvbb.db import MVBBLoader
from mvbb.CollisionCheck import CollisionTestInterpolate, CollisionTestPoseAll, WillCollideDuringClosure

class ShelfStateMachineController(object):
    """A more sophisticated controller that uses a state machine."""
    def __init__(self,sim,hand,dt):
        self.sim = sim
        self.world = sim.world
        self.robot = sim.world.robot(0)
        self.hand = hand
        self.dt = dt
        self.sim.updateWorld()
        self.base_xform = get_moving_base_xform(self.sim.controller(0).model())
        self.state = 'wait'
        self.objs_to_move = []
        self.obj = None
        self.t0 = sim.getTime()
        self.start_pose = None
        self.goal_pose = None
        self.p_T_h = np.array(se3.homogeneous(resource.get("default_initial_soft_hand.xform", description="Initial hand transform",
                             default=se3.identity(), world=sim.world, doedit=False)))
        self.db = MVBBLoader()

    def __call__(self,controller):
        self.sim.updateWorld()
        sim = self.sim
        xform = self.base_xform

        #controller state machine
        t_park = 1.0
        t_approach = 0.5
        t_grasp = 1.0
        t_move = 0.5
        t_ungrasp = 0.5
        t_wait = 1.0

        if self.state == 'wait':
            if sim.getTime() - self.t0 > t_wait:
                self.state = 'idle'
                print self.state

        if self.state == 'idle':
            self.obj = None
            if len(self._get_objs_to_move()) > 0:
                print "Found", len(self.objs_to_move),"objects"
                for i, obj in enumerate(self.objs_to_move):
                    goal_pose = self._get_hand_pose_for_obj(obj)
                    if goal_pose is not None:
                        self.obj =self.objs_to_move[i]
                        print "Found", len(goal_pose), "for object", obj.getName(), "- using first"
                        self.goal_pose = se3.from_homogeneous(goal_pose[0]) # TODO make multiple attempts? Make better indicator?
                        break

                if self.obj is None:
                    self.obj = self.objs_to_move.pop(0)
                    self.goal_pose = se3.from_homogeneous(self.get_hand_pose_to_throw_obj(obj))
                    print "No Pose Found, trying to throw", self.obj.getName()
                else:
                    print "Executing with specified Pose"

                self.start_pose = get_moving_base_xform(self.sim.controller(0).model())
                self.t0 = sim.getTime()
                self.state = 'parking'
                print self.state
            else:
                print "Done moving all objects"
                self.state = 'wait'
                self.t0 = sim.getTime()
                print self.state


        elif self.state == 'parking':
            u = (sim.getTime() - self.t0) / t_park
            goal_pose = deepcopy(self.goal_pose)
            goal_pose[1][0:2] = self.start_pose[1][0:2]
            t = vectorops.interpolate(self.start_pose[1], goal_pose[1], np.min((u,1.0)))
            desired = (goal_pose[0], t)
            send_moving_base_xform_PID(controller, desired[0], desired[1])

            if sim.getTime() - self.t0 > t_park:
                self.start_pose = get_moving_base_xform(self.sim.controller(0).model())
                self.t0 = sim.getTime()
                self.state = 'approaching'
                print self.state

        elif self.state == 'approaching':
            u = (sim.getTime() - self.t0) / t_approach
            t = vectorops.interpolate(self.start_pose[1], self.goal_pose[1], np.min((u, 1.0)))
            desired = (self.goal_pose[0], t)
            send_moving_base_xform_PID(controller, desired[0], desired[1])

            if sim.getTime() - self.t0 > t_approach:
                self.start_pose = get_moving_base_xform(self.sim.controller(0).model())
                self.t0 = sim.getTime()
                #this is needed to stop at the current position in case there's some residual velocity
                controller.setPIDCommand(controller.getCommandedConfig(),[0.0]*len(controller.getCommandedConfig()))
                self.hand.setCommand([1.0])
                self.state = 'grasping'
                print self.state

        elif self.state == 'grasping':
            if sim.getTime() - self.t0 > t_grasp:
                self.t0 = sim.getTime()
                self.start_pose = get_moving_base_xform(self.sim.controller(0).model())
                goal_pose = deepcopy(self.base_xform) # back to initial pose
                self.goal_pose = goal_pose
                self.state = 'moving'
                print self.state

        elif self.state == 'moving':
            u = (sim.getTime() - self.t0) / t_move
            goal_pose = deepcopy(self.goal_pose)
            t = vectorops.interpolate(self.start_pose[1], goal_pose[1], np.min((u, 1.0)))
            desired = (goal_pose[0], t)
            send_moving_base_xform_PID(controller, desired[0], desired[1])

            if sim.getTime() - self.t0 > t_move:

                self.obj = None
                self.hand.setCommand([0.0])
                self.state = 'wait'
                self.t0 = sim.getTime()
                print self.state

        # need to manually call the hand emulator
        self.hand.process({}, self.dt)

    def _get_objs_to_move(self):
        w = self.sim.world
        self.objs_to_move = []
        for i in range(w.numRigidObjects()):
            o = w.rigidObject(i)
            R, t = o.getTransform()
            if t[0] > -shelf_dims[0]/2.0 and t[0] < shelf_dims[0]/2.0 and \
               t[1] > shelf_offset - shelf_dims[1]/2.0 and  t[1] < shelf_dims[1]/2.0 + shelf_offset and \
               t[2] > shelf_height - shelf_dims[2]/2.0 and  t[2] < shelf_height + shelf_dims[2]:
                self.objs_to_move.append(o)
            else:
                print o.getName(), "out of shelf at", t

            self.objs_to_move = sorted(self.objs_to_move, key=lambda obj: obj.getTransform()[1][1])
        return self.objs_to_move

    def _get_hand_pose_for_obj(self, obj):
        w_T_o = np.array(se3.homogeneous(obj.getTransform()))
        poses = self.db.get_scored_poses(obj.getName())
        filtered_poses = []
        alternate_strategy = False

        if len(poses) == 0:
            alternate_strategy = True
        else:
            for pose in poses:
                p = w_T_o.dot(pose).dot(self.p_T_h)
                #if WillCollideDuringClosure(self.hand,obj):
                if not CollisionTestPoseAll(self.world, self.robot, p):
                    filtered_poses.append(p)
            if len(filtered_poses) == 0:
                alternate_strategy = True

        if alternate_strategy:
            return None
        else:
            return sorted(filtered_poses, key= lambda pose: pose[2,3], reverse=True)

    def get_hand_pose_to_throw_obj(self, obj):
        return np.array(se3.from_homogeneous(obj.getTransform()))

def make(sim,hand,dt):
    """The make() function returns a 1-argument function that takes a SimRobotController and performs whatever
    processing is needed when it is invoked."""
    return ShelfStateMachineController(sim,hand,dt)
