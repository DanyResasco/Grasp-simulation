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
from mvbb.CollisionCheck import CollisionTestInterpolate, CollisionTestPose, WillCollideDuringClosure
from create_mvbb_filtered import numpytokdl4
import PyKDL

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
        self.o_T_p = None
        self.p_T_h = np.array(se3.homogeneous(resource.get("default_initial_soft_hand.xform", description="Initial hand transform",
                             default=se3.identity(), world=sim.world, doedit=False)))
        self.db = MVBBLoader()
        self.t_wait = 3.5

    def __call__(self,controller):
        self.sim.updateWorld()
        sim = self.sim
        xform = self.base_xform
        #embed()

        #controller state machine
        t_park = 1.0
        t_park_settle = 0.5
        t_approach = 1.5
        t_approach_settle = 0.5
        t_grasp = 1.0
        t_move = 1.0
        t_raise = 0.5
        t_wait = self.t_wait

        if self.state == 'wait':
            if sim.getTime() - self.t0 > t_wait:
                self.state = 'idle'
                self.t_wait = 1.0
                print self.state

        if self.state == 'idle':
            self.obj = None
            if len(self._get_objs_to_move()) > 0:
                print "Found", len(self.objs_to_move),"objects"
                for i, obj in enumerate(self.objs_to_move):
                    goal_pose = self._get_hand_pose_for_obj(obj)
                    if goal_pose is not None:
                        self.obj =self.objs_to_move[i]
                        print "Found", len(goal_pose), "poses for object", obj.getName(), "- using first", goal_pose[0]
                        self.goal_pose = se3.from_homogeneous(goal_pose[0])

                        w_T_o = np.array(se3.homogeneous(obj.getTransform()))
                        self.o_T_p = np.linalg.inv(w_T_o).dot(goal_pose[0])
                        break # anyway
                    else:
                        print "found no poses for object", obj.getName()

                self.start_pose = get_moving_base_xform(self.sim.controller(0).model())
                self.t0 = sim.getTime()

                if self.obj is None:
                    print "No Pose Found"
                else:
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
            T = se3.interpolate(self.start_pose, goal_pose, np.min((u,1.0)))
            send_moving_base_xform_PID(controller, T[0], T[1])

            if sim.getTime() - self.t0 > t_park + t_park_settle:
                self.start_pose = get_moving_base_xform(self.sim.controller(0).model())
                self.t0 = sim.getTime()
                self.state = 'approaching'
                print self.state

        elif self.state == 'approaching':
            u = (sim.getTime() - self.t0) / t_approach
            t = vectorops.interpolate(self.start_pose[1], self.goal_pose[1], np.min((u, 1.0)))
            desired = (self.goal_pose[0], t)
            send_moving_base_xform_PID(controller, desired[0], desired[1])

            if sim.getTime() - self.t0 > t_approach + t_approach_settle:
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
                if self.obj.getName() == 'dove_beauty_bar':
                    self.goal_pose[1][2] += 0.13  # TODO if the grasp is not too high we can go higher...
                else:
                    self.goal_pose[1][2] += 0.07  # TODO if the grasp is not too high we can go higher...
                self.state = 'raising'
                print self.state

        elif self.state == 'raising':
            u = (sim.getTime() - self.t0) / t_raise
            t = vectorops.interpolate(self.start_pose[1], self.goal_pose[1], np.min((u, 1.0)))
            desired = (self.goal_pose[0], t)
            send_moving_base_xform_PID(controller, desired[0], desired[1])

            if sim.getTime() - self.t0 > t_raise:
                self.start_pose = get_moving_base_xform(self.sim.controller(0).model())
                #goal_pose = deepcopy(self.base_xform) # back to initial pose
                goal_pose = deepcopy(self.goal_pose)  # back to initial pose
                goal_pose[1][0] = self.base_xform[1][0]
                goal_pose[1][1] = self.base_xform[1][1]
                #set goal pose to have same palm pose wrt original pose
                w_T_p_initial = se3.mul(self.base_xform, se3.inv(se3.from_homogeneous(self.p_T_h)))
                w_T_p_current = se3.mul(self.start_pose, se3.inv(se3.from_homogeneous(self.p_T_h)))
                w_T_p_final = (w_T_p_current[0], w_T_p_initial[1])
                w_T_p_final[1][2] = w_T_p_current[1][2]
                self.goal_pose = se3.mul(w_T_p_final, se3.from_homogeneous(self.p_T_h))
                self.state = 'moving'
                self.t0 = sim.getTime()
                print self.state

        elif self.state == 'moving':
            curr_pose_p = se3.mul(get_moving_base_xform(self.sim.controller(0).model()), se3.inv(se3.from_homogeneous(self.p_T_h)))
            curr_g = se3.mul(self.obj.getTransform(), se3.from_homogeneous(self.o_T_p))

            if vectorops.distance(curr_pose_p[1], curr_g[1]) > 0.2:
                print "Object dropped"
                print curr_pose_p
                print curr_g
                u = 1.0
            else:
                u = (sim.getTime() - self.t0) / t_move
            t = vectorops.interpolate(self.start_pose[1], self.goal_pose[1], np.min((u, 1.0)))
            desired = (self.goal_pose[0], t)
            send_moving_base_xform_PID(controller, desired[0], desired[1])

            if sim.getTime() - self.t0 > t_move:
                self.hand.setCommand([0.0])

            if sim.getTime() - self.t0 > t_move + 0.5:
                send_moving_base_xform_PID(controller, self.goal_pose[0],
                                           vectorops.add(self.goal_pose[1], [0, 0, -0.15]))
            if sim.getTime() - self.t0 > t_move + 0.6:
                send_moving_base_xform_PID(controller, self.goal_pose[0], self.goal_pose[1])
            if sim.getTime() - self.t0 > t_move + 0.7:
                send_moving_base_xform_PID(controller, self.goal_pose[0],
                                           vectorops.add(self.goal_pose[1], [0, 0, -0.15]))
            if sim.getTime() - self.t0 > t_move + 0.8:
                send_moving_base_xform_PID(controller, self.goal_pose[0], self.goal_pose[1])
            if sim.getTime() - self.t0 > t_move + 0.9:
                self.state = 'wait'
                self.t0 = sim.getTime()
                self.obj = None
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

        self.objs_to_move = sorted(self.objs_to_move, key=lambda obj: obj.getTransform()[1][1])
        return self.objs_to_move

    def _has_pose_for_obj(self):
        return len(self._get_hand_pose_for_obj(self.obj)) > 0

    def _get_hand_pose_for_obj(self, obj):
        w_T_o = np.array(se3.homogeneous(obj.getTransform()))
        poses = self.db.get_scored_poses(obj.getName())
        filtered_poses = []
        alternate_strategy = False

        if len(poses) == 0:
            return None
        else:
            for pose in poses:
                p = w_T_o.dot(pose).dot(self.p_T_h)
                s = np.array(se3.homogeneous(self.base_xform)) # linear distance from s TODO use planning instead

                s[0:3,0:3] = p[0:3,0:3] # copying rotation from final pose
                s[0,3] = p[0,3]         # copying x from final pose
                s[2,3] = p[2,3]         # copying height from final pose

                #if WillCollideDuringClosure(self.hand,obj):
                if not CollisionTestPose(self.world, self.robot, obj, p) and \
                   not CollisionTestInterpolate(self.world, self.robot, obj, p, s):
                    filtered_poses.append(p)
            if len(filtered_poses) == 0:
                return None

            curr_pose_kdl = numpytokdl4(np.array(se3.homogeneous(get_moving_base_xform(self.sim.controller(0).model()))))

            filtered_poses = sorted(filtered_poses, key=lambda pose: PyKDL.diff(curr_pose_kdl, numpytokdl4(pose)).rot.Norm())
            return sorted(filtered_poses, key= lambda pose: pose[2,3], reverse=True) # TODO better sorting

    def _getObjectGlobalCom(self, obj):
        return se3.apply(obj.getTransform(), obj.getMass().getCom())

def make(sim,hand,dt):
    """The make() function returns a 1-argument function that takes a SimRobotController and performs whatever
    processing is needed when it is invoked."""
    return ShelfStateMachineController(sim,hand,dt)
