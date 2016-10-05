import pkg_resources
pkg_resources.require("klampt>=0.7.0")
from klampt import *
from klampt import vis 
from klampt.vis.glrobotprogram import *
from klampt.math import *
from klampt.model import collide
from klampt.io import resource
from klampt.sim import *
from moving_base_control import *
import importlib
import math
import os
import string
import sys
import time

from create_mvbb import MVBBVisualizer, compute_poses, skip_decimate_or_return
from klampt.math import so3, se3
import pydany_bb
import numpy as np
from IPython import embed
from mvbb.graspvariation import PoseVariation
from mvbb.TakePoses import SimulationPoses
from mvbb.draw_bbox import draw_GL_frame, draw_bbox
from i16mc import make_object, make_moving_base_robot
from mvbb.CollisionCheck import CheckCollision, CollisionTestInterpolate, CollisionTestPose


objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
robots = ['reflex_col', 'soft_hand', 'reflex']

object_geom_file_patterns = {
    'ycb':['data/objects/ycb/%s/meshes/tsdf_mesh.stl','data/objects/ycb/%s/meshes/poisson_mesh.stl'],
    'apc2015':['data/objects/apc2015/%s/textured_meshes/optimized_tsdf_textured_mesh.ply']
}

robot_files = {
    'reflex_col':'data/robots/reflex_col.rob',
    'soft_hand':'data/robots/soft_hand.urdf',
    'reflex':'data/robots/reflex.rob'
}

class FilteredMVBBTesterVisualizer(MVBBVisualizer):
    def __init__(self, poses, poses_variations, boxes, world, h_T_h2):
        MVBBVisualizer.__init__(self, poses, poses_variations, boxes, world, None)
        self.boxes = boxes
        self.world = world

    def display(self):
        if self.world.numRigidObjects() > 0:
            self.obj = self.world.rigidObject(0)

        self.world.drawGL()
        self.obj.drawGL()

        for pose in self.poses:
            draw_GL_frame(T)

        hand_xform = get_moving_base_xform(self.robot)
        h_T_g_np = np.array(se3.homogeneous(hand_xform)).dot(h_T_h2)
        T_h = se3.from_homogeneous(h_T_g_np)
        draw_GL_frame(T_h)

        for box in self.boxes:
            draw_bbox(box.Isobox, box.T)

def launch_test_mvbb_filtered(robotname, object_list, min_vertices = 0):
    """Launches a very simple program that spawns an object from one of the
    databases.
    It launches a visualization of the mvbb decomposition of the object, and corresponding generated poses.
    It then spawns a hand and tries all different poses to check for collision
    """

    world = WorldModel()
    robot = make_moving_base_robot(robotname, world)
    xform = resource.get("default_initial_%s.xform" % robotname, description="Initial hand transform",
                         default=se3.identity(), world=world, doedit=False)

    for object_name in object_list:
        for object_set in objects:
            if object_name in object_set:
                object = make_object(object_set, object_name, world)


        R,t = object.getTransform()
        object.setTransform(R, [0, 0, 0])
        object_vertices_or_none, tm_decimated = skip_decimate_or_return(object, min_vertices, 2000)
        if object_vertices_or_none is None:
            pass
        object_or_vertices = object_vertices_or_none

        print "------Computing poses:"
        poses, poses_variations, boxes = compute_poses(object_or_vertices)


        w_T_o = np.eye(4) # object is at origin
        h_T_h2 = np.array(se3.homogeneous(xform))

        poses_h = []
        poses_variations_h = []

        for i in range(len(poses)):
            poses_h.append(w_T_o.dot(poses[i]).dot(h_T_h2))
        for i in range(len(poses_variations)):
            poses_variations_h.append(w_T_o.dot(poses_variations[i]).dot(h_T_h2))

        print "-------Filtering poses:"
        filtered_poses = []
        for i in range(len(poses)):
            if not CollisionTestPose(world, robot, object, poses_h[i]):
                filtered_poses.append(poses[i])
        filtered_poses_variations = []
        for i in range(len(poses_variations)):
            if not CollisionTestPose(world, robot, object, poses_variations_h[i]):
                filtered_poses_variations.append(poses_variations[i])

        embed()

        hand = None
        program = FilteredMVBBTesterVisualizer(filtered_poses, filtered_poses_variations, world, h_T_h2)
        vis.setPlugin(program)
        program.reshape(800, 600)

        vis.show()
        # this code manually updates the visualization
        for pose in filtered_poses + filtered_poses_variations:
            object_com_z_0 = None # TODO
            world.loadElement("data/terrains/plane.env")
            object_fell = False
            time = 0.0

            while vis.shown() and time <= 4.0 or object_fell:
                object_com_z = None # TODO
                if time == 0:
                    #hand.send_command(0.7) # TODO
                elif time == 1.0:
                    world.remove(world.terrain(0))
                time += 0.01

                if object_com_z < object_com_z_0 - 1.0:
                    object_fell = True

        vis.kill()
    return

if __name__ == '__main__':
    all_objects = []
    for dataset in objects:
        all_objects += dataset
    print all_objects

    embed()

    launch_test_mvbb_filtered("soft_hand", all_objects, 100)
