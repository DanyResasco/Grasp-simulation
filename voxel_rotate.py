#!/usr/bin/env python


import importlib
import sys
import os
import random
import string
import pydany_bb
import numpy as np
import math
from IPython import embed
from klampt.math import so3,se3
import time
# from klampt import *
# import klampt.robotsim
# from klampt import vis
# from klampt.vis.glprogram import *
# from klampt.vis.glprogram import GLNavigationProgram    #Per il
# from klampt.sim import *
import importlib
import sys
import os
import random
from klampt.math import so3, se3
import string
import pydany_bb
import numpy as np
import math
from IPython import embed
# from mvbb.graspvariation import PoseVariation
# from mvbb.TakePoses import SimulationPoses
# from mvbb.draw_bbox import draw_GL_frame, draw_bbox
# from i16mc import make_object
from dany_make_rotate_voxel import make_objectRotate
import trimesh
import csv
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import pymesh
import random



objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
# objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
# objects['newObjdany'] = [f for f in os.listdir('data/objects/newObjdany')]
# objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]

Pose = {}
Pose['Pose'] = [f for f in os.listdir('3DCNN/NNSet/Pose')]


def Open_pose_file(object_list, vector_set):
    '''Read the poses and store it as rpy into a vector'''
    for object_name in object_list:
        # for object_set, objects_in_set in Pose.items():
            obj_dataset = '3DCNN/NNSet/Pose/pose/%s.csv'%(object_name)
            try:
                with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
                    file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
                    for row in file_reader:
                        T = row[9:]
                        Matrix_ = so3.matrix(row)
                        pp = row[:9]
                        P = pp,T
                        vector_set.append(P)
            except:
                print "problem with csvfile for ", object_name


def Write_Poses(dataset,poses):
    '''Write the dataset'''
    # embed()
    # pose_temp = [float(v) for v in poses]
    # posa.append(pose_temp)
    f = open(dataset, 'w')
    # embed()
    for i in range(0,len(poses)):
        f.write(','.join([str(v) for v in poses[i]]))
        f.write(',')
        # f.write(','.join([str(v) for v in poses[i][1]]))
        # f.write('\n')
    f.close()




def main(object_list):

    for object_name in object_list:
        for object_set, objects_in_set in objects.items():
            if object_name in objects_in_set:
                pose_original = []
                Open_pose_file([object_name],pose_original)
                # num = 0
                # print len(pose_original)
                pose_new = []
                for i in range(0,len(pose_original)):
                    new_mesh = pymesh.Mesh
                    if object_set == 'princeton':
                        objpath = 'data/objects/princeton/%s/tsdf_mesh.off'%object_name
                        respath = 'data/objects/voxelrotate/%s/%s/%s_rotate_%s.off'%(object_set,object_name,object_name,i)
                    elif object_set == 'apc2015':
                        objpath = 'data/objects/apc2015/%s/meshes/poisson.ply'%object_name
                        respath = 'data/objects/voxelrotate/%s/%s/%s_rotate_%s.stl'%(object_set,object_name,object_name,i)
                    # elif object_set == 'newObjdany':
                    #     objpath = 'data/objects/newObjdany/%s/poisson_mesh.ply'%object_name
                    #     respath = 'data/objects/newObjdany/%s/meshes/poisson_rotate_%s.ply'%(object_name,i)
                    else:
                        objpath = 'data/objects/%s/%s/meshes/poisson_mesh.stl'%(object_set,object_name)
                        respath = 'data/objects/voxelrotate/%s/%s/%s_rotate_%s.stl'%(object_set,object_name,object_name,i)
                    # directory = 'data/objects/voxelrotate/%s/%s'%(object_set,object_name)
                    # if not os.path.exists(directory):
                    #     os.makedirs(directory)
                    mesh = pymesh.load_mesh(objpath)
                    if i is not 0:
                        temp_vertex = []
                        n_axis = random.randrange(0,3)
                        if n_axis is 0:
                            axis = [1,0,0]
                            # print "x"
                        elif n_axis is 1:
                            axis = [0,1,0]
                            # print "y"
                        else:
                            axis = [0,0,1]

                            # print "z"
                        theta_deg = random.randrange(-5,5)
                        if theta_deg is 0:
                            theta_deg = random.randrange(-5,5)
                        theta = math.radians(theta_deg)
                        ROtation_matrix = so3.matrix(so3.from_axis_angle((axis,theta)))
                        temp_vertex = mesh.vertices.dot(ROtation_matrix)
                        new_mesh.vertices = temp_vertex
                        # embed()
                        mesh_new = pymesh.form_mesh(new_mesh.vertices, mesh.faces, mesh.voxels)
                        try:
                            pymesh.save_mesh(respath, mesh_new)
                            # embed()
                            pose_new = so3.mul(so3.from_axis_angle((axis,theta)),pose_original[i][0]),pose_original[i][1]
                            # pose_new.append(pose_mod)
                        except:
                            print "Problem with", object_name, "In", object_set

                    else:
                        print object_name
                        pymesh.save_mesh(respath, mesh)
                        # pose_new.append(pose_original[i])
                        pose_new = pose_original[i]
                    respose = '3DCNN/NNSet/Pose/PoseVariation/%s_rotate_%s.csv'%(object_name,str(i))
                    Write_Poses(respose,pose_new)







if __name__ == '__main__':
    all_objects = []
    for dataset in objects.values():
        all_objects += dataset

    try:
        nome = sys.argv[1]
        main([nome])
    except:
        main(all_objects)
