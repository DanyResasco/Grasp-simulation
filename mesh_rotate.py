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
from klampt import *
import klampt.robotsim
from klampt import vis
from klampt.vis.glprogram import *
from klampt.vis.glprogram import GLNavigationProgram    #Per il
from klampt.sim import *
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
from dany_make_rotate_voxel import make_objectRotate


'''Code to create a rotate mesh. The rotation is always on Z axis'''


objects = {}
# objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
# objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
# objects['newObjdany'] = [f for f in os.listdir('data/objects/newObjdany')]
objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]

Pose = {}
Pose['Pose'] = [f for f in os.listdir('3DCNN/NNSet/Pose')]


def Open_pose_file(object_list, vector_set):
    '''Read the poses and store it as rpy into a vector'''
    for object_name in object_list:
        # for object_set, objects_in_set in Pose.items():
            obj_dataset = '3DCNN/NNSet/Pose/pose/%s.csv'%(object_name)
            # embed()
            try:
                with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
                    file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
                    for row in file_reader:
                        T = row[9:]
                        # Matrix_ = so3.matrix(row)
                        pp = row[:9]
                        # embed()
                        P = np.array(se3.homogeneous((pp,T)))
                        vector_set.append(P)
            except:
                print "No pose in ", object_name


def Write_Poses(dataset,poses):
    '''Write the dataset'''
    f = open(dataset, 'w')
    # embed()
    temp = se3.from_homogeneous(poses)
    for i in range(0,len(temp)):
        f.write(','.join([str(v) for v in temp[i]]))
        f.write(',')
    f.close()

def WriteRotationObj(dataset,angle,Axis,T):
    f = open(dataset, 'w')
    f.write(str(angle))
    f.write(',')
    f.write(','.join([str(v) for v in Axis]))
    f.write(',')
    f.write(','.join([str(v) for v in T]))
    f.close()





def main(object_list):

    world = WorldModel()
    # world.loadElement("data/terrains/plane.env")
    for object_name in object_list:
        for object_set, objects_in_set in objects.items():
            if object_name in objects_in_set:
                o_T_p = []
                Open_pose_file([object_name],o_T_p)
                # embed()
                for i in range(1,len(o_T_p)):

                    if world.numRigidObjects() > 0:
                        world.remove(world.rigidObject(0))
                    pose_new = []
                    if object_set == 'princeton':
                        objpath = 'data/objects/princeton/%s/tsdf_mesh.off'%object_name
                        respath = 'data/objects/voxelrotate/princeton/%s/%s_rotate_%s.off'%(object_name,object_name,i)
                    # elif object_set == 'apc2015':
                    #     objpath = 'data/objects/apc2015/%s/meshes/poisson.ply'%object_name
                    #     respath = 'data/objects/voxelrotate/%s/%s/%s_rotate_%s.stl'%(object_set,object_name,object_name,i)
                    # elif object_set == 'newObjdany':
                    #     objpath = 'data/objects/newObjdany/%s/tsdf_mesh.stl'%object_name
                    #     respath = 'data/objects/newObjdany/%s/%s_rotate_%s.stl'%(object_name,object_name,i)
                    # else:
                    #     objpath = 'data/objects/%s/%s/meshes/poisson_mesh.stl'%(object_set,object_name)
                    #     respath = 'data/objects/voxelrotate/%s/%s/%s_rotate_%s.stl'%(object_set,object_name,object_name,i)
                    
                    mesh = pymesh.load_mesh(objpath)
                    # embed()
                    # if i is not 0:
                    axis = [0,0,1] #only on z
                    theta_deg = random.randrange(-90,90)
                    if theta_deg is 0:
                        theta_deg = random.randrange(-90,90)
                    theta = math.radians(theta_deg)
                    ROtation_matrix = so3.matrix(so3.from_axis_angle((axis,theta)))

                    temp_vertex = mesh.vertices.dot(np.array(ROtation_matrix).transpose())

                    mesh_new = pymesh.form_mesh(temp_vertex, mesh.faces,mesh.voxels)
                    obj = None
                    obj = make_objectRotate(object_set,object_name, world,i)
                    if obj is None:
                        continue
                    R = np.array((se3.homogeneous((so3.from_axis_angle((axis,theta)),[0,0,0]))))
                    w_T_o = np.array(se3.homogeneous((obj.getTransform())))
                        # embed()
                    pose_new = np.dot(R, np.dot(w_T_o, o_T_p[i])) #w_T_p_rotate
                    # embed()
                    try:
                        pymesh.save_mesh(respath, mesh_new)

                    except:
                        print "Problem with", object_name, "In", object_set

                    respose = '3DCNN/NNSet/Pose/ObjectsVariation/%s_rotate_%s.csv'%(object_name,str(i))
                    WriteRotationObj(respose,theta,axis,obj.getTransform()[1])

                    respose = '3DCNN/NNSet/Pose/PoseVariation/%s_rotate_%s.csv'%(object_name,str(i))
                    Write_Poses(respose,pose_new)
                    







if __name__ == '__main__':
    all_objects = []
    for dataset in objects.values():
        all_objects += dataset

    # try:
    nome = sys.argv[1]
    main([nome])
    # except:
    #     main(all_objects)
