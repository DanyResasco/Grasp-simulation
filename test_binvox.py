'''Script that call a binvox lybrary to make a voxel from mesh file.
You need binvox and viewvox execute file in the folder. Download it on http://www.patrickmin.com/binvox/ 
Supported 3d model file format: UG, OBJ, OFF, DXF, XGL, POV, BREP, PLY, JOT: only polygons supported'''

from subprocess import call
import os

objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]


def Binvox_Dany(obj_list):

    for objectName in obj_list:
        for object_set, objects_in_set in objects.items():
            if objectName in objects['princeton']:
                model_filepath = 'data/objects/princeton/%s/tsdf_mesh.off'%objectName
                vie_filepath = 'data/objects/princeton/%s/tsdf_mesh.binvox'%objectName
            else:
                model_filepath = 'data/objects/%s/%s/meshes/poisson_mesh.ply'%(object_set,objectName)
                vie_filepath = 'data/objects/%s/%s/meshes/poisson_mesh.binvox'%(object_set,objectName)
            print "///////////////////"
            print "working on:"
            print model_filepath
            print "//////////////"

            '''Called a binvox execute file.
            -d: specify voxel grid size (default 256, max 1024)
            -down: downsample voxels by a factor of 2 in each dimension (can be used multiple times)
            -ri: remove internal voxels '''

            call(["./binvox","-down","-d"," 250", "-ri",model_filepath])
            '''Called a viewvox execute to view a voxel'''
            call(["./viewvox", vie_filepath]) 


if __name__ == '__main__':
    all_objects = []
    for dataset in objects.values():
            all_objects += dataset

    to_check = []
    done = []

    for obj in to_check + done:
        all_objects.pop(all_objects.index(obj))

    try:
        objname = sys.argv[1]
        Binvox_Dany([objname])
    except:
        Binvox_Dany(all_objects)