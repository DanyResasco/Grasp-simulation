'''Script that call a binvox lybrary to make a voxel from mesh file.
You need binvox and viewvox execute file in the folder. Download it on http://www.patrickmin.com/binvox/ 
Supported 3d model file format: UG, OBJ, OFF, DXF, XGL, POV, BREP, PLY, JOT: only polygons supported'''

from subprocess import call
import os
from IPython import embed
import sys

objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]

def Binvox_Dany(obj_list):

    for objectName in obj_list:
        # print objectName
        for object_set, objects_in_set in objects.items():
            if objectName in objects_in_set: 
                if object_set == 'princeton':
                    model_filepath = 'data/objects/princeton/%s/tsdf_mesh.off'%objectName
                    vie_filepath = 'data/objects/princeton/%s/tsdf_mesh.binvox'%objectName
                elif object_set == 'acp':
                    model_filepath = 'data/objects/%s/%s/meshes/poisson.ply'%(object_set,objectName)
                    vie_filepath = 'data/objects/%s/%s/meshes/poisson.binvox'%(object_set,objectName)
                else:
                    model_filepath = 'data/objects/%s/%s/meshes/poisson_mesh.ply'%(object_set,objectName)
                    vie_filepath = 'data/objects/%s/%s/meshes/poisson_mesh.binvox'%(object_set,objectName)
                print "///////////////////"
                print "working on:"
                print model_filepath
                print "//////////////"

                '''Called a binvox execute file.
                -d: specify voxel grid size (default 256, max 1024). Number of voxels for the longest cube dimension. 
                    You need to tweak this number so that the final map (after octree creation) has the desired resolution.
                -down: downsample voxels by a factor of 2 in each dimension (can be used multiple times)
                -ri: remove internal voxels 
                -e: exact carving gives best results, but results e.g. in hollow walls (no room for compression in the octree)
                otherwise try a combination of -c and / or -v. This might or might not work for some meshes
                -fit: gives the smallest bounding box fit '''


                call(["./binvox","-down","-ri" ,"-d","128","-e", model_filepath])
                '''Called a viewvox execute to view a voxel'''
                # call(["./viewvox", vie_filepath]) 


if __name__ == '__main__':
    all_objects = []
    for dataset in objects.values():
            all_objects += dataset

    to_check = ['1in_metal_washer', #no poisson_mesh.ply
    ]
    done = []

    for obj in to_check + done:
        all_objects.pop(all_objects.index(obj))

    try:
        objname = sys.argv[1]
        Binvox_Dany([objname])
    except:
        Binvox_Dany(all_objects)