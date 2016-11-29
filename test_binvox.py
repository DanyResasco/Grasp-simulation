'''Script that call a binvox lybrary to make a voxel from mesh file.
You need binvox and viewvox execute file in the folder'''

from subprocess import call
import os

objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
all_objects = []
for dataset in objects.values():
        all_objects += dataset


for obj in all_objects:
    all_objects.pop(all_objects.index(obj))
    model_filepath = 'data/objects/ycb/%s/meshes/poisson_mesh.ply'%obj
    vie_filepath = 'data/objects/ycb/%s/meshes/poisson_mesh.binvox'%obj
    print
    print "working on:"
    print model_filepath

    call(["./binvox","-down", model_filepath]) #called a binvox
    call(["./viewvox", vie_filepath]) # called a function to view a voxel