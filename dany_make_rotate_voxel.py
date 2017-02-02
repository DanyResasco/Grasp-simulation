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
from klampt.math import so3,se3
import importlib
import os
from IPython import embed
import sys
import csv
import numpy as np

box_dims = (0.45,0.45,0.25)
box_dims_shelf = (0.5,0.5,0.3)
shelf_dims = (0.3,0.5,0.3)
shelf_offset = 0.8
shelf_height = 0.5
moving_base_template_fn = 'data/robots/moving_base_template.rob'
object_template_fn = 'data/objects/object_template.obj'
objects = {}
# objects['ycb'] = [f for f in sorted(os.listdir('data/objects/voxelrotate/ycb'))]
# objects['apc2015'] = [f for f in sorted(os.listdir('data/objects/voxelrotate/apc2015'))]
objects['princeton'] = [f for f in sorted(os.listdir('data/objects/voxelrotate/princeton'))]
# objects['newObjdany'] = [f for f in sorted(os.listdir('data/objects/voxelrotate/newObjdany'))]



robots = ['reflex_col', 'soft_hand', 'reflex']

object_geom_file_patterns = {
    # 'ycb':['data/objects/voxelrotate/ycb/%s/%s.stl'],
    # 'apc2015':['data/objects/voxelrotate/apc2015/%s/%s.stl'],
    'princeton':['data/objects/voxelrotate/princeton/%s/%s.off'],
    # 'newObjdany':['data/objects/voxelrotate/newObjdany/%s/%s.stl']
}
#default mass for objects whose masses are not specified, in kg
default_object_mass = 0.5
object_masses = {
    'ycb':dict(),
    'apc2015':dict(),
    'princeton':dict(),
    # 'newObjdany':dict(),
}


def Write_Poses(dataset,poses):
    '''Write the dataset'''
    f = open(dataset, 'w')
    temp = se3.from_homogeneous(poses)
    for i in range(0,len(temp)):
        f.write(','.join([str(v) for v in temp[i]]))
        f.write(',')
    f.close()




def getTransform(nome):
    obj_dataset = '3DCNN/NNSet/Pose/ObjectsVariation/%s.csv'%nome
    print obj_dataset
    with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
        file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
        for row in file_reader:
            angle = row[0]
            axis = [0,0,1]
            R = so3.from_axis_angle((axis,angle))
            T = row[4:]
            # embed()
    return np.array((R,T))




def make_objectRotate(object_set,objectname,world,num):
    """Adds an object to the world using its geometry / mass properties
    and places it in a default location (x,y)=(0,0) and resting on plane."""
    # print "dentro make_object"
    # embed()
    for pattern in object_geom_file_patterns[object_set]:
        nome = objectname + '_rotate_' + str(num)

        objfile = pattern%(objectname,nome)
        # print "****",objfile
        # embed()
        # objfile = pattern%(nome)
        objmass = object_masses[object_set].get('mass',default_object_mass)
        if object_set == 'princeton':
            f = open( 'data/objects/template_obj_scale_princeton.obj','r')
        else:
            f = open(object_template_fn,'r')
        # embed()
        pattern = ''.join(f.readlines())
        # print pattern
        f.close()
        f2 = open("temp.obj",'w')
        f2.write(pattern % (objfile,objmass))
        f2.close()
        nobjs = world.numRigidObjects()
        if world.loadElement('temp.obj') < 0 :
            continue
        assert nobjs < world.numRigidObjects(),"Hmm... the object didn't load, but loadElement didn't return -1?"
        obj = world.rigidObject(world.numRigidObjects()-1)
        
        objT  = getTransform(nome)

        # embed()
        obj.setTransform(objT[0],objT[1])
        # embed()
        bmin,bmax = obj.geometry().getBB()
        T = obj.getTransform()
        spacing = 0.006
        T = (T[0],vectorops.add(T[1],(-(bmin[0]+bmax[0])*0.5,-(bmin[1]+bmax[1])*0.5,-bmin[2]+spacing)))
        obj.setTransform(*T)
        obj.appearance().setColor(0.2,0.5,0.7,1.0)
        obj.setName(nome)
        return obj