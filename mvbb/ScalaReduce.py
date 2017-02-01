#!/usr/bin/env python
from klampt import *
from klampt.vis.glrobotprogram import * #Per il simulatore
from klampt.model import collide
from moving_base_control import set_moving_base_xform, get_moving_base_xform
import numpy as np
from IPython import embed 
from klampt.math import se3,so3

object_geom_file_patterns = {
    'ycb':['data/objects/ycb/%s/meshes/tsdf_mesh.stl','data/objects/ycb/%s/meshes/poisson_mesh.stl'],
    'apc2015':['data/objects/apc2015/%s/textured_meshes/optimized_tsdf_textured_mesh.ply'],
    'newObjdany':['data/objects/newObjdany/%s/tsdf_mesh.stl', 'data/objects/newObjdany/%s/tsdf_mesh.ply' ],
    'princeton':['data/objects/princeton/%s/tsdf_mesh.off']
}

default_object_mass = 0.5
object_masses = {
    'ycb':dict(),
    'apc2015':dict(),
    'princeton':dict(),
    'newObjdany':dict(),
}

def DanyReduceScale(objectname, world,objfilename,object_set):
    """Simple function to scale the object in princeton databaset, 
    and add a obj in the word"""

    for pattern in object_geom_file_patterns[object_set]:
        objfile = pattern%(objectname,)
        f = open(objfilename,'r')
        pattern = ''.join(f.readlines())
        f.close()
        f2 = open("temp.obj",'w')
        objmass = object_masses[object_set].get('mass',default_object_mass)
        f2.write(pattern % (objfile,objmass))
        f2.close()
        nobjs = world.numRigidObjects()
        if world.loadElement('temp.obj') < 0 :
            continue
        assert nobjs < world.numRigidObjects(),"Hmm... the object didn't load, but loadElement didn't return -1?"
        # print "obj numRigidObjects"
        obj = world.rigidObject(world.numRigidObjects()-1)
        # print "obj numRigidObjects"
        obj.setTransform(*se3.identity())
        bmin,bmax = obj.geometry().getBB()
        T = obj.getTransform()
        if bmin[0] <0 and bmin[1] <0 and bmin[2] <0:
            T[1][0] = -bmin[0]
            T[1][1] = -bmin[1]
            T[1][2] = -bmin[2]
            obj.setTransform(T)
        spacing = 0.005
        T = (T[0],vectorops.add(T[1],(-(bmin[0]+bmax[0])*0.5,-(bmin[1]+bmax[1])*0.5,-bmin[2]+spacing)))
        obj.setTransform(*T)
        obj.appearance().setColor(0.2,0.5,0.7,1.0)
        obj.setName(objectname)
        return obj
    raise RuntimeError("Unable to load object name %s from set %s"%(objectname,object_set))