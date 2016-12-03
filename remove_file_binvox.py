import os
import sys

objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]



def remove_file(object_list):
    for objectName in object_list:
        # print objectName
        for object_set, objects_in_set in objects.items():
            # print objectName
            if objectName in objects_in_set: 
                if object_set == 'princeton':
                    objpath = 'data/objects/princeton/%s/tsdf_mesh.binvox'%objectName
                elif object_set == 'acp':
                    objpath = 'data/objects/%s/%s/meshes/poisson.binvox'%(object_set,objectName)
                else:
                    objpath = 'data/objects/%s/%s/meshes/poisson_mesh.binvox'%(object_set,objectName)
                try:
                    os.remove(objpath)
                except:
                    print "No voxel is", objectName


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
        remove_file([objname])
    except:
        remove_file(all_objects)