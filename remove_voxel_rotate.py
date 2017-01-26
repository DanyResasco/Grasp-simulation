import os
import sys

objects = {}
# objects['ycb'] = [f for f in os.listdir('data/objects/voxelrotate/ycb')]
# objects['apc2015'] = [f for f in os.listdir('data/objects/voxelrotate/apc2015')]
objects['princeton'] = [f for f in os.listdir('data/objects/voxelrotate/princeton')]
# objects['newObjdany'] = [f for f in os.listdir('data/objects/voxelrotate/newObjdany')]


def remove_file(object_list):
    for objectName in object_list:
        # print objectName
        for object_set, objects_in_set in objects.items():
            # print objectName
            if objectName in objects_in_set:
                objpath = 'data/objects/voxelrotate/%s/%s/'%(object_set,objectName)
                for i in range(0,len(list(os.listdir(objpath)))):
                    if object_set == 'princeton':
                        respath = 'data/objects/voxelrotate/princeton/%s/%s_rotate%s.off'%(objectName,objectName,str(i))
                    # elif object_set == 'apc2015':
                    #     respath = 'data/objects/voxelrotate/apc2015/%s/poisson_rotate_%s.ply'%(objectName,str(i))
                    # else:
                    #     respath = 'data/objects/voxelrotate/%s/%s/poisson_rotate_%s.ply'%(object_set,objectName,str(i))

                    try:
                        os.remove(respath)
                    except:
                        print "No voxel rotate ", objectName, "in",object_set


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