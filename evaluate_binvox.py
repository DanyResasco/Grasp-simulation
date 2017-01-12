
import binvox_read
import sys
import os
import numpy as np
objects = {}
# objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
# objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]
# objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]
objects['newObjDany'] = [f for f in os.listdir('data/objects/newObjDany')]
# no_binvox = []

def read_header(fp):
    """ Read binvox header. Mostly meant for internal use.
    """
    line = fp.readline().strip() #.strip() which removes whitespace from the beginning or end of the string
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale

def Write_Results(parameter,res_dataset):
    import csv
    with open(res_dataset, 'wb') as csvfilereader:
        writer = csv.writer(csvfilereader, delimiter=',')
        for i in parameter:
            writer.writerow([i])
        csvfilereader.close()


def read_binvox(object_list):

    for objectName in object_list:
        for object_set, objects_in_set in objects.items():
            if objectName in objects_in_set: 
                objpath = 'data/objects/newObjDany/%s/poisson_mesh.binvox'%objectName
                # if object_set == 'princeton':
                #     objpath = 'data/objects/princeton/%s/tsdf_mesh.binvox'%objectName
                #     pathbinvox2bt = 'data/objects/princeton/%s/tsdf_mesh.binvox.bt'%objectName
                #     pathoctree = 'data/objects/princeton/%s/tsdf_mesh.binvox.bt.ot'%objectName
                # elif object_set == 'apc2015':
                #     objpath = 'data/objects/%s/%s/meshes/poisson.binvox'%(object_set,objectName)
                #     pathbinvox2bt = 'data/objects/%s/%s/meshes/poisson.binvox.bt'%(object_set,objectName)
                #     pathoctree = 'data/objects/%s/%s/meshes/poisson.binvox.bt.ot'%(object_set,objectName)
                # else:
                #     objpath = 'data/objects/%s/%s/meshes/poisson_mesh.binvox'%(object_set,objectName)
                #     pathbinvox2bt = 'data/objects/%s/%s/meshes/poisson_mesh.binvox.bt'%(object_set,objectName)
                #     pathoctree = 'data/objects/%s/%s/meshes/poisson_mesh.binvox.bt.ot'%(object_set,objectName)
                try:
                    import shutil
                    qui = '3DCNN/NNSet/binvox/Binvox/%s.csv'%objectName
                    shutil.copyfile(objpath,qui)

                    # qui = '3DCNN/NNSet/binvox/binvox2bt/%s.csv'%objectName
                    # shutil.copyfile(pathbinvox2bt,qui)

                    # qui = '3DCNN/NNSet/binvox/octree/%s.csv'%objectName
                    # shutil.copyfile(pathoctree,qui)


                    with open(objpath, 'rb') as f:
                        dims, translate, scale = read_header(f)
                        # print "nome: ", objectName
                        # print "dimension:",dims
                        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
                        values, counts = raw_data[::2], raw_data[1::2]
                        data = np.repeat(values, counts).astype(np.int64)
                        data =  data.reshape(dims)
                        data = np.transpose(data, (0, 2, 1))
                        res_path = '3DCNN/NNSet/binvox/%s.csv'%objectName
                        Write_Results(data, res_path)
                except:
                    print "No binvox in", objectName

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
        read_binvox([objname])
    except:
        read_binvox(all_objects)


#No binvox in plastic_bolt_grey
# No binvox in plastic_nut_grey
# No binvox in extra_small_black_spring_clamp
# No binvox in plastic_wine_cup
# No binvox in 1in_metal_washer
