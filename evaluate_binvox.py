
import binvox_read
import sys
import os
import numpy as np
objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['princeton'] = [f for f in os.listdir('data/objects/princeton')]
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



def read_binvox(object_list):

    for objectName in object_list:
        # print objectName
        for object_set, objects_in_set in objects.items():
            # print objectName
            if objectName in objects_in_set: 
                if object_set == 'princeton':
                    objpath = 'data/objects/princeton/%s/tsdf_mesh.binvox'%objectName
                else:
                    objpath = 'data/objects/%s/%s/meshes/poisson_mesh.binvox'%(object_set,objectName)
                print objpath
                try:
                    with open(objpath, 'rb') as f:
                        dims, translate, scale = read_header(f)
                        print "nome: ", objectName
                        print "dimension:",dims
                        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
                        values, counts = raw_data[::2], raw_data[1::2]
                        # print len(values)
                        # print len(counts)
                        data = np.repeat(values, counts).astype(np.bool)
                        # print len(data)
                        # print "qui", data
                        data =  data.reshape(dims)
                        data = np.transpose(data, (0, 2, 1))
                except:
                    print "No binvox in", objectName
                        # no_binvox.append(object_name)
                                        



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