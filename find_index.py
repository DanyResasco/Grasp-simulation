#!/usr/bin/env python

import os
import sys

objects = {}
objects['ycb'] = [f for f in os.listdir('data/objects/ycb')]
objects['apc2015'] = [f for f in os.listdir('data/objects/apc2015')]

object_geom_file_patterns = {
    'ycb':['data/objects/ycb/%s/meshes/tsdf_mesh.stl','data/objects/ycb/%s/meshes/poisson_mesh.stl'],
    'apc2015':['data/objects/apc2015/%s/textured_meshes/optimized_tsdf_textured_mesh.ply']
}

def get_dataset_from_obj_name(obj_name):
	for dataset in objects:
		if obj_name in objects[dataset]:
			return dataset
	return None

if __name__ == '__main__':
    import random
    try:
        dataset = sys.argv[1]
    except IndexError:
        print "Usage find_index.py dataset_name object_name"

    try:
        objname = sys.argv[2]
    except IndexError:
        print "Usage find_index.py dataset_name object_name"

    try:
        index = objects[dataset].index(objname)
    except IndexError:
        print "Could not find dataset", dataset
    except ValueError:
        print "Could not find object", objname, "in dataset", dataset

    print index