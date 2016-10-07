#!/usr/bin/env python

import csv
from klampt.math import se3
import numpy as np
import pydany_bb

object_geom_file_patterns = {
	'ycb':['data/objects/ycb/%s/meshes/tsdf_mesh.stl','data/objects/ycb/%s/meshes/poisson_mesh.stl'],
	'apc2015':['data/objects/apc2015/%s/textured_meshes/optimized_tsdf_textured_mesh.ply']
}

class MVBBGenerator(object):
    def __init__(self, new_method = False):
        self.new_method = new_method

        self.use_tris_from_object = False
        """if set to False it will load the mesh from disk, otherwise from the loaded object"""

    def generate_poses(self, obj):
        object_name = obj.getName()

