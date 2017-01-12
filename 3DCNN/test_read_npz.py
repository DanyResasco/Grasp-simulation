import numpy as np
# import theano
import theano.tensor as T
import csv
# from __future__ import division
# import numpy as np
import os, sys, theano, traceback
# import theano.tensor as T
import matplotlib.pyplot as plt
from read_as_coord_array import read_as_coord_array,read_as_3d_array
# import csv
import sys
import random
import math
from klampt.math import so3
from IPython import embed
# from read_as_coord_array import read_as_coord_array,read_as_3d_array


if __name__ == '__main__':
		data = np.load('3d6Cnn3fcl_5000p.npz')
		o_w = data['output_W'].astype(theano.config.floatX)
		o_b = data['output_b'].astype(theano.config.floatX)
		print "weight",o_w
		print 'bias', o_b
		truth = []
		binvox = {}
		objpath = 'NNSet/binvox/Binvox/1_and_a_half_in_metal_washer.csv'
		with open(objpath, 'rb') as f:
		    # data = read_as_coord_array(f) # dimension not matched. are different for each objects
		    data = read_as_3d_array(f)
		    binvox['1_and_a_half_in_metal_washer'] = data

		pred = T.dot(binvox['1_and_a_half_in_metal_washer'], o_w) + o_b
		embed()

		with open('NNSet/Pose/1_and_a_half_in_metal_washer.csv', 'rb') as csvfile: #open the file in read mode
		    file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
		    for row in file_reader:
		        # Matrix_ = so3.matrix(row)
		        T = row[9:]
		        # print list(so3.rpy(row))
		        row_t = list(so3.rpy(row)) + list(T)
		        truth.append(np.array(row_t))
		print truth[0]
		print pred.eval()
		print pred - truth[0]
		print pred - truth[1]