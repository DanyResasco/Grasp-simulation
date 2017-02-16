
# root = ~/usr/local/cuda-7.0
from __future__ import division
import numpy as np
import os, sys, theano, traceback
import theano.tensor as T
import matplotlib.pyplot as plt
from read_as_coord_array import read_as_coord_array,read_as_3d_array
import csv
import sys
import random
import math
from klampt.math import so3
from IPython import embed
import Image

# prev_obj = {}
# prev_obj['ycb'] = [f for f in os.listdir('../data/objects/ycb')]
# prev_obj['apc2015'] = [f for f in os.listdir('../data/objects/apc2015')]
# prev_obj['princeton'] = [f for f in os.listdir('../data/objects/princeton')]


Input_name = [f for f in os.listdir('NNSet/Image')] #all binvox in binvox format
Output = {}
Output['Pose'] = [f for f in os.listdir('NNSet/pose')]

Input_training = []
Training_y = []
Input_validate = []
Validate_y = []
Test_y = []
Input_test = []
binvox = {}




def Set_input(objectName,vector_set,i):

    
    # for i in range(0,10)::
        nome = os.path.splitext(objectName)[0] + '_rotate_'+ str(i)+ '.jpg'
        obj_dataset = 'NNSet/Image/%s/%s'%(os.path.splitext(objectName)[0],nome)
        im = Image.open(obj_dataset).convert('L')
        vector_set.append(np.asarray(im).reshape(256,256))

def Set_vector(object_name, vector_set,input_set):
    '''Read the poses and store it into a vector'''
    name = os.path.splitext(object_name)[0]
    folder = 'NNSet/Image/%s'%(name)
    n_poses = len(list(os.listdir(folder)))//11

    # embed()
    for object_set, objects_in_set in Output.items():
        if object_name in objects_in_set:
            obj_dataset = 'NNSet/pose/%s'%(object_name)
            with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
                file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
                for row in file_reader:
                    T = row[9:]
                    row_t = list(so3.rpy(row)) + list(T)
                    n_var = int(10*n_poses)
                    # embed()
                    # for i in range(0,n_var): #10 variation for each label
                    vector_set.append(np.array(row_t))
                    Set_input(object_name,input_set,0)
                    break

# def Find_binvox(all_obj):
#     '''Remove all objects that doesn't has a binvox'''
#     vector = []
#     for object_name in all_obj:
#         if object_name in Input_name: #binvox exist!
#             vector.append(object_name) #save obj
#             Save_binvox(object_name)
#     return vector


#Mi da errore qui.. mi dice che non riesce a convertire il tipo di dato,
def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables """
    data_y, data_x = data_xy
 
    'data must be an numpy array'
    shared_x = theano.shared(np.array(data_x, theano.config.floatX), borrow=True)
    shared_y = theano.shared(np.array(data_y, theano.config.floatX), borrow=True)

    return shared_x, shared_y

def Input_output():
    all_obj = []
    for objects_name in Output.values():
        all_obj += objects_name

    #Sicuro ce' un metodo piu' intelligente
    # all_obj = Find_binvox(all_obj_tmp)

    random.shuffle(all_obj)
    #Temporaly vector. I store the file not the pose!! 
    Training_label_set = [x for i,x in enumerate(all_obj) if i <= len(all_obj)*.85 ]
    Validate_label_set = [x for i,x in enumerate(all_obj) if i >= len(all_obj)*.85 and i <len(all_obj)*.95]
    Test_label_set = [x for i,x in enumerate(all_obj) if i >= len(all_obj)*.95 and i<len(all_obj)*1]

    #Open the respectively file and take all poses
    for object_name in Training_label_set:
        Set_vector(object_name, Training_y,Input_training)

    for object_name in Validate_label_set:
        Set_vector(object_name, Validate_y,Input_validate)

    for object_name in Test_label_set:
        Set_vector(object_name, Test_y,Input_test)

    print len(Training_y) + len(Validate_y) + len(Test_y)
    print len(Input_training) + len(Input_validate)+ len(Input_test)

    Training_ = [Training_y, Input_training]
    Validate_ = [Validate_y,Input_validate ]
    Test_ = [Test_y,Input_test ]

    # embed()

    # print "Input_training", len(Input_training)
    # print "Training_y",len(Training_y)


    # print "Input_validate", len(Input_validate)
    # print "Validate_y",len(Validate_y)

    # print "Input_test", len(Input_test)
    # print "Test_label_set",len(Input_test)

    # print shared_dataset(Training_)[0]
    # embed()
    return shared_dataset(Training_), shared_dataset(Validate_),shared_dataset(Test_)


    # return Training_, Validate_ ,Test_


if __name__ == '__main__':
    T,V,TT = Input_output()
    # print "t[0]", T[0].get_value()
    # print "t[0].type", T[0].type
    # print 't[1]',T[1].get_value()
    # print 't[1].type', T[1].type




# for nome  in Training_label_set:
#     if nome not in Input_name:
#         print Training_label_set[Training_label_set.index(nome)]


# Output_name = [f for f in os.listdir('NNSet/Pose')]
# Remove = []

# Training = []
# Validate = []
# Test = []

# Input_net = []
# Output_net = []


# def Read_N_row(obj_list):
#     row_count = 0
#     for object_name in obj_list:
#         for object_set, objects_in_set in objects.items():
#             if object_name in objects_in_set:
#                 obj_dataset = 'NNSet/%s/%s'%(object_set,object_name)
#                 with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
#                     file_reader = csv.reader(csvfile, delimiter=',')
#                     row_count += sum(1 for row in file_reader)
#     return row_count





# class DataFeeder:
# 	def __init__(self, init_idx=1, test_idx=0):
# 		self.init_idx = init_idx
# 		self.cur_idx = init_idx
# 		self.test_idx = test_idx

# 	def next_training_set_shared(self):
# 		try:
# 			data = np.load('data/data_%i.npz'%self.cur_idx) #legge i file dal formato zippato di numpy
# 		except IOError:
# 			traceback.print_exc()
# 			print 'done all chunk files at %i'%self.cur_idx
# 			self.cur_idx = self.init_idx
# 		data = np.load('data/data_%i.npz'%self.cur_idx)
# 		X = data['X']
# 		y = data['y']
# 		X_loc = X[:, 0:1]
# 		X_occ = X[:, 1:]
# 		X_occ_shared = theano.shared(X_occ, borrow=True)
# 		X_loc_shared = theano.shared(X_loc, borrow=True)
# 		y_shared = theano.shared(y, borrow=True)
# 		self.cur_idx += 1
# 		return X_occ_shared, X_loc_shared, y_shared
	
# 	def next_training_set_raw(self):
# 		try:
# 			data = np.load('data/data_%i.npz'%self.cur_idx)
# 		except IOError:
# 			traceback.print_exc()
# 			print 'done all chunk files at %i'%self.cur_idx
# 			self.cur_idx = self.init_idx
# 		if self.cur_idx%1000==0:
# 			print 'using data %i'%self.cur_idx
# 		data = np.load('data/data_%i.npz'%self.cur_idx)
# 		X = data['X']
# 		y = data['y']
# 		X_loc = X[:, 0:1]
# 		X_occ = X[:, 1:]
# 		self.cur_idx += 1
# 		return X_occ, X_loc, y
	
# 	def test_set_shared(self):
# 		data = np.load('data/data_%i.npz'%self.test_idx)
# 		X = data['X']
# 		y = data['y']
# 		X_loc = X[:, 0:1]
# 		X_occ = X[:, 1:]
# 		X_occ_shared = theano.shared(X_occ, borrow=True)
# 		X_loc_shared = theano.shared(X_loc, borrow=True)
# 		y_shared = theano.shared(y, borrow=True)
# 		return X_occ_shared, X_loc_shared, y_shared