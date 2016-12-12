
# root = ~/usr/local/cuda-7.0
from __future__ import division
import numpy as np
import os, sys, theano, traceback
import theano.tensor as T
import matplotlib.pyplot as plt
from read_as_coord_array import read_as_coord_array
import csv
import sys
import random
import math

objects = {}
objects['Percentage'] = [f for f in os.listdir('NNSet/Percentage')]
objects['Nsuccessfull'] = [f for f in os.listdir('NNSet/Nsuccessfull')]

prev_obj = {}
prev_obj['ycb'] = [f for f in os.listdir('../data/objects/ycb')]
prev_obj['apc2015'] = [f for f in os.listdir('../data/objects/apc2015')]
prev_obj['princeton'] = [f for f in os.listdir('../data/objects/princeton')]


Input_name = [f for f in os.listdir('NNSet/binvox')]
Input_training = []
Training_y = []
Input_validate = []
Validate_y = []
Test_y = []
Input_test = []
binvox = {}


def Set_input(objectName,vector_set):
    # for objectName in object_list:

        objectName = os.path.splitext(objectName)[0]DataFeeder
        for object_set, objects_in_set in prev_obj.items():
            if objectName in objects_in_set: 
                if object_set == 'princeton':
                    objpath = '../data/objects/princeton/%s/tsdf_mesh.binvox'%objectName
                elif object_set == 'apc2015':
                    objpath = '../data/objects/apc2015/%s/meshes/poisson.binvox'%(objectName)
                else:
                    objpath = '../data/objects/ycb/%s/meshes/poisson_mesh.binvox'%(objectName)
                try:
                    with open(objpath, 'rb') as f:
                        data = read_as_coord_array(f)
                        vector_set.append(data[0])
                except:
                    print "No binvox in file %s "%(objpath)

        # try:
        #     with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
        #         file_reader = csv.reader(csvfile, delimiter=',')
        #         temp_vecto = []
        #         for row in file_reader:
        #             temp_vecto.append(row)
        #         vector_set.append(temp_vecto)
        # except:
        #     print "No binvox in file", obj_dataset



def Set_vector(object_name, vector_set,input_set):
    '''Read the poses and store it into a vector'''
    # for object_name in nome_obj:
    row_count =0
    for object_set, objects_in_set in objects.items():
        if object_name in objects_in_set:
            obj_dataset = 'NNSet/%s/%s'%(object_set,object_name)
            with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
                file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
                for row in file_reader:
                    row_temp = []
                    # for i in range(2,len(row)):
                    #     # print row[i].split(',')
                    #     row_temp = float(row[i].split(','))
                    #     # row_temp2 = [float(n) for n in r]
                    vector_set.append(row_temp)
                    # print len(vector_set)
                    Set_input(object_name,input_set)


def Save_binvox(nome):




def Find_binvox(all_obj):
    '''Remove all objects that doesn't has a binvox'''
    vector = []
    for object_name in all_obj:
        if object_name in Input_name: #binvox exist!
            vector.append(object_name) #save obj
            Save_binvx(object_name)
    return vector


#Mi da errore qui.. mi dice che non riesce a convertire il tipo di dato,
def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        # shared_x = theano.shared(np.asarray(data_x,
        #                                        dtype=theano.config.floatX),
        #                          borrow=borrow)
        # shared_y = theano.shared(np.asarray(data_y,
        #                                        dtype=theano.config.floatX),
        #                          borrow=borrow)


    # train_set_x = [train_set[i][0] for i in range(len(train_set))]



        'data must be an numpy array'
        shared_x = theano.shared(np.array(data_x, theano.config.floatX))
        shared_y = theano.shared(np.array(data_y, theano.config.floatX))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')






def Input_output():
    all_obj_tmp = []
    for objects_name in objects.values():
        all_obj_tmp += objects_name

    #Sicuro ce' un metodo piu' intelligente
    all_obj = Find_binvox(all_obj_tmp)

    random.shuffle(all_obj)
    Training_label_set = [x for i,x in enumerate(all_obj) if i <= len(all_obj)*.85 ]
    Validate_label_set = [x for i,x in enumerate(all_obj) if i >= len(all_obj)*.85 and i <len(all_obj)*.95]
    Test_label_set = [x for i,x in enumerate(all_obj) if i >= len(all_obj)*.95 and i<len(all_obj)*1]

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

    return shared_dataset(Training_), shared_dataset(Validate_),shared_dataset(Test_)
    # print "Input_training", len(Input_training)
    # print "Training_y",len(Training_y)


    # print "Input_validate", len(Input_validate)
    # print "Validate_y",len(Validate_y)

    # print "Input_test", len(Input_test)
    # print "Test_label_set",len(Input_test)

    # return Training_, Validate_ ,Test_


if __name__ == '__main__':
    Input_output()
    # import cProfile
    # import re
    # cProfile.run('Input_output()')







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