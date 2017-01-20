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
objects = {}
objects['Percentage'] = [f for f in os.listdir('NNSet/Percentage')]
objects['Nsuccessfull'] = [f for f in os.listdir('NNSet/Nsuccessfull')]
 
# prev_obj = {}
# prev_obj['ycb'] = [f for f in os.listdir('../data/objects/ycb')]
# prev_obj['apc2015'] = [f for f in os.listdir('../data/objects/apc2015')]
# prev_obj['princeton'] = [f for f in os.listdir('../data/objects/princeton')]
 
 
Input_name = [f for f in os.listdir('NNSet/binvox/Binvox')] #all binvox in binvox format
Output = {}
Output['Pose'] = [f for f in os.listdir('NNSet/Pose')]
 
Input_training = []
Training_y = []
Input_validate = []
Validate_y = []
Test_y = []
Input_test = []
binvox = {}
 
def Save_binvox(nome):
    objpath = 'NNSet/binvox/Binvox/%s'%nome
    try:
        with open(objpath, 'rb') as f:
            # data = read_as_coord_array(f) # dimension not matched. are different for each objects
            # embed()
            data = read_as_3d_array(f)
            binvox[nome] = data
    except:
        print "No binvox in file %s "%(objpath)
 
 
 
def Set_input(objectName,vector_set):
        vector_set.append(binvox[objectName])
 
 
def Set_vector(object_name, vector_set,input_set):
    '''Read the poses and store it as rpy into a vector'''
    # time_first = 0
    for object_set, objects_in_set in Output.items():
        if object_name in objects_in_set:
            obj_dataset = 'NNSet/Pose/%s'%(object_name)
            with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
                file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
                for row in file_reader:
                    # Matrix_ = so3.matrix(row)
                    # print 'object_name',object_name
                    T = row[9:]
                    # print 't',T
                    # print list(so3.rpy(row))
                    row_t = list(so3.rpy(row)) + list(T)
                    # print 'row_t',row_t
                    # embed()
                    # row_t = (list(so3.rpy(row)) + list(T))
                    vector_set.append(np.array(row_t))
                    input_set.append(binvox[object_name])
                    break #train with only one pose
                    # if time_first is 0:
                    #     input_set = binvox[object_name]
                    #     time_first = 1
                    # else:
                    #     input_set = np.concatenate((input_set,binvox[object_name]))
                    # Set_input(object_name,input_set)
    # time_first = 0
 
def Find_binvox(all_obj):
    '''Remove all objects that doesn't have a binvox'''
    vector = []
    for object_name in all_obj:
        if object_name in Input_name: #binvox exist!
            vector.append(object_name) #save obj
            Save_binvox(object_name)
    return vector
 
 
def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables """
    data_y, data_x = data_xy
 
    'data must be an numpy array'
    
    # embed()
    # print len(data_temp)
    shared_x = theano.shared(np.array(data_x, theano.config.floatX), borrow=True)
    # embed()

    shared_y = theano.shared(np.array(data_y, theano.config.floatX), borrow=True)
    # shared_y = theano.shared(data_y, borrow=True)
    # print shared_x.type()
    # embed()

    return shared_x, shared_y
 
def Input_output():
    all_obj_tmp = []
    for objects_name in Output.values():
        all_obj_tmp += objects_name
 
    #Sicuro ce' un metodo piu' intelligente
    all_obj = Find_binvox(all_obj_tmp)
 
    random.seed(0)
    random.shuffle(all_obj)
    #Temporaly vector. I store the file not the pose!! 
    Training_label_set = [x for i,x in enumerate(all_obj) if i <= len(all_obj)*.85 ]
    Validate_label_set = [x for i,x in enumerate(all_obj) if i >= len(all_obj)*.85 and i <len(all_obj)*.90]
    Test_label_set = [x for i,x in enumerate(all_obj) if i >= len(all_obj)*.90 and i<len(all_obj)*1]
 
 
    #Open the respectively file and take all poses
    for object_name in Training_label_set:
        Set_vector(object_name, Training_y,Input_training)
    # embed()

 
    for object_name in Validate_label_set:
        Set_vector(object_name, Validate_y,Input_validate)
 
    for object_name in Test_label_set:
        Set_vector(object_name, Test_y,Input_test)
 
    print "label", len(Training_y) + len(Validate_y) + len(Test_y)
    print 'input',len(Input_training) + len(Input_validate)+ len(Input_test)
 
    Training_ = [Training_y, Input_training]
    Validate_ = [Validate_y,Input_validate ]
    Test_ = [Test_y,Input_test ]
 
    # print "Input_training", len(Input_training)
    # print "Training_y",len(Training_y)
 
 
    # print "Input_validate", len(Input_validate)
    # print "Validate_y",len(Validate_y)
 
    # print "Input_test", len(Input_test)
    # print "Test_label_set",len(Input_test)
 
    # print shared_dataset(Training_)[0]
 
    return shared_dataset(Training_), shared_dataset(Validate_),shared_dataset(Test_)
 
 
    # return Training_, Validate_ ,Test_
 
 
if __name__ == '__main__':
    T,V,TT = Input_output()