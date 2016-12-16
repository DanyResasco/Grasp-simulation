
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



Input_name = [f for f in os.listdir('NNSet/binvox/Binvox')] #all binvox in binvox format
Output = {}
Output['Pose'] = [f for f in os.listdir('NNSet/Pose')]








#Mi da errore qui.. mi dice che non riesce a convertire il tipo di dato,
def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables """
    data_y, data_x = data_xy

    'data must be an numpy array'
    #convert 4d to matrix
    # data_x = np.ndarray.astype(np.array(data_x),dtype='float32')
    # data_x = data_x.reshape(data_x.shape[0], -1) 

    data_temp = []
    print'len(data_x)', len(data_x)
    for i in range(0,len(data_x)):
        data_temp.append(data_x[i].reshape(64,64,64))

    print len(data_temp)
    shared_x = theano.shared(np.array(data_temp, theano.config.floatX), borrow=True)
    # embed()
    #convert matrix to vector
    # data_y = np.ndarray.astype(np.array(data_y),dtype='float32')
    data_y_temp = []
    print'len(data_y)', len(data_y)
    for i in range(0,len(data_y)):
        data_y_temp.append(data_y[i].reshape(-1) )

    # print data_y_temp
    # data_y = data_y.reshape(-1) 
    shared_y = theano.shared(np.array(data_y_temp, theano.config.floatX), borrow=True)
    # print shared_y.type
    # print shared_y.get_value()
    return shared_x, shared_y


class DanyDataset():
    """docstring for DanyDataset"""
    def __init__(self, init_idx=0):
        self.init_idx = init_idx
        self.val_idx = init_idx
        self.test_idx = init_idx
        self.Input_training = []
        self.Training_y = []
        self.Input_validate = []
        self.Validate_y = []
        self.Test_y = []
        self.Input_test = []
        self.binvox = {}
        self._Set_All_Dataset()

    def Save_binvox(self, nome):
        objpath = 'NNSet/binvox/Binvox/%s'%nome
        try:
            with open(objpath, 'rb') as f:
                # data = read_as_coord_array(f) # dimension not matched. are different for each objects
                data = read_as_3d_array(f)
                self.binvox[nome] = data
        except:
            print "No binvox in file %s "%(objpath)



    def Set_vector(self,object_name, vector_set,input_set):
        '''Read the poses and store it into a vector'''
        for object_set, objects_in_set in Output.items():
            if object_name in objects_in_set:
                obj_dataset = 'NNSet/Pose/%s'%(object_name)
                with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
                    file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
                    for row in file_reader:
                        Matrix_ = so3.matrix(row)
                        T = row[7:10]
                        row_t = (list(so3.rpy(row)) + list(T))
                        vector_set.append(np.array(row_t))
                        input_set.append(self.binvox[object_name])


    def Find_binvox(self,all_obj):
        '''Remove all objects that doesn't has a binvox'''
        vector = []
        for object_name in all_obj:
            if object_name in Input_name: #binvox exist!
                vector.append(object_name) #save obj
                self.Save_binvox(object_name)
        return vector

    def _Set_All_Dataset(self):
        all_obj_tmp = []
        for objects_name in Output.values():
            all_obj_tmp += objects_name

        #Sicuro ce' un metodo piu' intelligente
        all_obj = self.Find_binvox(all_obj_tmp)

        random.seed(0)
        random.shuffle(all_obj)
        #Temporaly vector. I store the file not the pose!! 
        Training_label_set = [x for i,x in enumerate(all_obj) if i <= len(all_obj)*.85 ]
        Validate_label_set = [x for i,x in enumerate(all_obj) if i >= len(all_obj)*.85 and i <len(all_obj)*.95]
        Test_label_set = [x for i,x in enumerate(all_obj) if i >= len(all_obj)*.95 and i<len(all_obj)*1]

        # #Open the respectively file and take all poses
        for object_name in Training_label_set:
            self.Set_vector(object_name, self.Training_y,self.Input_training)

        for object_name in Validate_label_set:
            self.Set_vector(object_name, self.Validate_y,self.Input_validate)

        for object_name in Test_label_set:
            self.Set_vector(object_name, self.Test_y,self.Input_test)

        # print len(Training_y) + len(Validate_y) + len(Test_y)
        # print len(Input_training) + len(Input_validate)+ len(Input_test)

        # Training_ = [Training_y, Input_training]
        # Validate_ = [Validate_y,Input_validate ]
        # Test_ = [Test_y,Input_test ]

        # print "Input_training", len(Input_training)
        # print "Training_y",len(Training_y)


        # print "Input_validate", len(Input_validate)
        # print "Validate_y",len(Validate_y)

        # print "Input_test", len(Input_test)
        # print "Test_label_set",len(Input_test)

        # print shared_dataset(Training_)[0]

        # return shared_dataset(Training_), shared_dataset(Validate_),shared_dataset(Test_)


    # return Training_, Validate_ ,Test_

    def Take_one_element_training(self):
        # dx = T.tensor4('dx')
        # data_temp = dx.reshape((1, 1, 64, 64, 64))

        data_temp = (self.Input_training[self.init_idx]).reshape(( 1, 64, 64, 64))
        # print data_temp.shape
        # data_temp.append(self.Input_training[self.init_idx].reshape(64,64,64))
        shared_x = theano.shared(np.array(data_temp, theano.config.floatX), borrow=True)

        data_y_temp = []

        data_y_temp.append( self.Training_y[self.init_idx].reshape(-1) )

        shared_y = theano.shared(np.array(np.array(data_y_temp).reshape(-1) , theano.config.floatX), borrow=True)
        self.init_idx +=1

        return shared_x,shared_y



    def Take_one_element_test(self):
        # data_temp = []

        # data_temp.append(self.Input_test[self.test_idx].reshape(64,64,64))
        data_temp = (self.Input_test[self.init_idx]).reshape(( 1, 64, 64, 64))
        shared_x = theano.shared(np.array(data_temp, theano.config.floatX), borrow=True)

        data_y_temp = []

        data_y_temp.append( self.Test_y[self.test_idx].reshape(-1) )

        shared_y = theano.shared(np.array(np.array(data_y_temp).reshape(-1) , theano.config.floatX), borrow=True)
        self.test_idx +=1

        return shared_x,shared_y

    def Take_one_element_Validation(self):
        # data_temp = []

        # data_temp.append(self.Input_validate[self.val_idx].reshape(64,64,64))
        data_temp = (self.Input_validate[self.init_idx]).reshape(( 1, 64, 64, 64))
        shared_x = theano.shared(np.array(data_temp, theano.config.floatX), borrow=True)

        data_y_temp = []

        data_y_temp.append( self.Validate_y[self.val_idx].reshape(-1) )

        shared_y = theano.shared(np.array(np.array(data_y_temp).reshape(-1) , theano.config.floatX), borrow=True)
        self.val_idx +=1

        return shared_x,shared_y