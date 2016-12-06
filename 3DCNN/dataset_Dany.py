
from __future__ import division
import numpy as np
import os, sys, theano, traceback
import theano.tensor as T
import matplotlib.pyplot as plt


import csv
import sys
import random
import math

objects = {}
objects['Percentage'] = [f for f in os.listdir('NNSet/Percentage')]
objects['Nsuccessfull'] = [f for f in os.listdir('NNSet/Nsuccessfull')]

Input_name = [f for f in os.listdir('NNSet/binvox')]


def Set_vector(object_name, vector_set):
    # for object_name in nome_obj:
        for object_set, objects_in_set in objects.items():
            if object_name in objects_in_set:
                obj_dataset = 'NNSet/%s/%s'%(object_set,object_name)
                with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
                    file_reader = csv.reader(csvfile, delimiter=',')
                    for row in file_reader:
                        vector_set.append(row)



def Set_input(nome_obj,vector_set):
    if nome_obj in Input_name:
        obj_dataset = 'NNSet/binvox/%s'%Input_name[Input_name.index(nome_obj)]
        try:
            with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
                file_reader = csv.reader(csvfile, delimiter=',')
                temp_vecto = []
                for row in file_reader:
                    temp_vecto.append(row)
                vector_set.append(temp_vecto)
        except:
            print "No binvox in file", obj_dataset

def Find_binvox(all_obj):
    vector = []
    for object_name in all_obj:
        if object_name in Input_name:
            vector.append(object_name)
        # if object_name not in Input_name:
        #     print object_name
        #     del all_obj[all_obj.index(object_name)]
    return vector

# def Input_output():
all_obj_tmp = []
for objects_name in objects.values():
    all_obj_tmp += objects_name

#Sicuro ce' un metodo piu' intelligente
all_obj = Find_binvox(all_obj_tmp)

random.shuffle(all_obj)
Training_label_set = [x for i,x in enumerate(all_obj) if i <= len(all_obj)*.85 ]
Validate_label_set = [x for i,x in enumerate(all_obj) if i >= len(all_obj)*.85 and i <len(all_obj)*.95]
Test_label_set = [x for i,x in enumerate(all_obj) if i >= len(all_obj)*.95 and i<len(all_obj)*1]

Training_y = []
Input_training = []
for object_name in Training_label_set:
    # print object_name
    Set_vector(object_name, Training_y)
    Set_input(object_name,Input_training)
Validate_y = []
Input_validate = []
for object_name in Validate_label_set:
    Set_vector(object_name, Validate_y)
    Set_input(object_name,Input_validate)

Test_y = []
Input_test = []
for object_name in Test_label_set:
    Set_vector(object_name, Test_y)
    Set_input(object_name,Input_test)

print len(Training_label_set) + len(Validate_label_set) + len(Test_label_set)
print len(Input_training) + len(Input_validate)+ len(Input_test)

return Training_label_set,Input_training, Validate_label_set, Input_validate, Test_label_set, Input_test

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