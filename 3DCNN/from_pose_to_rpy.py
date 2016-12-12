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


def Read_and_set_vector(nome_obj):
    '''Read the poses and store it into a vector'''
    for object_name in nome_obj:
        for object_set, objects_in_set in objects.items():
            print object_set
            if object_name in objects_in_set:
                print "qui"
                obj_dataset = 'NNSet/%s/%s'%(object_set,object_name)
                print obj_dataset
                with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
                    file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC, delimiter = ',')
                    caracter = ['[',',']
                    for row in file_reader:
                        row_temp = []

                        # vector_set.append(row_temp)
                        # # print len(vector_set)
                        # Set_input(object_name,input_set)


if __name__ == '__main__':

    all_obj_tmp = []
    for objects_name in objects.values():
        all_obj_tmp += objects_name

    try:
        obj_name = sys.argv[1]
        Read_and_set_vector([obj_name])
    except:
        Read_and_set_vector(all_obj_tmp)