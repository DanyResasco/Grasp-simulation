
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
import cv2

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

def im2double(im):
    if im is None:
        assert 'Problem with import image'
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out
 
 

# def standardization(y):

#         roll_std  = np.std(y[:,0])
#         roll_mean = np.mean(y[:,0])
#         pitch_std  = np.std(y[:,1])
#         pitch_mean = np.mean(y[:,1])
#         yaw_std  = np.std(y[:,2])
#         yaw_mean = np.mean(y[:,2])

#         x_std = np.std(y[:,3])
#         x_mean = np.mean(y[:,3])
#         y_std = np.std(y[:,4])
#         y_mean = np.mean(y[:,4])
#         z_std = np.std(y[:,5])
#         z_mean = np.mean(y[:,5])

#         std_ = [roll_std,pitch_std,yaw_std,x_std,y_std,z_std]
#         mean_ = [roll_mean,pitch_mean,yaw_mean,x_mean,y_mean,z_mean]

#         vect_temp = []
#         for i in range(0,len(y)):

#             r = (y[i,0] - mean_[0])/ std_[0]
#             p = (y[i,1] - mean_[1] )/ std_[1]
#             w = (y[i,2] - mean_[2] )/std_[2]
#             x = (y[i,3] - mean_[3] )/std_[3]
#             v = (y[i,4] - mean_[4] )/std_[4]
#             z = (y[i,5] - mean_[5] )/std_[5]
#             # embed()
#             vect_temp.append(np.array([r,p,w,x,v,z]))

#         return vect_temp,std_,mean_




def Set_input(objectName,image_set,i):

    
    # for i in range(0,10)::
        nome = os.path.splitext(objectName)[0] + '_rotate_'+ str(i)+ '.png'
        try:
            obj_dataset = 'NNSet/Image/%s/%s'%(os.path.splitext(objectName)[0],nome)
            im = Image.open(obj_dataset)
            # img2 = cv2.imread(obj_dataset) # Read in your image
            # out = im2double(img2) # Convert to normalized floating poin
            # embed
            image_set.append(np.asarray(im))
        except:
            print obj_dataset,'NOT FOUND'


def Check_INterval(angle,intervallo):
    if angle < math.radians(-intervallo):
        angle = angle + math.radians(360)
    if angle < math.radians(intervallo):
        angle = angle - math.radians(360)
    return angle

def Check_Metric(rpy):
    roll, pitch,yaw = rpy
    roll = Check_INterval(roll,180)
    yaw = Check_INterval(yaw,180)
    pitch = Check_INterval(pitch,90)
    return [roll,pitch,yaw]




def Set_vector(object_name, vector_set,input_set):
    '''Read the poses and store it into a vector'''
    name = os.path.splitext(object_name)[0]
    folder = 'NNSet/Image/%s'%(name)
    n_poses = len(list(os.listdir(folder)))//11
    n_row = 1
    n_var = int(10*n_poses)
    # embed()
    temp=[]
    for object_set, objects_in_set in Output.items():
        if object_name in objects_in_set:
            obj_dataset = 'NNSet/pose/%s'%(object_name)
            with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
                file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
                for row in file_reader:
                    T = row[9:]
                    # r_klampt = so3.matrix(row[0:9])
                    quaternion= so3.quaternion(row[0:9])
                    row_t = list(quaternion) + list(T)
                    # row_t = list(so3.rpy(row)) + list(T)
                    # vector_set.append(np.array(row_t))
                    # Set_input(object_name,input_set,0)
                    temp.append(np.array(row_t))
                # csvfile.close()

            for i in range(0,len(temp)): #10 variation for each label
                if i <= n_var:
                    vector_set.append(temp[i])
                    print object_name
                    Set_input(object_name,input_set,i)
                else:
                    break  


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
    random.seed(0)
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

    print '***********************************'
    for object_name in Test_label_set:
       Set_vector(object_name, Test_y,Input_test)


    # vect_std,std_,mean_ =standardization(np.vstack((np.array(Training_y),np.array(Validate_y),np.array(Test_y))))

    # # embed()
    # Training_label_set_STD = [vect_std[i] for i in range(0,len(Training_y))]
    # Validate_label_set_STD = [vect_std[i] for i in range(0,len(Validate_y))]
    # Test_label_set_STD = [vect_std[i] for i in range(0,len(Test_y))]







    print 'Test_y: ',len(Test_y)
    print len(Training_y) + len(Validate_y) + len(Test_y)
    print len(Input_training) + len(Input_validate)+ len(Input_test)

    Training_ = [Training_y, Input_training]
    Validate_ = [Validate_y,Input_validate ]
    Test_ = [Test_y,Input_test ]

    # embed()

    print "Input_training", len(Input_training)
    print "Training_y",len(Training_y)


    print "Input_validate", len(Input_validate)
    print "Validate_y",len(Validate_y)

    print "Input_test", len(Input_test)
    print "Test_label_set",len(Input_test)

    # print shared_dataset(Training_)[0]
    # embed()
    return shared_dataset(Training_), shared_dataset(Validate_),shared_dataset(Test_)
     # std_,mean_


    # return Training_, Validate_ ,Test_


if __name__ == '__main__':
    T,V,TT = Input_output()
