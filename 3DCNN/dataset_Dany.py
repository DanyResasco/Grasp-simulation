
from __future__ import division
import numpy as np
import os, sys, theano, traceback
import theano.tensor as T
import matplotlib.pyplot as plt

# def make_dataset():
import csv
import sys
import random

objects = {}
objects['training'] = [f for f in os.listdir('3DCNN/NNSet')-15]
for Name in objects.values():
        all_objects += Name

print all_objects


# obj_dataset = '3DCNN/NNSet/%s.csv'%Name
# with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
#     file_reader = csv.reader(csvfile, delimiter=',')
#     row_count = sum(1 for row in file_reader)
#     csvfile.seek(0)
#     Line_old = []
#     End = False
#     for row in file_reader:
#     	print row
    # if End is not True:
    # 	line = random.randint(0,row_count)
    # 	if line not in Line_old:
	   #  	Line_old.append(line)
	   #  	set_train_x = [row for idx, row in enumerate(file_reader) if idx is line]
	   #  	print set_train_x[0][0]
	   #  	# set_train_y = [row for idx, row in enumerate(file_reader) if idx is line]
	   #  	# print set_train_y



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
