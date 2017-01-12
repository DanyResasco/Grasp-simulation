
from __future__ import division
import numpy as np
import os, sys, theano, traceback
import theano.tensor as T
import matplotlib.pyplot as plt

class DataFeeder:
	def __init__(self, init_idx=1, test_idx=0):
		self.init_idx = init_idx
		self.cur_idx = init_idx
		self.test_idx = test_idx

	def next_training_set_shared(self):
		try:
			data = np.load('data/%i.npz'%self.cur_idx)
		except IOError:
			traceback.print_exc()
			print 'done all chunk files at %i'%self.cur_idx
			self.cur_idx = self.init_idx
		data = np.load('data/%i.npz'%self.cur_idx)
		Xy = data['Xy'].astype(theano.config.floatX)
		X_loc = Xy[:, 0:2]
		X_occ = Xy[:, 2:10002]
		y = Xy[:, 10002]
		y_flag = Xy[:, 10003]
		X_occ_shared = theano.shared(X_occ, borrow=True)
		X_loc_shared = theano.shared(X_loc, borrow=True)
		y_shared = theano.shared(y, borrow=True)
		y_flag_shared = theano.shared(y_flag, borrow=True)
		self.cur_idx += 1
		return X_occ_shared, X_loc_shared, y_shared, y_flag_shared
	
	def next_training_set_raw(self):
		try:
			data = np.load('data/%i.npz'%self.cur_idx)
		except IOError:
			traceback.print_exc()
			print 'done all chunk files at %i'%self.cur_idx
			self.cur_idx = self.init_idx
		data = np.load('data/%i.npz'%self.cur_idx)
		Xy = data['Xy'].astype(theano.config.floatX)
		X_loc = Xy[:, 0:2]
		X_occ = Xy[:, 2:10002]
		y = Xy[:, 10002]
		y_flag = Xy[:, 10003]
		self.cur_idx += 1
		return X_occ, X_loc, y, y_flag
	
	def test_set_shared(self):
		data = np.load('data/%i.npz'%self.test_idx)
		Xy = data['Xy'].astype(theano.config.floatX)
		X_loc = Xy[:, 0:2]
		X_occ = Xy[:, 2:10002]
		y = Xy[:, 10002]
		y_flag = Xy[:, 10003]
		X_occ_shared = theano.shared(X_occ, borrow=True)
		X_loc_shared = theano.shared(X_loc, borrow=True)
		y_shared = theano.shared(y, borrow=True)
		y_flag_shared = theano.shared(y_flag, borrow=True)
		return X_occ_shared, X_loc_shared, y_shared, y_flag_shared
