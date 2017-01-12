
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
		data = np.load('data/data_%i.npz'%self.cur_idx)
		X, y = data['X'].astype(theano.config.floatX), data['y'].astype('int32')
		X_loc = X[:,0:4]
		X_occ = X[:,4:]
		self.cur_idx += 1
		X_occ_shared = theano.shared(X_occ, borrow=True)
		X_loc_shared = theano.shared(X_loc, borrow=True)
		y_shared = theano.shared(y, borrow=True)
		return X_occ_shared, X_loc_shared, y_shared
	
	def next_training_set_raw(self):
		try:
			data = np.load('data/data_%i.npz'%self.cur_idx)
		except IOError:
			traceback.print_exc()
			self.cur_idx = self.init_idx
			print 'done all chunk files'
		data = np.load('data/data_%i.npz'%self.cur_idx)
		X, y = data['X'].astype(theano.config.floatX), data['y'].astype('int32')
		X_loc = X[:,0:4]
		X_occ = X[:,4:]
		self.cur_idx += 1
		return X_occ, X_loc, y
	
	def test_set_shared(self):
		data = np.load('data/data_%i.npz'%self.test_idx)
		X, y = data['X'].astype(theano.config.floatX), data['y'].astype('int32')
		X_loc = X[:,0:4]
		X_occ = X[:,4:]
		X_occ_shared = theano.shared(X_occ, borrow=True)
		X_loc_shared = theano.shared(X_loc, borrow=True)
		y_shared = theano.shared(y, borrow=True)
		return X_occ_shared, X_loc_shared, y_shared
