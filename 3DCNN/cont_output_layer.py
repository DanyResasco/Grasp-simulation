

from __future__ import division

import numpy as np

import theano
import theano.tensor as T
from IPython import embed

class ContOutputLayer(object):
	"""
	Continuous Regression Class
	"""

	def __init__(self, input, n_in,n_out, W=None, b=None):
		""" Initialize the parameters of the logistic regression

		:type input: theano.tensor.TensorType
		:param input: symbolic variable that describes the input of the
					  architecture (one minibatch)

		:type input_mask: theano.tensor.TensorType
		:param input_mask: symbolic variable that describes relevance of input

		:type n_in: int
		:param n_in: number of input units, the dimension of the space in
					 which the datapoints lie

		"""
		# start-snippet-1

		self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX ), name='W', borrow=True )
		# initialize the biases b as a vector of n_out 0s
		self.b = theano.shared( value=np.zeros( n_out, dtype=theano.config.floatX ),name='b', borrow=True)

		# if W is None:
		# 	self.W = theano.shared(value=np.zeros(n_in, dtype=theano.config.floatX), name='W', borrow=True)
		# else:
		# 	assert isinstance(W, np.ndarray), 'W must be an numpy array'
		# 	self.W = theano.shared(value=W.astype(theano.config.floatX), borrow=True)

		# # initialize the biases b as a vector of n_out 0s
		# if b is None:
		# 	self.b = theano.shared(value=np.array(0, dtype=theano.config.floatX), name='b', borrow=True)
		# else:
		# 	assert isinstance(b, np.ndarray), 'b must be an numpy array'
		# 	self.b = theano.shared(value=b.astype(theano.config.floatX), borrow=True)

		self.y_pred = T.dot(input, self.W) + self.b

		# parameters of the model
		self.params = [self.W, self.b]

		# keep track of model input
		self.input = input



	def mse(self, y,nrow): # balanced penalization
		"""
		mean square-root error
		"""
		# embed()
		# One_ori = np.array([1,1,1,0,0,0])
		# One_tra = np.array([0,0,0,1,1,1])
		# diff = y-self.y_pred

		# windows = np.zeros((nrow,6))

		# NORMA = []
		# for i in range(0,nrow):
		# 	windows[i] = One_ori
		# 	dist_or = diff*windows
		# 	windows[i] = One_tra
		# 	dist_tra = diff*windows
		# 	NORMA.append(np.add(np.linalg.norm([dist_or[i]],2)*0.087,0.005*np.linalg.norm([dist_tra[i]],2)))
		# 	windows[i] = np.zeros(6) 


		# embed()

		NORMA =[] 
		for i in range(0,nrow):
			NORMA.append(np.add(np.linalg.norm([y[i,0:3]-self.y_pred[i,0:3]],2),np.linalg.norm([y[i,3:]-self.y_pred[i,3:]],2)) )


		# embed()

		# embed()
		return  T.mean(NORMA),y,self.y_pred

	def dany_error(self,y,nrow):
		import math
		pi = math.pi
		# embed()
		NORMA = 0
		for i in range(0,nrow):
			dx = T.min([abs(y[i,0]-self.y_pred[i,0]),(2*pi - abs(y[i,0]-self.y_pred[i,0]))])
			dy = T.min([abs(y[i,1]-self.y_pred[i,1]),(2*pi - abs(y[i,1]-self.y_pred[i,1]))])
			dz = T.min([abs(y[i,2]-self.y_pred[i,2]),(2*pi - abs(y[i,2]-self.y_pred[i,2]))])
			d_o = np.sqrt(np.add( np.add(T.pow(dx, 2),T.pow(dy, 2)),T.pow(dz, 2)))
			tx = np.array(y[i,3]-self.y_pred[i,3])
			ty = np.array(y[i,4]-self.y_pred[i,4])
			tz = np.array(y[i,5]-self.y_pred[i,5])
			t0 = np.array([tx,ty,tz])
			# embed()
			NORMA +=(np.add(d_o*0.01,0.005*np.linalg.norm(t0,2)))
			# embed()
		# NORMA = np.add(np.linalg.norm([y[0:3]-self.y_pred[0:3]],2),np.linalg.norm([y[3:]-self.y_pred[3:]],2)) #works if i use one pose for obj

		# return T.mean(NORMA),y,self.y_pred
		return NORMA,y,self.y_pred


	def cost(self, y, y_flag=None): # balanced penalization
		"""
		mean square-root error
		"""
		# check if y has same dimension of y_pred
		if y_flag is not None:
			assert y.ndim == self.y_pred.ndim and y_flag.ndim == self.y_pred.ndim, 'Dimension mismatch'
			valid_sum = T.sum(T.pow(y - self.y_pred, 2) * y_flag)
			valid_num = T.sum(y_flag)
			return valid_sum / valid_num
		else:
			assert y.ndim == self.y_pred.ndim, 'Dimension mismatch'
			a =T.mean(T.pow(y-self.y_pred, 2))
			embed()
			return a,y,self.y_pred


	def cost2(self, y):
		diff = y - self.y_pred
		diff_skewed = T.switch(diff>0, diff*20, diff)
		# diff_skewed = T.set_subtensor(diff[diff>0], diff[diff>0]*8) # penalize underestimation, favor overestimation
		return T.mean(T.pow(diff_skewed, 2))

	def errors(self, y):
		"""
		mean square-root error
		"""
		# check if y has same dimension of y_pred
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)
		return T.mean(T.pow(y-self.y_pred, 2))

	# def negative_log_likelihood(self, y):

	# 	return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
