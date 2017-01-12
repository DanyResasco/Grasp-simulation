

from __future__ import division

import numpy as np

import theano
import theano.tensor as T


class ContOutputLayer(object):
	"""
	Continuous Regression Class
	"""

	def __init__(self, input, n_in, W=None, b=None):
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

		if W is None:
			self.W = theano.shared(value=np.zeros(n_in, dtype=theano.config.floatX), name='W', borrow=True)
		else:
			assert isinstance(W, np.ndarray), 'W must be an numpy array'
			self.W = theano.shared(value=W.astype(theano.config.floatX), borrow=True)

		# initialize the biases b as a vector of n_out 0s
		if b is None:
			self.b = theano.shared(value=np.array(0, dtype=theano.config.floatX), name='b', borrow=True)
		else:
			assert isinstance(b, np.ndarray), 'b must be an numpy array'
			self.b = theano.shared(value=b.astype(theano.config.floatX), borrow=True)

		self.y_pred = T.dot(input, self.W) + self.b
		# parameters of the model
		self.params = [self.W, self.b]

		# keep track of model input
		self.input = input




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
			return T.mean(T.pow(y-self.y_pred, 2))

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
