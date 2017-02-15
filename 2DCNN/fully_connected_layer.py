
from __future__ import division

import numpy as np
from math import sqrt
from theano.tensor.nnet import sigmoid
import theano
import theano.tensor as T
from theano.tensor.nnet import relu

class FullyConnectedLayer(object):
	def __init__(self, rng, input, n_in, n_out, activation=sigmoid, W=None, b=None):
		"""
		Typical hidden layer of a MLP: units are fully-connected and have
		sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
		and the bias vector b is of shape (n_out,).

		NOTE : The nonlinearity used here is tanh

		Hidden unit activation is given by: tanh(dot(input,W) + b)

		:type rng: np.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.dmatrix
		:param input: a symbolic tensor of shape (n_examples, n_in)

		:type n_in: int
		:param n_in: dimensionality of input

		:type n_out: int
		:param n_out: number of hidden units

		:type activation: theano.Op or function
		:param activation: Non linearity to be applied in the hidden
						   layer
		"""
		self.input = input
		# end-snippet-1

		# `W` is initialized with `W_values` which is uniformely sampled
		# from sqrt(-6./(n_in+n_out)) and sqrt(6./(n_in+n_out))
		# for tanh activation function
		# the output of uniform if converted using asarray to dtype
		# theano.config.floatX so that the code is runable on GPU
		# Note : optimal initialization of weights is dependent on the
		#        activation function used (among other things).
		#        For example, results presented in [Xavier10] suggest that you
		#        should use 4 times larger initial weights for sigmoid
		#        compared to tanh
		#        We have no info for other function, so we use the same as
		#        tanh.

		# for RELU: w = np.random.randn(n) * sqrt(2/n) where n is number of inputs
			
		# W_values = np.asarray(rng.normal(n_in, size=(n_in, n_out)) / sqrt(2/n_in), dtype=theano.config.floatX)
		# self.W = theano.shared(value=W_values, name='W', borrow=True)

		if W is None:
			W_bound = np.sqrt(6. / (n_in + n_out))
			if activation is sigmoid:
				W_bound = W_bound*4
			self.W = theano.shared(
				value=rng.uniform(low=-W_bound, high=W_bound, size=(n_in, n_out)).astype(theano.config.floatX),
				borrow=True
			)
		else:
			assert isinstance(W, np.ndarray), 'W must be an numpy array'
			self.W = theano.shared(value=W.astype(theano.config.floatX), borrow=True)

		if b is None:
			self.b = theano.shared(value=np.zeros((n_out), dtype=theano.config.floatX), name='b', borrow=True)
		else:
			assert isinstance(b, np.ndarray), 'b must be an numpy array'
			self.b = theano.shared(value=b.astype(theano.config.floatX), borrow=True)


		lin_output = T.dot(input, self.W) + self.b
		self.output = activation(lin_output)
		# parameters of the model
		self.params = [self.W, self.b]