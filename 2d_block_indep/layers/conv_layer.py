
from __future__ import division

import numpy as np
from math import sqrt

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d, relu

class ConvLayer(object):
	"""Pool Layer of a convolutional network """

	def __init__(self, rng, input, filter_shape, image_shape, poolsize, activation=relu, W=None, b=None):
		"""
		Allocate a ConvLayer with shared variable internal parameters.

		:type rng: np.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.dtensor4
		:param input: symbolic image tensor, of shape image_shape

		:type filter_shape: tuple or list of length 4
		:param filter_shape: (number of filters, num input feature maps,
							  filter height, filter width)

		:type image_shape: tuple or list of length 4
		:param image_shape: (batch size, num input feature maps,
							 image height, image width)

		:type poolsize: tuple or list of length 2
		:param poolsize: the downsampling (pooling) factor (#rows, #cols)
		"""

		assert image_shape[1] == filter_shape[1]
		self.input = input

		# there are "num input feature maps * filter height * filter width"
		# inputs to each hidden unit
		fan_in = np.prod(filter_shape[1:])
		# each unit in the lower layer receives a gradient from:
		# "num output feature maps * filter height * filter width" /
		#   pooling size
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
				   np.prod(poolsize))

		# for RELU: w = np.random.randn(n) * sqrt(2/n) where n is number of inputs
		# W_values = np.asarray(rng.normal(fan_in, size=filter_shape) / sqrt(2/fan_in), dtype=theano.config.floatX)
		# self.W = theano.shared(value=W_values, borrow=True, name='W')

		if W is None:
			W_bound = np.sqrt(6. / (fan_in + fan_out))
			self.W = theano.shared(
				value=rng.uniform(low=-W_bound, high=W_bound, size=filter_shape).astype(theano.config.floatX), 
				borrow=True
			)
		else:
			assert isinstance(W, np.ndarray), 'W must be an numpy array'
			self.W = theano.shared(value=W.astype(theano.config.floatX), borrow=True)
		

		# the bias is a 1D tensor -- one bias per output feature map
		if b is None:
			self.b = theano.shared(value=np.zeros((filter_shape[0],), dtype=theano.config.floatX), borrow=True, name='b')
		else:
			assert isinstance(b, np.ndarray), 'b must be an numpy array'
			self.b = theano.shared(value=b.astype(theano.config.floatX), borrow=True)

		# convolve input feature maps with filters
		conv_out = conv2d(
			input=input,
			filters=self.W,
			filter_shape=filter_shape,
			input_shape=image_shape
		)

		# pool each feature map individually, using maxpooling
		pooled_out = pool.pool_2d(
			input=conv_out,
			ds=poolsize,
			ignore_border=True
		)

		# add the bias term. Since the bias is a vector (1D array), we first
		# reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
		# thus be broadcasted across mini-batches and feature map
		# width & height
		self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		# store parameters of this layer
		self.params = [self.W, self.b]

		# keep track of model input
		self.input = input
