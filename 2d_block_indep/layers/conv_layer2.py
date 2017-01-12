
'''
Custom convolution layer supporting convolution of different images with different kernels
'''

from __future__ import division

import numpy as np
from math import sqrt

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d, relu

class ConvLayer2(object):
	"""Pool Layer of a convolutional network """

	def __init__(self, rng, input, filter_shape, image_shape, poolsize, W, batch_size, b=None, activation=relu):
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

		self.W = W

		# the bias is a 1D tensor -- one bias per output feature map
		if b is None:
			self.b = theano.shared(value=np.zeros((filter_shape[0],), dtype=theano.config.floatX), borrow=True, name='b')
		else:
			assert isinstance(b, np.ndarray), 'b must be an numpy array'
			self.b = theano.shared(value=b.astype(theano.config.floatX), borrow=True)

		single_image_shape = [s for s in image_shape]
		single_image_shape[0] = 1
		convs = [
			conv2d(
				input=input[i,:,:,:].reshape(single_image_shape),
				filters=self.W[i,:,:,:,:].reshape(filter_shape),
				input_shape=image_shape, 
				filter_shape=filter_shape
			) for i in xrange(batch_size)
		]

		conv_out = T.concatenate(convs, axis=0)

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
