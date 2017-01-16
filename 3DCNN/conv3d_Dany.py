
from __future__ import division

import numpy as np
from math import sqrt

import theano
import theano.tensor as T

from maxpool3d import max_pool_3d
from theano.tensor.nnet import relu
from theano.tensor.nnet import sigmoid
from theano.tensor.nnet.conv3d2d import conv3d

class Conv3D(object):
	"""Pool Layer of a convolutional network """

	def __init__(self, rng, input, image_shape, filter_shape, poolsize, 
		activation=relu, W=None, b=None):
		"""
		image_shape is (num_imgs, num_channels, img_height, img_width, img_length)
		filter_shape is (num_kernels, num_channels, kernel_height, kernel_width, kernel_length)

		for theano.tensor.nnet.conv3d2d.conv3d(img, filter, image_shape, filter_shape, border_mode='valid'), 
		image_shape is (num_imgs, img_length, num_channels, img_height, img_width)
		filter_shape is (num_kernels, kernel_length, num_channels, kernel_height, kernel_width)
		"""
		assert image_shape[1] == filter_shape[1], 'Number of channels must be the same for input and kernel'
		# assert self.input.shape==tuple(image_shape), 'Actual and provided input shapes are not the same'

		fan_in = np.prod(filter_shape[1:])

		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
				   np.prod(poolsize))

		num_imgs, num_channels, img_height, img_width, img_length = image_shape
		num_kernels, num_channels, kernel_height, kernel_width, kernel_length = filter_shape

		# make shapes conforming to theano rules
		image_shape = (num_imgs, img_length, num_channels, img_height, img_width)
		filter_shape = (num_kernels, kernel_length, num_channels, kernel_height, kernel_width)

		if W is None:
			W_bound = np.sqrt(6. / (fan_in + fan_out)) #regola se usi tanh come activation function
			if activation is sigmoid:
				W_bound = W_bound*4
			self.W = theano.shared(
				value=rng.uniform(low=-W_bound, high=W_bound, size=filter_shape).astype(theano.config.floatX), 
				borrow=True
			)
		else:
			assert isinstance(W, np.ndarray), 'W must be an numpy array'
			assert W.shape==filter_shape, 'W must be of shape (%i,%i,%i,%i,%i)'%(num_kernels, kernel_length, num_channels, kernel_height, kernel_width)
			self.W = theano.shared(value=W.astype(theano.config.floatX), borrow=True)
		

		# the bias is a 1D tensor -- one bias per output feature map
		if b is None:
			self.b = theano.shared(value=np.zeros((filter_shape[0],), dtype=theano.config.floatX), borrow=True, name='b')
		else:
			assert isinstance(b, np.ndarray), 'b must be an numpy array'
			assert b.shape==(filter_shape[0],)
			self.b = theano.shared(value=b.astype(theano.config.floatX), borrow=True)


		input = input.dimshuffle(0, 4, 1, 2, 3)
		# convolve input feature maps with filters
		# conv_out is of shape (num_imgs, convolved_length, num_new_channels, convolved_width, convolved_height)
		conv_out = conv3d(
			signals=input,
			filters=self.W,
			filters_shape=filter_shape,
			signals_shape=image_shape
		)

		conv_out = conv_out.dimshuffle(0, 2, 3, 4, 1)

		# pool each feature map individually, using maxpooling
		if list(poolsize)==(1,1,1):
			pooled_out = conv_out
		else:
			pooled_out = max_pool_3d(
				input=conv_out,
				ds=poolsize,
				ignore_border=True
			)

		# add the bias term. Since the bias is a vector (1D array), we first
		# reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
		# thus be broadcasted across mini-batches and feature map
		# width & height
		self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x', 'x'))

		# store parameters of this layer
		self.params = [self.W, self.b]

		# keep track of model input
		self.input = input
