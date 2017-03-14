

from __future__ import division

import numpy as np

import theano
import theano.tensor as T
from IPython import embed
from theano_utils import _tensor_py_operators



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
		if W is None:
			self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX ), name='W', borrow=True )
		else:
			self.W = W
		# initialize the biases b as a vector of n_out 0s
		if b is None:
			self.b = theano.shared( value=np.zeros( n_out, dtype=theano.config.floatX ),name='b', borrow=True)
		else:
			self.b =b

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







	def dany_error(self,y,nrow):
		import math
		pi = math.pi
		ORI = []
		TRA = []
		temp = []
		NORMA = 0
		for i in range(0,nrow):
			dx = T.min([abs(y[i,0]-self.y_pred[i,0]),(2*pi - abs(y[i,0]-self.y_pred[i,0]))])
			dy = T.min([abs(y[i,1]-self.y_pred[i,1]),(2*pi - abs(y[i,1]-self.y_pred[i,1]))])
			dz = T.min([abs(y[i,2]-self.y_pred[i,2]),(2*pi - abs(y[i,2]-self.y_pred[i,2]))])

			d_o = np.sqrt(np.add( np.add(T.pow(dx, 2),T.pow(dy, 2)),T.pow(dz, 2)))

			tx = (y[i,3:]-self.y_pred[i,3:]).norm(2)
			NORMA +=(np.add(d_o*0.031,0.005*tx))
			ORI.append(d_o*0.031)
			TRA.append(0.005*tx)

		return NORMA,y,self.y_pred,T.cast(ORI,'float32'),T.cast(TRA,'float32')

	def cost_quaternion(self,y,nrow):

		ORI = []
		TRA = []
		temp = []
		NORMA = 0

		delta = [1*10^-12,0,0,0]
		y_p = []
		for i in range(0,nrow):

			q = 1 - abs(T.dot(y[i,0:4],((self.y_pred[i,0:4] + delta )/((self.y_pred[i,0:4]+delta).norm(2)) )))
			tx = (y[i,4:]-self.y_pred[i,4:]).norm(2)
			NORMA +=(np.add(q*0.031,0.005*tx  ))
			ORI.append(q*0.031)
			TRA.append(0.005*tx )
			# embed()

		return NORMA,y,self.y_pred,T.cast(ORI,'float32'),T.cast(TRA,'float32')



	def cost_quaternion_acos(self,y,nrow):

		ORI = []
		TRA = []
		temp = []
		NORMA = 0

		delta = [1*10^-12,0,0,0]
		y_p = []
		for i in range(0,nrow):

			q = T.arccos(abs(T.dot(y[i,0:4],((self.y_pred[i,0:4] + delta )/((self.y_pred[i,0:4]+delta).norm(2))))))
			tx = (y[i,4:]-self.y_pred[i,4:]).norm(2)
			NORMA +=(np.add(q*0.031,0.005*tx  ))
			ORI.append(q*0.031)
			TRA.append(0.005*tx )

		return NORMA,y,self.y_pred,T.cast(ORI,'float32'),T.cast(TRA,'float32')



	def cost_quaternion_min(self,y,nrow):

		ORI = []
		TRA = []
		# temp = []
		NORMA = 0
		import math
		delta = [1*10^-12,0,0,0]
		# y_p = []
		# embed()
		for i in range(0,nrow):
			q1_acos = T.arccos( T.dot( y[i,0:4],((self.y_pred[i,0:4] + delta )/((self.y_pred[i,0:4]+delta).norm(2))) ) ) 
			q2_1_acos = math.pi - T.arccos( T.dot( y[i,0:4],((self.y_pred[i,0:4] + delta )/((self.y_pred[i,0:4]+delta).norm(2))) ))

			# embed()
			q = T.min((q1_acos,q2_1_acos))
			tx = (y[i,4:]-self.y_pred[i,4:]).norm(2)
			NORMA +=(np.add(q*0.031,0.005*tx  ))
			ORI.append(q*0.031)
			TRA.append(0.005*tx )

		return NORMA,y,self.y_pred,T.cast(ORI,'float32'),T.cast(TRA,'float32')




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
			return T.mean(T.pow(y-self.y_pred, 2)),self.y_pred

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
