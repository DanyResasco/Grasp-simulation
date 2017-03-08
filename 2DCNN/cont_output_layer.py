

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
			tx = np.array(y[i,3]-self.y_pred[i,3])
			ty = np.array(y[i,4]-self.y_pred[i,4])
			tz = np.array(y[i,5]-self.y_pred[i,5])
			t0 = np.array([tx,ty,tz])
			# embed()
			# temp.append(np.add(d_o*0.031,0.005*np.linalg.norm(t0,2)))
			# NORMA +=(np.add(d_o*0.031,0.005*np.linalg.norm(t0,2)))
			NORMA +=(np.add(d_o*0.031,0.005*np.linalg.norm(t0,2)))
			ORI.append(d_o*0.031)
			TRA.append(0.005*np.linalg.norm(t0,2))

		return NORMA,y,self.y_pred,T.cast(ORI,'float32'),T.cast(TRA,'float32')

	def Normalization_dany(self,nrow):
		delta = [1*10^-12,0,0,0]
		y_p=[]
		for i in range(0,nrow):

			q0 = (np.array(self.y_pred[i,0]) + delta[0] )  / np.linalg.norm(np.array(self.y_pred[i,0]) + delta[0] )
			q1 = (np.array(self.y_pred[i,1]) + delta[1] )  /  np.linalg.norm(np.array(self.y_pred[i,1]) + delta[1] ) 
			q2 = (np.array(self.y_pred[i,2]) + delta[2] )  /  np.linalg.norm(np.array(self.y_pred[i,2]) + delta[2] )
			q3 = (np.array(self.y_pred[i,3]) + delta[3] )  /  np.linalg.norm(np.array(self.y_pred[i,3]) + delta[3] )

			x = self.y_pred[i,4]
			y = self.y_pred[i,5]
			z = self.y_pred[i,6]			

			temp = [q0,q1,q2,q3,x,y,z]
			y_p.append(temp)

		return y_p


	def cost_quaternion(self,y,nrow):

		ORI = []
		TRA = []
		temp = []
		NORMA = 0
		# p = self.y_pred
		# self.y_pred = self.Normalization_dany(nrow)
		# y_pred = self.y_pred
		# embed()
		delta = [1*10^-12,0,0,0]
		y_p = []
		for i in range(0,nrow):

			# q0 = np.dot(np.array(y[i,0]),np.array((np.array(self.y_pred[i,0]) + delta[0] )  / np.linalg.norm(np.array(self.y_pred[i,0]) + delta[0] )))
			# q1 = np.dot(np.array(y[i,1]),np.array((np.array(self.y_pred[i,1]) + delta[1] )  /  np.linalg.norm(np.array(self.y_pred[i,1]) + delta[1] )))
			# q2 = np.dot(np.array(y[i,2]),np.array((np.array(self.y_pred[i,2]) + delta[2] )  /  np.linalg.norm(np.array(self.y_pred[i,2]) + delta[2] )))
			# q3 = np.dot(np.array(y[i,3]),np.array((np.array(self.y_pred[i,3]) + delta[3] )  /  np.linalg.norm(np.array(self.y_pred[i,3]) + delta[3] )))

			# q = 1 - abs(np.add(q0 , np.add(q1, np.add(q1,q3))))
			# embed()
			# self.y_pred[i,0:4] = (self.y_pred[i,0:4] - delta )/(np.linalg.norm(np.array(self.y_pred[i,0:3]) + delta ))
			# embed()
			q = 1 - abs(T.dot(y[i,0:4],(self.y_pred[i,0:4] - delta )/((self.y_pred[i,0:4]+delta).norm(2))))

			# embed()
			tx = (y[i,4:]-self.y_pred[i,4:]).norm(2)
			# tx = np.array(y[i,4]-self.y_pred[i,4])
			# ty = np.array(y[i,5]-self.y_pred[i,5])
			# tz = np.array(y[i,6]-self.y_pred[i,6])
			# t0 = np.array([tx,ty,tz])

			NORMA +=(np.add(q*0.031,0.005*tx  ))
			ORI.append(q*0.031)
			TRA.append(0.005*tx )
			# embed()
			y_temp = (self.y_pred[i,0:4] - delta )/((self.y_pred).norm(2))
			y_t2 = (self.y_pred[i,4:])
		# 	# # embed()
			y_p.append(np.array([y_temp,y_t2]))
			# self.y_pred[i,0:4]=(self.y_pred[i,0:4] - delta )/(np.linalg.norm(np.array(([self.y_pred[i,0] + delta[0],self.y_pred[i,0] + delta[0],self.y_pred[i,0] + delta[0]]))))
		# h=T.matrix('h')
		# h = y_p
		# embed()
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
