
from __future__ import division

import numpy as np
import theano, random
import theano.tensor as T
from layers.conv_layer2 import ConvLayer2
from layers.fully_connected_layer import FullyConnectedLayer
from layers.output_layer import OutputLayer
from dataset import DataFeeder

feeder = DataFeeder()

def save_model(filename, **layer_dict):
	'''
	layer_dict is pairs of layer name to layer object. 
	each layer object is assumed to have W and b property of type SharedVariable, whose value will be saved
	'''
	np_dict = dict()
	for name, layer in layer_dict.iteritems():
		np_dict[name+'_W'] = layer.W.get_value()
		np_dict[name+'_b'] = layer.b.get_value()
	np.savez_compressed(filename, **np_dict)

rng = np.random.RandomState(12345)
batch_size = 200
nkerns = (50, 30)

print 'defining input'

index = T.lscalar()
X_occ = T.matrix('X_occ')
X_loc = T.matrix('X_loc')
y = T.ivector('y')
input_occ_batch = X_occ.reshape((batch_size, 1, 20, 20))
input_loc_batch = X_loc.reshape((batch_size, 4))

train_set_X_occ, train_set_X_loc, train_set_y = feeder.next_training_set_shared()

print 'defining architecture'

all_params = []

conv1_kernels = []
for i in xrange(nkerns[0]):
	kernel1_fc1 = FullyConnectedLayer(rng, input=input_loc_batch, n_in=4, n_out=20)
	kernel1_fc2 = FullyConnectedLayer(rng, input=kernel1_fc1.output, n_in=20, n_out=9)
	all_params += ( kernel1_fc1.params + kernel1_fc2.params )
	cur_kernel = kernel1_fc2.output.reshape((batch_size, 1, 1, 3, 3))
	conv1_kernels.append(cur_kernel)
conv1_kernel = T.concatenate(conv1_kernels, axis=1)

print conv1_kernel.shape.eval({X_loc:train_set_X_loc.eval()})

conv2_kernels = []
for i in xrange(nkerns[1]):
	kernel2_fc1 = FullyConnectedLayer(rng, input=input_loc_batch, n_in=4, n_out=1000)
	kernel2_fc2 = FullyConnectedLayer(rng, input=kernel2_fc1.output, n_in=1000, n_out=16*nkerns[0])
	all_params += ( kernel2_fc1.params + kernel2_fc2.params )
	conv2_kernels.append(kernel2_fc2.output.reshape((batch_size, 1, nkerns[0], 4, 4)))
conv2_kernel = T.concatenate(conv2_kernels, axis=1)

conv1 = ConvLayer2(rng, input=input_occ_batch, filter_shape=(nkerns[0], 1, 3, 3), 
	image_shape=(batch_size, 1, 20, 20), poolsize=(2,2), W=conv1_kernel, batch_size=batch_size)

print conv1.output.shape.eval({X_loc:train_set_X_loc.eval(), X_occ:train_set_X_occ.eval()})

conv2 = ConvLayer2(rng, input=conv1.output, filter_shape=(nkerns[1], nkerns[0], 4, 4), 
	image_shape=(batch_size, nkerns[0], 9, 9), poolsize=(2,2), W=conv2_kernel, batch_size=batch_size)#, W_shared=conv2_kernel)

fc_input = conv2.output.flatten(2)
fc1 = FullyConnectedLayer(rng, input=fc_input, n_in=3*3*nkerns[1], n_out=100)
fc2 = FullyConnectedLayer(rng, input=fc1.output, n_in=100, n_out=100)

output = OutputLayer(input=fc2.output, n_in=100, n_out=2)


print 'defining cost'

cost = output.negative_log_likelihood(y)

all_params += ( fc1.params + fc2.params + output.params )

all_grads = T.grad(cost, all_params)


print 'defining train model'

# Adam Optimizer Update
updates = []
adam_a=0.0002; adam_b1=0.1; adam_b2=0.001; adam_e=1e-8
adam_i = theano.shared(np.float32(0).astype(theano.config.floatX)) # iteration
adam_i_new = adam_i + 1 # iteration update
updates.append((adam_i, adam_i_new))
adam_const1 = 1. - (1. - adam_b1)**adam_i_new
adam_const2 = 1. - (1. - adam_b2)**adam_i_new
for p, g in zip(all_params, all_grads):
	adam_m = theano.shared(p.get_value() * 0.)
	adam_v = theano.shared(p.get_value() * 0.)
	adam_m_new = adam_b1 * g + ((1. - adam_b1) * adam_m)
	adam_v_new = (adam_b2 * T.sqr(g)) + ((1. - adam_b2) * adam_v)
	adam_p_new = p - adam_a * T.sqrt(adam_const2) / adam_const1* adam_m_new / (T.sqrt(adam_v_new) + adam_e)
	updates.append((adam_m, adam_m_new))
	updates.append((adam_v, adam_v_new))
	updates.append((p, adam_p_new))

train_model = theano.function( [index], cost, updates=updates, 
	givens={
		X_occ: train_set_X_occ[index * batch_size: (index + 1) * batch_size],
		X_loc: train_set_X_loc[index * batch_size: (index + 1) * batch_size],
		y: train_set_y[index * batch_size: (index + 1) * batch_size]
	}
)

print 'defining test model'
test_set_X_occ, test_set_X_loc, test_set_y = feeder.test_set_shared()

test_model = theano.function(
	[index], # index can be 0 to 24 because there are 5000 examples in the validation set
	output.errors(y),
	givens={
		X_occ: test_set_X_occ[index * batch_size: (index + 1) * batch_size],
		X_loc: test_set_X_loc[index * batch_size: (index + 1) * batch_size],
		y: test_set_y[index * batch_size: (index + 1) * batch_size]
	}
)

# learn
chunk_file_idx = 0
while True: # loop through chunk training file
	for i in xrange(500): # loop through mini-batch
		train_model(i)
		if i%50==0:
			print 'test error %f'%test_model(random.randint(0, 24))
	for i in xrange(500):
		train_model(i)
		if i%50==0:
			print 'test error %f'%test_model(random.randint(0, 24))
	X_occ, X_loc, y = feeder.next_training_set_raw()
	if X_occ is None:
		print 'done with all chunk files'
		break
	train_set_X_occ.set_value(X_occ, borrow=True)
	train_set_X_loc.set_value(X_loc, borrow=True)
	train_set_y.set_value(y, borrow=True)
	
	chunk_file_idx += 1

print 'done'
