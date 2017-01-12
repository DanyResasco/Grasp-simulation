
from __future__ import division

import numpy as np
import theano, random
import theano.tensor as T
from layers.conv_layer import ConvLayer
from layers.fully_connected_layer import FullyConnectedLayer
from layers.output_layer import OutputLayer
from dataset_stack import DataFeeder

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
x = T.matrix('x')
y = T.ivector('y')
input_batch = x.reshape((batch_size, 5, 20, 20))

print 'defining architecture'

if ckpt_file is None:
	conv1 = ConvLayer(rng, input=input_batch, filter_shape=(nkerns[0], 5, 3, 3), 
		image_shape=(batch_size, 5, 20, 20), poolsize=(2,2))
	conv2 = ConvLayer(rng, input=conv1.output, filter_shape=(nkerns[1], nkerns[0], 4, 4), 
		image_shape=(batch_size, nkerns[0], 9, 9), poolsize=(2,2))

	fc_input = conv2.output.flatten(2)
	fc1 = FullyConnectedLayer(rng, input=fc_input, n_in=3*3*nkerns[1], n_out=100)
	fc2 = FullyConnectedLayer(rng, input=fc1.output, n_in=100, n_out=100)
	
	output = OutputLayer(input=fc2.output, n_in=100, n_out=2)
else:
	weight_dict = np.load(ckpt_file)

	conv1 = ConvLayer(rng, input=input_batch, filter_shape=(nkerns[0], 5, 3, 3), 
		image_shape=(batch_size, 5, 20, 20), poolsize=(2,2), W=weight_dict['conv1_W'], b=weight_dict['conv1_b'])
	conv2 = ConvLayer(rng, input=conv1.output, filter_shape=(nkerns[1], nkerns[0], 4, 4), 
		image_shape=(batch_size, nkerns[0], 9, 9), poolsize=(2,2), W=weight_dict['conv2_W'], b=weight_dict['conv2_b'])

	fc_input = conv2.output.flatten(2)
	fc1 = FullyConnectedLayer(rng, input=fc_input, n_in=3*3*nkerns[1], n_out=100, W=weight_dict['fc1_W'], b=weight_dict['fc1_b'])
	fc2 = FullyConnectedLayer(rng, input=fc1.output, n_in=100, n_out=100, W=weight_dict['fc2_W'], b=weight_dict['fc2_b'])
	
	output = OutputLayer(input=fc2.output, n_in=100, n_out=2, W=weight_dict['output_W'], b=weight_dict['output_b'])


print 'defining cost'

cost = output.negative_log_likelihood(y)

all_params = conv1.params + conv2.params + fc1.params + fc2.params + output.params

all_grads = T.grad(cost, all_params)

print 'defining train model'
# learning_rate_sgd=0.1
# updates = [ (param_i, param_i - learning_rate_sgd * grad_i) for param_i, grad_i in zip(all_params, all_grads) ]

train_set_X, train_set_y = feeder.next_training_set_shared()

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
		x: train_set_X[index * batch_size: (index + 1) * batch_size],
		y: train_set_y[index * batch_size: (index + 1) * batch_size]
	}
)

print 'defining test model'
test_set_X, test_set_y = feeder.test_set_shared()

test_model = theano.function(
	[index], # index can be 0 to 24 because there are 5000 examples in the validation set
	output.errors(y),
	givens={
		x: test_set_X[index * batch_size: (index + 1) * batch_size],
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
	X, y = feeder.next_training_set_raw()
	if X is None:
		print 'done with all chunk files'
		break
	train_set_X.set_value(X, borrow=True)
	train_set_y.set_value(y, borrow=True)
	
	chunk_file_idx += 1
	if chunk_file_idx%100==0:
		ckpt_name = 'ckpt_stack/weights_%i_iter.npz'%(chunk_file_idx)
		save_model(ckpt_name, conv1=conv1, conv2=conv2, fc1=fc1, fc2=fc2, output=output)
		print 'saved successfully to %s'%ckpt_name

print 'done'
