
from __future__ import division

import numpy as np
import theano, random
import theano.tensor as T
from layers.conv_layer import ConvLayer
from layers.fully_connected_layer import FullyConnectedLayer
from layers.cont_output_layer import ContOutputLayer
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
nkerns = (50, 50, 50, 50)

print 'defining input'

index = T.lscalar()
X_occ = T.fmatrix('X_occ')
y = T.fvector('y')
y_flag = T.fvector('y_flag')
input_occ_batch = X_occ.reshape((batch_size, 1, 100, 100))

print 'defining architecture'

conv1 = ConvLayer(rng, input=input_occ_batch, filter_shape=(nkerns[0], 1, 7, 7), 
	image_shape=(batch_size, 1, 100, 100), poolsize=(2,2))
conv2 = ConvLayer(rng, input=conv1.output, filter_shape=(nkerns[1], nkerns[0], 6, 6), 
	image_shape=(batch_size, nkerns[2], 47, 47), poolsize=(2,2))
conv3 = ConvLayer(rng, input=conv2.output, filter_shape=(nkerns[2], nkerns[1], 4, 4), 
	image_shape=(batch_size, nkerns[3], 21, 21), poolsize=(2,2))

conv_out_flat = conv3.output.flatten(2)
fc1 = FullyConnectedLayer(rng, input=conv_out_flat, n_in=9*9*nkerns[1], n_out=1000)
fc2 = FullyConnectedLayer(rng, input=fc1.output, n_in=1000, n_out=100)

output = ContOutputLayer(input=fc2.output, n_in=100)


print 'defining cost'

cost = output.cost(y, y_flag)

all_params = conv1.params + conv2.params + conv3.params + fc1.params + fc2.params + output.params

all_grads = T.grad(cost, all_params)


print 'defining train model'
# learning_rate_sgd=0.1
# updates = [ (param_i, param_i - learning_rate_sgd * grad_i) for param_i, grad_i in zip(all_params, all_grads) ]

train_set_X_occ, _, train_set_y, train_set_y_flag = feeder.next_training_set_shared()


# Adam Optimizer Update
updates = []
one = np.float32(1)
zero = np.float32(0)
adam_a=np.float32(0.0001); adam_b1=np.float32(0.1); adam_b2=np.float32(0.001); adam_e=np.float32(1e-8)
adam_i = theano.shared(zero.astype(theano.config.floatX)) # iteration
adam_i_new = adam_i + one # iteration update
updates.append((adam_i, adam_i_new))
adam_const1 = one - (one - adam_b1)**adam_i_new
adam_const2 = one - (one - adam_b2)**adam_i_new
for p, g in zip(all_params, all_grads):
	adam_m = theano.shared(p.get_value() * zero)
	adam_v = theano.shared(p.get_value() * zero)
	adam_m_new = adam_b1 * g + ((one - adam_b1) * adam_m)
	adam_v_new = (adam_b2 * T.sqr(g)) + ((one - adam_b2) * adam_v)
	adam_p_new = p - adam_a * T.sqrt(adam_const2) / adam_const1* adam_m_new / (T.sqrt(adam_v_new) + adam_e)
	updates.append((adam_m, adam_m_new))
	updates.append((adam_v, adam_v_new))
	updates.append((p, adam_p_new))

# l = 0.00001
# updates = [(p, p-l*g) for (p,g) in zip(all_params, all_grads)]

train_model = theano.function( [index], cost, updates=updates, 
	givens={
		X_occ: train_set_X_occ[index * batch_size: (index + 1) * batch_size],
		y: train_set_y[index * batch_size: (index + 1) * batch_size], 
		y_flag: train_set_y_flag[index * batch_size: (index + 1) * batch_size]
	}
)

print 'defining test model'
test_set_X_occ, _, test_set_y, test_set_y_flag = feeder.test_set_shared()

test_model = theano.function(
	[index], # index can be 0 to 49 because there are 10000 examples in the validation set
	cost,
	givens={
		X_occ: test_set_X_occ[index * batch_size: (index + 1) * batch_size],
		y: test_set_y[index * batch_size: (index + 1) * batch_size], 
		y_flag: test_set_y_flag[index * batch_size: (index + 1) * batch_size]
	}
)

# learn
chunk_file_idx = 600
while True: # loop through chunk training files
	for i in xrange(50): # loop through mini-batch
		print train_model(i)
	for i in xrange(50):
		print train_model(i)
	print 'test error %f'%test_model(random.randint(0, 49))
	X_occ_data, _, y_data, y_flag_data = feeder.next_training_set_raw()
	if X_occ is None:
		print 'done with all chunk files'
		break
	train_set_X_occ.set_value(X_occ_data, borrow=True)
	train_set_y.set_value(y_data, borrow=True)
	train_set_y_flag.set_value(y_flag_data, borrow=True)
	
	chunk_file_idx += 1

	if chunk_file_idx%100==0:
		ckpt_name = 'ckpt_baseline/weights_%i_iter.npz'%chunk_file_idx
		save_model(ckpt_name, conv1=conv1, conv2=conv2, conv3=conv3, fc1=fc1, fc2=fc2, output=output)
		print 'saved successfully to %s'%ckpt_name

print 'done'
