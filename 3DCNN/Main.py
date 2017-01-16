
from __future__ import division
import timeit

import numpy as np
import theano, random
import theano.tensor as T
from conv3d_Dany import Conv3D
from fully_connected_layer import FullyConnectedLayer
from cont_output_layer import ContOutputLayer
from IPython import embed
from dataset_Dany import Input_output 
import matplotlib.pyplot as plt
# from test_dataset_split_dany import DanyDataset

def Draw_Grasph(truth,prediction):
    print "disegno"
    import matplotlib.pyplot as plt
    plt.figure(1)
#     print truth
#     print 'prediction', prediction
    # plt.plot(prediction,marker='x', color='r', label='Prediction values')
    # plt.plot(truth)
    for j in range(0,len(truth)):
      for i in range(0,2):
        plt.plot(truth[j][i], label='Truth values')
        plt.scatter(prediction[j][i],marker='x', color='r', label='Prediction values')
        plt.xlabel('True orientation', fontsize=9)
        plt.ylabel('Prediction orientatio', fontsize=9)
      # plt.figure(figsize=(8,6))
      plt.figure(2)
      for i in range(3,5):
        plt.plot(truth[j][i], label='Truth values')
        plt.scatter(prediction[j][i],marker='x', color='r', label='Prediction values')
        plt.xlabel('true translation', fontsize=9)
        plt.ylabel('Cost y', fontsize=9)
      # plt.figure(figsize=(8,6))
  # for i in truth.values():
    # plt.plot(truth[i][2], label='Truth values')
    # plt.scatter(prediction[i][2],marker='x', color='r', label='Prediction values')
    # plt.xlabel('Poses z', fontsize=9)
    # plt.ylabel('Cost z', fontsize=9)

# # for i in truth.values():
#   plt.plot(truth[i][3], label='Truth values')
#   plt.scatter(prediction[i][3],marker='x', color='r', label='Prediction values')
#   plt.xlabel('Translation x', fontsize=9)
#   plt.ylabel('Cost x', fontsize=9)
#   plt.figure(figsize=(8,6))
#   # for i in truth.values():
#   plt.plot(truth[i][4], label='Truth values')
#   plt.scatter(prediction[i][4],marker='x', color='r', label='Prediction values')
#   plt.xlabel('Translation y', fontsize=9)
#   plt.ylabel('Cost y', fontsize=9)
#   plt.figure(figsize=(8,6))
#   # for i in truth.values():
#   plt.plot(truth[i][5], label='Truth values')
#   plt.scatter(prediction[i][5],marker='x', color='r', label='Prediction values')
#   plt.xlabel('Translation z', fontsize=9)
#   plt.ylabel('Cost z', fontsize=9)

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

'''param rng: a random number generator used to initialize weights'''
# DanyDataset = DanyDataset()
np.random.seed(0)
rng = np.random.RandomState(65432)
batch_size = 5
# data_chunk_size = 50 #Non so ancora cosa sia minibatch?
nkerns = (10,10,10,10,10,10) # n filter for each layer

print 'defining input'

index = T.lscalar('index')
# X_occ = T.fmatrix('X_occ')
X_occ = T.tensor4('X_occ') 
y = T.fvector('y')

# 3 e' il numero di canali, 1 gray scale 3 rgb, 64 dimensione voxel 
input_occ_batch = X_occ.reshape((batch_size, 1, 64, 64, 64)) #64 voxel dimension

print 'defining architecture'

''' image_shape is (num_imgs, num_channels, img_height, img_width, img_length)
	filter_shape is (num_kernels, num_channels, kernel_height, kernel_width, kernel_length)
	size(image_shape[num_channels]) == size(filter_shape[num_channels]) '''
#(64-7+1)/poolsize
# conv1 = Conv3D(rng, input=input_occ_batch, filter_shape=(nkerns[0], 1, 7, 7, 7), 
# 	image_shape=(batch_size, 1, 64, 64,64), poolsize=(1,1,1))
# conv2 = Conv3D(rng, input=conv1.output, filter_shape=(nkerns[1], nkerns[0], 3, 3, 3), 
#  	image_shape=(batch_size, nkerns[0], 58,58,58), poolsize=(2,2,2))
# conv3 = Conv3D(rng, input=conv2.output, filter_shape=(nkerns[2], nkerns[1], 2, 2, 2), 
#  	image_shape=(batch_size, nkerns[1], 28, 28, 28), poolsize=(1,1,1))
# conv4 = Conv3D(rng, input=conv3.output, filter_shape=(nkerns[3], nkerns[2], 3, 3, 3), 
# 	image_shape=(batch_size, nkerns[2], 27, 27, 27), poolsize=(2,2,2))
# conv5 = Conv3D(rng, input=conv4.output, filter_shape=(nkerns[4], nkerns[3], 5, 5, 5), 
# 	image_shape=(batch_size, nkerns[3], 12, 12, 12), poolsize=(1,1,1))
# conv6 = Conv3D(rng, input=conv5.output, filter_shape=(nkerns[5], nkerns[4], 3, 3, 3), 
# 	image_shape=(batch_size, nkerns[4], 8, 8, 8), poolsize=(1,1,1))


# fc_input = conv6.output.flatten(2)
# fc1 = FullyConnectedLayer(rng, input=fc_input, n_in=6*6*6*nkerns[5], n_out=5500)
# # #output is a vector of 6 elements
# fc2 = FullyConnectedLayer(rng, input=fc1.output, n_in=5500, n_out=3000)
# fc3 = FullyConnectedLayer(rng, input=fc2.output, n_in=3000, n_out=6)
# output = ContOutputLayer(input=fc3.output, n_in=6)

conv1 = Conv3D(rng, input=input_occ_batch, filter_shape=(nkerns[0], 1, 3, 3, 3), 
  image_shape=(batch_size, 1, 64, 64,64), poolsize=(1,1,1))
conv2 = Conv3D(rng, input=conv1.output, filter_shape=(nkerns[1], nkerns[0], 3, 3, 3), 
  image_shape=(batch_size, nkerns[0], 62,62,62), poolsize=(2,2,2))
conv3 = Conv3D(rng, input=conv2.output, filter_shape=(nkerns[2], nkerns[1], 3, 3, 3), 
  image_shape=(batch_size, nkerns[1], 30, 30, 30), poolsize=(1,1,1))
conv4 = Conv3D(rng, input=conv3.output, filter_shape=(nkerns[3], nkerns[2], 3, 3, 3), 
  image_shape=(batch_size, nkerns[2], 28, 28, 28), poolsize=(2,2,2))
conv5 = Conv3D(rng, input=conv4.output, filter_shape=(nkerns[4], nkerns[3], 2, 2, 2), 
  image_shape=(batch_size, nkerns[3], 13, 13, 13), poolsize=(1,1,1))
conv6 = Conv3D(rng, input=conv5.output, filter_shape=(nkerns[5], nkerns[4], 2, 2, 2), 
  image_shape=(batch_size, nkerns[4], 12, 12, 12), poolsize=(1,1,1))
conv7 = Conv3D(rng, input=conv6.output, filter_shape=(nkerns[5], nkerns[4], 2, 2, 2), 
  image_shape=(batch_size, nkerns[4], 11, 11, 11), poolsize=(2,2,2))

fc_input = conv7.output.flatten(2)
fc1 = FullyConnectedLayer(rng, input=fc_input, n_in=5*5*5*nkerns[5], n_out=500)
# output is a vector of 6 elements
fc2 = FullyConnectedLayer(rng, input=fc1.output, n_in=500, n_out=250)
fc3 = FullyConnectedLayer(rng, input=fc2.output, n_in=250, n_out=50)
fc4 = FullyConnectedLayer(rng, input=fc3.output, n_in=50, n_out=6)


output = ContOutputLayer(input=fc4.output, n_in=6)

print 'defining cost'

# cost = output.cost(y, y_flag=None)
cost = output.mse(y, y_flag=None)


# all_params = (conv1.params + conv2.params + conv3.params + conv4.params + conv5.params +
# 	conv6.params +  fc1.params + fc2.params +fc3.params+ output.params)

all_params = (conv1.params + conv2.params + conv3.params + conv4.params + conv5.params +
  conv6.params + conv7.params +  fc1.params + fc2.params +fc3.params+fc4.params+ output.params)

# compute the gradient of cost
# embed()
all_grads = T.grad(cost[0], all_params)
print "grad"

Dataset_dany =  Input_output()
train_set_X_occ, train_set_y = Dataset_dany[0]
valid_set_x, valid_set_y = Dataset_dany[1]
test_set_X_occ, test_set_y  = Dataset_dany[2]

# print'train_set_y.type' ,train_set_y.type
# print' y.type', y.type
# print'valid_set_y',valid_set_y.type
# print 'test_set_y',test_set_y.type
# print'train_set_X_occ.type' ,train_set_X_occ.type
# print 'X_occ.type',X_occ.type
# print 'test_set_X_occ',test_set_X_occ.type
# print'valid_set_x',valid_set_x.type

print 'Adam Optimizer Update'
# Adam Optimizer Update
updates = []
one = np.float32(1)
zero = np.float32(0)
adam_a=np.float32(0.0001); adam_b1=np.float32(0.1); adam_b2=np.float32(0.001); adam_e=np.float32(1e-8)
# adam_a=np.float32(0.001); adam_b1=np.float32(0.9); adam_b2=np.float32(0.99); adam_e=np.float32(1e-8) #from article

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

''' compiling a Theano function `train_model` that returns the cost, but
    in the same time updates the parameter of the model based on the rules
    defined in `updates '''
print 'defining train model'
train_model = theano.function( [index], cost, updates=updates,
	givens={
		X_occ: train_set_X_occ[(index * batch_size): ((index + 1) * batch_size)],
		y: train_set_y[index * batch_size: (index + 1) * batch_size]
	}
)

print 'defining test model'
test_model = theano.function(	[index],	cost,
	givens={
		X_occ: test_set_X_occ[index * batch_size: (index + 1) * batch_size], 
		y: test_set_y[index * batch_size: (index + 1) * batch_size]
	}
)


print 'defining valid model'
validate_model = theano.function(
	inputs=[index],
	outputs= cost,
	givens={
		X_occ: valid_set_x[index * batch_size:(index + 1) * batch_size],
		y: valid_set_y[index * batch_size:(index + 1) * batch_size]
	}
)


print 'minibatches'
if theano.config.mode != "FAST_COMPILE":
    mode = "FAST_COMPILE"

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_X_occ.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_set_X_occ.get_value(borrow=True).shape[0] // batch_size

# early-stopping parameters
patience = 5000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                       # found
improvement_threshold = 0.995  # a relative improvement of this much is
                               # considered significant
validation_frequency = min(n_train_batches, patience // 2)
                              # go through this many
                              # minibatche before checking the network
                              # on the validation set;

best_validation_loss = np.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()

epoch = 0
done_looping = False
n_epochs =1000
print 'prima del while'
# test_error = []

# chunk_file_idx = 0
# while True: # loop through chunk training files
#   for i in xrange(int(n_train_batches)): # loop through mini-batch
#      train_model(i)
#   for i in xrange(int(n_train_batches)):
#      train_model(i)
#   if chunk_file_idx%10==0:
#     # test_error.append(test_model(random.randint(0, n_test_batches)))
#     test_losses = [test_model(i)
#                    for i in range(n_test_batches)]
#     # for i in range(n_test_batches):
#     #       test_losses = test_model(i)
#     test_score = np.mean(test_losses)
#     # print 'test error %f',test_error
#   # X_occ_data, _, y_data = feeder.next_training_set_raw()
#   if X_occ is None:
#     print 'done with all chunk files'
#     break
#   # train_set_X_occ.set_value(X_occ_data, borrow=True)
#   # train_set_y.set_value(y_data, borrow=True)
  
#   chunk_file_idx += 1

#   if chunk_file_idx%100==0:
#     ckpt_name = 'weights_%i_iter.npz'%chunk_file_idx
#     save_model(ckpt_name, conv1=conv1, conv2=conv2, conv3=conv3,conv4=conv4,
#             conv5=conv5,conv6=conv6, fc1=fc1, fc2=fc2,fc3=fc3, output=output)
#     print 'saved successfully to %s'%ckpt_name
#     print test_score
#     break

# print 'done'


Truth = []
pred = []
# count_dany = 0
while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    # for minibatch_index in range(n_train_batches): #loop on train examples
    #     # print minibatch_index
    #     train_model(minibatch_index)
    #     # iteration number
    for minibatch_index in range(n_train_batches): #loop on train examples
        train_model(minibatch_index)
        iter = (epoch - 1) * n_train_batches + minibatch_index
        if (iter + 1) % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_losses = [validate_model(i) for i
                                 in range(n_valid_batches)]
            
            validation_m = [validation_losses[i][0] for i
                                 in range(0,len(validation_losses))]


            this_validation_loss = np.mean(validation_m)

            print("Epoch %i, Minibatch %i/%i, Validation Error %f " 
                    % (epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss 
                      )
                  )

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                print 'validation'
                #improve patience if loss improvement is good enough
                if (this_validation_loss < best_validation_loss * improvement_threshold):
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                print 'test'
                test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                                   
                Truth = [test_losses[i][1] for i
                                 in range(0,len(test_losses))]
                pred = [test_losses[i][2] for i
                                 in range(0,len(test_losses))]
                # for i in range(n_test_batches):
                #       test_losses = test_model(i)
                test_m = [test_losses[i][0] for i
                                 in range(0,len(test_losses))]

                test_score = np.mean(test_m)

                #       Truth.append(list(test_losses[1]))
                #       pred.append(list(test_losses[2]))

                # count_dany +=1
                print(("Epoch %i, Minibatch %i/%i, Test error of"" best model %f ") 
                      % (   epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score 
                        )
                    )

        if patience <= iter:
            print 'save'
            done_looping = True
            res_name = '10X10FILTER/3d7Cnn4fcl_RELU_norma.npz'
            # save_model(res_name, conv1=conv1, conv2=conv2, conv3=conv3,conv4=conv4,
            # conv5=conv5,conv6=conv6, fc1=fc1, fc2=fc2,fc3=fc3, output=output)
            save_model(res_name, conv1=conv1, conv2=conv2, conv3=conv3,conv4=conv4,
            conv5=conv5,conv6=conv6,conv7=conv7, fc1=fc1, fc2=fc2,fc3=fc3,fc4=fc4, output=output)
            break

end_time = timeit.default_timer()
print(('Optimization complete. Best validation score of %f '
	'obtained at iteration %i, with test performance %f ') %
	(best_validation_loss , best_iter + 1, test_score ))
# for i in Truth.value
Draw_Grasph(Truth,pred)
  # (best_validation_loss , best_iter + 1, test_score ))



