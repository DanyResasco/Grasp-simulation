
from __future__ import division
import timeit

import numpy as np
import theano, random
import theano.tensor as T
from conv2d_Dany import Conv2D
from fully_connected_layer import FullyConnectedLayer
from cont_output_layer import ContOutputLayer
from IPython import embed
from dataset_Dany import Input_output 


def Draw_Grasph(truth,prediction,eo,et):
    print "disegno"
    # import matplotlib.pypl
    # embed()
    rot_y = []
    tra_y = []
    rot_yp = []
    tra_yp= []
    # for j in range(0,len(truth)):
    #   rot_y.append([truth[j][0],truth[j][1],truth[j][2]])
    #   tra_y.append([truth[j][3],truth[j][4],truth[j][5]])
    #   rot_yp.append([prediction[j][0],prediction[j][1],prediction[j][2]])
    #   tra_yp.append([prediction[j][3],prediction[j][4],prediction[j][5]])
    # print 'truth',truth
    # print 'prediction',prediction
    # print 'len(truth)',len(truth)
    # print 'len(prediction)',len(prediction)
    embed()









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
rng = np.random.RandomState(65432)
batch_size = 5
# data_chunk_size = 50 #Non so ancora cosa sia minibatch?
nkerns = (20, 20, 20, 10, 10, 10) # n felter for each layer

print 'defining input'

index = T.lscalar('index')
# X_occ = T.fmatrix('X_occ')
X_occ = T.tensor3('X_occ') 
y = T.fmatrix('y')
# 3 e' il numero di canali, 1 gray scale 3 rgb, 127 dimensione voxel controlla
input_occ_batch = X_occ.reshape((batch_size, 1, 256, 256)) #64 voxel dimension

print 'defining architecture'

''' image_shape is (num_imgs, num_channels, img_height, img_width, img_length)
	filter_shape is (num_kernels, num_channels, kernel_height, kernel_width, kernel_length)
	size(image_shape[num_channels]) == size(filter_shape[num_channels]) '''
#(64-7+1)/poolsize
conv1 = Conv2D(rng, input=input_occ_batch, filter_shape=(nkerns[0], 1, 7, 7), 
	image_shape=(batch_size, 1, 256, 256), poolsize=(1,1))
conv2 = Conv2D(rng, input=conv1.output, filter_shape=(nkerns[1], nkerns[0], 7, 7), 
 	image_shape=(batch_size, nkerns[0], 250,250), poolsize=(1,1))
conv3 = Conv2D(rng, input=conv2.output, filter_shape=(nkerns[2], nkerns[1], 5, 5), 
 	image_shape=(batch_size, nkerns[1], 244, 244), poolsize=(2,2))


conv4 = Conv2D(rng, input=conv3.output, filter_shape=(nkerns[3], nkerns[2], 5, 5), 
	image_shape=(batch_size, nkerns[2], 120, 120), poolsize=(2,2))


conv5 = Conv2D(rng, input=conv4.output, filter_shape=(nkerns[4], nkerns[3], 5, 5), 
	image_shape=(batch_size, nkerns[3], 58, 58), poolsize=(1,1))

# conv6 = Conv2D(rng, input=conv5.output, filter_shape=(nkerns[5], nkerns[4], 3, 3), 
# 	image_shape=(batch_size, nkerns[4], 8, 8), poolsize=(1,1))


fc_input = conv5.output.flatten(2)
fc1 = FullyConnectedLayer(rng, input=fc_input, n_in=54*54*nkerns[4], n_out=550)
#output is a vector of 12 elements
fc2 = FullyConnectedLayer(rng, input=fc1.output, n_in=550, n_out=250)
fc3 = FullyConnectedLayer(rng, input=fc2.output, n_in=250, n_out=150)
output = ContOutputLayer(input=fc3.output, n_in =150 ,n_out=6)

print 'defining cost'

cost = output.dany_error(y,batch_size)

all_params = (conv1.params + conv2.params + conv3.params + conv4.params + conv5.params + fc1.params + fc2.params +fc3.params+ output.params)

  # compute the gradient of cost
# embed()
all_grads = T.grad(cost[0], all_params)
print "grad"

print 'defining train model'

# train_set_X_occ, _, train_set_y = feeder.next_training_set_shared() #TO change
Dataset_dany =  Input_output()
train_set_X_occ, train_set_y = Dataset_dany[0]
valid_set_x, valid_set_y = Dataset_dany[1]
test_set_X_occ, test_set_y = Dataset_dany[2]

# embed()

#


print train_set_y.type
print y.type
print train_set_X_occ.type
print X_occ.type
print 'Adam Optimizer Update'
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
		X_occ: test_set_X_occ[index * batch_size: (index + 1) * batch_size], #Scorre la matrice? 
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
patience = 10000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                       # found
improvement_threshold = 0.995  # a relative improvement of this much is
                               # considered significant
validation_frequency = min(n_train_batches, patience // 2)
                              # go through this many
                              # minibatche before checking the network
                              # on the validation set; in this case we
                              # check every epoch

best_validation_loss = np.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()

epoch = 0
done_looping = False
n_epochs =1000
print 'prima del while'
Truth = []
pred = []
E_ori = []
E_tra = []
# count_dany = 0
while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    # for minibatch_index in range(n_train_batches): #loop on train examples
    #     # print minibatch_index
    #     train_model(minibatch_index)
    #     # iteration number
    for minibatch_index in range(n_train_batches): #loop on train examples
        train_model(minibatch_index)
        # print op
        iter = (epoch - 1) * n_train_batches + minibatch_index
        if (iter + 1) % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_losses = [validate_model(i) for i in range(n_valid_batches)]
            
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

                test_m = [test_losses[i][0] for i
                                 in range(0,len(test_losses))]

                E_ori = [test_losses[i][3] for i
                                 in range(0,len(test_losses))]

                E_tra = [test_losses[i][4] for i
                                 in range(0,len(test_losses))]

                test_score = np.mean(test_m)

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
            res_name = '2d6Cnn3fcl_10_3_20_3_10.npz'
            # save_model(res_name, conv1=conv1, conv2=conv2, conv3=conv3,conv4=conv4,
            # conv5=conv5,conv6=conv6, fc1=fc1, fc2=fc2,fc3=fc3, output=output)
            save_model(res_name, conv1=conv1, conv2=conv2, conv3=conv3,conv4=conv4,
            conv5=conv5, fc1=fc1, fc2=fc2,fc3=fc3, output=output)
            break

end_time = timeit.default_timer()
print(('Optimization complete. Best validation score of %f '
	'obtained at iteration %i, with test performance %f ') %
	(best_validation_loss , best_iter + 1, test_score ))
# for i in Truth.value
embed()
Draw_Grasph(Truth,pred,E_ori,E_tra)