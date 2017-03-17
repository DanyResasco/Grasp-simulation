

from __future__ import division

import numpy as np

import theano
import theano.tensor as T
from IPython import embed
from theano_utils import _tensor_py_operators
from cont_output_layer import ContOutputLayer
from fully_connected_layer import FullyConnectedLayer

def _dropsout(rng, layer, p):
    ''' n n of trial
    p (1-p) probability with which to retain the neurons 
    size is the shape of the output'''

    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(1000))
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer*T.cast(mask, theano.config.floatX) #put to zero some neurons
    return output / (1 - p)
    

class DropoutMLP(object):
    """Multi-Layer Perceptron Class with partial hidden units
    An implementation of Multilayer Perceptron with dropping of hidden units at a probability 
    given by ```1-dropout_rate```.
    """

    def __init__(self, rng, input, n_in_out, dropout_rates):
        """Initialize the parameters for the multilayer perceptron
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie
        :type n_hidden: int
        :param n_hidden: number of hidden units
        :type dropout_rate: list 
        :param dropout_rate: array containing probabilities of retaining a unit
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """
        
        #Dropping out the input layer
        inp_dropout_layer = _dropsout(rng, input, p=dropout_rates[0])
        
        self.drop_layer = FullyConnectedLayer(rng,
                    input=inp_dropout_layer,
                    n_in=n_in_out[0], n_out=n_in_out[1])

        self.drop_layer.output = _dropsout(rng, self.drop_layer.output, p=dropout_rates[0])

        self.drop_layer2 = FullyConnectedLayer(rng,
            input=self.drop_layer.output,
            n_in=n_in_out[1], n_out=n_in_out[2])

        self.drop_layer2.output = _dropsout(rng, self.drop_layer2.output, p=dropout_rates[0])


        self.drop_layer3 = FullyConnectedLayer(rng,
            input=self.drop_layer2.output,
            n_in=n_in_out[2], n_out=n_in_out[3])

        self.drop_layer3.output = _dropsout(rng, self.drop_layer3.output, p=dropout_rates[0])
        
        # embed()

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        # self.hiddenLayer = ContOutputLayer(
        #     input=input,
        #     n_in=n_in,
        #     n_out=n_hidden,

        #     W=self.drop_layer.W,

        #     b=self.drop_layer.b,
        # )
        
        
        self.drop_output_layer = ContOutputLayer(
        input=self.drop_layer3.output,
        n_in=n_in_out[3], 
        n_out=n_in_out[4] )
        

        #for test and validation
        fc1 = FullyConnectedLayer(rng, input=input, n_in=n_in_out[0], n_out=n_in_out[1],W=self.drop_layer.W,b=self.drop_layer.b)
		#output is a vector of 12 elements
        fc2 = FullyConnectedLayer(rng, input=fc1.output, n_in=n_in_out[1], n_out=n_in_out[2],W=self.drop_layer2.W,b=self.drop_layer2.b)
        fc3 = FullyConnectedLayer(rng, input=fc2.output, n_in=n_in_out[2], n_out=n_in_out[3],W=self.drop_layer3.W,b=self.drop_layer3.b)
        self.cont_OutputLayer = ContOutputLayer(input=fc3.output, n_in =n_in_out[3] ,n_out=n_in_out[4],W=self.drop_output_layer.W,b=self.drop_output_layer.b)
        # # The logistic regression layer gets as input the hidden units
        # # of the hidden layer
        # self.logRegressionLayer = ContOutputLayer(
        #     input=self.hiddenLayer.output,
        #     n_in=n_hidden,
        #     n_out=n_out,
        #     W=self.drop_output_layer.W,

        #     b=self.drop_output_layer.b,
        # )
        
        
        # self.drop_negative_log_likelihood = self.drop_output_layer.cost_quaternion
        self.dany_error_drop = self.drop_output_layer.cost_quaternion

        self.dany_error = self.cont_OutputLayer.cost_quaternion

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        # self.negative_log_likelihood = (
        #     self.logRegressionLayer.negative_log_likelihood
        # )
        # same holds for the function computing the number of errors
        # self.errors = self.ContOutputLayer.cost_quaternion

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.drop_layer.params +self.drop_layer2.params +self.drop_layer3.params + self.drop_output_layer.params

        # self.params = self.drop_layer.params + self.cont_OutputLayer.params

        self.input = input