import sys
import time
import random

import numpy as np
import scipy.io
import theano

from pylearn2.datasets import DenseDesignMatrix
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.models import mlp, maxout
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.models.mlp import ConvRectifiedLinear
from pylearn2.space import VectorSpace, CompositeSpace, Conv2DSpace


class sknn():

    """
    SK-learn like interface for pylearn2
    Notice how training the model and the training algorithm are now part of the same class, which I actually quite like
    This class is focused a bit on online learning, so you might need to modify it to include other pylearn2 options if
    you have access all your data upfront
    """

    def __init__(self, layers, dropout=False, input_scaler = None, output_scaler = None,   learning_rate=0.01, verbose=0):
        """

        :param layers: List of tuples of types of layers alongside the number of neurons
        :param learning_rate: The learning rate for all layers
        :param verbose: Verbosity level
        :return:
        """
        self.layers = layers
        self.mlp = None
        self.ds = None
        self.f = None
        self.verbose = verbose
        cost = Dropout() if dropout else None
        self.trainer = sgd.SGD(
            learning_rate=learning_rate,
            learning_rule=learning_rule.RMSProp(),
            cost=cost, batch_size=1000)

        self.input_normaliser = input_scaler
        self.output_normaliser = output_scaler
        self.conv_input = False


    def __scale(self,X,y):
        if self.input_normaliser is not None:
            X_s = self.input_normaliser.fit(X).transform(X)
        else:
            X_s = X
        if self.output_normaliser is not None and y is not None:
            y_s = self.output_normaliser.fit(y).transform(y)
        else:
            y_s = y

        return X_s, y_s

    def __original_y(self,y):
        if(self.output_normaliser is None):
            return y
        else:
            return self.output_normaliser.inverse_transform(y)

    def linit(self, X, y):
        if(self.verbose > 0):
            print "Lazy initialisation"

        layers = self.layers
        pylearn2mlp_layers = []
        self.units_per_layer = []
        #input layer units
        self.units_per_layer+=[X.shape[1]]

        for layer in layers[:-1]:
            self.units_per_layer+=[layer[1]]

        #Output layer units
        self.units_per_layer+=[y.shape[1]]

        if(self.verbose > 0):
            print "Units per layer", str(self.units_per_layer)


        for i, layer in enumerate(layers[:-1]):

            try:
                fan_in = self.units_per_layer[i] + 1
                fan_out = self.units_per_layer[i+1]
                lim = np.sqrt(6) / (np.sqrt(fan_in + fan_out))
            except:
                # TODO: Better way to compute this for conv layers?
                lim = 0.05

            layer_name = "Hidden_%i_%s"%(i,layer[0])
            activate_type = layer[0]
            if activate_type == "RectifiedLinear":
                hidden_layer = mlp.RectifiedLinear(
                    dim=layer[1],
                    layer_name=layer_name,
                    irange=lim)
            elif activate_type == "Sigmoid":
                hidden_layer = mlp.Sigmoid(
                    dim=layer[1],
                    layer_name=layer_name,
                    irange=lim)
            elif activate_type == "Tanh":
                hidden_layer = mlp.Tanh(
                    dim=layer[1],
                    layer_name=layer_name,
                    irange=lim)
            elif activate_type == "Maxout":
                hidden_layer = maxout.Maxout(
                    num_units=layer[1],
                    num_pieces=layer[2],
                    layer_name=layer_name,
                    irange=lim)
            elif activate_type == "ConvRectifiedLinear":
               hidden_layer = ConvRectifiedLinear(
                        layer_name=layer_name,
                        kernel_shape=layer[1]['kernel'],
                        pool_shape=[1, 1],
                        pool_stride= [1, 1],
                        output_channels=layer[1]['channels'],
                        irange=lim)
               self.conv_input = True
            else:
                raise NotImplementedError(
                    "Layer of type %s are not implemented yet" %
                    layer[0])
            pylearn2mlp_layers += [hidden_layer]

        output_layer_info = layers[-1]
        output_layer_name = "Output_%s"%output_layer_info[0]

        try:
            fan_in = self.units_per_layer[-2] + 1
            fan_out = self.units_per_layer[-1]
            lim = np.sqrt(6) / (np.sqrt(fan_in + fan_out))
        except:
            # TODO: Better way to compute this for post-conv layers?
            lim = 0.05

        if(output_layer_info[0] == "Linear"):
            output_layer = mlp.Linear(
                dim=self.units_per_layer[-1],
                layer_name=output_layer_name,
                irange=lim)
            pylearn2mlp_layers += [output_layer]

        if self.conv_input:
            nvis = None
            input_space = Conv2DSpace(shape=X.shape[1:], num_channels=1)
            self.view = input_space.get_origin_batch(100)
            self.ds = DenseDesignMatrix(topo_view=self.view, y=y)
        else:
            nvis = self.units_per_layer[0]
            input_space = None
            self.ds = DenseDesignMatrix(X=X, y=y)

        if self.mlp is None:
            self.mlp = mlp.MLP(pylearn2mlp_layers, input_space=input_space, nvis=nvis)
        self.trainer.setup(self.mlp, self.ds)
        inputs = self.mlp.get_input_space().make_theano_batch()
        self.f = theano.function([inputs], self.mlp.fprop(inputs))

    def fit(self, X, y, epochs=100):
        """
        :param X: Training data
        :param y:
        :return:
        """
        if(self.ds is None):
            self.linit(X, y)
        if(self.conv_input):
            X = np.array([X]).transpose(1,2,3,0)
        ds = self.ds
        X_s,y_s = self.__scale(X,y)
        if self.conv_input:
            ds.X = ds.view_converter.topo_view_to_design_mat(X_s)
        else:
            ds.X = X_s
        ds.y = y_s

        for e in range(epochs):
            self.trainer.train(dataset=ds)
        return self

    def predict(self, X):
        """

        :param X:
        :return:
        """
        if(self.conv_input):
            X = np.array([X]).transpose(1,2,3,0)
        X_s,_ = self.__scale(X, None)
        y =  self.f(X_s)
        y_s = self.__original_y(y)
        return y_s


class IncrementalMinMaxScaler():
    def __init__(self, feature_range=(-1.0,1.0)):
        self.feature_range = feature_range
        self.changed = False
        self.init = False
        self.times = 0


    def fit(self,X, y = None):
        #print "fitting"
        self.changed = False
        self.times+=1;
        #print "fitting"
        if (not self.init):
            self.min_ = np.array(X[0],dtype = np.float64)
            self.max_ = np.array(X[0],dtype = np.float64)
            self.data_min = self.min_
            self.data_max = self.max_
            self.init = True
        else:
            #print "appednign"
            X = np.array(X,ndmin=2)
            X = np.append(X, [self.data_min], axis = 0)
            X = np.append(X, [self.data_max], axis = 0)

        feature_range = self.feature_range
        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)

        if not (self.data_min == data_min).all():
            # print "min changed" # , data_min - self.data_min
            print 'X',
            self.changed = True

        if not (self.data_max == data_max).all():
            # print "max changed" # , data_max - self.data_max
            print 'X',
            self.changed = True

        self.data_min = data_min
        self.data_max = data_max

        data_range = data_max - data_min
        # Do not scale constant features
        #print data_range
        #exit()
        data_range[data_range == 0.0] = 1.0
        data_range[data_range == 0] = 1.0
        #print "data range", data_range.dtype, data_max.dtype, data_min.dtype
        self.scale_ = (feature_range[1] - feature_range[0]) / data_range
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_range = data_range
        self.data_min = data_min
        return self


    def transform(self, X):
        #X *= self.scale_#X += self.min_
        assert (len(X.shape) == 2), X
        transformed =  (X * self.scale_) + self.min_
        #transformed[:, -1] = 1.0
        return transformed


    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            Input data that will be transformed.
        """
        reverse_transformed =  (X - self.min_)/self.scale_
        return reverse_transformed
