import tensorflow as tf
from tensorflow.contrib import slim

from cleverhans.model import Model

from sp_func_nn import tf_sp_frontend, tf_sp_frontend_binary

class FourLayerModel(Model):
    """
    4 layer CNN with defense
    """
    def __init__(self, def_params):
        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'logits']
        self.nb_classes = 10
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        self.def_params = def_params

    def fprop(self, x):
        x = tf_sp_frontend(x, **self.def_params)
        x = tf.reshape(x, [-1, self.image_size, self.image_size, self.num_channels])
        _, end_points = four_layer_cnn(x, phase=False, reuse=tf.AUTO_REUSE)
        return end_points

    def predict(self, x):
        return self.fprop(x)['logits']


class FourLayerModelPlain(Model):
    """
    4 layer CNN without defense
    """
    def __init__(self):
        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'logits']
        self.nb_classes = 10
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

    def fprop(self, x):
        _, end_points = four_layer_cnn(x, phase=False, reuse=tf.AUTO_REUSE)
        return end_points

    def predict(self, x):
        return self.fprop(x)['logits']


class TwoLayerModel(Model):
    """
    2 layer NN with defense
    """
    def __init__(self, def_params):
        self.layer_names = ['layer1', 'logits']
        self.def_params = def_params
        self.dim = 784

    def fprop(self, x):
        x = tf_sp_frontend_binary(x, **self.def_params)
        x = tf.reshape(x, [-1, self.dim])
        _, end_points = two_layer_nn(x, phase=False, reuse=tf.AUTO_REUSE)
        return end_points


class TwoLayerModelPlain(Model):
    """
    2 layer NN without defense
    """
    def __init__(self):
        self.layer_names = ['layer1', 'logits']

    def fprop(self, x):
        _, end_points = two_layer_nn(x, phase=False, reuse=tf.AUTO_REUSE)
        return end_points


def four_layer_cnn(inputs, phase, reuse=tf.AUTO_REUSE, scope='four_layer_cnn'):
    """
    Four layer CNN from http://neuralnetworksanddeeplearning.com/chap6.html/. 
    TensorFlow-Slim implementation adapted from https://github.com/initialized/tensorflow-tutorial/.

    Layer 1: Convolutional, with a 5x5 receptive field and 20 feature maps. 
    Layer 2: Convolutional, with a 5x5 receptive field and 40 feature maps.
    Layer 3: Fully connected, with 1000 neurons.
    Layer 4: Fully connected, with 1000 neurons.

    inputs - Should be of shape [num_samples, 28, 28, 1].
    phase  - If TRUE, dropout is switched on in layers 3 and 4.
    """
    end_points = {}
    with tf.variable_scope(scope, reuse=reuse):
        net = slim.conv2d(inputs, 20, [5, 5], padding='SAME', scope='layer1-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer1-max-pool')
        end_points['layer1'] = net

        net = slim.conv2d(net, 40, [5, 5], padding='VALID', scope='layer2-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')
        net = tf.reshape(net, [-1, 5*5*40])
        end_points['layer2'] = net

        net = slim.fully_connected(net, 1000, scope='layer3')
        net = slim.dropout(net, is_training=phase, scope='layer3-dropout')
        end_points['layer3'] = net

        net = slim.fully_connected(net, 1000, scope='layer4')
        net = slim.dropout(net, is_training=phase, scope='layer4-dropout')
        end_points['layer4'] = net

        net = slim.fully_connected(net, 10, scope='logits', activation_fn=None)
        logits = slim.dropout(net, is_training=phase, scope='logits-dropout')
        end_points['logits'] = logits

        return logits, end_points

def two_layer_nn(inputs, phase, reuse=None, scope='two_layer_nn'):
    """
    Fully connected NN with 1 hidden layer (10 neurons) for binary classif.
    """
    end_points = {}
    with tf.variable_scope(scope, reuse=reuse):
        # Set the default weight _regularizer and activation for each fully_connected layer.
        with slim.arg_scope([slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(0.001)):
            # First layer: 10x1
            net = slim.fully_connected(inputs, 10, scope='layer1')
            end_points['layer1'] = net

            # Output layer: 1x1
            logits = slim.fully_connected(net, 1, scope='logits', activation_fn=None)
            end_points['logits'] = logits

            return logits, end_points