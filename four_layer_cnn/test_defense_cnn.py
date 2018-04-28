"""
Tests the efficacy of sparsity-based defenses against adversarial attacks on a four layer CNN.

Classifier: Four layer CNN, used for classification of the MNIST dataset.
Defense: Sparsifying front end with rho = 3% (sparsity level).
Adversarial attacks: FGSM and locally linear attacks (semi-white box and white box) with epsilon = 0.25 (L-infinity attack budget).
"""

import numpy as np
import pywt
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from cleverhans.model import Model
from cleverhans.attacks import FastGradientMethod

from sp_func_cnn import sp_frontend
from locally_linear_attacks import semi_white_box, white_box

def four_layer_cnn(inputs, phase, reuse=tf.AUTO_REUSE, scope='four_layer_cnn'):
    """
    Four layer CNN from Chapter 6 of the textbook 'Neural Networks and Deep Learning' by Michael A. Nielsen, Determination Press, 2015 (http://neuralnetworksanddeeplearning.com/chap6.html/). TensorFlow-Slim implementation adapted from https://github.com/initialized/tensorflow-tutorial/.

    Layer 1: Convolutional, with a 5x5 receptive field and 20 feature maps. 
    Layer 2: Convolutional, with a 5x5 receptive field and 40 feature maps.
    Layer 3: Fully connected, with 1000 neurons.
    Layer 4: Fully connected, with 1000 neurons.

    :param inputs: Should be of shape [num_samples, 28, 28, 1].
    :param phase: If TRUE, dropout is switched on in layers 3 and 4.
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


class FourLayerModel(Model):
    """Model class for CleverHans library."""

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'logits']

    def fprop(self, x):
        x_norm = (x - self.mu)/self.sigma
        _, end_points = four_layer_cnn(x_norm, phase=False, reuse=tf.AUTO_REUSE)
        return end_points


tf.reset_default_graph() 
tf.logging.set_verbosity(tf.logging.ERROR)

x = tf.placeholder(tf.float32, shape=[1, 28, 28, 1])
eps = tf.placeholder(tf.float32, shape=[], name='perturb')
mu = tf.placeholder(tf.float32, shape=[])
sigma = tf.placeholder(tf.float32, shape=[])

model = FourLayerModel(mu, sigma)
output = model.get_logits(x)
ind = tf.argmax(output, axis=1, output_type=tf.int32)[0]

# FGSM
fgsm = FastGradientMethod(model)
x_fgsm = fgsm.generate(x, eps=eps, clip_min=0., clip_max=1.)

# Semi-white box attack
x_sw = semi_white_box(x, ind, model, eps=eps)

# White box attack
x_w = white_box(x, ind, model, eps=eps)

# Test
x_inp = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='Inputs')
y_actual = tf.placeholder(tf.float32, shape=[None, 10], name='Labels')

logits = model.get_logits(x_inp)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

var_list = slim.get_model_variables(scope='four_layer_cnn')
saver = tf.train.Saver(var_list=var_list)

attacks_no_def = np.array(['FGSM', 'Locally linear attack'])
attacks = np.array(['FGSM', 'Semi-white box attack', 'White box attack'])
tf_var =  np.array([x_fgsm, x_sw, x_w])
mnist = mnist_data.read_data_sets('MNIST_data', one_hot=True)
images_adv = np.zeros([attacks.shape[0], mnist.test.images.shape[0], 28, 28, 1])
accuracies = np.zeros(attacks.shape)

epsilon = 0.25
rho = 0.03

print("\n***************************************")
print("MNIST classification via four layer CNN")
print("***************************************")
print("Attacks use epsilon = {:.2f} \nImages are in the range [0, 1]\n".format(epsilon))
print("**********")
print("No defense")
print("**********")

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver.restore(sess, "./checkpoints/plain.ckpt")  # Pre-trained model without front end
mu_train = np.mean(mnist.train.images)
sigma_train = np.std(mnist.train.images)

for j in range(attacks_no_def.shape[0]):
    for k in range(mnist.test.images.shape[0]):
        attack_data = {
            x: mnist.test.images[k].reshape([-1, 28, 28, 1]),
            eps: epsilon,
            mu: mu_train,
            sigma: sigma_train}
        images_adv[j,k] = np.reshape(sess.run(tf_var[j], feed_dict=attack_data),[28,28,1])
    test_data = {
        x_inp: images_adv[j],
        y_actual: mnist.test.labels,
        mu: mu_train,
        sigma: sigma_train}
    accuracies[j] = sess.run(accuracy, feed_dict=test_data)
    print('{:}: {:.2f}'.format(attacks_no_def[j], 100 * accuracies[j]))
sess.close()

print("\n********************************")
print("Sparsifying front end (rho = {:.0f}%)".format(100*rho))
print("********************************")

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver.restore(sess, "./checkpoints/sparse_3.ckpt") # Pre-trained model with sparsifying front end (rho=3%)
train_sp = sp_frontend(mnist.train.images.reshape([-1, 28, 28, 1]))
mu_train = np.mean(train_sp)
sigma_train = np.std(train_sp)

for j in range(attacks.shape[0]):
    for k in range(mnist.test.images.shape[0]):
        attack_data = {
            x: mnist.test.images[k].reshape([-1, 28, 28, 1]),
            eps: epsilon,
            mu: mu_train,
            sigma: sigma_train}
        images_adv[j, k] = np.reshape(sess.run(tf_var[j], feed_dict=attack_data), [28, 28, 1])

    test_data = {
        x_inp: sp_frontend(images_adv[j]),
        y_actual: mnist.test.labels,
        mu: mu_train,
        sigma: sigma_train}
    accuracies[j] = sess.run(accuracy, feed_dict=test_data)
    print('{:}: {:.2f}'.format(attacks[j], 100 * accuracies[j]))
sess.close()
