"""
Tests the efficacy of sparsity-based defense on a 2-layer NN for binary MNIST classification.

Defense: Front end with sparsity level = 3%
Classifier: 2 layer fully-connected NN, 3 vs 7 binary MNIST
"""
import numpy as np
from collections import OrderedDict as odict 
from tqdm import trange

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

import mnist_data_pair
from models import TwoLayerModel, TwoLayerModelPlain
from attacks import LocLinearAttack, IterLocLinearAttack

# Read MNIST data
digit1 = 3
digit2 = 7
mnist = mnist_data_pair.read_data_sets('MNIST_data',
                                       digit1 = digit1,
                                       digit2 = digit2,
                                       test_ratio = 0.25,
                                       validation_ratio = 0.1)
x_test = mnist.test.images
y_test = mnist.test.labels
num_images, dim = x_test.shape
num_classes = y_test.shape[1]

# Attack parameters
attack_params = dict(# Locally linear attack
                     loclin = dict(run = True,
                                   batch_size = num_images,
                                   params = dict(eps = 0.2,
                                                 clip_min = 0.,
                                                 clip_max = 1.)),
                     # Iterative locally linear attack
                     iterloclin = dict(run = False,
                                       batch_size = num_images,
                                       params = dict(eps = 0.2,
                                                     delta = 0.01, # Per-iteration L-inf budget
                                                     niter = 1000, # No of iterations
                                                     clip_min = 0.,
                                                     clip_max = 1.))
                     )
# Attacks use BPDA of 1 for gradient of the front end.
# To use iterated projections to calculate the gradient, set proj to True.
proj = False

# Defense parameters
rho = 0.03  # Sparsity level
wavelet = 'db5'
level = 1
psi = np.load('./wavelet_mat/{}_{}.npz'.format(wavelet, level))['psi']
def_params = dict(rho = rho,
                  psi = psi)
checkpoint = './checkpoints/{}-{}-{:.0f}-{}-{}.ckpt'.format(wavelet, level, 1000*rho, digit1, digit2)

# Create model and accuracy graph
tf.reset_default_graph()
model = TwoLayerModel(def_params=def_params)
x = tf.placeholder(tf.float32, shape=(None, dim))
y = tf.placeholder(tf.float32, shape=(None, num_classes))
logits = model.get_logits(x)
prediction = tf.maximum(tf.sign(logits), 0)
accuracy = 100*tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32))

# Restore model weights from checkpoint
sess = tf.Session()
sess.run(tf.global_variables_initializer())
var_list = slim.get_model_variables(scope='two_layer_nn')
saver = tf.train.Saver(var_list=var_list)
saver.restore(sess, checkpoint)

acc = odict()
# Calculate accuracy on natural images
acc['no_attack'] = sess.run(accuracy, feed_dict={x: x_test, 
                                             y: y_test})
print('\nAccuracy without attack: {:.2f}'.format(acc['no_attack']))

# Locally linear attack
if attack_params['loclin']['run'] is True:
    loclin_params = attack_params['loclin']['params']
    batch_size = attack_params['loclin']['batch_size']
    print('\nStarting locally linear attack with eps = {:.2f}'.format(loclin_params['eps']))

    loclin = LocLinearAttack(model, sess=sess)
    x_loclin = loclin.run_binary(x_test, y_test, batch_size, **loclin_params)

    acc['loclin'] = sess.run(accuracy, feed_dict={x: x_loclin, 
                                                     y: y_test})
    print('Accuracy: {:.2f}'.format(acc['loclin']))

    if proj is True:
        print('\nStarting locally linear attack with iterated projections, eps = {:.2f}'.format(loclin_params['eps']))
        niter_proj = 20
        x_loclin_proj = loclin.run_binary_proj(x_test, y_test, batch_size, niter_proj, def_params, **loclin_params)

        acc['loclin_proj'] = sess.run(accuracy, feed_dict={x: x_loclin_proj, 
                                                           y: y_test})
        print('\nAccuracy: {:.2f}'.format(acc['loclin_proj']))

# Iterative locally linear attack
if attack_params['iterloclin']['run'] is True:
    iterloclin_params = attack_params['iterloclin']['params']
    batch_size = attack_params['iterloclin']['batch_size']
    print('\nStarting iterative locally linear attack with eps = {:.2f}, delta = {:.2f}, niter = {}'.format(iterloclin_params['eps'], iterloclin_params['delta'], iterloclin_params['niter']))

    iterloclin = IterLocLinearAttack(model, sess=sess)
    x_iterloclin = iterloclin.run_binary(x_test, y_test, batch_size, **iterloclin_params)

    acc['iterloclin'] = sess.run(accuracy, feed_dict={x: x_iterloclin, 
                                                     y: y_test})
    print('\nAccuracy: {:.2f}'.format(acc['iterloclin']))

    if proj is True:
        print('\nStarting iterative locally linear attack with iterated projections, eps = {:.2f}, delta = {:.2f}, niter = {}'.format(iterloclin_params['eps'], iterloclin_params['delta'], iterloclin_params['niter']))
        niter_proj = 20
        x_iterloclin_proj = iterloclin.run_binary_proj(x_test, y_test, batch_size, niter_proj, def_params, **iterloclin_params)

        acc['iterloclin_proj'] = sess.run(accuracy, feed_dict={x: x_iterloclin_proj, 
                                                               y: y_test})
        print('\nAccuracy: {:.2f}'.format(acc['iterloclin_proj']))

print('\n--------\nSummary:\n--------')
for key, value in acc.items():
    print('{}: {:.2f}'.format(key, value))

sess.close()

