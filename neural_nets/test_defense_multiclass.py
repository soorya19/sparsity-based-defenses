"""
Tests the efficacy of sparsity-based defense on a CNN for multiclass MNIST classification.

Defense: Front end with sparsity level = 3.5%
Classifier: 4 layer CNN, MNIST
"""
import numpy as np
from collections import OrderedDict as odict 
from tqdm import trange

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

from cleverhans.model import Model
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, MomentumIterativeMethod

from models import FourLayerModel, FourLayerModelPlain
from attacks import LocLinearAttack, IterLocLinearAttack, PGDAttack, FGSMAttack

# Read MNIST data
mnist = mnist_data.read_data_sets('MNIST_data', one_hot=True)
x_test = mnist.test.images.reshape([-1, 28, 28, 1])
y_test = mnist.test.labels
num_images, num_rows, num_cols, num_channels = x_test.shape
num_classes = y_test.shape[1]

# Attack parameters
attack_params = dict(# FGSM
                     fgsm = dict(run = True, # Whether to run the attack
                                 batch_size = 5000,
                                 params = dict(eps = 0.2,
                                               clip_min = 0.,
                                               clip_max = 1.)),
                     # Locally linear attack
                     loclin = dict(run = True,
                                   batch_size = 5000,
                                   params = dict(eps = 0.2,
                                                 clip_min = 0.,
                                                 clip_max = 1.)),
                     # Iterative locally linear attack
                     iterloclin = dict(run = False,
                                       batch_size = 5000,
                                       params = dict(eps = 0.2,
                                                     delta = 0.01, # Per-iteration L-inf budget
                                                     niter = 1000, # No of iterations
                                                     clip_min = 0.,
                                                     clip_max = 1.)),
                     # Iterative FGSM
                     fgsm_iter = dict(run = False,
                                      batch_size = 5000,
                                      params = dict(eps = 0.2,
                                                    eps_iter = 0.01,
                                                    nb_iter = 1000,
                                                    clip_min = 0.,
                                                    clip_max = 1.)),
                     # Momentum iterative FGSM
                     momentum = dict(run = False,
                                     batch_size = 5000,
                                     params = dict(eps = 0.2,
                                                   eps_iter = 0.01,
                                                   nb_iter = 1000,
                                                   clip_min = 0.,
                                                   clip_max = 1.)),
                     # PGD with multiple random restarts
                     pgd = dict(run = False,
                                batch_size = 5000,
                                params = dict(eps = 0.2,
                                              num_runs = 100, # No of random restarts
                                              delta = 0.05,
                                              niter = 100,
                                              clip_min = 0.,
                                              clip_max = 1.,
                                              rand_init = True,
                                              rand_seed = None))
                     )
# Attacks use BPDA of 1 for gradient of the front end.
# To use iterated projections to calculate the gradient, set proj to True.
proj = False

# Defense parameters
rho = 0.035 # Sparsity level
wavelet = 'coif1'
level = 1
psi = np.load('./wavelet_mat/{}_{}.npz'.format(wavelet, level))['psi']
def_params = dict(rho = rho,
                  psi = psi)
checkpoint = './checkpoints/{:}-{:}-{:.0f}.ckpt'.format(wavelet, level, 1000*rho)

# Create model and accuracy graph
tf.reset_default_graph()
model = FourLayerModel(def_params=def_params)
x = tf.placeholder(tf.float32, shape=(None, num_rows, num_cols, num_channels))
y = tf.placeholder(tf.float32, shape=(None, num_classes))
logits = model.get_logits(x)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = 100*tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Restore model weights from checkpoint
sess = tf.Session()
sess.run(tf.global_variables_initializer())
var_list = slim.get_model_variables(scope='four_layer_cnn')
saver = tf.train.Saver(var_list=var_list)
saver.restore(sess, checkpoint)

acc = odict()
# Calculate accuracy on natural images
acc['no_attack'] = sess.run(accuracy, feed_dict={x: x_test, 
                                             y: y_test})
print('\nAccuracy without attack: {:.2f}'.format(acc['no_attack']))

# FGSM attack
if attack_params['fgsm']['run'] is True:
    fgsm_params = attack_params['fgsm']['params']
    batch_size = attack_params['fgsm']['batch_size']
    print('\nStarting FGSM with eps = ', fgsm_params['eps'])

    fgsm = FastGradientMethod(model, sess=sess)
    adv_fgsm = fgsm.generate(x, y=y, **fgsm_params)
    x_fgsm = np.zeros(x_test.shape)
    for i in trange(num_images//batch_size):
        batch = slice(i*batch_size, (i+1)*batch_size)
        x_fgsm[batch] = sess.run(adv_fgsm, feed_dict={x: x_test[batch], 
                                                      y: y_test[batch]})

    acc['fgsm'] = sess.run(accuracy, feed_dict={x: x_fgsm, 
                                                y: y_test})
    print('Accuracy: {:.2f}'.format(acc['fgsm']))

    if proj is True:
        print('\nStarting FGSM with iterated projections, eps = {:.2f}'.format(fgsm_params['eps']))
        niter_proj = 20
        fgsm = FGSMAttack(model, sess=sess)
        x_fgsm_proj = fgsm.run_proj(x_test, y_test, batch_size, niter_proj, def_params, **fgsm_params)

        acc['fgsm_proj'] = sess.run(accuracy, feed_dict={x: x_fgsm_proj, 
                                                         y: y_test})
        print('\nAccuracy: {:.2f}'.format(acc['fgsm_proj']))

# Locally linear attack
if attack_params['loclin']['run'] is True:
    loclin_params = attack_params['loclin']['params']
    batch_size = attack_params['loclin']['batch_size']
    print('\nStarting locally linear attack with eps = {:.2f}'.format(loclin_params['eps']))

    loclin = LocLinearAttack(model, sess=sess)
    x_loclin = loclin.run(x_test, y_test, batch_size, **loclin_params)

    acc['loclin'] = sess.run(accuracy, feed_dict={x: x_loclin, 
                                                     y: y_test})
    print('Accuracy: {:.2f}'.format(acc['loclin']))

    if proj is True:
        print('\nStarting locally linear attack with iterated projections, eps = {:.2f}'.format(loclin_params['eps']))
        niter_proj = 20
        x_loclin_proj = loclin.run_proj(x_test, y_test, batch_size, niter_proj, def_params, **loclin_params)

        acc['loclin_proj'] = sess.run(accuracy, feed_dict={x: x_loclin_proj, 
                                                           y: y_test})
        print('\nAccuracy: {:.2f}'.format(acc['loclin_proj']))

# Iterative locally linear attack
if attack_params['iterloclin']['run'] is True:
    iterloclin_params = attack_params['iterloclin']['params']
    batch_size = attack_params['iterloclin']['batch_size']
    print('\nStarting iterative locally linear attack with eps = {:.2f}, delta = {:.2f}, niter = {}'.format(iterloclin_params['eps'], iterloclin_params['delta'], iterloclin_params['niter']))

    iterloclin = IterLocLinearAttack(model, sess=sess)
    x_iterloclin = iterloclin.run(x_test, y_test, batch_size, **iterloclin_params)

    acc['iterloclin'] = sess.run(accuracy, feed_dict={x: x_iterloclin, 
                                                     y: y_test})
    print('\nAccuracy: {:.2f}'.format(acc['iterloclin']))

    if proj is True:
        print('\nStarting iterative locally linear attack with iterated projections, eps = {:.2f}, delta = {:.2f}, niter = {}'.format(iterloclin_params['eps'], iterloclin_params['delta'], iterloclin_params['niter']))
        niter_proj = 20
        x_iterloclin_proj = iterloclin.run_proj(x_test, y_test, batch_size, niter_proj, def_params, **iterloclin_params)

        acc['iterloclin_proj'] = sess.run(accuracy, feed_dict={x: x_iterloclin_proj, 
                                                               y: y_test})
        print('\nAccuracy: {:.2f}'.format(acc['iterloclin_proj']))

# Iterative FGSM attack
if attack_params['fgsm_iter']['run'] is True:
    fgsm_iter_params = attack_params['fgsm_iter']['params']
    batch_size = attack_params['fgsm_iter']['batch_size']
    print('\nStarting iterative FGSM with eps = {:.2f}, delta = {:.2f}, niter = {}'.format(fgsm_iter_params['eps'], fgsm_iter_params['eps_iter'], fgsm_iter_params['nb_iter']))

    fgsm_iter = BasicIterativeMethod(model, sess=sess)
    adv_fgsm_iter = fgsm_iter.generate(x, y=y, **fgsm_iter_params)
    x_fgsm_iter = np.zeros(x_test.shape)
    for i in trange(num_images//batch_size):
        batch = slice(i*batch_size, (i+1)*batch_size)
        x_fgsm_iter[batch] = sess.run(adv_fgsm_iter, feed_dict={x: x_test[batch], 
                                                                y: y_test[batch]})

    acc['fgsm_iter'] = sess.run(accuracy, feed_dict={x: x_fgsm_iter, 
                                                     y: y_test})
    print('\nAccuracy: {:.2f}'.format(acc['fgsm_iter']))

# Momentum iterative FGSM attack
if attack_params['momentum']['run'] is True:
    momentum_params = attack_params['momentum']['params']
    batch_size = attack_params['momentum']['batch_size']
    print('\nStarting momentum iterative FGSM with eps = {:.2f}, delta = {:.2f}, niter = {}'.format(momentum_params['eps'], momentum_params['eps_iter'], momentum_params['nb_iter']))

    momentum = MomentumIterativeMethod(model, sess=sess)
    adv_momentum = momentum.generate(x, y=y, **momentum_params)
    x_momentum = np.zeros(x_test.shape)
    for i in trange(num_images//batch_size):
        batch = slice(i*batch_size, (i+1)*batch_size)
        x_momentum[batch] = sess.run(adv_momentum, feed_dict={x: x_test[batch], 
                                                               y: y_test[batch]})

    acc['momentum'] = sess.run(accuracy, feed_dict={x: x_momentum, 
                                                    y: y_test})
    print('Accuracy: {:.2f}'.format(acc['momentum']))

# PGD attack with multiple random restarts
if attack_params['pgd']['run'] is True:
    pgd_params = attack_params['pgd']['params']
    batch_size = attack_params['pgd']['batch_size']
    print('\nStarting PGD with {} random restarts, using eps = {:.2f}, delta = {:.2f}, niter = {}'.format(pgd_params['num_runs'], pgd_params['eps'], pgd_params['delta'], pgd_params['niter']))

    pgd = PGDAttack(model, sess=sess)
    x_pgd = pgd.run_multiple(x_test, y_test, batch_size, **pgd_params)
    # Here accuracy is reported within the attack function.
    # Acc is calculated over the most successful restart(s) for *each* image

    if proj is True:
        print('\nStarting PGD with {} random restarts, using iterated projections, with eps = {:.2f}, delta = {:.2f}, niter = {}'.format(pgd_params['num_runs'], pgd_params['eps'], pgd_params['delta'], pgd_params['niter']))
        niter_proj = 20
        x_pgd_proj = pgd.run_multiple_proj(x_test, y_test, batch_size, niter_proj, def_params, **pgd_params)

print('\n--------\nSummary:\n--------')
for key, value in acc.items():
    print('{}: {:.2f}'.format(key, value))

sess.close()

