"""
Contains functions related to the sparsifying front end.
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

def sp_frontend(images, rho, psi):
    """
    Sparsifies images in the wavelet basis and returns reconstruction.

    Inputs:
        images - Numpy array of shape [batch_size, num_features, num_features, 1] and in the range [0, 1] 
        rho    - Sparsity level in the range [0, 1]
        psi    - Orthonormal basis to sparsify in. Numpy array of shape 
                 [num_features*num_features, num_features*num_features]

    Output:
        images_sp - Sparsified images. Numpy array of same shape as images
    """ 
    batch_size = images.shape[0]
    num_features = images.shape[1]
    dim = num_features*num_features
    images_sp = images.copy()

    coeffs = images_sp.reshape([batch_size, dim]) @ psi.T

    k = np.floor(rho*dim).astype('int')
    n = dim-k
    indices = np.argpartition(np.abs(coeffs), n, axis=1)[:, :n]
    coeffs[np.arange(batch_size)[:, None], indices] = 0

    images_sp = np.clip(coeffs @ psi, 0., 1.).reshape([batch_size, num_features, num_features, 1])
    
    return images_sp.astype(np.float32)

def sp_project(x, w, rho, psi):
    """
    Projects w onto the top rho% of the support of x (in the wavelet basis).

    Inputs:
        x   - Numpy array of shape [batch_size, num_features, num_features, 1] and in the range [0, 1] 
        w   - Numpy array of same shape as x
        rho - Sparsity level in the range [0, 1]
        psi - Orthonormal basis to sparsify in. Numpy array of shape 
              [num_features*num_features, num_features*num_features]

    Output:
        weights - Projected version of w. Same shape as w
    """
    images = x.copy()
    weights = w.copy()
    batch_size = images.shape[0]
    num_features = images.shape[1]
    dim = num_features*num_features

    coeffs = images.reshape([batch_size, dim]) @ psi.T
    coeffs_w = weights.reshape([batch_size, dim]) @ psi.T

    k = np.floor(rho*dim).astype('int')
    n = dim-k
    indices = np.argpartition(np.abs(coeffs), n, axis=1)[:, :n]
    coeffs_w[np.arange(batch_size)[:, None], indices] = 0

    weights = (coeffs_w @ psi).reshape([batch_size, num_features, num_features, 1])

    return weights.astype(np.float32)

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    """
    tf.py_func with custom gradient
    """
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def grad_sp_frontend(op, grad):
    """
    Returns the propagated gradient with respect to the first, second and third 
    argument of tf_sp_frontend.
    Uses BPDA of 1 for gradient of front end.
    """
    return [grad, None, None]

def tf_sp_frontend(images, rho, psi, name=None):
    """
    tf.py_func version of sp_frontend
    """
    with tf.name_scope(None, "mod", [images, rho, psi]) as name:
        z = py_func(sp_frontend,
                    [images, rho, psi],
                    [tf.float32],
                    name=name,
                    grad=grad_sp_frontend)  # <-- here's the call to the gradient
        return z[0]

###################################################################################################
# Functions for binary classification using fully connected NN 
# where images are of shape [batch_size, dim]

def sp_frontend_binary(images, rho, psi):
    """
    Sparsifies images in the wavelet basis and returns reconstruction.

    Inputs:
        images - Numpy array of shape [batch_size, dim] and in the range [0, 1] 
        rho    - Sparsity level in the range [0, 1]
        psi    - Orthonormal basis to sparsify in. Numpy array of shape [dim, dim]

    Output:
        images_sp - Sparsified images. Numpy array of same shape as images
    """ 
    batch_size = images.shape[0]
    dim = images.shape[1]
    images_sp = images.copy()

    coeffs = images_sp @ psi.T

    k = np.floor(rho*dim).astype('int')
    n = dim-k
    indices = np.argpartition(np.abs(coeffs), n, axis=1)[:, :n]
    coeffs[np.arange(batch_size)[:, None], indices] = 0

    images_sp = np.clip(coeffs @ psi, 0., 1.)
    
    return images_sp.astype(np.float32)

def sp_project_binary(x, w, rho, psi):
    """
    Projects w onto the top rho% of the support of x (in the wavelet basis).

    Inputs:
        x   - Numpy array of shape [batch_size, dim] and in the range [0, 1] 
        w   - Numpy array of same shape as x
        rho - Sparsity level in the range [0, 1]
        psi - Orthonormal basis to sparsify in. Numpy array of shape [dim, dim]

    Output:
        weights - Projected version of w. Same shape as w
    """
    images = x.copy()
    weights = w.copy()
    batch_size = images.shape[0]
    dim = images.shape[1]

    coeffs = images @ psi.T
    coeffs_w = weights @ psi.T

    k = np.floor(rho*dim).astype('int')
    n = dim-k
    indices = np.argpartition(np.abs(coeffs), n, axis=1)[:, :n]
    coeffs_w[np.arange(batch_size)[:, None], indices] = 0

    weights = coeffs_w @ psi

    return weights.astype(np.float32)

def grad_sp_frontend_binary(op, grad):
    """
    Returns the propagated gradient with respect to the first, second and third 
    argument of tf_sp_frontend_binary.
    Uses BPDA of 1 for gradient of front end.
    """
    return [grad, None, None]

def tf_sp_frontend_binary(images, rho, psi, name=None):
    """
    tf.py_func version of sp_frontend_binary
    """
    with tf.name_scope(None, "mod", [images, rho, psi]) as name:
        z = py_func(sp_frontend_binary,
                        [images, rho, psi],
                        [tf.float32],
                        name=name,
                        grad=grad_sp_frontend_binary)  # <-- here's the call to the gradient
        return z[0]
