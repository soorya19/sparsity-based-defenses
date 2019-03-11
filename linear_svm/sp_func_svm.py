"""
Contains functions related to the sparsifying front end.
Images are assumed to be in the range [-1, 1].
"""
import numpy as np

def sp_frontend(images, rho, psi):
    """
    Sparsifies images in the wavelet basis and returns reconstruction.

    Inputs:
        images - Numpy array of shape [batch_size, dim] and in the range [-1, 1] 
        rho    - Sparsity level in the range [0, 1]
        psi    - Orthonormal basis to sparsify in. Numpy array of shape [dim, dim]

    Output:
        images_sp - Sparsified images. Numpy array of same shape as images
    """ 
    batch_size = images.shape[0]
    dim = images.shape[1]
    images_sp = images.copy()

    coeffs = (0.5*images_sp + 0.5) @ psi.T

    k = np.floor(rho*dim).astype('int')
    n = dim-k
    indices = np.argpartition(np.abs(coeffs), n, axis=1)[:, :n]
    coeffs[np.arange(batch_size)[:, None], indices] = 0

    images_sp = 2.0*np.clip(coeffs @ psi, 0., 1.).reshape([batch_size, dim]) - 1.0
    
    return images_sp.astype(np.float32)

def sp_project(x, w, rho, psi):
    """
    Projects w onto the top rho% of the support of x (in the wavelet basis).

    Inputs:
        x   - Numpy array of shape [batch_size, dim] and in the range [-1, 1] 
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

    coeffs = (0.5*images + 0.5) @ psi.T
    coeffs_w = weights @ psi.T

    k = np.floor(rho*dim).astype('int')
    n = dim-k
    indices = np.argpartition(np.abs(coeffs), n, axis=1)[:, :n]
    coeffs_w[np.arange(batch_size)[:, None], indices] = 0

    weights = (coeffs_w @ psi).reshape([batch_size, dim])

    return weights.astype(np.float32)