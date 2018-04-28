"""
Contains functions related to the sparsifying front end.
Images are assumed to be in the range [-1, 1].
"""

import numpy as np
import pywt

def sp_frontend(images, rho=0.02, wavelet='bior4.4', mode='periodization', max_lev=1):
	"""
    Sparsifies input in the wavelet basis (using the PyWavelets package) and returns reconstruction.

    :param images: Should be in the range [-1, 1] and of shape [num_samples, num_features] where num_features is a perfect square.
    :param rho: Sparsity level, in the range [0, 1].
    :param wavelet: Wavelet to use in the transform. See https://pywavelets.readthedocs.io/ for more details.
    :param mode: Signal extension mode. See https://pywavelets.readthedocs.io/ for more details.
    :param max_lev: Maximum allowed level of decomposition.
    """
	# Input is assumed to be in the range [-1, 1] and of shape [num_samples, 784] 
	# Projects input onto 
	num_samples = images.shape[0]
	num_features = images.shape[1]
	num_features_per_dim = np.int(np.sqrt(num_features))
	images_sp = images.copy()
	for i in range(num_samples):
		image = 0.5*images[i,:].reshape(num_features_per_dim, num_features_per_dim) + 0.5 
		wp = pywt.WaveletPacket2D(image, wavelet, mode, max_lev)
		paths = [node.path for node in wp.get_level(max_lev)]
		m = wp[paths[0]].data.shape[0]
		l = (4**max_lev)*m*m
		k = np.floor(rho*l).astype('int')
		n = l-k
		coeffs = np.zeros(l)
		for j in range(4**max_lev):
			coeffs[j*m*m:(j+1)*m*m] = wp[paths[j]].data.flatten()
		indices = np.argpartition(np.abs(coeffs), n)[:n]
		coeffs[indices] = 0
		for j in range(4**max_lev):
			wp[paths[j]].data = coeffs[j*m*m:(j+1)*m*m].reshape([m, m])
		image_r = wp.reconstruct(update=False).astype('float32')
		image_r = 2.0*np.clip(image_r, 0.0, 1.0) - 1.0
		images_sp[i,:] = image_r.flatten()
	return images_sp

def sp_project(image, weights, wavelet='bior4.4', mode='periodization', max_lev=1, rho=0.02):
	"""
	Projects weights onto top rho% of the support of image (in the wavelet basis).

	:param image: Should be in the range [-1, 1] and of shape [num_features] where num_features is a perfect square.
	:param weights: Should be of shape [num_features].
    :param rho: Sparsity level, in the range [0, 1].
    :param wavelet: Wavelet to use in the transform. See https://pywavelets.readthedocs.io/ for more details.
    :param mode: Signal extension mode. See https://pywavelets.readthedocs.io/ for more details.
    :param max_lev: Maximum allowed level of decomposition.
    """
	num_features = image.shape[0]
	num_features_per_dim = np.int(np.sqrt(num_features))
	image = 0.5*image.reshape([num_features_per_dim, num_features_per_dim]) + 0.5 
	wp = pywt.WaveletPacket2D(image, wavelet, mode, max_lev)
	paths = [node.path for node in wp.get_level(max_lev)]
	m = wp[paths[0]].data.shape[0]
	l = (4**max_lev)*m*m
	k = np.floor(rho*l).astype('int')
	n = l-k
	coeffs = np.zeros(l)
	for j in range(4**max_lev):
		coeffs[j*m*m:(j+1)*m*m] = wp[paths[j]].data.flatten()
	indices = np.argpartition(np.abs(coeffs), n)[:n]

	wp_w = pywt.WaveletPacket2D(weights.reshape(num_features_per_dim, num_features_per_dim), wavelet, mode, max_lev)
	paths_w = [node.path for node in wp_w.get_level(max_lev)]
	coeffs_w = np.zeros(l)
	for j in range(4**max_lev):
		coeffs_w[j*m*m:(j+1)*m*m] = wp_w[paths_w[j]].data.flatten()
	coeffs_w_proj = coeffs_w.copy()
	coeffs_w_proj[indices] = 0
	for i2 in range(4**max_lev):
		wp_w[paths_w[i2]].data = coeffs_w_proj[i2*m*m:(i2+1)*m*m].reshape([m, m])
	weights_proj = wp_w.reconstruct(update=False).astype('float32').flatten()
	return weights_proj
