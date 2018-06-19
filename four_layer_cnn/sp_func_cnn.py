"""
Contains functions related to the sparsifying front end.
Images are assumed to be in the range [0, 1]. 
"""

import numpy as np
import pywt

def sp_frontend(images, rho=0.03, wavelet='bior4.4', mode='periodization', max_lev=1):
	"""
	Sparsifies input in the wavelet basis (using the PyWavelets package) and returns reconstruction.

	:param images: Should be in the range [0, 1] and of shape [num_samples, num_features, num_features, 1].
	:param rho: Sparsity level, in the range [0, 1].
	:param wavelet: Wavelet to use in the transform. See https://pywavelets.readthedocs.io/ for more details.
	:param mode: Signal extension mode. See https://pywavelets.readthedocs.io/ for more details.
	:param max_lev: Maximum allowed level of decomposition.
	"""
	num_samples = images.shape[0]
	num_features = images.shape[1]
	images_sp = images.copy()
	for i in range(num_samples):
		image = images[i].reshape(num_features,num_features)
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
			wp[paths[j]].data = coeffs[j*m*m:(j+1)*m*m].reshape([m,m])
		image_r = wp.reconstruct(update=False).astype('float32')
		image_r = np.clip(image_r, 0.0, 1.0)
		images_sp[i, :, :, 0] = image_r
	return images_sp

def sp_project(image, weights, wavelet='bior4.4', mode='periodization', max_lev=1, rho=0.03):
	"""
	Projects weights onto top rho% of the support of image (in the wavelet basis).

	:param image: Should be in the range [0, 1], and resizable to shape [num_features, num_features]
	:param weights: Should be resizable to shape [num_features, num_features].
	:param rho: Sparsity level, in the range [0, 1].
	:param wavelet: Wavelet to use in the transform. See https://pywavelets.readthedocs.io/ for more details.
	:param mode: Signal extension mode. See https://pywavelets.readthedocs.io/ for more details.
	:param max_lev: Maximum allowed level of decomposition.
	"""
	num_features = image.shape[1]
	weights_proj = np.array(weights).reshape([1, num_features, num_features, 1])
	wp = pywt.WaveletPacket2D(image.reshape(num_features,num_features), wavelet, mode, max_lev)
	paths = [node.path for node in wp.get_level(max_lev)]
	m = wp[paths[0]].data.shape[0]
	l = (4**max_lev)*m*m
	k = np.floor(rho*l).astype('int')
	n = l-k
	coeffs = np.zeros(l)
	for j in range(4**max_lev):
		coeffs[j*m*m:(j+1)*m*m] = wp[paths[j]].data.flatten()
	indices = np.argpartition(np.abs(coeffs), n)[:n]

	weight = weights_proj.reshape(num_features,num_features)
	wp_w = pywt.WaveletPacket2D(weight.reshape(num_features,num_features), wavelet, mode, max_lev)
	paths_w = [node.path for node in wp_w.get_level(max_lev)]
	coeffs_w = np.zeros(l)
	for j in range(4**max_lev):
		coeffs_w[j*m*m:(j+1)*m*m] = wp_w[paths_w[j]].data.flatten()
	coeffs_w[indices] = 0
	for j in range(4**max_lev):
		wp_w[paths_w[j]].data = coeffs_w[j*m*m:(j+1)*m*m].reshape([m,m])
	weights_proj[0, :, :, 0] = wp_w.reconstruct(update=False).astype('float32')
	return weights_proj
