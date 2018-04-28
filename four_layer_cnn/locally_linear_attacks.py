"""
Contains functions for attacks based on the locally linear model.
"""

import numpy as np
import tensorflow as tf
from sp_func_cnn import sp_project

def semi_white_box(x, source, model, eps=0.25):
	"""
	Locally linear semi-white box attack. Given L labels and the true label l, it finds the locally linear attacks required to target each of the L-1 false labels by calculating epsilon*sgn(weq_i - weq_l). It then picks the attack with the highest output distortion.

	:param x: Should be in the range [0, 1] and of shape [1, num_features, num_features, 1].
	:param source: True label.
	:param model: Model to attack. Should be an instance of cleverhans.model.Model.
	:param eps: L-infinity attack budget.
	"""
	func = lambda target: semi_white_box_each(x, source, target, model, eps)
	targets = np.arange(10).astype(np.int32)
	x_sw_temp = tf.map_fn(func, targets, dtype=tf.float32)
	func2 = lambda target: model.get_logits(x_sw_temp[target])[:, target][0] - model.get_logits(x_sw_temp[target])[:, source][0]
	output_temp = tf.map_fn(func2, targets, dtype=tf.float32)
	val, ind = tf.nn.top_k(tf.reshape(output_temp, [10]), k=2)
	target = tf.cond(tf.equal(ind[0],source), lambda: ind[1], lambda: ind[0])
	x_sw = x_sw_temp[target]
	return x_sw

def white_box(x, source, model, eps=0.25):
	"""
	Locally linear white box attack. Similar to the semi white box attack, but instead of using the equivalent weights directly, it uses the projection of the weights on the input.

	:param x: Should be in the range [0, 1] and of shape [1, num_features, num_features, 1].
	:param source: True label.
	:param model: Model to attack; should be an instance of cleverhans.model.Model.
	:param eps: L-infinity attack budget.
	"""
	func = lambda target: white_box_each(x, source, target, model, eps)
	targets = np.arange(10).astype(np.int32)
	x_w_temp = tf.map_fn(func, targets, dtype=tf.float32)
	func2 = lambda target: model.get_logits(x_w_temp[target])[:, target][0] - model.get_logits(x_w_temp[target])[:, source][0]
	output_temp = tf.map_fn(func2, targets, dtype=tf.float32)
	val, ind = tf.nn.top_k(tf.reshape(output_temp, [10]), k=2)
	target = tf.cond(tf.equal(ind[0],source), lambda: ind[1], lambda: ind[0])
	x_w = x_w_temp[target]
	return x_w    

def semi_white_box_each(x, source, target, model, eps=0.25):
	"""
	Helper function used in semi_white_box(.)
	"""
	output = model.get_logits(x)
	weq_source = tf.gradients(output[:,source], x)[0]
	weq_target = tf.gradients(output[:,target], x)[0]
	sweq = tf.reshape(tf.sign(tf.subtract(weq_target,weq_source)),[-1,28,28,1])
	x_sw_each = tf.clip_by_value(x + eps*sweq, 0., 1.)
	return x_sw_each

def white_box_each(x, source, target, model, eps=0.25):
	"""
	Helper function used in white_box(.)
	"""
	output = model.get_logits(x)
	weq_source = tf.gradients(output[:,source], x)[0]
	weq_source_proj = tf.py_func(sp_project, [x, weq_source], tf.float32)
	weq_target = tf.gradients(output[:,target], x)[0]
	weq_target_proj = tf.py_func(sp_project, [x, weq_target], tf.float32)
	sweq = tf.reshape(tf.sign(tf.subtract(weq_target_proj,weq_source_proj)),[-1,28,28,1])
	x_w_each = tf.clip_by_value(x + eps*sweq, 0., 1.)
	return x_w_each
