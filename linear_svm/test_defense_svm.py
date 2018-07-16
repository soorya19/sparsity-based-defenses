"""
Tests the efficacy of sparsity-based defenses against adversarial attacks on a linear SVM.

Classifier: Linear SVM, used for binary classification of digit pairs from the MNIST dataset.
Defense: Sparsifying front end with rho = 2% (sparsity level).
Adversarial attacks: Semi-white box and white box attacks with epsilon = 0.25 (L-infinity attack budget).
"""

import numpy as np
import pywt
from sklearn import datasets, utils, model_selection, metrics, svm
from sp_func_svm import sp_project, sp_frontend
from mnist_workaround import fetch_mnist

epsilon = 0.25 # L-infinity attack budget. Images are assumed to be in the range [-1, 1].
rho = 0.02 # Sparsity level used in the defense, in the range [0, 1].
digit_1 = 3
digit_2 = 7

fetch_mnist()
mnist = datasets.fetch_mldata("MNIST original")
digit_1_data = 2.0*mnist.data[mnist.target==digit_1]/255.0 - 1.0
digit_2_data = 2.0*mnist.data[mnist.target==digit_2]/255.0 - 1.0
data = np.vstack([digit_1_data, digit_2_data])  
labels = np.hstack([np.repeat(digit_1, digit_1_data.shape[0]), np.repeat(digit_2, digit_2_data.shape[0])])  
data, labels = utils.shuffle(data, labels, random_state=1234)
data_train, data_test, labels_train, labels_test = model_selection.train_test_split(data, labels, test_size=0.25, random_state=1234)

print("\n*************************************")
print("{:} vs. {:} classification via linear SVM".format(digit_1,digit_2))
print("*************************************")
print("Attacks use epsilon = {:.2f} \nImages are in the range [-1, 1]\n".format(epsilon))
print("**********")
print("No defense")
print("**********")
clf = svm.LinearSVC(loss="hinge", random_state=65)
clf.fit(data_train, labels_train)
pred = clf.predict(data_test)
score_bl_no_adv = 100*metrics.accuracy_score(labels_test, pred)
print("Baseline accuracy: {:.2f}%" .format(score_bl_no_adv))

data_test_adv = np.zeros(data_test.shape)
for i in range(labels_test.shape[0]):
	if labels_test[i]==digit_1:
		data_test_adv[i] = data_test[i] + epsilon*np.sign(clf.coef_)
	else:
		data_test_adv[i] = data_test[i] - epsilon*np.sign(clf.coef_)
data_test_adv = np.clip(data_test_adv, -1.0, 1.0)
pred_adv = clf.predict(data_test_adv)
score_adv = 100*metrics.accuracy_score(labels_test, pred_adv)
print("After attack: {:.2f}% \n" .format(score_adv))

print("********************************")
print("Sparsifying front end (rho = {:.0f}%)".format(100*rho))
print("********************************")
clf_sp = svm.LinearSVC(loss="hinge", random_state=1234)
data_train_sp = sp_frontend(data_train, rho=rho)
data_test_sp = sp_frontend(data_test, rho=rho)
clf_sp.fit(data_train_sp, labels_train)
pred_sp = clf_sp.predict(data_test_sp)
acc_bl = 100*metrics.accuracy_score(labels_test, pred_sp)
print("Baseline: {:.2f}%" .format(acc_bl))

# Semi-white box attack
data_test_adv_sw = np.zeros(data_test.shape)
for i in range(labels_test.shape[0]):
	if labels_test[i]==digit_1:
		data_test_adv_sw[i] = data_test[i] +  epsilon*np.sign(clf_sp.coef_)
	else:
		data_test_adv_sw[i] = data_test[i] - epsilon*np.sign(clf_sp.coef_)
data_test_adv_sw = np.clip(data_test_adv_sw, -1.0, 1.0)
data_test_adv_sw_sp = sp_frontend(data_test_adv_sw, rho=rho)
pred_adv_sw_sp = clf_sp.predict(data_test_adv_sw_sp)
acc_sw = 100*metrics.accuracy_score(labels_test, pred_adv_sw_sp)
print("Semi-white box attack: {:.2f}%" .format(acc_sw))

# White box attack
data_test_adv_w = np.zeros(data_test.shape)
for i in range(labels_test.shape[0]):
	weights_proj = sp_project(data_test[i], clf_sp.coef_, rho=rho)
	if labels_test[i]==digit_1:
		data_test_adv_w[i] = data_test[i] + epsilon*np.sign(weights_proj)
	else:
		data_test_adv_w[i] = data_test[i] - epsilon*np.sign(weights_proj)
data_test_adv_w = np.clip(data_test_adv_w, -1.0, 1.0)
data_test_adv_w_sp = sp_frontend(data_test_adv_w, rho=rho)
pred_adv_w_sp = clf_sp.predict(data_test_adv_w_sp)
acc_w = 100*metrics.accuracy_score(labels_test, pred_adv_w_sp)
print("White box attack: {:.2f}%" .format(acc_w))