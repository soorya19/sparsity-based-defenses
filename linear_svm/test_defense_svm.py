"""
Tests the efficacy of sparsity-based defense against adversarial attacks on a linear SVM.

Defense: Sparsifying front end with rho = 2% (sparsity level).
Classifier: Linear SVM, used for binary classification of digit pairs from the MNIST dataset.
"""
import numpy as np
from sklearn import datasets, utils, model_selection, metrics, svm

from sp_func_svm import sp_project, sp_frontend
from mnist_workaround import fetch_mnist

# Defense parameters
rho = 0.02 # Sparsity level
wavelet = 'db5'
level = 1
psi = np.load('./wavelet_mat/{}_{}.npz'.format(wavelet, level))['psi']
def_params = dict(rho = rho,
                  psi = psi)


# L-inf attack budget, corresponding to images in the range [-1, 1]
epsilon = 0.2
proj_iter = False # Change to True to run attack with iterated projections

# Read MNIST data
digit_1 = 3
digit_2 = 7
fetch_mnist()
mnist = datasets.fetch_mldata("MNIST original")
digit_1_data = 2.0*mnist.data[mnist.target==digit_1]/255.0 - 1.0
digit_2_data = 2.0*mnist.data[mnist.target==digit_2]/255.0 - 1.0
data = np.vstack([digit_1_data, digit_2_data])  
labels = np.hstack([np.repeat(digit_1, digit_1_data.shape[0]), np.repeat(digit_2, digit_2_data.shape[0])])  
data, labels = utils.shuffle(data, labels, random_state=1234)
x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.25, random_state=1234, stratify=labels)
num_test = x_test.shape[0]

print("\n*************************************")
print("{:} vs. {:} classification via linear SVM".format(digit_1,digit_2))
print("*************************************")
print("Attacks use epsilon = {:.2f} \nImages are in the range [-1, 1]\n".format(epsilon))
print("**********")
print("No defense")
print("**********")
clf = svm.LinearSVC(loss="hinge", random_state=65)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
score_no_adv = 100*metrics.accuracy_score(y_test, pred)
print("\nBaseline accuracy: {:.2f}%" .format(score_no_adv))

digit_smaller = min(digit_1, digit_2)
sgn_attack = np.tile(((y_test==digit_smaller)*2-1)[..., None], [1, 784])
weights_no_def = np.tile(clf.coef_, [num_test, 1])
x_adv = x_test + epsilon*np.sign(weights_no_def)*sgn_attack
x_adv = np.clip(x_adv, -1.0, 1.0)
pred_adv = clf.predict(x_adv)
score_adv = 100*metrics.accuracy_score(y_test, pred_adv)
print("After attack: {:.2f}% \n" .format(score_adv))

print("********************************")
print("Sparsifying front end (rho = {:.1f}%)".format(100*rho))
print("********************************")
clf_sp = svm.LinearSVC(loss="hinge", random_state=1234)
x_train_sp = sp_frontend(x_train, **def_params)
x_test_sp = sp_frontend(x_test, **def_params)
clf_sp.fit(x_train_sp, y_train)
pred_sp = clf_sp.predict(x_test_sp)
acc_bl = 100*metrics.accuracy_score(y_test, pred_sp)
print("\nBaseline: {:.2f}%" .format(acc_bl))

# Semi-white box attack
weights = np.tile(clf_sp.coef_, [num_test, 1])
x_adv_sw = x_test + epsilon*np.sign(weights)*sgn_attack
x_adv_sw = np.clip(x_adv_sw, -1.0, 1.0)
x_adv_sw_sp = sp_frontend(x_adv_sw, **def_params)
pred_adv_sw_sp = clf_sp.predict(x_adv_sw_sp)
acc_sw = 100*metrics.accuracy_score(y_test, pred_adv_sw_sp)
print("Semi-white box attack: {:.2f}%" .format(acc_sw))

# White box attack
weights_proj = sp_project(x_test, weights, **def_params)
x_adv_w = x_test + epsilon*np.sign(weights_proj)*sgn_attack
x_adv_w = np.clip(x_adv_w, -1.0, 1.0)
x_adv_w_sp = sp_frontend(x_adv_w, **def_params)
pred_adv_w_sp = clf_sp.predict(x_adv_w_sp)
acc_w = 100*metrics.accuracy_score(y_test, pred_adv_w_sp)
print("White box attack: {:.2f}%" .format(acc_w))

if proj_iter is True:
    # White box attack with iterated projections
    x_adv_w_proj = x_adv_sw.copy()
    niter_proj = 100
    for n in range(niter_proj):
        weights_proj = sp_project(x_adv_w_proj, weights, **def_params)
        x_adv_w_proj = x_test + epsilon*np.sign(weights_proj)*sgn_attack
        x_adv_w_proj = np.clip(x_adv_w_proj, -1.0, 1.0)
    x_adv_w_proj_sp = sp_frontend(x_adv_w_proj, **def_params)
    pred_adv_proj_sp = clf_sp.predict(x_adv_w_proj_sp)
    acc_w_proj = 100*metrics.accuracy_score(y_test, pred_adv_proj_sp)
    print("White box attack (iterated projections): {:.2f}% \n" .format(acc_w_proj))
