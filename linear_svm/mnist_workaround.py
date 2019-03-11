"""
A workaround to download MNIST data since mldata.org appears to be unstable
Taken from https://github.com/scikit-learn/scikit-learn/issues/8588#issuecomment-292634781
"""

from shutil import copyfileobj
from six.moves import urllib
from sklearn.datasets.base import get_data_home
import os

def fetch_mnist():
	mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
	data_home = get_data_home()
	data_home = os.path.join(data_home, 'mldata')
	if not os.path.exists(data_home):
	    os.makedirs(data_home)
	mnist_save_path = os.path.join(data_home, "mnist-original.mat")
	if not os.path.exists(mnist_save_path):
	    mnist_url = urllib.request.urlopen(mnist_alternative_url)
	    with open(mnist_save_path, "wb") as matlab_file:
	        copyfileobj(mnist_url, matlab_file)