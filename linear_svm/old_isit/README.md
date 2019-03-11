## About

Contains the older (slower) version of the code that used the [PyWavelets](https://pywavelets.readthedocs.io/) package to perform sparsification. Can be used to reproduce the ISIT paper results:

* Z. Marzi*, S. Gopalakrishnan*, U. Madhow, R. Pedarsani, "Sparsity-based Defense against Adversarial Attacks on Linear Classifiers", in *IEEE International Symposium on Information Theory (ISIT)*, June 2018. [ArXiv:1801.04695](https://arxiv.org/abs/1801.04695). (*Joint first authors.)

The new version of the code uses matrix operations instead of Pywavelet convolutions, and is parallelized across images.
