## About

Contains the older (slower) version of the code that used the [PyWavelets](https://pywavelets.readthedocs.io/) package to perform sparsification. Can be used to reproduce the ICLR workshop results:

* S. Gopalakrishnan*, Z. Marzi*, U. Madhow, R. Pedarsani, "Combating Adversarial Attacks Using Sparse Representations", in *ICLR Workshop*, April 2018. [ArXiv:1803.03880](https://arxiv.org/abs/1803.03880). (*Joint first authors.)

The new version of the code uses matrix operations instead of Pywavelet convolutions, and is parallelized across images.
