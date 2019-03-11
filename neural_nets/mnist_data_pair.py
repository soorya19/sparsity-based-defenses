# ==============================================================================
# Modified to read digit pairs instead of all digits
# Labels are not one-hot encoded; instead they take values in {0, 1}
# ==============================================================================
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading digit pairs from MNIST dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

from numpy.random import RandomState

# CVDF mirror of http://yann.lecun.com/exdb/mnist/
SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  # print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def extract_labels(f, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D uint8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  # print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    return labels


class DataSet(object):

  def __init__(self,
               images,
               labels,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)

    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
    if dtype == dtypes.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   dtype=dtypes.float32,
                   reshape=True,
                   digit1 = 3,
                   digit2 = 7,
                   test_ratio = 0.25,
                   validation_ratio = 0.1,
                   seed=None):

  TRAIN_IMAGES = '/train-images-idx3-ubyte.gz'
  TRAIN_LABELS = '/train-labels-idx1-ubyte.gz'
  TEST_IMAGES = '/t10k-images-idx3-ubyte.gz'
  TEST_LABELS = '/t10k-labels-idx1-ubyte.gz'

  # local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   # SOURCE_URL + TRAIN_IMAGES)
  with open(train_dir+TRAIN_IMAGES, 'rb') as f:
    train_images_all = extract_images(f)

  # local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   # SOURCE_URL + TRAIN_LABELS)
  with open(train_dir+TRAIN_LABELS, 'rb') as f:
    train_labels_all = extract_labels(f)

  # local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   # SOURCE_URL + TEST_IMAGES)
  with open(train_dir+TEST_IMAGES, 'rb') as f:
    test_images_all = extract_images(f)

  # local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   # SOURCE_URL + TEST_LABELS)
  with open(train_dir+TEST_LABELS, 'rb') as f:
    test_labels_all = extract_labels(f)

  images_all = numpy.concatenate((train_images_all, test_images_all))
  labels_all = numpy.concatenate((train_labels_all, test_labels_all))
  ind1 = numpy.where(labels_all==digit1)[0]
  ind2 = numpy.where(labels_all==digit2)[0]
  images_1 = images_all[ind1]
  images_2 = images_all[ind2]
  labels_1 = numpy.zeros([images_1.shape[0], 1])
  labels_2 = numpy.ones([images_2.shape[0], 2])

  perm1 = numpy.arange(images_1.shape[0])
  prng1 = RandomState(42)
  prng1.shuffle(perm1)
  images_1 = images_1[perm1]
  labels_1 = labels_1[perm1]

  perm2 = numpy.arange(images_2.shape[0])
  prng2 = RandomState(43)
  prng2.shuffle(perm2)
  images_2 = images_2[perm2]
  labels_2 = labels_2[perm2]

  train_size_1 = numpy.int((1-test_ratio)*(1-validation_ratio)*images_1.shape[0])
  train_size_2 = numpy.int((1-test_ratio)*(1-validation_ratio)*images_2.shape[0])
  test_size_1 = numpy.int(test_ratio*images_1.shape[0])
  test_size_2 = numpy.int(test_ratio*images_2.shape[0])

  train_images = numpy.concatenate((images_1[:train_size_1], images_2[:train_size_2]))
  train_labels = numpy.concatenate((labels_1[:train_size_1].reshape((-1,1)), labels_2[:train_size_2].reshape((-1,1))))
  validation_images = numpy.concatenate((images_1[train_size_1:-test_size_1], images_2[train_size_2:-test_size_2]))
  validation_labels = numpy.concatenate((labels_1[train_size_1:-test_size_1].reshape((-1,1)), labels_2[train_size_2:-test_size_2].reshape((-1,1))))
  test_images = numpy.concatenate((images_1[-test_size_1:], images_2[-test_size_2:]))
  test_labels = numpy.concatenate((labels_1[-test_size_1:].reshape((-1,1)), labels_2[-test_size_2:].reshape((-1,1))))

  perm3 = numpy.arange(train_images.shape[0])
  prng = RandomState(44)
  prng.shuffle(perm3)
  train_images = train_images[perm3]
  train_labels = train_labels[perm3]

  perm4 = numpy.arange(test_images.shape[0])
  prng = RandomState(45)
  prng.shuffle(perm4)
  test_images = test_images[perm4]
  test_labels = test_labels[perm4]

  perm5 = numpy.arange(validation_images.shape[0])
  prng = RandomState(46)
  prng.shuffle(perm5)
  validation_images = validation_images[perm5]
  validation_labels = validation_labels[perm5]

  options = dict(dtype=dtype, reshape=reshape, seed=seed)
  
  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)
  
  return base.Datasets(train=train, validation=validation, test=test)
