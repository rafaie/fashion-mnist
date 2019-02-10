"""
util.py: it includes Helper functions

"""

import tensorflow as tf
import numpy as np
import os

tf.logging.set_verbosity(tf.logging.INFO)  # if you want to see the log info

# Load data from npy file and create np data array
def get_data(data_dir='.data/', test_size = 0.2, valid_size = 0.15, need_valid=True):
     images = np.load(os.path.join(data_dir, 'fmnist_train_data.npy'))
     labels_tmp = np.load(os.path.join(data_dir, 'fmnist_train_labels.npy'))
     labels = labels_tmp.astype(np.int64)

     s = images.shape[0]

     s1 = round(s * (1 - test_size))
     test_images = images[s1:]
     test_labels = labels[s1:]

     if need_valid is True:
          s2 = round(s1 * (1 - valid_size))
          valid_images = images[s2:s1]
          valid_labels = labels[s2:s1]
          train_images = images[:s2]
          train_labels = labels[:s2]
     else:
          valid_images = []
          valid_labels = []
          train_images = images
          train_labels = labels


     tf.logging.debug("images shape {}, images shape {}".format(images.shape, images.shape))
     tf.logging.debug("train_images shape {}, train_labels shape {}".format(train_images.shape, train_labels.shape))
     if need_valid is True:
          tf.logging.debug("valid_images shape {}, valid_labels shape {}".format(valid_images.shape, valid_labels.shape))
     tf.logging.debug("test_images shape {}, test_labels shape {}".format(test_images.shape, test_labels.shape))

     return (train_images, train_labels,
               valid_images, valid_labels,
               test_images, test_labels)


# Get np data array and create a tensorflow dataset
def get_dataset(data_dir='.data/', test_size = 0.2, valid_size = 0.15, need_valid=True):
     (train_images, train_labels,
     valid_images, valid_labels,
     test_images, test_labels) = get_data(data_dir, test_size, valid_size, need_valid)

     ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
     if need_valid is True:
          ds_valid = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels))
     else:
          ds_valid = None
     ds_test  = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

     return (ds_train, ds_valid, ds_test, len(train_labels))
     

# Load data from npy file and create np data array
def get_data2(data_dir='.data/', test_size = 0.2, valid_size = 0.15, need_valid=True):
     images = np.load(os.path.join(data_dir, 'fmnist_train_data.npy'))
     labels_tmp = np.load(os.path.join(data_dir, 'fmnist_train_labels.npy'))
     def gen_lbl(x):
          a = [0]* 10
          a[int(x)] = 1
          return np.array(a)

     labels = np.array([gen_lbl(x) for x in labels_tmp])
     # labels = labels_tmp.apply(lambda x: gen_lbl(x))

     # labels = labels_tmp.astype(np.int64)

     s = images.shape[0]

     s1 = round(s * (1 - test_size))
     test_images = images[s1:]
     test_labels = labels[s1:]

     if need_valid is True:
          s2 = round(s1 * (1 - valid_size))
          valid_images = images[s2:s1]
          valid_labels = labels[s2:s1]
          train_images = images[:s2]
          train_labels = labels[:s2]
     else:
          valid_images = []
          valid_labels = []
          train_images = images
          train_labels = labels


     tf.logging.debug("images shape {}, images shape {}".format(images.shape, images.shape))
     tf.logging.debug("train_images shape {}, train_labels shape {}".format(train_images.shape, train_labels.shape))
     if need_valid is True:
          tf.logging.debug("valid_images shape {}, valid_labels shape {}".format(valid_images.shape, valid_labels.shape))
     tf.logging.debug("test_images shape {}, test_labels shape {}".format(test_images.shape, test_labels.shape))

     return (train_images, train_labels,
               valid_images, valid_labels,
               test_images, test_labels,
               images, labels)
