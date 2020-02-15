"""
Utilities for importing the CIFAR10 dataset.

Each image in the dataset is a numpy array of shape (32, 32, 3), with the values
being unsigned integers (i.e., in the range 0,1,...,255).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import sys
import tensorflow as tf
version = sys.version_info

import numpy as np
from PIL import Image
import os
import re
from sklearn.preprocessing import LabelEncoder
import platform


class GreebleData(object):
    """
    Loads greeble images from disk

    Inputs to constructor
    =====================

        - path: path to the dataset folder. 
        
        The training data must be pickled
        into five files named data_batch_i for i = 1, ..., 5, containing 10,000
        examples each, the test data
        must be pickled into a single file called test_batch containing 10,000
        examples, and the 10 class names must be
        pickled into a file called batches.meta. The pickled examples should
        be stored as a tuple of two objects: an array of 10,000 32x32x3-shaped
        arrays, and an array of their 10,000 true labels.

    """
    def __init__(self, path, greebles_mode):
        if (platform.system() == "Windows"):
            path_train = path + "\\train"
            path_test = path + "\\test"
        else:
            path_train = path + "/train"
            path_test = path + "/test"

        train_filenames = os.listdir(path_train)
        test_filenames = os.listdir(path_test)

        # Remove alpha channel from png file, just keep the first 3 channels
        if (platform.system() == "Windows"):
            train_images = np.array([np.array(Image.open(path_train + "\\" + fname))[...,:3] for fname in train_filenames])
            eval_images = np.array([np.array(Image.open(path_test + "\\" + fname))[...,:3] for fname in test_filenames])
        else:
            train_images = np.array([np.array(Image.open(path_train + "/" + fname))[...,:3] for fname in train_filenames])
            eval_images = np.array([np.array(Image.open(path_test + "/" + fname))[...,:3] for fname in test_filenames])

        '''
        File names denote the individual Greeble by defining the specific origin of the body type and parts, as well as its gender.

        The first character is the gender (m/f)

        The second number is the family (defined by body type, 1-5)
        Next there is a tilda (~) (is this referring to the dash in the filename?)

        The next few numbers describe where the parts came from in terms of the original Greebles.

        The third number is the family these particular parts ORIGINALLY came from. That is, a "2" would denote that the parts in the Greeble you are dealing with came from family 2 (1-5)

        The final number is which set of parts were taken from the specified family. Note that genders are never crossed (!), so that the number here only refers to the same gender parts as the Greeble you are dealing with. Depending on the number of individual Greebles in the original set, there could more more or less of these part sets (1-10, where 10 is the max possible as of August 2002).

        For example, "f1~16.max" is the model of a female Greeble of family 1, with body parts from family 1, set 6.
        '''

        train_labels = np.zeros(len(train_filenames), dtype='int32')
        eval_labels = np.zeros(len(test_filenames), dtype='int32')

        train_labels_temp = np.zeros(len(train_filenames), dtype=object)
        for idx, fname in enumerate(train_filenames):
            l = np.empty(greebles_mode,dtype=object)
            s = "-"
            #replace all non alphanumeric characters with nothing
            label = re.sub('[^A-Za-z0-9]+', '', fname)
            #match label structure
            matchObj = re.match( r'(f|m)([1-5]{1})([1-5]{1})(10|[1-9])', label, re.M|re.I)
            if matchObj:
                #male of female
                if(greebles_mode >=1):
                    l[0] = matchObj.group(1)
                #body type, 1-5
                if(greebles_mode >=2):
                    l[1] = matchObj.group(2)
                #original family, 1-5
                if(greebles_mode >=3):
                    l[2] = matchObj.group(3)
                #which set of parts, 1-10
                if(greebles_mode >=4):
                    l[3] = matchObj.group(4)
            else:
               raise NameError('Wrong file name structurem check the greebles documentation.')
            s = s.join(l)
            train_labels_temp[idx] = s

        eval_labels_temp = np.zeros(len(test_filenames), dtype=object)
        for idx, fname in enumerate(test_filenames):
            l = np.empty(greebles_mode,dtype=object)
            s = "-"
            #replace all non alphanumeric characters with nothing
            label = re.sub('[^A-Za-z0-9]+', '', fname)
            #match label structure
            matchObj = re.match( r'(f|m)([1-5]{1})([1-5]{1})(10|[1-9])', label, re.M|re.I)
            if matchObj:
                #male of female
                if(greebles_mode >=1):
                    l[0] = matchObj.group(1)
                #body type, 1-5
                if(greebles_mode >=2):
                    l[1] = matchObj.group(2)
                #original family, 1-5
                if(greebles_mode >=3):
                    l[2] = matchObj.group(3)
                #which set of parts, 1-10
                if(greebles_mode >=4):
                    l[3] = matchObj.group(4)
            else:
               raise NameError('Wrong file name structurem check the greebles documentation.')
            s = s.join(l)
            eval_labels_temp[idx] = s

        le = LabelEncoder()
        train_labels = np.asarray(le.fit_transform(train_labels_temp))
        eval_labels = np.asarray(le.fit_transform(eval_labels_temp))

        # label_names has the string versions of each category, change this later
        self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.train_data = Dataset(train_images, train_labels)
        self.eval_data = Dataset(eval_images, eval_labels)

class AugmentedGreebleData(object):
    """
    Data augmentation wrapper over a loaded dataset.

    Inputs to constructor
    =====================
        - raw_GreebleData: the loaded CIFAR10 dataset, via the GreebleData class
        - sess: current tensorflow session
    """
    def __init__(self, raw_GreebleData, sess):
        assert isinstance(raw_GreebleData, GreebleData)
        self.image_size = 32

        # create augmentation computational graph
        self.x_input_placeholder = tf.placeholder(tf.float32,
                                                  shape=[None, 32, 32, 3])

        # random transforamtion parameters
        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img),
                            self.x_input_placeholder)

        self.augmented = flipped

        self.train_data = AugmentedDataset(raw_GreebleData.train_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented)
        self.eval_data = AugmentedDataset(raw_GreebleData.eval_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented)
        self.label_names = raw_GreebleData.label_names


class Dataset(object):
    """
    Dataset object implementing a simple batching procedure.
    """
    def __init__(self, xs, ys):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False,
                       reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end],...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end],...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += actual_batch_size
        return batch_xs, batch_ys


class AugmentedDataset(object):
    """
    Dataset object with built-in data augmentation. When performing 
    adversarial attacks, we cannot include data augmentation as part of the
    model. If we do the adversary will try to backprop through it. 
    """
    def __init__(self, raw_datasubset, sess, x_input_placeholder,
                 augmented):
        self.sess = sess
        self.raw_datasubset = raw_datasubset
        self.x_input_placeholder = x_input_placeholder
        self.augmented = augmented

    def get_next_batch(self, batch_size, multiple_passes=False,
                       reshuffle_after_pass=True):
        raw_batch = self.raw_datasubset.get_next_batch(batch_size,
                                                       multiple_passes,
                                                       reshuffle_after_pass)
        images = raw_batch[0].astype(np.float32)
        return (self.sess.run(
                     self.augmented,
                     feed_dict={self.x_input_placeholder: raw_batch[0]}),
                raw_batch[1])
