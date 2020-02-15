"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import math
import os
import shutil
import sys
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import trange

import cifar10_input
import greebles_input
from eval import evaluate 
import resnet
from spatial_attack import SpatialAttack
import utilities

from math import sqrt
# import IPython

def put_kernels_on_grid (kernel, pad = 1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  # IPython.embed()
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x


def train(config):
    # seeding randomness
    tf.set_random_seed(config.training.tf_random_seed)
    np.random.seed(config.training.np_random_seed)

    # Setting up training parameters
    max_num_training_steps = config.training.max_num_training_steps
    step_size_schedule = config.training.step_size_schedule
    weight_decay = config.training.weight_decay
    momentum = config.training.momentum
    batch_size = config.training.batch_size
    adversarial_training = config.training.adversarial_training
    eval_during_training = config.training.eval_during_training
    if eval_during_training:
        num_eval_steps = config.training.num_eval_steps

    # Setting up output parameters
    num_output_steps = config.training.num_output_steps
    num_summary_steps = config.training.num_summary_steps
    num_checkpoint_steps = config.training.num_checkpoint_steps

    # Setting up the data and the model
    data_path = config.data.data_path
    # raw_cifar = cifar10_input.CIFAR10Data(data_path)
    raw_cifar = greebles_input.GreebleData(data_path, 2)
    global_step = tf.contrib.framework.get_or_create_global_step()
    model = resnet.Model(config.model)

    # uncomment to get a list of trainable variables
    # model_vars = tf.trainable_variables()
    # slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    # Setting up the optimizer
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values)
    total_loss = model.mean_xent + weight_decay * model.weight_decay_loss

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_step = optimizer.minimize( total_loss, global_step=global_step)

    # Set up adversary
    attack = SpatialAttack(model, config.attack)

    # Setting up the Tensorboard and checkpoint outputs
    model_dir = config.model.output_dir
    if eval_during_training:
        eval_dir = os.path.join(model_dir, 'eval')
        if not os.path.exists(eval_dir):
          os.makedirs(eval_dir)

    # We add accuracy and xent twice so we can easily make three types of
    # comparisons in Tensorboard:
    # - train vs eval (for a single run)
    # - train of different runs
    # - eval of different runs

    saver = tf.train.Saver(max_to_keep=3)

    tf.summary.scalar('accuracy_adv_train', model.accuracy, collections=['adv'])
    tf.summary.scalar('accuracy_adv', model.accuracy, collections=['adv'])
    tf.summary.scalar('xent_adv_train', model.xent / batch_size,
                                                        collections=['adv'])
    tf.summary.scalar('xent_adv', model.xent / batch_size, collections=['adv'])
    tf.summary.image('images_adv_train', model.x_image, collections=['adv'])
    # grid = put_kernels_on_grid (model.x_conv1)
    # tf.image.summary('conv1/kernels', grid, max_outputs=1, collections=['adv'])
    adv_summaries = tf.summary.merge_all('adv')

    tf.summary.scalar('accuracy_nat_train', model.accuracy, collections=['nat'])
    tf.summary.scalar('accuracy_nat', model.accuracy, collections = ['nat'])
    tf.summary.scalar('xent_nat_train', model.xent / batch_size,
                                                        collections=['nat'])
    tf.summary.scalar('xent_nat', model.xent / batch_size, collections=['nat'])
    tf.summary.image('images_nat_train', model.x_image, collections=['nat'])
    # grid = put_kernels_on_grid (model.x_conv1)
    # tf.image.summary('conv1/kernels', grid, max_outputs=1, collections=['nat'])
    tf.summary.scalar('learning_rate', learning_rate, collections=['nat'])
    nat_summaries = tf.summary.merge_all('nat')
    

    with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:

      # initialize data augmentation
      if config.training.data_augmentation:
          # cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess)
          cifar = greebles_input.AugmentedGreebleData(raw_cifar, sess)
      else:
          cifar = raw_cifar

      # Initialize the summary writer, global variables, and our time counter.
      summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
      if eval_during_training:
          eval_summary_writer = tf.summary.FileWriter(eval_dir)

      sess.run(tf.global_variables_initializer())
      training_time = 0.0

      # Main training loop
      for ii in range(max_num_training_steps+1):
        x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                           multiple_passes=True)

        noop_trans = np.zeros([len(x_batch), 3])
        # Compute Adversarial Perturbations
        if adversarial_training:
            start = timer()
            x_batch_adv, adv_trans = attack.perturb(x_batch, y_batch, sess)
            end = timer()
            training_time += end - start
        else:
            x_batch_adv, adv_trans = x_batch, noop_trans

        nat_dict = {model.x_input: x_batch,
                    model.y_input: y_batch,
                    model.transform: noop_trans,
                    model.is_training: False}

        adv_dict = {model.x_input: x_batch_adv,
                    model.y_input: y_batch,
                    model.transform: adv_trans,
                    model.is_training: False}

        # Output to stdout
        if ii % num_output_steps == 0:
          nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
          adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
          print('Step {}:    ({})'.format(ii, datetime.now()))
          print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
          print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
          if ii != 0:
            print('    {} examples per second'.format(
                num_output_steps * batch_size / training_time))
            training_time = 0.0

        # Tensorboard summaries
        if ii % num_summary_steps == 0:
          summary = sess.run(adv_summaries, feed_dict=adv_dict)
          summary_writer.add_summary(summary, global_step.eval(sess))
          summary = sess.run(nat_summaries, feed_dict=nat_dict)
          summary_writer.add_summary(summary, global_step.eval(sess))
          
        # Visualize conv1 kernels
        # with tf.variable_scope('input/init_conv'):
            # tf.get_variable_scope().reuse_variables()
            # weights = tf.get_variable('DW')
            # grid = put_kernels_on_grid (weights)
            # # IPython.embed()
            # summary_conv = tf.summary.image('init_conv/kernels', grid, max_outputs=1)
            # summary = sess.run(summary_conv)
            # summary_writer.add_summary(summary, global_step.eval(sess))

        # print([v.name for v in tf.trainable_variables()])
        # Desired variable is called "tower_2/filter:0".
        # var = [v for v in tf.trainable_variables() if v.name == "input/init_conv/DW:0"][0]
        
        # Write a checkpoint
        if ii % num_checkpoint_steps == 0:
          saver.save(sess,
                     os.path.join(model_dir, 'checkpoint'),
                     global_step=global_step)

        if eval_during_training and ii % num_eval_steps == 0:  
            evaluate(model, attack, sess, config, eval_summary_writer)

        # Actual training step
        start = timer()
        if adversarial_training:
            # print("\n\nADV\n\n")
            adv_dict[model.is_training] = True
            sess.run(train_step, feed_dict=adv_dict)
        else:
            # print("\n\nNAT\n\n")
            nat_dict[model.is_training] = True
            sess.run(train_step, feed_dict=nat_dict)
        end = timer()
        training_time += end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Train script options',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file',
                        default='config.json', required=False)
    args = parser.parse_args()

    config_dict = utilities.get_config(args.config)

    model_dir = config_dict['model']['output_dir']
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    # keep the configuration file with the model for reproducibility
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, sort_keys=True, indent=4)

    config = utilities.config_to_namedtuple(config_dict)
    train(config)
