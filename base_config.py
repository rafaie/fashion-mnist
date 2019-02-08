"""
base_config.py: it's the base config class

"""

import tensorflow as tf
import numpy as np
import os


class BaseConfig(object):
    def __init__(self, name='test_model'):
        self.name = name
        self.learning_rate = 0.001

        self.model_dir = './' + self.name
        os.makedirs(self.model_dir, exist_ok=True)
        # self.ckpt_dir = os.path.join(self.out_dir, 'ckpt_' + self.name)
        # self.eval_dir = os.path.join(self.out_dir, 'eval_' + self.name)


        # config for `tf.data.Dataset`
        self.shuffle_and_repeat = False
        self.shuffle_buffer_size = 10000  # dataset.shuffle(buffer_size)
        self.n_batch_train = 30  # for train
        self.n_epoch_train = 30  # for train
        self.n_epoch_eval = 1
        self.n_batch_eval = 10
        self.n_epoch_pred = 1
        self.n_batch_pred = 10

        # training configuration
        self.save_checkpoints_secs = 120
        self.stop_if_no_increase_hook_max_steps_without_increase = 50
        self.stop_if_no_increase_hook_min_steps = 1500
        self.stop_if_no_increase_hook_run_every_secs = 120
        self.train_throttle_secs = 120

        