"""
base_model.py: it's the base model class'

"""

import tensorflow as tf
import numpy as np
import os
import functools


tf.logging.set_verbosity(tf.logging.INFO)  # if you want to see the log info


class BaseModel(object):
    def __init__(self):
        self.config = None
        self.init_config()

    def load_dataset(self, dataset, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            if self.config.shuffle_buffer_size is True:
                dataset = dataset.shuffle(self.config.shuffle_buffer_size).repeat(
                    self.config.n_epoch_train).batch(self.config.n_batch_train)
            else:
                dataset = dataset.batch(self.config.n_batch_train)
        elif mode == tf.estimator.ModeKeys.EVAL:
            dataset = dataset.batch(self.config.n_batch_eval)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            dataset = dataset.batch(self.config.n_batch_pred)

        ds_iter = dataset.make_one_shot_iterator()

        return ds_iter.get_next()


    def get_hooks(self, estimator):
        hook1 = tf.contrib.estimator.stop_if_no_increase_hook(
            estimator, "f_" + self.config.name, 
            max_steps_without_increase=self.config.stop_if_no_increase_hook_max_steps_without_increase, 
            min_steps = self.config.stop_if_no_increase_hook_min_steps, 
            run_every_secs = self.config.stop_if_no_increase_hook_run_every_secs)

        return [hook1]


    def train_and_evaluate(self, ds_train, ds_valid):

        it_train = functools.partial(self.load_dataset,ds_train, tf.estimator.ModeKeys.TRAIN)
        it_valid = functools.partial(self.load_dataset,ds_valid, tf.estimator.ModeKeys.EVAL)    
        cfg = tf.estimator.RunConfig(save_checkpoints_secs=self.config.save_checkpoints_secs)
        estimator = tf.estimator.Estimator(self.model_fn, self.config.model_dir, cfg)
        hooks = self.get_hooks(estimator)
        os.makedirs(estimator.eval_dir(), exist_ok=True)
        train_spec = tf.estimator.TrainSpec(input_fn=it_train, hooks=hooks)
        eval_spec = tf.estimator.EvalSpec(input_fn=it_valid, throttle_secs=self.config.train_throttle_secs)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def train_and_evaluate(self, ds_train):


    def init_config(self,):
        raise NotImplementedError

    def model_fn(self, features, labels, mode):
        raise NotImplementedError

    