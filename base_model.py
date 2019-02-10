"""
base_model.py: it's the base model class'

"""

import tensorflow as tf
import numpy as np
import os
import functools
from util import get_data, get_dataset


tf.logging.set_verbosity(tf.logging.INFO)  # if you want to see the log info


class BaseModel(object):
    def __init__(self):
        self.config = None
        self.init_config()

    def load_dataset(self, dataset, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            if self.config.shuffle_and_repeat is True:
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


    def load_dataset2(self, features, labels, batch_size):
        """An input function for evaluation or prediction"""
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        # Return the dataset.
        return dataset

    def load_dataset3(self, train_images, train_labels):
        dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        dataset.batch(self.config.n_batch_train)
        # Return the dataset.
        return dataset



    def train_and_evaluate_without_hook(self, ds_train, ds_valid, ds_test, ds_train_size):
        # it_train = functools.partial(self.load_dataset,ds_train, tf.estimator.ModeKeys.TRAIN)
        # it_valid = functools.partial(self.load_dataset,ds_valid, tf.estimator.ModeKeys.EVAL)    
        # it_test = functools.partial(self.load_dataset,ds_valid, tf.estimator.ModeKeys.EVAL)

        (train_images, train_labels,
         valid_images, valid_labels,
         test_images, test_labels) = get_data(data_dir='data', test_size=0.2, valid_size=0.15, need_valid=True)

        my_feature_columns = []
        # for key in train_images.keys():
        #     my_feature_columns.append(tf.feature_column.numeric_column(key=key))

        cfg = tf.estimator.RunConfig()
        estimator = tf.estimator.Estimator(self.model_fn, self.config.model_dir, cfg)
        os.makedirs(estimator.eval_dir(), exist_ok=True)

        # train_spec = tf.estimator.TrainSpec(input_fn=it_train, max_steps=self.config.max_steps)
        # valid_spec = tf.estimator.EvalSpec(input_fn=it_valid)
        # test_spec = tf.estimator.EvalSpec(input_fn=it_valid)
        
        steps = round(ds_train_size/(self.config.n_batch_train * self.config.max_steps))
        for i in range(steps):
            # tf.estimator.train_and_evaluate(estimator, train_spec, valid_spec)
            estimator.train(input_fn=lambda:self.load_dataset2(train_images, train_labels, self.config.n_batch_train), 
                            steps=self.config.max_steps)

            eval_result = estimator.evaluate(input_fn=lambda:self.load_dataset2(valid_images, valid_labels, self.config.n_batch_train))


    def predict(self, ds):
        pass

    def init_config(self,):
        raise NotImplementedError

    def model_fn(self, features, labels, mode, params={}):
        raise NotImplementedError

    def get_hooks(self, estimator):
        raise NotImplementedError

