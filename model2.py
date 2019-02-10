"""
model.py: the sample model class

"""


import tensorflow as tf
from base_model import BaseModel
from base_config import BaseConfig

tf.logging.set_verbosity(tf.logging.INFO)


class model(BaseModel):

    def init_config(self):
        self.config = BaseConfig('model2')
        self.config.wit_hook = False
        self.config.n_epoch_eval = 1
        self.config.max_steps = 10
        self.config.shuffle_and_repeat = False
        self.config.n_batch_train = 25  # for train
        self.config.n_epoch_train = 25  # for train
 


    def model_fn(self, features, labels, mode, params):
        input_layer = tf.divide(features, tf.constant(255, tf.float64),
                                name='input_layer')
        hidden = tf.layers.dense(input_layer,
                                784,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                activation=tf.nn.relu,
                                name='hidden_layer')
        hidden2 = tf.layers.dense(hidden,
                                256,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                name='hidden_layer2')
        hidden3 = tf.layers.dense(hidden2,
                                256,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                name='hidden_layer3')
        output = tf.layers.dense(hidden3,
                                10,
                                # kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                # bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                name='output_layer')
        tf.identity(output, name='model_output')

        # The out of model which has not been softmax
        self._logits = output #tf.layers.dense(inputs=dropout, units=10)
        self._predictions = tf.argmax(self._logits, axis=1)

        # if mode == tf.estimator.ModeKeys.PREDICT:
        #     # return self._predictions
        #     return {'predictions': self._predictions,
        #             'probability': tf.nn.softmax(self._logits)}

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self._logits)
        metrics = {
            'acc': tf.metrics.accuracy(labels, self._predictions)
        }

        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)



