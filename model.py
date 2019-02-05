"""
model.py: the sample model class

"""


import tensorflow as tf
from base.model import BaseModel, BaseConfig

tf.logging.set_verbosity(tf.logging.INFO)


class model(BaseModel):
    def _init_model(self, features, labels, mode):
        # # For tf newer, the most important is to know the in/out shape of each layer
        # # Input size:  [batch_size, 28, 28, 1]
        # # Output size: [batch_size, 28, 28, 32]
        # # The strides is (1, 1) default, so it will not change
        # # the size of image which is (28, 28) when `padding="same"`
        # conv1 = tf.layers.conv2d(features,
        #                          filters=32,
        #                          kernel_size=[5, 5],
        #                          padding="same",
        #                          activation=tf.nn.relu)

        # # In:  [batch_size, 28, 28, 32]
        # # Out: [batch_size, 14, 14, 32], here 14 = 28/strides = 28/2 = 14
        # pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

        # # In:  [batch_size, 14, 14, 32]
        # # Out: [batch_size, 14, 14, 64]
        # conv2 = tf.layers.conv2d(pool1,
        #                          filters=64,
        #                          kernel_size=[5, 5],
        #                          padding="same",
        #                          activation=tf.nn.relu)

        # # In:  [batch_size, 14, 14, 64]
        # # Out: [batch_size, 7, 7, 64], where 7 is computed same as pool1
        # pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

        # # In:  [batch_size, 7, 7, 64]
        # # Out: [batch_size, 7*7*64]
        # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        # # In:  [batch_size, 7*7*64]
        # # Out: [batch_size, 1024]
        # dense = tf.layers.dense(pool2_flat, units=1024, activation=tf.nn.relu)

        hidden = tf.layers.dense(features,
                                784,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                activation=tf.nn.relu,
                                name='hidden_layer')
        output = tf.layers.dense(hidden,
                                10,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                name='output_layer')
        tf.identity(output, name='model_output')

        # # Use dropout only when (mode == TRAIN)
        # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == self.ModeKeys.TRAIN))

        # The out of model which has not been softmax
        self._logits = output #tf.layers.dense(inputs=dropout, units=10)
        self._predictions = tf.argmax(self._logits, axis=1)

        if mode == self.ModeKeys.PREDICT:
            # return self._predictions
            return {'predictions': self._predictions,
                    'probability': tf.nn.softmax(self._logits)}

        self._loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self._logits)
        _, self._accuracy = tf.metrics.accuracy(labels, self._predictions)

        self.add_log_tensors({'loss': self._loss,
                              'acc': self._accuracy})

        if mode == self.ModeKeys.EVALUATE:
            return [self._loss, self._accuracy]

        self._train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(
            self._loss, global_step=tf.train.get_global_step())

        if mode == self.ModeKeys.TRAIN:
            return [self._train_op, self._loss, self._accuracy]

