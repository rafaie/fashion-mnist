import argparse
import os

import numpy as np
import tensorflow as tf
from util import get_data2, get_dataset

class FashionMnist:
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        (self.train_images, self.train_labels,
         self.valid_images, self.valid_labels,
         self.test_images, self.test_labels) = get_data2(data_dir, test_size=0.15, 
                                                        valid_size=0.20, need_valid=True)

        self.models=[self.create_model_0]
        self.model = None
 
    def show_dataset_size(self):
        print('-----------------------------------------------')
        print('self.train_images:', self.train_images.shape)
        print('self.train_labels:', self.train_labels.shape)
        print('self.valid_images:', self.valid_images.shape)
        print('self.valid_labels:', self.valid_labels.shape)
        print('self.test_images:', self.test_images.shape)
        print('self.test_labels:', self.test_labels.shape)

    def create_model_0(self):
        tf.reset_default_graph()

        # specify the network
        x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
        with tf.name_scope('linear_model'):
            norm  = tf.divide(x, tf.constant(255, tf.float32),
                                name='norm')
            drop = tf.layers.dropout(norm, rate=self.dropout_factor)                    
            hidden = tf.layers.dense(drop,
                                    800,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_constant),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_constant),
                                    activation=tf.nn.relu,
                                    name='hidden_layer2')
            hidden2 = tf.layers.dense(hidden,
                                    400,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_constant),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_constant),
                                    activation=tf.nn.relu,
                                    name='hidden_layer3')
            hidden3 = tf.layers.dense(hidden2,
                                    200,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_constant),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_constant),
                                    activation=tf.nn.relu,
                                    name='hidden_layer4')
    
            output = tf.layers.dense(hidden3,
                                    10,
                                    name='output')
            tf.identity(output, name='output')
        
        # define classification loss
        y = tf.placeholder(tf.float32, [None, 10], name='label')
        total = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = total + self.reg_constant * sum(reg_losses)
        return (output, total_loss, x, y)

        


    def evaluate(self, test_images, test_labels, confusion_matrix_op, total_loss, output, session,
                 x, y, batch_size, test_num_examples):
        ce_vals = []
        conf_mxs = []

        c_pred = tf.cast(tf.equal(tf.argmax(y, axis=1), 
                                    tf.argmax(output, axis=1)), tf.int32)
        c_preds = []
        for i in range(test_num_examples // batch_size):
            batch_xs = test_images[i * batch_size:(i + 1) * batch_size, :]
            batch_ys = test_labels[i * batch_size:(i + 1) * batch_size, :]
            test_ce, conf_matrix, c_pred_val= session.run(
                [tf.reduce_mean(total_loss), confusion_matrix_op, c_pred], {
                    x: batch_xs,
                    y: batch_ys
                })
            ce_vals.append(test_ce)
            conf_mxs.append(conf_matrix)
            c_preds += c_pred_val.tolist()
        return (ce_vals, conf_mxs, sum(c_preds)/len(c_preds))
        

    def update_model_info(self, model_id=0):
        self.model_path = "homework_1_" + str(model_id)
        os.makedirs(self.model_path, exist_ok=True)

        self.episode = 200
        if model_id == 1:
            self.model = self.models[0]
            self.dropout_factor = 0.30
            self.learning_rate = 0.001
            self.reg_constant = 0.01
        else:
            self.model = self.models[0]
            self.dropout_factor = 0.30
            self.learning_rate = 0.001
            self.reg_constant = 0.01



    def train(self, model_id=0):
        self.update_model_info(model_id)
        train_num_examples = self.train_images.shape[0]
        valid_num_examples = self.valid_images.shape[0]
        test_num_examples = self.test_images.shape[0]

        output, total_loss, x, y = self.model()

        confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), 
                                                  tf.argmax(output, axis=1), num_classes=10)

        # set up training and saving functionality
        global_step_tensor = tf.get_variable(
            'global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            # run training
            batch_size = self.batch_size
            ce_vals = []
            print(train_num_examples // batch_size)
            for j in range(self.episode):
                for i in range(train_num_examples // batch_size):
                    batch_xs = self.train_images[i * batch_size:(i + 1) * batch_size, :]
                    batch_ys = self.train_labels[i * batch_size:(i + 1) * batch_size, :]
                    _, train_ce = session.run(
                        [train_op, tf.reduce_mean(total_loss)], {
                            x: batch_xs,
                            y: batch_ys
                        })
                    ce_vals.append(train_ce)
                    avg_train_ce = sum(ce_vals) / len(ce_vals)

                    if i % 50 == 0 and i > 10:
                        print('epoch:', j, ',step:', i, ', TRAIN CROSS ENTROPY: ' + str(avg_train_ce))
                        # report mean validation loss
                        ce_vals, conf_mxs, acc = self.evaluate(self.valid_images, self.valid_labels, confusion_matrix_op, 
                                                                    total_loss, output, session,
                                                                    x, y, batch_size, valid_num_examples)
                        avg_test_ce = sum(ce_vals) / len(ce_vals)
                        print('VALIDATION CROSS ENTROPY: ' + str(avg_test_ce))
                        print('VALIDATION Accuracy     : ' + str(acc))

                        print(','.join(['MODEL_STATE', str(model_id), str(j), str(i), str(avg_train_ce), 
                                        str(avg_test_ce), str(acc) ]))


                if j % 5 == 0 and j > 1:
                    saver.save(
                            session,
                            os.path.join(self.model_path, "homework_1_" + str(model_id)) ,
                            global_step=global_step_tensor)
                    saver.save(
                            session,
                            os.path.join(self.model_path, "homework_1"))
                     


            # report mean test loss
            ce_vals, conf_mxs, acc = self.evaluate(self.test_images, self.test_labels, confusion_matrix_op, 
                                                   total_loss, output, session,
                                                   x, y, batch_size, test_num_examples)
            avg_test_ce = sum(ce_vals) / len(ce_vals)
            print('------------------------------')
            print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
            print('TEST Accuracy     : ' + str(acc))
            print('TEST CONFUSION MATRIX:')
            print(str(sum(conf_mxs)))

            saver.save(
                session,
                os.path.join(self.model_path, "homework_1_" + str(model_id)) ,
                global_step=global_step_tensor)
            saver.save(
                session,
                os.path.join(self.model_path, "homework_1"))

