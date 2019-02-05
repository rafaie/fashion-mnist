import os
import doctest
import tensorflow as tf
from huaytools import Bunch


class BaseConfig(Bunch):

    def __init__(self, name='', **kwargs):
        """
        Examples:
            >>> config = BaseConfig('', n_batch=11, aaa="AAA")
            >>> config.ttt = 'TTT'
            >>> print(config.n_batch_train)
            64
            >>> print(config.aaa)
            AAA
            >>> print(config.ttt)
            TTT
            >>> print(config['ttt'])
            TTT
        """
        super(BaseConfig, self).__init__()

        self.name = name
        self.learning_rate = 0.001
        self.out_dir = './out_' + self.name
        os.makedirs(self.out_dir, exist_ok=True)

        self.ckpt_dir = os.path.join(self.out_dir, 'ckpt_' + self.name)
        self.eval_dir = os.path.join(self.out_dir, 'eval_' + self.name)
        self.summary_dir = os.path.join(self.out_dir, 'summary_' + self.name)  # no use

        self.n_feature = None
        self.n_class = None

        # config for `tf.data.Dataset`
        self.shuffle_buffer_size = 10000  # dataset.shuffle(buffer_size)
        self.n_batch_train = 64  # for train
        self.n_epoch_train = 30  # for train
        self.n_epoch_eval = 1
        self.n_batch_eval = 10
        self.n_epoch_pred = 1
        self.n_batch_pred = 10

        self.n_step = None  # `n_batch` and `n_epoch` decide the value of `n_step`
        self.max_steps = None  # If defined, it will add a `tf.train.StopAtStepHook`
        """How to use `max_steps`. 
            You can first train the model forever. 
            Than learn the step of over-fitting from tensorboard as `max_steps` to re-train the model
            However, if the model is very big, and have to train so long. It will be no use with this method.
        """

        # config for `tf.train.CheckpointSaverHook`
        # you can set only one of bellow
        self.save_ckpt_steps = 10  # save checkpoint every 10 steps
        self.save_ckpt_secs = None  # use save_steps

        # config for `tf.train.SummarySaverHook`
        # you can set only one of bellow
        # for train (train's SummarySaverHook is inside the MonitoredTrainingSession)
        self.save_sum_steps_train = 5
        self.save_sum_secs_train = None
        # for eval
        self.save_sum_steps_eval = self.save_sum_steps_train
        self.save_sum_secs_eval = self.save_sum_secs_train

        # config for `tf.train.LoggingTensorHook` of train
        self.log_n_steps_train = 1
        self.log_n_secs_train = None
        # config for `tf.train.LoggingTensorHook` of evaluate
        self.log_n_steps_eval = self.log_n_steps_train
        self.log_n_secs_eval = self.log_n_secs_train

        # config for `tf.Session`, ref: `tf.ConfigProto`
        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess_config.log_device_placement = True
        self.sess_config.allow_soft_placement = True

        # the `scaffold` of each Session can not be same
        # config for `tf.train.MonitoredTrainingSession`
        self.mon_sess_config_train = {'master': '',
                                      'is_chief': True,
                                      'checkpoint_dir': self.ckpt_dir,
                                      'scaffold': None,
                                      # Though there's a default `CheckpointSaverHook` inside `MonitoredTrainingSession`
                                      # but it can't set the arg of `save_checkpoint_steps`
                                      # so set this `save_checkpoint_secs` to None
                                      # then to use a outside `CheckpointSaverHook`
                                      'save_checkpoint_secs': None,
                                      'save_summaries_steps': self.save_sum_steps_train,
                                      'save_summaries_secs': self.save_sum_secs_train,
                                      'config': self.sess_config,
                                      'log_step_count_steps': self.log_n_steps_train}

        # config for the `session_creator` of evaluate `tf.train.MonitoredSession`
        self.mon_sess_config_eval = {'master': '',
                                     'scaffold': None,
                                     'config': self.sess_config,
                                     # 'checkpoint_dir': self.ckpt_dir  # use `latest_ckpt` instead
                                     }

        # config for the `session_creator` of predict `tf.train.MonitoredSession`
        self.mon_sess_config_pred = {'master': '',
                                     'scaffold': None,
                                     'config': self.sess_config,
                                     # 'checkpoint_dir': self.ckpt_dir  # use `latest_ckpt` instead
                                     }

        for k, v in kwargs.items():
            self[k] = v

        # # config for `tf.train.ChiefSessionCreator`
        # # for train
        # self.train_chief_sess_creator = {'master': '',
        #                                  'scaffold': None,
        #                                  'config': self.sess_config,
        #                                  'checkpoint_dir': self.ckpt_dir}
        # # for evaluate
        # self.eval_chief_sess_creator = {'master': '',
        #                                 'scaffold': None,
        #                                 'config': self.sess_config}

        # # config for `tf.train.Saver`
        # self.max_to_keep = 5
        # self.keep_checkpoint_every_n_hours = 10000.0  # use `tf.train.CheckpointSaverHook` replace

        # self.is_chief = True

        # self.train_scaffold = tf.train.Scaffold()
        # self.eval_scaffold = tf.train.Scaffold()

        # self.master = ''
        # self.is_chief = True
        # self.save_checkpoint_secs = None  # this work is given to the `CheckpointSaverHook`
        # self.save_summaries_steps = 5
        # self.log_step_count_steps = self.every_n_iter


if __name__ == '__main__':
    """"""
    doctest.testmod()
