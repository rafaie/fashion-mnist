import tensorflow as tf

from .config import BaseConfig
from .mode_key import ModeKeys

tf.logging.set_verbosity(tf.logging.INFO)  # if you want to see the log info


class BaseModel(object):

    def __init__(self, config
                 # , graph=None
                 ):
        """
        Args:
            config(BaseConfig):
        """
        self._config = config
        self.ModeKeys = ModeKeys

        self._loss = None
        self._train_op = None
        self._summary_op = None

        self._hooks = None
        self._chief_only_hooks = None

        self._ops_to_run = None

        self._log_tensors = dict()

        # if graph is None:
        #     self._graph = tf.Graph()
        # else:
        #     self._graph = graph

        # self._saver = None
        # self._global_step = None
        # self._accuracy = None

        # self._metric_ops = dict()
        # self._predict_ops = []
        # self._evaluate_ops = []

        # self.features_ph = None
        # self.labels_ph = None

        # self._global_step = tf.train.create_global_step(self.graph)

        # logger.debug(self.graph.get_collection(tf.GraphKeys.GLOBAL_STEP))
        # logging.debug(self.graph.get_collection(tf.GraphKeys.GLOBAL_STEP))

        # with self.graph.as_default():
        #     self._global_step = tf.Variable(0, trainable=False, name='global_step')
        #     self._init_placeholder()
        #     self._init_model(self.features_ph, self.labels_ph)
        #     self._saver = tf.train.Saver()

    def _init_model(self, features, labels, mode):
        """
        Define the whole model.
        The Basic part you have to do:
            1.

        """
        raise NotImplementedError

    def _init_hooks(self, mode):
        """
        If you want to add some hooks when evaluate, override this function.

        Todo(huay): more easy to add hooks without override this function.
        """
        self._hooks = []

        if mode == self.ModeKeys.TRAIN:
            self._hooks += [tf.train.NanTensorHook(self._loss)]
            self._hooks += [tf.train.LoggingTensorHook(self._log_tensors,
                                                       every_n_iter=self.config.log_n_steps_train,
                                                       every_n_secs=self.config.log_n_secs_train)]

            self._chief_only_hooks = []  # only for `MonitoredTrainingSession`
            # Actually, there's also a default `CheckpointSaverHook` inside the `MonitoredTrainingSession`
            # but it can't defined the arg `save_steps`, so use a outside `CheckpointSaverHook` instead
            self._chief_only_hooks += [tf.train.CheckpointSaverHook(self.config.ckpt_dir,
                                                                    save_secs=self.config.save_ckpt_secs,
                                                                    save_steps=self.config.save_ckpt_steps,
                                                                    scaffold=self.config.mon_sess_config_train[
                                                                        'scaffold'])]

            # These hooks have defined inside the `tf.train.MonitoredTrainingSession`
            # self._hooks += [tf.train.StepCounterHook(output_dir=self.config.summary_dir,
            #                                          every_n_steps=self.config.every_n_steps),
            #                 tf.train.SummarySaverHook(  # summary_op=self._summary_op,
            #                     scaffold=self.config.train_scaffold,
            #                     save_secs=self.config.summary_save_secs,
            #                     save_steps=self.config.summary_save_steps,
            #                     output_dir=self.config.summary_dir)]

            if self.config.max_steps is not None:
                self._hooks += [tf.train.StopAtStepHook(last_step=self.config.max_steps)]

        if mode == self.ModeKeys.EVALUATE:
            self._hooks += [tf.train.LoggingTensorHook(self._log_tensors,
                                                       every_n_iter=self.config.log_n_steps_eval,
                                                       every_n_secs=self.config.log_n_secs_eval)]
            if self._summary_op is not None:
                self._hooks += [tf.train.SummarySaverHook(summary_op=self._summary_op,
                                                          save_secs=self.config.save_sum_secs_eval,
                                                          save_steps=self.config.save_sum_steps_eval,
                                                          output_dir=self.config.eval_dir)]

    def train(self, dataset):
        """
        Generally, you need not to override this function.

        Args:
            dataset(tf.data.Dataset):
        """
        assert self.config.ckpt_dir is not None, "There's no ckpt_dir."
        mode = self.ModeKeys.TRAIN

        with tf.Graph().as_default() as g:
            global_step_tensor = tf.train.create_global_step()
            self._log_tensors['step'] = global_step_tensor

            features, labels = self._load_dataset(dataset, mode)
            self.ops_to_run = self._init_model(features, labels, mode)

            # self._summary_op = tf.summary.merge_all()  # done inside the `MonitoredTrainingSession`

            tf.add_to_collection(tf.GraphKeys.LOSSES, self._loss)  # no use

            # Init the hooks; some hooks have to define after the model
            # self._init_train_hooks()
            self._init_hooks(mode)

            self._access_ops({'loss': self._loss, 'train_op': self._train_op})
            with tf.train.MonitoredTrainingSession(hooks=self._hooks, chief_only_hooks=self._chief_only_hooks,
                                                   **self.config.mon_sess_config_train) as sess:
                while not sess.should_stop():
                    sess.run(self.ops_to_run)

    def evaluate(self, dataset):
        """"""
        mode = self.ModeKeys.EVALUATE

        latest_ckpt = tf.train.latest_checkpoint(self.config.ckpt_dir)
        assert latest_ckpt is not None, "There is no trained model in {}".format(self.config.ckpt_dir)
        
        g1 = tf.Graph()
        with g1.as_default() as g:
            global_step_tensor = tf.train.create_global_step(g)

            # tf.logging.info('Evaluate the model (global_step = {})'.format(self.get_global_step()))
            # self._log_tensors['step'] = tf.constant(self.get_global_step())
            self._log_tensors['step'] = global_step_tensor

            features, labels = self._load_dataset(dataset, mode)
            self.ops_to_run = self._init_model(features, labels, mode)

            # `MonitoredSession` have no `SummarySaverHook` inside, so you have to merge all the summary manual.
            self._summary_op = tf.summary.merge_all()

            # self._init_evaluate_hooks()
            self._init_hooks(mode)

            # Note: the scaffolds of train and evaluate must be different.
            with tf.train.MonitoredSession(hooks=self._hooks,
                                           session_creator=tf.train.ChiefSessionCreator(
                                               checkpoint_filename_with_path=latest_ckpt,
                                               **self.config.mon_sess_config_eval)) as sess:
                while not sess.should_stop():
                    sess.run(self.ops_to_run)
                    # tf.logging.info("loss = {}, acc = {}".format(loss, acc))

    def predict(self, dataset):
        """"""
        mode = self.ModeKeys.PREDICT

        latest_ckpt = tf.train.latest_checkpoint(self.config.ckpt_dir)
        assert latest_ckpt is not None, "There is no trained model in {}".format(self.config.ckpt_dir)

        with tf.Graph().as_default() as g:
            global_step_tensor = tf.train.create_global_step(g)
            self._log_tensors['step'] = global_step_tensor

            # labels is None
            features, labels = self._load_dataset(dataset, mode)
            self.ops_to_run = self._init_model(features, labels, mode)

            self._init_hooks(mode)

            with tf.train.MonitoredSession(hooks=self._hooks,
                                           session_creator=tf.train.ChiefSessionCreator(
                                               checkpoint_filename_with_path=latest_ckpt,
                                               **self.config.mon_sess_config_pred)) as sess:
                while not sess.should_stop():
                    pred_ret = sess.run(self.ops_to_run)
                    if isinstance(self.ops_to_run, dict):
                        for i in range(self.config.n_batch_pred):
                            try:  # the last batch may smaller than n_batch
                                yield {k: v[i] for k, v in pred_ret.items()}
                            except IndexError:
                                break
                    elif isinstance(self.ops_to_run, list):
                        for ret in zip(*pred_ret):
                            yield ret
                    else:
                        for one_ret in pred_ret:
                            yield one_ret

    def add_log_tensor(self, key, tensor, add_to_summary=True, summary_op=tf.summary.scalar, **kwargs):
        """
        Add tensor to the log info.
        The tensor will also be added to summary for Tensorboard if `add_to_summary` is True(default).

        Examples:
            before add `loss` and `accuracy`
                INFO:tensorflow:step = 9376 (0.090 sec)
            add_log_tensor('loss', self_loss)
            add_log_tensor('acc', self._accuracy)
                INFO:tensorflow:step = 231, loss = 0.101266466, acc = 0.968 (0.300 sec)

        Args:
            key(str):
            tensor(tf.Tensor):
            add_to_summary(bool): If true, the tensor will also be add to summary
            summary_op: can be `tf.summary.scalar` or `tf.summary.histogram`
            kwargs: the rest args for summary_ops

        """
        # assert summary_op in [tf.summary.scalar, tf.summary.histogram], (
        #     "summary_op should be one of [tf.summary.scalar, tf.summary.histogram]")

        self._log_tensors[key] = tensor
        if add_to_summary:
            summary_op(key, tensor, **kwargs)

    def add_log_tensors(self, tensors_dict):
        """
        Add tensors to the log info.
        These tensors will also be added to summary with summary_op `tf.summary.scalar` for Tensorboard.
        If you don't want to add some of them to summary or want to use other summary_op, use `add_log_tensor` instead.
        """
        for key, tensor in tensors_dict.items():
            self.add_log_tensor(key, tensor)

    def get_global_step(self):
        """
        Get global_step from ckpt

        Returns:
            numpy.int32 or numpy.int64
        """
        try:
            ckpt_reader = tf.train.NewCheckpointReader(
                tf.train.latest_checkpoint(self.config.ckpt_dir))
            return ckpt_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
        except:
            return 0

    def _load_dataset(self, dataset, mode):
        """
        Apply the config to the dataset, and generate the dataset iterator.
        Both for the train/eval and predict(no labels)

        Notes:
            The graph of dataset iterator must be same as the model.

        Args:
            dataset(tf.data.Dataset):
        """
        if mode == self.ModeKeys.TRAIN:  # use `shuffle`
            dataset = dataset.shuffle(self.config.shuffle_buffer_size).batch(self.config.n_batch_train).repeat(
                self.config.n_epoch_train)
        elif mode == self.ModeKeys.EVALUATE:  # n_epoch is smaller
            dataset = dataset.batch(self.config.n_batch_eval).repeat(self.config.n_epoch_eval)
        elif mode == self.ModeKeys.PREDICT:  # n_epoch is 1
            dataset = dataset.batch(self.config.n_batch_pred).repeat(self.config.n_epoch_pred)

        ds_iter = dataset.make_one_shot_iterator()

        if mode == self.ModeKeys.PREDICT:  # No labels
            features = ds_iter.get_next()
            return features, None
        else:
            features, labels = ds_iter.get_next()
            return features, labels

    @staticmethod
    def _access_ops(ops_dict):
        """
        Access all the ops in ops_dict have been defined.

        Examples:
            self._access_ops({'loss': self._loss, 'train_op': self._train_op})
        """
        for op_name, op in ops_dict.items():
            assert op is not None, "Not define the `{}`".format(op_name)
        # assert self._loss is not None, "Not define the `loss`"
        # assert self._train_op is not None, "Not define the `train_op`"

    @property
    def config(self):
        """"""
        return self._config

    @property
    def ops_to_run(self):
        return self._ops_to_run

    @ops_to_run.setter
    def ops_to_run(self, value):
        self._ops_to_run = value

    # @property
    # def saver(self):
    #     """"""
    #     if self._saver is None:
    #         self._saver = tf.train.Saver()
    #     return self._saver

    # @property
    # def global_step(self):
    #     """"""
    #     return self._global_step

    # @property
    # def graph(self):
    #     """"""
    #     return self._graph

    # def add_metric_ops(self, key, metric_op):
    #     """"""
    #     self._metric_ops[key] = metric_op

    # def _extract_metric_update_ops(self):
    #     """"""

    # def save(self, sess, ckpt_dir=None, **kwargs):
    #     """
    #     It is useless when using the MonitoredSession, it can save variables automatically.
    #     """
    #     if ckpt_dir is None:
    #         ckpt_dir = self.config.ckpt_dir
    #         assert ckpt_dir is not None, "`ckpt_dir` is None!"
    #
    #     ckpt_prefix = os.path.join(ckpt_dir, self.config.name)
    #     tf.logging.info("Saving model to {}".format(self.config.ckpt_dir))
    #
    #     self._saver.save(sess, ckpt_prefix, tf.train.get_global_step(), **kwargs)
    #     # logger.info("Model is saved.")
    #
    # def load(self, sess, ckpt_dir=None):
    #     """
    #     It is useless when using the MonitoredSession, it can load variables automatically.
    #     """
    #     if ckpt_dir is None:
    #         ckpt_dir = self.config.ckpt_dir
    #         assert ckpt_dir is not None, "`ckpt_dir` is None!"
    #
    #     ckpt = tf.train.get_checkpoint_state(self.config.ckpt_dir)
    #
    #     if ckpt and ckpt.model_checkpoint_path:
    #         tf.logging.info("Loading the latest model from checkpoint {}".format(ckpt.model_checkpoint_path))
    #         self._saver.restore(sess, ckpt.model_checkpoint_path)
    #     else:
    #         tf.logging.info("No model checkpoint.")
    #         pass

    # def _init_placeholder(self):
    #     """
    #     Actually, this template mostly uses the `tf.data.Dataset` to pass the input to the graph
    #     The aim to define these placeholder is to init whole graph when create a instance,
    #     then you will not wait the dataset to init the graph.
    #     """
    #     # raise NotImplementedError

    # def _init_train_hooks(self):
    #     """
    #     If the hook which you want to add depends on some op from the model, you have to override this function;
    #     Otherwise, you can just add the hook to self._hooks directly by `self._hooks.append(hook)` or
    #     `self._hooks += [some hooks]`
    #     """
    #     self._hooks = []
    #     self._hooks += [tf.train.NanTensorHook(self._loss),
    #                     tf.train.LoggingTensorHook(self._log_tensors,
    #                                                every_n_iter=self.config.every_n_steps)]
    #     if self.config.max_steps is not None:
    #         self._hooks += [tf.train.StopAtStepHook(last_step=self.config.max_steps)]
    #
    #     self._chief_hooks = []
    #     # the priority of ckpt_dir of `MonitoredTrainingSession` is higher than `CheckpointSaverHook`
    #     self._chief_hooks += [tf.train.CheckpointSaverHook(self.config.ckpt_dir,
    #                                                        save_secs=self.config.save_secs,
    #                                                        save_steps=self.config.save_steps,
    #                                                        scaffold=self.config.train_scaffold)]
    #     # There has a `tf.train.SummarySaverHook` inside the MonitoredTrainingSession
    #     # if self._summary_op is not None:
    #     #     self._hooks += [tf.train.SummarySaverHook(summary_op=self._summary_op,
    #     #                                               save_secs=self.config.summary_save_secs,
    #     #                                               save_steps=self.config.summary_save_steps,
    #     #                                               output_dir=self.config.summary_dir)]

    # def _init_evaluate_hooks(self):
    #     """
    #     If you want to add some hooks when evaluate, override this function;
    #     """
    #     self._hooks = []
    #     self._hooks += [tf.train.LoggingTensorHook(self._log_tensors,
    #                                                every_n_iter=self.config.every_n_steps)]
    #     if self._summary_op is not None:
    #         self._hooks += [tf.train.SummarySaverHook(summary_op=self._summary_op,
    #                                                   save_secs=self.config.summary_save_secs,
    #                                                   save_steps=self.config.summary_save_steps,
    #                                                   output_dir=self.config.eval_dir)]

    # def _train_model(self, sess, features, labels):
    #     """
    #     The logic of train.
    #     """
    #     raise NotImplementedError

    # def _assert_mode(self, func):
    #
    #     def _func(*args, **kwargs):
    #         assert kwargs['mode'] in self.ModeKeys.keys, "mode must one of the {}".format(self.ModeKeys.keys)
    #         return func(*args, **kwargs)
    #
    #     return _func
