class ModeKeys(object):
    """
    Standard names for model modes.

        * `TRAIN`: training mode.
        * `EVALUATE`: evaluation mode.
        * `PREDICT`: inference mode.
        ref: https://www.tensorflow.org/versions/master/api_docs/python/tf/estimator/ModeKeys
    """

    TRAIN = 'train'
    EVALUATE = 'evaluate/test/validate'
    PREDICT = 'infer/predict'

    @property
    def keys(self):
        return [self.TRAIN, self.EVALUATE, self.PREDICT]