import tensorflow as tf


class ThreeDimensionalR2Score(R2Score):
    def __init__(self, name='three_dimensional_r2_score', **kwargs):
        super(ThreeDimensionalR2Score, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1, y_true.shape[-1]])
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super(ThreeDimensionalR2Score, self).get_config()
        return config
