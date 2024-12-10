import tensorflow as tf
from tensorflow.keras.metrics import R2Score


class ThreeDimensionalR2Score(R2Score):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1, y_true.shape[-1]])
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        return super().update_state(y_true, y_pred, sample_weight)
