import numpy as np
import tensorflow as tf


class FourierKANLayer(tf.keras.layers.Layer):
    def __init__(self, inputdim, outdim, gridsize=300, **kwargs):
        super(FourierKANLayer, self).__init__(**kwargs)
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim
        initializer = tf.keras.initializers.RandomNormal(
            stddev=1.0 / (np.sqrt(inputdim) * np.sqrt(self.gridsize))
        )
        self.fouriercoeffs = self.add_weight(
            shape=(2, outdim, inputdim, gridsize),
            initializer=initializer,
            trainable=True,
            name='fouriercoeffs'
        )

    def call(self, inputs):
        # inputs shape: (batch_size, inputdim)
        x = inputs
        k = tf.reshape(tf.range(1, self.gridsize + 1, dtype=tf.float32), [1, 1, 1, self.gridsize])
        x_reshaped = tf.reshape(x, [x.shape[0], 1, x.shape[1], 1])
        c = tf.cos(k * x_reshaped)
        s = tf.sin(k * x_reshaped)
        c = tf.reshape(c, [1, x.shape[0], x.shape[1], self.gridsize])
        s = tf.reshape(s, [1, x.shape[0], x.shape[1], self.gridsize])
        cs_combined = tf.concat([c, s], axis=0)
        y = tf.einsum('dbik,djik->bj', cs_combined, self.fouriercoeffs)
        return y

    def get_config(self):
        config = super(FourierKANLayer, self).get_config()
        config.update({
            'inputdim': self.inputdim,
            'outdim': self.outdim,
            'gridsize': self.gridsize
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_loss = tf.reduce_mean(tf.abs(self.fouriercoeffs))
        p = tf.abs(self.fouriercoeffs) / (tf.reduce_sum(tf.abs(self.fouriercoeffs)) + 1e-10)
        entropy_loss = -tf.reduce_sum(p * tf.math.log(p + 1e-10))
        return regularize_activation * l1_loss + regularize_entropy * entropy_loss
