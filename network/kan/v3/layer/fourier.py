import numpy as np
import tensorflow as tf


class FourierKANLayer(tf.keras.layers.Layer):
    def __init__(self, inputdim, outdim, gridsize=300, scale_base=1.0, scale_fourier=1.0, **kwargs):
        super(FourierKANLayer, self).__init__(**kwargs)
        self.inputdim = inputdim
        self.outdim = outdim
        self.gridsize = gridsize
        self.scale_base = scale_base
        self.scale_fourier = scale_fourier

        # Initialize base weights for linear transformation
        self.base_weight = self.add_weight(
            shape=(inputdim, outdim),
            initializer=tf.keras.initializers.HeUniform(),
            trainable=True,
            name='base_weight'
        )

        # Initialize Fourier coefficients
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

        # Compute base output
        base_output = tf.matmul(x, self.base_weight)
        base_output = tf.nn.silu(base_output)

        # Generate k values
        k = tf.reshape(tf.range(1, self.gridsize + 1, dtype=tf.float32), [1, 1, 1, self.gridsize])

        # Reshape x for broadcasting
        x_reshaped = tf.reshape(x, [x.shape[0], 1, x.shape[1], 1])

        # Compute cosine and sine components
        c = tf.cos(k * x_reshaped)
        s = tf.sin(k * x_reshaped)

        # Reshape for einsum
        c = tf.reshape(c, [1, x.shape[0], x.shape[1], self.gridsize])
        s = tf.reshape(s, [1, x.shape[0], x.shape[1], self.gridsize])

        # Combine cosine and sine components
        cs_combined = tf.concat([c, s], axis=0)

        # Compute Fourier output
        fourier_output = tf.einsum('dbik,djik->bj', cs_combined, self.fouriercoeffs)

        # Combine base and Fourier outputs
        output = self.scale_base * base_output + self.scale_fourier * fourier_output

        return output

    def get_config(self):
        config = super(FourierKANLayer, self).get_config()
        config.update({
            'inputdim': self.inputdim,
            'outdim': self.outdim,
            'gridsize': self.gridsize,
            'scale_base': self.scale_base,
            'scale_fourier': self.scale_fourier
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # L1 regularization on base weights and Fourier coefficients
        base_l1 = tf.reduce_sum(tf.abs(self.base_weight))
        fourier_l1 = tf.reduce_sum(tf.abs(self.fouriercoeffs))
        total_l1 = base_l1 + fourier_l1

        # Entropy regularization
        p_base = tf.abs(self.base_weight) / (total_l1 + 1e-10)
        p_fourier = tf.abs(self.fouriercoeffs) / (total_l1 + 1e-10)

        entropy_base = -tf.reduce_sum(p_base * tf.math.log(p_base + 1e-10))
        entropy_fourier = -tf.reduce_sum(p_fourier * tf.math.log(p_fourier + 1e-10))

        return regularize_activation * total_l1 + regularize_entropy * (entropy_base + entropy_fourier)
