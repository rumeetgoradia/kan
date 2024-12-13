import tensorflow as tf


class ChebyshevKANLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, degree, **kwargs):
        super(ChebyshevKANLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        # Initialize Chebyshev coefficients
        self.cheby_coeffs = self.add_weight(
            shape=(input_dim, output_dim, degree + 1),
            initializer='random_normal',
            trainable=True,
            name='cheby_coeffs'
        )

    def call(self, inputs):
        # inputs shape: (batch_size, input_dim)
        x = inputs
        # Normalize x to [-1, 1] using tanh
        x = tf.tanh(x)
        # Compute Chebyshev polynomials
        cheby_polys = self.chebyshev_polynomials(x)
        # Compute the Chebyshev interpolation
        y = tf.einsum('bid,iod->bo', cheby_polys, self.cheby_coeffs)
        return y

    def chebyshev_polynomials(self, x):
        # x shape: (batch_size, input_dim)
        T = [tf.ones_like(x), x]
        for n in range(2, self.degree + 1):
            T.append(2 * x * T[-1] - T[-2])
        return tf.stack(T, axis=-1)

    def get_config(self):
        config = super(ChebyshevKANLayer, self).get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'degree': self.degree
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # L1 regularization on Chebyshev coefficients
        l1_loss = tf.reduce_mean(tf.abs(self.cheby_coeffs))

        # Entropy regularization
        p = tf.abs(self.cheby_coeffs) / (tf.reduce_sum(tf.abs(self.cheby_coeffs)) + 1e-10)
        entropy_loss = -tf.reduce_sum(p * tf.math.log(p + 1e-10))

        return regularize_activation * l1_loss + regularize_entropy * entropy_loss
