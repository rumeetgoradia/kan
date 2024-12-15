import tensorflow as tf


class ChebyshevKANLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, degree, scale_base=1.0, scale_cheby=1.0, **kwargs):
        super(ChebyshevKANLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.scale_base = scale_base
        self.scale_cheby = scale_cheby
        self.base_weight = self.add_weight(
            shape=(input_dim, output_dim),
            initializer=tf.keras.initializers.HeUniform(),
            trainable=True,
            name='base_weight'
        )
        self.cheby_coeffs = self.add_weight(
            shape=(input_dim, output_dim, degree + 1),
            initializer=tf.keras.initializers.HeUniform(),
            trainable=True,
            name='cheby_coeffs'
        )

    def call(self, inputs):
        x = inputs
        # Apply base transformation
        base_output = tf.matmul(x, self.base_weight)
        base_output = tf.nn.silu(base_output)
        # Normalize x to [-1, 1] using a custom normalization
        x_min = tf.reduce_min(x, axis=-1, keepdims=True)
        x_max = tf.reduce_max(x, axis=-1, keepdims=True)
        x_norm = 2.0 * (x - x_min) / (x_max - x_min + 1e-8) - 1.0
        # Compute Chebyshev polynomials
        cheby_polys = self.chebyshev_polynomials(x_norm)
        # Compute the Chebyshev interpolation
        cheby_output = tf.einsum('bid,iod->bo', cheby_polys, self.cheby_coeffs)
        # Combine base and Chebyshev outputs
        return self.scale_base * base_output + self.scale_cheby * cheby_output

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
            'degree': self.degree,
            'scale_base': self.scale_base,
            'scale_cheby': self.scale_cheby
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

    def get_weights(self):
        return [self.base_weight, self.cheby_coeffs]

    def set_weights(self, weights):
        if len(weights) != 2:
            raise ValueError(f"Expected 2 weight arrays, got {len(weights)}")
        if weights[0].shape != self.base_weight.shape:
            raise ValueError(f"Expected base_weight shape {self.base_weight.shape}, got {weights[0].shape}")
        if weights[1].shape != self.cheby_coeffs.shape:
            raise ValueError(f"Expected cheby_coeffs shape {self.cheby_coeffs.shape}, got {weights[1].shape}")
        self.base_weight.assign(weights[0])
        self.cheby_coeffs.assign(weights[1])
