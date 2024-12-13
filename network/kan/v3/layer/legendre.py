import tensorflow as tf


class LegendreKANLayer(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, degree=5, scale_base=1.0, scale_legendre=1.0, **kwargs):
        super(LegendreKANLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.scale_base = scale_base
        self.scale_legendre = scale_legendre

        # Initialize base weights for linear transformation
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(in_features, out_features),
            initializer=tf.keras.initializers.HeUniform(),
            dtype=tf.float32,
            trainable=True
        )

        # Initialize Legendre weights
        self.legendre_weight = self.add_weight(
            name="legendre_weight",
            shape=(in_features, out_features, degree + 1),
            initializer=tf.keras.initializers.HeUniform(),
            dtype=tf.float32,
            trainable=True
        )

    @tf.function
    def legendre_basis(self, x):
        # Normalize input to [-1, 1] range
        x_min = tf.reduce_min(x, axis=-1, keepdims=True)
        x_max = tf.reduce_max(x, axis=-1, keepdims=True)
        x_norm = 2.0 * (x - x_min) / (x_max - x_min + tf.keras.backend.epsilon()) - 1.0

        # Compute Legendre polynomials using recurrence relation
        legendre_polys = [tf.ones_like(x_norm), x_norm]
        for n in range(2, self.degree + 1):
            p_n = ((2 * n - 1) * x_norm * legendre_polys[-1] - (n - 1) * legendre_polys[-2]) / n
            legendre_polys.append(p_n)

        return tf.stack(legendre_polys, axis=-1)

    @tf.function
    def call(self, inputs):
        # inputs shape: (batch_size, in_features)
        x = inputs

        # Compute base output
        base_output = tf.matmul(x, self.base_weight)
        base_output = tf.nn.silu(base_output)

        # Compute Legendre output
        legendre_basis = self.legendre_basis(x)
        legendre_output = tf.einsum('bi,iod,bid->bo', x, self.legendre_weight, legendre_basis)

        # Combine base and Legendre outputs
        output = self.scale_base * base_output + self.scale_legendre * legendre_output

        return output

    @tf.function
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # L1-like regularization for base_weight
        base_l1 = tf.reduce_sum(tf.abs(self.base_weight))

        # L1-like regularization for legendre_weight
        legendre_l1 = tf.reduce_sum(tf.abs(self.legendre_weight))

        # Combine regularization losses
        total_l1 = base_l1 + legendre_l1

        # Entropy regularization
        p_base = tf.abs(self.base_weight) / (total_l1 + tf.keras.backend.epsilon())
        p_legendre = tf.abs(self.legendre_weight) / (total_l1 + tf.keras.backend.epsilon())

        entropy_base = -tf.reduce_sum(p_base * tf.math.log(p_base + tf.keras.backend.epsilon()))
        entropy_legendre = -tf.reduce_sum(p_legendre * tf.math.log(p_legendre + tf.keras.backend.epsilon()))

        return regularize_activation * total_l1 + regularize_entropy * (entropy_base + entropy_legendre)

    def get_config(self):
        config = super(LegendreKANLayer, self).get_config()
        config.update({
            'in_features': self.in_features,
            'out_features': self.out_features,
            'degree': self.degree,
            'scale_base': self.scale_base,
            'scale_legendre': self.scale_legendre,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)