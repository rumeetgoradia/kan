import tensorflow as tf


class LegendreKANLayer(tf.keras.layers.Layer):
    def __init__(self, out_features, degree=5, scale_base=1.0, scale_legendre=1.0, in_features=None, **kwargs):
        super(LegendreKANLayer, self).__init__(**kwargs)
        self.out_features = out_features
        self.degree = degree
        self.scale_base = scale_base
        self.scale_legendre = scale_legendre
        self.in_features = in_features
        self.activity_regularizer = tf.keras.regularizers.L2(1e-5)

    def build(self, input_shape):
        self.in_features = self.in_features or input_shape[-1]
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.in_features, self.out_features),
            initializer=tf.keras.initializers.GlorotUniform(),
            dtype=tf.float32,
            trainable=True
        )
        self.legendre_weight = self.add_weight(
            name="legendre_weight", shape=(self.in_features, self.out_features, self.degree + 1),
            initializer=tf.keras.initializers.GlorotUniform(),
            dtype=tf.float32,
            trainable=True
        )

    @tf.function
    def legendre_basis(self, x):
        x_min = tf.reduce_min(x, axis=-1, keepdims=True)
        x_max = tf.reduce_max(x, axis=-1, keepdims=True)
        x_norm = 2.0 * (x - x_min) / (x_max - x_min + tf.keras.backend.epsilon()) - 1.0
        legendre_polys = [tf.ones_like(x_norm), x_norm]
        for n in range(2, self.degree + 1):
            p_n = ((2 * n - 1) * x_norm * legendre_polys[-1] - (n - 1) * legendre_polys[-2]) / n
            legendre_polys.append(p_n)
        return tf.stack(legendre_polys, axis=-1)

    @tf.function
    def call(self, inputs):
        # inputs shape: (batch_size, in_features)
        x = inputs
        base_output = tf.matmul(x, self.base_weight)
        base_output = tf.nn.silu(base_output)
        legendre_basis = self.legendre_basis(x)
        legendre_output = tf.einsum('bi,iod,bid->bo', x, self.legendre_weight, legendre_basis)
        output = self.scale_base * base_output + self.scale_legendre * legendre_output
        return output

    @tf.function
    def regularization_loss(self):
        base_l1 = tf.reduce_sum(tf.abs(self.base_weight))
        legendre_l1 = tf.reduce_sum(tf.abs(self.legendre_weight))
        total_l1 = base_l1 + legendre_l1
        p_base = tf.abs(self.base_weight) / (total_l1 + tf.keras.backend.epsilon())
        p_legendre = tf.abs(self.legendre_weight) / (total_l1 + tf.keras.backend.epsilon())
        entropy_base = -tf.reduce_sum(p_base * tf.math.log(p_base + tf.keras.backend.epsilon()))
        entropy_legendre = -tf.reduce_sum(p_legendre * tf.math.log(p_legendre + tf.keras.backend.epsilon()))
        return total_l1 + entropy_base + entropy_legendre

    def get_config(self):
        config = super(LegendreKANLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'degree': self.degree,
            'scale_base': self.scale_base,
            'scale_legendre': self.scale_legendre,
            'in_features': self.in_features,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
