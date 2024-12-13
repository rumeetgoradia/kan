import tensorflow as tf

from network.kan.layer import BaseKANLayer


class ChebyshevKANLayer(BaseKANLayer):
    def __init__(self, out_features, degree=5, scale_base=1.0, scale_cheb=1.0, in_features=None, **kwargs):
        super(ChebyshevKANLayer, self).__init__(**kwargs)
        self.out_features = out_features
        self.degree = degree
        self.scale_base = scale_base
        self.scale_cheb = scale_cheb
        self.activity_regularizer = tf.keras.regularizers.L2(1e-5)

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.in_features, self.out_features),
            initializer=tf.keras.initializers.GlorotUniform(),
            dtype=tf.float32,
            trainable=True
        )
        self.cheb_weight = self.add_weight(
            name="cheb_weight",
            shape=(self.in_features, self.out_features, self.degree),
            initializer=tf.keras.initializers.GlorotUniform(),
            dtype=tf.float32,
            trainable=True
        )

    @tf.function
    def chebyshev_basis(self, x):
        x_min = tf.reduce_min(x, axis=-1, keepdims=True)
        x_max = tf.reduce_max(x, axis=-1, keepdims=True)
        x_scaled = 2.0 * (x - x_min) / (x_max - x_min + tf.keras.backend.epsilon()) - 1.0

        T = [tf.ones_like(x_scaled), x_scaled]
        for _ in range(2, self.degree):
            T.append(2 * x_scaled * T[-1] - T[-2])
        return tf.stack(T, axis=-1)

    @tf.function
    def call(self, x):
        # Compute base output
        base_output = tf.matmul(x, self.base_weight)
        base_output = tf.nn.silu(base_output)

        # Compute Chebyshev output
        cheb_basis = self.chebyshev_basis(x)
        cheb_output = tf.einsum('bi,iod,bid->bo', x, self.cheb_weight, cheb_basis)

        output = base_output + cheb_output
        return output

    @tf.function
    def regularization_loss(self):
        # L1-like regularization for base_weight
        base_l1 = tf.reduce_sum(tf.abs(self.base_weight))

        # L1-like regularization for cheb_weight
        cheb_l1 = tf.reduce_sum(tf.abs(self.cheb_weight))

        # Combine regularization losses
        total_l1 = base_l1 + cheb_l1

        # Entropy regularization
        p_base = tf.abs(self.base_weight) / (total_l1 + tf.keras.backend.epsilon())
        p_cheb = tf.abs(self.cheb_weight) / (total_l1 + tf.keras.backend.epsilon())

        entropy_base = -tf.reduce_sum(p_base * tf.math.log(p_base + tf.keras.backend.epsilon()))
        entropy_cheb = -tf.reduce_sum(p_cheb * tf.math.log(p_cheb + tf.keras.backend.epsilon()))

        return total_l1 + entropy_base + entropy_cheb

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_features)

    def get_config(self):
        config = super(ChebyshevKANLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'degree': self.degree,
            'scale_base': self.scale_base,
            'scale_cheb': self.scale_cheb,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)