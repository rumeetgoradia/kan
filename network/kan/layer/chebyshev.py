import tensorflow as tf

from network.kan.layer import BaseKANLayer


class ChebyshevKANLayer(BaseKANLayer):
    def __init__(self, out_features, degree=5, scale_base=1.0, scale_cheb=1.0):
        super(ChebyshevKANLayer, self).__init__()
        self.out_features = out_features
        self.degree = degree
        self.scale_base = scale_base
        self.scale_cheb = scale_cheb

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.in_features, self.out_features),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_base),
            dtype=tf.float64,
            trainable=True
        )
        self.cheb_weight = self.add_weight(
            name="cheb_weight",
            shape=(self.in_features, self.out_features, self.degree),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_cheb),
            dtype=tf.float64,
            trainable=True
        )

    def chebyshev_basis(self, x):
        x_scaled = 2.0 * (x - tf.reduce_min(x, axis=-1, keepdims=True)) / (
                tf.reduce_max(x, axis=-1, keepdims=True) - tf.reduce_min(x, axis=-1, keepdims=True)) - 1.0
        T = [tf.ones_like(x, dtype=tf.float64), x_scaled]
        for _ in range(2, self.degree):
            T.append(2 * x_scaled * T[-1] - T[-2])
        return tf.stack(T, axis=-1)

    @tf.function
    def call(self, x):
        x = tf.cast(x, tf.float64)
        original_shape = tf.shape(x)
        x = tf.reshape(x, [-1, self.in_features])

        base_output = tf.keras.activations.swish(x) @ self.base_weight
        cheb_basis = self.chebyshev_basis(x)
        cheb_output = tf.einsum('bi,iod,bid->bo', x, self.cheb_weight, cheb_basis)
        output = base_output + cheb_output

        output = tf.reshape(output, tf.concat([original_shape[:-1], [self.out_features]], axis=0))
        return output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = tf.reduce_mean(tf.abs(self.cheb_weight), axis=-1)
        regularization_loss_activation = tf.reduce_sum(l1_fake)
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -tf.reduce_sum(p * tf.math.log(p + 1e-10))
        return (regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy)

    def get_config(self):
        config = super(ChebyshevKANLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'degree': self.degree,
            'scale_base': self.scale_base,
            'scale_cheb': self.scale_cheb
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
