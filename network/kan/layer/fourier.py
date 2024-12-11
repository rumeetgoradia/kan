import numpy as np
import tensorflow as tf

from network.kan.layer import BaseKANLayer


class FourierKANLayer(BaseKANLayer):
    def __init__(self, out_features, num_frequencies=5, scale_base=1.0, scale_fourier=1.0,
                 fourier_weight=None, base_weight=None, **kwargs):
        super(FourierKANLayer, self).__init__()
        self.fourier_weight = fourier_weight
        self.base_weight = base_weight
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        self.scale_base = scale_base
        self.scale_fourier = scale_fourier


    def build(self, input_shape):
        self.in_features = input_shape[-1]  # Store the number of input features
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.in_features, self.out_features),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_base),
            dtype=tf.float64,  # Ensure consistent dtype
            trainable=True
        )
        self.fourier_weight = self.add_weight(
            name="fourier_weight",
            shape=(self.in_features, self.out_features, 2 * self.num_frequencies),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_fourier),
            dtype=tf.float64,  # Ensure consistent dtype
            trainable=True
        )

    def fourier_basis(self, x):
        # Normalize x to the range [0, 2π]
        x_scaled = 2 * np.pi * (x - tf.reduce_min(x, axis=-1, keepdims=True)) / (
                tf.reduce_max(x, axis=-1, keepdims=True) - tf.reduce_min(x, axis=-1, keepdims=True))

        # Compute sine and cosine terms
        frequencies = tf.range(1, self.num_frequencies + 1, dtype=tf.float64)
        sin_terms = tf.sin(frequencies * x_scaled[..., tf.newaxis])
        cos_terms = tf.cos(frequencies * x_scaled[..., tf.newaxis])

        # Concatenate sine and cosine terms
        return tf.concat([sin_terms, cos_terms], axis=-1)

    @tf.function
    def call(self, x):
        x = tf.cast(x, tf.float64)
        base_output = tf.keras.activations.swish(x) @ self.base_weight
        fourier_basis = self.fourier_basis(x)
        fourier_output = tf.einsum('bi,iod,bid->bo', x, self.fourier_weight, fourier_basis)
        return base_output + fourier_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = tf.reduce_mean(tf.abs(self.fourier_weight), axis=-1)
        regularization_loss_activation = tf.reduce_sum(l1_fake)
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -tf.reduce_sum(p * tf.math.log(p + 1e-10))
        return (regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy)

    def get_config(self):
        config = super(FourierKANLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'num_frequencies': self.num_frequencies,
            'scale_base': self.scale_base,
            'scale_fourier': self.scale_fourier,
            'fourier_weight': self.fourier_weight,
            'base_weight': self.base_weight
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)