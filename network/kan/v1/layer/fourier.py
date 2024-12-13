import tensorflow as tf
from network.kan.layer import BaseKANLayer
import numpy as np

class FourierKANLayer(BaseKANLayer):
    def __init__(self, out_features, num_frequencies=5, scale_base=1.0, scale_fourier=1.0, in_features=None, **kwargs):
        super(FourierKANLayer, self).__init__(**kwargs)
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        self.scale_base = scale_base
        self.scale_fourier = scale_fourier
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
        self.fourier_weight = self.add_weight(
            name="fourier_weight",
            shape=(self.in_features, self.out_features, 2 * self.num_frequencies),
            initializer=tf.keras.initializers.GlorotUniform(),
            dtype=tf.float32,
            trainable=True
        )

    @tf.function
    def fourier_basis(self, x):
        x_min = tf.reduce_min(x, axis=-1, keepdims=True)
        x_max = tf.reduce_max(x, axis=-1, keepdims=True)
        x_scaled = 2 * np.pi * (x - x_min) / (x_max - x_min + tf.keras.backend.epsilon())

        frequencies = tf.range(1, self.num_frequencies + 1, dtype=tf.float32)
        sin_terms = tf.sin(frequencies * x_scaled[..., tf.newaxis])
        cos_terms = tf.cos(frequencies * x_scaled[..., tf.newaxis])

        return tf.concat([sin_terms, cos_terms], axis=-1)

    @tf.function
    def call(self, x):
        # Compute base output
        base_output = tf.matmul(x, self.base_weight)
        base_output = tf.nn.silu(base_output)

        # Compute Fourier output
        fourier_basis = self.fourier_basis(x)
        fourier_output = tf.einsum('bi,iod,bid->bo', x, self.fourier_weight, fourier_basis)

        output = base_output + fourier_output
        return output

    @tf.function
    def regularization_loss(self):
        # L1-like regularization for base_weight
        base_l1 = tf.reduce_sum(tf.abs(self.base_weight))

        # L1-like regularization for fourier_weight
        fourier_l1 = tf.reduce_sum(tf.abs(self.fourier_weight))

        # Combine regularization losses
        total_l1 = base_l1 + fourier_l1

        # Entropy regularization
        p_base = tf.abs(self.base_weight) / (total_l1 + tf.keras.backend.epsilon())
        p_fourier = tf.abs(self.fourier_weight) / (total_l1 + tf.keras.backend.epsilon())

        entropy_base = -tf.reduce_sum(p_base * tf.math.log(p_base + tf.keras.backend.epsilon()))
        entropy_fourier = -tf.reduce_sum(p_fourier * tf.math.log(p_fourier + tf.keras.backend.epsilon()))

        return total_l1 + entropy_base + entropy_fourier

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_features)

    def get_config(self):
        config = super(FourierKANLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'num_frequencies': self.num_frequencies,
            'scale_base': self.scale_base,
            'scale_fourier': self.scale_fourier,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)