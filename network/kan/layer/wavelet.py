import numpy as np
import tensorflow as tf

from network.kan.layer import BaseKANLayer


class WaveletKANLayer(BaseKANLayer):

    def __init__(self, out_features, num_levels=3, scale_base=1.0, scale_wavelet=1.0, **kwargs):
        super(WaveletKANLayer, self).__init__()
        self.out_features = out_features
        self.num_levels = num_levels
        self.scale_base = scale_base
        self.scale_wavelet = scale_wavelet
        self.base_weight = None
        self.wavelet_weight = None

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.in_features, self.out_features),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_base),
            dtype=tf.float64,
            trainable=True
        )
        self.wavelet_weight = self.add_weight(
            name="wavelet_weight",
            shape=(self.in_features, self.out_features, self.num_levels),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_wavelet),
            dtype=tf.float64,
            trainable=True
        )

    def haar_wavelet_transform(self, x):
        coeffs = []
        for level in range(self.num_levels):
            # Compute approximation and detail coefficients
            approx = (x[:, ::2] + x[:, 1::2]) / np.sqrt(2)
            detail = (x[:, ::2] - x[:, 1::2]) / np.sqrt(2)
            coeffs.append(detail)
            x = approx
        coeffs.append(x)  # Add the final approximation
        return tf.stack(coeffs[::-1], axis=-1)  # Stack in reverse order

    @tf.function
    def call(self, x):
        x = tf.cast(x, tf.float64)
        base_output = tf.keras.activations.swish(x) @ self.base_weight
        wavelet_coeffs = self.haar_wavelet_transform(x)
        wavelet_output = tf.einsum('bi,iod,bid->bo', x, self.wavelet_weight, wavelet_coeffs)
        return base_output + wavelet_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = tf.reduce_mean(tf.abs(self.wavelet_weight), axis=-1)
        regularization_loss_activation = tf.reduce_sum(l1_fake)
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -tf.reduce_sum(p * tf.math.log(p + 1e-10))
        return (regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy)
