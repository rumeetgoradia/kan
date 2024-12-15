"""
rg3072

Inspired by EasyTSF (https://github.com/2448845600/EasyTSF)
"""

import math
import tensorflow as tf


class WaveletKANLayer(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat',
                 scale_base=1.0, scale_wavelet=1.0, with_bn=True, **kwargs):
        super(WaveletKANLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        self.scale_base = scale_base
        self.scale_wavelet = scale_wavelet
        self.with_bn = with_bn
        # Parameters for wavelet transformation
        self.scale = self.add_weight(shape=(out_features, in_features),
                                     initializer='ones',
                                     trainable=True,
                                     name='scale')
        self.translation = self.add_weight(shape=(out_features, in_features),
                                           initializer='zeros',
                                           trainable=True,
                                           name='translation')
        # Base weights for linear transformation
        self.base_weight = self.add_weight(shape=(in_features, out_features),
                                           initializer=tf.keras.initializers.HeUniform(),
                                           trainable=True,
                                           name='base_weight')
        self.wavelet_weights = self.add_weight(shape=(out_features, in_features),
                                               initializer=tf.keras.initializers.HeUniform(),
                                               trainable=True,
                                               name='wavelet_weights')
        self.base_activation = tf.keras.activations.swish
        # Batch normalization
        if self.with_bn:
            self.bn = tf.keras.layers.BatchNormalization()

    def wavelet_transform(self, x):
        x_expanded = tf.expand_dims(x, 1)
        translation_expanded = tf.expand_dims(self.translation, 0)
        scale_expanded = tf.expand_dims(self.scale, 0)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded
        if self.wavelet_type == 'mexican_hat':
            term1 = (tf.square(x_scaled) - 1)
            term2 = tf.exp(-0.5 * tf.square(x_scaled))
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = tf.cos(omega0 * x_scaled)
            envelope = tf.exp(-0.5 * tf.square(x_scaled))
            wavelet = envelope * real
        elif self.wavelet_type == 'dog':
            wavelet = -x_scaled * tf.exp(-0.5 * tf.square(x_scaled))
        elif self.wavelet_type == 'meyer':
            v = tf.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return tf.where(v <= 1 / 2, tf.ones_like(v),
                                tf.where(v >= 1, tf.zeros_like(v),
                                         tf.cos(pi / 2 * self.nu(2 * v - 1))))

            wavelet = tf.sin(pi * v) * meyer_aux(v)
        elif self.wavelet_type == 'shannon':
            pi = math.pi
            sinc = tf.sin(pi * x_scaled) / (pi * x_scaled + 1e-8)  # Add small epsilon to avoid division by zero
            window = tf.signal.hamming_window(tf.shape(x_scaled)[-1], periodic=False)
            wavelet = sinc * window
        else:
            raise ValueError("Unsupported wavelet type")
        wavelet_weighted = wavelet * tf.expand_dims(self.wavelet_weights, 0)
        wavelet_output = tf.reduce_sum(wavelet_weighted, axis=2)
        return wavelet_output

    def call(self, inputs):
        # inputs shape: (batch_size, in_features)
        x = inputs
        # Compute base output
        base_output = tf.matmul(x, self.base_weight)
        base_output = self.base_activation(base_output)
        # Compute wavelet output
        wavelet_output = self.wavelet_transform(x)
        # Combine base and wavelet outputs
        combined_output = self.scale_base * base_output + self.scale_wavelet * wavelet_output
        # Apply batch normalization
        if self.with_bn:
            combined_output = self.bn(combined_output)
        return combined_output

    def nu(self, t):
        return t * 4 * (35 - 84 * t + 70 * t * 2 - 20 * t ** 3)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # L1 regularization on base and wavelet weights
        base_l1 = tf.reduce_sum(tf.abs(self.base_weight))
        wavelet_l1 = tf.reduce_sum(tf.abs(self.wavelet_weights))
        total_l1 = base_l1 + wavelet_l1
        # Entropy regularization
        p_base = tf.abs(self.base_weight) / (total_l1 + 1e-10)
        p_wavelet = tf.abs(self.wavelet_weights) / (total_l1 + 1e-10)
        entropy_base = -tf.reduce_sum(p_base * tf.math.log(p_base + 1e-10))
        entropy_wavelet = -tf.reduce_sum(p_wavelet * tf.math.log(p_wavelet + 1e-10))
        return regularize_activation * total_l1 + regularize_entropy * (entropy_base + entropy_wavelet)

    def get_config(self):
        config = super(WaveletKANLayer, self).get_config()
        config.update({
            'in_features': self.in_features,
            'out_features': self.out_features,
            'wavelet_type': self.wavelet_type,
            'scale_base': self.scale_base,
            'scale_wavelet': self.scale_wavelet,
            'with_bn': self.with_bn
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_weights(self):
        return [self.scale, self.translation, self.base_weight, self.wavelet_weights]

    def set_weights(self, weights):
        if len(weights) != 4:
            raise ValueError(f"Expected 4 weight arrays, got {len(weights)}")
        if weights[0].shape != self.scale.shape:
            raise ValueError(f"Expected scale shape {self.scale.shape}, got {weights[0].shape}")
        if weights[1].shape != self.translation.shape:
            raise ValueError(f"Expected translation shape {self.translation.shape}, got {weights[1].shape}")
        if weights[2].shape != self.weight1.shape:
            raise ValueError(f"Expected weight1 shape {self.weight1.shape}, got {weights[2].shape}")
        if weights[3].shape != self.wavelet_weights.shape:
            raise ValueError(f"Expected wavelet_weights shape {self.wavelet_weights.shape}, got {weights[3].shape}")
        self.scale.assign(weights[0])
        self.translation.assign(weights[1])
        self.weight1.assign(weights[2])
        self.wavelet_weights.assign(weights[3])
