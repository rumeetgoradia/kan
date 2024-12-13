import math
import tensorflow as tf


class WaveletKANLayer(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat', with_bn=True, **kwargs):
        super(WaveletKANLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
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
        # Weights for linear transformation
        self.weight1 = self.add_weight(shape=(out_features, in_features),
                                       initializer=tf.keras.initializers.HeUniform(),
                                       trainable=True,
                                       name='weight1')
        self.wavelet_weights = self.add_weight(shape=(out_features, in_features),
                                               initializer=tf.keras.initializers.HeUniform(),
                                               trainable=True,
                                               name='wavelet_weights')
        self.base_activation = tf.keras.activations.swish
        # Batch normalization
        if self.with_bn:
            self.bn = tf.keras.layers.BatchNormalization()

    def wavelet_transform(self, x):
        # x shape: (batch_size, in_features)
        x_expanded = tf.expand_dims(x, 1)  # (batch_size, 1, in_features)
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
        wavelet_output = self.wavelet_transform(inputs)
        base_output = tf.matmul(inputs, self.weight1, transpose_b=True)
        combined_output = wavelet_output + base_output
        # Apply batch normalization
        if self.with_bn:
            combined_output = self.bn(combined_output)
        return combined_output

    def nu(self, t):
        return t * 4 * (35 - 84 * t + 70 * t * 2 - 20 * t ** 3)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # L1 regularization on wavelet weights
        l1_loss = tf.reduce_mean(tf.abs(self.wavelet_weights))
        # Entropy regularization
        p = tf.abs(self.wavelet_weights) / (tf.reduce_sum(tf.abs(self.wavelet_weights)) + 1e-10)
        entropy_loss = -tf.reduce_sum(p * tf.math.log(p + 1e-10))
        return regularize_activation * l1_loss + regularize_entropy * entropy_loss

    def get_config(self):
        config = super(WaveletKANLayer, self).get_config()
        config.update({
            'in_features': self.in_features,
            'out_features': self.out_features,
            'wavelet_type': self.wavelet_type,
            'with_bn': self.with_bn
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
