"""
rg3072
"""

import tensorflow as tf


class BaseKANLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BaseKANLayer, self).__init__(**kwargs)
        self.in_features = None
        self.out_features = None

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.in_features, self.out_features),
            initializer='glorot_uniform',
            trainable=True
        )
        super(BaseKANLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        raise NotImplementedError

    def regularization_loss(self):
        return 0.0

    def get_config(self):
        config = super(BaseKANLayer, self).get_config()
        config.update({
            'in_features': self.in_features,
            'out_features': self.out_features
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
