"""
rg3072

Inspired by efficient-kan (https://github.com/Blealtan/efficient-kan)
"""

import tensorflow as tf


class BSplineKANLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=tf.nn.silu,
            grid_eps=0.02,
            grid_range=[-1, 1],
            **kwargs
    ):
        super(BSplineKANLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation
        self.grid_eps = grid_eps
        self.grid_range = grid_range
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = tf.range(-spline_order, grid_size + spline_order + 1, dtype=tf.float32) * h + grid_range[0]
        self.grid = tf.Variable(tf.tile(tf.expand_dims(grid, 0), [in_features, 1]), trainable=False)
        self.base_weight = self.add_weight(
            shape=(out_features, in_features),
            initializer=tf.keras.initializers.HeUniform(seed=None),
            trainable=True,
            name='base_weight'
        )
        self.spline_weight = self.add_weight(
            shape=(out_features, in_features, grid_size + spline_order),
            initializer='random_normal',
            trainable=True,
            name='spline_weight'
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = self.add_weight(
                shape=(out_features, in_features),
                initializer=tf.keras.initializers.HeUniform(seed=None),
                trainable=True,
                name='spline_scaler'
            )

    def call(self, inputs):
        # inputs shape: (batch_size, in_features)
        x = inputs
        base_output = tf.matmul(self.base_activation(x), self.base_weight, transpose_b=True)
        spline_output = tf.matmul(
            tf.reshape(self.b_splines(x), [tf.shape(x)[0], -1]),
            tf.reshape(self.scaled_spline_weight, [self.out_features, -1]),
            transpose_b=True
        )
        output = base_output + spline_output
        return output

    def b_splines(self, x):
        x = tf.expand_dims(x, -1)
        bases = tf.cast(tf.logical_and(x >= self.grid[:, :-1], x < self.grid[:, 1:]), tf.float32)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - self.grid[:, : -(k + 1)]) / (self.grid[:, k:-1] - self.grid[:, : -(k + 1)]) * bases[:,
                                                                                                               :, :-1]
                    ) + (
                            (self.grid[:, k + 1:] - x) / (self.grid[:, k + 1:] - self.grid[:, 1:-k]) * bases[:, :, 1:]
                    )
        return bases

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * tf.expand_dims(self.spline_scaler, -1)
        return self.spline_weight

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = tf.reduce_mean(tf.abs(self.spline_weight), axis=-1)
        regularization_loss_activation = tf.reduce_sum(l1_fake)
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -tf.reduce_sum(p * tf.math.log(p + 1e-10))
        return (
                regularize_activation * regularization_loss_activation +
                regularize_entropy * regularization_loss_entropy
        )

    def get_config(self):
        config = super(BSplineKANLayer, self).get_config()
        config.update({
            'in_features': self.in_features,
            'out_features': self.out_features,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'scale_noise': self.scale_noise,
            'scale_base': self.scale_base,
            'scale_spline': self.scale_spline,
            'enable_standalone_scale_spline': self.enable_standalone_scale_spline,
            'base_activation': tf.keras.activations.serialize(self.base_activation),
            'grid_eps': self.grid_eps,
            'grid_range': self.grid_range
        })
        return config

    @classmethod
    def from_config(cls, config):
        base_activation = tf.keras.activations.deserialize(config['base_activation'])
        config['base_activation'] = base_activation
        return cls(**config)

    def get_weights(self):
        weights = [self.base_weight, self.spline_weight]
        if self.enable_standalone_scale_spline:
            weights.append(self.spline_scaler)
        return weights

    def set_weights(self, weights):
        expected_count = 3 if self.enable_standalone_scale_spline else 2
        if len(weights) != expected_count:
            raise ValueError(f"Expected {expected_count} weight arrays, got {len(weights)}")

        if weights[0].shape != self.base_weight.shape:
            raise ValueError(f"Expected base_weight shape {self.base_weight.shape}, got {weights[0].shape}")
        if weights[1].shape != self.spline_weight.shape:
            raise ValueError(f"Expected spline_weight shape {self.spline_weight.shape}, got {weights[1].shape}")

        self.base_weight.assign(weights[0])
        self.spline_weight.assign(weights[1])

        if self.enable_standalone_scale_spline:
            if weights[2].shape != self.spline_scaler.shape:
                raise ValueError(f"Expected spline_scaler shape {self.spline_scaler.shape}, got {weights[2].shape}")
            self.spline_scaler.assign(weights[2])

