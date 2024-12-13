import tensorflow as tf
from network.kan.layer import BaseKANLayer

class BSplineKANLayer(BaseKANLayer):
    def __init__(
            self,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=tf.nn.silu,
            grid_range=[-1, 1],
            **kwargs
    ):
        super(BSplineKANLayer, self).__init__(**kwargs)
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation
        self.grid_range = grid_range

    def build(self, input_shape):
        self.in_features = input_shape[-1]

        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.in_features, self.out_features),
            initializer=tf.keras.initializers.GlorotUniform(),
            dtype=tf.float32,
            trainable=True
        )

        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(self.out_features, self.in_features, self.grid_size + self.spline_order),
            initializer=tf.keras.initializers.GlorotUniform(),
            dtype=tf.float32,
            trainable=True
        )

    @tf.function
    def b_splines(self, x):
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid = tf.range(-self.spline_order, self.grid_size + self.spline_order + 1, dtype=tf.float32) * h + self.grid_range[0]
        grid = tf.tile(tf.expand_dims(grid, 0), [self.in_features, 1])

        x = tf.expand_dims(x, -1)
        bases = tf.cast((x >= grid[:, :-1]) & (x < grid[:, 1:]), dtype=tf.float32)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)] + tf.keras.backend.epsilon())
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k] + tf.keras.backend.epsilon())
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases

    @tf.function
    def call(self, x):
        x = tf.cast(x, tf.float32)

        # Handle input shape (batch_size, sequence_length, in_features)
        original_shape = tf.shape(x)
        if len(x.shape) == 3:
            x = tf.reshape(x, [-1, self.in_features])

        # Apply base layer
        base_output = tf.matmul(self.base_activation(x), self.base_weight)

        # Apply spline layer
        splines = self.b_splines(x)
        spline_output = tf.einsum('bi,oid,bid->bo', x, self.spline_weight, splines)

        output = base_output + spline_output

        # Reshape output if input was 3D
        if len(original_shape) == 3:
            output = tf.reshape(output, [original_shape[0], original_shape[1], self.out_features])

        return output

    @tf.function
    def regularization_loss(self):
        base_l1 = tf.reduce_sum(tf.abs(self.base_weight))
        spline_l1 = tf.reduce_sum(tf.abs(self.spline_weight))
        total_l1 = base_l1 + spline_l1

        p_base = tf.abs(self.base_weight) / (total_l1 + tf.keras.backend.epsilon())
        p_spline = tf.abs(self.spline_weight) / (total_l1 + tf.keras.backend.epsilon())

        entropy_base = -tf.reduce_sum(p_base * tf.math.log(p_base + tf.keras.backend.epsilon()))
        entropy_spline = -tf.reduce_sum(p_spline * tf.math.log(p_spline + tf.keras.backend.epsilon()))

        return total_l1 + entropy_base + entropy_spline

    def get_config(self):
        config = super(BSplineKANLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'scale_base': self.scale_base,
            'scale_spline': self.scale_spline,
            'base_activation': tf.keras.activations.serialize(self.base_activation),
            'grid_range': self.grid_range,
        })
        return config

    @classmethod
    def from_config(cls, config):
        base_activation = tf.keras.activations.deserialize(config['base_activation'])
        config['base_activation'] = base_activation
        return cls(**config)