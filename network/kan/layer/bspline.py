import tensorflow as tf

from network.kan.layer import BaseKANLayer


class BSplineKANLayer(BaseKANLayer):
    def __init__(
            self,
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
        super(BSplineKANLayer, self).__init__()
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

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid = tf.range(-self.spline_order, self.grid_size + self.spline_order + 1, dtype=tf.float32) * h + \
               self.grid_range[0]
        self.grid = tf.Variable(
            tf.tile(tf.expand_dims(grid, 0), [self.in_features, 1]),
            trainable=False,
            dtype=tf.float32,
            name="grid"
        )

        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.in_features, self.out_features),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_base),
            dtype=tf.float32,
            trainable=True
        )

        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(self.out_features, self.in_features, self.grid_size + self.spline_order),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_spline),
            dtype=tf.float32,
            trainable=True
        )

        if self.enable_standalone_scale_spline:
            self.spline_scaler = self.add_weight(
                name="spline_scaler",
                shape=(self.out_features, self.in_features),
                initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_spline),
                dtype=tf.float32,
                trainable=True
            )

        self.reset_parameters()

    def reset_parameters(self):
        noise = (tf.random.uniform(
            (self.grid_size + 1, self.in_features, self.out_features)) - 1 / 2) * self.scale_noise / self.grid_size

        x = tf.cast(self.grid[:, self.spline_order:-self.spline_order], tf.float32)

        coeff = self.curve2coeff(tf.transpose(x), noise)
        if not self.enable_standalone_scale_spline:
            coeff *= self.scale_spline
        self.spline_weight.assign(coeff)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32)])
    def b_splines(self, x):
        x = tf.expand_dims(x, -1)
        bases = tf.cast(
            (x >= self.grid[:, :-1]) & (x < self.grid[:, 1:]),
            dtype=tf.float32
        )
        for k in range(1, self.spline_order + 1):
            left = (x - self.grid[:, : -(k + 1)]) / (self.grid[:, k:-1] - self.grid[:, : -(k + 1)])
            right = (self.grid[:, k + 1:] - x) / (self.grid[:, k + 1:] - self.grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases

    @tf.function
    def curve2coeff(self, x, y):
        A = tf.transpose(self.b_splines(x), [1, 0, 2])
        B = tf.transpose(y, [1, 0, 2])

        A = tf.cast(A, tf.float32)
        B = tf.cast(B, tf.float32)

        solution = tf.linalg.lstsq(A, B)
        result = tf.transpose(solution, [2, 0, 1])
        return result

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * tf.expand_dims(self.spline_scaler, -1)
        return self.spline_weight

    @tf.function
    def call(self, x):
        x = tf.cast(x, tf.float32)
        original_shape = tf.shape(x)
        x = tf.reshape(x, [-1, self.in_features])

        base_output = tf.matmul(self.base_activation(x), self.base_weight)

        splines = self.b_splines(x)
        splines_reshaped = tf.reshape(splines, [tf.shape(x)[0], -1])
        scaled_spline_weight_reshaped = tf.reshape(self.scaled_spline_weight, [self.out_features, -1])

        spline_output = tf.matmul(splines_reshaped, scaled_spline_weight_reshaped, transpose_b=True)

        output = base_output + spline_output
        output = tf.reshape(output, tf.concat([original_shape[:-1], [self.out_features]], axis=0))
        return output

    @tf.function
    def update_grid(self, x, margin=0.01):
        x = tf.cast(x, tf.float32)
        batch = tf.shape(x)[0]
        splines = self.b_splines(x)
        splines = tf.transpose(splines, [1, 0, 2])
        orig_coeff = tf.transpose(self.scaled_spline_weight, [1, 2, 0])
        unreduced_spline_output = tf.matmul(splines, orig_coeff)
        unreduced_spline_output = tf.transpose(unreduced_spline_output, [1, 0, 2])
        x_sorted = tf.sort(x, axis=0)
        grid_adaptive = tf.gather(
            x_sorted,
            tf.cast(
                tf.linspace(0, tf.cast(batch - 1, tf.float32), self.grid_size + 1),
                tf.int32
            )
        )
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                tf.range(self.grid_size + 1, dtype=tf.float32)[:, tf.newaxis]
                * uniform_step
                + x_sorted[0]
                - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = tf.concat(
            [
                grid[:1]
                - uniform_step
                * tf.range(self.spline_order, 0, -1, dtype=tf.float32)[:, tf.newaxis],
                grid,
                grid[-1:]
                + uniform_step
                * tf.range(1, self.spline_order + 1, dtype=tf.float32)[:, tf.newaxis],
            ],
            axis=0
        )
        self.grid.assign(tf.transpose(grid))
        self.spline_weight.assign(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = tf.reduce_mean(tf.abs(self.spline_weight), axis=-1)
        regularization_loss_activation = tf.reduce_sum(l1_fake)
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -tf.reduce_sum(p * tf.math.log(p + 1e-10))
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )

    def get_config(self):
        config = super(BSplineKANLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'scale_noise': self.scale_noise,
            'scale_base': self.scale_base,
            'scale_spline': self.scale_spline,
            'enable_standalone_scale_spline': self.enable_standalone_scale_spline,
            'base_activation': tf.keras.activations.serialize(self.base_activation),
            'grid_eps': self.grid_eps,
            'grid_range': self.grid_range,
            'spline_scaler': self.spline_scaler,
            'spline_weight': self.spline_weight,
            'base_weight': self.base_weight,
            'grid': self.grid
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize the base_activation function
        base_activation = tf.keras.activations.deserialize(config['base_activation'])
        config['base_activation'] = base_activation
        return cls(**config)

