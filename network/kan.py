import numpy as np
import tensorflow as tf


class BaseKANLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BaseKANLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.in_features, self.out_features),
            initializer='glorot_uniform',
            trainable=True
        )
        super(BaseKANLayer, self).build(input_shape)

    def call(self, inputs):
        raise NotImplementedError

    def regularization_loss(self):
        return 0.0


class ChebyshevKANLayer(BaseKANLayer):
    def __init__(self, out_features, degree=5, scale_base=1.0, scale_cheb=1.0):
        super(ChebyshevKANLayer, self).__init__()
        self.base_weight = None
        self.cheb_weight = None
        self.out_features = out_features
        self.degree = degree
        self.scale_base = scale_base
        self.scale_cheb = scale_cheb

    def build(self, input_shape):
        in_features = input_shape[-1]
        self.base_weight = self.add_weight(
            "base_weight",
            shape=(in_features, self.out_features),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_base),
            trainable=True
        )
        self.cheb_weight = self.add_weight(
            "cheb_weight",
            shape=(in_features, self.out_features, self.degree),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_cheb),
            trainable=True
        )

    def chebyshev_basis(self, x):
        x_scaled = 2.0 * (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x)) - 1.0
        T = [tf.ones_like(x_scaled), x_scaled]
        for _ in range(2, self.degree):
            T.append(2 * x_scaled * T[-1] - T[-2])
        return tf.stack(T, axis=-1)

    def call(self, x):
        base_output = tf.keras.activations.swish(x) @ self.base_weight
        cheb_basis = self.chebyshev_basis(x)
        cheb_output = tf.reduce_sum(cheb_basis[..., tf.newaxis, :] * self.cheb_weight, axis=-1)
        return base_output + cheb_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = tf.reduce_mean(tf.abs(self.cheb_weight), axis=-1)
        regularization_loss_activation = tf.reduce_sum(l1_fake)
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -tf.reduce_sum(p * tf.math.log(p + 1e-10))
        return (regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy)


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
            grid_range=None,
    ):
        if grid_range is None:
            grid_range = [-1, 1]
        super(BSplineKANLayer, self).__init__()
        self.spline_scaler = None
        self.spline_weight = None
        self.base_weight = None
        self.in_features = None
        self.grid = None
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
        print("Input shape:", input_shape)
        print("in_features:", self.in_features)

        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid = tf.range(self.grid_range[0], self.grid_range[1] + h/2, h, dtype=tf.float32)
        self.grid = tf.expand_dims(grid, 0)  # Shape: [1, grid_size + 1]
        print("Grid shape:", tf.shape(self.grid))

        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.in_features, self.out_features),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_base),
            trainable=True
        )
        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(self.out_features, self.in_features, self.grid_size),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_spline),
            trainable=True
        )
        if self.enable_standalone_scale_spline:
            self.spline_scaler = self.add_weight(
                name="spline_scaler",
                shape=(self.out_features, self.in_features),
                initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_spline),
                trainable=True
            )

        self.reset_parameters()

    def reset_parameters(self):
        noise = (tf.random.uniform((self.grid_size, self.in_features, self.out_features))
                 - 1 / 2) * self.scale_noise / self.grid_size

        x = self.grid[0, :-1]  # Use all grid points except the last one
        x = tf.tile(tf.expand_dims(x, 0), [self.in_features, 1])  # Shape: (in_features, grid_size)

        coeff = self.curve2coeff(x, noise)

        if not self.enable_standalone_scale_spline:
            coeff *= self.scale_spline

        self.spline_weight.assign(coeff)

    @tf.function
    def b_splines(self, x):
        x = tf.expand_dims(x, -1)  # Shape: (in_features, grid_size, 1)
        grid = tf.tile(self.grid, [tf.shape(x)[0], 1])  # Shape: (in_features, grid_size + 1)
        grid = tf.expand_dims(grid, -1)  # Shape: (in_features, grid_size + 1, 1)

        bases = tf.cast(
            (x >= grid[:, :-1, :]) & (x < grid[:, 1:, :]),
            dtype=tf.float32
        )

        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1), :]) / (grid[:, k:-1, :] - grid[:, : -(k + 1), :])
            right = (grid[:, k + 1:, :] - x) / (grid[:, k + 1:, :] - grid[:, 1:-k, :])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]

        return bases  # Shape: (in_features, grid_size, spline_order)

    @tf.function
    def curve2coeff(self, x, y):
        A = self.b_splines(x)  # Shape: (in_features, grid_size, spline_order)

        # Reshape A to 2D: (in_features * grid_size, spline_order)
        A_reshaped = tf.reshape(A, [-1, tf.shape(A)[-1]])

        # Reshape y to 2D: (in_features * grid_size, out_features)
        y_reshaped = tf.reshape(tf.transpose(y, [1, 0, 2]), [-1, tf.shape(y)[-1]])

        # Perform least squares
        solution = tf.linalg.lstsq(A_reshaped, y_reshaped)

        # Reshape the solution back to 3D: (out_features, in_features, grid_size)
        result = tf.reshape(tf.transpose(solution),
                            [tf.shape(y)[-1], tf.shape(x)[0], tf.shape(x)[1]])

        return result

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * tf.expand_dims(self.spline_scaler, -1)
        return self.spline_weight

    def call(self, x):
        original_shape = tf.shape(x)
        x = tf.reshape(x, [-1, self.in_features])

        base_output = tf.matmul(self.base_activation(x), self.base_weight)
        spline_output = tf.matmul(
            tf.reshape(self.b_splines(x), [tf.shape(x)[0], -1]),
            tf.reshape(self.scaled_spline_weight, [self.out_features, -1]),
            transpose_b=True
        )
        output = base_output + spline_output

        output = tf.reshape(output, tf.concat([original_shape[:-1], [self.out_features]], axis=0))
        return output

    @tf.function
    def update_grid(self, x, margin=0.01):
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


class FourierKANLayer(BaseKANLayer):
    def __init__(self, out_features, num_frequencies=5, scale_base=1.0, scale_fourier=1.0):
        super(FourierKANLayer, self).__init__()
        self.fourier_weight = None
        self.base_weight = None
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        self.scale_base = scale_base
        self.scale_fourier = scale_fourier

    def build(self, input_shape):
        in_features = input_shape[-1]
        self.base_weight = self.add_weight(
            "base_weight",
            shape=(in_features, self.out_features),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_base),
            trainable=True
        )
        self.fourier_weight = self.add_weight(
            "fourier_weight",
            shape=(in_features, self.out_features, 2 * self.num_frequencies),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_fourier),
            trainable=True
        )

    def fourier_basis(self, x):
        # Normalize x to the range [0, 2π]
        x_scaled = 2 * np.pi * (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))

        # Compute sine and cosine terms
        frequencies = tf.range(1, self.num_frequencies + 1, dtype=tf.float32)
        sin_terms = tf.sin(frequencies * x_scaled[..., tf.newaxis])
        cos_terms = tf.cos(frequencies * x_scaled[..., tf.newaxis])

        # Concatenate sine and cosine terms
        return tf.concat([sin_terms, cos_terms], axis=-1)

    def call(self, x):
        base_output = tf.keras.activations.swish(x) @ self.base_weight
        fourier_basis = self.fourier_basis(x)
        fourier_output = tf.reduce_sum(fourier_basis[..., tf.newaxis, :] * self.fourier_weight, axis=-1)
        return base_output + fourier_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = tf.reduce_mean(tf.abs(self.fourier_weight), axis=-1)
        regularization_loss_activation = tf.reduce_sum(l1_fake)
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -tf.reduce_sum(p * tf.math.log(p + 1e-10))
        return (regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy)
