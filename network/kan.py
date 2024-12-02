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
        grid = tf.range(-self.spline_order, self.grid_size + self.spline_order + 1, dtype=tf.float64) * h + \
               self.grid_range[0]
        self.grid = tf.tile(tf.expand_dims(grid, 0), [self.in_features, 1])
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.in_features, self.out_features),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_base),
            dtype=tf.float64,
            trainable=True
        )
        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(self.out_features, self.in_features, self.grid_size + self.spline_order),
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
        noise = (tf.random.uniform(
            (self.grid_size + 1, self.in_features, self.out_features)) - 1 / 2) * self.scale_noise / self.grid_size

        x = tf.cast(self.grid[:, self.spline_order:-self.spline_order], tf.float64)

        coeff = self.curve2coeff(tf.transpose(x), noise)
        if not self.enable_standalone_scale_spline:
            coeff *= self.scale_spline
        self.spline_weight.assign(coeff)

    @tf.function
    def b_splines(self, x):
        x = tf.expand_dims(x, -1)
        bases = tf.cast(
            (x >= self.grid[:, :-1]) & (x < self.grid[:, 1:]),
            dtype=tf.float64
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

        A = tf.cast(A, tf.float64)
        B = tf.cast(B, tf.float64)

        solution = tf.linalg.lstsq(A, B)
        result = tf.transpose(solution, [2, 0, 1])
        return result

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * tf.expand_dims(self.spline_scaler, -1)
        return self.spline_weight

    def call(self, x):
        x = tf.cast(x, tf.float64)
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
        x = tf.cast(x, tf.float64)
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


class ChebyshevKANLayer(BaseKANLayer):
    def __init__(self, out_features, degree=5, scale_base=1.0, scale_cheb=1.0):
        super(ChebyshevKANLayer, self).__init__()
        self.in_features = None
        self.base_weight = None
        self.cheb_weight = None
        self.out_features = out_features
        self.degree = degree
        self.scale_base = scale_base
        self.scale_cheb = scale_cheb

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.in_features, self.out_features),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_base),
            dtype=tf.float64,
            trainable=True
        )
        self.cheb_weight = self.add_weight(
            name="cheb_weight",
            shape=(self.in_features, self.out_features, self.degree),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_cheb),
            dtype=tf.float64,
            trainable=True
        )

    def chebyshev_basis(self, x):
        x_scaled = 2.0 * (x - tf.reduce_min(x, axis=-1, keepdims=True)) / (
                tf.reduce_max(x, axis=-1, keepdims=True) - tf.reduce_min(x, axis=-1, keepdims=True)) - 1.0
        T = [tf.ones_like(x, dtype=tf.float64), x_scaled]
        for _ in range(2, self.degree):
            T.append(2 * x_scaled * T[-1] - T[-2])
        return tf.stack(T, axis=-1)

    @tf.function
    def call(self, x):
        x = tf.cast(x, tf.float64)
        original_shape = tf.shape(x)
        x = tf.reshape(x, [-1, self.in_features])

        base_output = tf.keras.activations.swish(x) @ self.base_weight
        cheb_basis = self.chebyshev_basis(x)
        cheb_output = tf.einsum('bi,iod,bid->bo', x, self.cheb_weight, cheb_basis)
        output = base_output + cheb_output

        output = tf.reshape(output, tf.concat([original_shape[:-1], [self.out_features]], axis=0))
        return output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = tf.reduce_mean(tf.abs(self.cheb_weight), axis=-1)
        regularization_loss_activation = tf.reduce_sum(l1_fake)
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -tf.reduce_sum(p * tf.math.log(p + 1e-10))
        return (regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy)


class FourierKANLayer(BaseKANLayer):
    def __init__(self, out_features, num_frequencies=5, scale_base=1.0, scale_fourier=1.0):
        super(FourierKANLayer, self).__init__()
        self.fourier_weight = None
        self.base_weight = None
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        self.scale_base = scale_base
        self.scale_fourier = scale_fourier
        self.in_features = None  # Add this line to store input features

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


class WaveletKANLayer(tf.keras.layers.Layer):
    """
    Wavelet Kolmogorov-Arnold Network Layer
    This layer implements a Kolmogorov-Arnold Network using Haar wavelet basis functions.
    It combines a base linear transformation with a wavelet transformation to capture
    both global and local features in the input data.
    Attributes:
        out_features (int): Number of output features.
        num_levels (int): Number of wavelet decomposition levels.
        scale_base (float): Scaling factor for base weight initialization.
        scale_wavelet (float): Scaling factor for wavelet weight initialization.
        base_weight (tf.Variable): Weights for the base linear transformation.
        wavelet_weight (tf.Variable): Weights for the wavelet transformation.
    """

    def __init__(self, out_features, num_levels=3, scale_base=1.0, scale_wavelet=1.0):
        """
        Initialize the WaveletKANLayer.
        Args:
            out_features (int): Number of output features.
            num_levels (int): Number of wavelet decomposition levels.
            scale_base (float): Scaling factor for base weight initialization.
            scale_wavelet (float): Scaling factor for wavelet weight initialization.
        """
        super(WaveletKANLayer, self).__init__()
        self.out_features = out_features
        self.num_levels = num_levels
        self.scale_base = scale_base
        self.scale_wavelet = scale_wavelet
        self.base_weight = None
        self.wavelet_weight = None

    def build(self, input_shape):
        """
        Build the layer weights.
        Args:
            input_shape (tf.TensorShape): Shape of the input tensor.
        """
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
        """
        Perform Haar wavelet transform on the input.
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            tf.Tensor: Wavelet coefficients of shape (batch_size, in_features, num_levels).
        """
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
        """
        Forward pass of the layer.
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = tf.cast(x, tf.float64)
        base_output = tf.keras.activations.swish(x) @ self.base_weight
        wavelet_coeffs = self.haar_wavelet_transform(x)
        wavelet_output = tf.einsum('bi,iod,bid->bo', x, self.wavelet_weight, wavelet_coeffs)
        return base_output + wavelet_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute regularization loss for the layer.
        Args:
            regularize_activation (float): Weight for activation regularization.
            regularize_entropy (float): Weight for entropy regularization.
        Returns:
            tf.Tensor: Regularization loss.
        """
        l1_fake = tf.reduce_mean(tf.abs(self.wavelet_weight), axis=-1)
        regularization_loss_activation = tf.reduce_sum(l1_fake)
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -tf.reduce_sum(p * tf.math.log(p + 1e-10))
        return (regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy)


class LegendreKANLayer(tf.keras.layers.Layer):
    """
    Legendre Kolmogorov-Arnold Network Layer
    This layer implements a Kolmogorov-Arnold Network using Legendre polynomial basis functions.
    It combines a base linear transformation with a Legendre polynomial transformation to capture
    both global and local features in the input data.
    Attributes:
        out_features (int): Number of output features.
        degree (int): Degree of Legendre polynomials to use.
        scale_base (float): Scaling factor for base weight initialization.
        scale_legendre (float): Scaling factor for Legendre weight initialization.
        base_weight (tf.Variable): Weights for the base linear transformation.
        legendre_weight (tf.Variable): Weights for the Legendre polynomial transformation.
    """

    def __init__(self, out_features, degree=5, scale_base=1.0, scale_legendre=1.0):
        """
        Initialize the LegendreKANLayer.
        Args:
            out_features (int): Number of output features.
            degree (int): Degree of Legendre polynomials to use.
            scale_base (float): Scaling factor for base weight initialization.
            scale_legendre (float): Scaling factor for Legendre weight initialization.
        """
        super(LegendreKANLayer, self).__init__()
        self.out_features = out_features
        self.degree = degree
        self.scale_base = scale_base
        self.scale_legendre = scale_legendre
        self.base_weight = None
        self.legendre_weight = None

    def build(self, input_shape):
        """
        Build the layer weights.
        Args:
            input_shape (tf.TensorShape): Shape of the input tensor.
        """
        self.in_features = input_shape[-1]
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.in_features, self.out_features),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_base),
            dtype=tf.float64,
            trainable=True
        )
        self.legendre_weight = self.add_weight(
            name="legendre_weight",
            shape=(self.in_features, self.out_features, self.degree + 1),
            initializer=tf.keras.initializers.VarianceScaling(scale=self.scale_legendre),
            dtype=tf.float64,
            trainable=True
        )

    def legendre_basis(self, x):
        """
        Compute Legendre polynomial basis functions.
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            tf.Tensor: Legendre polynomial values of shape (batch_size, in_features, degree + 1).
        """
        # Normalize x to the range [-1, 1]
        x_norm = 2.0 * (x - tf.reduce_min(x, axis=-1, keepdims=True)) / (
                    tf.reduce_max(x, axis=-1, keepdims=True) - tf.reduce_min(x, axis=-1, keepdims=True)) - 1.0
        legendre_polys = [tf.ones_like(x_norm), x_norm]
        for n in range(2, self.degree + 1):
            p_n = ((2 * n - 1) * x_norm * legendre_polys[-1] - (n - 1) * legendre_polys[-2]) / n
            legendre_polys.append(p_n)
        return tf.stack(legendre_polys, axis=-1)

    @tf.function
    def call(self, x):
        """
        Forward pass of the layer.
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = tf.cast(x, tf.float64)
        base_output = tf.keras.activations.swish(x) @ self.base_weight
        legendre_basis = self.legendre_basis(x)
        legendre_output = tf.einsum('bi,iod,bid->bo', x, self.legendre_weight, legendre_basis)
        return base_output + legendre_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute regularization loss for the layer.
        Args:
            regularize_activation (float): Weight for activation regularization.
            regularize_entropy (float): Weight for entropy regularization.
        Returns:
            tf.Tensor: Regularization loss.
        """
        l1_fake = tf.reduce_mean(tf.abs(self.legendre_weight), axis=-1)
        regularization_loss_activation = tf.reduce_sum(l1_fake)
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -tf.reduce_sum(p * tf.math.log(p + 1e-10))
        return (regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy)
