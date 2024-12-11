import tensorflow as tf

from network.kan.layer import BaseKANLayer


class LegendreKANLayer(BaseKANLayer):
    def __init__(self, out_features, degree=5, scale_base=1.0, scale_legendre=1.0,
                 legendre_weight=None, base_weight=None, **kwargs):
        super(LegendreKANLayer, self).__init__()
        self.legendre_weight = legendre_weight
        self.base_weight = base_weight
        self.out_features = out_features
        self.degree = degree
        self.scale_base = scale_base
        self.scale_legendre = scale_legendre

    def build(self, input_shape):
        """
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

    def get_config(self):
        config = super(LegendreKANLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'degree': self.degree,
            'scale_base': self.scale_base,
            'scale_legendre': self.scale_legendre,
            'legendre_weight': self.legendre_weight,
            'base_weight': self.base_weight
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
