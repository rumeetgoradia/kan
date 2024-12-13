import tensorflow as tf
from tensorflow.keras.models import load_model
from network.kan.layer import *
from network.kan.common import *
from network import ThreeDimensionalR2Score


class TimeSeriesKANV3(tf.keras.Model):
    def __init__(self, hidden_size, lookahead, num_output_features, kan_layer, dropout_rate=0.1, **kwargs):
        super(TimeSeriesKANV3, self).__init__()

        self.hidden_size = hidden_size
        self.lookahead = lookahead
        self.num_output_features = num_output_features
        self.dropout_rate = dropout_rate

        if not isinstance(kan_layer, BaseKANLayer):
            raise TypeError("kan_layer must be an instance of BaseKANLayer")

        self.kan_layer = kan_layer

        # Simplified architecture
        self.positional_encoding = PositionalEncoding(position=1000, d_model=hidden_size)
        self.lstm_layer = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.transformer_layer = TransformerEncoderLayer(hidden_size, num_heads=4, dff=128, rate=dropout_rate)

        self.attention = MultiHeadAttentionLayer(hidden_size, num_heads=4)
        self.output_layer = tf.keras.layers.Dense(num_output_features * lookahead)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.input_projection = tf.keras.layers.Dense(hidden_size)

    def call(self, inputs, training=False):
        # Project input to hidden_size
        x = self.input_projection(inputs)

        # Apply positional encoding
        x = self.positional_encoding(x)

        # LSTM layer
        x = self.lstm_layer(x)

        # Transformer layer
        x = self.transformer_layer(x, training=training, mask=None)

        # Multi-head attention
        context_vector, _ = self.attention(x, x, x, mask=None)

        # KAN layer
        kan_out = self.dropout(self.kan_layer(context_vector[:, -1, :]))

        # Output layer
        output = self.output_layer(kan_out)

        # Reshape the output
        batch_size = tf.shape(inputs)[0]
        output = tf.reshape(output, [batch_size, self.lookahead, self.num_output_features])

        return output

    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            loss += self.kan_layer.regularization_loss()

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super(TimeSeriesKANV3, self).get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'lookahead': self.lookahead,
            'num_output_features': self.num_output_features,
            'kan_layer': tf.keras.utils.serialize_keras_object(self.kan_layer),
            'dropout_rate': self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        kan_layer_config = config.pop('kan_layer')
        kan_layer = tf.keras.utils.deserialize_keras_object(kan_layer_config)
        return cls(kan_layer=kan_layer, **config)

    def save(self, filepath, **kwargs):
        # Save the model using the Keras save method
        super(TimeSeriesKANV3, self).save(filepath, **kwargs)

    @classmethod
    def load(cls, filepath, **kwargs):
        custom_objects = {
            'TimeSeriesKAN': cls,
            'PositionalEncoding': PositionalEncoding,
            'MultiHeadAttentionLayer': MultiHeadAttentionLayer,
            'TransformerEncoderLayer': TransformerEncoderLayer,
            'BaseKANLayer': BaseKANLayer,
            'ThreeDimensionalR2Score': ThreeDimensionalR2Score,
            'silu': tf.nn.silu,
            'BSplineKANLayer': BSplineKANLayer,
            'ChebyshevKANLayer': ChebyshevKANLayer,
            'FourierKANLayer': FourierKANLayer,
            'LegendreKANLayer': LegendreKANLayer
        }
        model = load_model(filepath, custom_objects=custom_objects, compile=False, **kwargs)
        return model