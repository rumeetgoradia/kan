import tensorflow as tf
from tensorflow.keras.models import load_model
from network.kan.layer import BaseKANLayer
from network.kan.common import *
from network import ThreeDimensionalR2Score
from network.kan.layer import *


class TimeSeriesKANV2(tf.keras.Model):
    def __init__(self, hidden_size, lookahead, num_output_features, kan_layer, num_lstm_layers=1,
                 num_transformer_layers=1, lstm_kwargs=None, dropout_rate=0.1, output_activation=None,
                 num_heads=8, dff=256, **kwargs):
        super(TimeSeriesKANV2, self).__init__()

        self.hidden_size = hidden_size
        self.lookahead = lookahead
        self.num_output_features = num_output_features
        self.num_lstm_layers = num_lstm_layers
        self.num_transformer_layers = num_transformer_layers
        self.lstm_kwargs = lstm_kwargs or {}
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.num_heads = num_heads
        self.dff = dff

        if not isinstance(kan_layer, BaseKANLayer):
            raise TypeError("kan_layer must be an instance of BaseKANLayer")

        self.kan_layer = kan_layer

        self.positional_encoding = PositionalEncoding(position=1000, d_model=hidden_size)
        self.lstm_layers = [tf.keras.layers.LSTM(hidden_size, return_sequences=True, **self.lstm_kwargs) for _ in
                            range(num_lstm_layers)]
        self.transformer_layers = [TransformerEncoderLayer(hidden_size, num_heads, dff, dropout_rate) for _ in
                                   range(num_transformer_layers)]

        self.attention = MultiHeadAttentionLayer(hidden_size, num_heads)
        self.output_layer = tf.keras.layers.Dense(num_output_features * lookahead, activation=output_activation)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.input_projection = tf.keras.layers.Dense(hidden_size)

    def call(self, inputs, training=False):
        # Project input to hidden_size
        x = self.input_projection(inputs)

        # Apply positional encoding
        x = self.positional_encoding(x)

        # LSTM layers with residual connections
        for lstm_layer in self.lstm_layers:
            lstm_out = lstm_layer(x)
            x = x + lstm_out  # Residual connection

        # Transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, training=training, mask=None)

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

    def predict_step(self, data):
        x = data
        output_sequence = []

        for _ in range(self.lookahead):
            y_pred = self(x, training=False)
            output_sequence.append(y_pred)
            x = tf.concat([x[:, 1:, :], y_pred[:, tf.newaxis, :]], axis=1)

        return tf.concat(output_sequence, axis=1)

    def get_config(self):
        config = super(TimeSeriesKANV2, self).get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'lookahead': self.lookahead,
            'num_output_features': self.num_output_features,
            'kan_layer': tf.keras.utils.serialize_keras_object(self.kan_layer),
            'num_lstm_layers': self.num_lstm_layers,
            'num_transformer_layers': self.num_transformer_layers,
            'lstm_kwargs': self.lstm_kwargs,
            'dropout_rate': self.dropout_rate,
            'output_activation': self.output_activation,
            'num_heads': self.num_heads,
            'dff': self.dff
        })
        return config

    @classmethod
    def from_config(cls, config):
        kan_layer_config = config.pop('kan_layer')
        kan_layer = tf.keras.utils.deserialize_keras_object(kan_layer_config)
        return cls(kan_layer=kan_layer, **config)

    def save(self, filepath, **kwargs):
        # Save the model using the Keras save method
        super(TimeSeriesKANV2, self).save(filepath, **kwargs)

    @classmethod
    def load(cls, filepath, **kwargs):
        # Load the model using the Keras load method
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
