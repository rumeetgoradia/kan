"""
rg3072
"""

import tensorflow as tf
from keras.src.layers import Flatten, Reshape
from tensorflow.keras.layers import LSTM, Dropout, Dense

from .layer import *


class TimeSeriesKANV3(tf.keras.Model):
    NAME = 'kan'
    ALLOWED_KAN_LAYERS = ['bspline', 'chebyshev', 'legendre', 'wavelet', 'fourier']

    def __init__(self, seq_length, num_features, lookahead, num_outputs, lstm_units_list, kan_units_list,
                 kan_layer_type, dropout_rate=0.2, **kwargs):
        super(TimeSeriesKANV3, self).__init__()

        self.seq_length = seq_length
        self.num_features = num_features
        self.lookahead = lookahead
        self.num_outputs = num_outputs
        self.dropout_rate = dropout_rate
        self.lstm_units_list = lstm_units_list
        self.kan_units_list = kan_units_list
        self.kan_layer_type = kan_layer_type

        # Define LSTM layers and corresponding dropout layers
        self.lstm_layers = []
        self.dropout_layers = []
        for units in lstm_units_list:
            self.lstm_layers.append(LSTM(units, return_sequences=True))
            self.dropout_layers.append(Dropout(self.dropout_rate))

        # Flatten layer
        self.flatten = Flatten()

        # Define KAN layers
        self.kan_layers = []
        input_dim = seq_length * lstm_units_list[-1]
        for units in kan_units_list:
            self.kan_layers.append(self._create_kan_layer(input_dim, units))
            input_dim = units

        # Final output layer
        self.output_layer = Dense(lookahead * num_outputs)

        # Reshape layer
        self.reshape = Reshape((lookahead, num_outputs))

    def _create_kan_layer(self, in_features, out_features):
        if self.kan_layer_type == 'bspline':
            return BSplineKANLayer(in_features=in_features, out_features=out_features)
        elif self.kan_layer_type == 'chebyshev':
            return ChebyshevKANLayer(input_dim=in_features, output_dim=out_features, degree=5)
        elif self.kan_layer_type == 'legendre':
            return LegendreKANLayer(in_features=in_features, out_features=out_features)
        elif self.kan_layer_type == 'wavelet':
            return WaveletKANLayer(in_features=in_features, out_features=out_features)
        elif self.kan_layer_type == 'fourier':
            return FourierKANLayer(inputdim=in_features, outdim=out_features)
        else:
            raise ValueError(f"Invalid KAN layer type: {self.kan_layer_type}")

    def call(self, inputs, training=False):
        x = inputs
        for lstm_layer, dropout_layer in zip(self.lstm_layers, self.dropout_layers):
            x = lstm_layer(x)
            if training:
                x = dropout_layer(x)

        # Flatten the output from LSTM layers
        x = self.flatten(x)

        for kan_layer in self.kan_layers:
            x = kan_layer(x)

        # Apply the final dense layer
        x = self.output_layer(x)

        # Reshape the output to match the target shape
        x = self.reshape(x)
        return x

    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compiled_loss(y, predictions)

            # Add regularization loss from each KAN layer
            # reg_loss = sum(layer.regularization_loss() for layer in self.kan_layers)
            # loss = loss + reg_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, predictions)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def predict_values(self, X):
        predictions = self.predict(X)
        return predictions.reshape(-1, self.lookahead, self.num_outputs)

    def get_config(self):
        config = super(TimeSeriesKANV3, self).get_config()
        config.update({
            "seq_length": self.seq_length,
            "num_features": self.num_features,
            "lookahead": self.lookahead,
            "num_outputs": self.num_outputs,
            "dropout_rate": self.dropout_rate,
            "lstm_units_list": self.lstm_units_list,
            "kan_units_list": self.kan_units_list,
            "kan_layer_type": self.kan_layer_type,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
