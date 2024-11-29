import tensorflow as tf

from network.kan import BaseKANLayer


class TimeSeriesKAN(tf.keras.Model):
    def __init__(self, hidden_size, output_size, kan_layer, num_lstm_layers=1, lstm_kwargs=None,
                 output_activation=None):
        super(TimeSeriesKAN, self).__init__()

        # Ensure kan_layer is an instance of BaseKANLayer
        if not isinstance(kan_layer, BaseKANLayer):
            raise TypeError("kan_layer must be an instance of BaseKANLayer")

        # LSTM layer(s)
        lstm_kwargs = lstm_kwargs or {}
        if num_lstm_layers > 1:
            self.lstm = tf.keras.Sequential([tf.keras.layers.LSTM(hidden_size, return_sequences=True, **lstm_kwargs)
                                             for _ in range(num_lstm_layers - 1)
                                             ] +
                                            [tf.keras.layers.LSTM(hidden_size, return_sequences=True,
                                                                  **lstm_kwargs)])
        else:
            self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, **lstm_kwargs)

        # KAN layer
        self.kan_layer = kan_layer

        # Output layer
        self.output_layer = tf.keras.layers.Dense(output_size, activation=output_activation)

    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length, num_input_features)
        lstm_out = self.lstm(inputs)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)

        # We'll use the last output from the LSTM sequence
        last_lstm_out = lstm_out[:, -1, :]
        # last_lstm_out shape: (batch_size, hidden_size)

        kan_out = self.kan_layer(last_lstm_out)
        # kan_out shape: (batch_size, kan_size)

        output = self.output_layer(kan_out)
        # output shape: (batch_size, num_output_features)

        return output

    def regularization_loss(self):
        return self.kan_layer.regularization_loss()
