import tensorflow as tf

from network.kan import BaseKANLayer


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, encoder_output):
        score = self.V(tf.nn.tanh(self.W(encoder_output)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


class TimeSeriesKAN(tf.keras.Model):
    def __init__(self, hidden_size, output_size, kan_layer, num_lstm_layers=1, lstm_kwargs=None,
                 output_activation=None):
        super(TimeSeriesKAN, self).__init__()

        if not isinstance(kan_layer, BaseKANLayer):
            raise TypeError("kan_layer must be an instance of BaseKANLayer")

        lstm_kwargs = lstm_kwargs or {}
        if num_lstm_layers > 1:
            self.lstm = tf.keras.Sequential([
                                                tf.keras.layers.LSTM(hidden_size, return_sequences=True, **lstm_kwargs)
                                                for _ in range(num_lstm_layers - 1)
                                            ] + [tf.keras.layers.LSTM(hidden_size, return_sequences=True,
                                                                      **lstm_kwargs)])
        else:
            self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, **lstm_kwargs)

        self.attention = AttentionLayer(hidden_size)
        self.kan_layer = kan_layer
        self.output_layer = tf.keras.layers.Dense(output_size, activation=output_activation)

    def call(self, inputs):
        lstm_out = self.lstm(inputs)
        context_vector = self.attention(lstm_out)
        kan_out = self.kan_layer(context_vector)
        output = self.output_layer(kan_out)
        return output

    def regularization_loss(self):
        return self.kan_layer.regularization_loss()
