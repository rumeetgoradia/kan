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
        # Ensure kan_layer is built with the correct input shape
        self.kan_layer.build((None, None, hidden_size))  # (batch_size, time_steps, hidden_size)
        self.output_layer = tf.keras.layers.Dense(output_size, activation=output_activation)
        self.reshape = None  # We'll set this in the main.py

    def call(self, inputs):
        lstm_out = self.lstm(inputs)
        context_vector = self.attention(lstm_out)
        kan_out = self.kan_layer(context_vector)
        output = self.output_layer(kan_out)
        if self.reshape is not None:
            output = self.reshape(output)
        return output

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            reg_loss = self.regularization_loss()
            total_loss = loss + reg_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {**{m.name: m.result() for m in self.metrics}, "reg_loss": reg_loss}

    def regularization_loss(self):
        return self.kan_layer.regularization_loss()
