from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout


class LSTMNetwork(keras.Model):
    NAME = 'lstm'

    def __init__(self, input_shape, lookahead, num_output_features, units=64, dropout_rate=0.2, **kwargs):
        super(LSTMNetwork, self).__init__()
        self.input_shape = input_shape
        self.lookahead = lookahead
        self.num_output_features = num_output_features

        self.lstm1 = LSTM(units, return_sequences=True)
        self.dropout1 = Dropout(dropout_rate)
        self.lstm2 = LSTM(units)
        self.dropout2 = Dropout(dropout_rate)
        self.dense = Dense(num_output_features * lookahead)
        self.reshape = keras.layers.Reshape((lookahead, num_output_features))

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.dropout1(x)
        x = self.lstm2(x)
        x = self.dropout2(x)
        x = self.dense(x)
        return self.reshape(x)

    def get_config(self):
        config = super(LSTMNetwork, self).get_config()
        config.update({
            'input_shape': self.input_shape,
            'lookahead': self.lookahead,
            'num_output_features': self.num_output_features,
            'units': self.lstm1.units,
            'dropout_rate': self.dropout1.rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

