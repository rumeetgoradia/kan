from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout


class MLPNetwork(keras.Model):
    NAME = 'mlp'

    def __init__(self, input_shape, lookahead, num_output_features, units=64, dropout_rate=0.2):
        super(MLPNetwork, self).__init__()
        self.input_shape = input_shape
        self.lookahead = lookahead
        self.num_output_features = num_output_features

        self.flatten = keras.layers.Flatten()
        self.dense1 = Dense(units, activation='relu', kernel_initializer='he_normal')
        self.dropout = Dropout(dropout_rate)
        self.dense2 = Dense(units // 2, activation='relu', kernel_initializer='he_normal')
        self.dense3 = Dense(num_output_features * lookahead, activation='linear')
        self.reshape = keras.layers.Reshape((lookahead, num_output_features))

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.reshape(x)

    def get_config(self):
        config = super(MLPNetwork, self).get_config()
        config.update({
            'input_shape': self.input_shape,
            'lookahead': self.lookahead,
            'num_output_features': self.num_output_features,
            'units': self.dense1.units,
            'dropout_rate': self.dropout.rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
