from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

from network import ThreeDimensionalR2Score


class MLPNetwork(keras.Model):
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

    @classmethod
    def load(cls, filepath, **kwargs):
        # Load the model using the Keras load method
        custom_objects = {
            'MLPNetwork': cls,
            'ThreeDimensionalR2Score': ThreeDimensionalR2Score,
        }
        model = load_model(filepath, custom_objects=custom_objects, compile=False, **kwargs)
        return model
