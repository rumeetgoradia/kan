from typing import Type, List, Union, Dict

import tensorflow as tf
from keras.layers import Layer, RNN
from tensorflow import keras


class TKANCell(Layer):
    def __init__(
            self,
            units: int,
            kan_layer_class: Type[Layer],
            sub_kan_configs: List[Union[None, int, Dict]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kan_layer_class = kan_layer_class
        self.sub_kan_configs = sub_kan_configs or [None]

        self.lstm_layer = keras.layers.LSTMCell(units)
        self.kan_layers = []

    def build(self, input_shape):
        input_dim = input_shape[-1]

        for config in self.sub_kan_configs:
            if config is None:
                layer = self.kan_layer_class(self.units)
            elif isinstance(config, int):
                layer = self.kan_layer_class(self.units, spline_order=config)
            elif isinstance(config, dict):
                layer = self.kan_layer_class(self.units, **config)
            else:
                raise ValueError(f"Unsupported config type: {type(config)}")

            self.kan_layers.append(layer)

        self.combine_dense = keras.layers.Dense(self.units)

        self.built = True

    def call(self, inputs, states, training=None):
        h_prev, c_prev = states

        # LSTM part
        lstm_out, [h, c] = self.lstm_layer(inputs, [h_prev, c_prev])

        # KAN part
        kan_outputs = [layer(inputs) for layer in self.kan_layers]
        kan_concat = tf.concat(kan_outputs, axis=-1)

        # Combine LSTM and KAN outputs
        combined = tf.concat([lstm_out, kan_concat], axis=-1)
        output = self.combine_dense(combined)

        return output, [h, c]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.lstm_layer.get_initial_state(inputs, batch_size, dtype)


class TKAN(RNN):
    def __init__(
            self,
            units: int,
            kan_layer_class: Type[Layer],
            sub_kan_configs: List[Union[None, int, Dict]] = None,
            return_sequences: bool = False,
            return_state: bool = False,
            **kwargs
    ):
        cell = TKANCell(units, kan_layer_class, sub_kan_configs)
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            **kwargs
        )

    def call(self, inputs, initial_state=None, training=None):
        return super().call(inputs, initial_state=initial_state, training=training)
