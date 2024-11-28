import logging
import time

import keras
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TKANCell(keras.layers.Layer):
    def __init__(self, units, input_dim, kan_layer_class, kan_params,
                 activation='tanh', recurrent_activation='sigmoid',
                 use_bias=True, dropout=0.0, recurrent_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self.kan_layer_class = kan_layer_class
        self.kan_params = kan_params
        self.activation = keras.activations.get(activation)
        self.recurrent_activation = keras.activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        self.state_size = [units, units, units]  # [h, c, kan_state]
        self.output_size = units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units * 4),
                                      name='kernel',
                                      initializer='glorot_uniform')
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4),
                                                name='recurrent_kernel',
                                                initializer='orthogonal')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer='zeros')

        self.kan_layer = self.kan_layer_class(self.units, **self.kan_params)

        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1, c_tm1, kan_state = states

        if 0 < self.dropout < 1 and training:
            inputs = keras.backend.dropout(inputs, self.dropout)

        z = keras.backend.dot(inputs, self.kernel)
        z += keras.backend.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
            z = keras.backend.bias_add(z, self.bias)

        z0, z1, z2, z3 = keras.backend.split(z, 4, axis=-1)

        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)

        h = o * self.activation(c)

        kan_output, new_kan_state = self.kan_layer(inputs, kan_state)

        output = keras.layers.concatenate([h, kan_output])
        output = keras.layers.Dense(self.units)(output)

        return output, [h, c, new_kan_state]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [keras.backend.zeros((batch_size, self.units)),
                keras.backend.zeros((batch_size, self.units)),
                keras.backend.zeros((batch_size, self.units))]


class TKAN(keras.layers.RNN):
    def __init__(self, units, input_dim, kan_layer_class, kan_params,
                 activation='tanh', recurrent_activation='sigmoid',
                 use_bias=True, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=False, return_state=False, **kwargs):
        cell = TKANCell(units, input_dim, kan_layer_class, kan_params,
                        activation, recurrent_activation, use_bias,
                        dropout, recurrent_dropout)
        super().__init__(cell, return_sequences=return_sequences,
                         return_state=return_state, **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super().call(inputs, mask=mask, training=training,
                            initial_state=initial_state)


def create_model(input_shape, units, kan_layer_class, kan_params, dropout_rate=0.2, output_dim=15):
    input_dim = input_shape[-1]  # Get the number of features from the input shape
    model = Sequential([
        TKAN(units, input_dim, kan_layer_class, kan_params, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        TKAN(units, units, kan_layer_class, kan_params, return_sequences=True),
        Dropout(dropout_rate),
        TKAN(units, units, kan_layer_class, kan_params),
        Dropout(dropout_rate),
        Dense(output_dim)
    ])
    return model


def train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs, learning_rate):
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    return history


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f'Test Results - MSE: {mse:.8f}, MAE: {mae:.8f}, R2: {r2:.8f}')

    return mse, mae, r2


def main(X_train_path: str, y_train_path: str, X_test_path: str, y_test_path: str, kan_layer_class, kan_params):
    start_time = time.time()
    logger.info("Starting the main function")

    # Load data
    logger.info("Loading data...")
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    logger.info(
        f"Data loaded. Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Split training data into train and validation
    val_size = int(0.2 * len(X_train))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    # Hyperparameters
    batch_size = 64
    epochs = 100
    learning_rate = 0.001
    units = 64
    dropout_rate = 0.2

    # Get the output dimension from y_train
    output_dim = y_train.shape[-1]

    # Create and train the model
    model = create_model(X_train.shape[1:], units, kan_layer_class, kan_params, dropout_rate, output_dim)
    logger.info(
        f"Model created with {units} units, dropout rate: {dropout_rate}, KAN layer: {kan_layer_class.__name__}, and KAN params: {kan_params}")

    history = train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs, learning_rate)

    # Evaluate the model
    mse, mae, r2 = evaluate_model(model, X_test, y_test)

    # Save metrics to file
    with open('model_metrics.txt', 'w') as f:
        f.write(f'Mean Squared Error: {mse}\n')
        f.write(f'Mean Absolute Error: {mae}\n')
        f.write(f'R-squared Score: {r2}\n')

    logger.info('Testing completed. Metrics saved to model_metrics.txt')

    # Save the model
    model.save('tkan_model.h5')
    logger.info('Model saved as tkan_model.h5')

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    from network.kan.bspline import BSplineKANLayer

    # Example usage with BSplineKANLayer
    bspline_params = {
        'grid_size': 5,
        'spline_order': 3,
        'scale_base': 1.0,
        'scale_spline': 1.0,
    }
    main('../data/prepared/sequenced/X_train.npy', '../data/prepared/sequenced/y_train.npy',
         '../data/prepared/sequenced/X_test.npy', '../data/prepared/sequenced/y_test.npy',
         BSplineKANLayer, bspline_params)

    # # Example usage with ChebyshevKANLayer
    # chebyshev_params = {
    #     'degree': 5,
    #     'scale_base': 1.0,
    #     'scale_cheb': 1.0,
    # }
    # main('../data/prepared/sequenced/X_train.npy', '../data/prepared/sequenced/y_train.npy',
    #      '../data/prepared/sequenced/X_test.npy', '../data/prepared/sequenced/y_test.npy',
    #      ChebyshevKANLayer, chebyshev_params)
