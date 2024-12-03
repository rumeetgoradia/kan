import argparse
import json
import logging
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.metrics import R2Score, MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError
from tqdm import tqdm
import time

from data.processor import prepare_data, get_ticker_split
from data.window_generator import WindowGenerator
from network.kan import ChebyshevKANLayer, BSplineKANLayer, FourierKANLayer, LegendreKANLayer, WaveletKANLayer
from network.ts_kan import TimeSeriesKAN

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available. Using GPU.")
else:
    print("No GPU available. Using CPU.")

# Set the device
device = "/gpu:0" if len(physical_devices) > 0 else "/cpu:0"


def create_lstm_model(input_shape, output_features, label_width, units=64, dropout_rate=0.2):
    inputs = keras.Input(shape=input_shape, dtype="float64")
    x = inputs
    x = LSTM(units, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(output_features * label_width)(x)
    outputs = keras.layers.Reshape((label_width, output_features))(outputs)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_mlp_model(input_shape, output_features, label_width, units=64, dropout_rate=0.2):
    inputs = keras.Input(shape=input_shape, dtype="float64")
    x = keras.layers.Flatten()(inputs)
    x = Dense(units, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units // 2, activation='relu', kernel_initializer='he_normal')(x)
    outputs = Dense(output_features * label_width, activation='linear')(x)
    outputs = keras.layers.Reshape((label_width, output_features))(outputs)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_kan_model(input_shape, output_features, label_width, kan_type, hidden_size=64, kan_size=32,
                     **kan_kwargs):
    if kan_type.lower() == 'chebyshev':
        kan_layer = ChebyshevKANLayer(out_features=kan_size, **kan_kwargs)
    elif kan_type.lower() == 'bspline':
        kan_layer = BSplineKANLayer(out_features=kan_size, **kan_kwargs)
    elif kan_type.lower() == 'fourier':
        kan_layer = FourierKANLayer(out_features=kan_size, **kan_kwargs)
    elif kan_type.lower() == 'legendre':
        kan_layer = LegendreKANLayer(out_features=kan_size, **kan_kwargs)
    elif kan_type.lower() == 'wavelet':
        kan_layer = WaveletKANLayer(out_features=kan_size, **kan_kwargs)
    else:
        raise ValueError(f"Unsupported KAN type: {kan_type}")

    model = TimeSeriesKAN(hidden_size=hidden_size, output_size=output_features * label_width,
                          kan_layer=kan_layer)
    model.reshape = tf.keras.layers.Reshape((label_width, output_features))
    return model


class CustomR2Score(R2Score):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1, y_true.shape[-1]])
        y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        return super().update_state(y_true, y_pred, sample_weight)


def create_and_compile_model(model_type, input_shape, output_features, label_width, learning_rate=0.001, **kwargs):
    if model_type.lower() == 'lstm':
        model = create_lstm_model(input_shape, output_features, label_width, **kwargs)
    elif model_type.lower() == 'mlp':
        model = create_mlp_model(input_shape, output_features, label_width, **kwargs)
    elif model_type.lower() in ['bspline', 'chebyshev', 'fourier']:
        model = create_kan_model(input_shape, output_features, label_width, kan_type=model_type, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[MeanAbsoluteError(name='mae'),
                           MeanSquaredError(name='mse'),
                           RootMeanSquaredError(name='rmse'),
                           CustomR2Score(name='r2')])
    return model


def check_for_nan(tensor):
    return tf.reduce_any(tf.math.is_nan(tensor))


def train_model(model, train_data, val_data, epochs):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    class NanCallback(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if logs is not None and 'loss' in logs:
                if np.isnan(logs['loss']):
                    print(f"NaN loss encountered at batch {batch}")
                    self.model.stop_training = True
            else:
                print(f"Warning: No logs available at batch {batch}")

    # Use mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    logger.info("Training model...")

    start_time = time.time()  # Start timing

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[early_stopping, lr_scheduler, NanCallback()],
        verbose=1
    )

    end_time = time.time()  # End timing
    training_time = end_time - start_time

    return history, training_time


def evaluate_model(model, test_data):
    y_true = []
    y_pred = []
    for x, y in tqdm(test_data, desc="Evaluating"):
        y_true.append(y.numpy())
        pred = model.predict(x, verbose=0)
        if np.isnan(pred).any():
            logger.warning("NaN values found in model predictions")
        y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Check for NaN values
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        logger.warning("NaN values found in true values or predictions")

    # Flatten the arrays
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

    # Calculate metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_flat, y_pred_flat)

    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    return metrics


def save_metrics(metrics, file_path):
    with open(file_path, 'w') as f:
        for metric_name, value in metrics.items():
            f.write(f'{metric_name}: {value}\n')



def generate_sample_data(num_samples=1000, input_width=30, input_features=43, output_features=3, label_width=5):
    # Generate random input data
    X = np.random.randn(num_samples, input_width, input_features)

    # Generate random output data
    y = np.random.randn(num_samples, label_width, output_features)

    # Split into train, validation, and test sets
    train_split = int(0.7 * num_samples)
    val_split = int(0.85 * num_samples)

    X_train, y_train = X[:train_split], y[:train_split]
    X_val, y_val = X[train_split:val_split], y[train_split:val_split]
    X_test, y_test = X[val_split:], y[val_split:]

    # Create TensorFlow datasets
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    return train_data, val_data, test_data, input_features, output_features


def main(model_type, use_sample_data=False):
    tf.keras.backend.set_floatx('float64')
    # Parameters
    input_width = 30
    label_width = 5
    shift = 5

    if use_sample_data:
        logger.info("Generating sample data...")
        train_data, val_data, test_data, input_features, output_features = generate_sample_data(
            input_width=input_width, label_width=label_width)
    else:
        logger.info("Preparing data...")
        df, input_features, output_features, stock_scalers, market_scaler, encoders = prepare_data(
            'data/processed/sp500.csv',
            'data/processed/market.csv',
            'data/ticker_split.json'
        )
        logger.info("Getting train/val/test ticker split...")
        train_tickers, val_tickers, test_tickers = get_ticker_split('data/ticker_split.json')

        # Create WindowGenerator for train and validation data
        window_generator = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            train_df=df[df['ticker'].isin(train_tickers)],
            val_df=df[df['ticker'].isin(val_tickers)],
            test_df=None,  # We don't use test data here
            label_columns=output_features
        )

        # Get the datasets
        train_data = window_generator.train
        val_data = window_generator.val
        input_features = len(input_features)
        output_features = len(output_features)

    input_shape = (input_width, input_features)
    logger.info(f"Creating {model_type} model...")
    with tf.device(device):
        if model_type.startswith('kan-'):
            kan_type = model_type.split('-')[1]
            model = create_and_compile_model(kan_type, input_shape, output_features, label_width, learning_rate=0.001,
                                             hidden_size=64, kan_size=32)
        else:
            model = create_and_compile_model(model_type, input_shape, output_features, label_width,
                                             learning_rate=0.001)


        history, training_time = train_model(model, train_data, val_data, epochs=100)

        logger.info("Saving model...")

        if use_sample_data:
            model.save(f'results/models/{model_type}_sample_model.keras')
        else:
            model.save(f'results/models/{model_type}_model.keras')

        # Save training history and time
        history_dict = history.history
        history_dict['training_time'] = training_time

        if use_sample_data:
            with open(f'results/metrics/{model_type}_sample_history.json', 'w') as f:
                json.dump(history_dict, f)
        else:
            with open(f'results/metrics/{model_type}_history.json', 'w') as f:
                json.dump(history_dict, f)

        logger.info(f"Training time: {training_time:.2f} seconds")

    logger.info("Process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test a model.")
    parser.add_argument("model_type",
                        choices=['lstm', 'mlp', 'kan-bspline', 'kan-chebyshev', 'kan-fourier', 'kan-legendre',
                                 'kan-wavelet'],
                        help="Type of model to use")
    parser.add_argument("--test", action="store_true", help="Test the model instead of training")
    parser.add_argument("--sample", action="store_true", help="Use sample data for quick testing")
    args = parser.parse_args()
    main(args.model_type, args.sample)
