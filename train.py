import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError

from constants import *
from data import *
from network import ThreeDimensionalR2Score, LSTMNetwork, MLPNetwork
from network.kan.v2 import TimeSeriesKAN
from network.kan.layer import *

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_data(stock_market_data_filepath, ticker_splits_filepath, sequence_length, lookahead, output_features,
             batch_size):
    logger.info("Loading stock market data...")
    all_data = pd.read_csv(stock_market_data_filepath, parse_dates=['Date'])

    logger.info("Splitting data...")
    ticker_splits = create_ticker_splits({'train': 0.7, 'val': 0.15, 'test': 0.15}, all_data, ticker_splits_filepath)
    split_dataframes = create_split_dataframes(ticker_splits, all_data)

    logger.info("Creating training dataset...")
    train = create_sequenced_tf_dataset(split_dataframes[TRAIN_KEY], sequence_length, lookahead,
                                        output_features, batch_size)
    logger.info("Creating validation dataset...")
    val = create_sequenced_tf_dataset(split_dataframes[VAL_KEY], sequence_length, lookahead, output_features,
                                      batch_size)

    return {TRAIN_KEY: train, VAL_KEY: val}


def create_lstm_model(input_shape, lookahead, num_output_features, units=64, dropout_rate=0.2):
    return LSTMNetwork(input_shape=input_shape, lookahead=lookahead, num_output_features=num_output_features, units=units,
                       dropout_rate=dropout_rate)


def create_mlp_model(input_shape, lookahead, num_output_features, units=64, dropout_rate=0.2):
    return MLPNetwork(input_shape=input_shape, lookahead=lookahead, num_output_features=num_output_features, units=units,
                      dropout_rate=dropout_rate)


def create_kan_model(input_shape, lookahead, num_output_features, kan_type, kan_size=32,
                     num_lstm_layers=1, num_transformer_layers=1, num_heads=8, dff=256,
                     dropout_rate=0.1, output_activation=None,
                     **kan_kwargs):
    # Ensure hidden_size is divisible by num_heads
    hidden_size = max(input_shape[-1], ((input_shape[-1] + num_heads - 1) // num_heads) * num_heads)

    if kan_type.lower() == 'chebyshev':
        kan_layer = ChebyshevKANLayer(out_features=kan_size, **kan_kwargs)
    elif kan_type.lower() == 'bspline':
        kan_layer = BSplineKANLayer(out_features=kan_size, **kan_kwargs)
    elif kan_type.lower() == 'fourier':
        kan_layer = FourierKANLayer(out_features=kan_size, **kan_kwargs)
    elif kan_type.lower() == 'legendre':
        kan_layer = LegendreKANLayer(out_features=kan_size, **kan_kwargs)
    else:
        raise ValueError(f"Unsupported KAN type: {kan_type}")

    model = TimeSeriesKAN(
        hidden_size=hidden_size,
        lookahead=lookahead,
        num_output_features=num_output_features,
        kan_layer=kan_layer,
        num_lstm_layers=num_lstm_layers,
        num_transformer_layers=num_transformer_layers,
        lstm_kwargs=None,
        dropout_rate=dropout_rate,
        output_activation=output_activation,
        num_heads=num_heads,
        dff=dff
    )
    return model


def create_and_compile_model(model_type, input_shape, lookahead, num_output_features, learning_rate, **kwargs):
    if model_type.lower() == 'lstm':
        model = create_lstm_model(input_shape=input_shape,
                                  lookahead=lookahead,
                                  num_output_features=num_output_features,
                                  **kwargs)
    elif model_type.lower() == 'mlp':
        model = create_mlp_model(input_shape=input_shape,
                                 lookahead=lookahead,
                                 num_output_features=num_output_features,
                                 **kwargs)
    elif model_type.lower() in ['bspline', 'chebyshev', 'fourier', 'legendre', 'wavelet']:
        model = create_kan_model(input_shape=input_shape,
                                 lookahead=lookahead,
                                 num_output_features=num_output_features,
                                 kan_type=model_type,
                                 **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[MeanAbsoluteError(name='mae'),
                           MeanSquaredError(name='mse'),
                           RootMeanSquaredError(name='rmse'),
                           ThreeDimensionalR2Score(name='r2')])
    return model


def train_model(model, train_data, val_data, epochs):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    class NanCallback(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if logs is not None and 'loss' in logs:
                if np.isnan(logs['loss']):
                    logger.warning(f"NaN loss encountered at batch {batch}")
                    self.model.stop_training = True
            else:
                logger.warning(f"No logs available at batch {batch}")

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


def main(model_name, model_save_directory:str=None, history_save_directory:str=None):
    tf.keras.backend.set_floatx('float32')

    data = get_data(stock_market_data_filepath=STOCK_MARKET_DATA_FILEPATH,
                    ticker_splits_filepath=TICKER_SPLITS_FILEPATH,
                    output_features=OUTPUT_FEATURES,
                    lookahead=LOOKAHEAD,
                    sequence_length=SEQUENCE_LENGTH,
                    batch_size=BATCH_SIZE
                    )

    train_data, train_input_shape, train_output_shape = data[TRAIN_KEY]
    val_data, val_input_shape, val_output_shape = data[VAL_KEY]

    if train_input_shape != val_input_shape or train_output_shape != val_output_shape:
        raise ValueError(f"Train and validation shapes do not match. "
                         f"Train shapes: input {train_input_shape}, output {train_output_shape}. "
                         f"Validation shapes: input {val_input_shape}, output {val_output_shape}.")
    else:
        logger.info(f"Train and validation shapes: input {train_input_shape}, output {train_output_shape}")

    logger.info(f"Creating {model_name} model...")

    model_type = model_name
    if model_name.startswith('kan-'):
        model_type = model_name.split('-')[1]

    model = create_and_compile_model(model_type=model_type,
                                     input_shape=train_input_shape,
                                     lookahead=LOOKAHEAD,
                                     num_output_features=len(OUTPUT_FEATURES),
                                     learning_rate=LEARNING_RATE)

    history, training_time = train_model(model, train_data, val_data, epochs=EPOCHS)

    logger.info(f"Training time: {training_time:.2f} seconds")

    history_dict = history.history
    history_dict['training_time'] = training_time

    if model_save_directory is not None:
        if not os.path.exists(model_save_directory):
            os.makedirs(model_save_directory)

        model.save(os.path.join(model_save_directory, MODEL_FILE_NAME(model_name)))
        logger.info(f"Model saved to {model_save_directory}")

    if history_save_directory is not None:
        if not os.path.exists(history_save_directory):
            os.makedirs(history_save_directory)

        with open(os.path.join(f'{model_name}_history.json'), 'w') as f:
            json.dump(history_dict, f)
            logger.info(f"History saved to {model_name}_history.json")

    logger.info("Process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("model", type=str,
                        choices=['lstm', 'mlp', 'kan-bspline', 'kan-chebyshev', 'kan-fourier', 'kan-legendre'],
                        help="Type of model to use")
    parser.add_argument("--model-save-dir", type=str, required=False, help="Directory to save the model")
    parser.add_argument("--history-save-dir", type=str, required=False, help="Directory to save the model training history")
    args = parser.parse_args()
    main(args.model, args.model_save_dir, args.history_save_dir)
