"""
rg3072
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from constants import *
from data import *
from network import LSTMNetwork, MLPNetwork
from network.kan.v3 import TimeSeriesKANV3
from network.common.compile_model import compile_model

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
    return LSTMNetwork(input_shape=input_shape, lookahead=lookahead, num_output_features=num_output_features,
                       units=units,
                       dropout_rate=dropout_rate)


def create_mlp_model(input_shape, lookahead, num_output_features, units=64, dropout_rate=0.2):
    return MLPNetwork(input_shape=input_shape, lookahead=lookahead, num_output_features=num_output_features,
                      units=units,
                      dropout_rate=dropout_rate)


def create_kan_model(input_shape, lookahead, num_output_features, kan_layer_type,
                     dropout_rate=0.2):
    model = TimeSeriesKANV3(
        seq_length=input_shape[0],
        num_features=input_shape[1],
        lookahead=lookahead,
        num_outputs=num_output_features,
        lstm_units_list=[256, 128, 64],
        kan_units_list=[32, 16],
        kan_layer_type=kan_layer_type,
        dropout_rate=dropout_rate
    )
    return model


def create_and_compile_model(model_type, input_shape, lookahead, num_output_features, learning_rate, **kwargs):
    if model_type.lower() == LSTMNetwork.NAME.lower():
        model = create_lstm_model(input_shape=input_shape,
                                  lookahead=lookahead,
                                  num_output_features=num_output_features,
                                  **kwargs)
    elif model_type.lower() == MLPNetwork.NAME.lower():
        model = create_mlp_model(input_shape=input_shape,
                                 lookahead=lookahead,
                                 num_output_features=num_output_features,
                                 **kwargs)
    elif (model_type.lower().startswith(TimeSeriesKANV3.NAME.lower())
          and model_type.split('-')[1] in TimeSeriesKANV3.ALLOWED_KAN_LAYERS):
        model = create_kan_model(input_shape=input_shape,
                                 lookahead=lookahead,
                                 num_output_features=num_output_features,
                                 kan_layer_type=model_type.split('-')[1])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return compile_model(model, learning_rate)


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


def main(model_type, model_save_directory: str = None, history_save_directory: str = None):
    tf.keras.backend.set_floatx('float32')

    data = get_data(stock_market_data_filepath='data/processed/stock_market.csv',
                    ticker_splits_filepath='data/processed/ticker_splits.json',
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

    logger.info(f"Creating {model_type} model...")
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

        model_save_filepath = os.path.join(model_save_directory, MODEL_FILE_NAME(model_type))
        model.save(model_save_filepath)
        logger.info(f"Model saved to {model_save_filepath}")

    if history_save_directory is not None:
        if not os.path.exists(history_save_directory):
            os.makedirs(history_save_directory)

        history_save_filepath = os.path.join(history_save_directory, TRAIN_METRICS_FILE_NAME(model_type))
        with open(history_save_filepath, 'w') as f:
            json.dump(history_dict, f)
            logger.info(f"History saved to {history_save_filepath}")

    logger.info("Process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("model", type=str,
                        choices=MODEL_TYPES,
                        help="Type of model to train")
    parser.add_argument("--model-save-dir", type=str, required=False, help="Directory to save the model")
    parser.add_argument("--history-save-dir", type=str, required=False,
                        help="Directory to save the model training history")
    args = parser.parse_args()
    main(args.model, args.model_save_dir, args.history_save_dir)
