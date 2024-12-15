"""
rg3072
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from constants import *
from data import *
from network import ThreeDimensionalR2Score, LSTMNetwork, MLPNetwork
from network.common import compile_model
from network.kan.v3 import TimeSeriesKANV3

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_data(stock_market_data_filepath, ticker_splits_filepath, sequence_length, lookahead, output_features,
             batch_size):
    logger.info("Loading stock market data")
    all_data = pd.read_csv(stock_market_data_filepath, parse_dates=['Date'])

    logger.info("Splitting data")
    ticker_splits = create_ticker_splits({'train': 0.7, 'val': 0.15, 'test': 0.15}, all_data, ticker_splits_filepath)
    split_dataframes = create_split_dataframes(ticker_splits, all_data)

    logger.info("Creating testing dataset...")
    test = create_sequenced_tf_dataset(split_dataframes[TEST_KEY], sequence_length, lookahead,
                                       output_features, batch_size)

    return test


def load_and_compile_model(model_name: str, models_directory: str):
    model_path = os.path.join(models_directory, MODEL_FILE_NAME(model_name))

    custom_objects = {
        "ThreeDimensionalR2Score": ThreeDimensionalR2Score,
    }

    if model_name == LSTMNetwork.NAME:
        custom_objects['LSTMNetwork'] = LSTMNetwork
    elif model_name == MLPNetwork.NAME:
        custom_objects['MLPNetwork'] = MLPNetwork
    elif model_name.startswith(TimeSeriesKANV3.NAME):
        custom_objects['TimeSeriesKANV3'] = TimeSeriesKANV3

    else:
        raise ValueError(f"Invalid model type: {model_name}")

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    return compile_model(model=model)


def log_layer_weights(model):
    for layer in model.layers:
        logger.info(f"Layer: {layer.name} (Type: {type(layer).__name__})")

        if isinstance(layer, tf.keras.layers.TimeDistributed):
            # For TimeDistributed layers, we need to access the inner layer
            inner_layer = layer.layer
            weights = inner_layer.get_weights()
        else:
            weights = layer.get_weights()

        if not weights:
            logger.info("  No weights in this layer.")
        else:
            for i, w in enumerate(weights):
                logger.info(f"  Weight {i} shape: {w.shape}")
                logger.info(f"  Weight {i} mean: {np.mean(w):.6f}, std: {np.std(w):.6f}")
                logger.info(f"  Weight {i} min: {np.min(w):.6f}, max: {np.max(w):.6f}")

        logger.info("")  # Add a blank line for readability


def main(model_name: str, models_directory: str, metrics_save_directory: str = None):
    # Load the model
    model = load_and_compile_model(model_name, models_directory)
    log_layer_weights(model)

    # Load the test data
    test_data, _, _ = get_data(stock_market_data_filepath='data/processed/stock_market.csv',
                               ticker_splits_filepath='data/processed/ticker_splits.json',
                               output_features=OUTPUT_FEATURES,
                               lookahead=LOOKAHEAD,
                               sequence_length=SEQUENCE_LENGTH,
                               batch_size=BATCH_SIZE
                               )

    # Evaluate the model on the dataset
    logger.info("Evaluating model...")
    results = model.evaluate(test_data, verbose=1, return_dict=True)

    if metrics_save_directory is not None:
        # Define the path for the JSON file
        if not os.path.exists(metrics_save_directory):
            os.makedirs(metrics_save_directory)
        scores_path = os.path.join(metrics_save_directory, TEST_METRICS_FILE_NAME(model_name))

        # Save the metrics to a JSON file
        with open(scores_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)
            logger.info(f"Metrics saved to {scores_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model.")
    parser.add_argument("model_type",
                        choices=MODEL_TYPES,
                        help="Type of model to test")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to load the model from")
    parser.add_argument("--metrics-save-dir", type=str, required=False, help="Directory to save the model metrics")
    args = parser.parse_args()
    main(args.model_type, args.models_dir, args.metrics_save_dir)
