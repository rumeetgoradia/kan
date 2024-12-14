import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    logger.info("Loading stock market data...")
    all_data = pd.read_csv(stock_market_data_filepath, parse_dates=['Date'])

    logger.info("Splitting data...")
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


def calculate_scores(y_true, y_pred):
    y_true_reshaped = y_true.reshape(-1, y_true.shape[-1])
    y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])

    # Calculate metrics
    mse = mean_squared_error(y_true_reshaped, y_pred_reshaped)
    mae = mean_absolute_error(y_true_reshaped, y_pred_reshaped)
    rmse = np.sqrt(mse)

    # Calculate R2 score for each feature
    r2_scores = [r2_score(y_true_reshaped[:, i], y_pred_reshaped[:, i]) for i in range(y_true.shape[-1])]

    # Average R2 score across features
    r2_avg = np.mean(r2_scores)

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2_avg,
        'R2_per_feature': r2_scores
    }


def main(model_name: str, models_directory: str, metrics_save_directory: str = None):
    # Load the test data
    test_data, _, _ = get_data(stock_market_data_filepath=STOCK_MARKET_DATA_FILEPATH,
                               ticker_splits_filepath=TICKER_SPLITS_FILEPATH,
                               output_features=OUTPUT_FEATURES,
                               lookahead=LOOKAHEAD,
                               sequence_length=SEQUENCE_LENGTH,
                               batch_size=BATCH_SIZE
                               )

    # Load the model
    model = load_and_compile_model(model_name, models_directory)

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
                        choices=['lstm', 'mlp', 'kan-bspline', 'kan-chebyshev', 'kan-fourier', 'kan-legendre'],
                        help="Type of model to use")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to load the model from")
    parser.add_argument("--metrics-save-dir", type=str, required=False, help="Directory to save the model metrics")
    args = parser.parse_args()
    main(args.model_type, args.models_dir, args.metrics_save_dir)
