import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

from network import ThreeDimensionalR2Score
from network.kan.v1 import TimeSeriesKAN, TimeSeriesKANAttentionLayer
from network.kan.layer import *
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError
import logging
from constants import *
from data import *

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

def load_and_compile_model(model_path, learning_rate):
    logger.info(f"Loading model from {model_path}...")
    custom_objects = {
        'BaseKANLayer': BaseKANLayer,
        'BSplineKANLayer': BSplineKANLayer,
        'ChebyshevKANLayer': ChebyshevKANLayer,
        'FourierKANLayer': FourierKANLayer,
        'LegendreKANLayer': LegendreKANLayer,
        'TimeSeriesKAN': TimeSeriesKAN,
        'AttentionLayer': TimeSeriesKANAttentionLayer,
        'CustomR2Score': ThreeDimensionalR2Score
    }

    # Load the model with custom objects
    model = load_model(model_path, custom_objects=custom_objects, compile=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            MeanAbsoluteError(name='mae'),
            MeanSquaredError(name='mse'),
            RootMeanSquaredError(name='rmse'),
            ThreeDimensionalR2Score(name='r2')
        ]
    )

    return model


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

def main(model_name:str, models_directory:str, save_directory:str=None):
    # Load the test data
    test_data, _, _ = get_data(stock_market_data_filepath=STOCK_MARKET_DATA_FILEPATH,
                         ticker_splits_filepath=TICKER_SPLITS_FILEPATH,
                         output_features=OUTPUT_FEATURES,
                         lookahead=LOOKAHEAD,
                         sequence_length=SEQUENCE_LENGTH,
                         batch_size=BATCH_SIZE
                         )

    # Load the model
    model_path = os.path.join(models_directory, MODEL_FILE_NAME(model_name))
    model = load_and_compile_model(model_path, LEARNING_RATE)

    # Evaluate the model on the dataset
    logger.info("Evaluating model...")
    results = model.evaluate(test_data, verbose=1, return_dict=True)


    if save_directory is not None:
        # Define the path for the JSON file
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        scores_path = os.path.join(save_directory, f'{model_name}_scores.json')

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
    parser.add_argument("--save-dir", type=str, required=False, help="Directory to save the model metrics")
    args = parser.parse_args()
    main(args.model_type, args.models_dir, args.save_dir)


