"""
rg3072
"""

import argparse
import json
import logging
import os

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from constants import *
from data import create_ticker_splits, create_split_dataframes, create_sequenced_tf_dataset

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_test_data(stock_market_data_filepath, ticker_splits_filepath, sequence_length, lookahead, output_features,
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


def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        return None


def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON file: {file_path}")


def compile_metrics(model_type, train_metrics_dir, test_metrics_dir):
    compiled_metrics = {}

    model_types = [model_type] if model_type else MODEL_TYPES
    for model_name in tqdm(model_types, desc="Compiling metrics"):
        train_metrics_path = os.path.join(train_metrics_dir, TRAIN_METRICS_FILE_NAME(model_name))
        test_metrics_path = os.path.join(test_metrics_dir, TEST_METRICS_FILE_NAME(model_name))

        train_metrics = load_json(train_metrics_path)
        test_metrics = load_json(test_metrics_path)

        if train_metrics is None or test_metrics is None:
            logger.warning(f"Skipping {model_name} due to missing metrics files")
            continue

        compiled_metrics[model_name] = {
            "train": {
                "loss": train_metrics["loss"],
                "mse": train_metrics["mse"],
                "mae": train_metrics["mae"],
                "r2": train_metrics["r2"],
                'training_time': train_metrics['training_time'],
            },
            "val": {
                "loss": train_metrics["val_loss"],
                "mse": train_metrics["val_mse"],
                "mae": train_metrics["val_mae"],
                "r2": train_metrics["val_r2"]
            },
            "test": test_metrics
        }

    return compiled_metrics


def create_predictions_json(model_type, models_dir, test_data):
    predictions = {}

    model_types = [model_type] if model_type else MODEL_TYPES
    for model_name in model_types:
        model_path = os.path.join(models_dir, MODEL_FILE_NAME(model_name))

        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {model_path}: {e}")
            continue

        try:
            logger.info(f"Creating predictions for model: {model_name}")
            model_predictions = model.predict(test_data, verbose=1)
            predictions[model_name] = model_predictions.tolist()
        except Exception as e:
            logger.error(f"Failed to create predictions for model: {model_name}: {e}")

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile model metrics and create predictions JSON")
    parser.add_argument("--model_type", choices=MODEL_TYPES, help="Specific model type to compile (optional)")
    parser.add_argument("--models_dir", required=True, help="Directory containing model files")
    parser.add_argument("--train_metrics_dir", required=True, help="Directory containing training metrics files")
    parser.add_argument("--test_metrics_dir", required=True, help="Directory containing test metrics files")
    parser.add_argument("--output_dir", default="compiled", help="Directory to save output files")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Compile metrics
    compiled_metrics = compile_metrics(args.model_type, args.train_metrics_dir, args.test_metrics_dir)
    save_json(compiled_metrics, os.path.join(args.output_dir, "compiled_metrics.json"))

    # Load test data
    test_data, _, _ = get_test_data(stock_market_data_filepath='../data/processed/stock_market.csv',
                                    ticker_splits_filepath='../data/processed/ticker_splits.json',
                                    output_features=OUTPUT_FEATURES,
                                    lookahead=LOOKAHEAD,
                                    sequence_length=SEQUENCE_LENGTH,
                                    batch_size=BATCH_SIZE
                                    )

    # Create predictions JSON
    predictions = create_predictions_json(args.model_type, args.models_dir, test_data)
    # Extract actual values from the test dataset
    actual_values = []
    for x, y in test_data:
        actual_values.extend(y.numpy().tolist())

    # Save predictions and actual values
    save_json({"predictions": predictions, "actual": actual_values},
              os.path.join(args.output_dir, "predictions_and_actual.json"))
