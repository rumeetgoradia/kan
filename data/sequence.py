import logging
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sequenced_tf_dataset(df: pd.DataFrame, sequence_length: int, lookahead: int, output_features: list,
                                batch_size: int) -> Tuple[tf.data.Dataset, Tuple[int, int], Tuple[int, int]]:
    """
    Create a TensorFlow dataset from the dataframe for use in a TensorFlow network.

    Args:
    df (pd.DataFrame): Input dataframe containing stock data.
    sequence_length (int): Length of input sequences.
    lookahead (int): Number of future time steps to predict.
    output_features (list): List of column names to use as output features.
    batch_size (int): Batch size for the dataset.

    Returns:
    Tuple[tf.data.Dataset, Tuple[int, int], Tuple[int, int]]:
        - A TensorFlow dataset that can be used for training
        - Shape of each input sample (sequence_length, num_features)
        - Shape of each output sample (lookahead, num_output_features)
    """
    # Identify input features (exclude 'Ticker', 'Date')
    date_column = 'Date' if 'Date' in df.columns else df.index.name
    input_features = [col for col in df.columns if col not in ['Ticker', date_column]]

    max_sequences = 100_000_000

    # Preallocate arrays with an estimated size
    input_sequences = np.empty((max_sequences, sequence_length, len(input_features)))
    output_sequences = np.empty((max_sequences, lookahead, len(output_features)))

    current_index = 0

    # Group by ticker to ensure sequences don't span multiple tickers
    for ticker, group in tqdm(df.groupby('Ticker'), desc="Sequencing tickers"):
        # Sort group by date
        group = group.sort_values(by=date_column)

        # Create sequences for this ticker
        for i in range(len(group) - sequence_length - lookahead + 1):
            input_seq = group[input_features].iloc[i:i + sequence_length].values
            output_seq = group[output_features].iloc[i + sequence_length:i + sequence_length + lookahead].values

            # Add sequences to the preallocated arrays
            input_sequences[current_index] = input_seq
            output_sequences[current_index] = output_seq
            current_index += 1

    # Trim the arrays to the actual size
    input_sequences = input_sequences[:current_index]
    output_sequences = output_sequences[:current_index]

    logger.info("Creating TensorFlow dataset...")
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((input_sequences, output_sequences))

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=len(input_sequences)).batch(batch_size)

    # Get the shape of input and output samples
    input_shape = (sequence_length, len(input_features))
    output_shape = (lookahead, len(output_features))

    return dataset, input_shape, output_shape


if __name__ == "__main__":
    # Example usage
    from constants import SEQUENCE_LENGTH, LOOKAHEAD, OUTPUT_FEATURES, BATCH_SIZE

    df = pd.read_csv('processed/stock_market.csv', parse_dates=['Date'])

    dataset, input_shape, output_shape = create_sequenced_tf_dataset(df, SEQUENCE_LENGTH, LOOKAHEAD, OUTPUT_FEATURES,
                                                                     BATCH_SIZE)

    # Print the first batch
    for inputs, outputs in dataset.take(1):
        print(f"Input shape: {inputs.shape}")
        print(f"Output shape: {outputs.shape}")
