import json
import logging
import os
import random
import math
from typing import Dict, List

import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_ticker_splits(split_ratios: Dict[str, float], stock_data: pd.DataFrame, output_file: str) -> Dict[
    str, List[str]]:
    """
    Create randomized splits of tickers for training, validation, and test sets.

    Args:
    split_ratios (Dict[str, float]): A dictionary with 'train', 'val', 'test' as keys and their respective percentages as values.
    stock_data (pd.DataFrame): The stock market dataframe containing the tickers.
    output_file (str): The filepath to save/load the ticker splits.

    Returns:
    Dict[str, List[str]]: A dictionary with 'train', 'val', 'test' as keys and lists of tickers as values.
    """
    if os.path.exists(output_file):
        logger.info(f"Loading existing ticker splits from {output_file}...")
        with open(output_file, 'r') as f:
            return json.load(f)

    logger.info("Creating new ticker splits...")

    # Validate split ratios
    if not math.isclose(sum(split_ratios.values()), 1, rel_tol=1e-9):
        raise ValueError("Split ratios must sum to 1")

    # Get unique tickers
    tickers = stock_data['Ticker'].unique().tolist()
    random.shuffle(tickers)

    # Calculate number of tickers for each split
    total_tickers = len(tickers)
    split_counts = {k: int(v * total_tickers) for k, v in split_ratios.items()}

    last_split = list(split_counts.keys())[-1]
    split_counts[last_split] += total_tickers - sum(split_counts.values())

    # Create splits
    splits = {}
    start = 0
    for split_name, count in split_counts.items():
        end = start + count
        splits[split_name] = tickers[start:end]
        start = end

    # Save splits to file
    with open(output_file, 'w') as f:
        json.dump(splits, f)

    logger.info(f"Ticker splits saved to {output_file}")
    return splits


def create_split_dataframes(splits: Dict[str, List[str]], stock_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create separate dataframes for training, validation, and test sets based on ticker splits.

    Args:
    splits (Dict[str, List[str]]): A dictionary with 'train', 'val', 'test' as keys and lists of tickers as values.
    stock_data (pd.DataFrame): The stock market dataframe containing all data.

    Returns:
    Dict[str, pd.DataFrame]: A dictionary with 'train', 'val', 'test' as keys and their respective dataframes as values.
    """
    logger.info("Creating split dataframes...")
    split_dfs = {}

    for split_name, tickers in splits.items():
        split_df = stock_data[stock_data['Ticker'].isin(tickers)]

        # Check if 'Date' is in columns or index
        if 'Date' in stock_data.columns:
            split_df = split_df.sort_values('Date')
        elif stock_data.index.name == 'Date':
            split_df = split_df.sort_index()
        else:
            logger.warning("'Date' not found in columns or index. Dataframe will not be sorted by date.")

        split_dfs[split_name] = split_df

    return split_dfs


if __name__ == "__main__":
    from constants import SPLIT_RATIOS
    # Example usage
    stock_data = pd.read_csv('processed/stock_market.csv', parse_dates=['Date'])

    splits = create_ticker_splits(SPLIT_RATIOS, stock_data, 'processed/ticker_splits.json')
    split_dfs = create_split_dataframes(splits, stock_data)
    print(split_dfs['train'].head())
