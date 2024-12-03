import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_ticker_split(file_path='data/ticker_split.json'):
    """
    Retrieve the ticker split from a JSON file.

    Args:
    file_path (str): Path to the JSON file containing the ticker split.

    Returns:
    tuple: A tuple containing lists of train, validation, and test tickers.
           Returns (None, None, None) if the file doesn't exist.

    Raises:
    json.JSONDecodeError: If there's an error decoding the JSON file.
    """
    if not os.path.exists(file_path):
        logger.warning(f"Ticker split file not found: {file_path}")
        return None, None, None

    try:
        with open(file_path, 'r') as f:
            split = json.load(f)

        train_tickers = split.get('train', [])
        val_tickers = split.get('val', [])
        test_tickers = split.get('test', [])

        logger.info(f"Loaded ticker split from {file_path}")
        logger.info(f"Train tickers: {len(train_tickers)}, "
                    f"Validation tickers: {len(val_tickers)}, "
                    f"Test tickers: {len(test_tickers)}")

        return train_tickers, val_tickers, test_tickers

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        raise


def create_ticker_split(df, train_ratio=0.7, val_ratio=0.15, file_path='ticker_split.json'):
    """
    Create and save a new ticker split.

    Args:
    df (pd.DataFrame): The dataframe containing the ticker data.
    train_ratio (float): The ratio of tickers to use for training.
    val_ratio (float): The ratio of tickers to use for validation.
    file_path (str): Path to save the JSON file containing the ticker split.

    Returns:
    tuple: A tuple containing lists of train, validation, and test tickers.
    """
    logger.info("Creating new ticker split")
    all_tickers = df['ticker'].unique().tolist()
    train_tickers, temp_tickers = train_test_split(all_tickers, train_size=train_ratio)
    val_tickers, test_tickers = train_test_split(temp_tickers, train_size=(val_ratio / (1 - train_ratio)))

    split = {
        'train': train_tickers,
        'val': val_tickers,
        'test': test_tickers
    }
    with open(file_path, 'w') as f:
        json.dump(split, f)
    logger.info(f"Saved ticker split to {file_path}")

    return train_tickers, val_tickers, test_tickers


def load_and_merge_data(stock_file, market_file):
    logger.info("Loading stock data...")
    stock_df = pd.read_csv(stock_file)
    stock_df['date'] = pd.to_datetime(stock_df['date'])

    logger.info("Loading market data...")
    market_df = pd.read_csv(market_file)
    market_df['date'] = pd.to_datetime(market_df['date'])

    logger.info("Merging stock and market data...")
    df = pd.merge(stock_df, market_df, on='date', how='left')

    logger.info("Forward filling market data...")
    market_columns = market_df.columns.drop('date')
    df[market_columns] = df[market_columns].ffill()

    return df


def preprocess_data(df):
    logger.info("Starting data preprocessing...")
    stock_price_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    market_columns = [col for col in df.columns if col.startswith('^') or '=' in col]
    categorical_columns = ['industry', 'sector']

    stock_scalers = {}
    market_scaler = MinMaxScaler()
    encoders = {}

    logger.info("Scaling stock price features per ticker...")
    for ticker in tqdm(df['ticker'].unique(), desc="Scaling data"):
        stock_scalers[ticker] = MinMaxScaler()
        ticker_mask = df['ticker'] == ticker
        df.loc[ticker_mask, stock_price_columns] = stock_scalers[ticker].fit_transform(
            df.loc[ticker_mask, stock_price_columns].astype(float))

    logger.info("Scaling market features globally...")
    df[market_columns] = market_scaler.fit_transform(df[market_columns].astype(float))

    logger.info("Encoding categorical features...")
    for col in tqdm(categorical_columns, desc="Encoding categories"):
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    return df, stock_scalers, market_scaler, encoders


def split_data(df, ticker, train_split=0.7, val_split=0.2):
    # logger.info(f"Splitting data for ticker {ticker}")
    ticker_data = df[df['ticker'] == ticker].sort_values('date')
    n = len(ticker_data)
    train_df = ticker_data[:int(n * train_split)]
    val_df = ticker_data[int(n * train_split):int(n * (train_split + val_split))]
    test_df = ticker_data[int(n * (train_split + val_split)):]

    return train_df, val_df, test_df


def prepare_data(stock_file, market_file, train_test_ratio=0.8, ticker_split_file='ticker_split.json'):
    logger.info("Starting data preparation...")
    df = load_and_merge_data(stock_file, market_file)

    logger.info(f"Columns after merging: {df.columns.tolist()}")

    df, stock_scalers, market_scaler, encoders = preprocess_data(df)

    logger.info(f"Columns after preprocessing: {df.columns.tolist()}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    logger.info(f"Columns after removing NaNs: {df.columns.tolist()}")

    stock_price_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    market_columns = [col for col in df.columns if col.startswith('^') or '=' in col]
    categorical_columns = ['industry', 'sector']

    input_features = stock_price_columns + market_columns + categorical_columns
    output_features = ['Open', 'Close', 'Volume']

    logger.info("Data preparation completed.")
    logger.info(f"Number of input features: {len(input_features)}")
    logger.info(f"Input features: {input_features}")
    logger.info(f"Number of columns in DataFrame: {len(df.columns)}")
    logger.info(f"Columns in DataFrame: {df.columns.tolist()}")

    if not os.path.exists(ticker_split_file):
        create_ticker_split(df, train_test_ratio, ticker_split_file)

    return df, input_features, output_features, stock_scalers, market_scaler, encoders


if __name__ == "__main__":
    df, input_features, output_features, stock_scalers, market_scaler, encoders = prepare_data('processed/sp500.csv',
                                                                                               'processed/market.csv')
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Input features: {input_features}")
    logger.info(f"Output features: {output_features}")
    logger.info(f"Number of stocks: {len(stock_scalers)}")
    logger.info(f"Number of encoders: {len(encoders)}")
