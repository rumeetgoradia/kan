import logging

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    stock_price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
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


def prepare_data(stock_file, market_file):
    logger.info("Starting data preparation...")
    df = load_and_merge_data(stock_file, market_file)

    df, stock_scalers, market_scaler, encoders = preprocess_data(df)

    # Add this after your data preparation
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    stock_price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    market_columns = [col for col in df.columns if col.startswith('^') or '=' in col]
    categorical_columns = ['industry', 'sector']

    input_features = stock_price_columns + market_columns + categorical_columns
    output_features = ['Open', 'Close', 'Volume']

    logger.info("Data preparation completed.")
    return df, input_features, output_features, stock_scalers, market_scaler, encoders


if __name__ == "__main__":
    df, input_features, output_features, stock_scalers, market_scaler, encoders = prepare_data('processed/sp500.csv',
                                                                                               'processed/market.csv')
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Input features: {input_features}")
    logger.info(f"Output features: {output_features}")
    logger.info(f"Number of stocks: {len(stock_scalers)}")
    logger.info(f"Number of encoders: {len(encoders)}")
