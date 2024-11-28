import csv
import os
from datetime import datetime
from typing import Dict, List, Union

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm


def prepare_market_data(market_data_file_path: str) -> Dict[datetime.date, Dict[str, float]]:
    market_data: Dict[datetime.date, Dict[str, float]] = {}
    with open(market_data_file_path, mode='r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)

        # Collect all data for scaling
        data: List[List[float]] = []
        dates: List[datetime.date] = []
        fieldnames = [field for field in reader.fieldnames if field != 'date']  # Exclude 'date' from fieldnames

        for row in reader:
            date_str = row['date']
            date = datetime.strptime(date_str, "%Y-%m-%d")  # Convert string to datetime
            dates.append(date.date())
            data.append([float(row[key]) if row[key] else 0.0 for key in fieldnames])

    # Convert to numpy array for scaling
    data_array = np.array(data)

    # Apply Min-Max Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_array)

    # Reconstruct the scaled data into the dictionary
    for i, date in enumerate(dates):
        market_data[date] = {key: scaled_data[i][j] for j, key in enumerate(fieldnames)}

    return market_data


def prepare_stock_data(file_path: str) -> Dict[
    str, Dict[str, Union[Dict[str, Union[float, int]], Dict[datetime.date, Dict[str, float]]]]]:
    df = pd.read_csv(file_path, parse_dates=['date'])
    stock_data: Dict[str, Dict[str, Union[Dict[str, Union[float, int]], Dict[datetime.date, Dict[str, float]]]]] = {}

    # Label encode industry and sector
    industry_encoder = LabelEncoder()
    sector_encoder = LabelEncoder()
    df['industry'] = industry_encoder.fit_transform(df['industry'])
    df['sector'] = sector_encoder.fit_transform(df['sector'])

    # Min-Max Scale the numerical data
    scaler = MinMaxScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Use tqdm to track progress
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing stock data"):
        ticker = row['ticker']
        date = row['date'].date()  # Convert to datetime.date
        if ticker not in stock_data:
            stock_data[ticker] = {'industry': row['industry'], 'sector': row['sector'], 'history': {}}

        stock_data[ticker]['history'][date] = {col: row[col] for col in df.columns if
                                               col not in ['ticker', 'date', 'industry', 'sector']}

    return stock_data


def prepare_merged_data(
        market_data: Dict[str, Dict[str, float]],
        stock_data: Dict[str, Dict[str, Union[Dict[str, Union[float, int]], Dict[datetime.date, Dict[str, float]]]]],
        output_file_path: str,
        top: int = None  # Limit the number of elements if specified
) -> Dict[str, Dict[datetime.date, Dict[str, Union[float, int]]]]:
    combined_data: Dict[str, Dict[datetime.date, Dict[str, Union[float, int]]]] = {}

    for ticker, ticker_data in tqdm(stock_data.items(), desc="Merging data"):
        combined_ticker_data: Dict[datetime.date, Dict[str, Union[float, int]]] = {}
        ticker_info = {
            'industry': ticker_data['industry'],
            'sector': ticker_data['sector']
        }

        for date, stock_values in ticker_data['history'].items():
            # Convert market_data date keys to date objects
            market_date = pd.to_datetime(date).date()
            if market_date in market_data:
                # Combine stock data and market data for the date
                combined_values = {}
                for sd_key, sd_value in stock_values.items():
                    combined_values[sd_key] = sd_value
                for md_key, md_value in market_data[market_date].items():
                    combined_values[md_key] = md_value
                for ti_key, ti_value in ticker_info.items():
                    combined_values[ti_key] = ti_value
                combined_ticker_data[date] = combined_values

        combined_data[ticker] = combined_ticker_data

    # Limit the combined data to the top N elements if specified
    if top is not None:
        combined_data = dict(sorted(combined_data.items(), key=lambda item: len(item[1]), reverse=True)[:top])

    # Prepare data for CSV
    rows = []
    for ticker, dates in combined_data.items():
        for date, values in dates.items():
            row = {'ticker': ticker, 'date': date}
            row.update(values)
            rows.append(row)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_file_path, index=False)

    return combined_data


def read_prepared_merged_data(
        file_path: str,
) -> Dict[str, Dict[datetime.date, Dict[str, Union[float, int]]]]:
    # Read from CSV
    df = pd.read_csv(file_path, parse_dates=['date'])
    combined_data: Dict[datetime.date, Dict[Union[str, pd.Timestamp], Dict[str, Union[float, int]]]] = {}

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Reading merged data"):
        ticker = row['ticker']
        date = row['date'].date()  # Convert to datetime.date
        if ticker not in combined_data:
            combined_data[ticker] = {}
        combined_data[ticker][date] = {col: row[col] for col in df.columns if col not in ['ticker', 'date']}

    return combined_data


def prepare_sequenced_data(
        combined_data: Dict[str, Dict[datetime.date, Dict[str, Union[float, int]]]],
        sequence_length: int = 30,
        n_ahead: int = 5
) -> (np.ndarray, np.ndarray):
    X, y = [], []
    for ticker, data in combined_data.items():
        dates = sorted(data.keys())

        for i in tqdm(range(len(dates) - sequence_length - n_ahead + 1), desc="Sequencing data for " + ticker):
            # Create a sequence of vectors
            sequence = []
            for j in range(sequence_length):
                date = dates[i + j]
                vector = list(data[date].values())
                sequence.append(vector)

            X.append(sequence)

            # Target is the 'Open' and 'Close' prices for the next n days
            target = []
            for k in range(n_ahead):
                next_date = dates[i + sequence_length + k]
                target.extend([data[next_date]['Open'], data[next_date]['Close'], data[next_date]['Volume']])
            y.append(target)

    X, y = np.array(X), np.array(y)

    return X, y


def split_data(X: np.ndarray, y: np.ndarray, test_split: float, output_dir: str):
    # Calculate the split index
    split_index = math.ceil(len(X) * test_split)

    # Split the data
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to .npy files
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # market_data = prepare_market_data('processed/market.csv')
    # stock_data = prepare_stock_data('processed/sp500.csv')
    # merged_data = prepare_merged_data(market_data, stock_data, 'prepared/merged_data_top_50.csv', 50)
    # merged_data = read_prepared_merged_data('prepared/merged_data_top_50.csv')
    # X, y = prepare_sequenced_data(merged_data, 30, n_ahead=5)
    # split_data(X, y, 0.8, 'prepared/sequenced')

    # Load the data from .npy files
    X_train = np.load('prepared/sequenced/X_train.npy')
    X_test = np.load('prepared/sequenced/X_test.npy')
    y_train = np.load('prepared/sequenced/y_train.npy')
    y_test = np.load('prepared/sequenced/y_test.npy')

    # Check the shapes to ensure they loaded correctly
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
