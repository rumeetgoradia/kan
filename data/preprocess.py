import datetime
from typing import Dict, List, Union
from tqdm import tqdm


import pandas as pd
import yfinance as yf


def preprocess_market_data(commodities_csv_file_path: str, indices_csv_file_path: str, output_file_path: str):
    # Load the first CSV
    commodities_csv = pd.read_csv(commodities_csv_file_path)

    # Load the second CSV
    indices_csv = pd.read_csv(indices_csv_file_path)

    # Select the required columns from the second CSV
    indices_csv = indices_csv[['Date', 'Dow Jones (^DJI)', 'Nasdaq (^IXIC)', 'S&P500 (^GSPC)', 'NYSE Composite (^NYA)',
                               'Russell 2000 (^RUT)', 'CBOE Volitility (^VIX)', 'Treasury Yield 5 Years (^FVX)',
                               'Treasury Bill 13 Week (^IRX)', 'Treasury Yield 10 Years (^TNX)',
                               'Treasury Yield 30 Years (^TYX)']]

    # Rename columns to use symbols only
    indices_csv.columns = ['Date', '^DJI', '^IXIC', '^GSPC', '^NYA', '^RUT', '^VIX', '^FVX', '^IRX', '^TNX', '^TYX']

    # Convert percentage columns to decimal
    for col in ['^VIX', '^FVX', '^IRX', '^TNX', '^TYX']:
        indices_csv[col] = indices_csv[col] / 100

    # Pivot CSV #1 to have tickers as columns
    csv1_pivot = commodities_csv.pivot(index='date', columns='ticker', values='close').reset_index()

    # Merge the pivoted CSV #1 with CSV #2 on the date column
    merged = pd.merge(csv1_pivot, indices_csv, left_on='date', right_on='Date', how='inner')

    # Sort by date to ensure proper forward filling
    merged.sort_values(by='date', inplace=True)

    # Forward fill missing values
    merged.fillna(method='ffill', inplace=True)

    # Drop the extra 'Date' column from the merge
    merged.drop(columns=['Date'], inplace=True)

    # Save the consolidated CSV
    merged.to_csv(output_file_path, index=False)

    print(f"Processed data saved to {output_file_path}")


def preprocess_stock_data(symbols_file_path: str, file_path: str):
    tickers = []
    with open(symbols_file_path, mode='r') as file:
        for line in file:
            tickers.append(line.strip())

    yf_tickers = yf.Tickers(" ".join(tickers))
    stock_data: Dict[str, Dict[Union[str, datetime.date], Union[Dict[str, float], str]]] = {}

    # Process each ticker
    for ticker in tqdm(tickers, desc="Processing tickers"):
        try:
            ticker_info = yf_tickers.tickers[ticker].info
            industry = ticker_info.get('industryKey', 'Unknown')
            sector = ticker_info.get('sectorKey', 'Unknown')

            ticker_df = yf_tickers.tickers[ticker].history(period='max', interval='1d')
            ticker_df.reset_index(inplace=True)
            ticker_df['Date'] = pd.to_datetime(ticker_df['Date']).dt.date

            # Create the nested dictionary structure
            ticker_data: Dict[Union[str, datetime.date], Union[Dict[str, float], str]] = {}
            for _, row in ticker_df.iterrows():
                date = row['Date']
                ticker_data[date] = {col: row[col] for col in ticker_df.columns if col != 'Date'}

            # Add industry and sector
            ticker_data['industry'] = industry
            ticker_data['sector'] = sector
            stock_data[ticker] = ticker_data
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    rows = []
    for ticker, data in stock_data.items():
        industry = data['industry']
        sector = data['sector']
        for date, values in data.items():
            if isinstance(date, datetime.date):
                row = {'ticker': ticker, 'date': date}
                row.update(values)
                row['industry'] = industry
                row['sector'] = sector
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)

    print(f"Processed stock data saved to {file_path}")


if __name__ == '__main__':
    preprocess_market_data('raw/commodities.csv', 'raw/indices.csv', 'processed/market.csv')
    preprocess_stock_data('raw/sp500.txt', 'processed/sp500.csv')
