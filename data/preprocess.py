import logging
import os

import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_and_scale_market_data(commodities_csv_file_path: str, indices_csv_file_path: str, output_file_path: str):
    if os.path.exists(output_file_path):
        logger.info(f"Output file {output_file_path} already exists. Loading from file.")
        return pd.read_csv(output_file_path, parse_dates=['Date'], index_col='Date')

    logger.info("Processing and scaling market data...")
    commodities_csv = pd.read_csv(commodities_csv_file_path)
    indices_csv = pd.read_csv(indices_csv_file_path)

    indices_csv = indices_csv[
        ['Date', 'Dow Jones (^DJI)', 'Nasdaq (^IXIC)', 'S&P500 (^GSPC)', 'NYSE Composite (^NYA)',
         'Russell 2000 (^RUT)', 'CBOE Volitility (^VIX)', 'Treasury Yield 5 Years (^FVX)',
         'Treasury Bill 13 Week (^IRX)', 'Treasury Yield 10 Years (^TNX)',
         'Treasury Yield 30 Years (^TYX)']]

    indices_csv.columns = ['Date', '^DJI', '^IXIC', '^GSPC', '^NYA', '^RUT', '^VIX', '^FVX', '^IRX', '^TNX', '^TYX']

    csv1_pivot = commodities_csv.pivot(index='date', columns='ticker', values='close').reset_index()

    merged = pd.merge(csv1_pivot, indices_csv, left_on='date', right_on='Date', how='outer')
    merged.sort_values(by='Date', inplace=True)
    merged.set_index('Date', inplace=True)
    merged = merged.drop(columns=['date'], errors='ignore')
    merged = merged.ffill()
    merged = merged.dropna(how='any')

    scaler = MinMaxScaler()
    merged.loc[:, merged.columns] = scaler.fit_transform(merged)

    merged.to_csv(output_file_path)
    logger.info(f"Processed and scaled data saved to {output_file_path}")

    return merged


def generate_scaled_stock_data(market_data, symbols_file_path, output_file_path):
    # Check if the output file already exists
    if os.path.exists(output_file_path):
        logger.info(f"Output file {output_file_path} already exists. Loading from file.")
        return pd.read_csv(output_file_path, parse_dates=['Date'])

    logger.info("Generating scaled stock data...")
    if 'Date' in market_data.columns:
        market_dates = set(pd.to_datetime(market_data['Date']).dt.date)
    else:
        market_dates = set(pd.to_datetime(market_data.index).date)

    with open(symbols_file_path, mode='r') as file:
        tickers = [line.strip() for line in file]

    yf_tickers = yf.Tickers(" ".join(tickers))

    industry_encoder = LabelEncoder()
    sector_encoder = LabelEncoder()

    all_data = []
    industries = []
    sectors = []

    for ticker in tqdm(tickers, desc="Processing tickers"):
        try:
            ticker_info = yf_tickers.tickers[ticker].info
            industry = ticker_info.get('industryKey', 'Unknown')
            sector = ticker_info.get('sectorKey', 'Unknown')

            industries.append(industry)
            sectors.append(sector)

            ticker_df = yf_tickers.tickers[ticker].history(period='max', interval='1d')
            ticker_df = ticker_df.reset_index()

            ticker_df['Date'] = pd.to_datetime(ticker_df['Date']).dt.date
            ticker_df = ticker_df[ticker_df['Date'].isin(market_dates)]

            if not ticker_df.empty:
                ticker_df = ticker_df.copy()  # Create a copy to avoid SettingWithCopyWarning
                ticker_df['Ticker'] = ticker
                ticker_df['Industry'] = industry
                ticker_df['Sector'] = sector

                columns = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
                           'Industry', 'Sector']
                ticker_df = ticker_df[columns]

                all_data.append(ticker_df)
            else:
                logger.warning(f"No data available for {ticker} within the market date range")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")

    if not all_data:
        raise ValueError(
            "No valid data was processed for any ticker. Please check your input data and market date range.")

    combined_df = pd.concat(all_data, ignore_index=True)

    combined_df['Industry'] = industry_encoder.fit_transform(combined_df['Industry'])
    combined_df['Sector'] = sector_encoder.fit_transform(combined_df['Sector'])

    scaler = MinMaxScaler()
    columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Industry', 'Sector']

    for ticker in tqdm(combined_df['Ticker'].unique(), desc="Scaling tickers"):
        mask = combined_df['Ticker'] == ticker
        combined_df.loc[mask, columns_to_scale] = scaler.fit_transform(
            combined_df.loc[mask, columns_to_scale].astype(float))

    if 'Date' in market_data.columns:
        market_data['Date'] = pd.to_datetime(market_data['Date']).dt.date
        final_df = pd.merge(combined_df, market_data, on='Date', how='left')
    else:
        market_data_reset = market_data.reset_index()
        market_data_reset['Date'] = pd.to_datetime(market_data_reset['Date']).dt.date
        final_df = pd.merge(combined_df, market_data_reset, on='Date', how='left')

    final_df.to_csv(output_file_path, index=False)
    logger.info(f"Processed data saved to {output_file_path}")

    return final_df


if __name__ == '__main__':
    market_df = process_and_scale_market_data('raw/commodities.csv', 'raw/indices.csv', 'processed/market.csv')
    stocks_df = generate_scaled_stock_data(market_df, 'raw/sp500.txt', 'processed/stock_market.csv')
    print(stocks_df.head())
