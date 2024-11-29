import logging

from tqdm import tqdm

from processor import prepare_data, split_data
from window_generator import WindowGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parameters
input_width = 30  # 30 days of input
label_width = 5  # Predict next 5 days
shift = 5  # Shift by 5 days


def main():
    # Prepare data
    df, input_features, output_features, stock_scalers, market_scaler, encoders = prepare_data('processed/sp500.csv',
                                                                                               'processed/market.csv')
    logger.info(f"Data prepared. Shape: {df.shape}")
    logger.info(f"Input features: {input_features}")
    logger.info(f"Output features: {output_features}")

    # Create a dictionary to store WindowGenerator objects for each ticker
    windows = {}

    unique_tickers = df['ticker'].unique()
    logger.info(f"Creating windows for {len(unique_tickers)} tickers...")

    for ticker in tqdm(unique_tickers, desc="Windowing ticker data"):
        train_df, val_df, test_df = split_data(df, ticker)

        window = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            input_features=input_features,
            output_features=output_features
        )

        windows[ticker] = window

    logger.info("Window generation completed.")

    # Access the datasets for each ticker
    logger.info("Accessing datasets for each ticker...")
    for ticker, window in tqdm(windows.items(), desc="Accessing datasets"):
        train_data = window.train
        val_data = window.val
        test_data = window.test

        logger.info(f"Created windows for {ticker}. "
                    f"Train: {len(list(train_data))}, "
                    f"Val: {len(list(val_data))}, "
                    f"Test: {len(list(test_data))} batches")

    logger.info("Process completed successfully.")


if __name__ == "__main__":
    main()
