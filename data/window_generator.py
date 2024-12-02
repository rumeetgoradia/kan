import logging

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df=None, val_df=None, test_df=None,
                 label_columns=None):
        # Store the raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        # Work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        # Determine which DataFrame to use for column indices
        if train_df is not None:
            df_for_columns = train_df
        elif val_df is not None:
            df_for_columns = val_df
        else:
            df_for_columns = test_df
        self.column_indices = {name: i for i, name in enumerate(df_for_columns.columns)}
        # Work out the window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the tf.data.Datasets are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, data):
        datasets = []
        logger.info(f"Creating datasets for {len(data['ticker'].unique())} tickers")
        for ticker in tqdm(data['ticker'].unique(), desc="Processing tickers"):
            # Filter data for this ticker and sort by date
            ticker_data = data[data['ticker'] == ticker].sort_values('date')
            # Drop 'ticker' and 'date' columns
            numeric_data = ticker_data.drop(columns=['ticker', 'date'])

            # Check for NaN values
            if np.isnan(numeric_data.values).any():
                logger.warning(f"NaN values found in data for ticker {ticker}")
                numeric_data = numeric_data.fillna(method='ffill').fillna(method='bfill')

            data_array = np.array(numeric_data, dtype=np.float32)
            ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data_array,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=False,  # Keep the time order
                batch_size=32, )
            ds = ds.map(self.split_window)
            datasets.append(ds)

        logger.info("Combining datasets from all tickers")
        combined_ds = datasets[0]
        for ds in tqdm(datasets[1:], desc="Combining datasets"):
            combined_ds = combined_ds.concatenate(ds)

        logger.info("Shuffling the combined dataset")
        combined_ds = combined_ds.shuffle(buffer_size=1000)
        return combined_ds

    @property
    def train(self):
        return self.make_dataset(self.train_df) if self.train_df is not None else None

    @property
    def val(self):
        return self.make_dataset(self.val_df) if self.val_df is not None else None

    @property
    def test(self):
        return self.make_dataset(self.test_df) if self.test_df is not None else None
