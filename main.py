import argparse
import json
import logging
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tqdm import tqdm

from data.processor import prepare_data, split_data
from data.window_generator import WindowGenerator
from network.kan import ChebyshevKANLayer, BSplineKANLayer
from network.ts_kan import TimeSeriesKAN

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_lstm_model(input_shape, output_features, label_width, units=64, dropout_rate=0.2):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = LSTM(units, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(output_features * label_width)(x)
    outputs = keras.layers.Reshape((label_width, output_features))(outputs)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_mlp_model(input_shape, output_features, label_width, units=64, dropout_rate=0.2):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Flatten()(inputs)
    x = Dense(units, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units // 2, activation='relu', kernel_initializer='he_normal')(x)
    outputs = Dense(output_features * label_width, activation='linear')(x)
    outputs = keras.layers.Reshape((label_width, output_features))(outputs)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_ts_kan_model(input_shape, output_features, label_width, kan_type='chebyshev', hidden_size=64, kan_size=32,
                        **kan_kwargs):
    if kan_type.lower() == 'chebyshev':
        kan_layer = ChebyshevKANLayer(out_features=kan_size, **kan_kwargs)
    elif kan_type.lower() == 'bspline':
        kan_layer = BSplineKANLayer(out_features=kan_size, **kan_kwargs)
    else:
        raise ValueError(f"Unsupported KAN type: {kan_type}")
    model = TimeSeriesKAN(hidden_size=hidden_size, output_size=output_features * label_width, kan_layer=kan_layer)
    # Add a Reshape layer to the model if necessary
    model.add(keras.layers.Reshape((label_width, output_features)))
    return model


def create_and_compile_model(model_type, input_shape, output_features, label_width, learning_rate=0.001, **kwargs):
    if model_type.lower() == 'lstm':
        model = create_lstm_model(input_shape, output_features, label_width, **kwargs)
    elif model_type.lower() == 'mlp':
        model = create_mlp_model(input_shape, output_features, label_width, **kwargs)
    elif model_type.lower() in ['chebyshev', 'bspline']:
        model = create_ts_kan_model(input_shape, output_features, label_width, kan_type=model_type, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    return model


def check_for_nan(tensor):
    return tf.reduce_any(tf.math.is_nan(tensor))


def train_model(model, train_data, val_data, epochs):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    class NanCallback(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if np.isnan(logs.get('loss')):
                print(f"NaN loss encountered at batch {batch}")
                self.model.stop_training = True

    def filter_nans(x, y):
        is_finite = tf.reduce_all(tf.math.is_finite(x)) & tf.reduce_all(tf.math.is_finite(y))
        return is_finite

    logger.info("Removing NaNs from data...")
    filtered_train_data = train_data.batch(32).filter(filter_nans).unbatch()
    filtered_val_data = val_data.batch(32).filter(filter_nans).unbatch()

    # Calculate steps per epoch
    train_steps = sum(1 for _ in filtered_train_data)
    val_steps = sum(1 for _ in filtered_val_data)

    logger.info("Training model...")
    history = model.fit(
        filtered_train_data,
        validation_data=filtered_val_data,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[early_stopping, lr_scheduler, NanCallback()],
        verbose=1
    )
    return history


def evaluate_model(model, test_data):
    y_true = []
    y_pred = []
    for x, y in test_data:
        y_true.append(y.numpy())
        y_pred.append(model.predict(x))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred).numpy()
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred).numpy()
    rmse = np.sqrt(mse)

    # R2 score calculation
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-7))  # adding small epsilon to avoid division by zero

    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2)
    }


def save_metrics(metrics, file_path):
    with open(file_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f'{key}: {value}\n')


def combine_windows(windows):
    combined_train = windows[0].train
    combined_val = windows[0].val
    combined_test = windows[0].test

    for window in windows[1:]:
        combined_train = combined_train.concatenate(window.train)
        combined_val = combined_val.concatenate(window.val)
        combined_test = combined_test.concatenate(window.test)

    combined_train = combined_train.shuffle(buffer_size=10000)
    combined_val = combined_val.shuffle(buffer_size=10000)
    combined_test = combined_test.shuffle(buffer_size=10000)

    return combined_train, combined_val, combined_test


def get_or_create_ticker_split(df, train_test_ratio, file_path='ticker_split.json'):
    if os.path.exists(file_path):
        logger.info(f"Loading existing ticker split from {file_path}")
        with open(file_path, 'r') as f:
            split = json.load(f)
        train_tickers = split['train']
        test_tickers = split['test']
    else:
        logger.info("Creating new ticker split")
        all_tickers = df['ticker'].unique().tolist()
        train_tickers, test_tickers = train_test_split(all_tickers, train_size=train_test_ratio)

        split = {
            'train': train_tickers,
            'test': test_tickers
        }
        with open(file_path, 'w') as f:
            json.dump(split, f)
        logger.info(f"Saved ticker split to {file_path}")

    return train_tickers, test_tickers


def main(model_type, kan_type=None, train=True):
    tf.keras.backend.set_floatx('float64')

    # Parameters
    input_width = 30
    label_width = 5
    shift = 5
    train_test_ratio = 0.8

    logger.info("Preparing data...")
    df, input_features, output_features, _, _, _ = prepare_data(
        stock_file='data/processed/sp500.csv',
        market_file='data/processed/market.csv'
    )

    logger.info("Getting train/test ticker split...")
    train_tickers, test_tickers = get_or_create_ticker_split(df, train_test_ratio)

    logger.info("Creating windows for each ticker...")
    train_windows = []
    test_windows = []

    for ticker in tqdm(train_tickers, desc="Windowing train tickers"):
        train_df, val_df, _ = split_data(df[df['ticker'] == ticker], ticker)
        window = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            train_df=train_df,
            val_df=val_df,
            test_df=val_df,
            input_features=input_features,
            output_features=output_features
        )
        train_windows.append(window)

    for ticker in tqdm(test_tickers, desc="Windowing test tickers"):
        _, _, test_df = split_data(df[df['ticker'] == ticker], ticker)
        window = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            train_df=test_df,
            val_df=test_df,
            test_df=test_df,
            input_features=input_features,
            output_features=output_features
        )
        test_windows.append(window)

    logger.info("Combining and shuffling windows...")
    train_data, val_data, _ = combine_windows(train_windows)
    _, _, test_data = combine_windows(test_windows)

    input_shape = (input_width, len(input_features))

    logger.info(f"Creating {model_type} model...")
    if model_type.lower() in ['chebyshev', 'bspline']:
        model = create_and_compile_model(kan_type, input_shape, len(output_features), label_width, learning_rate=0.001,
                                         hidden_size=64, kan_size=32)
    else:
        model = create_and_compile_model(model_type, input_shape, len(output_features), label_width,
                                         learning_rate=0.001)

    if train:
        train_model(model, train_data, val_data, epochs=100)
        logger.info("Saving model...")
        model.save(f'results/{model_type}_model.keras')
    else:
        logger.info("Loading pre-trained model...")
        model = keras.models.load_model(f'results/{model_type}_model.keras')

    logger.info("Evaluating model...")
    metrics = evaluate_model(model, test_data)
    save_metrics(metrics, f'results/{model_type}_metrics.txt')

    for key, value in metrics.items():
        logger.info(f"{key}: {value}")

    logger.info("Process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test a model.")
    parser.add_argument("model_type", choices=['lstm', 'mlp', 'ts_kan'], help="Type of model to use")
    parser.add_argument("--kan_type", choices=['chebyshev', 'bspline'], help="Type of KAN layer (only for ts_kan)")
    parser.add_argument("--test", action="store_true", help="Test the model instead of training")
    args = parser.parse_args()

    if args.model_type == 'ts_kan' and args.kan_type is None:
        parser.error("ts_kan model type requires --kan_type")

    main(args.model_type, args.kan_type, not args.test)
