import logging
import time

import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_model(input_shape, units, dropout_rate=0.2):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units),
        Dropout(dropout_rate),
        Dense(15)
    ])
    return model


def train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs, learning_rate):
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    return history


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f'Test Results - MSE: {mse:.8f}, MAE: {mae:.8f}, R2: {r2:.8f}')

    return mse, mae, r2


def main(X_train_path: str, y_train_path: str, X_test_path: str, y_test_path: str):
    start_time = time.time()
    logger.info("Starting the main function")

    # Load data
    logger.info("Loading data...")
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    logger.info(
        f"Data loaded. Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Split training data into train and validation
    val_size = int(0.2 * len(X_train))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    # Hyperparameters
    batch_size = 64
    epochs = 100
    learning_rate = 0.001
    units = 64
    dropout_rate = 0.2

    # Create and train the model
    model = create_model(X_train.shape[1:], units, dropout_rate)
    logger.info(f"Model created with {units} units and dropout rate: {dropout_rate}")

    history = train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs, learning_rate)

    # Evaluate the model
    mse, mae, r2 = evaluate_model(model, X_test, y_test)

    # Save metrics to file
    with open('model_metrics.txt', 'w') as f:
        f.write(f'Mean Squared Error: {mse}\n')
        f.write(f'Mean Absolute Error: {mae}\n')
        f.write(f'R-squared Score: {r2}\n')

    logger.info('Testing completed. Metrics saved to model_metrics.txt')

    # Save the model
    model.save('lstm_model.h5')
    logger.info('Model saved as lstm_model.h5')

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main('../data/prepared/sequenced/X_train.npy', '../data/prepared/sequenced/y_train.npy',
         '../data/prepared/sequenced/X_test.npy', '../data/prepared/sequenced/y_test.npy')
