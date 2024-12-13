# Data
STOCK_MARKET_DATA_FILEPATH = 'data/processed/stock_market.csv'
TICKER_SPLITS_FILEPATH = 'data/processed/ticker_splits.json'

# Data processing
SEQUENCE_LENGTH = 30
LOOKAHEAD = 5
OUTPUT_FEATURES = ['Open', 'Close', 'Volume']

# Model parameters
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 0.001

# Train-Val-Test split
TRAIN_KEY = 'train'
VAL_KEY = 'val'
TEST_KEY = 'test'
SPLIT_RATIOS = {TRAIN_KEY: 0.7, VAL_KEY: 0.15, TEST_KEY: 0.15}

# Random seed for reproducibility
RANDOM_SEED = 42

MODEL_FILE_NAME = lambda model_name: f"{model_name}_model.keras"
TRAIN_METRICS_FILE_NAME = lambda model_name: f"{model_name}_train_metrics.json"
TEST_METRICS_FILE_NAME = lambda model_name: f"{model_name}_test_metrics.json"
