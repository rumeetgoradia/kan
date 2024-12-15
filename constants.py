"""
rg3072
"""

from network import LSTMNetwork, MLPNetwork
from network.kan.v3 import TimeSeriesKANV3

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

KAN_MODEL_TYPES = [TimeSeriesKANV3.NAME + "-" + kan for kan in TimeSeriesKANV3.ALLOWED_KAN_LAYERS]
MODEL_TYPES = KAN_MODEL_TYPES + [LSTMNetwork.NAME, MLPNetwork.NAME]
