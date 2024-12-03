import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

from data.processor import get_ticker_split, prepare_data
from data.window_generator import WindowGenerator

# Set up directories
MODELS_DIR = 'results/models'
METRICS_DIR = 'results/metrics'
ANALYSIS_DIR = 'results/analysis'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Load models
models = {}
for file in os.listdir(MODELS_DIR):
    if file.endswith('_model.keras'):
        model_name = file.split('_')[0]
        model_path = os.path.join(MODELS_DIR, file)
        try:
            models[model_name] = load_model(model_path)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
print(f"Loaded models: {list(models.keys())}")


def load_test_data():
    try:
        # Parameters
        input_width = 30
        label_width = 5
        shift = 5
        # Prepare data
        df, input_features, output_features, stock_scalers, market_scaler, encoders = prepare_data(
            'data/processed/sp500.csv',
            'data/processed/market.csv'
        )
        _, _, test_tickers = get_ticker_split('data/ticker_split.json')
        # Create WindowGenerator for test data only
        window_generator = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            train_df=None,
            val_df=None,
            test_df=df[df['ticker'].isin(test_tickers)],
            label_columns=output_features
        )
        # Get the test dataset
        test_data = window_generator.test
        return test_data, len(output_features)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None, None


def load_training_history(model_name):
    history_path = os.path.join(METRICS_DIR, f'{model_name}_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: No training history found for {model_name}")
        return None


def get_model_stats(model, X_test):
    start_time = time.time()
    model.predict(X_test)
    inference_time = time.time() - start_time
    num_params = model.count_params()
    return inference_time, num_params


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}


# Load test data
test_data, num_features = load_test_data()
if test_data is None:
    print("Failed to load test data. Exiting.")
    exit(1)

# Separate features and labels
X_test, y_test = [], []
for x, y in test_data:
    X_test.append(x)
    y_test.append(y)

X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

# Computational efficiency comparison
efficiency_data = []
for name, model in models.items():
    inference_time, num_params = get_model_stats(model, X_test)
    efficiency_data.append({
        'Model': name,
        'Inference Time (s)': inference_time,
        'Number of Parameters': num_params
    })

efficiency_df = pd.DataFrame(efficiency_data)
print("Computational Efficiency Comparison:")
print(efficiency_df)
efficiency_df.to_csv(os.path.join(ANALYSIS_DIR, 'computational_efficiency.csv'), index=False)


# Hyperparameter settings table
def get_hyperparameters(model):
    hyperparams = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LSTM):
            hyperparams['LSTM Units'] = layer.units
        elif isinstance(layer, tf.keras.layers.Dense):
            hyperparams['Dense Units'] = layer.units
    hyperparams['Learning Rate'] = model.optimizer.lr.numpy()
    return hyperparams


hyperparameter_data = []
for name, model in models.items():
    hyperparams = get_hyperparameters(model)
    hyperparams['Model'] = name
    hyperparameter_data.append(hyperparams)
hyperparameter_df = pd.DataFrame(hyperparameter_data)
print("\nHyperparameter Settings:")
print(hyperparameter_df)
hyperparameter_df.to_csv(os.path.join(ANALYSIS_DIR, 'hyperparameter_settings.csv'), index=False)

# Calculate and save metrics for each model
metrics_data = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    metrics['Model'] = name
    metrics_data.append(metrics)
metrics_df = pd.DataFrame(metrics_data)
print("\nModel Performance Metrics:")
print(metrics_df)
metrics_df.to_csv(os.path.join(ANALYSIS_DIR, 'model_metrics.csv'), index=False)

# Figures
# Predicted vs. Actual Values (for the first output feature)
plt.figure(figsize=(12, 6))
plt.plot(y_test[:100, 0], label='Actual', alpha=0.7)
for name, model in models.items():
    y_pred = model.predict(X_test)
    plt.plot(y_pred[:100, 0], label=f'{name} Predicted', alpha=0.7)
plt.legend()
plt.title('Predicted vs Actual Values (First 100 samples, first feature)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig(os.path.join(ANALYSIS_DIR, 'predicted_vs_actual.png'))
plt.close()

# Learning Curves
for name, model in models.items():
    history = load_training_history(name)
    if history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'Learning Curves - {name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(ANALYSIS_DIR, f'learning_curves_{name}.png'))
        plt.close()

# Computational Efficiency vs. Prediction Accuracy
plt.figure(figsize=(10, 6))
for name, model in models.items():
    inference_time, _ = get_model_stats(model, X_test)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    plt.scatter(inference_time, rmse, label=name)
plt.xlabel('Inference Time (s)')
plt.ylabel('RMSE')
plt.title('Computational Efficiency vs. Prediction Accuracy')
plt.legend()
plt.savefig(os.path.join(ANALYSIS_DIR, 'efficiency_vs_accuracy.png'))
plt.close()

# Distribution of Prediction Errors
error_data = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    errors = y_test - y_pred
    error_data.append(pd.DataFrame({'Model': name, 'Error': errors.flatten()}))
error_df = pd.concat(error_data)
plt.figure(figsize=(12, 6))
sns.violinplot(x='Model', y='Error', data=error_df)
plt.title('Distribution of Prediction Errors')
plt.savefig(os.path.join(ANALYSIS_DIR, 'error_distribution.png'))
plt.close()

print(f"\nAnalysis complete. Results saved in {ANALYSIS_DIR}")
