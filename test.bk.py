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
from network import ThreeDimensionalR2Score
from network.kan import TimeSeriesKAN, TimeSeriesKANAttentionLayer
from network.kan.layer import *
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError

# Set up directories
MODELS_DIR = 'results/models'
METRICS_DIR = 'results/metrics'
ANALYSIS_DIR = 'results/analysis'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Parameters
input_width = 30
label_width = 5
output_features = 3
shift = 5


def custom_load_model(model_path, learning_rate=0.001):
    custom_objects = {
        'BaseKANLayer': BaseKANLayer,
        'BSplineKANLayer': BSplineKANLayer,
        'ChebyshevKANLayer': ChebyshevKANLayer,
        'FourierKANLayer': FourierKANLayer,
        'LegendreKANLayer': LegendreKANLayer,
        'TimeSeriesKAN': TimeSeriesKAN,
        'AttentionLayer': TimeSeriesKANAttentionLayer,
        'CustomR2Score': ThreeDimensionalR2Score
    }

    # Load the model with custom objects
    model = load_model(model_path, custom_objects=custom_objects, compile=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            MeanAbsoluteError(name='mae'),
            MeanSquaredError(name='mse'),
            RootMeanSquaredError(name='rmse'),
            ThreeDimensionalR2Score(name='r2')
        ]
    )

    return model


# Load models
models = {}
for file in os.listdir(MODELS_DIR):
    if file.endswith('_model.keras'):
        model_name = file.split('_')[0]
        model_path = os.path.join(MODELS_DIR, file)
        try:
            models[model_name] = custom_load_model(model_path)
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
print(f"Loaded models: {list(models.keys())}")


def load_test_data():
    try:
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
    # Reshape the arrays to 2D: (num_samples * look_ahead, num_features)
    y_true_reshaped = y_true.reshape(-1, y_true.shape[-1])
    y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])

    # Calculate metrics
    mse = mean_squared_error(y_true_reshaped, y_pred_reshaped)
    mae = mean_absolute_error(y_true_reshaped, y_pred_reshaped)
    rmse = np.sqrt(mse)

    # Calculate R2 score for each feature
    r2_scores = [r2_score(y_true_reshaped[:, i], y_pred_reshaped[:, i]) for i in range(y_true.shape[-1])]

    # Average R2 score across features
    r2_avg = np.mean(r2_scores)

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2_avg,
        'R2_per_feature': r2_scores
    }


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
# efficiency_data = []
# for name, model in models.items():
#     inference_time, num_params = get_model_stats(model, X_test)
#     efficiency_data.append({
#         'Model': name,
#         'Inference Time (s)': inference_time,
#         'Number of Parameters': num_params
#     })
#
# efficiency_df = pd.DataFrame(efficiency_data)
# print("Computational Efficiency Comparison:")
# print(efficiency_df)
# efficiency_df.to_csv(os.path.join(ANALYSIS_DIR, 'computational_efficiency.csv'), index=False)
#
#
# # Hyperparameter settings table
# def get_hyperparameters(model):
#     hyperparams = {}
#     for layer in model.layers:
#         if isinstance(layer, tf.keras.layers.LSTM):
#             hyperparams['LSTM Units'] = layer.units
#         elif isinstance(layer, tf.keras.layers.Dense):
#             hyperparams['Dense Units'] = layer.units
#
#     # Handle different ways to access learning rate
#     if hasattr(model.optimizer, 'lr'):
#         lr = model.optimizer.lr
#     elif hasattr(model.optimizer, 'learning_rate'):
#         lr = model.optimizer.learning_rate
#     else:
#         lr = None
#
#     if lr is not None:
#         if callable(lr):
#             hyperparams['Learning Rate'] = lr().numpy()
#         else:
#             hyperparams['Learning Rate'] = lr.numpy()
#     else:
#         hyperparams['Learning Rate'] = 'Unknown'
#
#     return hyperparams
#
#
# hyperparameter_data = []
# for name, model in models.items():
#     if not hasattr(model, 'optimizer') or model.optimizer is None:
#         print(f"Warning: Model {name} is not compiled. Skipping hyperparameter extraction.")
#         continue
#     hyperparams = get_hyperparameters(model)
#     hyperparams['Model'] = name
#     hyperparameter_data.append(hyperparams)
# hyperparameter_df = pd.DataFrame(hyperparameter_data)
# print("\nHyperparameter Settings:")
# print(hyperparameter_df)
# hyperparameter_df.to_csv(os.path.join(ANALYSIS_DIR, 'hyperparameter_settings.csv'), index=False)

# Calculate and save metrics for each model
metrics_data = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    print("Calculated metrics for", name)
    metrics['Model'] = name
    metrics_data.append(metrics)
metrics_df = pd.DataFrame(metrics_data)
print("\nModel Performance Metrics:")
print(metrics_df)
metrics_df.to_csv(os.path.join(ANALYSIS_DIR, 'model_metrics.csv'), index=False)

# Figures
# Predicted vs. Actual Values (for the first output feature and first time step)
plt.figure(figsize=(12, 6))
plt.plot(y_test[:100, 0, 0], label='Actual', alpha=0.7)
for name, model in models.items():
    y_pred = model.predict(X_test)
    plt.plot(y_pred[:100, 0, 0], label=f'{name} Predicted', alpha=0.7)
plt.legend()
plt.title('Predicted vs Actual Values (First 100 samples, first feature, first time step)')
plt.xlabel('Sample')
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
    rmse = np.sqrt(mean_squared_error(y_test.reshape(-1, y_test.shape[-1]),
                                      y_pred.reshape(-1, y_pred.shape[-1])))
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
    error_df = pd.DataFrame({
        'Model': [name] * errors.size,
        'Error': errors.flatten(),
        'Feature': np.tile(np.repeat(range(errors.shape[2]), errors.shape[1]), errors.shape[0]),
        'Time Step': np.tile(np.repeat(range(errors.shape[1]), errors.shape[2]), errors.shape[0])
    })
    error_data.append(error_df)
error_df = pd.concat(error_data, ignore_index=True)

plt.figure(figsize=(12, 6))
sns.violinplot(x='Model', y='Error', data=error_df)
plt.title('Distribution of Prediction Errors (All Features and Time Steps)')
plt.savefig(os.path.join(ANALYSIS_DIR, 'error_distribution_all.png'))
plt.close()

# Distribution of Prediction Errors by Feature
plt.figure(figsize=(15, 5 * ((len(models) - 1) // 3 + 1)))
for i, feature in enumerate(range(y_test.shape[2])):
    plt.subplot(((len(models) - 1) // 3 + 1), 3, i + 1)
    sns.violinplot(x='Model', y='Error', data=error_df[error_df['Feature'] == feature])
    plt.title(f'Distribution of Prediction Errors (Feature {feature})')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'error_distribution_by_feature.png'))
plt.close()

# Distribution of Prediction Errors by Time Step
plt.figure(figsize=(15, 5 * ((len(models) - 1) // 3 + 1)))
for i, time_step in enumerate(range(y_test.shape[1])):
    plt.subplot(((len(models) - 1) // 3 + 1), 3, i + 1)
    sns.violinplot(x='Model', y='Error', data=error_df[error_df['Time Step'] == time_step])
    plt.title(f'Distribution of Prediction Errors (Time Step {time_step})')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'error_distribution_by_time_step.png'))
plt.close()

print(f"\nAnalysis complete. Results saved in {ANALYSIS_DIR}")
