"""
rg3072
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        return None


def get_model_complexity(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model.count_params()
    except:
        logger.error(f"Failed to load model: {model_path}")
        return None


def complexity_vs_performance(compiled_metrics, models_dir, output_dir):
    """
    Plot a scatter of model complexity (number of parameters) against performance metrics.
    This can show if more complex models are justified by their performance improvements.
    """
    logger.info("Analyzing model complexity vs performance...")

    output_dir_full = os.path.join(output_dir, 'complexity_vs_performance')
    if not os.path.exists(output_dir_full):
        os.makedirs(output_dir_full)

    model_data = []
    for model_name, metrics in compiled_metrics.items():
        model_path = Path(models_dir) / f"{model_name}_model.keras"
        num_params = get_model_complexity(model_path)
        if num_params is not None:
            model_data.append({
                'Model': model_name,
                'Number of Parameters': num_params,
                'Test MSE': metrics['test']['mse'],
                'Test MAE': metrics['test']['mae'],
                'Test R2': metrics['test']['r2']
            })

    df = pd.DataFrame(model_data)

    metrics = ['Test MSE', 'Test MAE', 'Test R2']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Number of Parameters', y=metric, hue='Model')
        plt.title(f"Model Complexity vs {metric}")
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_full, f"complexity_vs_{metric.lower().replace(' ', '_')}.png"))
        plt.close()

    logger.info("Model complexity analysis plots saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model complexity vs performance")
    parser.add_argument("--metrics-file", required=True, help="Path to the compiled metrics JSON file")
    parser.add_argument("--models-dir", required=True, help="Directory containing the model files")
    parser.add_argument("--output-dir", default='plots', help="Path to the directory to save output plots")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load compiled metrics
    compiled_metrics = load_json(args.metrics_file)

    if compiled_metrics is None:
        logger.error("Failed to load compiled metrics. Exiting.")
        exit(1)

    # Run analysis
    complexity_vs_performance(compiled_metrics, args.models_dir, args.output_dir)

    logger.info("Model complexity analysis complete. Plots saved in: %s", args.output_dir)
