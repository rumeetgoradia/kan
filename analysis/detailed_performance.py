import argparse
import json
import logging
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

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


def error_distribution(data, output_dir):
    """
    Create box plots of prediction errors for each model.
    This shows the spread and central tendency of errors.
    """
    logger.info("Creating error distribution plots...")

    output_dir_full = os.path.join(output_dir, "error_distribution")
    if not os.path.exists(output_dir_full):
        os.makedirs(output_dir_full)

    errors = {}
    for model, preds in data['predictions'].items():
        error = np.array(preds) - np.array(data['actual'])
        errors[model] = error.flatten()  # Flatten the error array

    df = pd.DataFrame(errors)

    # Boxplot
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df)
    plt.title("Error Distribution Across Models (Boxplot)", fontsize=16)
    plt.ylabel("Error", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_full, "error_distribution_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Error distribution boxplot saved.")

    # Histogram with KDE
    plt.figure(figsize=(15, 10))

    # Calculate common x-axis limits
    all_errors = np.concatenate(list(errors.values()))
    x_min, x_max = np.percentile(all_errors, [1, 99])

    for i, (model, error) in enumerate(errors.items()):
        ax = plt.subplot(3, 3, i + 1)
        sns.histplot(error, kde=True, label=model, ax=ax)
        ax.set_xlim(x_min, x_max)
        ax.set_title(model, fontsize=12)
        ax.set_xlabel("Error" if i >= 6 else "")
        ax.set_ylabel("Frequency" if i % 3 == 0 else "")
        ax.legend().set_visible(False)

    plt.suptitle("Error Distribution Across Models (Histogram with KDE)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_full, "error_distribution_histogram.png"), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Error distribution histogram saved.")

    # Overlay KDE plot
    plt.figure(figsize=(15, 8))
    for model, error in errors.items():
        sns.kdeplot(error, label=model)
    plt.title("Error Distribution Across Models (KDE Overlay)", fontsize=16)
    plt.xlabel("Error", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_full, "error_distribution_kde_overlay.png"), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Error distribution KDE overlay plot saved.")


def residual_analysis(data, output_dir):
    """
    Plot residuals (predicted - actual) over time for each model.
    This can reveal any patterns in prediction errors.
    """
    logger.info("Performing residual analysis...")

    output_dir_full = os.path.join(output_dir, "residual_analysis")
    if not os.path.exists(output_dir_full):
        os.makedirs(output_dir_full)

    actual = np.array(data['actual'])

    for model in tqdm(data['predictions'].keys(), desc="Creating residual plots"):
        predictions = np.array(data['predictions'][model])

        # Ensure predictions and actual have the same shape
        if predictions.shape != actual.shape:
            logger.warning(
                f"Shape mismatch for model {model}. Predictions: {predictions.shape}, Actual: {actual.shape}")
            continue

        residuals = predictions - actual

        # Flatten the residuals and create a corresponding time axis
        flat_residuals = residuals.flatten()
        time_steps = np.arange(len(flat_residuals))

        plt.figure(figsize=(15, 8))
        plt.scatter(time_steps, flat_residuals, alpha=0.1, s=1)
        plt.title(f"Residual Plot for {model}", fontsize=16)
        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Residual", fontsize=12)
        plt.axhline(y=0, color='r', linestyle='--')

        # Add a rolling mean line
        window = len(time_steps) // 100  # Adjust this value to change the smoothness
        rolling_mean = np.convolve(flat_residuals, np.ones(window) / window, mode='valid')
        plt.plot(time_steps[window - 1:], rolling_mean, color='green', linewidth=2)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_full, f"residual_plot_{model}.png"), dpi=300)
        plt.close()

    logger.info("Residual analysis plots saved.")

    # Additional summary plot
    plt.figure(figsize=(15, 8))
    for model in data['predictions'].keys():
        predictions = np.array(data['predictions'][model])
        residuals = predictions - actual
        flat_residuals = residuals.flatten()
        sns.kdeplot(flat_residuals, label=model, shade=True)

    plt.title("Residual Distribution Across Models", fontsize=16)
    plt.xlabel("Residual", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_full, "residual_distribution.png"), dpi=300)
    plt.close()

    logger.info("Residual distribution plot saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze detailed model performance")
    parser.add_argument("--predictions-file", required=True, help="Compiled predictions file path")
    parser.add_argument("--output-dir", default='plots', help="Path to the directory to save output plots")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_json(args.predictions_file)

    if data is None:
        logger.error("Failed to load predictions data. Exiting.")
        exit(1)

    # Run analyses
    error_distribution(data, args.output_dir)
    residual_analysis(data, args.output_dir)

    logger.info("Detailed performance analysis complete. Plots saved in: %s", args.output_dir)
