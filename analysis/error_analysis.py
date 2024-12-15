import argparse
import json
import logging
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def predicted_vs_actual(data, output_dir):
    """
    Create improved plots of predicted vs actual values for each model,
    handling 3D time series data (samples, look ahead factor, number of output features).
    """
    logger.info("Creating predicted vs actual plots...")

    output_dir_full = os.path.join(output_dir, 'predicted_vs_actual')
    if not os.path.exists(output_dir_full):
        os.makedirs(output_dir_full)

    actual = np.array(data['actual'])

    # Flatten the actual values
    actual_flat = actual.reshape(-1)

    for model, predictions in tqdm(data['predictions'].items(), desc="Processing models"):
        predicted = np.array(predictions)

        # Flatten the predicted values
        predicted_flat = predicted.reshape(-1)

        plt.figure(figsize=(12, 10))

        # Hexbin plot
        plt.hexbin(actual_flat, predicted_flat, gridsize=50, cmap='Blues')
        plt.colorbar(label='Count in bin')

        # Add identity line
        min_val = min(actual_flat.min(), predicted_flat.min())
        max_val = max(actual_flat.max(), predicted_flat.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        # Add contour lines
        h, xedges, yedges = np.histogram2d(actual_flat, predicted_flat, bins=50)
        plt.contour(h.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    cmap='Reds', alpha=0.5)

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Predicted vs Actual Values - {model}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_full, f"predicted_vs_actual_{model}.png"), dpi=300)
        plt.close()

    logger.info("Predicted vs actual plots saved.")


def error_heatmap(data, output_dir):
    """
    Create heatmaps of errors across different features and time periods.
    This can show if certain models struggle with specific types of data or time periods.
    """
    logger.info("Creating error heatmaps...")

    output_dir_full = os.path.join(output_dir, 'error_heatmaps')
    if not os.path.exists(output_dir_full):
        os.makedirs(output_dir_full)

    actual = np.array(data['actual'])

    # Create a summary heatmap showing average errors for all models
    plt.figure(figsize=(20, 12))
    avg_errors_all_models = []
    model_names = []

    for model, predictions in data['predictions'].items():
        errors = np.array(predictions) - actual
        avg_errors = np.mean(errors, axis=(1, 2))  # Average across prediction steps and features
        avg_errors_all_models.append(avg_errors)
        model_names.append(model)

    summary_heatmap = np.array(avg_errors_all_models)

    # Calculate the maximum absolute error for symmetric color scaling
    max_abs_error = np.max(np.abs(summary_heatmap))

    # Create the heatmap with improved aesthetics
    sns.heatmap(summary_heatmap, cmap='coolwarm', center=0,
                yticklabels=model_names,
                vmin=-max_abs_error, vmax=max_abs_error,
                cbar_kws={'label': 'Average Error'})

    plt.title("Average Error Heatmap - All Models", fontsize=16)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Models", fontsize=12)

    # Improve x-axis labels
    num_ticks = 10
    step = len(summary_heatmap[0]) // num_ticks
    plt.xticks(range(0, len(summary_heatmap[0]), step),
               range(0, len(summary_heatmap[0]), step),
               rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_full, 'error_heatmap_all_models_summary.png'))
    plt.close()

    logger.info("Summary error heatmap saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze prediction errors")
    parser.add_argument("--predictions-file", required=True, help="Compiled predictions file path")
    parser.add_argument("--output-dir", required=True, help="Path to the directory to save output plots")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_json(args.predictions_file)

    if data is None:
        logger.error("Failed to load predictions data. Exiting.")
        exit(1)

    # Run analyses
    predicted_vs_actual(data, args.output_dir)
    error_heatmap(data, args.output_dir)

    logger.info("Error analysis complete. Plots saved in: %s", args.output_dir)
