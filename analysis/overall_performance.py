"""
rg3072
"""

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

METRICS = ['loss', 'mse', 'mae', 'r2']


def overall_metrics_comparison(data, output_dir):
    """
    Create a bar chart comparing metric scores for all models.
    This gives a quick overview of which models perform best overall.
    """
    logger.info("Generating overall metrics comparison chart")
    models = list(data.keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1])
    fig.suptitle('Overall Metrics Comparison', fontsize=16)

    x = np.arange(len(models))
    width = 0.2

    for i, metric in enumerate(METRICS):
        values = [data[model]['test'][metric.lower()] for model in models]

        if metric.lower() != 'r2':
            bars = ax1.bar(x + width * i, values, width, label=metric.upper())
            ax1.bar_label(bars, fmt='%.2e', label_type='edge', fontsize=8, rotation=90, padding=2)
        else:
            bars = ax2.bar(x, values, width, label=metric.upper(), color='red')
            ax2.bar_label(bars, fmt='%.4f', label_type='edge', fontsize=8, padding=2)

    ax1.set_ylabel('Score (log scale)', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title('LOSS, MSE, MAE (Log Scale)', fontsize=14)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.set_ylabel('R2 Score', fontsize=12)
    ax2.set_title('R2 Score', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim(0.8, 1)  # Adjust this range based on your R2 scores
    ax2.legend(loc='lower right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, 'overall_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Overall metrics comparison chart saved as 'overall_metrics_comparison.png'")


def learning_curves(data, output_dir):
    """
    Plot training and validation loss over epochs for each model.
    This helps visualize how well each model learned and if there's overfitting.
    """
    logger.info("Generating learning curves...")

    num_models = len(data)
    rows = (num_models + 1) // 2  # Calculate number of rows needed
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if num_models > 1 else [axes]

    for i, (model, metrics) in enumerate(tqdm(data.items(), desc="Processing models")):
        ax = axes[i]
        epochs = range(1, len(metrics['train']['loss']) + 1)
        ax.plot(epochs, metrics['train']['loss'], label='Training Loss')
        ax.plot(epochs, metrics['val']['loss'], label='Validation Loss')
        ax.set_title(f'{model} Learning Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()
    logger.info("Learning curves saved as 'learning_curves.png'")


def metric_evolution(data, output_dir):
    """
    Create line plots showing how metrics evolved over epochs for each model.
    This can reveal which models converged faster or more stably.
    """
    logger.info("Generating metric evolution plots...")

    # Define a color cycle for consistent colors across models
    color_cycle = plt.cm.tab10(np.linspace(0, 1, len(data)))

    for metric in tqdm(METRICS, desc="Processing metrics"):
        fig, ax = plt.subplots(figsize=(15, 8))

        for (model, metrics_data), color in zip(data.items(), color_cycle):
            epochs = range(1, len(metrics_data['train'][metric]) + 1)
            ax.plot(epochs, metrics_data['train'][metric], label=f'{model} (Train)',
                    color=color, linewidth=2, alpha=0.7)
            ax.plot(epochs, metrics_data['val'][metric], label=f'{model} (Validation)',
                    color=color, linestyle='--', linewidth=2, alpha=0.7)

        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} Evolution Over Epochs', fontsize=16)

        # Use log scale for y-axis if the metric is not R2
        if metric.lower() != 'r2':
            ax.set_yscale('log')
            ax.set_ylabel(f'{metric.upper()} (log scale)', fontsize=12)

        # Improve legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        # Add grid for better readability
        ax.grid(True, which="both", ls="--", alpha=0.5)

        # Improve tick labels
        ax.tick_params(axis='both', which='major', labelsize=10)

        plt.tight_layout()

        output_dir_full = os.path.join(output_dir, 'metric_evolution')
        if not os.path.exists(output_dir_full):
            os.makedirs(output_dir_full)

        plt.savefig(os.path.join(output_dir_full, f'{metric}_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"{metric.upper()} evolution plot saved as '{metric}_evolution.png'")

    # Create a separate legend plot
    fig, ax = plt.subplots(figsize=(12, len(data) * 0.5))
    ax.axis('off')
    for (model, _), color in zip(data.items(), color_cycle):
        ax.plot([], [], color=color, label=model, linewidth=2)
        ax.plot([], [], color=color, linestyle='--', label=f'{model} (Validation)', linewidth=2)
    ax.legend(loc='center', fontsize=12, ncol=2)
    plt.tight_layout()

    output_dir_full = os.path.join(output_dir, 'metric_evolution')
    if not os.path.exists(output_dir_full):
        os.makedirs(output_dir_full)

    plt.savefig(os.path.join(output_dir_full, 'legend.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Separate legend saved as 'legend.png'")


def training_times(data, output_dir):
    models = []
    training_times = []

    for model, model_data in data.items():
        models.append(model)
        training_times.append(model_data['train']['training_time'])

    plt.figure(figsize=(10, 6))
    plt.bar(models, training_times)
    plt.title('Training Time Comparison')
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, 'training_times_comparison.png'))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze overall model performance")
    parser.add_argument("--metrics-file", required=True, help="Path to the compiled metrics JSON file")
    parser.add_argument("--output-dir", default="plots", help="Directory to save output plots")
    args = parser.parse_args()

    logger.info(f"Loading data from {args.metrics_file}")
    try:
        with open(args.metrics_file, 'r') as f:
            data = json.load(f)
        logger.info("Data loaded successfully")
    except FileNotFoundError:
        logger.error(f"File not found: {args.metrics_file}")
        exit(1)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {args.metrics_file}")
        exit(1)

    # Run analyses
    overall_metrics_comparison(data, output_dir=args.output_dir)
    learning_curves(data, output_dir=args.output_dir)
    metric_evolution(data, output_dir=args.output_dir)
    training_times(data, output_dir=args.output_dir)
