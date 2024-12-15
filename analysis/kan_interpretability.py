import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from constants import *
from network.kan.v3.layer import *

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


def load_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except:
        logger.error(f"Failed to load model: {model_path}")
        return None


def visualize_kan_activations(model, model_name, output_dir):
    """
    Visualize the learned activation functions of a KAN-based model.
    """
    logger.info(f"Visualizing activation functions for {model_name}...")

    output_dir_full = os.path.join(output_dir, 'kan_interpretability')
    if not os.path.exists(output_dir_full):
        os.makedirs(output_dir_full)

    # Find KAN layers in the model
    kan_layers = [layer for layer in model.layers if 'KANLayer' in layer.__class__.__name__]

    for i, layer in enumerate(kan_layers):
        plt.figure(figsize=(12, 8))

        # Generate input values and cast to float32
        x = np.linspace(-1, 1, 1000).reshape(-1, 1).astype(np.float32)

        # Compute activations based on the layer type
        if isinstance(layer, BSplineKANLayer):
            activations = layer.b_splines(tf.constant(x)).numpy()
        elif isinstance(layer, ChebyshevKANLayer):
            x_norm = tf.constant(x)
            cheby_polys = layer.chebyshev_polynomials(x_norm)

            # print(f"cheby_polys shape: {cheby_polys.shape}")
            # print(f"cheby_coeffs shape: {layer.cheby_coeffs.shape}")

            # Reshape cheby_polys to match the input dimension of cheby_coeffs
            cheby_polys_reshaped = tf.tile(cheby_polys, [1, layer.cheby_coeffs.shape[0], 1])

            # Perform the calculation
            activations = tf.einsum('bip,iop->bo', cheby_polys_reshaped, layer.cheby_coeffs)
            activations = activations.numpy()
        elif isinstance(layer, LegendreKANLayer):
            x_tf = tf.constant(x)
            legendre_basis = layer.legendre_basis(x_tf)

            # print(f"x shape: {x.shape}")
            # print(f"legendre_basis shape: {legendre_basis.shape}")
            # print(f"legendre_weight shape: {layer.legendre_weight.shape}")

            # Adjust legendre_basis to match the expected input dimension
            legendre_basis_expanded = tf.tile(legendre_basis, [1, layer.legendre_weight.shape[0], 1])

            # Compute activations
            activations = tf.einsum('iod,bid->bo', layer.legendre_weight, legendre_basis_expanded)
            activations = activations.numpy()
        elif isinstance(layer, FourierKANLayer):
            k = np.arange(1, layer.gridsize + 1)
            c = np.cos(k * np.pi * x)
            s = np.sin(k * np.pi * x)
            cs_combined = np.concatenate([c, s], axis=1)
            activations = np.dot(cs_combined, layer.fouriercoeffs)
        elif isinstance(layer, WaveletKANLayer):
            scale, translation, base_weight, wavelet_weights = layer.get_weights()
            x_expanded = np.repeat(x, layer.in_features, axis=1)
            x_scaled = (x_expanded - translation[0]) / scale[0]
            activations = layer.wavelet_transform(tf.constant(x_scaled)).numpy()
        else:
            logger.warning(f"Unknown KAN layer type in {model_name}, layer {i}")
            continue

        # Ensure activations are 2D
        if activations.ndim == 3:
            activations = activations.reshape(activations.shape[0], -1)
        elif activations.ndim == 1:
            activations = activations.reshape(-1, 1)

        # Plot activations
        num_functions = min(10, activations.shape[1])
        for j in range(num_functions):
            plt.plot(x, activations[:, j], label=f'Function {j + 1}')

        plt.title(f'{model_name} - {layer.__class__.__name__} Layer {i + 1} Activation Functions')
        plt.xlabel('Input')
        plt.ylabel('Activation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_full, f"{model_name}_{layer.__class__.__name__}_layer_{i + 1}_activations.png"))
        plt.close()

    logger.info(f"Activation function plots for {model_name} saved.")


def analyze_kan_interpretability(models_dir, output_dir):
    """
    Analyze and visualize the interpretability of KAN-based models.
    """
    for model_type in KAN_MODEL_TYPES:
        model_path = Path(models_dir) / f"{model_type}_model.keras"
        model = load_model(model_path)
        if model is not None:
            visualize_kan_activations(model, model_type, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze KAN model interpretability")
    parser.add_argument("--models-dir", required=True, help="Directory containing the model files")
    parser.add_argument("--output-dir", default='plots', help="Path to the directory to save output plots")
    args = parser.parse_args()
    # Analyze KAN interpretability
    analyze_kan_interpretability(args.models_dir, args.output_dir)
    logger.info("KAN interpretability analysis complete. Plots saved in: %s", args.output_dir)
