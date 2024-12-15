"""
rg3072
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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


def perform_statistical_tests(data, output_dir):
    """
    Conduct statistical tests (t-tests and ANOVA) to determine if the differences
    in performance between models are significant.
    """
    logger.info("Performing statistical tests...")

    output_dir_full = os.path.join(output_dir, 'statistical_tests')
    if not os.path.exists(output_dir_full):
        os.makedirs(output_dir_full)

    # Prepare data for analysis
    model_errors = {}
    for model, predictions in data['predictions'].items():
        errors = np.array(predictions) - np.array(data['actual'])
        model_errors[model] = errors.flatten()

    df = pd.DataFrame(model_errors)

    # Perform pairwise t-tests
    models = list(model_errors.keys())
    t_test_results = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1, model2 = models[i], models[j]
            t_stat, p_value = stats.ttest_ind(model_errors[model1], model_errors[model2])
            t_test_results.append({
                'Model 1': model1,
                'Model 2': model2,
                't-statistic': t_stat,
                'p-value': p_value
            })

    t_test_df = pd.DataFrame(t_test_results)
    t_test_df.to_csv(os.path.join(output_dir_full, "pairwise_t_tests.csv"), index=False)
    logger.info("Pairwise t-test results saved to pairwise_t_tests.csv")

    # Perform one-way ANOVA
    f_statistic, p_value = stats.f_oneway(*[model_errors[model] for model in models])

    with open(os.path.join(output_dir_full, "anova_results.txt"), "w") as f:
        f.write(f"One-way ANOVA results:\n")
        f.write(f"F-statistic: {f_statistic}\n")
        f.write(f"p-value: {p_value}\n")
    logger.info("ANOVA results saved to anova_results.txt")

    # Perform Tukey's HSD test
    model_names = np.repeat(models, [len(model_errors[model]) for model in models])
    all_errors = np.concatenate([model_errors[model] for model in models])

    tukey_results = pairwise_tukeyhsd(all_errors, model_names)

    with open(os.path.join(output_dir_full, "tukey_hsd_results.txt"), "w") as f:
        f.write(str(tukey_results))
    logger.info("Tukey's HSD test results saved to tukey_hsd_results.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform statistical tests on model performance")
    parser.add_argument("--predictions-file", required=True, help="Path to the predictions JSON file")
    parser.add_argument("--output-dir", default='plots', help="Path to the directory to save output files")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load predictions data
    data = load_json(args.predictions_file)

    if data is None:
        logger.error("Failed to load predictions data. Exiting.")
        exit(1)

    # Perform statistical tests
    perform_statistical_tests(data, args.output_dir)

    logger.info("Statistical tests complete. Results saved in: %s", args.output_dir)
