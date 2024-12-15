# Comparison of polynomial basis functions for time series forecasting with Kolmogorov-Arnold networks.

__Rumeet Goradia (rg3072)__

## Prerequisites

This project was built with __Python 3.11__.

Initialize a virtual environment and install the required packages with the following commands:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

This project will be compressed with the necessary data. In case you need to re-process the data, you can access the
indices data from [Kaggle](https://www.kaggle.com/datasets/asimislam/30-yrs-stock-market-data).

Save the data in `data/raw` as `indices.csv`. The existing `sp500.txt` file will contain all tickers in the S&P 500
index.

Update the arguments in the `__main__` function of `preprocess.py` and run the script to generate the necessary data.

To reinitialize the data, the files under `data/processed` can be deleted.

> Note: During training, a `ticker_splits.json` file will be created in the `data/processed` directory if it does not
> already exist. This file maintains the training, validation, and testing splits across all data, so that every model
> is trained, validated, and tested with the same data.

## Usage

The enabled model types are:

1. `mlp`
2. `lstm`
3. `kan-bspline`
4. `kan-chebyshev`
5. `kan-legendre`
6. `kan-wavelet`
7. `kan-fourier`

### Training

Run `train.py` with a model type and, optionally, the desired directories to which to save the model and its training
history. If no such directories are specified, training will continue, but no results will be persisted. For example:

```bash
python train.py kan-bspline --model-save-dir results/models --history-save-dir results/metrics/train
````

### Testing

Run `test.py` with a model type and the directory containing the `.keras` model file. You can also optionally
specify the directory to save the test metrics. For example:

```bash
python test.py kan-bspline --models-dir results/models --metrics-save-dir results/metrics/test
```

### Analysis

All analysis is controlled by the files in the `analysis` package. By default, plots will be stored in `analysis/plots`.

Before analysis can take place, results from training and testing must be compiled via `compile.py`. The script can be
run as follows:

```bash
python analysis/compile.py --models_dir ../results/models --train_metrics_dir ../results/metrics/train --test_metrics_dir ../results/metrics/test
```