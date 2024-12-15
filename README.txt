Comparison of polynomial basis functions for time series forecasting with Kolmogorov-Arnold networks.

Rumeet Goradia (rg3072)

Prerequisites
-------------

This project was built with Python 3.11.

Initialize a virtual environment and install the required packages with the following commands:

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Usage
-----

The enabled model types are:

1. mlp
2. lstm
3. kan-bspline
4. kan-chebyshev
5. kan-legendre
6. kan-wavelet
7. kan-fourier

Training
--------

Run train.py with a model type and, optionally, the desired directories to which to save the model and its training
history. If no such directories are specified, training will continue, but no results will be persisted. For example:

python train.py kan-bspline --model-save-dir results/models --history-save-dir results/metrics/train

Testing
-------

Run test.py with a model type and the directory containing the .keras model file. You can also optionally
specify the directory to save the test metrics. For example:

python test.py kan-bspline --models-dir results/models --metrics-save-dir results/metrics/test

Analysis
--------

All analysis is controlled by the files in the analysis package. By default, plots will be stored in analysis/plots.

Before analysis can take place, results from training and testing must be compiled via compile.py. The script can be
run as follows:

python analysis/compile.py --models_dir ../results/models --train_metrics_dir ../results/metrics/train --test_metrics_dir ../results/metrics/test

References
----------

The following open-source libraries are heavily utilized throughout the codebase, as documented in the
requirements.txt file:

- TensorFlow + Keras
- NumPy
- Pandas
- SciPy + scikit-learn
- Matplotlib
- Seaborn

Additionally, the following repositories were used as inspiration for the KAN layers:

- B-spline: efficient-kan (https://github.com/Blealtan/efficient-kan/)
- Chebyshev: ChebyKAN (https://github.com/SynodicMonth/ChebyKAN)
- Fourier: FourierKAN (https://github.com/GistNoesis/FourierKAN)
- Wavelet: EasyTSF (https://github.com/2448845600/EasyTSF/)

All files in the codebase were written by me, as denoted by the CUID at the top of each file. KAN layer references are
mentioned in those comments where applicable.