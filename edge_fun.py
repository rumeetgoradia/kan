from abc import ABC, abstractmethod
from typing import Dict, Callable, Tuple

import numpy as np
from scipy.interpolate import BSpline
from scipy.signal import cwt, ricker


class EdgeFunctionProvider(ABC):
    @abstractmethod
    def get_functions(self) -> Tuple[Dict[int, Callable], Dict[int, Callable]]:
        """Return edge functions and their derivatives."""
        pass


class BSplineProvider(EdgeFunctionProvider):
    def __init__(self, x_bounds, n_fun, degree=3):
        self.x_bounds = x_bounds
        self.n_fun = n_fun
        self.degree = degree

    def get_functions(self):
        grid_len = self.n_fun - self.degree + 1
        step = (self.x_bounds[1] - self.x_bounds[0]) / (grid_len - 1)
        edge_fun, edge_fun_der = {}, {}

        # SiLU bias function
        edge_fun[0] = lambda x: x / (1 + np.exp(-x))
        edge_fun_der[0] = lambda x: (1 + np.exp(-x) + x * np.exp(-x)) / np.power((1 + np.exp(-x)), 2)

        # B-splines
        t = np.linspace(self.x_bounds[0] - self.degree * step, self.x_bounds[1] + self.degree * step,
                        grid_len + 2 * self.degree)
        t[self.degree], t[-self.degree - 1] = self.x_bounds[0], self.x_bounds[1]
        for ind_spline in range(self.n_fun - 1):
            edge_fun[ind_spline + 1] = BSpline.basis_element(t[ind_spline:ind_spline + self.degree + 2],
                                                             extrapolate=False)
            edge_fun_der[ind_spline + 1] = edge_fun[ind_spline + 1].derivative()
        return edge_fun, edge_fun_der


class ChebyshevProvider(EdgeFunctionProvider):
    def __init__(self, x_bounds, n_fun):
        self.x_bounds = x_bounds
        self.n_fun = n_fun

    def get_functions(self):
        edge_fun, edge_fun_der = {}, {}
        for deg in range(self.n_fun):
            edge_fun[deg] = np.polynomial.chebyshev.Chebyshev.basis(deg=deg, domain=self.x_bounds)
            edge_fun_der[deg] = edge_fun[deg].deriv(1)
        return edge_fun, edge_fun_der


class FourierProvider(EdgeFunctionProvider):
    def __init__(self, x_bounds, n_fun):
        self.x_bounds = x_bounds
        self.n_fun = n_fun

    def get_functions(self):
        edge_fun, edge_fun_der = {}, {}
        L = self.x_bounds[1] - self.x_bounds[0]
        for k in range(self.n_fun):
            edge_fun[k] = lambda x, k=k: np.cos(2 * np.pi * k * x / L) + np.sin(2 * np.pi * k * x / L)
            edge_fun_der[k] = lambda x, k=k: -2 * np.pi * k / L * (
                    np.sin(2 * np.pi * k * x / L) - np.cos(2 * np.pi * k * x / L))
        return edge_fun, edge_fun_der


class WaveletProvider(EdgeFunctionProvider):
    def __init__(self, x_bounds, n_fun, widths=None):
        self.x_bounds = x_bounds
        self.n_fun = n_fun
        self.widths = widths if widths is not None else np.arange(1, n_fun + 1)

    def get_functions(self):
        edge_fun, edge_fun_der = {}, {}
        for i, width in enumerate(self.widths):
            edge_fun[i] = lambda x, width=width: cwt(x, ricker, [width])[0]
            edge_fun_der[i] = lambda x, width=width: np.gradient(cwt(x, ricker, [width])[0])
        return edge_fun, edge_fun_der
