import numpy as np

from neuron.neuron_template import Neuron
from utils.activations import tanh_act


class NeuronKAN(Neuron):

    def __init__(self, n_in, n_weights_per_edge, weights_range, edge_function, edge_function_derivative, **kwargs):
        super().__init__(n_in, n_weights_per_edge=n_weights_per_edge, weights_range=weights_range)
        self.edge_function = edge_function
        self.edge_function_derivative = edge_function_derivative

    def get_xmid(self):
        # apply edge functions
        self.phi_x_mat = np.array([self.edge_function[b](self.xin) for b in self.edge_function]).T
        self.phi_x_mat[np.isnan(self.phi_x_mat)] = 0
        self.xmid = (self.weights * self.phi_x_mat).sum(axis=1)

    def get_xout(self):
        # note: node function <- tanh to avoid any update of spline grids
        self.xout = tanh_act(sum(self.xmid.flatten()), get_derivative=False)

    def get_dxout_dxmid(self):
        self.dxout_dxmid = tanh_act(sum(self.xmid.flatten()), get_derivative=True) * np.ones(self.n_in)

    def get_dxmid_dw(self):
        self.dxmid_dw = self.phi_x_mat

    def get_dxmid_dxin(self):
        phi_x_der_mat = np.array(
            [self.edge_function_derivative[b](self.xin) if self.edge_function[b](self.xin) is not None else 0
             for b in self.edge_function_derivative]).T  # shape (n_in, n_weights_per_edge)
        phi_x_der_mat[np.isnan(phi_x_der_mat)] = 0
        self.dxmid_dxin = (self.weights * phi_x_der_mat).sum(axis=1)

    def get_dxout_dbias(self):
        # no bias in KAN!
        self.dxout_dbias = 0
