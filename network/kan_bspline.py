import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import BSpline

from data import provide as data_provider
from train import train_and_validate


class BSplineKAN(nn.Module):
    def __init__(self, layers_hidden, spline_order=3, base_activation=nn.SiLU):
        super(BSplineKAN, self).__init__()

        self.layers_hidden = layers_hidden
        self.spline_order = spline_order
        self.base_activation = base_activation()

        self.base_weights = nn.ParameterList()
        self.spline_weights = nn.ParameterList()
        self.layer_norms = nn.ModuleList()

        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            self.spline_weights.append(nn.Parameter(torch.randn(out_features, in_features * (spline_order + 1))))
            self.layer_norms.append(nn.LayerNorm(out_features))

        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.spline_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    def compute_bspline_basis(self, x, order, num_knots):
        # Generate uniform knots
        knots = np.linspace(0, 1, num_knots + 2 * order)
        knots = np.concatenate(([0] * order, knots, [1] * order))

        # Normalize x to [0, 1]
        x_normalized = (x - x.min(dim=1, keepdim=True)[0]) / (
                x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0])

        # Compute B-spline basis functions for each feature
        basis = []
        for i in range(num_knots + order):
            c = np.zeros(num_knots + order)
            c[i] = 1
            spline = BSpline(knots, c, order)
            basis.append(spline(x_normalized.cpu().numpy()))

        return torch.tensor(np.stack(basis, axis=-1), dtype=torch.float32, device=x.device)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.size()
        x = x.view(batch_size * seq_len, feature_dim)

        x = x.to(self.base_weights[0].device)

        for i, (base_weight, spline_weight, layer_norm) in enumerate(
                zip(self.base_weights, self.spline_weights, self.layer_norms)):
            base_output = F.linear(self.base_activation(x), base_weight)

            num_knots = feature_dim  # Number of knots can be set to the number of input features
            bspline_basis = self.compute_bspline_basis(x.view(batch_size, seq_len, feature_dim), self.spline_order,
                                                       num_knots)
            bspline_basis = bspline_basis.view(batch_size * seq_len, -1)

            spline_output = F.linear(bspline_basis, spline_weight)
            x = self.base_activation(layer_norm(base_output + spline_output))

        x = x.view(batch_size, seq_len, -1)
        return x




if __name__ == '__main__':
    X_train = np.load('../data/prepared/sequenced/X_train.npy')
    y_train = np.load('../data/prepared/sequenced/y_train.npy')
    print(X_train[0].size(0))
    print(y_train.size(0))
    train_and_validate(X_train, y_train, BSplineKAN, {'layers_hidden': [43, 64, 1], 'spline_order': 3})
