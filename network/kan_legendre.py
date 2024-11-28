from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


class LegendreKAN(nn.Module):
    def __init__(self, layers_hidden, polynomial_order=3, base_activation=nn.SiLU):
        super(LegendreKAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.polynomial_order = polynomial_order
        self.base_activation = base_activation()
        self.base_weights = nn.ParameterList()
        self.poly_weights = nn.ParameterList()
        self.layer_norms = nn.ModuleList()
        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            self.poly_weights.append(nn.Parameter(torch.randn(out_features, in_features * (polynomial_order + 1))))
            self.layer_norms.append(nn.LayerNorm(out_features))
        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.poly_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    @lru_cache(maxsize=128)
    def compute_legendre_polynomials(self, x, order):
        P0 = x.new_ones(x.shape)
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x
        legendre_polys = [P0, P1]
        for n in range(1, order):
            Pn = ((2.0 * n + 1.0) * x * legendre_polys[-1] - n * legendre_polys[-2]) / (n + 1.0)
            legendre_polys.append(Pn)
        return torch.stack(legendre_polys, dim=-1)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.size()
        x = x.view(batch_size * seq_len, feature_dim)
        x = x.to(self.base_weights[0].device)
        for i, (base_weight, poly_weight, layer_norm) in enumerate(
                zip(self.base_weights, self.poly_weights, self.layer_norms)):
            base_output = F.linear(self.base_activation(x), base_weight)
            x_normalized = 2 * (x - x.min(dim=1, keepdim=True)[0]) / (
                        x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0]) - 1
            legendre_basis = self.compute_legendre_polynomials(x_normalized, self.polynomial_order)
            legendre_basis = legendre_basis.view(batch_size * seq_len, -1)
            poly_output = F.linear(legendre_basis, poly_weight)
            x = self.base_activation(layer_norm(base_output + poly_output))
        x = x.view(batch_size, seq_len, -1)
        return x
