import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierKAN(nn.Module):  # Kolmogorov Arnold Network with Fourier series
    def __init__(self, layers_hidden, fourier_order=3, base_activation=nn.SiLU):
        super(FourierKAN, self).__init__()

        self.layers_hidden = layers_hidden
        self.fourier_order = fourier_order
        self.base_activation = base_activation()

        self.base_weights = nn.ParameterList()
        self.fourier_weights = nn.ParameterList()
        self.layer_norms = nn.ModuleList()

        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            self.fourier_weights.append(nn.Parameter(torch.randn(out_features, in_features * (2 * fourier_order + 1))))
            self.layer_norms.append(nn.LayerNorm(out_features))

        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.fourier_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    def compute_fourier_basis(self, x, order):
        # Normalize x to [0, 2π] for Fourier basis computation
        x_normalized = 2 * torch.pi * (x - x.min(dim=1, keepdim=True)[0]) / (
                    x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0])

        # Compute Fourier basis functions for each feature
        basis = [x_normalized]
        for n in range(1, order + 1):
            basis.append(torch.sin(n * x_normalized))
            basis.append(torch.cos(n * x_normalized))

        return torch.cat(basis, dim=-1)

    def forward(self, x):
        # Reshape x to combine batch and sequence dimensions for processing
        batch_size, seq_len, feature_dim = x.size()
        x = x.view(batch_size * seq_len, feature_dim)

        x = x.to(self.base_weights[0].device)

        for i, (base_weight, fourier_weight, layer_norm) in enumerate(
                zip(self.base_weights, self.fourier_weights, self.layer_norms)):
            base_output = F.linear(self.base_activation(x), base_weight)

            fourier_basis = self.compute_fourier_basis(x.view(batch_size, seq_len, feature_dim), self.fourier_order)
            fourier_basis = fourier_basis.view(batch_size * seq_len, -1)

            fourier_output = F.linear(fourier_basis, fourier_weight)
            x = self.base_activation(layer_norm(base_output + fourier_output))

        # Reshape back to original batch and sequence dimensions
        x = x.view(batch_size, seq_len, -1)
        return x
