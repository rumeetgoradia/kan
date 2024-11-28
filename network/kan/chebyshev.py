import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.kan.base import BaseKANLayer


class ChebyshevKANLayer(BaseKANLayer):
    def __init__(
            self,
            in_features,
            out_features,
            degree=5,
            scale_base=1.0,
            scale_cheb=1.0,
            base_activation=nn.SiLU,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.scale_base = scale_base
        self.scale_cheb = scale_cheb
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.cheb_weight = nn.Parameter(torch.Tensor(out_features, in_features, degree))
        self.base_activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        nn.init.kaiming_uniform_(self.cheb_weight, a=math.sqrt(5) * self.scale_cheb)

    def chebyshev_basis(self, x):
        # Assume x is scaled to [-1, 1]
        T = [torch.ones_like(x), x]
        for i in range(2, self.degree):
            T.append(2 * x * T[-1] - T[-2])
        return torch.stack(T, dim=-1)

    def forward(self, x):
        # Scale x to [-1, 1]
        x_scaled = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        base_output = F.linear(self.base_activation(x), self.base_weight)
        cheb_basis = self.chebyshev_basis(x_scaled)
        cheb_output = torch.sum(cheb_basis.unsqueeze(-2) * self.cheb_weight, dim=-1)
        return base_output + cheb_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.cheb_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )
