from abc import ABC, abstractmethod

import torch.nn as nn


class BaseKANLayer(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        pass
