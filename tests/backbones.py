"""Small backbone modules shared by tests."""

import torch
from torch import nn


class TinyBackbone(nn.Module):
    """Small backbone used to build architecture fixtures without pretrained weights."""

    def __init__(self) -> None:
        """Initializes a randomly weighted convolutional feature layer."""
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1))
        nn.init.normal_(self.features[0].weight)
        nn.init.normal_(self.features[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes convolutional features.

        Args:
            x (Tensor): Input image batch.

        Returns:
            Convolutional features.
        """
        return self.features(x)
