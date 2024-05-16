import torch.nn as nn
import torch
from torch import Tensor
from loguru import logger


class L2Similarities(nn.Module):
    r"""Default layer for computing similarity scores based on L2 distance in the convolutional space.

    Attributes:
        protopnet_compatibility: If True, uses the order of operations of ProtoPNet to compute the L2 distance.
    """

    def __init__(self, num_prototypes: int, num_features: int, protopnet_compatibility: bool = False) -> None:
        r"""Initializes a L2Similarities layer.

        Args:
            num_prototypes (int): Number of prototypes.
            num_features (int): Size of each prototype.
            protopnet_compatibility (bool, optional): If True, enables compatibility mode with ProtoPNet.
                Default: False.

        """
        super().__init__()
        self.register_buffer("_summation_kernel", torch.ones((num_prototypes, num_features, 1, 1)))
        self.protopnet_compatibility = protopnet_compatibility
        if self.protopnet_compatibility:
            logger.warning("Compatibility mode with ProtoPNet enabled when computing similarity scores ")

    def L2_square_distance(self, features: Tensor, prototypes: Tensor):
        r"""Returns the square of the L2 distance between each vector of features and each prototype.

        Args:
            features (tensor): Input tensor. Shape (N, D, H, W).
            prototypes (tensor): Tensor of prototypes. Shape (P, D, 1, 1).

        Returns:
            ||x-y||^2 for each feature vector and each prototype. Shape (N, P, H, W).
        """
        # ||X||² Shape (N, D, H, W)
        features_l2_squared = torch.conv2d(input=features**2, weight=self._summation_kernel)

        # ||P||² Shape (P, 1, 1)
        prototypes_l2_squared = torch.sum(prototypes**2, dim=(1, 2, 3)).view(-1, 1, 1)

        # features x prototypes (N, P, H, W)
        features_x_prototypes = torch.conv2d(input=features, weight=prototypes)
        if self.protopnet_compatibility:
            # Swap order of operations to reproduce ProtoPNet legacy code
            # Introduces slight changes on the output of the layer due to floating point operations
            intermediate = -2 * features_x_prototypes + prototypes_l2_squared
            return features_l2_squared + intermediate
        return features_l2_squared + prototypes_l2_squared - 2 * features_x_prototypes

    def forward(self, features: Tensor, prototypes: Tensor) -> Tensor:
        r"""Computes similarity based on L2 distance using ||x - y||² = ||x||² + ||y||² - 2 x.y.

        Args:
            features (tensor): Input tensor. Shape (N, D, H, W).
            prototypes (tensor): Tensor of prototypes. Shape (P, D, 1, 1).

        Returns:
            Tensor of similarities. Shape (N, P, H, W).
        """
        raise NotImplementedError
