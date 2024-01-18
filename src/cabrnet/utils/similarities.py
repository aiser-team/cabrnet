import torch.nn as nn
import torch
from torch import Tensor


class L2Similarities(nn.Module):
    def __init__(self, num_prototypes: int, num_features: int) -> None:
        """
        Create module for computing similarities based on L2 distance
        Args:
            num_prototypes: Number of prototypes
            num_features: Size of each prototype

        """
        super().__init__()
        self.register_buffer("_summation_kernel", torch.ones((num_prototypes, num_features, 1, 1)))

    def L2_square_distance(self, features: Tensor, prototypes: Tensor):
        """Returns the square of the L2 distance between each vector of features
        and each prototype.

        Args:
            features: Input tensor. Shape (N, D, H, W)
            prototypes: Tensor of prototypes. Shape (P, D, 1, 1)

        Returns:
            ||x-y||^2 for each feature vector and each prototype. Shape (N, P, H, W)
        """
        # ||X||² Shape (N, D, H, W)
        features_l2_squared = torch.conv2d(input=features**2, weight=self._summation_kernel)

        # ||P||² Shape (P, 1, 1)
        prototypes_l2_squared = torch.sum(prototypes**2, dim=(1, 2, 3)).view(-1, 1, 1)

        # features x prototypes (N, P, H, W)
        features_x_prototypes = torch.conv2d(input=features, weight=prototypes)
        return features_l2_squared + prototypes_l2_squared - 2 * features_x_prototypes

    def forward(self, features: Tensor, prototypes: Tensor) -> Tensor:
        """
        Compute similarity based on L2 distance using ||x - y||² = ||x||² + ||y||² - 2 x.y
        Args:
            features: Input tensor. Shape (N, D, H, W)
            prototypes: Tensor of prototypes. Shape (P, D, 1, 1)

        Returns:
            Tensor of similarities. Shape (N, P, H, W)
        """
        raise NotImplementedError
