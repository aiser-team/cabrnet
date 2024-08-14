from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor


class SimilarityLayer(nn.Module, ABC):
    r"""Abstract layer for computing similarity scores based on distances in the convolutional space."""

    def __init__(self, **kwargs):
        r"""Uses nn.Module init function."""
        nn.Module.__init__(self)

    @abstractmethod
    def distances(self, features: Tensor, prototypes: Tensor, **kwargs) -> Tensor:
        r"""Computes pairwise distances between a tensor of features and a tensor of prototypes.

        Args:
            features (tensor): Input tensor. Shape (N, D, H, W).
            prototypes (tensor): Tensor of prototypes. Shape (P, D, 1, 1).

        Returns:
             Distance between each feature vector and each prototype. Shape (N, P, H, W).
        """
        raise NotImplementedError

    @abstractmethod
    def similarities(self, features: Tensor, prototypes: Tensor, **kwargs) -> Tensor:
        r"""Computes pairwise similarity scores between a tensor of features and a tensor of prototypes.

        Args:
            features (tensor): Input tensor. Shape (N, D, H, W).
            prototypes (tensor): Tensor of prototypes. Shape (P, D, 1, 1).

        Returns:
            Tensor of similarity scores. Shape (N, P, H, W).
        """
        raise NotImplementedError

    def forward(self, features: Tensor, prototypes: Tensor, **kwargs) -> Tensor:
        r"""Return pairwise similarity scores between a tensor of features and a tensor of prototypes.

        Args:
            features (tensor): Input tensor. Shape (N, D, H, W).
            prototypes (tensor): Tensor of prototypes. Shape (P, D, 1, 1).

        Returns:
            Tensor of similarity scores. Shape (N, P, H, W).
            Tensor of distances. Shape (N, P, H, W).
        """
        return self.similarities(features, prototypes, **kwargs)


class SquaredEuclideanDistance(SimilarityLayer):
    r"""Layer for computing Euclidean (L2) distances in the convolutional space."""

    def distances(self, features: Tensor, prototypes: Tensor, **kwargs) -> Tensor:
        r"""Computes pairwise squared Euclidean (L2) distances between a tensor of features and a tensor of prototypes.

        Args:
            features (tensor): Input tensor. Shape (N, D, H, W).
            prototypes (tensor): Tensor of prototypes. Shape (P, D, 1, 1).

        Returns:
            Tensor of distances. Shape (N, P, H, W).
        """
        N, D, H, W = features.shape
        features = features.view((N, D, -1))  # Shape (N, D, HxW)
        features = torch.transpose(features, 1, 2)  # Shape (N, HxW, D)
        prototypes = prototypes.squeeze(dim=(2, 3))  # Shape (P, D)
        distances = torch.cdist(features, prototypes) ** 2  # Shape (N, HxW, P)
        distances = torch.transpose(distances, 1, 2)  # Shape (N, P, HxW)
        distances = distances.view(distances.shape[:2] + (H, W))  # Shape (N, P, H, W)
        return distances


class ProtoPNetDistance(SimilarityLayer):
    r"""Layer for computing Euclidean (L2) distances in the convolutional space (ProtoPNet original implementation).

    Attributes:
        _summation_kernel: Accumulation tensor used to compute the Euclidean distance.
    """

    def __init__(self, num_prototypes: int, num_features: int, **kwargs) -> None:
        r"""Initializes a ProtoPNetDistance layer.

        Args:
            num_prototypes (int): Number of prototypes.
            num_features (int): Size of each prototype.
        """
        super().__init__(**kwargs)
        self.register_buffer("_summation_kernel", torch.ones((num_prototypes, num_features, 1, 1)))
        logger.warning("Enabling compatibility mode with ProtoPNet legacy code when computing similarity scores.")

    def distances(self, features: Tensor, prototypes: Tensor, **kwargs) -> Tensor:
        r"""Computes pairwise squared Euclidean (L2) distances between a tensor of features and a tensor of prototypes,
        as originally implemented in ProtoPNet, i.e., using the formula ||x - y||² = ||x||² - 2 x.y. + ||y||².

        Args:
            features (tensor): Input tensor. Shape (N, D, H, W).
            prototypes (tensor): Tensor of prototypes. Shape (P, D, 1, 1).

        Returns:
            Tensor of distances. Shape (N, P, H, W).
        """
        features_l2_squared = torch.conv2d(input=features**2, weight=self._summation_kernel)

        # ||P||² Shape (P, 1, 1)
        prototypes_l2_squared = torch.sum(prototypes**2, dim=(1, 2, 3)).view(-1, 1, 1)

        # features x prototypes (N, P, H, W)
        features_x_prototypes = torch.conv2d(input=features, weight=prototypes)
        # Swap order of operations to reproduce ProtoPNet legacy code
        # Introduces slight changes on the output of the layer due to floating point operations
        intermediate = -2 * features_x_prototypes + prototypes_l2_squared
        return features_l2_squared + intermediate


class ProtoTreeDistance(SimilarityLayer):
    r"""Layer for computing Euclidean (L2) distances in the convolutional space (ProtoTree original implementation).

    Attributes:
        _summation_kernel: Accumulation tensor used to compute the Euclidean distance.
    """

    def __init__(self, num_prototypes: int, num_features: int, **kwargs) -> None:
        r"""Initializes a ProtoTreeDistance layer.

        Args:
            num_prototypes (int): Number of prototypes.
            num_features (int): Size of each prototype.
        """
        super().__init__(**kwargs)
        self.register_buffer("_summation_kernel", torch.ones((num_prototypes, num_features, 1, 1)))
        logger.warning("Enabling compatibility mode with ProtoTree legacy code when computing similarity scores.")

    def distances(self, features: Tensor, prototypes: Tensor, **kwargs) -> Tensor:
        r"""Computes pairwise squared Euclidean (L2) distances between a tensor of features and a tensor of prototypes,
        as originally implemented in ProtoTree, i.e., using the formula ||x - y||² = ||x||² + ||y||² - 2 x.y.

        Args:
            features (tensor): Input tensor. Shape (N, D, H, W).
            prototypes (tensor): Tensor of prototypes. Shape (P, D, 1, 1).

        Returns:
            Tensor of distances. Shape (N, P, H, W).
        """
        # ||X||² Shape (N, D, H, W)
        features_l2_squared = torch.conv2d(input=features**2, weight=self._summation_kernel)

        # ||P||² Shape (P, 1, 1)
        prototypes_l2_squared = torch.sum(prototypes**2, dim=(1, 2, 3)).view(-1, 1, 1)

        # features x prototypes (N, P, H, W)
        features_x_prototypes = torch.conv2d(input=features, weight=prototypes)

        return features_l2_squared + prototypes_l2_squared - 2 * features_x_prototypes


class LogDistance(SimilarityLayer):
    r"""Abstract layer for computing similarity scores based on the log of distances in the convolutional space.

    Attributes:
        stability_factor: Stability factor.
    """

    def __init__(self, stability_factor: float = 1e-4, **kwargs) -> None:
        r"""Initializes a ProtoPNetDistance layer.

        Args:
            stability_factor (float, optional): Stability factor. Default: 1e-4.
        """
        super().__init__(**kwargs)
        self.stability_factor = stability_factor

    def similarities(self, features: Tensor, prototypes: Tensor, **kwargs) -> Tensor:
        r"""Computes pairwise similarity scores between a tensor of features and a tensor of prototypes as
        log((1+distances)/(1+epsilon)) where epsilon is the stability factor.

        Args:
            features (tensor): Input tensor. Shape (N, D, H, W).
            prototypes (tensor): Tensor of prototypes. Shape (P, D, 1, 1).

        Returns:
            Tensor of similarity scores. Shape (N, P, H, W).
        """
        # Ensures that distances are greater than 0
        distances = torch.relu(self.distances(features, prototypes))
        return torch.log((distances + 1) / (distances + self.stability_factor))


class ExpDistance(SimilarityLayer):
    r"""Abstract layer for computing similarity scores based on the exponential of distances in the convolutional space.

    Attributes:
        log_probabilities: If True, returns similarity scores as log of probabilities.
        stability_factor: Stability factor.
    """

    def __init__(self, log_probabilities: bool = False, stability_factor: float = 1e-14, **kwargs) -> None:
        r"""Initializes a ProtoPNetDistance layer.

        Args:
            log_probabilities (bool, optional): If True, returns similarity scores as log of probabilities.
                Default: False.
            stability_factor (float, optional): Stability factor. Default: 1e-14.
        """
        super().__init__(**kwargs)
        self.log_probabilities = log_probabilities
        self.stability_factor = stability_factor

    def similarities(self, features: Tensor, prototypes: Tensor, **kwargs) -> Tensor:
        r"""Computes pairwise similarity scores between a tensor of features and a tensor of prototypes
        as exp(-sqrt(distances+epsilon))$$ where epsilon is the stability factor.
        If log_probabilities is enabled, simply returns -sqrt(distances+epsilon).

        Args:
            features (tensor): Input tensor. Shape (N, D, H, W).
            prototypes (tensor): Tensor of prototypes. Shape (P, D, 1, 1).

        Returns:
            Tensor of similarity scores. Shape (N, P, H, W).
        """
        distances = torch.sqrt(
            torch.abs(self.distances(features=features, prototypes=prototypes)) + self.stability_factor
        )
        if self.log_probabilities:
            return -distances
        return torch.exp(-distances)


class LegacyProtoPNetSimilarity(ProtoPNetDistance, LogDistance):
    r"""Layer for computing similarity scores based on Euclidean (L2) distances in the convolutional space
        (ProtoPNet original implementation).

    Attributes:
        stability_factor: Stability factor.
        _summation_kernel: Accumulation tensor used to compute the Euclidean distance.
    """

    def __init__(
        self, num_prototypes: int = 0, num_features: int = 0, stability_factor: float = 1e-4, **kwargs
    ) -> None:
        r"""Initializes a LegacyProtoPNetSimilarity layer.

        Args:
            num_prototypes (int, optional): Number of prototypes (this should be set automatically). Default: 0.
            num_features (int, optional): Size of each prototype (this should be set automatically). Default: 0.
            stability_factor (float, optional): Stability factor. Default: 1e-4.
        """
        super().__init__(
            num_prototypes=num_prototypes, num_features=num_features, stability_factor=stability_factor, **kwargs
        )


class ProtoPNetSimilarity(SquaredEuclideanDistance, LogDistance):
    r"""Layer for computing similarity scores based on Euclidean (L2) distances in the convolutional space
        (ProtoPNet implementation updated with cdist function).

    Attributes:
        stability_factor: Stability factor.
    """

    def __init__(self, stability_factor: float = 1e-4, **kwargs) -> None:
        r"""Initializes a ProtoPNetSimilarity layer.

        Args:
            stability_factor (float, optional): Stability factor. Default: 1e-4.
        """
        super().__init__(stability_factor=stability_factor, **kwargs)


class LegacyProtoTreeSimilarity(ProtoTreeDistance, ExpDistance):
    r"""Layer for computing similarity scores based on Euclidean (L2) distances in the convolutional space
        (ProtoPTree original implementation).

    Attributes:
        log_probabilities: If True, returns similarity scores as log of probabilities.
        stability_factor: Stability factor.
        _summation_kernel: Accumulation tensor used to compute the Euclidean distance.
    """

    def __init__(
        self,
        num_prototypes: int = 0,
        num_features: int = 0,
        log_probabilities: bool = False,
        stability_factor: float = 1e-14,
        **kwargs,
    ) -> None:
        r"""Initializes a LegacyProtoTreeSimilarity layer.

        Args:
            num_prototypes (int, optional): Number of prototypes (this should be set automatically). Default: 0.
            num_features (int, optional): Size of each prototype (this should be set automatically). Default: 0.
            log_probabilities (bool, optional): If True, returns similarity scores as log of probabilities.
                Default: False.
            stability_factor (float, optional): Stability factor. Default: 1e-14.
        """
        super().__init__(
            num_prototypes=num_prototypes,
            num_features=num_features,
            log_probabilities=log_probabilities,
            stability_factor=stability_factor,
            **kwargs,
        )


class ProtoTreeSimilarity(SquaredEuclideanDistance, ExpDistance):
    r"""Layer for computing similarity scores based on Euclidean (L2) distances in the convolutional space
        (ProtoPTree implementation updated with cdist function).

    Attributes:
        log_probabilities: If True, returns similarity scores as log of probabilities.
        stability_factor: Stability factor.
    """

    def __init__(self, log_probabilities: bool = False, stability_factor: float = 1e-14, **kwargs) -> None:
        r"""Initializes a ProtoTreeSimilarity layer.

        Args:

            log_probabilities (bool, optional): If True, returns similarity scores as log of probabilities.
                Default: False.
            stability_factor (float, optional): Stability factor. Default: 1e-14.
        """
        super().__init__(log_probabilities=log_probabilities, stability_factor=stability_factor, **kwargs)
