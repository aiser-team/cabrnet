from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cabrnet.archs.generic.decision import CaBRNetClassifier


class NonNegLinear(nn.Linear):
    r"""Linear layer with non-negative weights."""

    def forward(self, input: Tensor) -> Tensor:
        r"""Applies linear layer.

        Args:
            input (tensor): Input tensor.

        Returns:
            ReLU(A) * input (+ bias).
        """
        return F.linear(input, torch.relu(self.weight), self.bias)


class PIPNetClassifier(CaBRNetClassifier):
    r"""Classification pipeline for PIPNet architecture.

    Attributes:
        num_classes: Number of output classes.
        num_features: Size of the features extracted by the convolutional extractor. In PIPNet, this is also the
            number of prototypes.
        normalization_multiplier: Learnable normalization value.
        last_layer: Linear layer in charge of weighting similarity scores and computing the final logit vector.
    """

    def __init__(
        self,
        num_classes: int,
        num_features: int,
        use_bias: bool = False,
        **kwargs,
    ) -> None:
        r"""Initializes a PIPNet classifier.

        Args:
            num_classes (int): Number of classes.
            num_features (int): Number of features. This is also the number of prototypes.
            use_bias (bool, optional): Use bias in the final linear layer. Default: False.
        """
        super().__init__(num_classes=num_classes, num_features=num_features)

        # PIPNet does not train a dedicated set of prototypes
        self.prototypes = nn.Parameter(torch.empty(0))  # type: ignore

        # Final layers
        self.last_layer = NonNegLinear(in_features=self.num_features, out_features=self.num_classes, bias=use_bias)

        torch.nn.init.normal_(self.last_layer.weight, mean=1.0, std=0.1)
        if use_bias:
            torch.nn.init.constant_(self.last_layer.bias, val=0.0)

        # Learnable normalization multiplier
        self.normalization_multiplier = nn.Parameter(torch.ones((1,), requires_grad=False) * 2.0)

    def build_similarity(self, config: dict[str, Any]) -> None:
        r"""Builds the similarity layer based on information regarding the decision process.
        WARNING: Irrelevant in PIPNet.

        Args:
            config (dict): Configuration of the similarity layer.
        """
        raise NotImplementedError

    @property
    def num_prototypes(self) -> int:
        r"""Returns the maximum number of prototypes."""
        return self.num_features

    def prototype_is_active(self, proto_idx: int) -> bool:
        r"""Is the prototype *proto_idx* active or disabled?

        Args:
            proto_idx (int): Prototype index.
        """
        # A prototype is active if it is associated with at least one class
        class_mapping = self.prototype_class_mapping
        return max(class_mapping[proto_idx])

    def similarities(self, features: Tensor, **kwargs) -> Tensor:
        r"""Returns similarity scores.
        WARNING: Irrelevant in PIPNet.

        Args:
            features (tensor): Feature tensor.

        Returns:
            Tensor of similarity scores.
        """
        return F.softmax(features, dim=1)

    def distances(self, features: Tensor, **kwargs) -> Tensor:
        r"""Returns pairwise distances between each feature vector and each prototype.
        WARNING: Irrelevant in PIPNet.

        Args:
            features (tensor): Features tensor.

        Returns:
            Tensor of distances.
        """
        raise NotImplementedError

    @property
    def prototype_class_mapping(self) -> np.ndarray:
        r"""Mapping between prototypes and classes.

        Returns:
            Binary array of shape (P, C).
        """
        # After clamping, relevant associations have non-zero weights
        return (self.last_layer.weight > 1e-3).swapaxes(0, 1).detach().cpu().numpy()

    def forward(self, features: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        r"""Performs classification using a linear layer.

        Args:
            features (tensor): Convolutional features from extractor. Shape (N, D, H, W).

        Returns:
            Tensor of prototypical features. Shape (N, P, H, W).
            Tensor of prototype presence. Shape (N, P).
            Vector of logits. Shape (N, C).
        """
        features = F.softmax(features, dim=1)
        prototype_presence = F.adaptive_max_pool2d(input=features, output_size=(1, 1)).flatten(start_dim=1)
        prediction = self.last_layer(prototype_presence)
        return features, prototype_presence, prediction

    def clamp_parameters(self):
        r"""Clamps parameters in-between epochs."""
        with torch.no_grad():
            self.last_layer.weight.copy_(torch.clamp(self.last_layer.weight.data - 1e-3, min=0.0))
            self.normalization_multiplier.copy_(torch.clamp(self.normalization_multiplier.data, min=1.0))
            if self.last_layer.bias is not None:
                self.last_layer.bias.copy_(torch.clamp(self.last_layer.bias.data, min=0.0))
