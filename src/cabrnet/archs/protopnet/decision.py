from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from cabrnet.archs.generic.decision import CaBRNetClassifier
from cabrnet.core.utils.prototypes import init_prototypes


class ProtoPNetClassifier(CaBRNetClassifier):
    r"""Classification pipeline for ProtoPNet architecture.

    Attributes:
        num_classes: Number of output classes.
        num_features: Size of the features extracted by the convolutional extractor.
        num_proto_per_class: Initial number of prototypes per class.
        prototypes: Tensor of prototypes.
        prototypes_init_mode: Initialization mode for the tensor of prototypes.
        similarity_layer: Layer used to compute similarity scores between the prototypes and the convolutional features.
        last_layer: Linear layer in charge of weighting similarity scores and computing the final logit vector.
    """

    def __init__(
        self,
        similarity_config: dict[str, Any],
        num_classes: int,
        num_features: int,
        num_proto_per_class: int,
        proto_init_mode: str = "SHIFTED_NORMAL",
        incorrect_class_penalty: float = -0.5,
        compatibility_mode: bool = False,
    ) -> None:
        r"""Initializes a ProtoPNet classifier.

        Args:
            similarity_config (dict): Configuration of the layer used to compute similarity scores between the
                prototypes and the convolutional features.
            num_classes (int): Number of classes.
            num_features (int): Number of features (size of each prototype).
            num_proto_per_class (int): Number of prototypes per class.
            proto_init_mode (str, optional): Init mode for prototypes. Default: Shifted normal distribution.
            incorrect_class_penalty (float, optional): Initial penalty for incorrect classes in the linear layer.
                Default: 0.5.
            compatibility_mode (bool, optional): If True, enables compatibility mode with legacy ProtoPNet.
                Default: False.
        """
        super().__init__(num_classes=num_classes, num_features=num_features, proto_init_mode=proto_init_mode)

        assert num_proto_per_class > 0, f"Invalid number of prototypes per class: {num_proto_per_class}"
        self.num_proto_per_class = num_proto_per_class
        self._compatibility_mode = compatibility_mode

        # Init prototypes
        self.prototypes = nn.Parameter(  # type: ignore
            init_prototypes(
                num_prototypes=num_proto_per_class * num_classes,
                num_features=self.num_features,
                init_mode=proto_init_mode,
            )
        )

        # Init similarity layer
        self.build_similarity(similarity_config)

        # Initialize last layer
        proto_class_map = torch.zeros(self.num_prototypes, self.num_classes)
        for j in range(self.num_prototypes):
            proto_class_map[j, j // self.num_proto_per_class] = 1
        self.register_buffer("proto_class_map", proto_class_map, persistent=True)
        self.last_layer = nn.Linear(in_features=self.num_prototypes, out_features=self.num_classes, bias=False)
        correct_locations = torch.t(self.proto_class_map)
        incorrect_locations = 1 - correct_locations
        self.last_layer.weight.data.copy_(correct_locations + incorrect_class_penalty * incorrect_locations)

    def prototype_is_active(self, proto_idx: int) -> bool:
        r"""Is the prototype *proto_idx* active or disabled?

        Args:
            proto_idx (int): Prototype index.
        """
        return not (int(torch.max(self.proto_class_map[proto_idx])) == 0)

    def forward(self, features: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        r"""Performs classification using a linear layer.

        Args:
            features (tensor): Convolutional features from extractor. Shape (N, D, H, W).

        Returns:
            Vector of logits. Shape (N, C).
            Tensor of min distances. Shape (N, P).
        """
        similarities = self.similarity_layer.similarities(features, self.prototypes)  # Shape (N, P, H, W)
        distances = self.similarity_layer.distances(features, self.prototypes)  # Shape (N, P, H, W)
        if self._compatibility_mode:
            # Reproduce legacy ProtoPNet operations
            min_distances = -torch.nn.functional.max_pool2d(
                -distances, kernel_size=(distances.size()[2], distances.size()[3])
            )
            min_distances = min_distances.view(-1, self.num_prototypes)
            similarities = torch.log((min_distances + 1) / (min_distances + 1e-4))
        else:
            min_distances = torch.min(distances.view(distances.shape[:2] + (-1,)), dim=2)[0]  # Shape (N, P)
            similarities = torch.max(similarities.view(similarities.shape[:2] + (-1,)), dim=2)[0]  # Shape (N, P)
        prediction = self.last_layer(similarities)

        return prediction, min_distances
