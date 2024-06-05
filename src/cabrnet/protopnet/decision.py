from __future__ import annotations

from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from cabrnet.generic.decision import CaBRNetGenericClassifier
from cabrnet.utils.prototypes import init_prototypes
from cabrnet.utils.similarities import L2Similarities
from torch import Tensor


class ProtoPNetSimilarityScore(L2Similarities):
    r"""Class for computing similarity scores based on L2 distance in the convolutional space.

    Attributes:
        protopnet_compatibility: If True, uses the order of operations of ProtoPNet to compute the L2 distance.
    """

    def forward(
        self, features: Tensor, prototypes: Tensor, output_distances: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""Computes similarity based on L2 distance using ||x - y||² = ||x||² + ||y||² - 2 x.y.

        Args:
            features (tensor): Input tensor. Shape (N, D, H, W).
            prototypes (tensor): Tensor of prototypes. Shape (P, D, 1, 1).
            output_distances (bool, optional): If True, also outputs raw distances. Default: False.

        Returns:
            Tensor of similarities. Shape (N, P, H, W).
            Tensor of distances. Shape (N, P, H, W) (only if output_distances is enabled).
        """
        distances = torch.relu(self.L2_square_distance(features=features, prototypes=prototypes))
        similarities = torch.log((distances + 1) / (distances + 1e-4))
        if output_distances:
            return similarities, distances
        return similarities


class ProtoPNetClassifier(CaBRNetGenericClassifier):
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
        num_classes: int,
        num_features: int,
        num_proto_per_class: int,
        proto_init_mode: str = "SHIFTED_NORMAL",
        incorrect_class_penalty: float = -0.5,
        compatibility_mode: bool = False,
    ) -> None:
        r"""Initializes a ProtoPNet classifier.

        Args:
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

        # Sanity check on all parameters
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
        self.similarity_layer = ProtoPNetSimilarityScore(
            num_prototypes=self.num_prototypes,
            num_features=self.num_features,
            protopnet_compatibility=compatibility_mode,
        )

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

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        r"""Performs classification using a linear layer.

        Args:
            features (tensor): Convolutional features from extractor. Shape (N, D, H, W).

        Returns:
            Vector of logits. Shape (N, C).
            Tensor of min distances. Shape (N, P).
        """
        similarities, distances = self.similarity_layer(
            features, self.prototypes, output_distances=True
        )  # Shape (N, P, H, W)
        if self._compatibility_mode:
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

    @staticmethod
    def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
        r"""Adds arguments for creating a ProtoPNetClassifier.

        Args:
            parser (ArgumentParser, optional): Existing argument parser (if any). Default: None.

        Returns:
            Parser with arguments.
        """
        if parser is None:
            parser = ArgumentParser(description="builds a ProtoPNetClassifier object.")
        parser = CaBRNetGenericClassifier.create_parser(parser)
        parser.add_argument(
            "--num-proto-per-class",
            type=int,
            default=10,
            metavar="num",
            help="number of prototype for each category.",
        )
        return parser

    @staticmethod
    def build_from_parser(args: Namespace) -> ProtoPNetClassifier:
        r"""Builds a classifier from the command line.

        Args:
            args (Namespace): Parsed command line.

        Returns:
            ProtoPNet classifier.
        """
        return ProtoPNetClassifier(
            num_classes=args.num_classes,
            num_features=args.num_features,
            num_proto_per_class=args.num_proto_per_class,
            proto_init_mode=args.prototype_init_mode,
            incorrect_class_penalty=args.incorrect_class_penalty,
            compatibility_mode=args.compatibility_mode,
        )
