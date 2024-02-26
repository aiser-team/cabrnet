from __future__ import annotations

from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from cabrnet.generic.decision import CaBRNetAbstractClassifier
from cabrnet.utils.prototypes import init_prototypes
from cabrnet.utils.similarities import L2Similarities
from torch import Tensor


class ProtoPNetSimilarityScore(L2Similarities):
    def forward(self, features: Tensor, prototypes: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore
        """
        Compute similarity based on L2 distance using ||x - y||² = ||x||² + ||y||² - 2 x.y
        Args:
            features: Input tensor. Shape (N, D, H, W)
            prototypes: Tensor of prototypes. Shape (P, D, 1, 1)

        Returns:
            Tensor of similarities. Shape (N, P, H, W)
            Tensor of distances. Shape (N, P, H, W)
        """
        distances = torch.relu(self.L2_square_distance(features=features, prototypes=prototypes))
        return torch.log((distances + 1) / (distances + 1e-4)), distances


class ProtoPNetClassifier(CaBRNetAbstractClassifier, nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_features: int,
        num_proto_per_class: int,
        proto_init_mode: str = "SHIFTED_NORMAL",
        incorrect_class_penalty: float = -0.5,
        compatibility_mode: bool = False,
    ) -> None:
        """Create a ProtoPNet classifier.

        Args:
            num_classes: Number of classes
            num_features: Number of features (size of each prototype)
            num_proto_per_class: Number of prototypes per class
            proto_init_mode: Init mode for prototypes
            compatibility_mode: Compatibility mode with legacy ProtoPNet
        """
        nn.Module.__init__(self)
        CaBRNetAbstractClassifier.__init__(
            self, num_classes=num_classes, num_features=num_features, proto_init_mode=proto_init_mode
        )

        # Sanity check on all parameters
        assert num_proto_per_class > 0, f"Invalid number of prototypes per class: {num_proto_per_class}"

        self.num_proto_per_class = num_proto_per_class
        self._compatibility_mode = compatibility_mode

        # Init prototypes
        self.prototypes = nn.Parameter(
            init_prototypes(
                num_prototypes=self.max_num_prototypes, num_features=self.num_features, init_mode=proto_init_mode
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

    @property
    def max_num_prototypes(self) -> int:
        """
        Returns: Maximum number of prototypes (might differ from current number of prototypes due to pruning)
        """
        return self.num_proto_per_class * self.num_classes

    @property
    def num_prototypes(self) -> int:
        """Retrieve the current number of prototypes. Note: this value may change after pruning

        Returns:
            The total number of prototypes.
        """
        return self.prototypes.size(0)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Perform classification using decision tree.

        Args:
            features: Convolutional features from extractor. Shape (N, D, H, W)

        Returns:
            Vector of logits. Shape (N, C)
            Tensor of min distances. Shape (N, P)
        """
        similarities, distances = self.similarity_layer(features, self.prototypes)  # Shape (N, P, H, W)
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
        """Add arguments for creating a ProtoPNetClassifier.

        Args:
            parser: Existing argument parser (if any)

        Returns:
            Parser with arguments
        """
        if parser is None:
            parser = ArgumentParser(description="builds a ProtoPNetClassifier object.")
        parser = CaBRNetAbstractClassifier.create_parser(parser)
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
        """Builds a classifier from the command line

        Args:
            args: Parsed command line

        Returns:
            ProtoPNet classifier
        """
        return ProtoPNetClassifier(
            num_classes=args.num_classes,
            num_features=args.num_features,
            num_proto_per_class=args.num_proto_per_class,
            proto_init_mode=args.prototype_init_mode,
            incorrect_class_penalty=args.incorrect_class_penalty,
            compatibility_mode=args.compatibility_mode,
        )
