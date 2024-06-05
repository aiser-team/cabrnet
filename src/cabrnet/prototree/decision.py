from __future__ import annotations
import argparse
from argparse import ArgumentParser
from enum import Enum
import torch.nn as nn
import torch
from cabrnet.generic.decision import CaBRNetGenericClassifier
from cabrnet.utils.prototypes import init_prototypes
from cabrnet.utils.similarities import L2Similarities
from cabrnet.utils.tree import BinaryNode
from torch import Tensor


class SamplingStrategy(Enum):
    r"""Sampling strategy inside the decision tree."""

    DISTRIBUTED = 1  # The output is the sum of all leaf distributions, weighted by their respective probability
    SAMPLE_MAX = 2  # The output is the distribution of the leaf with the highest probability
    GREEDY = 3  # The output is computed by following the branch with the highest probability at each decision node.


class ProtoTreeSimilarityScore(L2Similarities):
    r"""Class for computing similarity scores based on L2 distance in the convolutional space.

    Attributes:
        protopnet_compatibility: If True, uses the order of operations of ProtoPNet to compute the L2 distance.
        log_probabilities: If True, returns similarity scores as log of probabilities.
    """

    def __init__(self, num_prototypes: int, num_features: int, log_probabilities: bool = False) -> None:
        r"""Creates module for computing similarities based on L2 distance.

        Args:
            num_prototypes (int): Number of prototypes.
            num_features (int): Size of each prototype.
            log_probabilities (bool, optional): If True, returns values as log of probabilities. Default: False.

        """
        super().__init__(num_prototypes=num_prototypes, num_features=num_features)
        self.log_probabilities = log_probabilities

    def forward(self, features: Tensor, prototypes: Tensor) -> Tensor:
        r"""Computes similarity based on L2 distance using ||x - y||² = ||x||² + ||y||² - 2 x.y.

        Args:
            features (tensor): Input tensor. Shape (N, D, H, W).
            prototypes (tensor): Tensor of prototypes. Shape (P, D, 1, 1).

        Returns:
            Tensor of similarities. Shape (N, P, H, W).
        """
        distances = torch.sqrt(torch.abs(self.L2_square_distance(features=features, prototypes=prototypes)) + 1e-14)
        if self.log_probabilities:
            return -distances
        return torch.exp(-distances)


class ProtoTreeClassifier(CaBRNetGenericClassifier):
    r"""Classification pipeline for ProtoTree architecture.

    Attributes:
        num_classes: Number of output classes.
        num_features: Size of the features extracted by the convolutional extractor.
        prototypes: Tensor of prototypes.
        prototypes_init_mode: Initialization mode for the tensor of prototypes.
        similarity_layer: Layer used to compute similarity scores between the prototypes and the convolutional features.
        tree: Decision tree used to compute the final logit vector.
        depth: Depth of the decision tree.
        leaves_init_mode: Initialization mode for the leaf distributions.
        log_probabilities: If true, the decision tree treats similarity scores as log of probabilities.
    """

    def __init__(
        self,
        num_classes: int,
        depth: int,
        num_features: int,
        leaves_init_mode: str = "ZEROS",
        proto_init_mode: str = "SHIFTED_NORMAL",
        log_probabilities: bool = False,
    ) -> None:
        r"""Initializes a ProtoTree classifier.

        Args:
            num_classes (int): Number of classes.
            depth (int): Depth of the binary decision tree.
            num_features (int): Number of features (size of each prototype).
            leaves_init_mode (str, optional): Init mode for leaves distributions. Default: Zero distribution.
            proto_init_mode (str, optional): Init mode for prototypes. Default: Shifted normal distribution.
            log_probabilities (bool, optional): If True, uses log of probabilities. Default: False.
        """
        super().__init__(num_classes=num_classes, num_features=num_features, proto_init_mode=proto_init_mode)

        # Sanity check on all parameters
        assert depth > 0, f"Invalid tree depth: {depth}"

        self.depth = depth
        self.leaves_init_mode = leaves_init_mode
        self.log_probabilities = log_probabilities
        self.tree = BinaryNode.create_binary_tree(
            depth=self.depth,
            num_classes=num_classes,
            leaves_init_mode=leaves_init_mode,
            log_probabilities=log_probabilities,
        )

        # Init prototypes
        num_prototypes = self.tree.num_prototypes
        self.prototypes = nn.Parameter(  # type: ignore
            init_prototypes(num_prototypes=num_prototypes, num_features=self.num_features, init_mode=proto_init_mode)
        )
        self._active_prototypes = self.tree.active_prototypes
        self.similarity_layer = ProtoTreeSimilarityScore(
            num_prototypes=num_prototypes, num_features=self.num_features, log_probabilities=log_probabilities
        )
        if log_probabilities:
            self.register_buffer("_root_prob", torch.zeros(1))
        else:
            self.register_buffer("_root_prob", torch.ones(1))
        self.register_buffer("_root_greedy_path", torch.Tensor([True]))

    def prototype_is_active(self, proto_idx: int) -> bool:
        r"""Is the prototype *proto_idx* active or disabled?
        Args:
            proto_idx (int): Prototype index.
        """
        return proto_idx in self._active_prototypes

    def forward(
        self, features: Tensor, strategy: SamplingStrategy = SamplingStrategy.DISTRIBUTED
    ) -> tuple[Tensor, dict] | None:
        r"""Performs classification using a decision tree.

        Args:
            features (tensor): Convolutional features from extractor. Shape (N, D, H, W).
            strategy (SamplingStrategy, optional): Sampling strategy. Default: Distributed.

        Returns:
            Vector of logits. Shape (N, C).
        """
        similarities = self.similarity_layer(features, self.prototypes)  # Shape (N, P, H, W)
        # Use only maximum similarity score for each prototype
        similarities = torch.max(similarities.view(similarities.shape[:2] + (-1,)), dim=2)[0]  # Shape (N, P)

        prediction, tree_info = self.tree(
            parent_probs=self._root_prob,
            conditional_probs=self._root_prob,
            similarities=similarities,
            greedy_path=self._root_greedy_path,
        )
        if strategy in [SamplingStrategy.SAMPLE_MAX, SamplingStrategy.GREEDY]:
            leaf_names = [leaf.node_id for leaf in self.tree.leaves]
            # Aggregate information from all leaves. Shape num_leaves x batch_size x 1
            if strategy == SamplingStrategy.SAMPLE_MAX:
                leaf_probabilities = torch.stack(
                    [tree_info[leaf.node_id]["absolute_probability"] for leaf in self.tree.leaves],
                    dim=0,
                )
            else:
                leaf_probabilities = (
                    torch.stack(
                        [tree_info[leaf.node_id]["absolute_probability"] for leaf in self.tree.leaves],
                        dim=0,
                    )
                    * 1.0
                )  # Convert boolean to float prior to argmax operation
            # leaf_distributions has shape num_leaves x num_classes
            leaf_distributions = torch.cat([leaf.distribution for leaf in self.tree.leaves], dim=0)
            # leaf_idxs has shape batch_size
            leaf_idxs = torch.argmax(leaf_probabilities, dim=0)
            prediction = torch.index_select(input=leaf_distributions, dim=0, index=leaf_idxs[..., 0])
            tree_info["decision_leaf"] = [leaf_names[int(leaf_idx.item())] for leaf_idx in leaf_idxs]
        return prediction, tree_info

    @staticmethod
    def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
        r"""Adds arguments for creating a ProtoTreeClassifier.

        Args:
            parser (ArgumentParser, optional): Existing argument parser (if any). Default: None.

        Returns:
            Parser with arguments.
        """
        if parser is None:
            parser = ArgumentParser(description="builds a ProtoTreeClassifier object.")
        parser = CaBRNetGenericClassifier.create_parser(parser)
        parser.add_argument(
            "--leaves-init-mode",
            type=str,
            default="zeros",
            choices=["zeros", "normal"],
            metavar="mode",
            help="initialisation mode for the leaves distributions.",
        )
        parser.add_argument("--tree-depth", type=int, default=9, metavar="num", help="depth of the decision tree.")
        parser.add_argument(
            "--use-log-probabilities",
            action="store_true",
            help="use log probabilities.",
        )
        return parser

    @staticmethod
    def build_from_parser(args: argparse.Namespace) -> ProtoTreeClassifier:
        r"""Builds a classifier from the command line.

        Args:
            args (Namespace): Parsed command line.

        Returns:
            Prototree classifier.
        """
        return ProtoTreeClassifier(
            num_classes=args.num_classes,
            depth=args.tree_depth,
            num_features=args.num_features,
            leaves_init_mode=args.leaves_init_mode,
            proto_init_mode=args.prototype_init_mode,
            log_probabilities=args.use_log_probabilities,
        )
