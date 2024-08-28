from __future__ import annotations

from enum import Enum
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from cabrnet.archs.generic.decision import CaBRNetClassifier
from cabrnet.core.utils.prototypes import init_prototypes
from cabrnet.core.utils.tree import BinaryNode


class SamplingStrategy(Enum):
    r"""Sampling strategy inside the decision tree."""

    DISTRIBUTED = 1  # The output is the sum of all leaf distributions, weighted by their respective probability
    SAMPLE_MAX = 2  # The output is the distribution of the leaf with the highest probability
    GREEDY = 3  # The output is computed by following the branch with the highest probability at each decision node.


class ProtoTreeClassifier(CaBRNetClassifier):
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
        similarity_config: dict[str, Any],
        num_classes: int,
        depth: int,
        num_features: int,
        leaves_init_mode: str = "ZEROS",
        proto_init_mode: str = "SHIFTED_NORMAL",
        log_probabilities: bool = False,
    ) -> None:
        r"""Initializes a ProtoTree classifier.

        Args:
            similarity_config (dict): Configuration of the layer used to compute similarity scores between the
                prototypes and the convolutional features.
            num_classes (int): Number of classes.
            depth (int): Depth of the binary decision tree.
            num_features (int): Number of features (size of each prototype).
            leaves_init_mode (str, optional): Init mode for leaves distributions. Default: Zero distribution.
            proto_init_mode (str, optional): Init mode for prototypes. Default: Shifted normal distribution.
            log_probabilities (bool, optional): If True, uses log of probabilities. Default: False.
        """
        super().__init__(num_classes=num_classes, num_features=num_features, proto_init_mode=proto_init_mode)

        # Sanity check
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

        if log_probabilities:
            self.register_buffer("_root_prob", torch.zeros(1))
        else:
            self.register_buffer("_root_prob", torch.ones(1))
        self.register_buffer("_root_greedy_path", torch.Tensor([True]))

        # Init similarity layer
        similarity_config["log_probabilities"] = log_probabilities
        self.build_similarity(similarity_config)

    def prototype_is_active(self, proto_idx: int) -> bool:
        r"""Is the prototype *proto_idx* active or disabled?

        Args:
            proto_idx (int): Prototype index.
        """
        return proto_idx in self._active_prototypes

    def forward(
        self, features: Tensor, strategy: SamplingStrategy = SamplingStrategy.DISTRIBUTED, **kwargs
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
