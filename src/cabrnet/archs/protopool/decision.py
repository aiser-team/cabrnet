from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from cabrnet.archs.generic.decision import CaBRNetClassifier
from cabrnet.core.utils.prototypes import init_prototypes
from torch import Tensor
from typing import Any


class ProtoPoolClassifier(CaBRNetClassifier):
    r"""Classification pipeline for ProtoPool architecture.

    Attributes:
        num_classes: Number of output classes.
        num_features: Size of the features extracted by the convolutional extractor.
        num_slots_per_class: Number of slots allocated to each class.
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
        num_prototypes: int,
        num_slots_per_class: int,
        proto_init_mode: str = "UNIFORM",
        incorrect_class_penalty: float = 0,
        compatibility_mode: bool = False,
        **kwargs,
    ) -> None:
        r"""Initializes a ProtoPool classifier.

        Args:
            similarity_config (dict): Configuration of the layer used to compute similarity scores between the
                prototypes and the convolutional features.
            num_classes (int): Number of classes.
            num_features (int): Number of features (size of each prototype).
            num_prototypes (int): Maximum number of prototypes.
            num_slots_per_class (int): Number of slots allocated to each class.
            proto_init_mode (str, optional): Init mode for prototypes. Default: UNIFORM.
            incorrect_class_penalty (float, optional): Initial penalty for incorrect classes in the linear layer.
                Default: 0.
            compatibility_mode (bool, optional): If True, enables compatibility mode with legacy ProtoPool.
                Default: False.
        """
        super().__init__(num_classes=num_classes, num_features=num_features, proto_init_mode=proto_init_mode)

        assert num_prototypes > 0, f"Invalid number of prototypes per class: {num_prototypes}"
        self._compatibility_mode = compatibility_mode
        self.num_slots_per_class = num_slots_per_class

        # For a given class, and after Softmax/Gumbel-Softmax normalization, each slot represents a distribution
        # over the prototypes
        self.proto_slot_map = nn.Parameter(torch.zeros((num_classes, num_prototypes, num_slots_per_class)))
        nn.init.xavier_normal_(self.proto_slot_map, gain=1)

        # Init prototypes
        self.prototypes = nn.Parameter(  # type: ignore
            init_prototypes(
                num_prototypes=num_prototypes,
                num_features=self.num_features,
                init_mode=proto_init_mode,
            )
        )

        # Init similarity layer
        if compatibility_mode:
            # Stability factor used in legacy code
            similarity_config["stability_factor"] = 1e-4
        self.build_similarity(similarity_config)

        # Initialize last layer based on the (fixed) association between a slot and a class
        num_slots = self.num_classes * self.num_slots_per_class
        slot_class_map = torch.zeros(num_slots, self.num_classes)
        for j in range(num_slots):
            slot_class_map[j, j // self.num_slots_per_class] = 1

        self.register_buffer("slot_class_map", slot_class_map, persistent=True)
        self.last_layer = nn.Linear(in_features=num_slots, out_features=self.num_classes, bias=False)
        correct_locations = torch.t(self.slot_class_map)
        incorrect_locations = 1 - correct_locations
        self.last_layer.weight.data.copy_(correct_locations + incorrect_class_penalty * incorrect_locations)

    @property
    def class_mapping(self) -> np.ndarray:
        r"""Returns the mapping between each class and a subset of prototypes."""
        with torch.no_grad():
            if self._compatibility_mode:
                # Makes the computation of the class mapping deterministic
                torch.manual_seed(0)
            normalized_proto_slot_map = (
                nn.functional.gumbel_softmax(self.proto_slot_map * 1e4, tau=0.5, dim=1).detach().cpu()
            )  # Shape C x P x S
            # For each class, keep track of the related prototypes (one for each slot)
            return torch.argmax(normalized_proto_slot_map, dim=1).cpu().numpy()  # Shape C x S

    def prototype_is_active(self, proto_idx: int) -> bool:
        r"""Is the prototype *proto_idx* active or disabled? True iff the prototype is associated with at least
        one slot.

        Args:
            proto_idx (int): Prototype index.
        """
        return proto_idx in list(self.class_mapping.reshape(-1))

    @property
    def prototype_class_mapping(self) -> np.ndarray:
        r"""Mapping between prototypes and classes.

        Returns:
            Binary array of shape (P, C).
        """
        slot_class_mapping = self.class_mapping  # Shape C x S
        proto_class_map = np.zeros((self.num_prototypes, self.num_classes))
        for cidx in range(self.num_classes):
            for pidx in slot_class_mapping[cidx]:
                proto_class_map[pidx, cidx] = 1
        return proto_class_map

    def forward(self, features: Tensor, gumbel_scale: int = 0, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        r"""Performs classification using a linear layer and based on the slot assignments.

        Args:
            features (tensor): Convolutional features from extractor. Shape (N, D, H, W).
            gumbel_scale (int, optional): Scale factor of the Gumbel Softmax. If 0, uses Softmax instead. Default: 0.

        Returns:
            Vector of logits. Shape (N, C).
            Tensor of min distances. Shape (N, P).
            Tensor of probabilities that each prototype belongs to a given slot associated with a given class.
                Shape (C,P,S).
        """
        # Probability that each prototype belongs to a given slot associated with a given class
        if gumbel_scale == 0:
            if self._compatibility_mode:
                proto_slot_probs = torch.softmax(self.proto_slot_map, dim=1)
            else:
                # Assume that self.proto_slot_map is already a one-hot distribution
                proto_slot_probs = self.proto_slot_map
        else:
            proto_slot_probs = nn.functional.gumbel_softmax(self.proto_slot_map * gumbel_scale, tau=0.5, dim=1)

        distances = self.similarity_layer.distances(features, self.prototypes)  # Shape (N, P, H, W)

        if self._compatibility_mode:
            # Reproduce legacy ProtoPool operations
            min_distances = -torch.nn.functional.max_pool2d(
                -distances, kernel_size=(distances.size()[2], distances.size()[3])
            ).flatten(start_dim=1)
            avg_distances = torch.nn.functional.avg_pool2d(
                distances, kernel_size=(distances.size()[2], distances.size()[3])
            ).flatten(start_dim=1)
        else:
            min_distances = torch.min(distances.flatten(start_dim=2), dim=2)[0]  # Shape (N, P)
            avg_distances = torch.mean(distances.flatten(start_dim=2), dim=2)  # Shape (N, P)

        # Distribute each prototype distance to the class slot(s) it belongs to, i.e.
        # convert a tensor of shape N x P into a tensor of shape N x C x S based on the probability of association
        # between a prototype and a class slot
        expanded_min_distances = torch.einsum("bp,cpn->bcn", min_distances, proto_slot_probs)
        expanded_avg_distances = torch.einsum("bp,cpn->bcn", avg_distances, proto_slot_probs)
        similarities = self.similarity_layer.distances_to_similarities(
            expanded_min_distances
        ) - self.similarity_layer.distances_to_similarities(expanded_avg_distances)

        similarities = similarities.flatten(start_dim=1)
        prediction = self.last_layer(similarities)

        return prediction, min_distances, proto_slot_probs
