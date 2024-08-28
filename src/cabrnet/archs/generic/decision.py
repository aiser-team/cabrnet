import importlib
from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn
from torch import Tensor

from cabrnet.core.utils.prototypes import prototype_init_modes
from cabrnet.core.utils.similarities import SimilarityLayer


class CaBRNetClassifier(nn.Module, ABC):
    r"""Abstract class for CaBRNet classification based on extracted features.

    Attributes:
        num_classes: Number of output classes.
        num_features: Size of the features extracted by the convolutional extractor.
        prototypes: Tensor of prototypes.
        prototypes_init_mode: Initialization mode for the tensor of prototypes.
        similarity_layer: Layer used to compute similarity scores between the prototypes and the convolutional features.
    """

    prototypes: nn.Parameter
    similarity_layer: SimilarityLayer

    def __init__(
        self,
        num_classes: int,
        num_features: int,
        proto_init_mode: str = "SHIFTED_NORMAL",
    ) -> None:
        r"""Initializes a CaBRNetClassifier.

        Args:
            num_classes (int): Number of classes.
            num_features (int): Number of features (size of each prototype).
            proto_init_mode (str, optional): Init mode for prototypes. Default: Shifted normal distribution.
        """
        super().__init__()

        # Sanity check on all parameters
        assert num_classes > 1, f"Invalid number of classes: {num_classes}"
        assert num_features > 0, f"Invalid number of features: {num_features}"
        assert (
            proto_init_mode in prototype_init_modes
        ), f"Unsupported prototype initialization mode: {proto_init_mode}. Choices are: {prototype_init_modes}"

        self.num_classes = num_classes
        self.num_features = num_features
        self.prototypes_init_mode = proto_init_mode

    def build_similarity(self, config: dict[str, Any]) -> None:
        r"""Builds the similarity layer based on information regarding the decision process.

        Args:
            config (dict): Configuration of the similarity layer.
        """
        if "name" not in config:
            raise ValueError(f"Missing mandatory field 'name' in similarity layer configuration")

        # Load similarity module and fetch parameters (if any)
        module = importlib.import_module(config.get("module", "cabrnet.core.utils.similarities"))
        params = config.get("params", {})

        # Create layer, providing all relevant information that could be useful
        self.similarity_layer = getattr(module, config["name"])(
            num_features=self.num_features, num_prototypes=self.num_prototypes, **params
        )

    @property
    def num_prototypes(self) -> int:
        r"""Returns the maximum number of prototypes, as given by the corresponding tensor.

        Note: some prototypes might be inactive.
        """
        return self.prototypes.size(0)

    def prototype_is_active(self, proto_idx: int) -> bool:
        r"""Is the prototype *proto_idx* active or disabled?

        Args:
            proto_idx (int): Prototype index.
        """
        return True

    @abstractmethod
    def forward(self, features: Any, **kwargs) -> Any:
        r"""Performs classification using the extracted features.

        Args:
            features (tensor): Convolutional features from extractor.

        Returns:
            Model output (usually a vector of logits).
        """
        raise NotImplementedError

    def similarities(self, features: Tensor, **kwargs) -> Tensor:
        r"""Returns similarity scores.

        Args:
            features (tensor): Feature tensor.

        Returns:
            Tensor of similarity scores.
        """
        return self.similarity_layer.similarities(features, self.prototypes, **kwargs)

    def distances(self, features: Tensor, **kwargs) -> Tensor:
        r"""Returns pairwise distances between each feature vector and each prototype.

        Args:
            features (tensor): Features tensor.

        Returns:
            Tensor of distances.
        """
        return self.similarity_layer.distances(features, self.prototypes, **kwargs)
