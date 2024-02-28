from abc import ABC, abstractmethod
from cabrnet.utils.prototypes import prototype_init_modes
from argparse import ArgumentParser
import torch.nn as nn


class CaBRNetAbstractClassifier(ABC):
    prototypes: nn.Parameter()

    """Abstract class for CaBRNet classification based on extracted features
    Args:
        num_classes: Number of classes
        num_features: Number of features (size of each prototype)
        proto_init_mode: Init mode for prototypes
    """

    def __init__(
        self,
        num_classes: int,
        num_features: int,
        proto_init_mode: str = "SHIFTED_NORMAL",
    ) -> None:
        # Sanity check on all parameters
        assert num_classes > 1, f"Invalid number of classes: {num_classes}"
        assert num_features > 0, f"Invalid number of features: {num_features}"
        assert (
            proto_init_mode in prototype_init_modes
        ), f"Unsupported prototype initialization mode: {proto_init_mode}. Choices are: {prototype_init_modes}"

        self.num_classes = num_classes
        self.num_features = num_features
        self.prototypes_init_mode = proto_init_mode
        # Dummy initialisation of prototypes and similarity layer
        self.prototypes = None
        self.similarity_layer = None

    @property
    @abstractmethod
    def num_prototypes(self) -> int:
        """
        Returns: Current number of prototypes
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def max_num_prototypes(self) -> int:
        """
        Returns: Maximum number of prototypes (might differ from current number of prototypes due to pruning)
        """
        raise NotImplementedError

    @staticmethod
    def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
        """Add essential arguments for creating a CaBRNet classifier
        Args:
            parser: Existing argument parser (if any)

        Returns:
            Parser with arguments
        """
        if parser is None:
            parser = ArgumentParser(description="builds a CaBRNet classifier object.")
        parser.add_argument(
            "--num-classes",
            type=int,
            default=200,
            metavar="num",
            help="number of categories in the classification task.",
        )
        parser.add_argument(
            "--num-features", type=int, default=256, metavar="num", help="number of features for each prototype."
        )
        parser.add_argument(
            "---prototype-init-mode",
            type=str,
            default="shifted_normal",
            choices=["shifted_normal", "normal", "uniform"],
            metavar="mode",
            help="initialisation mode for the prototypes.",
        )
        return parser
