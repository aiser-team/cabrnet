from cabrnet.utils.prototypes import prototype_init_modes
from argparse import ArgumentParser
import torch.nn as nn


class CaBRNetGenericClassifier(nn.Module):
    r"""Abstract class for CaBRNet classification based on extracted features.

    Attributes:
        num_classes: Number of output classes.
        num_features: Size of the features extracted by the convolutional extractor.
        prototypes: Tensor of prototypes.
        prototypes_init_mode: Initialization mode for the tensor of prototypes.
        similarity_layer: Layer used to compute similarity scores between the prototypes and the convolutional features.
    """

    prototypes: nn.Parameter
    similarity_layer: nn.Module

    def __init__(
        self,
        num_classes: int,
        num_features: int,
        proto_init_mode: str = "SHIFTED_NORMAL",
    ) -> None:
        r"""Initializes a CaBRNetGenericClassifier.

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

    @property
    def num_prototypes(self) -> int:
        r"""Returns the maximum number of prototypes, as given by the corresponding tensor.
        Note: some prototypes might be inactive."""
        return self.prototypes.size(0)

    def prototype_is_active(self, proto_idx: int) -> bool:
        r"""Is the prototype *proto_idx* active or disabled?

        Args:
            proto_idx (int): Prototype index.
        """
        raise NotImplementedError

    @staticmethod
    def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
        r"""Adds essential arguments for creating a CaBRNet classifier.

        Args:
            parser (ArgumentParser, optional): Existing argument parser (if any). Default: None.

        Returns:
            Parser with arguments.
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
