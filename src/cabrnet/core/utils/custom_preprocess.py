"""Custom module for data preprocessing."""

from cabrnet.core.utils.transform import load_transform
from typing import Tuple, Dict, Callable, Any
import numpy as np
import torchvision.transforms
import torch
from torch import Tensor
from PIL import Image


def convert_to_rgb() -> Callable:
    r"""Converts MNIST dataset to RGB.

    Returns:
        A function that converts PIL images to RGB.
    """

    def _convert_to_rgb_fn(img: Image.Image) -> Image.Image:
        return img.convert("RGB")

    return _convert_to_rgb_fn


def batch_mixup(data: Tensor, labels: Tensor, alpha: float = 0) -> tuple[Tensor, Tensor, float]:
    r"""Performs a random mix between elements inside a batch of data.
    Adapted from https://github.com/gmum/ProtoPool/.

    Args:
        data (tensor): Batch data. Shape (N x D).
        labels (tensor): Batch labels. Shape (N x L).
        alpha (float, optional): Parameter of the Beta-distribution controlling the percentage of mix. Default: 0.

    Returns:
        Modified batch data. Shape (N x A).
        Modified batch labels. Shape (N x L).
        Percentage of mix.
    """
    percentage = np.random.beta(a=alpha, b=alpha) if alpha > 0 else 1.0
    # Draws a random permutation of indices inside the batch
    index = torch.randperm(data.size(0), dtype=torch.long, device=data.device)
    return percentage * data + (1 - percentage) * data[index], labels[index], percentage


class TrivialAugmentWideShapeOnly(torchvision.transforms.TrivialAugmentWide):
    r"""Variant of TrivialAugmentWide used in PIPNet."""

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        r"""Returns a data augmentation space, containing only shape transforms.

        Args:
            num_bins (int): Default number of magnitude bins.
        """
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True),
        }


class TrivialAugmentWideNoShape(torchvision.transforms.TrivialAugmentWide):
    r"""Variant of TrivialAugmentWide used in PIPNet."""

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        r"""Returns a data augmentation space, containing everything but shape transforms.

        Args:
            num_bins (int): Default number of magnitude bins.
        """
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


class TrivialAugmentWideNoShapeNoColor(torchvision.transforms.TrivialAugmentWide):
    r"""Variant of TrivialAugmentWide used in PIPNet."""

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        r"""Returns a data augmentation space, containing everything but shape/color transforms.

        Args:
            num_bins (int): Default number of magnitude bins.
        """
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True),  # Minimal change to color
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


class AugmentationSplitter:
    r"""Applies a random transformation to *n* copies of the same input tensor."""

    def __init__(self, num_samples: int, transform: dict[str, Any] | Callable):
        r"""Initializes an AugmentationSplitter transform module.
        Args:
            num_samples (int): Number of samples to create.
            transform (Callable | dict): Transformation to apply to the input.
        """
        self.num_samples = num_samples
        if isinstance(transform, dict):
            self.transform = load_transform(transform)
        else:
            self.transform = transform

    def __call__(self, img: Any) -> tuple[Any, ...]:
        r"""Returns a tuple of transformed inputs.

        Args:
            img (any): Input image or tensor.
        """
        return tuple([self.transform(img) for _ in range(self.num_samples)])

    def __repr__(self) -> str:
        r"""Returns object configuration."""
        format_string = self.__class__.__name__ + "("
        format_string += "\n"
        format_string += f"    {self.transform}"
        format_string += "\n)"
        return format_string
