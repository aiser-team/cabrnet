"""Custom module for data preprocessing."""

from typing import Callable

from PIL import Image


def convert_to_rgb() -> Callable:
    r"""Converts MNIST dataset to RGB.

    Returns:
        A function that converts PIL images to RGB.
    """

    def _convert_to_rgb_fn(img: Image.Image) -> Image.Image:
        return img.convert("RGB")

    return _convert_to_rgb_fn
