import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch import Tensor


def polarity_and_collapse(
    array: np.ndarray,
    polarity: str | None = None,
    dim: int | None = None,
) -> np.ndarray:
    r"""Applies polarity filter (optional) followed by average over channels (optional).

    Args:
        array (Numpy array): Target array.
        polarity (str, optional): If given, applies a polarity filter (positive, negative, absolute). Default: None.
        dim (int, optional): If given, dimension across which channels are averaged. Default: None.

    Returns:
        Modified array.
    """
    assert polarity in [None, "positive", "negative", "absolute"], f"Invalid polarity {polarity}"

    # Polarity first
    if polarity == "positive":
        array = np.maximum(0, array)
    elif polarity == "negative":
        array = np.abs(np.minimum(0, array))
    elif polarity == "absolute":
        array = np.abs(array)

    # Channel average
    if dim is not None:
        array = np.average(array, axis=dim)
    return array


def normalize_min_max(array: np.ndarray) -> np.ndarray:
    r"""Performs min-max normalization of a numpy array.

    Args:
        array (Numpy array): Target array.

    Returns:
        Normalized array.
    """
    v_min = np.amin(array)
    v_max = np.amax(array)
    if v_min == v_max:
        return np.zeros_like(array)
    # Avoid division by zero
    return (array - v_min) / (v_max - v_min)


def post_process(
    array: np.ndarray,
    img: Image.Image,
    img_tensor: Tensor,
    resize: bool = True,
    polarity: str | None = "absolute",
    gaussian_ksize: int = 5,
    normalize: bool = False,
    grads_x_input: bool = False,
) -> np.ndarray:
    r"""Applies post-processing on numpy array.

    Args:
        array (Numpy array): Source array.
        img (Image): Raw input image.
        img_tensor (tensor): Input image tensor.
        resize (bool, optional): If True, resizes the array to the original image size. Default: True.
        polarity (str, optional): Polarity filter (None, "absolute", "positive", or "negative"). Default: absolute.
        gaussian_ksize (int, optional): Size of gaussian filter kernel size. Default: 5.
        normalize (bool, optional): If True, performs min-max normalization. Default: False.
        grads_x_input (bool, optional): If True, performs element-wise multiplication between gradient and image.
            Default: False.

    Returns:
        Result array.
    """
    if img_tensor.dim() > 3:
        img_tensor = img_tensor[0]
    if grads_x_input:
        assert (
            array.shape == img_tensor.shape
        ), f"Mismatching image tensor size. Expected {array.shape} but found {img_tensor.shape}"
        # Element-wise multiplication with source image
        array *= img_tensor.detach().cpu().numpy()

    # Post-processing
    array = polarity_and_collapse(array, polarity=polarity, dim=0)
    if gaussian_ksize:
        array = gaussian_filter(array, sigma=gaussian_ksize)
    if resize:
        array = cv2.resize(src=array, dsize=(img.width, img.height), interpolation=cv2.INTER_CUBIC)
    if normalize:
        array = normalize_min_max(array)
    return array
