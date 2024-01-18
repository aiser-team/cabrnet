import numpy as np
from PIL import Image
from torch import Tensor
import cv2
from scipy.ndimage import gaussian_filter


def polarity_and_collapse(
    array: np.ndarray,
    polarity: str | None = None,
    dim: int | None = None,
) -> np.ndarray:
    """Apply polarity filter (optional) followed by average over channels (optional)
    Args:
        array: target array
        polarity: polarity (positive, negative, absolute)
        dim: dimension across which channels are averaged

    Returns:
        modified array
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
    """Perform min-max normalization of a numpy array

    Args:
        array: target array

    Returns:
        normalized array
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
) -> np.array:
    """Apply post-processing on numpy array

    Args:
        array: source array
        img: raw input image
        img_tensor: input image tensor
        resize: resize array to original image size
        polarity: polarity filter (either None, "absolute", "positive", or "negative")
        gaussian_ksize: size of gaussian filter kernel size
        normalize: perform min-max normalization
        grads_x_input: perform element-wise multiplication between gradient and image

    Returns:
        result array
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
