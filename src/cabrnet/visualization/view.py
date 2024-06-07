import numpy as np
from cv2 import COLORMAP_JET, applyColorMap
from PIL import Image, ImageDraw


def compute_bbox(array: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    r"""Returns the coordinates of the bounding box.

    It is the smallest bounding box such that all values outside the box are smaller or equal to threshold.

    Args:
        array (Numpy array): Target array.
        threshold (float): Threshold value.

    Returns:
        Coordinates of the bounding box (x_min, x_max, y_min, y_max) where x and y correspond to the first and second
        dimension of the array.
    """
    assert array.ndim == 2, f"Invalid number of dimensions in array. Expected 2 but got {array.ndim}."
    x_min, x_max, y_min, y_max = 0, 0, 0, 0
    for y_min in range(array.shape[0]):
        if np.max(array[y_min]) > threshold:
            break
    for y_max in reversed(range(array.shape[0])):
        if np.max(array[y_max]) > threshold:
            break
    for x_min in range(array.shape[1]):
        if np.max(array[:, x_min]) > threshold:
            break
    for x_max in reversed(range(array.shape[1])):
        if np.max(array[:, x_max]) > threshold:
            break
    return x_min, x_max + 1, y_min, y_max + 1


def crop_to_percentile(img: Image.Image, sim_map: np.ndarray, percentile: float, **kwargs) -> Image.Image:
    r"""Crops an image based on a bounding box computed on the similarity map with a given percentile.

    Args:
        img (Image): Original image.
        sim_map (Numpy array): Similarity map, normalized between 0.0 and 1.0.
        percentile (float): Crop percentile.

    Returns:
        Cropped image.
    """
    assert 0 <= np.amin(sim_map) and np.amax(sim_map) <= 1.0, f"Expected normalized similarity map in {__name__}"
    x_min, x_max, y_min, y_max = compute_bbox(array=sim_map, threshold=1.0 - percentile)
    return img.crop(box=(x_min, y_min, x_max, y_max))


def bbox_to_percentile(
    img: Image.Image, sim_map: np.ndarray, percentile: float, thickness: int = 4, **kwargs
) -> Image.Image:
    r"""Shows image with a bounding box computed on the similarity map with a given percentile.

    Args:
        img (Image): Original image.
        sim_map (Numpy array): Similarity map, normalized between 0.0 and 1.0.
        percentile (float): Crop percentile.
        thickness (int, optional): Rectangle thickness. Default: 4.

    Returns:
        Image with bounding box.
    """
    assert 0 <= np.amin(sim_map) and np.amax(sim_map) <= 1.0, f"Expected normalized similarity map in {__name__}"
    x_min, x_max, y_min, y_max = compute_bbox(array=sim_map, threshold=1.0 - percentile)
    # Copy image before drawing
    dst_img = img.copy()
    # Draw bounding box
    img_with_bbox = ImageDraw.Draw(im=dst_img)
    if x_max > x_min and y_max > y_min:
        # Handle the case when no bounding box is found
        img_with_bbox.rectangle(xy=((x_min, y_min), (x_max, y_max)), width=thickness, outline="yellow")
    return dst_img


def heatmap(
    img: Image.Image, sim_map: np.ndarray, overlay: bool = False, percentile: float = 0, thickness: int = 4, **kwargs
) -> Image.Image:
    r"""Converts a similarity map into a heatmap.

    Args:
        img (Image): Original image.
        sim_map (Numpy array): Similarity map, normalized between 0.0 and 1.0.
        overlay (bool, optional): If True, overlays the heatmap on top of original image. Default: False.
        percentile (float, optional): Selection percentile. Default: 0.0.
        thickness (int, optional): Rectangle thickness. Default: 4.

    Returns:
        Image with bounding box.
    """
    assert (img.height, img.width) == sim_map.shape, (
        f"Mismatching shapes between heatmap and original image: "
        f"expected ({img.height}, {img.width}) but found {sim_map.shape}"
    )
    assert 0 <= np.amin(sim_map) and np.amax(sim_map) <= 1.0, f"Expected normalized similarity map in {__name__}"
    sim_heatmap = applyColorMap((255 * sim_map).astype(np.uint8), COLORMAP_JET).astype(np.uint8)
    # Convert BGR format returned by OpenCV into RGB
    sim_heatmap = sim_heatmap[..., ::-1]

    if overlay:
        img_array = np.array(img)
        if img_array.ndim < 3:
            img_array = np.expand_dims(img_array, -1)
        sim_heatmap = (0.4 * sim_heatmap + 0.6 * img_array).astype(np.uint8)

    if percentile > 0.0:
        return bbox_to_percentile(Image.fromarray(sim_heatmap), sim_map, percentile=percentile, thickness=thickness)
    return Image.fromarray(sim_heatmap)


supported_viewing_functions = {
    "crop_to_percentile": crop_to_percentile,
    "bbox_to_percentile": bbox_to_percentile,
    "heatmap": heatmap,
}
