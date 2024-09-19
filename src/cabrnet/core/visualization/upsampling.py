import cv2
import numpy as np
import torch
import torch.nn as nn
from cabrnet.core.visualization.postprocess import normalize_min_max
from PIL import Image
from torch import Tensor


def cubic_upsampling(
    model: nn.Module,
    img: Image.Image,
    img_tensor: Tensor,
    proto_idx: int,
    device: str | torch.device,
    location: tuple[int, int] | str | None = None,
    normalize: bool = False,
    **kwargs,
) -> np.ndarray:
    r"""Performs patch visualization using upsampling with cubic interpolation.

    Args:
        model (Module): Target model.
        img (Image): Raw input image.
        img_tensor (tensor): Input image tensor.
        proto_idx (int): Prototype index.
        device (str | device): Hardware device.
        location (tuple[int,int], str or None, optional): Location inside the similarity map.
                Can be given as an explicit location (tuple) or "max" for the location of maximum similarity.
                Default: None.
        normalize (bool, optional): If True, performs min-max normalization. Default: False.

    Returns:
        Upsampled similarity map.
    """
    if img_tensor.dim() != 4:
        # Fix number of dimensions if necessary
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # Map to device
    img_tensor = img_tensor.to(device)
    model.eval()
    model.to(device)

    with torch.no_grad():
        # Compute similarity map
        sim_map = model.similarities(img_tensor)[0, proto_idx].cpu().numpy()

    # Location of interest (if any)
    if location is not None:
        if location == "max":
            # Find location of feature vector with the highest similarity
            h, w = np.where(sim_map == np.max(sim_map))
            h, w = h[0], w[0]
        elif isinstance(location, tuple):
            # Location is predefined
            h, w = location
        else:
            raise ValueError(f"Invalid target location {location}")

        sim_map = np.zeros_like(sim_map)
        sim_map[h, w] = 1

    # Upsample to image size
    sim_map = cv2.resize(src=sim_map, dsize=(img.width, img.height), interpolation=cv2.INTER_CUBIC)
    if normalize:
        sim_map = normalize_min_max(sim_map)
    return sim_map
