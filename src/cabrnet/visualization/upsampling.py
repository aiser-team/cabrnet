import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from cabrnet.visualization.postprocess import normalize_min_max


def cubic_upsampling(
    model: nn.Module,
    img: Image.Image,
    img_tensor: Tensor,
    proto_idx: int,
    device: str,
    location: tuple[int, int] | None = None,
    single_location: bool = True,
    normalize: bool = False,
) -> np.array:
    """
    Perform patch visualization using cubic interpolation
    Args:
        model: target model
        img: raw input image
        img_tensor: input image tensor
        proto_idx: prototype index
        device: target hardware device
        location: coordinates of feature vector
        single_location: keep only a single location
        normalize: perform min-max normalization

    Returns:
        upsampled similarity map
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

    if single_location:
        if location is None:
            # Find location of feature vector with the highest similarity
            h, w = np.where(sim_map == np.max(sim_map))
            h, w = h[0], w[0]
        else:
            # Location is predefined
            h, w = location
        sim_map = np.zeros_like(sim_map)
        sim_map[h, w] = 1

    # Upsample to image size
    sim_map = cv2.resize(src=sim_map, dsize=(img.width, img.height), interpolation=cv2.INTER_CUBIC)
    if normalize:
        sim_map = normalize_min_max(sim_map)
    return sim_map
