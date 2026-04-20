import numpy as np
import torch
import torch.nn as nn
from captum.attr import LRP, Saliency
from loguru import logger
from PIL import Image
from torch import Tensor

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.visualization.postprocess import post_process
from cabrnet.core.visualization.prp_utils import (
    attach_lrp_comp_rules,
    get_cabrnet_lrp_composite_model,
)


def _check_tensor_dims(x: Tensor) -> Tensor:
    r"""Checks and extends (if necessary) number of dimensions to 4.

    Args:
        x (tensor): Input tensor.

    Returns:
        Modified tensor (if necessary).
    """
    if x.dim() not in [3, 4]:
        raise ValueError(f"Unsupported number of dimensions in tensor. Expected 3 or 4, got {x.dim()}")
    if x.dim() == 3:
        # Fix number of dimensions
        x = torch.unsqueeze(x, dim=0)
    elif x.size(0) != 1:
        raise ValueError(f"Gradient operations only support single images. Received batch of size {x.size(0)}")
    return x


def apply_augmentors(input: Tensor, augmentors: list[nn.Module]) -> Tensor:
    result = input.unsqueeze(0)
    for augment_module in augmentors:
        result = torch.cat([augment_module(x) for x in result])
    return result


def attribute_prototypes(
    model: CaBRNet,
    algorithm: str,
    img: Image.Image,
    img_tensor: Tensor,
    proto_idx: int,
    device: str | torch.device,
    augmentors: list[nn.Module] = [],
    post_augmentation_transform: nn.Module = nn.Identity(),
    location: tuple[int, int] | str | None = None,
    polarity: str | None = "absolute",
    gaussian_ksize: int = 5,
    normalize: bool = False,
    grads_x_input: bool = False,
    similarity_threshold: float | None = None,
    stability_factor: float = 1e-6,
    **kwargs,
) -> np.ndarray:
    r"""Computes attributions using a post-hoc explanation method from Captum.

    Args:
        model (Module): Target model.
        algorithm (str): Name of the attribution method.
        img (Image): Raw input image.
        img_tensor (tensor): Input image tensor.
        proto_idx (int): Prototype index.
        device (str | device): Hardware device.
        location (tuple[int,int], str or None, optional): Location inside the similarity map.
                Can be given as an explicit location (tuple) or "max" for the location of maximum similarity.
                Default: None.
        polarity (str, optional): Polarity filter (None, "absolute", "positive", or "negative"). Default: absolute.
        gaussian_ksize (int, optional): Size of gaussian filter kernel size. Default: 5.
        normalize (bool, optional): If True, performs min-max normalization. Default: False.
        grads_x_input (bool, optional): If True, performs element-wise multiplication between gradient and image.
            Default: False.
        similarity_threshold (float, optional): Ignore locations in the similarity map with a score lower than this
            threshold. Default: 0.1.

    Returns:
        Similarity map.
    """
    if similarity_threshold is not None and location is not None:
        raise ValueError(
            f"Both location={location} and similatiy_threshold={similarity_threshold} were provided for feature attribution, aborting"
        )
    if similarity_threshold is None and location is None:
        similarity_threshold = 0.1

    img_tensor = _check_tensor_dims(img_tensor)

    if algorithm == "lrp":
        if not hasattr(model, "lrp_ready"):
            logger.warning(
                "Canonizing model on-the-fly for PRP. For multiple explanations, "
                "consider performing canonization beforehand."
            )
            model = get_cabrnet_lrp_composite_model(
                model=model, set_bias_to_zero=True, stability_factor=stability_factor, use_zbeta=True
            )

    # Map model to device
    model.eval()
    model.to(device)

    # Map to device
    img_tensor = img_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        # Compute similarity map
        sim_map = model.similarities(img_tensor.to(device))[0, proto_idx].cpu().numpy()
        sim_map_height, sim_map_width = sim_map.shape[0], sim_map.shape[1]

    # Location of interest (if any)
    if location is None:
        positions_to_consider = [
            (h, w) for h in range(sim_map_height) for w in range(sim_map_width) if sim_map[h, w] > similarity_threshold
        ]
    else:
        if location == "max":
            # Find location of feature vector with the highest similarity
            h_max, w_max = np.where(sim_map == np.max(sim_map))
            positions_to_consider = [(h_max[0], w_max[0])]
        elif isinstance(location, tuple):
            # Location is predefined
            positions_to_consider = [location]
        else:
            raise ValueError(f"Invalid target location {location}")

    if algorithm == "saliency":
        attributor = Saliency(model)
    elif algorithm == "lrp":
        attributor = LRP(model)
    else:
        raise ValueError(f"Unsupported attribution method: {algorithm}")

    augmented_imgs = apply_augmentors(img_tensor, augmentors)
    attribution_inputs = post_augmentation_transform(augmented_imgs)

    # Init gradient accumulator
    grads = np.zeros_like(img_tensor[0].detach().cpu().numpy())
    for h, w in positions_to_consider:
        model.zero_grad()

        if algorithm == "lrp":
            # LRP already weights the attribution map by the output value
            weight = 1
        else:
            weight = sim_map[h, w].item()

        attributions = torch.cat([attributor.attribute(x, target=(proto_idx, h, w)) for x in attribution_inputs])
        grads += weight * attributions.mean(0)  # average all the attributions by default: smoothgrad-like behaviour

        if algorithm == "lrp":
            # Reattach LRP-Comp rules to underlying model
            attach_lrp_comp_rules(model)

    return post_process(
        array=grads,
        img=img,
        img_tensor=img_tensor,
        resize=True,
        polarity=polarity,
        gaussian_ksize=gaussian_ksize,
        normalize=normalize,
        grads_x_input=grads_x_input,
    )
