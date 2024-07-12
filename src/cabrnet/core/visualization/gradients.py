import numpy as np
import torch
import torch.nn as nn
from captum._utils.typing import TargetType
from captum.attr import LRP, Attribution, NoiseTunnel, Saliency
from loguru import logger
from PIL import Image
from torch import Tensor

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


def _captum_saliency_wrapper(model: nn.Module) -> Attribution:
    r"""Wrapper for the Saliency module in Captum.

    Args:
        model (Module): Target module.

    Returns:
        Attribution object.
    """
    return Saliency(model.similarities)


def _captum_saliency_nt_wrapper(model: nn.Module) -> Attribution:
    r"""Wrapper for the Smoothgrad module in Captum.

    Args:
        model (Module): Target module.

    Returns:
        Attribution object.
    """
    return NoiseTunnel(attribution_method=Saliency(model.similarities))


def _captum_attribute(captum_model: Attribution, img_tensor: Tensor, target: TargetType, **kwargs) -> np.ndarray:
    r"""Default Captum attribution.

    Args:
        captum_model (Attribution): Model prepared for Captum.
        img_tensor (tensor): Image tensor.
        target (TargetType): Target in model output (e.g. class index).

    Returns:
        Captum attribution.
    """
    return captum_model.attribute(img_tensor, target=target)[0].detach().cpu().numpy()


def _captum_smoothgrad_attribute(
    captum_model: Attribution,
    img_tensor: Tensor,
    target: TargetType,
    num_samples: int = 10,
    noise_ratio: float = 0.2,
    **kwargs,
) -> np.ndarray:
    r"""Smoothgrad attribution.

    Args:
        captum_model (Attribution): Model prepared for Captum.
        img_tensor (tensor): Image tensor.
        target (TargetType): Target in model output (e.g. class index).
        num_samples (int, optional): Number of random samples. Default: 10.
        noise_ratio (float, optional): Noise ratio for random samples. Default: 0.2.

    Returns:
        Captum attribution.
    """
    # Compute standard deviation dynamically based on noise ratio
    stdev = float(
        (img_tensor.max() - img_tensor.min()).detach().cpu().numpy() * noise_ratio
    )  # Explicit cast into float to prevent Captum from using float64
    return (
        captum_model.attribute(
            img_tensor,
            nt_type="smoothgrad",
            nt_samples=num_samples,
            nt_samples_batch_size=16,  # Maximum batch size
            stdevs=stdev,
            target=target,
        )[0]
        .detach()
        .cpu()
        .numpy()
    )


supported_captum_algorithms = {
    "smoothgrad": {"wrapper": _captum_saliency_nt_wrapper, "attribute": _captum_smoothgrad_attribute},
    "saliency": {"wrapper": _captum_saliency_wrapper, "attribute": _captum_attribute},
    "lrp": {"wrapper": LRP, "attribute": _captum_attribute},
}


def _captum_attribution(
    model: nn.Module,
    algorithm: str,
    img: Image.Image,
    img_tensor: Tensor,
    proto_idx: int,
    device: str | torch.device,
    location: tuple[int, int] | str | None = None,
    polarity: str | None = "absolute",
    gaussian_ksize: int = 5,
    normalize: bool = False,
    grads_x_input: bool = False,
    similarity_threshold: float = 0.1,
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
    img_tensor = _check_tensor_dims(img_tensor)

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
    h_max, w_max = -1, -1
    if location is not None:
        if location == "max":
            # Find location of feature vector with the highest similarity
            h_max, w_max = np.where(sim_map == np.max(sim_map))
            h_max, w_max = h_max[0], w_max[0]
        elif isinstance(location, tuple):
            # Location is predefined
            h_max, w_max = location
        else:
            raise ValueError(f"Invalid target location {location}")
        # Update similarity threshold to ensure that at least one location is used
        similarity_threshold = sim_map[h_max, w_max]

    # Create Captum wrapper
    assert algorithm in supported_captum_algorithms, f"Unsupported attribution method: {algorithm}"
    captum_model = supported_captum_algorithms[algorithm]["wrapper"](model)
    attribution_fn = supported_captum_algorithms[algorithm]["attribute"]

    # Init gradient accumulator
    grads = np.zeros_like(img_tensor[0].detach().cpu().numpy())
    for h in range(sim_map_height):
        if 0 <= h_max != h:
            # Skip location
            continue
        for w in range(sim_map_width):
            if 0 <= w_max != w or sim_map[h, w] < similarity_threshold:
                # Skip location
                continue
            model.zero_grad()
            attribution = attribution_fn(
                captum_model=captum_model, img_tensor=img_tensor, target=(proto_idx, h, w), **kwargs
            )
            if algorithm == "lrp":
                # LRP already weights the attribution map by the output value
                grads += attribution
                # Reattach LRP-Comp rules to underlying model
                attach_lrp_comp_rules(captum_model.model)
            else:
                grads += sim_map[h, w].item() * attribution

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


def smoothgrad(
    model: nn.Module,
    img: Image.Image,
    img_tensor: Tensor,
    proto_idx: int,
    device: str | torch.device,
    location: tuple[int, int] | str | None = None,
    polarity: str | None = "absolute",
    gaussian_ksize: int = 5,
    normalize: bool = False,
    num_samples: int = 10,
    noise_ratio: float = 0.2,
    grads_x_input: bool = False,
    similarity_threshold: float = 0.1,
    **kwargs,
) -> np.ndarray:
    r"""Performs patch visualization using SmoothGrad (https://arxiv.org/abs/1706.03825).

    Args:
        model (Module): Target model.
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
        num_samples (int, optional): Number of random samples. Default: 10.
        noise_ratio (float, optional): Noise ratio for random samples. Default: 0.2.
        grads_x_input (bool, optional): If True, performs element-wise multiplication between gradient and image.
            Default: False.
        similarity_threshold (float, optional): Ignore locations in the similarity map with a score lower than this
            threshold. Default: 0.1.

    Returns:
        Similarity map.
    """
    return _captum_attribution(
        model=model,
        algorithm="smoothgrad",
        img=img,
        img_tensor=img_tensor,
        proto_idx=proto_idx,
        device=device,
        location=location,
        polarity=polarity,
        gaussian_ksize=gaussian_ksize,
        normalize=normalize,
        grads_x_input=grads_x_input,
        num_samples=num_samples,
        noise_ratio=noise_ratio,
        similarity_threshold=similarity_threshold,
    )


def saliency(
    model: nn.Module,
    img: Image.Image,
    img_tensor: Tensor,
    proto_idx: int,
    device: str | torch.device,
    location: tuple[int, int] | str | None = None,
    polarity: str | None = "absolute",
    gaussian_ksize: int = 5,
    normalize: bool = False,
    grads_x_input: bool = False,
    similarity_threshold: float = 0.1,
    **kwargs,
) -> np.ndarray:
    r"""Performs patch visualization using saliency (https://arxiv.org/abs/1312.6034).

    Args:
        model (Module): Target model.
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
    return _captum_attribution(
        model=model,
        algorithm="saliency",
        img=img,
        img_tensor=img_tensor,
        proto_idx=proto_idx,
        device=device,
        location=location,
        polarity=polarity,
        gaussian_ksize=gaussian_ksize,
        normalize=normalize,
        grads_x_input=grads_x_input,
        similarity_threshold=similarity_threshold,
    )


def randgrad(
    img: Image.Image,
    img_tensor: Tensor,
    polarity: str | None = "absolute",
    gaussian_ksize: int = 5,
    normalize: bool = False,
    grads_x_input: bool = False,
    **kwargs,
) -> np.ndarray:
    r"""Returns random patch visualization (used as a baseline for evaluating properties of other retracing functions).

    Args:
        img (Image): Raw input image.
        img_tensor (tensor): Input image tensor.
        polarity (str, optional): Polarity filter (None, "absolute", "positive", or "negative"). Default: absolute.
        gaussian_ksize (int, optional): Size of gaussian filter kernel size. Default: 5.
        normalize (bool, optional): If True, performs min-max normalization. Default: False.
        grads_x_input (bool, optional): If True, performs element-wise multiplication between gradient and image.
            Default: False.

    Returns:
        Similarity map.
    """
    img_tensor = _check_tensor_dims(img_tensor)

    grads = np.random.random(img_tensor[0].shape)
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


def prp(
    model: nn.Module,
    img: Image.Image,
    img_tensor: Tensor,
    proto_idx: int,
    device: str | torch.device,
    location: tuple[int, int] | str | None = None,
    stability_factor: float = 1e-6,
    polarity: str | None = "absolute",
    gaussian_ksize: int = 5,
    normalize: bool = False,
    grads_x_input: bool = False,
    similarity_threshold: float = 0.1,
    **kwargs,
) -> np.ndarray:
    r"""Performs patch visualization using Prototype Relevance Propagation.

    See (https://www.sciencedirect.com/science/article/pii/S0031320322006513#bib0030).

    Args:
        model (Module): Target model.
        img (Image): Raw input image.
        img_tensor (tensor): Input image tensor.
        proto_idx (int): Prototype index.
        device (str | device): Hardware device.
        location (tuple[int,int], str or None, optional): Location inside the similarity map.
                Can be given as an explicit location (tuple) or "max" for the location of maximum similarity.
                Default: None.
        stability_factor (float, optional): LRP stability factor (epsilon). Default: 1e-6.
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
    if not hasattr(model, "lrp_ready"):
        logger.warning(
            "Canonizing model on-the-fly for PRP. For multiple explanations, "
            "consider performing canonization beforehand."
        )
        model = get_cabrnet_lrp_composite_model(
            model=model, set_bias_to_zero=True, stability_factor=stability_factor, use_zbeta=True
        )

    return _captum_attribution(
        model=model,
        algorithm="lrp",
        img=img,
        img_tensor=img_tensor,
        proto_idx=proto_idx,
        device=device,
        location=location,
        polarity=polarity,
        gaussian_ksize=gaussian_ksize,
        normalize=normalize,
        grads_x_input=grads_x_input,
        stability_factor=stability_factor,
        similarity_threshold=similarity_threshold,
    )
