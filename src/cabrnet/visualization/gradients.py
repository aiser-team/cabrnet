import torch
from torch import Tensor
from PIL import Image
import torch.nn as nn
import numpy as np
from loguru import logger
from cabrnet.visualization.postprocess import post_process
from cabrnet.visualization.prp_utils import get_cabrnet_lrp_composite_model, attach_lrp_comp_rules
from captum.attr import LRP, NoiseTunnel, Saliency, Attribution
from captum._utils.typing import TargetType


def _check_tensor_dims(x: Tensor) -> Tensor:
    if x.dim() not in [3, 4]:
        raise ValueError(f"Unsupported number of dimensions in tensor. Expected 3 or 4, got {x.dim()}")
    if x.dim() == 3:
        # Fix number of dimensions
        x = torch.unsqueeze(x, dim=0)
    elif x.size(0) != 1:
        raise ValueError(f"Gradient operations only support single images. Received batch of size {x.size(0)}")
    return x


def _captum_saliency_wrapper(model: nn.Module) -> Attribution:
    return Saliency(model.similarities)


def _captum_saliency_nt_wrapper(model: nn.Module) -> Attribution:
    return NoiseTunnel(attribution_method=Saliency(model.similarities))


def _captum_attribute(captum_model: Attribution, img_tensor: Tensor, target=TargetType, **kwargs) -> np.ndarray:
    """Default Captum attribution"""
    return captum_model.attribute(img_tensor, target=target)[0].detach().cpu().numpy()


def _captum_smoothgrad_attribute(
    captum_model: Attribution,
    img_tensor: Tensor,
    target=TargetType,
    num_samples: int = 10,
    noise_ratio: float = 0.2,
    **kwargs,
) -> np.ndarray:
    """Smoothgrad attribution"""
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
    device: str,
    location: tuple[int, int] | None = None,
    single_location: bool = True,
    polarity: str | None = "absolute",
    gaussian_ksize: int = 5,
    normalize: bool = False,
    grads_x_input: bool = False,
    **kwargs,
) -> np.array:
    """
    Compute attributions using a post-hoc explanation method from Captum
    Args:
        model: target model
        algorithm: name of the explanation method
        img: raw input image
        img_tensor: input image tensor
        proto_idx: prototype index
        device: target hardware device
        location: coordinates of feature vector
        single_location: keep only a single location
        polarity: polarity filter (either None, "absolute", "positive", or "negative")
        gaussian_ksize: size of gaussian filter kernel size
        normalize: perform min-max normalization
        num_samples: number of random samples
        noise_ratio: noise ratio for random samples
        grads_x_input: perform element-wise multiplication between gradient and image

    Returns:
        similarity map
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
    if single_location:
        if location is None:
            # Find location of feature vector with the highest similarity
            h_max, w_max = np.where(sim_map == np.max(sim_map))
            h_max, w_max = h_max[0], w_max[0]
        else:
            # Location is predefined
            h_max, w_max = location

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
            if 0 <= w_max != w:
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
    device: str,
    location: tuple[int, int] | None = None,
    single_location: bool = True,
    polarity: str | None = "absolute",
    gaussian_ksize: int = 5,
    normalize: bool = False,
    num_samples: int = 10,
    noise_ratio: float = 0.2,
    grads_x_input: bool = False,
) -> np.array:
    """
    Perform patch visualization using SmoothGrad (https://arxiv.org/abs/1706.03825)
    Args:
        model: target model
        img: raw input image
        img_tensor: input image tensor
        proto_idx: prototype index
        device: target hardware device
        location: coordinates of feature vector
        single_location: keep only a single location
        polarity: polarity filter (either None, "absolute", "positive", or "negative")
        gaussian_ksize: size of gaussian filter kernel size
        normalize: perform min-max normalization
        num_samples: number of random samples
        noise_ratio: noise ratio for random samples
        grads_x_input: perform element-wise multiplication between gradient and image

    Returns:
        similarity map
    """
    return _captum_attribution(
        model=model,
        algorithm="smoothgrad",
        img=img,
        img_tensor=img_tensor,
        proto_idx=proto_idx,
        device=device,
        location=location,
        single_location=single_location,
        polarity=polarity,
        gaussian_ksize=gaussian_ksize,
        normalize=normalize,
        grads_x_input=grads_x_input,
        num_samples=num_samples,
        noise_ratio=noise_ratio,
    )


def saliency(
    model: nn.Module,
    img: Image.Image,
    img_tensor: Tensor,
    proto_idx: int,
    device: str,
    location: tuple[int, int] | None = None,
    single_location: bool = True,
    polarity: str | None = "absolute",
    gaussian_ksize: int = 5,
    normalize: bool = False,
    grads_x_input: bool = False,
) -> np.array:
    """
    Perform patch visualization using saliency (https://arxiv.org/abs/1312.6034)
    Args:
        model: target model
        img: raw input image
        img_tensor: input image tensor
        proto_idx: prototype index
        device: target hardware device
        location: coordinates of feature vector
        single_location: keep only a single location
        polarity: polarity filter (either None, "absolute", "positive", or "negative")
        gaussian_ksize: size of gaussian filter kernel size
        normalize: perform min-max normalization
        grads_x_input: perform element-wise multiplication between gradient and image

    Returns:
        similarity map
    """
    return _captum_attribution(
        model=model,
        algorithm="saliency",
        img=img,
        img_tensor=img_tensor,
        proto_idx=proto_idx,
        device=device,
        location=location,
        single_location=single_location,
        polarity=polarity,
        gaussian_ksize=gaussian_ksize,
        normalize=normalize,
        grads_x_input=grads_x_input,
    )


def randgrad(
    model: nn.Module,
    img: Image.Image,
    img_tensor: Tensor,
    proto_idx: int,
    device: str,
    location: tuple[int, int] | None = None,
    polarity: str | None = "absolute",
    gaussian_ksize: int = 5,
    normalize: bool = False,
    grads_x_input: bool = False,
) -> np.array:
    """
    Return random patch visualization (used as a baseline for evaluating properties of other retracing functions)
    Args:
        model: target model
        img: raw input image
        img_tensor: input image tensor
        proto_idx: prototype index
        device: target hardware device
        location: coordinates of feature vector
        polarity: polarity filter (either None, "absolute", "positive", or "negative")
        gaussian_ksize: size of gaussian filter kernel size
        normalize: perform min-max normalization
        grads_x_input: perform element-wise multiplication between gradient and image

    Returns:
        similarity map
    """
    img_tensor = _check_tensor_dims(img_tensor)

    grads = np.random.random(img_tensor.shape)
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
    device: str,
    location: tuple[int, int] | None = None,
    single_location: bool = True,
    stability_factor: float = 1e-6,
    polarity: str | None = "absolute",
    gaussian_ksize: int = 5,
    normalize: bool = False,
    grads_x_input: bool = False,
) -> np.array:
    """Perform patch visualization using Prototype Relevance Propagation
        (https://www.sciencedirect.com/science/article/pii/S0031320322006513#bib0030)
    Args:
        model: target model
        img: raw input image
        img_tensor: input image tensor
        proto_idx: prototype index
        device: target hardware device
        location: coordinates of feature vector
        single_location: keep only a single location
        stability_factor: LRP stability factor (epsilon)
        polarity: polarity filter (either None, "absolute", "positive", or "negative")
        gaussian_ksize: size of gaussian filter kernel size
        normalize: perform min-max normalization
        grads_x_input: perform element-wise multiplication between gradient and image
    Returns:
        similarity map
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
        single_location=single_location,
        polarity=polarity,
        gaussian_ksize=gaussian_ksize,
        normalize=normalize,
        grads_x_input=grads_x_input,
        stability_factor=stability_factor,
    )
