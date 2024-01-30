import torch
from torch import Tensor
from PIL import Image
import torch.nn as nn
import numpy as np
from loguru import logger
from cabrnet.visualisation.postprocess import post_process
from cabrnet.visualisation.prp_utils import get_cabrnet_lrp_composite_model, attach_lrp_comp_rules
from captum.attr import LRP


def _check_tensor_dims(x: Tensor) -> Tensor:
    if x.dim() not in [3, 4]:
        raise ValueError(f"Unsupported number of dimensions in tensor. Expected 3 or 4, got {x.dim()}")
    if x.dim() == 3:
        # Fix number of dimensions
        x = torch.unsqueeze(x, dim=0)
    elif x.size(0) != 1:
        raise ValueError(f"Gradient operations only support single images. Received batch of size {x.size(0)}")
    return x


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
    img_tensor = _check_tensor_dims(img_tensor)

    # Map model to device
    model.eval()
    model.to(device)

    # Location of interest (if any)
    h_max, w_max = -1, -1
    if single_location:
        if location is None:
            with torch.no_grad():
                # Compute similarity map
                sim_map = model.similarities(img_tensor.to(device))[0, proto_idx].cpu().numpy()
            # Find location of feature vector with the highest similarity
            h_max, w_max = np.where(sim_map == np.max(sim_map))
            h_max, w_max = h_max[0], w_max[0]
        else:
            # Location is predefined
            h_max, w_max = location

    if num_samples == 1:
        noisy_images = [img_tensor]
    else:
        # Compute variance from noise ratio
        sigma = (img_tensor.max() - img_tensor.min()).detach().cpu().numpy() * noise_ratio
        # Generate noisy images around original.
        noisy_images = [img_tensor + torch.randn(img_tensor.shape) * sigma for _ in range(num_samples)]

    # Compute gradients
    grads = []
    for noisy_image in noisy_images:
        # Init gradient accumulator
        weighted_grad_sum = np.zeros_like(noisy_image[0])
        # Map to device
        noisy_image = noisy_image.to(device)

        # Forward pass
        noisy_image.requires_grad_()
        sim_map = model.similarities(noisy_image)[0, proto_idx]
        sim_map_height, sim_map_width = sim_map.size(0), sim_map.size(1)

        for h in range(sim_map_height):
            if 0 <= h_max != h:
                # Skip location
                continue
            for w in range(sim_map_width):
                if 0 <= w_max != w:
                    # Skip location
                    continue
                output = sim_map[h, w]
                output.backward(retain_graph=True)
                # Weight gradients from location (h,w) by corresponding similarity score
                weighted_grad_sum += sim_map[h, w].item() * noisy_image.grad.data[0].detach().cpu().numpy()
        grads.append(weighted_grad_sum)

    # grads has shape (num_samples) x img_tensor.shape => average across all samples
    grads = np.mean(np.array(grads), axis=0)
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
    img_tensor = _check_tensor_dims(img_tensor)

    # Map model to device
    model.eval()
    model.to(device)

    # Map to device
    img_tensor = img_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        # Compute similarity map
        sim_map = model(img_tensor.to(device))[0, proto_idx].cpu().numpy()
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
    lrp_model = LRP(model)

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
            # LRP already weights the attribution map by the output value
            grads += lrp_model.attribute(img_tensor, target=(proto_idx, h, w))[0].detach().cpu().numpy()
            # Reattach LRP-Comp rules to underlying model
            attach_lrp_comp_rules(lrp_model.model)
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
