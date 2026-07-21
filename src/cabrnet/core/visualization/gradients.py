from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from captum.attr import LRP, IntegratedGradients, Saliency
from captum.attr._utils.attribution import GradientAttribution
from loguru import logger
from torch import Tensor

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.visualization.prp_utils import (
    attach_lrp_comp_rules,
    get_cabrnet_lrp_composite_model,
)


class RandGrad(GradientAttribution):
    """Random gradient attribution method.

    A placeholder class for future implementation.
    """

    def attribute(self, inputs: Tensor, *args, **kwargs) -> Tensor:
        """Return random attributions (baseli).

        Args:
            inputs: Input tensor.
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Returns:
            Random tensor with the same shape as inputs.
        """
        return torch.randn(inputs.shape)


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


def apply_augmentors(input_tensor: Tensor, augmentors: Sequence[nn.Module]) -> Tensor:
    """Apply a sequence of augmentors to an input tensor.

    Each augmentor takes a single sample and returns a batch of augmented versions.
    The augmentors are applied sequentially, with each augmentor processing all
    samples produced by the previous one.

    Args:
        input_tensor: Input tensor to augment.
        augmentors: List of augmentor modules to apply sequentially.

    Returns:
        Tensor containing all augmented samples.
    """
    assert input_tensor.shape[0] == 1
    result = input_tensor
    for augment_module in augmentors:
        # invariant: always has a batch dimension
        result = torch.cat([augment_module(x.unsqueeze(0)) for x in result])
    return result


def _resolve_positions(
    sim_map: np.ndarray,
    location: tuple[int, int] | str | None,
    similarity_threshold: float,
) -> list[tuple[int, int]]:
    if location is None:
        h_idx, w_idx = np.where(sim_map > similarity_threshold)
        return list(zip(h_idx.tolist(), w_idx.tolist()))
    if location == "max":
        h, w = np.unravel_index(np.argmax(sim_map), sim_map.shape)
        return [(int(h), int(w))]
    if isinstance(location, tuple):
        return [location]
    raise ValueError(f"Invalid target location: {location!r}")


def _ensure_lrp_ready(model: CaBRNet, stability_factor: float) -> CaBRNet:
    if not hasattr(model, "lrp_ready"):
        logger.warning(
            "Canonizing model on-the-fly for PRP. For multiple explanations, "
            "consider performing canonization beforehand."
        )
        return get_cabrnet_lrp_composite_model(
            model=model,
            set_bias_to_zero=True,
            stability_factor=stability_factor,
            use_zbeta=True,
        )
    return model


def attribute_prototypes(
    model: CaBRNet,
    algorithm: Literal["saliency", "prp", "randgrad", "ig"],
    input_tensor: Tensor,
    proto_idx: int,
    device: str | torch.device,
    augmentors: Sequence[nn.Module] = [],
    post_augmentation_transform: nn.Module = nn.Identity(),
    location: tuple[int, int] | str | None = None,
    similarity_threshold: float = 0.1,
    stability_factor: float = 1e-6,
    **kwargs,
) -> np.ndarray:
    input_tensor = (input_tensor).unsqueeze(0)

    if algorithm == "prp":
        model = _ensure_lrp_ready(model, stability_factor)

    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)
    input_tensor_transformed = post_augmentation_transform(input_tensor)

    with torch.no_grad():
        sim_map = model.similarities(input_tensor_transformed)[0, proto_idx].cpu().numpy()

    positions = _resolve_positions(sim_map, location, similarity_threshold)

    # Build attributor — each algorithm wraps a different target function/model
    if algorithm == "saliency":
        attributor = Saliency(model.similarities)
    elif algorithm == "ig":
        attributor = IntegratedGradients(model.similarities)
    elif algorithm == "prp":
        attributor = LRP(model)
    else:  # randgrad
        attributor = RandGrad(model.similarities)

    augmented_imgs = apply_augmentors(input_tensor, augmentors)
    attribution_inputs = post_augmentation_transform(augmented_imgs)

    # PRP (LRP) already scales by output value internally; other methods weight by similarity score
    weights = [1.0 if algorithm == "prp" else sim_map[h, w].item() for h, w in positions]

    grads = np.zeros_like(input_tensor_transformed[0].detach().cpu().numpy())
    for (h, w), weight in zip(positions, weights):
        model.zero_grad()
        attributions = torch.stack(
            [attributor.attribute(x.unsqueeze(0), target=(proto_idx, h, w)).squeeze(0) for x in attribution_inputs]
        ).mean(0)
        grads += weight * attributions.detach().cpu().numpy()
        if algorithm == "prp":
            attach_lrp_comp_rules(model)

    return grads
