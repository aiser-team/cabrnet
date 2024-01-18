from torch import Tensor
import torch

prototype_init_modes = ["NORMAL", "SHIFTED_NORMAL"]


def init_prototypes(
    num_prototypes: int,
    num_features: int,
    init_mode: str = "SHIFTED_NORMAL",
) -> Tensor:
    """
    Create tensor of prototypes
    Args:
        num_prototypes: Number of prototypes
        num_features: Size of each prototype
        init_mode: Initialisation mode (default: SHIFTED_NORMAL = N(0.5, 0.1))
    """
    if init_mode not in prototype_init_modes:
        raise ValueError(f"Unknown prototype initialisation mode {init_mode}.")
    prototypes = torch.randn((num_prototypes, num_features, 1, 1), requires_grad=True)
    if init_mode == "SHIFTED_NORMAL":
        torch.nn.init.normal_(prototypes, mean=0.5, std=0.1)
    return prototypes
