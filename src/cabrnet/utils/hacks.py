import torch
from torch.optim import Optimizer


def optimizer_to(optim: Optimizer, device: str) -> None:
    """
    Move optimizer to target device. Solution from https://github.com/pytorch/pytorch/issues/8741
    Args:
        optim: Optimizer
        device: Target device
    """
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
