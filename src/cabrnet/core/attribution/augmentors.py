import torch
from torch import Tensor, nn

"""
An augmentor is any torch module that takes a single sample and return a batch of augmented versions.
They are used to improve the stability of feature attribution methods, typically to implement Smoothgrad.

For simplicity and consistency, the input is always supposed to have a batch dimension of one.
"""


class GaussianNoiseAugmentor(nn.Module):
    def __init__(self, num_samples: int, noise_ratio: float) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.noise_ratio = noise_ratio

    def forward(self, input_tensor: Tensor) -> Tensor:
        assert (
            input_tensor.shape[0] == 1
        ), f"GaussianNoiseAugmentor input must be a single image, got batch dimension {input_tensor.shape[0]}"
        noise_std = (input_tensor.max() - input_tensor.min()) * self.noise_ratio
        return torch.cat(
            [
                input_tensor + torch.randn(input_tensor.shape, device=input_tensor.device) * noise_std
                for _ in range(self.num_samples)
            ]
        )
