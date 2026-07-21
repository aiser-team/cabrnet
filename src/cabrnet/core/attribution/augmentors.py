import torch
from torch import Tensor, nn

"""
An augmentor is any torch module that takes a single sample and return a batch of augmented versions.
They are used to improve the stability of feature attribution methods, typically to implement Smoothgrad.

For simplicity and consistency, the input is always supposed to have a batch dimension of one.
"""


class GaussianNoiseAugmentor(nn.Module):
    r"""Creates augmented tensor views by adding Gaussian noise.

    The noise ratio is a proportion of the signal peak-to-peak amplitude.

    Attributes:
        num_samples: Number of augmented views to create.
        noise_ratio: Noise standard-deviation ratio relative to the input amplitude.
    """

    num_samples: int
    noise_ratio: float

    def __init__(self, num_samples: int, noise_ratio: float) -> None:
        r"""Initializes a Gaussian-noise augmentor.

        Args:
            num_samples (int): Number of augmented views to create.
            noise_ratio (float): Noise standard-deviation ratio relative to the input amplitude.
        """
        super().__init__()
        self.num_samples = num_samples
        self.noise_ratio = noise_ratio

    def forward(self, input_tensor: Tensor) -> Tensor:
        r"""Generates Gaussian-noise augmented views of an input tensor.

        Args:
            input_tensor (tensor): Single-sample input tensor with a batch dimension of one.

        Returns:
            Batch of augmented tensor views.
        """
        assert input_tensor.shape[0] == 1, (
            f"GaussianNoiseAugmentor input must be a single image, got batch dimension {input_tensor.shape[0]}"
        )
        noise_std = (input_tensor.max() - input_tensor.min()) * self.noise_ratio
        return torch.cat(
            [
                input_tensor + torch.randn(input_tensor.shape, device=input_tensor.device) * noise_std
                for _ in range(self.num_samples)
            ]
        )
