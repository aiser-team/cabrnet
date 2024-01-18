import torch
import torch.nn as nn
from torch.nn import functional as F
from abc import ABC, abstractmethod
from typing import Any

from captum.attr._utils.lrp_rules import PropagationRule


class GradientRule(PropagationRule):
    """Dummy propagation rule used when ignoring LRP and using gradients directly"""

    def forward_hook(self, module, inputs, outputs):
        pass

    def forward_hook_weights(self, module, inputs, outputs):
        pass

    def _manipulate_weights(self, module, inputs, outputs):
        pass

    def forward_pre_hook_activations(self, module, inputs):
        pass


class ZBetaLayer(ABC):
    """Abstract layer for applying Z^Beta rule during relevance backpropagation.
    For more information, see https://arxiv.org/pdf/1512.02479.pdf

    Use for first (pixel) layer.

    Args:
        set_bias_to_zero (bool): ignore bias during backpropagation
        lower_bound (float): smallest admissible pixel value
        upper_bound (float): largest admissible pixel value
        stability_factor (float): epsilon value used for numerical stability

    Note: admissible range for pixel values must take into account input normalization
    """

    def __init__(
        self,
        set_bias_to_zero: bool = True,
        lower_bound: float = min([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]),  # from imagenet normalization
        upper_bound: float = max(
            [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]
        ),  # from imagenet normalization
        stability_factor: float = 1e-6,
    ) -> None:
        self.set_bias_to_zero = set_bias_to_zero
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.stability_factor = stability_factor
        self.autograd_func = self._autograd_func()
        self.rule = GradientRule()

    @abstractmethod
    def _legacy_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy forward function for this layer"""
        raise NotImplementedError

    @abstractmethod
    def _modified_forward(self, x: torch.Tensor, lower_bound: torch.Tensor, upper_bound: torch.Tensor) -> torch.Tensor:
        """Modified forward function"""
        raise NotImplementedError

    def _autograd_func(self) -> torch.autograd.Function:
        class ZBetaAutoGradFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, *args, **kwargs) -> torch.Tensor:
                ctx.save_for_backward(*args)
                # Keep original inference
                return self._legacy_forward(*args)

            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                x = x.clone().detach().requires_grad_(True)
                lower_bound_tensor = (self.lower_bound * torch.ones_like(x)).requires_grad_(True)
                upper_bound_tensor = (self.upper_bound * torch.ones_like(x)).requires_grad_(True)
                with torch.enable_grad():
                    # Forward pass with data-independent bounds corresponding to range of admissible pixel values
                    output = self._modified_forward(x, lower_bound_tensor, upper_bound_tensor)
                    output_detached = output.clone().detach()
                    normalized = grad_output[0].detach() / (
                        output_detached + self.stability_factor * output_detached.sign()
                    )
                    # Backward pass
                    output.backward(normalized)
                grads = (
                    x * x.grad
                    + lower_bound_tensor * lower_bound_tensor.grad
                    + upper_bound_tensor * upper_bound_tensor.grad
                )
                return grads

        return ZBetaAutoGradFunc()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.autograd_func.apply(x)


class ZBetaLinear(ZBetaLayer, nn.Linear):  # ZBetaLayer takes precedence to supersede "forward" function from nn.Module
    """Replacement layer for applying Z^Beta rule on a nn.Linear operator during relevance backpropagation.
    For more information, see https://arxiv.org/pdf/1512.02479.pdf

    Use for first (pixel) layer.

    Args:
        module (nn.Module): layer to be replaced
        set_bias_to_zero (bool): ignore bias during backpropagation
        lower_bound (float): smallest admissible pixel value
        upper_bound (float): largest admissible pixel value
        stability_factor (float): epsilon value used for numerical stability

    Note: admissible range for pixel values must take into account input normalization
    """

    def __init__(
        self,
        module: nn.Linear,
        set_bias_to_zero: bool = True,
        lower_bound: float = min([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]),  # from imagenet normalization
        upper_bound: float = max(
            [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]
        ),  # from imagenet normalization
        stability_factor: float = 1e-6,
    ) -> None:
        # Init ZBetaLayer
        ZBetaLayer.__init__(
            self,
            set_bias_to_zero=set_bias_to_zero,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            stability_factor=stability_factor,
        )

        # Copy source module configuration and parameters
        nn.Linear.__init__(
            self, in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None
        )
        self.load_state_dict(module.state_dict())

        # Make two additional copies of the module weights, taking signs into account
        self.positive_w = module.weight.data.clamp(min=0)
        self.negative_w = module.weight.data.clamp(max=0)
        self.positive_b = None if module.bias is None else module.bias.data.clamp(min=0)
        self.negative_b = None if module.bias is None else module.bias.data.clamp(max=0)

    def _legacy_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy forward function for this layer"""
        return F.linear(x, self.weight, self.bias)

    def _modified_forward(
        self, x: torch.Tensor, l_bound_tensor: torch.Tensor, u_bound_tensor: torch.Tensor
    ) -> torch.Tensor:
        return (
            F.linear(x, self.weight, self.bias if not self.set_bias_to_zero else None)
            - F.linear(l_bound_tensor, self.positive_w, self.positive_b if not self.set_bias_to_zero else None)
            - F.linear(u_bound_tensor, self.negative_w, self.negative_b if not self.set_bias_to_zero else None)
        )


class ZBetaConv2d(ZBetaLayer, nn.Conv2d):  # ZBetaLayer takes precedence to supersede "forward" function from nn.Module
    """Replacement layer for applying Z^Beta rule on a nn.Conv2d operator during relevance backpropagation.
    For more information, see https://arxiv.org/pdf/1512.02479.pdf

    Use for first (pixel) layer.

    Args:
        module (nn.Module): layer to be replaced
        set_bias_to_zero (bool): ignore bias during backpropagation
        lower_bound (float): smallest admissible pixel value
        upper_bound (float): largest admissible pixel value
        stability_factor (float): epsilon value used for numerical stability

    Note: admissible range for pixel values must take into account input normalization
    """

    def __init__(
        self,
        module: nn.Conv2d,
        set_bias_to_zero: bool = True,
        lower_bound: float = min([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]),  # from imagenet normalization
        upper_bound: float = max(
            [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]
        ),  # from imagenet normalization
        stability_factor: float = 1e-6,
    ) -> None:
        # Init ZBetaLayer
        ZBetaLayer.__init__(
            self,
            set_bias_to_zero=set_bias_to_zero,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            stability_factor=stability_factor,
        )

        # Copy source module configuration and parameters first
        nn.Conv2d.__init__(
            self,
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,  # type: ignore
            stride=module.stride,  # type: ignore
            padding=module.padding,
            dilation=module.dilation,  # type: ignore
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        )
        self.load_state_dict(module.state_dict())

        # Make two additional copies of the module weights, taking signs into account
        self.positive_w = module.weight.data.clamp(min=0)
        self.negative_w = module.weight.data.clamp(max=0)
        self.positive_b = None if module.bias is None else module.bias.data.clamp(min=0)
        self.negative_b = None if module.bias is None else module.bias.data.clamp(max=0)

    def _legacy_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy forward function for this layer"""
        return self._conv_forward(x, self.weight, self.bias)

    def _modified_forward(
        self, x: torch.Tensor, l_bound_tensor: torch.Tensor, u_bound_tensor: torch.Tensor
    ) -> torch.Tensor:
        return (
            self._conv_forward(x, self.weight, self.bias if not self.set_bias_to_zero else None)
            - self._conv_forward(
                l_bound_tensor, self.positive_w, self.positive_b if not self.set_bias_to_zero else None
            )
            - self._conv_forward(
                u_bound_tensor, self.negative_w, self.negative_b if not self.set_bias_to_zero else None
            )
        )
