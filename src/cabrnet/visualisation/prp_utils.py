import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.fx import symbolic_trace
from loguru import logger
import copy
import operator
from abc import ABC, abstractmethod
from typing import Any
from torch import Tensor
from cabrnet.utils.similarities import L2Similarities
from captum.attr._utils.lrp_rules import PropagationRule, IdentityRule


class DecisionLRPWrapper(nn.Module):
    """Replacement for the decision layer of a ProtoClassifier

    Args:
        classifier (nn.Module): source decision layer
        stability_factor (float): epsilon value used for numerical stability
    """

    def __init__(self, classifier: nn.Module, stability_factor: float = 1e-12) -> None:
        super().__init__()
        for attr in ["prototypes", "num_prototypes", "num_features"]:
            if not hasattr(classifier, attr):
                raise ValueError(
                    f"Invalid source module when building {self.__class__.__name__}: " f"Missing attribute {attr}"
                )
        self.prototypes = copy.deepcopy(classifier.prototypes)
        self.similarity_layer = L2Similarities(
            # Do not used classifier.num_prototypes as it might have changed after pruning
            num_prototypes=self.prototypes.size(0),
            num_features=classifier.num_features,
        )

        self.stability_factor = stability_factor
        self.rule = GradientRule()
        self.autograd_func = self._autograd_func()

    def _autograd_func(self) -> torch.autograd.Function:
        class SimilarityAutoGradFunc(torch.autograd.Function):
            def forward(ctx: Any, features: Tensor) -> Tensor:
                # Compute raw L2 distances
                distances = self.similarity_layer.L2_square_distance(features=features, prototypes=self.prototypes)
                # Use ReLU to filter out negative values coming from approximations
                distances = F.relu(distances)
                # Converts to similarity scores following ProtoPNet method
                # i.e. sim = log( (distance+1)/(distance + epsilon))
                similarities = torch.log((distances + 1) / (distances + 1e-4))
                ctx.save_for_backward(features)
                return similarities

            def backward(ctx, grad_output):
                features = ctx.saved_tensors
                i = features.shape[2]
                j = features.shape[3]
                c = features.shape[1]
                p = self.prototypes.shape[0]
                ## Broadcast conv to Nxsize(conv) (No. of prototypes)
                conv = features.repeat(p, 1, 1, 1)
                prototype = self.prototypes.repeat(1, 1, i, j)
                conv = conv.squeeze()

                l2 = (conv - prototype) ** 2
                print(l2.shape)
                d = 1 / (l2**2 + 1e-12)
                denom = torch.sum(d, dim=1, keepdim=True) + 1e-12
                denom = denom.repeat(1, c, 1, 1) + 1e-12
                R = torch.div(d, denom)
                grad_output = grad_output.repeat(c, 1, 1, 1)
                grad_output = grad_output.permute(1, 0, 2, 3)

                R = R * grad_output
                R = torch.sum(R, dim=0)
                R = torch.unsqueeze(R, dim=0)
                return R, None, None

        return SimilarityAutoGradFunc()

    def forward(self, x: Tensor) -> Tensor:
        return self.autograd_func.apply(x)


class GradientRule(PropagationRule):
    """Dummy propagation rule used when ignoring LRP and using gradients directly"""

    def forward_hook(self, module, inputs, outputs):
        pass

    def _manipulate_weights(self, module, inputs, outputs):
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
    def _legacy_forward(self, x: Tensor) -> Tensor:
        """Legacy forward function for this layer"""
        raise NotImplementedError

    @abstractmethod
    def _modified_forward(self, x: Tensor, lower_bound: Tensor, upper_bound: Tensor) -> Tensor:
        """Modified forward function"""
        raise NotImplementedError

    def _autograd_func(self) -> torch.autograd.Function:
        class ZBetaAutoGradFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, *args, **kwargs) -> Tensor:
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
                    normalized = grad_output.detach() / (
                        output_detached + self.stability_factor * (output_detached.sign() + (output_detached == 0))
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

    def forward(self, x: Tensor) -> Tensor:
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

    def _legacy_forward(self, x: Tensor) -> Tensor:
        """Legacy forward function for this layer"""
        return F.linear(x, self.weight, self.bias)

    def _modified_forward(self, x: Tensor, l_bound_tensor: Tensor, u_bound_tensor: Tensor) -> Tensor:
        return (
            F.linear(x, self.weight, self.bias if not self.set_bias_to_zero else None)
            - F.linear(
                l_bound_tensor,
                self.weight.data.clamp(min=0),
                self.bias.data.clamp(min=0) if not self.set_bias_to_zero else None,
            )
            - F.linear(
                u_bound_tensor,
                self.weight.data.clamp(max=0),
                self.bias.data.clamp(max=0) if not self.set_bias_to_zero else None,
            )
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

    def _legacy_forward(self, x: Tensor) -> Tensor:
        """Legacy forward function for this layer"""
        return self._conv_forward(x, self.weight, self.bias)

    def _modified_forward(self, x: Tensor, l_bound_tensor: Tensor, u_bound_tensor: Tensor) -> Tensor:
        return (
            self._conv_forward(x, self.weight, self.bias if not self.set_bias_to_zero else None)
            - self._conv_forward(
                l_bound_tensor,
                self.weight.data.clamp(min=0),
                self.bias.data.clamp(min=0) if not self.set_bias_to_zero else None,
            )
            - self._conv_forward(
                u_bound_tensor,
                self.weight.data.clamp(max=0),
                self.bias.data.clamp(max=0) if not self.set_bias_to_zero else None,
            )
        )


class Alpha1Beta0Layer(ABC):
    """Abstract layer for applying Alpha1-Beta0 rule during relevance backpropagation.
    For more information, see https://arxiv.org/pdf/1512.02479.pdf

    Use for convolution layers.

    Args:
        set_bias_to_zero (bool): ignore bias during backpropagation
        stability_factor (float): epsilon value used for numerical stability
    """

    def __init__(
        self,
        set_bias_to_zero: bool = True,
        stability_factor: float = 1e-6,
    ) -> None:
        self.set_bias_to_zero = set_bias_to_zero
        self.stability_factor = stability_factor
        self.autograd_func = self._autograd_func()
        self.rule = GradientRule()

    @abstractmethod
    def _legacy_forward(self, x: Tensor) -> Tensor:
        """Legacy forward function for this layer"""
        raise NotImplementedError

    @abstractmethod
    def _modified_forward(self, x: Tensor) -> Tensor:
        """Modified forward function"""
        raise NotImplementedError

    def _autograd_func(self) -> torch.autograd.Function:
        class ZBetaAutoGradFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, *args, **kwargs) -> Tensor:
                ctx.save_for_backward(*args)
                # Keep original inference
                return self._legacy_forward(*args)

            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                x = x.clone().detach().requires_grad_(True)
                # Epsilon-rule for LRP propagation
                with torch.enable_grad():
                    output = self._modified_forward(x)
                    output_detached = output.clone().detach()

                    normalized = grad_output.detach() / (
                        output_detached
                        + self.stability_factor * output_detached.sign()
                        + 1e-12 * (output_detached == 0)  # Fixed stability factor when grads are 0
                    )
                    # Backward pass
                    output.backward(normalized)
                grads = x * x.grad
                return grads

        return ZBetaAutoGradFunc()

    def forward(self, x: Tensor) -> Tensor:
        return self.autograd_func.apply(x)


class Alpha1Beta0Linear(Alpha1Beta0Layer, nn.Linear):
    # Alpha1Beta0Layer takes precedence to supersede "forward" function from nn.Module
    """Replacement layer for applying Alpha1-Beta0 rule on a nn.Linear operator during relevance backpropagation.
    For more information, see https://arxiv.org/pdf/1512.02479.pdf

    Use for convolution layers.

    Args:
        module (nn.Module): layer to be replaced
        set_bias_to_zero (bool): ignore bias during backpropagation
        stability_factor (float): epsilon value used for numerical stability
    """

    def __init__(
        self,
        module: nn.Linear,
        set_bias_to_zero: bool = True,
        stability_factor: float = 1e-6,
    ) -> None:
        # Init Alpha1Beta0Layer
        Alpha1Beta0Layer.__init__(
            self,
            set_bias_to_zero=set_bias_to_zero,
            stability_factor=stability_factor,
        )

        # Copy source module configuration and parameters
        nn.Linear.__init__(
            self, in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None
        )
        self.load_state_dict(module.state_dict())

    def _legacy_forward(self, x: Tensor) -> Tensor:
        """Legacy forward function for this layer"""
        return F.linear(x, self.weight, self.bias)

    def _modified_forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x.clamp(min=0),
            self.weight.data.clamp(min=0),
            self.bias.data.clamp(min=0) if not self.set_bias_to_zero else None,
        ) + F.linear(x.clamp(max=0), self.weight.data.clamp(max=0), None)


class Alpha1Beta0Conv2d(Alpha1Beta0Layer, nn.Conv2d):
    # Alpha1Beta0Layer takes precedence to supersede "forward" function from nn.Module
    """Replacement layer for applying Alpha1-Beta0 rule on a nn.Conv2d operator during relevance backpropagation.
    For more information, see https://arxiv.org/pdf/1512.02479.pdf

    Use for convolution layers.

    Args:
        module (nn.Module): layer to be replaced
        set_bias_to_zero (bool): ignore bias during backpropagation
        stability_factor (float): epsilon value used for numerical stability
    """

    def __init__(
        self,
        module: nn.Conv2d,
        set_bias_to_zero: bool = True,
        stability_factor: float = 1e-6,
    ) -> None:
        # Init ZBetaLayer
        Alpha1Beta0Layer.__init__(
            self,
            set_bias_to_zero=set_bias_to_zero,
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

    def _legacy_forward(self, x: Tensor) -> Tensor:
        """Legacy forward function for this layer"""
        return self._conv_forward(x, self.weight, self.bias)

    def _modified_forward(self, x: Tensor) -> Tensor:
        return self._conv_forward(
            x.clamp(min=0),
            self.weight.data.clamp(min=0),
            self.bias.data.clamp(min=0) if not self.set_bias_to_zero else None,
        ) + self._conv_forward(x.clamp(max=0), self.weight.data.clamp(max=0), None)


class StackedSum(nn.Module):
    """Replacement layer used in order to enforce Epsilon-rule through the addition operator inside residual blocks.
    For more information, see https://github.com/AlexBinder/LRP_Pytorch_Resnets_Densenet/blob/master/lrp_general6.py

    Use for residual blocks (eg. ResNet).

    Args:
        stability_factor (float): epsilon value used for numerical stability

    """

    def __init__(self, stability_factor: float = 1e-6):
        super().__init__()
        self.stability_factor = stability_factor
        self.autograd_func = self._autograd_func()
        self.rule = GradientRule()

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.autograd_func.apply(a, b)

    def _autograd_func(self) -> torch.autograd.Function:
        class StackedSumAutoGradFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, a: Tensor, b: Tensor) -> Tensor:
                """Replacement operator for the addition of residual blocks"""
                stacked = torch.stack([a, b], dim=0)
                ctx.save_for_backward(stacked)
                return torch.sum(stacked, dim=0)

            @staticmethod
            def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, Tensor]:
                (stacked,) = ctx.saved_tensors
                stacked = stacked.clone().detach().requires_grad_(True)
                with torch.enable_grad():
                    # Epsilon-rule for LRP propagation
                    output = torch.sum(stacked, dim=0)
                    output_detached = output.clone().detach()

                    normalized = grad_output.detach() / (
                        output_detached + self.stability_factor * output_detached.sign()
                    )
                    # Backward pass
                    output.backward(normalized)
                grads = stacked * stacked.grad
                return grads[0], grads[1]

        return StackedSumAutoGradFunc()


def get_extractor_lrp_composite_model(
    model: nn.Module,
    set_bias_to_zero: bool,
    stability_factor: float = 1e-6,
    use_zbeta: bool = True,
    zbeta_lower_bound: float = min([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]),
    zbeta_upper_bound: float = max([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]),
) -> nn.Module:
    """Prepare a feature extractor for composite LRP

    Args:
        model: target model
        set_bias_to_zero: ignore bias in linear layers
        stability_factor (float): epsilon value used for numerical stability
        use_zbeta: use z-beta rule on first convolution
        zbeta_lower_bound (float): smallest admissible pixel value (used in z-beta rule)
        zbeta_upper_bound (float): largest admissible pixel value (used in z-beta rule)

    Returns:
        copy of the model, ready for running Captum LRP
    """
    lrp_model = copy.deepcopy(model)

    def _search_and_replace_addition(module: nn.Module) -> nn.Module:
        # Adaptation of https://github.com/pytorch/examples/blob/main/fx/replace_op.py
        add_idx = 0
        # Trace module to find all operations
        traced = symbolic_trace(module)
        patterns = {operator.add, torch.add, "add"}
        for n in traced.graph.nodes:
            if any(n.target == pattern for pattern in patterns):
                # Add a new layer to the module
                traced.add_module(f"stacked_adder_{add_idx}", StackedSum(stability_factor=stability_factor))
                with traced.graph.inserting_after(n):
                    new_node = traced.graph.call_module(f"stacked_adder_{add_idx}", n.args, n.kwargs)
                    n.replace_all_uses_with(new_node)
                # Remove the old node from the graph
                traced.graph.erase_node(n)
                add_idx += 1
        traced.recompile()
        return traced

    lrp_model = _search_and_replace_addition(lrp_model)

    def _find_batch_normalization_and_convolution_attribute_names(module: nn.Module):
        """Find the name and location of all convolutions that are followed by batch normalization inside a module

        Args:
            module: current top module

        Returns:
            the top module and the name of each convolution/batch normalization inside that module
        """
        target_conv_name = None
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                target_conv_name = name
            elif isinstance(child, nn.BatchNorm2d):
                if target_conv_name is None:
                    raise RuntimeError(f"Could not merge BatchNormalization layer {name}: missing target Conv2d.")
                yield module, target_conv_name, name
                target_conv_name = None
            else:
                target_conv_name = None
                yield from _find_batch_normalization_and_convolution_attribute_names(child)

    # Merge batch normalization layers into convolutional layers, as described in
    # https://iphome.hhi.de/samek/pdf/MonXAI19.pdf on page 205
    for top_module, conv_name, bn_name in _find_batch_normalization_and_convolution_attribute_names(module=lrp_model):
        bn_layer = getattr(top_module, bn_name)
        conv_layer = getattr(top_module, conv_name)
        # Merge as in https://github.com/AlexBinder/LRP_Pytorch_Resnets_Densenet/blob/master/lrp_general6.py#L242C1-L263C16
        s = (bn_layer.running_var + bn_layer.eps) ** 0.5
        w = bn_layer.weight
        b = bn_layer.bias
        m = bn_layer.running_mean
        conv_layer.weight = torch.nn.Parameter(conv_layer.weight * (w / s).reshape(-1, 1, 1, 1))
        if conv_layer.bias is None:
            conv_layer.bias = torch.nn.Parameter((0 - m) * (w / s) + b)
        else:
            conv_layer.bias = torch.nn.Parameter((conv_layer.bias - m) * (w / s) + b)
        # Reset BatchNormalization parameters back to Identity
        bn_layer.reset_parameters()

    def _find_convolution_attribute_names(module: nn.Module):
        """Find the name and location of all convolutions inside a module

        Args:
            module: current top module

        Returns:
            the top module and the name of each convolution inside that module
        """
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                yield module, name
            yield from _find_convolution_attribute_names(child)

    for conv_idx, (top_module, conv_name) in enumerate(_find_convolution_attribute_names(module=lrp_model)):
        conv_layer = getattr(top_module, conv_name)
        if conv_idx == 0 and use_zbeta:
            # Apply Z^Beta rule on the first convolution
            setattr(
                top_module,
                conv_name,
                ZBetaConv2d(
                    module=conv_layer,
                    set_bias_to_zero=set_bias_to_zero,
                    stability_factor=stability_factor,
                    lower_bound=zbeta_lower_bound,
                    upper_bound=zbeta_upper_bound,
                ),
            )
        else:
            # Apply Alpha1Beta0 rule on all other convolutions
            setattr(
                top_module,
                conv_name,
                Alpha1Beta0Conv2d(
                    module=conv_layer, set_bias_to_zero=set_bias_to_zero, stability_factor=stability_factor
                ),
            )

    # Set custom rule policy
    policy = {"MaxPool2d": GradientRule, "BatchNorm2d": IdentityRule, "Sigmoid": IdentityRule}

    def _set_policy(module: nn.Module, child_path: str = "") -> None:
        """Set propagation rules depending on module type

        Args:
            module: target module
            child_path: path inside module

        """
        for name, child in module.named_children():
            key = type(child).__name__
            if key in policy:
                logger.debug(f"Setting rule {policy[key].__name__} to module {child_path}{name} of type {key}")
                child.rule = policy[key]()
            _set_policy(child, child_path=f"{child_path}{name}.")

    _set_policy(lrp_model)
    return lrp_model


def get_cabrnet_lrp_composite_model(
    model: nn.Module,
    set_bias_to_zero: bool,
    stability_factor: float = 1e-6,
    use_zbeta: bool = True,
    zbeta_lower_bound: float = min([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]),
    zbeta_upper_bound: float = max([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]),
) -> nn.Module:
    """Prepare a CaBRNet ProtoClassifier model for composite LRP

    Args:
        model: target model
        set_bias_to_zero: ignore bias in linear layers
        stability_factor (float): epsilon value used for numerical stability
        use_zbeta: use z-beta rule on first convolution
        zbeta_lower_bound (float): smallest admissible pixel value (used in z-beta rule)
        zbeta_upper_bound (float): largest admissible pixel value (used in z-beta rule)

    Returns:
        copy of the model, ready for running Captum LRP
    """
    if not hasattr(model, "extractor"):
        # Check attribute presence rather than using isinstance(model, ProtoClassifier) to avoid circular dependencies
        logger.warning("Target model is not a ProtoClassifier, using generic function instead.")
        # Try to convert the model using the more generic function
        return get_extractor_lrp_composite_model(
            model=model,
            set_bias_to_zero=set_bias_to_zero,
            stability_factor=stability_factor,
            use_zbeta=use_zbeta,
            zbeta_lower_bound=zbeta_lower_bound,
            zbeta_upper_bound=zbeta_upper_bound,
        )

    lrp_model = copy.deepcopy(model)

    # Convert feature extractor
    lrp_model.extractor = get_extractor_lrp_composite_model(
        model=lrp_model.extractor,
        set_bias_to_zero=set_bias_to_zero,
        stability_factor=stability_factor,
        use_zbeta=use_zbeta,
        zbeta_lower_bound=zbeta_lower_bound,
        zbeta_upper_bound=zbeta_upper_bound,
    )

    # Replace decision layer
    lrp_model.classifier = DecisionLRPWrapper(classifier=lrp_model.classifier, stability_factor=1e-12)
    # Mark the model as ready for LRP
    lrp_model.lrp_ready = True
    return lrp_model
