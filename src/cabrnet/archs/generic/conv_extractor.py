import importlib
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torchvision.models as torch_models
from loguru import logger
from torch import nn
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

from cabrnet.archs.custom_extractors.onnx_backbone import GenericONNXModel
from cabrnet.core.utils.exceptions import check_mandatory_fields
from cabrnet.core.utils.init import LAYER_INIT_FUNCTIONS
from cabrnet.core.utils.state_dict import state_dict_for

warnings.filterwarnings("ignore")

CHANNEL_PRESERVING_ADD_ON_LAYERS = (
    nn.AdaptiveAvgPool2d,
    nn.AvgPool2d,
    nn.BatchNorm2d,
    nn.Dropout,
    nn.Dropout2d,
    nn.ELU,
    nn.GELU,
    nn.Identity,
    nn.LeakyReLU,
    nn.MaxPool2d,
    nn.PReLU,
    nn.ReLU,
    nn.SELU,
    nn.SiLU,
    nn.Sigmoid,
    nn.Softplus,
    nn.Tanh,
)


def _normalize_optional_weights(weights: Any, location: str) -> Any:
    r"""Converts the legacy string sentinel for absent weights to ``None``.

    Args:
        weights (any): Configured weight value.
        location (str): Configuration field being normalized.

    Returns:
        The normalized weight value.
    """
    if isinstance(weights, str) and weights == "None":
        logger.warning(
            f"Deprecated weights value 'None' in {location}; use YAML null instead. "
            "Support for the string value will be removed in a future release."
        )
        return None
    return weights


def _validate_custom_weights(weights: Any, location: str) -> None:
    r"""Validates weights configured for a custom component.

    Args:
        weights (any): Configured weight value.
        location (str): Custom component type for error reporting.

    Raises:
        ValueError: If weights are neither absent nor a ``.pth`` checkpoint path.
    """
    if weights is not None and (not isinstance(weights, str) or not weights.lower().endswith(".pth")):
        raise ValueError(f"Custom {location} only support null weights or a path ending in '.pth'. Got: {weights!r}")


def _create_layer(
    layer_name: str, config: dict[str, Any], input_channels: int | None, weights: Any
) -> tuple[nn.Module, int | None]:
    r"""Builds an add-on layer and determines its output channels when possible.
    This allows to create a sequence of addon layers without specifying the number of channels for each component.

    Args:
        layer_name (str): Add-on layer identifier.
        config (dictionary): Layer configuration.
        input_channels (int, optional): Number of channels produced by the preceding layer.
        weights (any): Optional layer checkpoint path.

    Returns:
        The layer and its output channel count, if known.

    Raises:
        ValueError: If the configuration is invalid.
    """
    module_name = config.get("module")
    if module_name is not None:
        _validate_custom_weights(weights, "add-on layers")
        add_on_module = importlib.import_module(module_name)
        try:
            layer_constructor = getattr(add_on_module, config["type"])
        except AttributeError as error:
            raise ValueError(f"Add-on class '{config['type']}' does not exist in module '{module_name}'.") from error
    else:
        if not hasattr(nn, config["type"]):
            raise ValueError(f"Module {config['type']} not found in torch.nn")
        layer_constructor = getattr(nn, config["type"])

    params = config.get("params")
    if module_name is None and config["type"] == "Conv2d":
        layer_module = _create_conv2d(layer_name, params, input_channels)
    elif params is not None:
        layer_module = layer_constructor(**params)
    else:
        layer_module = layer_constructor()
    if not isinstance(layer_module, nn.Module):
        raise ValueError(f"Add-on class '{config['type']}' must return a torch.nn.Module.")

    return layer_module, _output_channels_for(layer_module, input_channels, module_name)


def _create_conv2d(layer_name: str, params: dict[str, Any] | None, input_channels: int | None) -> nn.Conv2d:
    r"""Builds a convolutional add-on layer with validated input channels.

    Args:
        layer_name (str): Add-on layer identifier.
        params (dictionary, optional): Convolution parameters.
        input_channels (int, optional): Number of channels produced by the preceding layer.

    Returns:
        Configured convolutional layer.

    Raises:
        ValueError: If input channels cannot be inferred or do not match the configuration.
    """
    if params is None:
        raise ValueError(f"Convolutional add-on layer {layer_name} must provide parameters.")
    configured_channels = params.get("in_channels")
    if configured_channels is None:
        if input_channels is None:
            raise ValueError(
                f"Could not infer input channels for convolutional add-on layer {layer_name}. "
                "Set its in_channels explicitly."
            )
        params["in_channels"] = input_channels
    elif input_channels is not None and configured_channels != input_channels:
        raise ValueError(
            f"Invalid number of input channels for layer {layer_name}. "
            f"Should be {input_channels} but {configured_channels} was given."
        )
    return nn.Conv2d(**params)


def _output_channels_for(layer: nn.Module, input_channels: int | None, module_name: str | None) -> int | None:
    r"""Returns output channels for a layer whose channel behavior is known.

    Args:
        layer (Module): Configured add-on layer.
        input_channels (int, optional): Number of channels produced by the preceding layer.
        module_name (str, optional): Module that provided the layer class.

    Returns:
        Output channels, or None for custom and channel-changing layers.
    """
    if isinstance(layer, nn.Conv2d):
        return layer.out_channels
    if module_name is None and isinstance(layer, CHANNEL_PRESERVING_ADD_ON_LAYERS):
        return input_channels
    return None


class ConvExtractor(nn.Module):
    r"""Class representing the feature extractor.

    Attributes:
        arch_name: Architecture name.
        weights: Weights of the neural network.
        convnet: Graph module that represents the intermediate nodes from the given model.
        add_on: Add-on layer(s).
        num_pipelines: Number of extracted layers.
        output_channels: Number of output channels of the feature extractor.
    """

    def __init__(
        self,
        config: dict[str, dict],
        seed: int | None = None,
        ignore_weight_errors: bool = False,
    ) -> None:
        r"""Initializes a ConvExtractor from a configuration dictionary.

        Args:
            config (dictionary): Configuration dictionary.
            seed (int, optional): Random seed (used only to resynchronise random number generators in
                compatibility tests). Default: None.
            ignore_weight_errors (bool, optional): Ignore all errors regarding model weights
                (they will be overwritten later on). Default: False.

        Raises:
            ValueError when configuration is invalid.
        """
        super().__init__()

        # Check mandatory fields
        check_mandatory_fields(
            config_dict=config,
            mandatory_fields=["backbone"],
            location="extractor configuration",
        )
        backbone_config = config["backbone"]
        check_mandatory_fields(
            config_dict=backbone_config,
            mandatory_fields=["arch", "weights"],
            location="backbone configuration",
        )

        arch = backbone_config["arch"]
        module_name = backbone_config.get("module")
        arch_params = backbone_config.get("params", {})
        weights = _normalize_optional_weights(backbone_config["weights"], "backbone configuration")

        if module_name:
            _validate_custom_weights(weights, "backbones")
            backbone_module = importlib.import_module(module_name)
            try:
                model_constructor = getattr(backbone_module, arch)
            except AttributeError as error:
                raise ValueError(f"Backbone class '{arch}' does not exist in module '{module_name}'.") from error
        else:
            # Check that model architecture is supported
            assert arch.lower() in torch_models.list_models(), f"Unsupported model architecture: {arch}"

            def model_constructor(**kwargs):
                return torch_models.get_model(arch, **kwargs)

        if weights is None:
            weights = ""

        if Path(weights).is_file():
            if not ignore_weight_errors:
                logger.info(f"Loading state dict for feature extractor: {weights}")
            logger.warning(
                f"Loading file {weights}. Please ensure that you trust this file. "
                f"For more information regarding the potential risk of arbitrary code execution, "
                f"see https://docs.pytorch.org/docs/stable/generated/torch.load.html"
            )
            loaded_weights = torch.load(weights, map_location="cpu", weights_only=False)

            model = model_constructor(**arch_params)
            if isinstance(loaded_weights, Mapping):
                model.load_state_dict(state_dict_for(loaded_weights, "extractor.convnet"))
            elif isinstance(loaded_weights, nn.Module):
                model.load_state_dict(loaded_weights.state_dict(), strict=False)
            else:
                raise ValueError(f"Unsupported weights type: {type(loaded_weights)}")
        elif not module_name and weights and hasattr(torch_models.get_model_weights(arch), weights):
            if not ignore_weight_errors:
                logger.info(f"Loading pytorch weights: {weights}")
            loaded_weights = getattr(torch_models.get_model_weights(arch), weights)
            model = torch_models.get_model(arch, weights=loaded_weights, **arch_params)
        elif not weights or ignore_weight_errors:
            logger.warning(
                "Could not load initial weights for the feature extractor. "
                "This might be OK if the model state dictionary is loaded afterwards, "
                "or the model is in ONNX format and all parameters are provided in the ONNX file."
            )
            model = model_constructor(**arch_params)
        else:
            raise ValueError(f"Cannot load weights {weights} for model of type {arch}. Possible typo or missing file.")

        # Backbone post-processing (if any)
        for op, op_config in backbone_config.get("postprocess", {}).items():
            preprocess_fn = None
            match op:
                case "stride_divider":
                    ratio = op_config["ratio"]
                    min_channels = op_config["min_channels"]

                    def divide_stride(module: nn.Module):
                        if (
                            isinstance(module, nn.Conv2d)
                            and module.in_channels > min_channels
                            and min(module.stride) >= ratio
                        ):
                            module.stride = tuple(s // ratio for s in module.stride)

                    preprocess_fn = divide_stride
                case _:
                    raise ValueError(f"Unsupported preprocessing function {op}")

            model.apply(preprocess_fn)

        if seed is not None:
            # Reset random generator (compatibility tests only)
            torch.manual_seed(seed)

        self.arch_name = arch.lower()
        self.weights = weights

        # Find the source layer for each pipeline
        self.source_layers = {
            pipeline_name: pipeline_config["source_layer"]
            for pipeline_name, pipeline_config in config.items()
            if pipeline_name != "backbone"
        }
        self.num_pipelines = len(self.source_layers)
        assert self.num_pipelines > 0, "No pipeline defined for feature extraction"

        # Reverse mapping between pipeline names and source layers to build return nodes
        return_nodes = {val: key for key, val in self.source_layers.items()}
        if isinstance(model, GenericONNXModel):
            try:
                model.trim_model(return_nodes)
                self.convnet = model
            except ValueError as e:
                logger.error(
                    f"Could not create feature extractor from ONNX model. Possible layer names: "
                    f"{model.available_node_names()}"  # type: ignore[reportCallIssue]
                )
                raise e
        else:
            try:
                self.convnet = create_feature_extractor(model=model, return_nodes=return_nodes)
            except ValueError as e:
                # FIXME: Should be fixed with model loading rewrite.
                logger.error(
                    f"Could not create feature extractor. Possible layer names: {get_graph_node_names(model)}"  # type: ignore[reportCallIssue]
                )
                logger.error("See model architecture below")
                logger.info(model)
                raise e

        # Dummy inference to recover feature shapes and build add-on layers
        self.convnet.eval()
        try:
            output_tensors: dict[str, torch.Tensor] | None = self.convnet(torch.zeros((1, 3, 224, 224)))
        except Exception as error:
            logger.warning(f"Could not infer feature shapes from the backbone dummy tensor: {error}")
            output_tensors = None

        add_ons, output_channels = {}, {}
        for pipeline_name in self.source_layers:
            feature_tensor = output_tensors[pipeline_name] if output_tensors is not None else None
            add_on_module_path = "extractor.add_on"
            if self.num_pipelines > 1:
                add_on_module_path = f"{add_on_module_path}.{pipeline_name}"
            add_on_result = self.create_add_on(
                config=config[pipeline_name].get("add_on"),
                in_channels=feature_tensor.size(1) if feature_tensor is not None else None,
                module_path=add_on_module_path,
            )
            if add_on_result is None:
                add_ons[pipeline_name] = None
                output_channels[pipeline_name] = feature_tensor.size(1) if feature_tensor is not None else None
            else:
                add_ons[pipeline_name], output_channels[pipeline_name] = add_on_result

        # Create a ModuleDict to register add-on layers as submodules, or simply use a single add-on module
        self.add_on = nn.ModuleDict(add_ons) if self.num_pipelines > 1 else add_ons[next(iter(add_ons))]
        self.output_channels = output_channels

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor | dict[str, torch.Tensor]:
        r"""Computes convolutional features.

        Args:
            x (tensor): Input tensor.

        Returns:
            Dictionary of tensors of convolutional features or tensor of convolutional features if the dictionary
            contains a single entry.
        """
        features = self.convnet(x)
        if self.num_pipelines == 1:
            # Single layer extraction (features contains a single entry)
            features = features[next(iter(features))]  # type: ignore
            if self.add_on:
                features = self.add_on(features)
        else:
            # Multi-layer extraction
            for pipeline_name, add_on_layer in self.add_on.items():
                # Apply add-on layers independently
                if add_on_layer:
                    features[pipeline_name] = add_on_layer(features[pipeline_name])
        return features

    @staticmethod
    def create_add_on(
        config: dict[str, dict] | None, in_channels: int | None, module_path: str = "extractor.add_on"
    ) -> tuple[nn.Sequential, int | None] | None:
        r"""Builds configured add-on layers.

        ``in_channels`` is inferred by an optional dummy backbone pass. Channel flow is
        propagated through built-in convolutional and channel-preserving layers without
        executing the add-on. Custom layers have unknown output channels.

        Args:
            config (dictionary): Add-on layers configuration.
            in_channels (int, optional): Input channel count from the feature extractor.
            module_path (str, optional): Path of the add-on in a complete model state dictionary.
                Default: "extractor.add_on".

        Returns:
            The add-on module and its output channel count, if known; None when no add-on is configured.

        Raises:
            ValueError when the configuration is invalid.
        """
        if config is None:
            # No add-on layers
            return None

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        weights_paths: dict[str, str] = {}
        init_mode = None
        output_channels = in_channels

        for key, val in config.items():
            if key == "init_mode":
                # Extract initialisation mode
                if val not in LAYER_INIT_FUNCTIONS:
                    raise ValueError(f"Unsupported add_on layers initialisation mode {val}")
                init_mode = val
                continue
            weights = _normalize_optional_weights(val.get("weights"), f"add-on layer '{key}'")
            layer_module, output_channels = _create_layer(key, val, output_channels, weights)
            if weights is not None:
                if not Path(weights).is_file():
                    raise ValueError(f"Cannot load add-on weights from '{weights}': file does not exist.")
                weights_paths[key] = weights
            layers[key] = layer_module

        add_on = nn.Sequential(layers)

        # Apply initialisation function (if any)
        if init_mode:
            add_on.apply(LAYER_INIT_FUNCTIONS[init_mode])

        for key, weights_path in weights_paths.items():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            if not isinstance(state_dict, Mapping):
                raise ValueError(f"Checkpoint at '{weights_path}' does not contain a state dictionary.")
            add_on._modules[key].load_state_dict(state_dict_for(state_dict, f"{module_path}.{key}"))

        return add_on, output_channels
