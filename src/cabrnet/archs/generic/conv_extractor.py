import os
import warnings
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as torch_models
from loguru import logger
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

from cabrnet.core.utils.exceptions import check_mandatory_fields
from cabrnet.core.utils.init import layer_init_functions

warnings.filterwarnings("ignore")


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
        super(ConvExtractor, self).__init__()

        # Check mandatory fields
        check_mandatory_fields(config_dict=config, mandatory_fields=["backbone"], location="extractor configuration")
        backbone_config = config["backbone"]
        check_mandatory_fields(
            config_dict=backbone_config,
            mandatory_fields=["arch", "weights"],
            location="backbone configuration",
        )

        arch = backbone_config["arch"]
        arch_params = backbone_config.get("params", {})
        weights = backbone_config["weights"]

        # Check that model architecture is supported
        assert arch.lower() in torch_models.list_models(), f"Unsupported model architecture: {arch}"

        if weights is None:
            weights = ""

        if os.path.isfile(weights):
            if not ignore_weight_errors:
                logger.info(f"Loading state dict for feature extractor: {weights}")
            loaded_weights = torch.load(weights, map_location="cpu")
            model = torch_models.get_model(arch, **arch_params)
            if isinstance(loaded_weights, dict):
                model.load_state_dict(loaded_weights)
            elif isinstance(loaded_weights, nn.Module):
                model.load_state_dict(loaded_weights.state_dict(), strict=False)
            else:
                raise ValueError(f"Unsupported weights type: {type(loaded_weights)}")
        elif hasattr(torch_models.get_model_weights(arch), weights):
            if not ignore_weight_errors:
                logger.info(f"Loading pytorch weights: {weights}")
            loaded_weights = getattr(torch_models.get_model_weights(arch), weights)
            model = torch_models.get_model(arch, weights=loaded_weights, **arch_params)
        elif ignore_weight_errors:
            logger.warning(
                f"Could not load initial weights for the feature extractor. "
                f"This might be OK if the model state dictionary is loaded afterwards."
            )
            model = torch_models.get_model(arch, **arch_params)
        else:
            raise ValueError(f"Cannot load weights {weights} for model of type {arch}. Possible typo or missing file.")

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
        try:
            self.convnet = create_feature_extractor(model=model, return_nodes=return_nodes)
        except ValueError as e:
            logger.error(f"Could not create feature extractor. Possible layer names: {get_graph_node_names(model)}")
            logger.error("See model architecture below")
            logger.info(model)
            raise e
        # Dummy inference to recover number of output channels from the feature extractor
        self.convnet.eval()
        output_tensors = self.convnet(torch.zeros((1, 3, 224, 224)))

        add_ons, self.output_channels = {}, {}
        for pipeline_name in self.source_layers.keys():
            layer, num_channels = self.create_add_on(
                config=config[pipeline_name].get("add_on"), in_channels=output_tensors[pipeline_name].size(1)
            )
            add_ons[pipeline_name] = layer
            self.output_channels[pipeline_name] = num_channels

        # Create a ModuleDict to register add-on layers as submodules, or simply use a single add-on module
        self.add_on = nn.ModuleDict(add_ons) if self.num_pipelines > 1 else add_ons[next(iter(add_ons))]

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
    def create_add_on(config: dict[str, dict] | None, in_channels: int) -> Tuple[nn.Sequential | None, int]:
        r"""Builds add-on layers based on configuration.

        Args:
            config (dictionary): Add-on layers configuration.
            in_channels (int): Number of input channels (as given by the feature extractor).

        Returns:
            Module containing all add-on layers.

        Raises:
            ValueError when the configuration is invalid.
        """
        if config is None:
            # No add-on layers
            return None, in_channels

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        init_mode = None
        for key, val in config.items():
            if key == "init_mode":
                # Extract initialisation mode
                if val not in layer_init_functions:
                    raise ValueError(f"Unsupported add_on layers initialisation mode {val}")
                init_mode = val
                continue
            if not hasattr(nn, val["type"]):
                raise ValueError(f"Module {val['type']} not found in torch.nn")
            params = val.get("params")
            if params is not None:
                if val["type"] == "Conv2d":
                    # Check or update in_channels
                    if params.get("in_channels") is None:
                        params["in_channels"] = in_channels
                    elif params["in_channels"] != in_channels:
                        raise ValueError(
                            f"Invalid number of input channels for layer {key}. "
                            f"Should be {in_channels} but {params['in_channels']} was given."
                        )
                    in_channels = params["out_channels"]
                layer_module = getattr(nn, val["type"])(**params)
            else:
                layer_module = getattr(nn, val["type"])()
            layers[key] = layer_module
        add_on = nn.Sequential(layers)

        # Apply initialisation function (if any)
        if init_mode:
            add_on.apply(layer_init_functions[init_mode])

        return add_on, in_channels
