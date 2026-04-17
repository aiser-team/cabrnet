from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from torch import Tensor

from cabrnet.core.attribution.augmentors import GaussianNoiseAugmentor
from cabrnet.core.utils.exceptions import check_mandatory_fields
from cabrnet.core.utils.parser import load_config
from cabrnet.core.visualization.gradients import attribute_prototypes
from cabrnet.core.visualization.prp_utils import get_cabrnet_lrp_composite_model
from cabrnet.core.visualization.upsampling import cubic_upsampling
from cabrnet.core.visualization.view import SUPPORTED_VIEWING_FUNCTIONS

# FIXME: support cubic and randgrad
# SUPPORTED_ATTRIBUTION_FUNCTIONS = {
#     "cubic_upsampling": cubic_upsampling,
#     "smoothgrad": smoothgrad,
#     "saliency": saliency,
#     "randgrad": randgrad,
#     "prp": prp,
# }


class SimilarityVisualizer(nn.Module):
    r"""Object used to extract patch visualizations from a model.

    Attributes:
        attribution: Function used to compute the relative importance of each pixel w.r.t. a given similarity score.
        attribution_params: Parameters of the attribution function.
        view: Function used to generate an image from the attribution map.
        view_params: Parameters of the viewing function.
        config_file: Path to the configuration file used to create this object.
        model: Target CaBRNet model.
    """

    def __init__(
        self,
        model: nn.Module,
        attribution_method: str,
        view_fn: Callable,
        attribution_params: dict | None = None,
        view_params: dict | None = None,
        config_file: Path | None = None,
        augmentors: list[nn.Module] = [],
        *args,
        **kwargs,
    ):
        r"""Initializes a patch visualizer.

        Args:
            model (Module): Attach visualizer to a specific model.
            attribution_fn (Callable): Attribution function.
            view_fn (Callable): Viewing function.
            attribution_params (dictionary, optional): Parameters to attribution function. Default: None.
            view_params (dictionary, optional): Parameters to viewing function. Default: None.
            config_file (Path, optional): Path to the file used to configure the visualizer. Default: None.
        """
        super().__init__(*args, **kwargs)
        self.attribution_method = attribution_method
        self.attribution_params = attribution_params if attribution_params is not None else {}
        self.augmentors = augmentors
        self.view = view_fn
        self.view_params = view_params if view_params is not None else {}
        self.config_file = config_file

        self.model = model
        if self.attribution_method == "prp":
            logger.info("Canonizing model for PRP")
            self.model = get_cabrnet_lrp_composite_model(
                model=model,
                set_bias_to_zero=True,
                stability_factor=self.attribution_params.get("stability_factor", 1e-6),
                use_zbeta=True,
            )

    def forward(
        self,
        img: Image.Image,
        img_tensor: Tensor,
        proto_idx: int,
        device: str | torch.device,
        location: tuple[int, int] | str | None = "max",
    ) -> Image.Image:
        r"""Generates a visualization of the most similar patch to a given prototype.

        Args:
            img (Image): Original image.
            img_tensor (tensor): Image tensor.
            proto_idx (int): Prototype index.
            device (str | device): Hardware device.
            location (tuple[int,int], str or None, optional): Location inside the similarity map.
                Can be given as an explicit location (tuple) or "max" for the location of maximum similarity.
                Default: max.

        Returns:
            Patch visualization.
        """
        sim_map = self.get_attribution(
            img=img, img_tensor=img_tensor, proto_idx=proto_idx, device=device, location=location
        )
        return self.view(img=img, sim_map=sim_map, **self.view_params)

    def get_attribution(
        self,
        img: Image.Image,
        img_tensor: Tensor,
        proto_idx: int,
        device: str | torch.device,
        location: tuple[int, int] | str | None = None,
    ) -> np.ndarray:
        r"""Identifies the most similar pixels to a given prototype.

        Args:
            img (Image): Original image.
            img_tensor (tensor): Image tensor.
            proto_idx (int): Prototype index.
            device (str | device): Hardware device.
            location (tuple[int,int], str or None, optional): Location inside the similarity map.
                Can be given as an explicit location (tuple) or "max" for the location of maximum similarity.
                Default: None.

        Returns:
            Importance map.
        """
        attribution_params = self.attribution_params.copy()
        if location is not None:
            # Overwrite parameter in attribution_params
            attribution_params["location"] = location

        return attribute_prototypes(
            model=self.model,
            algorithm=self.attribution_method,
            img=img,
            img_tensor=img_tensor,
            proto_idx=proto_idx,
            device=device,
            augmentors=self.augmentors,
            **attribution_params,
        )

    DEFAULT_VISUALIZATION_CONFIG = Path("visualization.yml")

    @staticmethod
    def create_parser(
        parser: argparse.ArgumentParser | None = None,
        mandatory_config: bool = False,
    ) -> argparse.ArgumentParser:
        r"""Creates the argument parser for a ProtoVisualizer.

        Args:
            parser (ArgumentParser, optional): Existing parser (if any). Default: None.
            mandatory_config (bool, optional): If True, makes the configuration mandatory. Default: False.

        Returns:
            The parser itself.
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="Build a ProtoVisualizer")
        parser.add_argument(
            "-z",
            "--visualization",
            type=Path,
            required=mandatory_config,
            metavar="/path/to/file.yml",
            help="path to the visualization configuration file",
        )
        return parser

    @staticmethod
    def build_from_config(config: Path | dict[str, Any], model: nn.Module) -> SimilarityVisualizer:
        r"""Builds a ProtoVisualizer from a configuration file or dictionary.

        Args:
            config (Path): Path to configuration file or dictionary.
            model (Module): Target model.

        Returns:
            ProtoVisualizer.
        """
        if isinstance(config, Path):
            logger.info(f"Loading patch visualizer from {config}.")
            config_dict = load_config(config)
        else:
            # Configuration is given in dictionary form
            config_dict = config

        # Sanity checks on mandatory field
        check_mandatory_fields(
            config_dict=config_dict, mandatory_fields=["attribution", "view"], location="visualizer configuration"
        )

        # Visualization function
        # TODO: upsampling, randgrad
        attribution_method = config_dict["attribution"]["type"]
        if attribution_method not in ["saliency", "smoothgrad", "lrp"]:
            raise NotImplementedError(f"Unknown visualization function {config_dict['attribution']['type']}")
        attribution_params = config_dict["attribution"]["params"] if "params" in config_dict["attribution"] else None

        # Viewing function
        if config_dict["view"]["type"] in SUPPORTED_VIEWING_FUNCTIONS:
            view_fn = SUPPORTED_VIEWING_FUNCTIONS[config_dict["view"]["type"]]
        else:
            raise NotImplementedError(f"Unknown viewing function {config_dict['view']['type']}")
        view_params = config_dict["view"]["params"] if "params" in config_dict["view"] else None

        if attribution_method == "smoothgrad":
            assert attribution_params is not None  # FIXME: fallback with default params
            augmentors = [GaussianNoiseAugmentor(attribution_params["num_samples"], attribution_params["noise_ratio"])]
        else:
            augmentors = []

        return SimilarityVisualizer(
            model=model,
            attribution_method=attribution_method,
            view_fn=view_fn,
            attribution_params=attribution_params,
            view_params=view_params,
            config_file=config if isinstance(config, Path) else None,
            augmentors=augmentors,
        )
