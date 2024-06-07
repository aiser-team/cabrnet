from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import torch.nn as nn
from cabrnet.utils.parser import load_config
from cabrnet.visualization.gradients import prp, randgrad, saliency, smoothgrad
from cabrnet.visualization.prp_utils import get_cabrnet_lrp_composite_model
from cabrnet.visualization.upsampling import cubic_upsampling
from cabrnet.visualization.view import supported_viewing_functions
from loguru import logger
from PIL import Image
from torch import Tensor

supported_attribution_functions = {
    "cubic_upsampling": cubic_upsampling,
    "smoothgrad": smoothgrad,
    "saliency": saliency,
    "randgrad": randgrad,
    "prp": prp,
}


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
        attribution_fn: Callable,
        view_fn: Callable,
        attribution_params: dict | None = None,
        view_params: dict | None = None,
        config_file: str | None = None,
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
            config_file (str, optional): Path to the file used to configure the visualizer. Default: None.
        """
        super().__init__(*args, **kwargs)
        self.attribution = attribution_fn
        self.attribution_params = attribution_params if attribution_params is not None else {}
        self.view = view_fn
        self.view_params = view_params if view_params is not None else {}
        self.config_file = config_file

        self.model = model
        if self.attribution == prp:
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
        device: str,
        location: tuple[int, int] | None = None,
    ) -> Image.Image:
        r"""Generates a visualization of the most similar patch to a given prototype.

        Args:
            img (Image): Original image.
            img_tensor (tensor): Image tensor.
            proto_idx (int): Prototype index.
            device (str): Target hardware device.
            location (tuple[int,int], optional): Location inside the similarity map. Default: None.

        Returns:
            Patch visualization.
        """
        sim_map = self.attribution(
            model=self.model,
            img=img,
            img_tensor=img_tensor,
            proto_idx=proto_idx,
            device=device,
            location=location,
            **self.attribution_params,
        )
        return self.view(img=img, sim_map=sim_map, **self.view_params)

    def get_attribution(
        self,
        img: Image.Image,
        img_tensor: Tensor,
        proto_idx: int,
        device: str,
        location: tuple[int, int] | None = None,
    ) -> np.ndarray:
        r"""Identifies the most similar pixels to a given prototype.

        Args:
            img (Image): Original image.
            img_tensor (tensor): Image tensor.
            proto_idx (int): Prototype index.
            device (str): Target hardware device.
            location (tuple[int,int], optional): Location inside the similarity map. Default: None.

        Returns:
            Importance map.
        """
        return self.attribution(
            model=self.model,
            img=img,
            img_tensor=img_tensor,
            proto_idx=proto_idx,
            device=device,
            location=location,
            **self.attribution_params,
        )

    DEFAULT_VISUALIZATION_CONFIG = "visualization.yml"

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
            required=mandatory_config,
            metavar="/path/to/file.yml",
            help="path to the visualization configuration file",
        )
        return parser

    @staticmethod
    def build_from_config(config_file: str, model: nn.Module) -> SimilarityVisualizer:
        r"""Builds a ProtoVisualizer from a configuration file.

        Args:
            config_file (str): Path to configuration file.
            model (Module): Target model.

        Returns:
            ProtoVisualizer.
        """
        logger.info(f"Loading patch visualizer from {config_file}.")
        config_dict = load_config(config_file)

        # Sanity checks on mandatory field
        for mandatory_field in ["attribution", "view"]:
            if mandatory_field not in config_dict:
                raise ValueError(f"Missing mandatory field {mandatory_field} in configuration")

        # Visualization function
        attribution_fn = supported_attribution_functions.get(config_dict["attribution"]["type"])
        if attribution_fn is None:
            raise NotImplementedError(f"Unknown visualization function {config_dict['attribution']['type']}")
        attribution_params = config_dict["attribution"]["params"] if "params" in config_dict["attribution"] else None

        # Viewing function
        if config_dict["view"]["type"] in supported_viewing_functions:
            view_fn = supported_viewing_functions[config_dict["view"]["type"]]
        else:
            raise NotImplementedError(f"Unknown viewing function {config_dict['view']['type']}")
        view_params = config_dict["view"]["params"] if "params" in config_dict["view"] else None

        return SimilarityVisualizer(
            model=model,
            attribution_fn=attribution_fn,
            view_fn=view_fn,
            attribution_params=attribution_params,
            view_params=view_params,
            config_file=config_file,
        )
