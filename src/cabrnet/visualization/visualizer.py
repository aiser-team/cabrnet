from __future__ import annotations
import torch.nn as nn
from torch import Tensor
from loguru import logger
import argparse
from cabrnet.utils.parser import load_config
from cabrnet.visualization.upsampling import cubic_upsampling
from cabrnet.visualization.gradients import smoothgrad, randgrad, prp, saliency
from cabrnet.visualization.prp_utils import get_cabrnet_lrp_composite_model
from cabrnet.visualization.view import *
from typing import Callable


supported_attribution_functions = {
    "cubic_upsampling": cubic_upsampling,
    "smoothgrad": smoothgrad,
    "saliency": saliency,
    "randgrad": randgrad,
    "prp": prp,
}


class SimilarityVisualizer(nn.Module):
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
        """
        Init a patch visualizer
        Args:
            model: attach visualizer to a specific model
            attribution_fn: attribution function
            view_fn: viewing function
            attribution_params: optional parameters to attribution function
            view_params: optional parameters to viewing function
            config_file: optional path to the file used to configure the visualizer
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
        """
        Generates a visualization of the most similar patch to a given prototype
        Args:
            img: original image
            img_tensor: image tensor
            proto_idx: prototype index
            device: target hardware device
            location: location inside the similarity map

        Returns:
            patch visualization
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
        """
        Identify the most similar pixels to a given prototype
        Args:
            img: original image
            img_tensor: image tensor
            proto_idx: prototype index
            device: target hardware device
            location: location inside the similarity map

        Returns:
            importance map
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
        """Create the argument parser for a ProtoVisualizer.
        Args:
            parser: Existing parser (if any)
            mandatory_config: Make configuration mandatory

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
        """
        Builds a ProtoVisualizer from a YAML configuration file
        Args:
            config_file: path to configuration file
            model: target model

        Returns:
            ProtoVisualizer
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
