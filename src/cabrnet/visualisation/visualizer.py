from __future__ import annotations
import torch.nn as nn
from torch import Tensor
from loguru import logger
from PIL import Image
import argparse
from cabrnet.utils.parser import load_config
from cabrnet.visualisation.upsampling import cubic_upsampling
from cabrnet.visualisation.gradients import smoothgrad, randgrad, prp
from cabrnet.visualisation.prp_utils import get_cabrnet_lrp_composite_model
import cabrnet.visualisation.view as viewing_module
from typing import Callable


supported_retrace_functions = {
    "cubic_upsampling": cubic_upsampling,
    "smoothgrad": smoothgrad,
    "randgrad": randgrad,
    "prp": prp,
}


class SimilarityVisualizer(nn.Module):
    def __init__(
        self,
        retrace_fn: Callable,
        view_fn: Callable,
        retrace_params: dict | None = None,
        view_params: dict | None = None,
        config_file: str | None = None,
        *args,
        **kwargs,
    ):
        """
        Init a patch visualizer
        Args:
            retrace_fn: visualization function
            view_fn: viewing function
            retrace_params: optional parameters to retrace function
            view_params: optional parameters to viewing function
            config_file: optional path to the file used to configure the visualizer
        """
        super().__init__(*args, **kwargs)
        self.retrace = retrace_fn
        self.retrace_params = retrace_params if retrace_params is not None else {}
        self.view = view_fn
        self.view_params = view_params if view_params is not None else {}
        self.config_file = config_file

    def forward(
        self,
        model: nn.Module,
        img: Image.Image,
        img_tensor: Tensor,
        proto_idx: int,
        device: str,
        location: tuple[int, int] | None = None,
    ) -> Image.Image:
        """
        Generates a visualization of the most similar patch to a given prototype
        Args:
            model: target model
            img: original image
            img_tensor: image tensor
            proto_idx: prototype index
            device: target hardware device
            location: location inside the similarity map

        Returns:
            patch visualization
        """
        sim_map = self.retrace(
            model=model,
            img=img,
            img_tensor=img_tensor,
            proto_idx=proto_idx,
            device=device,
            location=location,
            **self.retrace_params,
        )
        return self.view(img=img, sim_map=sim_map, **self.view_params)

    def prepare_model(self, model: nn.Module) -> nn.Module:
        # Perform model preparation (depends on retrace function)
        if self.retrace == prp:
            return get_cabrnet_lrp_composite_model(model)
        return model

    @staticmethod
    def create_parser(
        parser: argparse.ArgumentParser | None = None,
    ) -> argparse.ArgumentParser:
        """Create the argument parser for a ProtoVisualizer.
        Args:
            parser: Existing parser (if any)

        Returns:
            The parser itself.
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="Build a ProtoVisualizer")
        parser.add_argument(
            "--visualization",
            default="configs/prototree/visualization.yml",
            metavar="/path/to/file.yml",
            help="Path to the visualization configuration file",
        )
        return parser

    @staticmethod
    def build_from_config(config_file: str, target: str | None = None) -> SimilarityVisualizer:
        """
        Builds a ProtoVisualizer from a YAML configuration file
        Args:
            config_file: path to configuration file
            target: name of target in configuration file

        Returns:
            ProtoVisualizer
        """
        logger.info(
            f"Loading patch visualizer from {config_file}." + f" Target: {target}." if target is not None else ""
        )
        config_dict = load_config(config_file)
        if target is not None:
            if target not in config_dict:
                raise ValueError(f"Missing target {target} from configuration file {config_file}")
            config_dict = config_dict[target]

        # Sanity checks on mandatory field
        for mandatory_field in ["retrace", "view"]:
            if mandatory_field not in config_dict:
                raise ValueError(f"Missing mandatory field {mandatory_field} in configuration")

        # Visualization function
        retrace_fn = supported_retrace_functions.get(config_dict["retrace"]["type"])
        if retrace_fn is None:
            raise NotImplementedError(f"Unknown visualization function {config_dict['retrace']['type']}")
        retrace_params = config_dict["retrace"]["params"] if "params" in config_dict["retrace"] else None

        # Viewing function
        if hasattr(viewing_module, config_dict["view"]["type"]):
            view_fn = getattr(viewing_module, config_dict["view"]["type"])
        else:
            raise NotImplementedError(f"Unknown viewing function {config_dict['view']['type']}")
        view_params = config_dict["view"]["params"] if "params" in config_dict["view"] else None

        return SimilarityVisualizer(
            retrace_fn=retrace_fn,
            view_fn=view_fn,
            retrace_params=retrace_params,
            view_params=view_params,
            config_file=config_file,
        )
