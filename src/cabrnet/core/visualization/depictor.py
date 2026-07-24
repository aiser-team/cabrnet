import argparse
import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.visualization.view import SUPPORTED_VIEWING_FUNCTIONS

# Attribution types renamed in a previous version, mapped to their current name, so that
# stale configuration files (e.g. from checkpoints exported before the rename) get an
# actionable error message instead of a generic "unsupported type" one.
_RENAMED_ATTRIBUTION_TYPES = {"cubic": "cubic_upsampling"}


def check_attribution_type(
    attribution_type: str, supported_methods: tuple[str, ...], config_file: Path | None = None
) -> str:
    r"""Validates that an attribution type is supported by a depictor class, raising a clear error naming the
    configuration file. Types renamed in a previous version are still accepted: a warning is logged and the
    current name is returned instead.

    Args:
        attribution_type (str): Attribution type read from a configuration file.
        supported_methods (tuple): Attribution types supported by the target depictor class.
        config_file (Path, optional): Path to the configuration file, if any, used to build the error message.
            Default: None.

    Returns:
        The resolved attribution type, identical to attribution_type unless it was renamed in a previous version.
    """
    if attribution_type in supported_methods:
        return attribution_type
    location = f" in {config_file}" if config_file is not None else ""
    if attribution_type in _RENAMED_ATTRIBUTION_TYPES:
        resolved_type = _RENAMED_ATTRIBUTION_TYPES[attribution_type]
        logger.warning(
            f"Attribution type '{attribution_type}'{location} was renamed to '{resolved_type}'. "
            "Update the configuration file to silence this warning."
        )
        return resolved_type
    raise NotImplementedError(
        f"Unknown attribution type '{attribution_type}'{location}. Supported types: {', '.join(supported_methods)}."
    )


def check_view_type(view_type: str, config_file: Path | None = None) -> None:
    r"""Validates that a view type is supported, raising a clear error naming the configuration file.

    Args:
        view_type (str): View type read from a configuration file.
        config_file (Path, optional): Path to the configuration file, if any, used to build the error message.
            Default: None.
    """
    if view_type in SUPPORTED_VIEWING_FUNCTIONS:
        return
    location = f" in {config_file}" if config_file is not None else ""
    raise NotImplementedError(
        f"Unknown view type '{view_type}'{location}. Supported types: {', '.join(SUPPORTED_VIEWING_FUNCTIONS)}."
    )


class ProtoDepictor(ABC):
    r"""Base class for *depictors*.

    A depictor generates and saves interpretations of a prototype to a human-readable format (image, audio ...).

    Attributes:
        config_file: Path to the configuration file used to build this object, if any.
        transform: Preprocessing transform applied to raw input. Must be set to (lambda x: x) if no transform.
    """

    DEFAULT_VISUALIZATION_CONFIG = Path("visualization.yml")

    # Supported attribution methods (overridden by subclasses)
    SUPPORTED_ATTRIBUTION_METHODS: tuple[str, ...] = ()

    config_file: Path | None

    transform: Callable

    @property
    @abstractmethod
    def extension(self) -> str:
        r"""File extension for output files (without dot).

        Returns:
            File extension without a leading dot.
        """
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        raw_input: Any,
        folder: Path,
        filename: str,
        proto_idx: int,
        device: str | torch.device,
        location: tuple[int, int] | str | None = None,
    ) -> Path:
        r"""Generates and saves a visualization.

        Args:
            raw_input (Any): Raw input data (e.g., PIL Image, audio tensor).
            folder (Path): Output directory.
            filename (str): Filename without extension.
            proto_idx (int): Prototype index.
            device (str | device): Hardware device.
            location (tuple[int, int] | str | None, optional): Location inside the similarity map.
                Default: None.

        Returns:
            Path to the saved file.
        """
        raise NotImplementedError

    @staticmethod
    def create_parser(
        parser: argparse.ArgumentParser | None = None,
        mandatory_config: bool = False,
    ) -> argparse.ArgumentParser:
        r"""Creates the argument parser for a depictor.

        Args:
            parser (ArgumentParser, optional): Existing parser (if any). Default: None.
            mandatory_config (bool, optional): If True, makes the configuration mandatory. Default: False.

        Returns:
            The parser itself.
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="Build a Depictor")
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
    def build_from_config(
        config: Path | dict[str, Any], model: CaBRNet, dataset_config: dict[str, Any]
    ) -> "ProtoDepictor":
        r"""Builds a depictor from a configuration file or dictionary.

        Args:
            config (Path | dict): Path to configuration file or dictionary.
            model (CaBRNet): Target model.
            dataset_config (dict): Dataset configuration dictionary.

        Returns:
            Depictor instance.
        """
        from cabrnet.core.utils.parser import load_config

        if isinstance(config, Path):
            logger.info(f"Loading depictor from {config}.")
            config_dict = load_config(config)
            config_path = config
        else:
            config_dict = config
            config_path = None

        # Get depictor type, default to similarity
        depictor_module = config_dict.pop("module", "cabrnet.core.visualization.visualizer")
        depictor_classname = config_dict.pop("type", None) or config_dict.get("name", None) or "SimilarityVisualizer"
        # Import and build the appropriate depictor
        module = importlib.import_module(depictor_module)
        depictor_class: ProtoDepictor = getattr(module, depictor_classname)

        result = depictor_class.build_from_config(config_dict, model=model, dataset_config=dataset_config)
        result.config_file = config_path
        return result
