import argparse
import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import torch
from loguru import logger

from cabrnet.archs.generic.model import CaBRNet


class Depictor(ABC):
    r"""Base class for visualization depictors.

    A depictor generates and saves visualizations for model interpretations.
    """

    DEFAULT_VISUALIZATION_CONFIG = Path("visualization.yml")

    # Supported attribution methods (overridden by subclasses)
    SUPPORTED_ATTRIBUTION_METHODS: tuple[str, ...] = ()

    config_file: Path | None

    @property
    @abstractmethod
    def extension(self) -> str:
        r"""File extension for output files (without dot)."""
        pass

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
            raw_input: Raw input data (e.g., PIL Image, audio tensor).
            folder: Output directory.
            filename: Filename without extension.
            proto_idx: Prototype index.
            device: Hardware device.
            location: Optional location inside the similarity map.

        Returns:
            Path to the saved file.
        """
        pass

    @staticmethod
    def create_parser(
        parser: argparse.ArgumentParser | None = None,
        mandatory_config: bool = False,
    ) -> argparse.ArgumentParser:
        r"""Creates the argument parser for a depictor.

        Args:
            parser: Existing parser (if any). Default: None.
            mandatory_config: If True, makes the configuration mandatory. Default: False.

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
    @abstractmethod
    def build_from_config(config: Path | dict[str, Any], model: CaBRNet, transform: Callable) -> "Depictor":
        r"""Builds a depictor from a configuration file or dictionary.

        Args:
            config: Path to configuration file or dictionary.
            model: Target model.

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
        depictor_class: Depictor = getattr(module, depictor_classname)

        result = depictor_class.build_from_config(config_dict, model=model, transform=transform)
        result.config_file = config_path
        return result
