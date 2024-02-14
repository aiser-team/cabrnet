"""This file contains all the necessary tools to parse the various config files."""

import argparse
import os
import yaml


def load_config(config_file: str) -> dict:
    """Load a configuration file for CaBRNet.

    Args:
        config_file: Path to the configuration file.

    Returns:
        The properly loaded config file.

    Raises:
        ValueError: The config file is in an unsupported file format.
    """
    file_ext = os.path.splitext(config_file)[-1].lower()

    if file_ext in [".yml", ".yaml"]:
        with open(config_file, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    raise ValueError("Unsupported file format, only YAML is supported.")


def create_training_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Create the argument parser for CaBRNet training configuration.

    Args:
        parser: Existing parser (if any)
    Returns:
        The parser itself.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Load datasets.")

    parser.add_argument(
        "--training",
        "-t",
        type=str,
        metavar="/path/to/file.yml",
        help="Path to the training configuration file",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        metavar="/path/to/checkpoint/directory",
        help="Path to existing checkpoint directory",
    )
    parser.add_argument(
        "--training-dir",
        type=str,
        required=True,
        metavar="path/to/training/directory",
        help="Path to output directory",
    )
    parser.add_argument(
        "--save-best",
        type=str,
        required=False,
        choices=["acc", "loss"],
        default="acc",
        metavar="metric",
        help="Save best model based on accuracy or loss",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        required=False,
        metavar="num_epochs",
        help="Checkpoint frequency (in epochs)",
    )
    return parser
