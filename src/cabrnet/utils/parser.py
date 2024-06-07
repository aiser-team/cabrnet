"""This file contains all the necessary tools to parse the various config files."""

import os

import yaml


def load_config(config_file: str) -> dict:
    r"""Loads a configuration file in YAML format for CaBRNet.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        The properly loaded config file.

    Raises:
        ValueError: The config file is in an unsupported file format.
    """
    file_ext = os.path.splitext(config_file)[-1].lower()

    if file_ext in [".yml", ".yaml"]:
        with open(config_file, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    raise ValueError(f"Error opening {config_file}. Unsupported file format, only YAML is supported.")
