"""This file contains all the necessary tools to parse the various config files."""
import torch  # necessary in case a dtype is torch.something
from pathlib import Path
import yaml


def dtype_safe(d: dict) -> dict:
    r"""Makes a YAML dictionary 'safe' by replacing the string associated with key 'dtype'
    with the class that is represented by this string.

    Args:
        d (dict): YAML dictionary.

    Returns:
        Modified dictionary.
    """
    if isinstance(d, dict):
        for k in d:
            if k == "dtype":  # Tries to replace the value d[k] with evaluation of d[k].
                try:
                    d[k] = eval(d[k])
                except:
                    pass
            d[k] = dtype_safe(d[k])
    return d


def load_config(config_file: Path) -> dict:
    r"""Loads a configuration file in YAML format for CaBRNet.

    Args:
        config_file (Path): Path to the configuration file.

    Returns:
        The properly loaded config file.

    Raises:
        ValueError: The config file is in an unsupported file format.
    """
    if config_file.suffix.lower() in [".yml", ".yaml"]:
        with open(config_file, "r", encoding="utf-8") as file:
            return dtype_safe(yaml.safe_load(file))
    raise ValueError(f"Error opening {config_file}. Unsupported file format, only YAML is supported.")
