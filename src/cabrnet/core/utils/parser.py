"""This file contains all the necessary tools to parse the various config files."""

import os

import torch  # necessary in case a dtype is torch.something
import yaml


def dtype_safe(d: dict) -> dict:
    r"""Makes a yaml dictionary 'safe' by replacing the string associated with key 'dtype'
    with the class that is represented by this string.

    Args:
        d (dict): Dictionary yaml.

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
            return dtype_safe(yaml.safe_load(file))
    raise ValueError(f"Error opening {config_file}. Unsupported file format, only YAML is supported.")
