from typing import Any
import importlib
import torchvision

# Common torchvision datasets use one of the following keywords to indicate data transformations.
TRANSFORM_FIELDS = ["transform", "target_transform"]

# Type of transformation compositions
TRANSFORM_COMPOSITIONS = ["Compose", "RandomOrder", "RandomChoice"]


def load_transform(config: dict[str, Any] | list[dict[str, Any]]) -> Any:
    r"""Builds a data transform from a dictionary or a list of dictionaries.

    Args:
        config (dictionary | list): Transform configuration.

    Returns:
        Transform function, or list of functions.

    Raises:
        ValueError whenever the configuration is incorrect.
    """
    # Unit transform
    if isinstance(config, dict) and "type" in config:
        # Load custom module if necessary
        module = importlib.import_module(config["module"]) if "module" in config else torchvision.transforms

        # Check for recursive types
        if config["type"] in TRANSFORM_COMPOSITIONS:
            if "transforms" not in config:
                raise ValueError(f"Missing <transforms> field for operation {config['type']}.")
            ops = load_transform(config["transforms"])
            return getattr(module, config["type"])(ops)

        if "params" in config:
            return getattr(module, config["type"])(**config["params"])
        return getattr(module, config["type"])()

    # Convert dictionary to list, ignoring keys
    if isinstance(config, dict):
        config = [value for value in config.values()]

    # Process each transform independently
    ops = [load_transform(c) for c in config]
    if len(ops) == 1:
        ops = ops[0]
    return ops
