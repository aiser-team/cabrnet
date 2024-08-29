from typing import Any


class ArgumentError(Exception):
    r"""Dedicated exception raised when parsing CaBRNet command-line arguments."""
    pass


def check_mandatory_fields(config_dict: dict[str, Any], mandatory_fields: list[str], location: str | None = None):
    r"""Checks that the list of mandatory fields appear in the specified config dict and, if not, raises an error
    pointing to the specified location. Ideally, the configuration dictionary is only accessed after the sanity check
    has been performed to avoid ugly errors.

    Args:
        config_dict (dict): Configuration dictionary where the fields are searched for.
        mandatory_fields (list): List of fields that ought to be checked.
        location (str, optional): If given, location that should be written in the error message. Default: None.
    """
    for field in mandatory_fields:
        if field not in config_dict:
            raise ValueError(f"Missing mandatory field {field}" f"{'in {location}' if location is not None else ''}")
