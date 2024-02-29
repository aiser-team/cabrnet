"""This file holds all the necessary functions to create datasets and dataloaders from configuration files."""

import argparse
import importlib
from typing import Any, Callable

import torchvision.transforms
from loguru import logger
from cabrnet.utils.parser import load_config
from torch.utils.data import DataLoader, Dataset


def create_dataset_parser(
    parser: argparse.ArgumentParser | None = None, mandatory_config: bool = False
) -> argparse.ArgumentParser:
    """Create the argument parser for CaBRNet datasets.

    Args:
        parser: Existing parser (if any)
        mandatory_config: Make dataset configuration mandatory

    Returns:
        The parser itself.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Load datasets.")
    parser.add_argument(
        "--dataset",
        "-d",
        required=mandatory_config,
        metavar="/path/to/file.yml",
        help="path to the dataset config",
    )
    return parser


# Common torchvision datasets use one of the following keywords to indicate data transformations.
TRANSFORM_FIELDS = ["transform", "target_transform"]

# Type of transformation compositions
TRANSFORM_COMPOSITIONS = ["Compose", "RandomOrder", "RandomChoice"]


def get_transform(trans_config: dict[str, Any]) -> Callable:
    """Build a data transformation from a dictionary.

    Args:
        trans_config: Transformation configuration
    Returns:
        transformation
    Raises:
        ValueError whenever the configuration is incorrect.
    """
    if "type" not in trans_config:
        raise ValueError("Missing transformation type.")
    # Load custom module if necessary
    module = importlib.import_module(trans_config["module"]) if "module" in trans_config else torchvision.transforms
    # Check for recursive types
    if trans_config["type"] in TRANSFORM_COMPOSITIONS:
        if "transforms" not in trans_config:
            raise ValueError(f"Missing <transforms> field for operation {trans_config['type']}.")
        ops = [get_transform(trans_config["transforms"][op_name]) for op_name in trans_config["transforms"]]
        return getattr(module, trans_config["type"])(ops)

    if "params" in trans_config:
        return getattr(module, trans_config["type"])(**trans_config["params"])
    return getattr(module, trans_config["type"])()


def get_datasets(config_file: str) -> dict[str, dict[str, Dataset | int | bool]]:
    """Load datasets from yaml configuration file.

    Args:
        config_file: path to configuration file

    Returns:
        dictionary of datasets with their respective batch size and shuffle property

    Raises:
        ValueError whenever a dataset could not be loaded
    """
    config = load_config(config_file)
    datasets: dict[str, dict[str, Dataset | int | bool]] = {}

    # Configuration should include at least train and projection sets
    mandatory_sets = ["train_set", "projection_set", "test_set"]
    for dataset_name in mandatory_sets:
        if dataset_name not in config:
            logger.error(f"Missing configuration for {dataset_name}.")

    for dataset_name in config:
        dataset: dict[str, Dataset | int | bool] = {}
        logger.info(f"Loading dataset {dataset_name} from file {config_file}")
        dconfig = config[dataset_name]
        for key in ["name", "module", "params", "batch_size", "shuffle"]:
            if key not in dconfig:
                raise ValueError(f"Missing dataset {key} information")

        params = dconfig["params"]
        for field in params:
            if field in TRANSFORM_FIELDS:
                # Replace configuration with actual transformation function
                ops = [get_transform(params[field][op_name]) for op_name in params[field]]
                ops = torchvision.transforms.Compose(ops) if len(ops) > 1 else ops[0]
                logger.debug(f"{field}: {ops}")
                params[field] = ops
            else:
                # Add parameter as is
                params[field] = params[field]

        batch_size: int = dconfig["batch_size"]
        if batch_size < 1:
            raise ValueError(f"Invalid batch size: {batch_size}.")
        shuffle: bool = dconfig["shuffle"]

        # Load dataset
        module = importlib.import_module(dconfig["module"])
        dataset["dataset"] = getattr(module, dconfig["name"])(**params)
        if "transform" in params:
            # Remove image preprocessing to recover raw images
            params["transform"] = None
        dataset["raw_dataset"] = getattr(module, dconfig["name"])(**params)
        dataset["batch_size"] = batch_size
        dataset["shuffle"] = shuffle
        # Recover optional number of workers
        dataset["num_workers"] = dconfig.get("num_workers", 0)
        datasets[dataset_name] = dataset
    return datasets


def get_dataloaders(config_file: str) -> dict[str, DataLoader]:
    """Create dataloaders from yaml configuration file.

    Args:
        config_file: path to configuration file

    Returns:
        dictionary of dataloaders

    Raises:
        ValueError whenever a dataset could not be loaded or a parameter is invalid
    """
    datasets = get_datasets(config_file=config_file)
    dataloaders: dict[str, DataLoader] = {}
    for dataset_name in datasets:
        dataset: Dataset = datasets[dataset_name]["dataset"]  # type: ignore
        raw_dataset: Dataset = datasets[dataset_name]["raw_dataset"]  # type: ignore
        batch_size: int = datasets[dataset_name]["batch_size"]  # type: ignore
        shuffle: bool = datasets[dataset_name]["shuffle"]  # type: ignore
        num_workers: int = datasets[dataset_name]["num_workers"]  # type: ignore
        dataloaders[dataset_name] = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        dataloaders[dataset_name + "_raw"] = DataLoader(
            dataset=raw_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
    return dataloaders


def get_dataset_transform(config_file: str, dataset: str = "test_set") -> Callable | None:
    """Return transform function associated with a given dataset

    Args:
        config_file: path to configuration file
        dataset: name of target dataset

    Returns:
        transform function if any

    Raises:
        ValueError whenever the configuration is incorrect.
    """
    config = load_config(config_file)
    if dataset not in config:
        raise ValueError(f"Missing configuration for dataset {dataset} in file {config_file}.")
    if "params" not in config[dataset]:
        raise ValueError(f"Missing parameters for dataset {dataset}.")
    if "transform" not in config[dataset]["params"]:
        return None
    transform_config = config[dataset]["params"]["transform"]

    ops = [get_transform(transform_config[op_name]) for op_name in transform_config]
    ops = torchvision.transforms.Compose(ops) if len(ops) > 1 else ops[0]
    logger.debug(f"Transform function for dataset {dataset}: {ops}")
    return ops
