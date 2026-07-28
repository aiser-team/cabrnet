"""This file holds all the necessary functions to create datasets and dataloaders from configuration files."""

import argparse
import copy
import importlib
import os
import random
from collections.abc import Sized
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torchvision.transforms
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Sampler

from cabrnet.core.utils.parser import load_config
from cabrnet.core.utils.transform import TRANSFORM_FIELDS, load_transform


# Custom collate functions
def concat_collate(data: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Collate function using concatenation. Used in PIPNet.

    Args:
        data (list of tuples): Input data, in the form [((a1,a2),y1),((b1,b2),y2), ...].

    Returns:
        A tuple of tensors [a1|a2, b1|b2, ...], [y1, y2, ....].
    """
    xs, ys = zip(*data)
    xs1, xs2 = zip(*xs)
    return torch.cat([torch.stack(xs1), torch.stack(xs2)]), torch.tensor(ys)


SUPPORTED_COLLATE_FUNCTIONS = {"concat_collate": concat_collate}
DATA_ROOT_VARENV = "CABRNET_DATA_ROOT"


# Dataset roles that can be declared either as a single legacy entry or as a named collection.
_SINGULAR_TO_PLURAL = {"test_set": "test_sets", "val_set": "val_sets"}


def named_dataset_configs(dataset_cfg: dict, singular: str) -> dict[str, dict]:
    r"""Returns named dataset configurations for a role declared either as a single legacy entry or as a
    collection of named entries, using their internal loader names.

    A legacy config has one top-level entry (e.g. ``test_set``). A collection under the plural key (e.g.
    ``test_sets``) is flattened to ``<plural>/<name>`` for the dataloader and metric namespaces, while the
    user-facing YAML remains nested.

    Args:
        dataset_cfg (dict): Dataset configuration dictionary.
        singular (str): Legacy top-level key for this role ('test_set' or 'val_set').

    Returns:
        Mapping of internal loader name to dataset configuration, for each declared entry of this role.
    """
    plural = _SINGULAR_TO_PLURAL[singular]
    has_legacy_entry = singular in dataset_cfg
    has_collection = plural in dataset_cfg
    if has_legacy_entry and has_collection:
        raise ValueError(f"Specify either '{singular}' or '{plural}', not both.")
    if has_legacy_entry:
        return {singular: dataset_cfg[singular]}
    if not has_collection:
        return {}

    collection = dataset_cfg[plural]
    if not isinstance(collection, dict):
        raise TypeError(f"'{plural}' must be a mapping of dataset names to dataset configurations.")
    if not collection:
        raise ValueError(f"'{plural}' must contain at least one dataset configuration.")
    return {f"{plural}/{name}": config for name, config in collection.items()}


def flatten_dataset_collections(dataset_cfg: dict) -> dict:
    r"""Flattens named dataset collections to the format expected by ``DatasetManager``.

    Args:
        dataset_cfg (dict): Dataset configuration dictionary.

    Returns:
        Dataset configuration dictionary with named collections flattened to internal loader names.
    """
    reserved_keys = set(_SINGULAR_TO_PLURAL) | set(_SINGULAR_TO_PLURAL.values())
    flattened = {name: config for name, config in dataset_cfg.items() if name not in reserved_keys}
    for singular, plural in _SINGULAR_TO_PLURAL.items():
        for name, config in named_dataset_configs(dataset_cfg, singular).items():
            if name in flattened:
                raise ValueError(f"Duplicate dataset name after flattening {plural}: {name}")
            flattened[name] = config
    return flattened


class IndexSampler(Sampler[int]):
    r"""Sampler over resolved source-dataset indices.

    It is used to turn a dataset into a dataloader, managing subseting (selecting indices) and shuffling.
    This way, datasets are never modified.
    """

    def __init__(self, indices: np.ndarray, shuffle: bool) -> None:
        r"""Initializes the sampler.

        Args:
            indices (numpy.ndarray): Source-dataset indices to iterate over.
            shuffle (bool): Whether to randomize their order on each iteration.
        """
        self.indices = np.asarray(indices, dtype=np.intp)
        self.shuffle = shuffle
        self.iterated_indices = self.indices

    def __iter__(self):
        r"""Yields source-dataset indices in configured order."""
        if self.shuffle:
            order = torch.randperm(len(self.indices)).tolist()
            self.iterated_indices = self.indices[order]
        else:
            self.iterated_indices = self.indices
        yield from (int(index) for index in self.iterated_indices)

    def __len__(self) -> int:
        r"""Returns the number of selected indices.

        Returns:
            Number of selected source-dataset indices.
        """
        return len(self.indices)


def get_dataloader_indices(dataloader: DataLoader) -> np.ndarray:
    r"""Returns the source-dataset indices in a dataloader's current sampling order.

    Args:
        dataloader (DataLoader): Dataloader to inspect.

    Returns:
        Source-dataset indices in sampling order.

    Raises:
        TypeError: If the dataloader's dataset has no defined length.
    """
    if isinstance(dataloader.sampler, IndexSampler):
        return dataloader.sampler.iterated_indices
    if not isinstance(dataloader.dataset, Sized):
        raise TypeError("Projection requires a map-style dataset with a defined length.")
    return np.arange(len(dataloader.dataset), dtype=np.intp)


class DatasetManager:
    r"""Class for handling datasets in CaBRNet."""

    DEFAULT_DATASET_CONFIG = Path("dataset.yml")
    DATASET_ALTERNATIVE = [("--dataset", DEFAULT_DATASET_CONFIG)]  # Used by CaBRNet.create_checkpoint_parser

    @staticmethod
    def create_parser(
        parser: argparse.ArgumentParser | None = None, mandatory_config: bool = False
    ) -> argparse.ArgumentParser:
        r"""Creates the argument parser for CaBRNet datasets.

        Args:
            parser (ArgumentParser, optional): Existing parser (if any). Default: None.
            mandatory_config (bool, optional): If True, makes the configuration mandatory. Default: False.

        Returns:
            The parser itself.
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="Load datasets.")
        parser.add_argument(
            "-d",
            "--dataset",
            type=Path,
            required=mandatory_config,
            metavar="/path/to/file.yml",
            help="path to the dataset config",
        )
        parser.add_argument(
            "--sampling-ratio",
            type=int,
            required=False,
            default=1,
            metavar="ratio",
            help="data sampling ratio (e.g. 5 means only one image in five is used). Default: 1",
        )
        return parser

    @staticmethod
    @staticmethod
    def get_datasets_and_indices(
        config: Path | dict[str, Any], sampling_ratio: int = 1, load_segmentation: bool = False
    ) -> tuple[dict[str, dict[str, Dataset]], dict[str, np.ndarray]]:
        r"""Loads datasets and resolves their selected source indices.

        Args:
            config (Path, dict): Path to configuration file, or configuration dictionary.
            sampling_ratio (int, optional): Sampling ratio (e.g. 5 means only one image in five is used). Default: 1.
            load_segmentation (bool, optional): If True, loads segmentation datasets if available. Default: False.

        Returns:
            Unmodified datasets and selected source indices for each configured
            dataset.
        Raises:
            ValueError whenever a dataset could not be loaded.
        """
        if not isinstance(config, (Path, dict)):
            raise ValueError(f"Unsupported configuration format: {type(config)}")
        if isinstance(config, Path):
            config = load_config(config)
        has_test_collection = "test_sets" in config
        config = flatten_dataset_collections(config)
        if sampling_ratio < 1:
            raise ValueError(f"sampling_ratio must be at least 1, got {sampling_ratio}")
        if sampling_ratio > 1:
            logger.warning(f"{'=' * 20} SAMPLING RATIO > 1: PROCESSING 1/{sampling_ratio} IMAGES {'=' * 20}")
        datasets: dict[str, dict[str, Dataset]] = {}
        selected_indices: dict[str, np.ndarray] = {}

        # Legacy configurations expose one ``test_set``. New configurations
        # may provide a named ``test_sets`` collection instead.
        mandatory_sets = ["train_set", "projection_set"]
        if not has_test_collection:
            mandatory_sets.append("test_set")
        for dataset_name in mandatory_sets:
            if dataset_name not in config:
                logger.error(f"Missing configuration for {dataset_name}.")

        for dataset_name in config:
            # Top-level underscore-prefixed mappings provide YAML anchors and other
            # metadata; they are not dataset definitions.
            if str(dataset_name).startswith("_"):
                continue
            dataset: dict[str, Dataset] = {}
            logger.info(f"Loading dataset {dataset_name}")
            dconfig = config[dataset_name]
            for key in ["name", "module", "params", "batch_size", "shuffle"]:
                if key not in dconfig:
                    raise ValueError(f"Missing dataset {key} information")

            params = copy.copy(dconfig["params"])
            # Update the root location if necessary
            if "root" in params:
                root = params["root"]
                if not Path(root).is_dir():
                    if (data_folder := os.getenv(DATA_ROOT_VARENV)) is not None:
                        new_root = Path(data_folder) / root
                        if not Path(new_root).is_dir():
                            logger.error(
                                f"Folder '{root}' of dataset {dataset_name} not found. "
                                f"Did you wrongly set environment variable {DATA_ROOT_VARENV}?"
                            )
                        else:
                            params["root"] = new_root
                    else:
                        logger.error(
                            f"Folder '{root}' of dataset {dataset_name} not found. "
                            f"You could set environment variable {DATA_ROOT_VARENV}."
                        )
            for field, value in params.items():
                if field in TRANSFORM_FIELDS:
                    # Replace configuration with actual transform function
                    params[field] = DatasetManager.get_dataset_transform(config=config, dataset=dataset_name)

            # Load dataset
            module = importlib.import_module(dconfig["module"])
            dataset["dataset"] = getattr(module, dconfig["name"])(**params)
            if "transform" in params:
                # Remove image preprocessing to recover raw images
                params["transform"] = None
            dataset["raw_dataset"] = getattr(module, dconfig["name"])(**params)
            if load_segmentation:
                try:
                    params["root"] += "_seg"
                    dataset["seg_dataset"] = getattr(module, dconfig["name"])(**params)
                except FileNotFoundError:
                    logger.warning(f"Segmentation set unavailable for dataset {dataset_name}")

            dataset_to_select = dataset["dataset"]
            if not isinstance(dataset_to_select, Sized):
                raise TypeError(f"Dataset {dataset_name} does not define a length")
            total_len = len(dataset_to_select)
            indices = np.arange(total_len, dtype=np.intp)

            # Handle deterministic partitioning (splitting train into train/val).
            if "partition" in dconfig:
                start_frac, end_frac = dconfig["partition"]

                # Use the existing Python-random ordering for backwards-compatible partitions.
                if "partition_seed" in dconfig:
                    seed = dconfig["partition_seed"]
                    logger.info(f"Shuffling {dataset_name} indices with seed {seed} before partitioning.")
                    shuffled_indices = indices.tolist()
                    random.Random(seed).shuffle(shuffled_indices)
                    indices = np.asarray(shuffled_indices, dtype=np.intp)

                start_idx = int(start_frac * total_len)
                end_idx = int(end_frac * total_len)
                indices = indices[start_idx:end_idx]

                logger.info(
                    f"Partitioning {dataset_name}: using range [{start_frac}-{end_frac}] ({len(indices)} samples)."
                )

            if sampling_ratio > 1:
                indices = indices[::sampling_ratio]

            for variant_name, variant in dataset.items():
                if not isinstance(variant, Sized):
                    raise TypeError(f"Dataset variant {dataset_name}/{variant_name} does not define a length")
                if len(variant) != total_len:
                    raise ValueError(
                        f"Dataset variant {dataset_name}/{variant_name} has {len(variant)} samples, "
                        f"but the main dataset has {total_len}."
                    )

            datasets[dataset_name] = dataset
            selected_indices[dataset_name] = indices
        return datasets, selected_indices

    @staticmethod
    def get_dataloaders(
        config: Path | dict[str, Any], sampling_ratio: int = 1, load_segmentation: bool = False
    ) -> dict[str, DataLoader]:
        r"""Creates dataloaders from a configuration file.

        Args:
            config (Path, dict): Path to configuration file, or configuration dictionary.
            sampling_ratio (int, optional): Sampling ratio (e.g. 5 means only one image in five is used). Default: 1.
            load_segmentation (bool, optional): If True, loads segmentation datasets if available. Default: False.

        Returns:
            Dictionary of dataloaders.

        Raises:
            ValueError whenever a dataset could not be loaded or a parameter is invalid.
        """
        if not isinstance(config, (Path, dict)):
            raise ValueError(f"Unsupported configuration format: {type(config)}")
        if isinstance(config, Path):
            config = load_config(config)
        datasets, selected_indices = DatasetManager.get_datasets_and_indices(
            config=config, sampling_ratio=sampling_ratio, load_segmentation=load_segmentation
        )
        config = flatten_dataset_collections(config)
        dataloaders: dict[str, DataLoader] = {}

        def _safe_item_load(item, t):
            if not isinstance(item, t):
                raise TypeError(f"{item} is of type {type(item)} but should be of type {t}.")
            return item

        for dataset_name, variants in datasets.items():
            dataset = _safe_item_load(variants["dataset"], Dataset)
            raw_dataset = _safe_item_load(variants["raw_dataset"], Dataset)

            dconfig = config[dataset_name]
            for key in ["batch_size", "shuffle"]:
                if key not in dconfig:
                    raise ValueError(f"Missing dataset {key} information")

            # Dataloader parameters
            batch_size = _safe_item_load(dconfig["batch_size"], int)
            shuffle = _safe_item_load(dconfig["shuffle"], bool)
            num_workers = _safe_item_load(dconfig.get("num_workers", 0), int)
            drop_last = _safe_item_load(dconfig.get("drop_last", False), bool)
            pin_memory = _safe_item_load(dconfig.get("pin_memory", False), bool)
            # Optional collate function
            collate_fn = dconfig.get("collate_fn")
            if collate_fn:
                if collate_fn not in SUPPORTED_COLLATE_FUNCTIONS:
                    raise ValueError(f"Unsupported collate function {collate_fn}")
                collate_fn = SUPPORTED_COLLATE_FUNCTIONS[collate_fn]

            dataloaders[dataset_name] = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=IndexSampler(selected_indices[dataset_name], shuffle=shuffle),
                num_workers=num_workers,
                collate_fn=collate_fn,
                drop_last=drop_last,
                pin_memory=pin_memory,
            )
            dataloaders[dataset_name + "_raw"] = DataLoader(
                dataset=raw_dataset,
                batch_size=batch_size,
                sampler=IndexSampler(selected_indices[dataset_name], shuffle=shuffle),
                num_workers=num_workers,
            )
            if load_segmentation:
                try:
                    seg_dataset = _safe_item_load(variants["seg_dataset"], Dataset)
                    dataloaders[dataset_name + "_seg"] = DataLoader(
                        dataset=seg_dataset,
                        batch_size=batch_size,
                        sampler=IndexSampler(selected_indices[dataset_name], shuffle=shuffle),
                        num_workers=num_workers,
                    )
                except KeyError:
                    pass
        return dataloaders

    @staticmethod
    def named_dataloaders(dataloaders: dict[str, DataLoader], singular: str) -> list[str]:
        r"""Returns the names of all dataloaders for a role declared either as a single legacy entry or as a
        named collection (e.g. 'test_set'/'test_sets' or 'val_set'/'val_sets').

        Args:
            dataloaders (dict): Dictionary of dataloaders, as returned by get_dataloaders.
            singular (str): Legacy top-level key for this role ('test_set' or 'val_set').

        Returns:
            Names of the legacy loader and/or any collection loaders for this role, excluding their
            '_raw'/'_seg' companion loaders.
        """
        plural = _SINGULAR_TO_PLURAL[singular]
        return [
            name
            for name in dataloaders
            if (name == singular or name.startswith(f"{plural}/")) and not name.endswith(("_raw", "_seg"))
        ]

    @staticmethod
    def test_set_names(dataloaders: dict[str, DataLoader]) -> list[str]:
        r"""Returns the names of all test-set dataloaders in a dataloader dictionary.

        Args:
            dataloaders (dict): Dictionary of dataloaders, as returned by get_dataloaders.

        Returns:
            Names of the legacy 'test_set' loader and/or any 'test_sets/<name>' collection loaders.
        """
        return DatasetManager.named_dataloaders(dataloaders, "test_set")

    @staticmethod
    def val_set_names(dataloaders: dict[str, DataLoader]) -> list[str]:
        r"""Returns the names of all validation-set dataloaders in a dataloader dictionary.

        Args:
            dataloaders (dict): Dictionary of dataloaders, as returned by get_dataloaders.

        Returns:
            Names of the legacy 'val_set' loader and/or any 'val_sets/<name>' collection loaders.
        """
        return DatasetManager.named_dataloaders(dataloaders, "val_set")

    @staticmethod
    def get_dataset_transform(
        config: Path | dict[str, Any], dataset: str = "test_set", keyword: str = "transform"
    ) -> Callable | None:
        r"""Returns the transform function associated with a given dataset.

        Args:
            config (Path | dict): Path to configuration file, or configuration dictionary.
            dataset (str, optional): Name of target dataset. Default: test_set.
            keyword (str, optional): Name of the transform keyword. Default: transform.

        Returns:
            Transform function (if any).

        Raises:
            ValueError whenever the configuration is incorrect.
        """
        if isinstance(config, Path):
            config = load_config(config)
        config = flatten_dataset_collections(config)
        if dataset not in config:
            raise ValueError(f"Missing configuration for dataset {dataset} in {config}.")
        if "params" not in config[dataset]:
            raise ValueError(f"Missing parameters for dataset {dataset}.")
        if keyword not in config[dataset]["params"]:
            return None
        transform_config = config[dataset]["params"][keyword]
        ops = load_transform(transform_config)
        ops = torchvision.transforms.Compose(ops) if isinstance(ops, list) else ops
        logger.debug(f"Transform function for dataset {dataset}: {ops}")
        return ops
