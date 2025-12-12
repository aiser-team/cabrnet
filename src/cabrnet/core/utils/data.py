"""This file holds all the necessary functions to create datasets and dataloaders from configuration files."""

import argparse
import copy
import importlib
import random
from typing import Any, Callable

import torch
import torchvision.transforms
from cabrnet.core.utils.parser import load_config
from cabrnet.core.utils.transform import load_transform, TRANSFORM_FIELDS
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Subset


# Custom collate functions
def concat_collate(data: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Collate function using concatenation. Used in PIPNet.

    Args:
        data (list of tuples): Input data, in the form [((a1,a2),y1),((b1,b2),y2), ...]

    Returns:
        A tuple of tensors [a1|a2, b1|b2, ...], [y1, y2, ....].
    """
    xs, ys = zip(*data)
    xs1, xs2 = zip(*xs)
    return torch.cat([torch.stack(xs1), torch.stack(xs2)]), torch.tensor(ys)


SUPPORTED_COLLATE_FUNCTIONS = {"concat_collate": concat_collate}


class VisionDatasetSubset(Subset):
    r"""Overwrites the Subset class so that it exposes all properties of a VisionDataset."""

    @property
    def transform(self) -> Any:
        r"""Returns the 'transform' function of the original dataset."""
        return getattr(self.dataset, "transform", None)

    def target_transform(self) -> Any:
        r"""Returns the 'target_transform' function of the original dataset."""
        return getattr(self.dataset, "target_transform", None)

    def transforms(self) -> Any:
        r"""Returns the 'transforms' function of the original dataset."""
        return getattr(self.dataset, "transforms", None)


class DatasetManager:
    r"""Class for handling datasets in CaBRNet."""

    DEFAULT_DATASET_CONFIG: str = "dataset.yml"

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
    def get_datasets(
        config: str | dict[str, Any], sampling_ratio: int = 1, load_segmentation: bool = False
    ) -> dict[str, dict[str, Dataset | int | bool]]:
        r"""Loads datasets from a configuration file.

        Args:
            config (str, dict): Path to configuration file, or configuration dictionary.
            sampling_ratio (int, optional): Sampling ratio (e.g. 5 means only one image in five is used). Default: 1.
            load_segmentation (bool, optional): If True, loads segmentation datasets if available. Default: False.

        Returns:
            Dictionary datasets:
                - dataset: Dataset with data preprocessing
                - raw_dataset: Dataset without data preprocessing
                - seg_dataset: If load_segmentation is True, dataset of segmentation masks without data preprocessing
        Raises:
            ValueError whenever a dataset could not be loaded.
        """
        if not isinstance(config, (str, dict)):
            raise ValueError(f"Unsupported configuration format: {type(config)}")
        if isinstance(config, str):
            config = load_config(config)
        if sampling_ratio > 1:
            logger.warning(f"{'=' * 20} SAMPLING RATIO > 1: PROCESSING 1/{sampling_ratio} IMAGES {'=' * 20}")
        datasets: dict[str, dict[str, Dataset | int | bool]] = {}

        # Configuration should include at least train and projection sets
        mandatory_sets = ["train_set", "projection_set", "test_set"]
        for dataset_name in mandatory_sets:
            if dataset_name not in config:
                logger.error(f"Missing configuration for {dataset_name}.")

        for dataset_name in config:
            dataset: dict[str, Dataset | int | bool] = {}
            logger.info(f"Loading dataset {dataset_name}")
            dconfig = config[dataset_name]
            for key in ["name", "module", "params", "batch_size", "shuffle"]:
                if key not in dconfig:
                    raise ValueError(f"Missing dataset {key} information")

            params = copy.copy(dconfig["params"])
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

            # Handle Deterministic Partitioning (Splitting Train into Train/Val)
            if "partition" in dconfig:
                start_frac, end_frac = dconfig["partition"]
                total_len = len(dataset["dataset"])
                
                # Create the full list of indices
                indices = list(range(total_len))
                
                # Deterministic Shuffle if requested
                if "partition_seed" in dconfig:
                    seed = dconfig["partition_seed"]
                    logger.info(f"Shuffling {dataset_name} indices with seed {seed} before partitioning.")
                    # Use a local Random instance to avoid affecting global state
                    random.Random(seed).shuffle(indices)
                
                # Calculate integer slice points
                start_idx = int(start_frac * total_len)
                end_idx = int(end_frac * total_len)
                
                # Select the specific indices for this split
                selected_indices = indices[start_idx:end_idx]
                
                logger.info(f"Partitioning {dataset_name}: using range [{start_frac}-{end_frac}] ({len(selected_indices)} samples).")
                
                # Apply subsetting to all loaded dataset variants (main, raw, seg)
                for key in ["dataset", "raw_dataset", "seg_dataset"]:
                    if dataset.get(key) is not None:
                        dataset[key] = VisionDatasetSubset(dataset[key], selected_indices)
            
            if sampling_ratio > 1:
                # Apply data sub-selection
                selected_indices = [idx for idx in range(len(dataset["dataset"]))][::sampling_ratio]
                for key in ["dataset", "raw_dataset", "seg_dataset"]:
                    if dataset.get(key) is not None:
                        dset = dataset[key]
                        if not isinstance(dset, Dataset):
                            raise TypeError(f"{dataset[key]} should be a dataset, but is of type {type(dataset[key])}")
                        dataset[key] = VisionDatasetSubset(dset, selected_indices)

            datasets[dataset_name] = dataset
        return datasets

    @staticmethod
    def get_dataloaders(
        config: str | dict[str, Any], sampling_ratio: int = 1, load_segmentation: bool = False
    ) -> dict[str, DataLoader]:
        r"""Creates dataloaders from a configuration file.

        Args:
            config (str, dict): Path to configuration file, or configuration dictionary.
            sampling_ratio (int, optional): Sampling ratio (e.g. 5 means only one image in five is used). Default: 1.
            load_segmentation (bool, optional): If True, loads segmentation datasets if available. Default: False.

        Returns:
            Dictionary of dataloaders.

        Raises:
            ValueError whenever a dataset could not be loaded or a parameter is invalid.
        """
        if not isinstance(config, (str, dict)):
            raise ValueError(f"Unsupported configuration format: {type(config)}")
        if isinstance(config, str):
            config = load_config(config)

        datasets = DatasetManager.get_datasets(
            config=config, sampling_ratio=sampling_ratio, load_segmentation=load_segmentation
        )
        dataloaders: dict[str, DataLoader] = {}

        def _safe_item_load(item, t):
            if not isinstance(item, t):
                raise TypeError(f"{item} is of type {type(item)} but should be of type {t}.")
            return item

        for dataset_name in datasets:
            dataset = _safe_item_load(datasets[dataset_name]["dataset"], Dataset)
            raw_dataset = _safe_item_load(datasets[dataset_name]["raw_dataset"], Dataset)

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
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                drop_last=drop_last,
                pin_memory=pin_memory,
            )
            dataloaders[dataset_name + "_raw"] = DataLoader(
                dataset=raw_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
            )
            if load_segmentation:
                try:
                    seg_dataset = _safe_item_load(datasets[dataset_name]["seg_dataset"], Dataset)
                    dataloaders[dataset_name + "_seg"] = DataLoader(
                        dataset=seg_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
                    )
                except KeyError:
                    pass
        return dataloaders

    @staticmethod
    def get_dataset_transform(
        config: str | dict[str, Any], dataset: str = "test_set", keyword: str = "transform"
    ) -> Callable | None:
        r"""Returns the transform function associated with a given dataset.

        Args:
            config (str | dict): Path to configuration file, or configuration dictionary.
            dataset (str, optional): Name of target dataset. Default: test_set.
            keyword (str, optional): Name of the transform keyword. Default: transform.

        Returns:
            Transform function (if any).

        Raises:
            ValueError whenever the configuration is incorrect.
        """
        if isinstance(config, str):
            config = load_config(config)
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
