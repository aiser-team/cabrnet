"""Implements the saving and loading capabilities for a CaBRNet model."""

import csv
from pathlib import Path
import pickle
import random
import shutil
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
import torch
import yaml

from cabrnet.archs.custom_extractors.onnx_backbone import GenericONNXModel
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.optimizers import OptimizerManager


def safe_copy(src: Path, dst: Path) -> None:
    r"""Copies a file to a given destination, ignoring copies of a file onto itself.

    Args:
        src (Path): Path to source file.
        dst (Path): Path to destination file.
    """
    try:
        shutil.copyfile(src=src, dst=dst)
    except shutil.SameFileError:
        logger.warning(f"Ignoring file copy from {src} to itself.")
        pass


def save_checkpoint(
    directory_path: Path,
    model: CaBRNet,
    model_arch: Path | dict[str, Any],
    optimizer_mngr: OptimizerManager | None,
    training_config: Path | dict[str, Any] | None,
    dataset_config: Path | dict[str, Any],
    projection_info: list[dict] | None,
    epoch: int | str,
    seed: int | None,
    device: str | torch.device,
    stats: dict[str, Any] | None = None,
) -> None:
    r"""Saves everything needed to restart a training process.

    Args:
        directory_path (Path): Target location.
        model (Module): CaBRNet model.
        model_arch (Path|dict): Path to the model configuration file, or configuration dictionary.
        optimizer_mngr (OptimizerManager): Optimizer manager.
        training_config (Path|dict): Path to the training configuration file, or configuration dictionary.
        dataset_config (Path|dict): Path to the dataset configuration file, or configuration dictionary.
        projection_info (list, optional): Projection information, generated during training epilogue.
        epoch (int or str): Current epoch.
        seed (int): Initial random seed (recorded for reproducibility).
        device (str | device): Hardware device (recorded for reproducibility).
        stats (dictionary, optional): Other optional statistics. Default: None.
    """
    directory_path.mkdir(parents=True, exist_ok=True)

    model.eval()

    torch.save(model.state_dict(), directory_path / CaBRNet.DEFAULT_MODEL_STATE)
    model.export_arch(directory_path)  # Export auxiliary infos if necessary

    if optimizer_mngr is not None:
        torch.save(optimizer_mngr.state_dict(), directory_path / OptimizerManager.DEFAULT_TRAINING_STATE)
    if isinstance(model_arch, Path):
        safe_copy(src=model_arch, dst=directory_path / CaBRNet.DEFAULT_MODEL_CONFIG)
    else:
        with open(directory_path / CaBRNet.DEFAULT_MODEL_CONFIG, "w") as fout:
            # Save dictionary to file
            yaml.dump(model_arch, fout, sort_keys=False)
    if training_config is not None:
        if isinstance(training_config, Path):
            safe_copy(src=training_config, dst=directory_path / OptimizerManager.DEFAULT_TRAINING_CONFIG)
        else:
            with open(directory_path / OptimizerManager.DEFAULT_TRAINING_CONFIG, "w") as fout:
                # Save dictionary to file
                yaml.dump(training_config, fout, sort_keys=False)

    if isinstance(dataset_config, Path):
        safe_copy(src=dataset_config, dst=directory_path / DatasetManager.DEFAULT_DATASET_CONFIG)
    else:
        with open(directory_path / DatasetManager.DEFAULT_DATASET_CONFIG, "w") as fout:
            # Save dictionary to file
            yaml.dump(dataset_config, fout, sort_keys=False)

    state = {
        "random_generators": {
            "torch_rng_state": torch.random.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        "epoch": epoch,
        "seed": seed,
        "device": str(device),
        "stats": stats,
    }

    with open(directory_path / "state.pickle", "wb") as file:
        pickle.dump(state, file)  # type: ignore

    # Save projection information if it exists
    if projection_info:
        save_projection_info(projection_info, directory_path / CaBRNet.DEFAULT_PROJECTION_INFO)

    # Save ONNX model if it exists
    # Important assertion: the same onnx model is used for all pipelines
    if model.extractor.arch_name == "generic_onnx_model":
        extractor_pipelines = list(model.extractor.named_modules())
        _, extractor_model = extractor_pipelines[1]
        assert isinstance(extractor_model, GenericONNXModel)
        original_path = extractor_model.get_original_onnx_model_path()
        saved_path_dir = directory_path / original_path.parent
        saved_path = saved_path_dir / original_path.name
        saved_path_dir.mkdir(parents=True, exist_ok=True)
        safe_copy(src=original_path, dst=saved_path)
        logger.info(f"Successfully saved ONNX model at {saved_path}.")

    logger.info(f"Successfully saved checkpoint at epoch {epoch}.")


def load_checkpoint(
    directory_path: Path,
    model: CaBRNet,
    optimizer_mngr: OptimizerManager | None = None,
) -> dict[str, Any]:
    r"""Restores training process using checkpoint directory.

    Args:
        directory_path (Path): Target location.
        model (Module): CaBRNet mode.
        optimizer_mngr (OptimizerManager, optional): Optimizer manager. Default: None.

    Returns:
        Dictionary containing auxiliary state information (epoch, seed, device, stats).
    """
    if not directory_path.is_dir():
        raise ValueError(f"Unknown checkpoint directory {directory_path}")

    model.load_state_dict(
        torch.load(directory_path / CaBRNet.DEFAULT_MODEL_STATE, map_location="cpu", weights_only=True)
    )
    model.import_arch(directory_path)  # Import auxiliary infos if necessary

    if optimizer_mngr is not None:
        optimizer_state_path = directory_path / OptimizerManager.DEFAULT_TRAINING_STATE
        if optimizer_state_path.is_file():
            optimizer_mngr.load_state_dict(torch.load(optimizer_state_path, map_location="cpu", weights_only=True))
        else:
            logger.warning(f"Could not find optimizer state {optimizer_state_path}. Using default state instead.")

    # Restore RNG state
    with open(directory_path / "state.pickle", "rb") as file:
        state = pickle.load(file)

    torch_rng = state.get("random_generators").get("torch_rng_state")
    numpy_rng = state.get("random_generators").get("numpy_rng_state")
    python_rng = state.get("random_generators").get("python_rng_state")
    torch.random.set_rng_state(torch_rng)
    np.random.set_state(numpy_rng)
    random.setstate(python_rng)

    epoch = state.get("epoch")
    stats = state.get("stats")
    seed = state.get("seed")
    device = state.get("device")

    logger.info(f"Successfully loaded checkpoint from epoch {epoch}.")

    return {
        "epoch": epoch,
        "seed": seed,
        "device": device,
        "stats": stats,
    }


def save_projection_info(projection_info: list[dict], filename: Path) -> None:
    r"""Saves projection information, either in pickle or CSV format.

    Args:
        projection_info (list): Projection list, generated during training epilogue.
        filename (Path): Path to output file. Based on the file extension, the file is stored in
          pickle format (pickle or pkl extension) or CSV format (any other extension).
    """
    if filename.suffix.lower() in [".pickle", ".pkl"]:
        with open(filename, "wb") as f:
            pickle.dump(projection_info, f)
    else:
        # CSV format
        with open(filename, "w") as f:
            # Extract fields from first entry in the dictionary
            fields = projection_info[0].keys()
            writer = csv.DictWriter(f, fieldnames=list(fields))
            writer.writeheader()
            for proto_info in projection_info:
                writer.writerow(proto_info)


def load_projection_info(filename: Path) -> list[dict]:
    r"""Loads projection information, either in pickle or CSV format.

    Args:
        filename (Path): Path to input file. Based on the file extension, the file is loaded in
          pickle format (pickle or pkl extension) or CSV format (any other extension).

    Returns:
        Projection dictionary, generated during training epilogue.
    """
    if not filename.is_file():
        raise FileNotFoundError(f"Could not find projection information file {filename}")
    if filename.suffix.lower() in [".pickle", ".pkl"]:
        with open(filename, "rb") as file:
            projection_info = pickle.load(file)
    else:
        projection_info = pd.read_csv(filename).to_dict(orient="records")
    return projection_info
