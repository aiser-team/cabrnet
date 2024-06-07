"""Implements the saving and loading capabilities for a CaBRNet model."""

import csv
import os
import pickle
import random
import shutil
from typing import Any

import numpy as np
import pandas as pd
import torch
from cabrnet.generic.model import CaBRNet
from cabrnet.utils.data import DatasetManager
from cabrnet.utils.optimizers import OptimizerManager
from loguru import logger


def safe_copy(src: str, dst: str) -> None:
    r"""Copies a file to a given destination, ignoring copies of a file onto itself.

    Args:
        src (str): Path to source file.
        dst (str): Path to destination file.
    """
    try:
        shutil.copyfile(src=src, dst=dst)
    except shutil.SameFileError:
        logger.warning(f"Ignoring file copy from {src} to itself.")
        pass


def save_checkpoint(
    directory_path: str,
    model: CaBRNet,
    model_config: str,
    optimizer_mngr: OptimizerManager | None,
    training_config: str | None,
    dataset_config: str,
    projection_info: dict[int, dict[str, int | float]] | None,
    epoch: int | str,
    seed: int | None,
    device: str,
    stats: dict[str, Any] | None = None,
) -> None:
    r"""Saves everything needed to restart a training process.

    Args:
        directory_path (str): Target location.
        model (Module): CaBRNet model.
        model_config (str): Path to the model configuration file.
        optimizer_mngr (OptimizerManager): Optimizer manager.
        training_config (str): Path to the training configuration file.
        dataset_config (str): Path to the dataset configuration file.
        projection_info (dictionary, optional): Projection dictionary, generated during training epilogue.
        epoch (int or str): Current epoch.
        seed (int): Initial random seed (recorded for reproducibility).
        device (str): Target hardware device (recorded for reproducibility).
        stats (dictionary, optional): Other optional statistics. Default: None.
    """
    os.makedirs(directory_path, exist_ok=True)

    model.eval()

    torch.save(model.state_dict(), os.path.join(directory_path, CaBRNet.DEFAULT_MODEL_STATE))
    if optimizer_mngr is not None:
        torch.save(optimizer_mngr.state_dict(), os.path.join(directory_path, OptimizerManager.DEFAULT_TRAINING_STATE))
    safe_copy(src=model_config, dst=os.path.join(directory_path, CaBRNet.DEFAULT_MODEL_CONFIG))
    if training_config is not None:
        safe_copy(src=training_config, dst=os.path.join(directory_path, OptimizerManager.DEFAULT_TRAINING_CONFIG))
    safe_copy(src=dataset_config, dst=os.path.join(directory_path, DatasetManager.DEFAULT_DATASET_CONFIG))

    state = {
        "random_generators": {
            "torch_rng_state": torch.random.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        "epoch": epoch,
        "seed": seed,
        "device": device,
        "stats": stats,
    }

    with open(os.path.join(directory_path, "state.pickle"), "wb") as file:
        pickle.dump(state, file)

    # Save projection information if it exists
    if projection_info is not None:
        save_projection_info(projection_info, os.path.join(directory_path, CaBRNet.DEFAULT_PROJECTION_INFO))

    logger.info(f"Successfully saved checkpoint at epoch {epoch}.")


def load_checkpoint(
    directory_path: str,
    model: CaBRNet,
    optimizer_mngr: OptimizerManager | None = None,
) -> dict[str, Any]:
    r"""Restores training process using checkpoint directory.

    Args:
        directory_path (str): Target location.
        model (Module): CaBRNet mode.
        optimizer_mngr (OptimizerManager, optional): Optimizer manager. Default: None.

    Returns:
        Dictionary containing auxiliary state information (epoch, seed, device, stats).
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Unknown checkpoint directory {directory_path}")

    model.load_state_dict(torch.load(os.path.join(directory_path, CaBRNet.DEFAULT_MODEL_STATE), map_location="cpu"))
    if optimizer_mngr is not None:
        optimizer_state_path = os.path.join(directory_path, OptimizerManager.DEFAULT_TRAINING_STATE)
        if os.path.isfile(optimizer_state_path):
            optimizer_mngr.load_state_dict(torch.load(optimizer_state_path, map_location="cpu"))
        else:
            logger.warning(f"Could not find optimizer state {optimizer_state_path}. Using default state instead.")

    # Restore RNG state
    with open(os.path.join(directory_path, "state.pickle"), "rb") as file:
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


def save_projection_info(projection_info: dict[int, dict[str, int | float]], filename: str) -> None:
    r"""Saves projection information, either in pickle or CSV format.

    Args:
        projection_info (dictionary): Projection dictionary, generated during training epilogue.
        filename (str): Path to output file. Based on the file extension, the file is stored in
          pickle format (pickle or pkl extension) or CSV format (any other extension).
    """
    if filename.lower().endswith(("pickle", "pkl")):
        with open(filename, "wb") as f:
            pickle.dump(projection_info, f)
    else:
        # CSV format
        with open(filename, "w") as f:
            writer = csv.DictWriter(f, fieldnames=["proto_idx"] + list(projection_info[0].keys()))
            writer.writeheader()
            for proto_idx in projection_info.keys():
                writer.writerow(projection_info[proto_idx] | {"proto_idx": proto_idx})


def load_projection_info(filename: str) -> dict[int, dict[str, int | float]]:
    r"""Loads projection information, either in pickle or CSV format.

    Args:
        filename (str): Path to input file. Based on the file extension, the file is loaded in
          pickle format (pickle or pkl extension) or CSV format (any other extension).

    Returns:
        Projection dictionary, generated during training epilogue.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Could not find projection information file {filename}")
    if filename.lower().endswith(tuple(["pickle", "pkl"])):
        with open(filename, "rb") as file:
            projection_info = pickle.load(file)
    else:
        projection_list = pd.read_csv(filename).to_dict(orient="records")
        projection_info = {entry["proto_idx"]: entry for entry in projection_list}
    return projection_info
