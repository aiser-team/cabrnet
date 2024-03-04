"""Implement the saving and loading capabilities for a CaBRNet model."""

import os
import pickle
import random
import shutil
from typing import Any, Mapping
import numpy as np
import torch
from loguru import logger
from cabrnet.utils.optimizers import OptimizerManager
from cabrnet.utils.data import DatasetManager
from cabrnet.visualization.visualizer import SimilarityVisualizer
from cabrnet.generic.model import CaBRNet


def save_checkpoint(
    directory_path: str,
    model: CaBRNet,
    model_config: str,
    optimizer_mngr: OptimizerManager | None,
    training_config: str | None,
    dataset_config: str,
    visualization_config: str,
    epoch: int | str,
    seed: int | None,
    device: str,
    stats: dict[str, Any] | None = None,
) -> None:
    """Save everything needed to restart a training process.

    Args:
        directory_path: Target location
        model: CaBRNet model
        model_config: Path to the model configuration file
        optimizer_mngr: Optimizer manager
        training_config: Path to the training configuration file
        dataset_config: Path to the dataset configuration file
        visualization_config: Path to the visualization configuration file
        epoch: Current epoch
        seed: Initial random seed (recorded for reproducibility)
        device: Target hardware device (recorded for reproducibility)
        stats: Other optional statistics
    """

    def safe_copy(src: str, dst: str):
        try:
            shutil.copyfile(src=src, dst=dst)
        except shutil.SameFileError:
            logger.warning(f"Ignoring file copy from {src} to itself.")
            pass

    os.makedirs(directory_path, exist_ok=True)

    model.eval()  # NOTE: do we want this?

    torch.save(model.state_dict(), os.path.join(directory_path, CaBRNet.DEFAULT_MODEL_STATE))
    if optimizer_mngr is not None:
        torch.save(optimizer_mngr.state_dict(), os.path.join(directory_path, OptimizerManager.DEFAULT_TRAINING_STATE))
    safe_copy(src=model_config, dst=os.path.join(directory_path, CaBRNet.DEFAULT_MODEL_CONFIG))
    if training_config is not None:
        safe_copy(src=training_config, dst=os.path.join(directory_path, OptimizerManager.DEFAULT_TRAINING_CONFIG))
    safe_copy(src=dataset_config, dst=os.path.join(directory_path, DatasetManager.DEFAULT_DATASET_CONFIG))
    safe_copy(
        src=visualization_config, dst=os.path.join(directory_path, SimilarityVisualizer.DEFAULT_VISUALIZATION_CONFIG)
    )

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

    logger.info(f"Successfully saved checkpoint at epoch {epoch}.")


def load_checkpoint(
    directory_path: str,
    model: CaBRNet,
    optimizer_mngr: OptimizerManager | None = None,
) -> Mapping[str, Any]:
    """Restore training process using checkpoint directory.

    Args:
        directory_path: Target location
        model: CaBRNet mode
        optimizer_mngr: Optimizer manager

    Returns:
        dictionary containing auxiliary state information (epoch, seed, device, stats)
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Unknown checkpoint directory {directory_path}")

    model.load_state_dict(torch.load(os.path.join(directory_path, CaBRNet.DEFAULT_MODEL_STATE), map_location="cpu"))
    if optimizer_mngr is not None:
        optimizer_mngr.load_state_dict(
            torch.load(os.path.join(directory_path, OptimizerManager.DEFAULT_TRAINING_STATE), map_location="cpu")
        )

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
