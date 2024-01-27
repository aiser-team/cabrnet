"""Implement the saving and loading capabilities for a generic ProtoClassifier."""

import os
import pickle
import random
import shutil
from typing import Any, Mapping
import numpy as np
import torch
from loguru import logger
from cabrnet.generic.model import ProtoClassifier
from cabrnet.utils.parser import get_optimizer, get_param_groups, get_scheduler, load_config
from cabrnet.utils.data import get_dataloaders


def save_checkpoint(
    directory_path: str,
    model: ProtoClassifier,
    model_config: str,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    training_config: str | None,
    dataset_config: str,
    epoch: int,
    seed: int | None,
    device: str,
    stats: dict[str, Any] | None = None,
) -> None:
    """Save everything needed to restart a training process.

    Args:
        directory_path: Target location
        model: ProtoClassifier
        model_config: Path to the model configuration file
        optimizer: Optimizer
        scheduler: Scheduler
        training_config: Path to the training configuration file
        dataset_config: Path to the dataset configuration file
        epoch: Current epoch
        seed: Initial random seed (recorded for reproducibility)
        device: Target hardware device (recorded for reproducibility)
        stats: Other optional statistics
    """
    os.makedirs(directory_path, exist_ok=True)

    model.eval()  # NOTE: do we want this?

    torch.save(model.state_dict(), os.path.join(directory_path, "model_state.pth"))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(directory_path, "optimizer_state.pth"))
    if scheduler is not None:  # NOTE: do we save something if there is no scheduler?
        torch.save(scheduler.state_dict(), os.path.join(directory_path, "scheduler_state.pth"))
    shutil.copyfile(src=model_config, dst=os.path.join(directory_path, "model.yml"))
    if training_config is not None:
        shutil.copyfile(src=training_config, dst=os.path.join(directory_path, "training.yml"))
    shutil.copyfile(src=dataset_config, dst=os.path.join(directory_path, "dataset.yml"))

    state = {
        "random_generators": {
            "torch_rng_state": torch.random.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        "epoch": epoch,
        "stats": stats,
    }

    with open(os.path.join(directory_path, "state.pickle"), "wb") as file:
        pickle.dump(state, file)

    # Add reproducibility information
    with open(os.path.join(directory_path, "reproducibility.txt"), "w") as file:
        file.write(f"seed: {seed}\n" f"device: {device}")

    logger.info(f"Successfully saved checkpoint at epoch {epoch}.")


# TODO: this function returns dataloaders by default, add hidden `return_datasets` option to return datasets instead
def load_checkpoint(directory_path: str) -> Mapping[str, Any]:
    """Restore training process using checkpoint directory.

    Args:
        directory_path: Target location

    Returns:
        dictionary containing checkpoint state (model, optimizer, scheduler, dataloaders, epoch, stats)
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Unknown checkpoint directory {directory_path}")

    model = ProtoClassifier.build_from_config(os.path.join(directory_path, "model.yml"))
    model.load_state_dict(torch.load(os.path.join(directory_path, "model_state.pth"), map_location="cpu"))

    trainer = load_config(os.path.join(directory_path, "training.yml"))
    param_groups = get_param_groups(trainer, model)
    optimizer = get_optimizer(trainer, param_groups)
    scheduler = get_scheduler(trainer, optimizer)

    dataloaders = get_dataloaders(os.path.join(directory_path, "dataset.yml"))

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

    logger.info(f"Successfully loaded checkpoint from epoch {epoch}.")

    with open(os.path.join(directory_path, "reproducibility.txt"), "r") as file:
        for line in file.readlines():
            logger.info(f"Reproducibility information. {line.rstrip()}")

    return {
        "model": model,
        "optimize": optimizer,
        "scheduler": scheduler,
        "dataloaders": dataloaders,
        "epoch": epoch,
        "stats": stats,
    }
