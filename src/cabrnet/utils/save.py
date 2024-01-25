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
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    training_config: str,
    dataset_config: str,
    epoch: int,
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
        stats: Other optional statistics
    """
    os.makedirs(directory_path, exist_ok=True)

    model.eval()  # NOTE: do we want this?

    torch.save(model.state_dict(), os.path.join(directory_path, "model_state.pth"))
    torch.save(optimizer.state_dict(), os.path.join(directory_path, "optimizer_state.pth"))
    if scheduler is not None:  # NOTE: do we save something if there is no scheduler?
        torch.save(scheduler.state_dict(), os.path.join(directory_path, "scheduler_state.pth"))
    shutil.copyfile(src=model_config, dst=os.path.join(directory_path, "model.yml"))
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

    return {
        "model": model,
        "optimize": optimizer,
        "scheduler": scheduler,
        "dataloaders": dataloaders,
        "epoch": epoch,
        "stats": stats,
    }


# TODO: Check whether the functions below are still useful
def save_model(
    directory_path: str,
    model: ProtoClassifier,
    model_config: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_config: str,
    dataset_config: str,
    epoch: int,
    best_train_acc: float,
    best_test_acc: float,
    leaf_labels: dict,
    checkpoint_frequency: int = 10,
) -> None:
    """Save a tree and all its parameters.

    Args:
        directory_path: Target location
        model: ProtoClassifier
        model_config: Path to the model configuration file
        optimizer: Optimizer
        scheduler: Scheduler
        training_config: Path to the training configuration file
        dataset_config: Path to the dataset configuration file
        epoch: Current epoch
        best_train_acc: Best train accuracy
        best_test_acc: Best test accuracy
        leaf_labels: Labels of the leaves
        checkpoint_frequency: Frequency to which to make checkpoints
    """
    assert checkpoint_frequency > 0, f"Invalid checkpoint frequency {checkpoint_frequency}"
    # Save latest model
    save_checkpoint(
        os.path.join(directory_path, "latest"),
        model,
        model_config,
        optimizer,
        scheduler,
        training_config,
        dataset_config,
        epoch,
        stats={"best_train_acc": best_train_acc, "best_test_acc": best_test_acc, "leaf_labels": leaf_labels},
    )

    # Save model every 10 epochs
    if epoch % checkpoint_frequency == 0:
        save_checkpoint(
            os.path.join(directory_path, f"epoch_{epoch}"),
            model,
            model_config,
            optimizer,
            scheduler,
            training_config,
            dataset_config,
            epoch,
            stats={"best_train_acc": best_train_acc, "best_test_acc": best_test_acc, "leaf_labels": leaf_labels},
        )


def save_best_train_tree(
    directory_path: str,
    model: ProtoClassifier,
    model_config: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_config: str,
    dataset_config: str,
    epoch: int,
    train_acc: float,
    best_train_acc: float,
    best_test_acc: float,
    leaf_labels: dict,
) -> float:
    """Save the best tree at training and all its parameters.

    Args:
        directory_path: Target location
        model: ProtoClassifier
        model_config: Path to the model configuration file
        optimizer: Optimizer
        scheduler: Scheduler
        training_config: Path to the training configuration file
        dataset_config: Path to the dataset configuration file
        epoch: Current epoch
        train_acc: Current train accuracy
        best_train_acc: Best train accuracy
        best_test_acc: Best test accuracy
        leaf_labels: Labels of the leaves
    """
    if train_acc > best_train_acc:
        best_train_acc = train_acc
        save_checkpoint(
            os.path.join(directory_path, "best_train_model"),
            model,
            model_config,
            optimizer,
            scheduler,
            training_config,
            dataset_config,
            epoch,
            stats={"best_train_acc": best_train_acc, "best_test_acc": best_test_acc, "leaf_labels": leaf_labels},
        )
    return best_train_acc


def save_best_test_tree(
    directory_path: str,
    model: ProtoClassifier,
    model_config: str,
    optimizer,
    scheduler,
    training_config: str,
    dataset_config: str,
    epoch: int,
    best_train_acc: float,
    test_acc: float,
    best_test_acc: float,
    leaf_labels: dict,
) -> float:
    """Save the best tree at testing and all its parameters.

    Args:
        directory_path: Target location
        model: ProtoClassifier
        model_config: Path to the model configuration file
        optimizer: Optimizer
        scheduler: Scheduler
        training_config: Path to the training configuration file
        dataset_config: Path to the dataset configuration file
        epoch: Current epoch
        best_train_acc: Best train accuracy
        test_acc: Current test accuracy
        best_test_acc: Best test accuracy
        leaf_labels: Labels of the leaves
    """
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        save_checkpoint(
            os.path.join(directory_path, "best_test_model"),
            model,
            model_config,
            optimizer,
            scheduler,
            training_config,
            dataset_config,
            epoch,
            stats={"best_train_acc": best_train_acc, "best_test_acc": best_test_acc, "leaf_labels": leaf_labels},
        )
    return best_test_acc
