import os
import sys
from typing import Any, Iterable

import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.monitoring import metrics_to_str
from cabrnet.core.utils.optimizers import OptimizerManager
from cabrnet.core.utils.parser import load_config
from cabrnet.core.utils.save import load_checkpoint, save_checkpoint
from cabrnet.core.utils.system_info import get_parent_directory


def training_loop(
    working_dir: str,
    model: CaBRNet,
    epoch_range: Iterable,
    dataloaders: dict[str, DataLoader],
    optimizer_mngr: OptimizerManager,
    metric: str,
    maximize: bool,
    best_metric: float,
    num_epochs: int,
    model_arch: dict[str, Any] | str,
    training_config: dict[str, Any] | str,
    dataset_config: dict[str, Any] | str,
    save_final: bool = True,
    checkpoint_frequency: int | None = None,
    resume_dir: str | None = None,
    seed: int = 42,
    device: str | torch.device = "cuda:0",
    verbose: bool = False,
    logger_level: str | None = None,
) -> dict[str, Any]:
    r"""Implements the main training loop for CaBRNet models.

    Args:
        working_dir (str): Working directory.
        model (Module): CaBRNet model to be trained.
        epoch_range (iterable): Range of epoch indices.
        dataloaders (dict): Dictionary of dataloaders.
        optimizer_mngr (OptimizerManager): Optimizer manager.
        metric (str): Metric to optimize.
        maximize (bool): Optimization mode (either maximize or minimize the chosen metric).
        best_metric (float): Current best value for the metric.
        num_epochs (int): Total number of epochs.
        model_arch (dict | str): Path to model configuration file, or configuration dictionary.
        training_config (dict | str): Path to training configuration file, or configuration dictionary.
        dataset_config (dict | str): Path to dataset configuration file, or configuration dictionary.
        save_final (bool, optional): If True, saves the final model after epilogue. Default: True.
        checkpoint_frequency (int, optional): Save training checkpoint every <checkpoint_frequency> epochs.
            Default: None.
        resume_dir (str, optional): If given, directory from which training was resumed. Default: None.
        seed (int, optional): Initial random seed. Default: 42.
        device (str | device, optional): Hardware device. Default: cuda:0.
        verbose (bool, optional): If True, enables verbose mode. Default: False.
        logger_level (str, optional): If given, change logger level inside function. Default: None.

    Returns:
        Dictionary of statistics on the test set.
    """
    if logger_level is not None:
        # Adjust logger level and set log file
        logger.configure(handlers=[{"sink": sys.stderr, "level": logger_level}])
        logger.add(sink=os.path.join(working_dir, "log.txt"), level=logger_level)

    tboard_dir = os.path.join(working_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir=tboard_dir)

    epochs_since_best = 0
    trained = False

    for epoch in epoch_range:
        # Handle early abort
        if epochs_since_best >= optimizer_mngr.get_patience(epoch):
            logger.warning(f"Ran out of patience after {optimizer_mngr.get_patience(epoch)} epochs.")
            break
        epochs_since_best += 1

        # At least one epoch took place
        trained = True

        # Freeze parameters if necessary depending on current epoch and parameter group
        optimizer_mngr.freeze(epoch=epoch)
        train_info = model.train_epoch(
            dataloaders=dataloaders,
            optimizer_mngr=optimizer_mngr,
            device=device,
            tqdm_position=1,
            epoch_idx=epoch,
            verbose=verbose,
        )
        # Apply scheduler
        optimizer_mngr.scheduler_step(epoch=epoch)

        # Add all stats to Tensorboard
        for key, value in train_info.items():
            writer.add_scalar(key, value, epoch)
        writer.flush()

        save_best_checkpoint = False
        if train_info.get(metric) is None:
            raise ValueError(f"Unknown training metric '{metric}'. Candidates are {list(train_info.keys())}")
        if (maximize and best_metric < train_info[metric]) or (not maximize and best_metric > train_info[metric]):
            best_metric = train_info[metric]
            save_best_checkpoint = True
            epochs_since_best = 0

        # Add information regarding current best metric
        train_info[f"best_{metric}"] = best_metric
        logger.info(f"Metrics at epoch {epoch}: {metrics_to_str(train_info)}")
        if save_best_checkpoint:
            logger.success(f"Better model found at epoch {epoch}. Saving checkpoint.")
            save_checkpoint(
                directory_path=os.path.join(working_dir, "best"),
                model=model,
                model_arch=model_arch,
                optimizer_mngr=optimizer_mngr,
                training_config=training_config,
                dataset_config=dataset_config,
                projection_info=None,
                epoch=epoch,
                seed=seed,
                device=device,
                stats=train_info,
            )
        if checkpoint_frequency is not None and (epoch % checkpoint_frequency == 0):
            save_checkpoint(
                directory_path=os.path.join(working_dir, f"epoch_{epoch}"),
                model=model,
                model_arch=model_arch,
                optimizer_mngr=optimizer_mngr,
                training_config=training_config,
                dataset_config=dataset_config,
                projection_info=None,
                epoch=epoch,
                seed=seed,
                device=device,
                stats=train_info,
            )
    writer.close()

    if trained:
        # Seek best model
        if os.path.isdir(os.path.join(working_dir, "best")):
            path_to_best = os.path.join(working_dir, "best")
        elif resume_dir is not None and os.path.isdir(os.path.join(get_parent_directory(resume_dir), "best")):
            # Best checkpoint occurred before training was resumed
            path_to_best = os.path.join(get_parent_directory(resume_dir), "best")
        else:
            raise FileNotFoundError("Could not find path to best model. Aborting epilogue.")
        logger.info(f"Loading best model from checkpoint {path_to_best}")
        load_checkpoint(directory_path=path_to_best, model=model, optimizer_mngr=optimizer_mngr)

    # Call epilogue
    epilogue_params = (
        training_config.get("epilogue", {})
        if isinstance(training_config, dict)
        else load_config(training_config).get("epilogue", {})
    )
    projection_info = model.epilogue(
        dataloaders=dataloaders,
        optimizer_mngr=optimizer_mngr,
        output_dir=working_dir,
        device=device,
        verbose=verbose,
        **epilogue_params,
    )

    # Evaluate model
    eval_info = model.evaluate(dataloader=dataloaders["test_set"], device=device, verbose=verbose)
    logger.info(f"Metrics on test set: {metrics_to_str(eval_info)}")
    if save_final:
        save_checkpoint(
            directory_path=os.path.join(working_dir, "final"),
            model=model,
            model_arch=model_arch,
            optimizer_mngr=None,
            training_config=training_config,
            dataset_config=dataset_config,
            projection_info=projection_info,
            epoch=num_epochs,
            seed=seed,
            device=device,
            stats=eval_info,
        )
    return eval_info
