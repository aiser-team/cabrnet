from pathlib import Path
import sys
from typing import Any, Iterable
from shutil import rmtree, copytree

import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.monitoring import metrics_to_str
from cabrnet.core.utils.optimizers import OptimizerManager
from cabrnet.core.utils.parser import load_config
from cabrnet.core.utils.save import load_checkpoint, save_checkpoint

LATEST_DIR = Path("latest")
FINAL_DIR = Path("final")
BEST_DIR = Path("best")
INITIAL_DIR = Path("init")
BACKUP_DIR = Path("tmp")
LOG_FILE = Path("log.txt")
TB_LOGS = Path("tensorboard_logs")


def best_dir(working_dir: Path) -> Path:
    r"""Provides the name of the subdirectory that will contain the best version of the model,
    which is populated at every iteration.

    Args:
        working_dir (Path): The output dir.

    Returns:
        The subdirectory.
    """
    return working_dir / BEST_DIR


def final_dir(working_dir: Path) -> Path:
    r"""Provides the name of the subdirectory that will contain the final version of the model,
    which is populated at every iteration.

    Args:
        working_dir (Path): The output dir.

    Returns:
        The subdirectory.
    """
    return working_dir / FINAL_DIR


def latest_dir(working_dir: Path) -> Path:
    r"""Provides the name of the subdirectory that will contain the latest version of the model,
    which is populated at every iteration.

    Args:
        working_dir (Path): The output dir.

    Returns:
        The subdirectory.
    """
    return working_dir / LATEST_DIR


def backup_dir(working_dir: Path) -> Path:
    r"""Provides the name of the subdirectory that will contain the backup of the latest version of the model,
    which is populated at every iteration.

    Args:
        working_dir (Path): The output dir.

    Returns:
        The subdirectory.
    """
    return working_dir / BACKUP_DIR


def training_loop(
    working_dir: Path,
    model: CaBRNet,
    epoch_range: Iterable,
    dataloaders: dict[str, DataLoader],
    optimizer_mngr: OptimizerManager,
    metric: str,
    maximize: bool,
    best_metric: float,
    num_epochs: int,
    model_arch: dict[str, Any] | Path,
    training_config: dict[str, Any] | Path,
    dataset_config: dict[str, Any] | Path,
    save_final: bool = True,
    checkpoint_frequency: int | None = None,
    resume_dir: Path | None = None,
    seed: int = 42,
    device: str | torch.device = "cuda:0",
    verbose: bool = False,
    logger_level: str | None = None,
) -> dict[str, Any]:
    r"""Implements the main training loop for CaBRNet models.

    Args:
        working_dir (Path): Working directory.
        model (Module): CaBRNet model to be trained.
        epoch_range (iterable): Range of epoch indices.
        dataloaders (dict): Dictionary of dataloaders.
        optimizer_mngr (OptimizerManager): Optimizer manager.
        metric (str): Metric to optimize.
        maximize (bool): Optimization mode (either maximize or minimize the chosen metric).
        best_metric (float): Current best value for the metric.
        num_epochs (int): Total number of epochs.
        model_arch (dict | Path): Path to model configuration file, or configuration dictionary.
        training_config (dict | Path): Path to training configuration file, or configuration dictionary.
        dataset_config (dict | Path): Path to dataset configuration file, or configuration dictionary.
        save_final (bool, optional): If True, saves the final model after epilogue. Default: True.
        checkpoint_frequency (int, optional): Save training checkpoint every <checkpoint_frequency> epochs.
            Default: None.
        resume_dir (Path, optional): If given, directory from which training was resumed. Default: None.
        seed (int, optional): Initial random seed. Default: 42.
        device (str | device, optional): Hardware device. Default: cuda:0.
        verbose (bool, optional): If True, enables verbose mode. Default: False.
        logger_level (str, optional): If given, change logger level inside function. Default: None.

    Returns:
        Dictionary of statistics on the test set.
    """

    projection_info = None  # Default value used by subroutine [save]. Might be a parameter of [save] instead.
    train_info = None

    def save(dir_name: Path, epoch: int | str, optimizer: OptimizerManager | None) -> None:
        r"""Saves the model by calling :func:`~cabrnet.core.utils.save.save_checkpoint`. Most parameters are already known
        in [training_loop], which is why they do not need to be repeated when calling this subroutine.

        Args:
            dir_name (Path): The name of the folder in which the model is saved.  This is added to the working dir.
            epoch (int|str): The epoch parameter of :func:`~cabrnet.core.utils.save.save_checkpoint`.
            optimizer (OptimizerManager): Current optimizer (if any).
        """
        save_checkpoint(
            directory_path=working_dir / dir_name,
            model=model,
            model_arch=model_arch,
            optimizer_mngr=optimizer,
            training_config=training_config,
            dataset_config=dataset_config,
            projection_info=projection_info,
            epoch=epoch,
            seed=seed,
            device=device,
            stats=train_info,
        )

    if logger_level is not None:
        # Adjust logger level and set log file
        logger.configure(handlers=[{"sink": sys.stderr, "level": logger_level}])
        logger.add(sink=working_dir / LOG_FILE, level=logger_level)

    tboard_dir = working_dir / TB_LOGS
    writer = SummaryWriter(log_dir=tboard_dir)  # type: ignore

    epochs_since_best = 0
    trained = False

    # Save initial model before training
    if list(epoch_range) and next(iter(epoch_range)) == 0 and checkpoint_frequency is not None:
        train_info = model.evaluate(dataloaders=dataloaders, dataset_name="train_set", device=device, verbose=verbose)

        # Add all stats to Tensorboard
        for key, value in train_info.items():
            writer.add_scalar(key, value, 0)
        writer.flush()

        save(dir_name=INITIAL_DIR, epoch="init", optimizer=optimizer_mngr)

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
        optimizer_mngr.scheduler_step(epoch=epoch, metric=train_info.get(metric))

        if "val_set" in dataloaders.keys():
            val_info = model.evaluate(dataloaders, "val_set", device=device, tqdm_position=1, verbose=verbose)
            train_info |= val_info

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

        def safe_save(output_dir: Path):
            r"""Save checkpoint, with extra care to prevent data loss."""
            rmtree(backup_dir(working_dir), ignore_errors=True)  # Delete old backup (if any)
            if output_dir.is_dir():  # Backup existing directory (if any)
                # Perform a copy rather than a renaming to maintain path to configuration files
                # (in case computation is restarted from this directory)
                copytree(output_dir, backup_dir(working_dir))
            save(dir_name=output_dir, epoch=epoch, optimizer=optimizer_mngr)
            rmtree(backup_dir(working_dir), ignore_errors=True)  # Delete backup (if any)

        # Save latest checkpoint
        safe_save(LATEST_DIR)
        if save_best_checkpoint:
            logger.success(f"Better model found at epoch {epoch}. Saving checkpoint.")
            safe_save(BEST_DIR)
        if checkpoint_frequency is not None and (epoch % checkpoint_frequency == 0):
            save(dir_name=Path(f"epoch_{epoch}"), epoch=epoch, optimizer=optimizer_mngr)
    writer.close()

    if trained:
        # Seek best model
        if best_dir(working_dir).is_dir():
            path_to_best = best_dir(working_dir)
        elif resume_dir is not None and best_dir(resume_dir.parent).is_dir():
            # Best checkpoint occurred before training was resumed
            path_to_best = best_dir(resume_dir.parent)
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
    projection_info = model.epilogue(  # Save projection infos before final checkpoint
        dataloaders=dataloaders,
        optimizer_mngr=optimizer_mngr,
        output_dir=working_dir,
        device=device,
        verbose=verbose,
        **epilogue_params,
    )

    # Evaluate model
    eval_info = model.evaluate(dataloaders=dataloaders, dataset_name="test_set", device=device, verbose=verbose)
    logger.info(f"Metrics on test set: {metrics_to_str(eval_info)}")
    if save_final:
        save(dir_name=FINAL_DIR, epoch=num_epochs, optimizer=None)
    return eval_info
