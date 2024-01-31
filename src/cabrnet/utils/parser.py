"""This file contains all the necessary tools to parse the various config files."""

import argparse
import os

import yaml
from loguru import logger
from torch import nn, optim


def load_config(config_file: str) -> dict:
    """Load a configuration file for CaBRNet.

    Args:
        config_file: Path to the configuration file.

    Returns:
        The properly loaded config file.

    Raises:
        ValueError: The config file is in an unsupported file format.
    """
    file_ext = os.path.splitext(config_file)[-1].lower()

    if file_ext in [".yml", ".yaml"]:
        with open(config_file, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    raise ValueError("Unsupported file format, only YAML is supported.")


# TODO: Allow for a single parameter group
def get_param_groups(trainer: dict, model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Build the groups of parameters for the training.

    Args:
        trainer: Options to build the groups.
        model: Model to take the parameters from.

    Returns:
        The groups of parameters.
    """
    param_groups = trainer["param_groups"]
    param_dict: dict[str, list] = {}
    covered_names = []

    for group_name, group_value in param_groups.items():
        param_dict[group_name] = []
        if isinstance(group_value, str):
            for name, param in model.named_parameters():
                if name.startswith(group_value):
                    param_dict[group_name].append(param)
                    covered_names.append(name)
        else:
            start = group_value["start"]
            stop = group_value["stop"]
            record = start is None
            for name, param in model.named_parameters():
                if start and name.startswith(start):
                    record = True
                if record and (name not in covered_names):
                    param_dict[group_name].append(param)
                    covered_names.append(name)
                if stop and name.startswith(stop):
                    record = False
        if not param_dict[group_name]:
            logger.warning(f"Empty parameter group {group_name}.")
    not_covered_count = 0
    first_not_covered_param = ""
    for name, _ in model.named_parameters():
        if name not in covered_names:
            if not_covered_count == 0:
                first_not_covered_param = name
            not_covered_count += 1
    if not_covered_count > 0:
        logger.warning(
            f"{first_not_covered_param} does not belong to any parameter group ({not_covered_count-1} similar messages)."
        )

    return param_dict


# TODO: check is global lr useful?
def get_optimizer(trainer: dict, param_groups: dict[str, list[nn.Parameter]]) -> optim.Optimizer:
    """Build the optimizer.

    Args:
        trainer: Options to build the optimizer.
        param_groups: Groups of parameters.

    Returns:
        Initialized optimizer.
    """
    optim_config = trainer["optimizer"]["config"]
    optim_type = trainer["optimizer"]["name"]
    param_list = [{"params": value, **optim_config[key]} for key, value in param_groups.items()]
    return getattr(optim, optim_type)(params=param_list, **trainer["optimizer"]["params"])


def get_scheduler(trainer: dict, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler | None:
    """Build the learning rate scheduler for the optimizer.

    Args:
        trainer: Training configuration
        optimizer: Target optimizer

    Returns:
        Initialized LR scheduler
    """
    if "scheduler" in trainer["optimizer"]:
        params = trainer["optimizer"]["scheduler"]["params"]
        type = trainer["optimizer"]["scheduler"]["type"]
        return getattr(optim.lr_scheduler, type)(optimizer=optimizer, **params)
    logger.warning("No Learning Rate Scheduler defined")
    return None


def freeze(epoch: int, param_groups: dict, trainer: dict) -> None:
    """Freeze the parameter groups declared in the training configuration file at the current epoch.

    Args:
        epoch: Current epoch.
        param_groups: Parameter groups defined in the training configuration file.
        trainer: Dictionary from training configuration file.
    """
    freeze_groups = trainer.get("freeze")

    def _build_params_lists(group):
        if group not in param_groups.keys():
            logger.warning(f"Attempting to freeze non-existing group {group}.")
        else:
            for p_group in param_groups.keys():
                if p_group not in group:
                    params_to_train.append(param_groups[p_group])
                else:
                    params_to_freeze.append(param_groups[p_group])

    if freeze_groups is not None:
        for f_group in freeze_groups:
            targets_to_freeze = freeze_groups[f_group]["targets"]
            params_to_train: list[list[nn.Parameter]] = []
            params_to_freeze: list[list[nn.Parameter]] = []
            if isinstance(targets_to_freeze, list):
                for target in targets_to_freeze:
                    _build_params_lists(target)
            else:
                _build_params_lists(targets_to_freeze)

            for param_list in params_to_train:
                for param in param_list:
                    param.requires_grad = True
            epoch_range = freeze_groups[f_group]["epoch_range"]
            assert len(epoch_range) == 2
            if epoch in range(epoch_range[0], epoch_range[1] + 1):
                logger.debug(f"Epoch {epoch}: parameter group {f_group} frozen")
                for param_list in params_to_freeze:
                    for param in param_list:
                        param.requires_grad = False
            else:
                logger.debug(f"Epoch {epoch}: parameter group {f_group} unfrozen")
                for param_list in params_to_freeze:
                    for param in param_list:
                        param.requires_grad = True


def create_training_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Create the argument parser for CaBRNet training configuration.

    Args:
        parser: Existing parser (if any)
    Returns:
        The parser itself.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Load datasets.")

    # Either provide a training configuration file or the path to a checkpoint directory
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--training",
        "-t",
        type=str,
        metavar="/path/to/file.yml",
        help="Path to the training configuration file",
    )
    group.add_argument(
        "--resume-from",
        type=str,
        metavar="/path/to/checkpoint/directory",
        help="Path to existing checkpoint directory",
    )
    parser.add_argument(
        "--training-dir",
        type=str,
        required=True,
        metavar="path/to/training/directory",
        help="Path to output directory",
    )
    parser.add_argument(
        "--save-best",
        type=str,
        required=False,
        choices=["acc", "loss"],
        default="acc",
        metavar="metric",
        help="Save best model based on accuracy or loss",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        required=False,
        metavar="num_epochs",
        help="Checkpoint frequency (in epochs)",
    )
    return parser
