from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn as nn
from cabrnet.core.utils.parser import load_config
from loguru import logger


def move_optimizer_to(optim: torch.optim.Optimizer, device: str | torch.device) -> None:
    r"""Moves optimizer to target device. Solution from https://github.com/pytorch/pytorch/issues/8741.

    Args:
        optim (Optimizer): Optimizer.
        device (str | device): Hardware device.
    """
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


class OptimizerManager:
    r"""Manager in charge of optimizers, learning rate schedulers and freezing parameters.

    Attributes:
        config: Configuration dictionary used to build this object.
        param_groups: Dictionary separating the model parameters into groups.
        optimizers: Dictionary of optimizers associated with different parameter groups.
        schedulers: Dictionary of learning rate schedulers associated with optimizers.
        periods: Dictionary defining training periods.
    """

    def __init__(self, config_dict: dict, module: nn.Module) -> None:
        r"""Initializes a OptimizeManager object.

        Args:
            config_dict (dictionary): Configuration dictionary.
            module (Module): Target module.

        """
        self.config = config_dict

        self.param_groups: dict[str, list[nn.Parameter]] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {}
        self.periods: dict[str, dict[str, Any]] = {}

        self._set_param_groups(module=module)
        self._set_optimizers()
        self._set_periods()

    @staticmethod
    def build_from_config(config: str | dict[str, Any], model: nn.Module) -> OptimizerManager:
        r"""Builds an OptimizerManager object from a YML file.

        Args:
            config (str | dict): Path to configuration file, or configuration dictionary.
            model (Module): Target module.

        Returns:
            OptimizerManager.
        """
        if not isinstance(config, (str, dict)):
            raise ValueError(f"Unsupported configuration format: {type(config)}")
        if isinstance(config, str):
            config_dict = load_config(config)
        else:
            config_dict = config
        return OptimizerManager(config_dict=config_dict, module=model)

    def _set_param_groups(self, module: nn.Module) -> None:
        r"""Builds the groups of parameters from the configuration dictionary.

        Args:
            module (Module): Target module.
        """
        param_groups = self.config.get("param_groups")

        if param_groups is None:
            # Create a default param group encompassing all model parameters
            self.param_groups["main"] = [param for _, param in module.named_parameters()]
            return

        covered_names = []

        for group_name, group_value in param_groups.items():
            self.param_groups[group_name] = []
            if isinstance(group_value, list | str):
                # Submodule or list of submodules
                if isinstance(group_value, str):
                    group_value = [group_value]
                for submodule_name in group_value:
                    match_found = False
                    for name, param in module.named_parameters():
                        if name.startswith(submodule_name):
                            match_found = True
                            self.param_groups[group_name].append(param)
                            covered_names.append(name)
                    if not match_found:
                        logger.warning(f"No parameter matching keyword {submodule_name} in model")
            else:
                start = group_value.get("start")
                stop = group_value.get("stop")
                if start is None and stop is None:
                    raise ValueError(f"Invalid format for group {group_name}: {group_value}.")
                record = start is None
                stop_found = False
                for name, param in module.named_parameters():
                    if start and name.startswith(start):
                        record = True
                    elif stop_found and not name.startswith(stop):
                        break
                    if record:
                        self.param_groups[group_name].append(param)
                        covered_names.append(name)
                    if stop and name.startswith(stop):
                        stop_found = True
            if not self.param_groups[group_name]:
                logger.warning(f"Empty parameter group {group_name}.")

        # Check which model parameters are not covered by any group
        not_covered_count = 0
        first_not_covered_param = ""
        for name, _ in module.named_parameters():
            if name not in covered_names:
                if not_covered_count == 0:
                    first_not_covered_param = name
                not_covered_count += 1
        if not_covered_count > 0:
            logger.warning(
                f"{first_not_covered_param} does not belong to any parameter group "
                f"({not_covered_count-1} similar messages)."
            )

    def _set_optimizers(self) -> None:
        r"""Sets optimizers."""
        optim_config = self.config["optimizers"]
        for optim_name in optim_config:
            config = optim_config[optim_name]
            global_params = config.get("params")
            optim_fn = config["type"]
            if config.get("groups") is None:
                param_group = self.param_groups["main"]
                self.optimizers[optim_name] = getattr(torch.optim, optim_fn)(params=param_group, **global_params)
            else:
                optimizer_params = []
                for group_name, group_config in config["groups"].items():
                    if group_name not in self.param_groups.keys():
                        raise ValueError(f"Parameter group not found for optimizer {optim_name}: {group_name}")
                    optimizer_params.append({"params": self.param_groups[group_name], **group_config})
                self.optimizers[optim_name] = (
                    getattr(torch.optim, optim_fn)(optimizer_params, **global_params)
                    if global_params is not None
                    else getattr(torch.optim, optim_fn)(optimizer_params)
                )
            if config.get("scheduler") is not None:
                scheduler_type = config["scheduler"]["type"]
                scheduler_params = config["scheduler"].get("params")
                self.schedulers[optim_name] = (
                    getattr(torch.optim.lr_scheduler, scheduler_type)(
                        optimizer=self.optimizers[optim_name], **scheduler_params
                    )
                    if scheduler_params is not None
                    else getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer=self.optimizers[optim_name])
                )
            else:
                logger.warning(f"No scheduler defined for optimizer {optim_name}")

    def _set_periods(self) -> None:
        r"""Sets training periods."""
        num_epochs = self.config["num_epochs"]
        if self.config.get("periods") is None:
            # Single training period
            self.periods = {
                "main_period": {
                    "epoch_range": [0, num_epochs - 1],
                    "freeze": None,
                    "optimizers": self.optimizers.keys(),
                }
            }
        else:
            current_epoch = 0
            for period_name, period_config in self.config["periods"].items():
                # Update configuration if necessary
                if period_config.get("num_epochs") is not None:
                    if period_config.get("epoch_range") is not None:
                        # num_epochs parameter takes precedence over epoch_range
                        logger.warning(
                            f"Overwriting epoch range for period {period_name} based on provided value for num_epochs"
                        )
                    # Convert num_epochs into epoch_range
                    period_config["epoch_range"] = [
                        current_epoch,
                        current_epoch + period_config["num_epochs"] - 1,
                    ]
                    current_epoch += period_config["num_epochs"]
                    del period_config["num_epochs"]
                elif period_config.get("epoch_range") is None:
                    # Default epoch range
                    period_config["epoch_range"] = [current_epoch, num_epochs - 1]
                if period_config.get("optimizers") is None:
                    raise ValueError(f"Missing optimizer for training period {period_name}")
                self.periods[period_name] = period_config
                if isinstance(self.periods[period_name]["optimizers"], str):
                    self.periods[period_name]["optimizers"] = [self.periods[period_name]["optimizers"]]
                if isinstance(self.periods[period_name].get("freeze"), str):
                    self.periods[period_name]["freeze"] = [self.periods[period_name]["freeze"]]

                # Sanity checks
                epoch_range = self.periods[period_name]["epoch_range"]
                if not isinstance(epoch_range, list) or len(epoch_range) != 2:
                    raise ValueError(f"Invalid epoch range format for training period {period_name}: {epoch_range}")
                if self.periods[period_name].get("freeze") is not None:
                    for group_name in self.periods[period_name]["freeze"]:
                        if group_name not in self.param_groups.keys():
                            raise ValueError(f"Unknown parameter group for training period {period_name}: {group_name}")
                for optim_name in self.periods[period_name]["optimizers"]:
                    if optim_name not in self.optimizers.keys():
                        raise ValueError(f"Unknown optimizers name for training period {period_name}: {optim_name}")

            # Create periods for all non-covered epochs
            create_period = False
            full_train_period_idx = 0
            for epoch in range(num_epochs):
                # Extend search range to num_epochs + 1 to ensure that last period is created
                active_periods = self.get_active_periods(epoch)
                if not active_periods and not create_period:
                    create_period = True
                    self.periods[f"full_train_period_{full_train_period_idx}"] = {
                        "epoch_range": [epoch, -1],  # No freeze
                        "optimizers": list(self.optimizers.keys()),  # Enable all optimizers
                    }
                elif create_period and active_periods:
                    # Current full train period ended last epoch
                    self.periods[f"full_train_period_{full_train_period_idx}"]["epoch_range"][1] = epoch - 1
                    logger.info(
                        f"Creating full training period for epoch range "
                        f"{self.periods[f'full_train_period_{full_train_period_idx}']['epoch_range']}"
                    )
                    full_train_period_idx += 1
                    create_period = False
            # Complete final period (if any)
            if create_period:
                self.periods[f"full_train_period_{full_train_period_idx}"]["epoch_range"][1] = num_epochs - 1
                logger.info(
                    f"Creating full training period for epoch range "
                    f"{self.periods[f'full_train_period_{full_train_period_idx}']['epoch_range']}"
                )
        logger.info("Training periods")
        for period_name, period_config in self.periods.items():
            logger.info(
                f"+ Period {period_name}: "
                f"range [{period_config['epoch_range'][0]}-{period_config['epoch_range'][1]}], "
                f"applied on {period_config['optimizers']}"
            )

    def get_active_periods(self, epoch: int) -> list[str]:
        r"""Returns all active periods associated with a given epoch index.

        Args:
            epoch (int): Current epoch.

        Returns:
            List of period names.
        """
        p_names = []
        for p_name in self.periods:
            min_epoch, max_epoch = self.periods[p_name]["epoch_range"]
            if min_epoch <= epoch <= max_epoch:
                p_names.append(p_name)
        return p_names

    def freeze_group(self, name: str, freeze: bool) -> None:
        r"""Freezes all parameters of a given group.

        Args:
            name (str): Group name.
            freeze (bool): Whether this parameter should be frozen or unfrozen.
        """
        logger.debug(f"Parameter group {name} is {'frozen' if freeze  else 'trainable'}")
        for param in self.param_groups[name]:
            param.requires_grad = not freeze

    def freeze(self, epoch: int) -> None:
        r"""Applies parameter freeze depending on current epoch.

        Args:
            epoch (int): Current epoch.
        """
        groups_to_freeze = []
        for period_name in self.get_active_periods(epoch):
            period_config = self.periods[period_name]
            if period_config.get("freeze") is not None:
                groups_to_freeze += period_config["freeze"]
                logger.info(f"Period {period_name} applies for epoch {epoch}: freezing groups {groups_to_freeze}")

        for group_name in self.param_groups:
            self.freeze_group(name=group_name, freeze=(group_name in groups_to_freeze))

    def zero_grad(self):
        r"""Resets all optimizer gradients."""
        for name in self.optimizers:
            self.optimizers[name].zero_grad()

    def optimizer_step(self, epoch: int):
        r"""Applies optimizer step depending on current epoch.

        Args:
            epoch (int): Current epoch.
        """
        for period_name in self.get_active_periods(epoch):
            period_config = self.periods[period_name]
            for optim_name in period_config["optimizers"]:
                self.optimizers[optim_name].step()

    def scheduler_step(self, epoch: int):
        r"""Applies learning rate scheduler step depending on current epoch.

        Args:
            epoch (int): Current epoch.
        """
        for period_name in self.get_active_periods(epoch):
            period_config = self.periods[period_name]
            for optim_name in period_config["optimizers"]:
                # Not all optimizers are associated with a scheduler
                if self.schedulers.get(optim_name) is not None:
                    self.schedulers[optim_name].step()

    def to(self, device: str | torch.device):
        r"""Moves OptimizerManager to a given device.

        Args:
            device (str | device): Hardware device.
        """
        for optim_name in self.optimizers:
            move_optimizer_to(self.optimizers[optim_name], device)

    def state_dict(self) -> dict[str, Any]:
        r"""Returns the state of the Optimizer manager as a dictionary."""
        state = {"optimizers": {}, "schedulers": {}}
        for optim_name in self.optimizers:
            state["optimizers"][optim_name] = self.optimizers[optim_name].state_dict()
        for optim_name in self.schedulers:
            state["schedulers"][optim_name] = self.schedulers[optim_name].state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Loads state dictionary, restoring optimizers and schedulers to a previous state.

        Args:
            state_dict (dictionary): State dictionary.
        """
        for optim_name in self.optimizers:
            self.optimizers[optim_name].load_state_dict(state_dict["optimizers"][optim_name])
        for optim_name in self.schedulers:
            self.schedulers[optim_name].load_state_dict(state_dict["schedulers"][optim_name])

    DEFAULT_TRAINING_CONFIG: str = "training.yml"
    DEFAULT_TRAINING_STATE: str = "optimizer_state.pth"

    @staticmethod
    def create_parser(
        parser: argparse.ArgumentParser | None = None, mandatory_config: bool = False
    ) -> argparse.ArgumentParser:
        r"""Creates the argument parser for CaBRNet training configuration.

        Args:
            parser (ArgumentParser, optional): Existing parser (if any). Default: False.
            mandatory_config (bool, optional): If True, makes the configuration mandatory. Default: False.

        Returns:
            The parser itself.
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="Load training configuration.")

        parser.add_argument(
            "-t",
            "--training",
            type=str,
            required=mandatory_config,
            metavar="/path/to/file.yml",
            help="path to the training configuration file",
        )
        return parser
