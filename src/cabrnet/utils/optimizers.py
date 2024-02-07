from __future__ import annotations
import torch
import torch.nn as nn
from loguru import logger
from typing import Any
from cabrnet.utils.parser import load_config


def move_optimizer_to(optim: torch.optim.Optimizer, device: str) -> None:
    """
    Move optimizer to target device. Solution from https://github.com/pytorch/pytorch/issues/8741
    Args:
        optim: Optimizer
        device: Target device
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
    def __init__(self, config_dict: dict, module: nn.Module) -> None:
        """Manager in charge of optimizers, learning rate schedulers and freezing parameters

        Args:
            config_dict: configuration dictionary
            module: target model

        """
        self.config = config_dict

        self.param_groups: dict[str, list[nn.Parameter]] = {}
        self.optimizers: dict[str, torch.optim.optimizer] = {}
        self.schedulers: dict[str, torch.optim.lr_scheduler] = {}
        self.periods: dict[str, dict[str, Any]] = {}

        self._set_param_groups(module=module)
        self._set_optimizers()
        self._set_periods()

    @staticmethod
    def build_from_config(config_file: str, model: nn.Module) -> OptimizerManager:
        """Build a OptimizerManager object from a YML file

        Args:
            config_file: path to configuration file
            model: target model

        Returns:
            OptimizerManager
        """
        return OptimizerManager(config_dict=load_config(config_file), module=model)

    def _set_param_groups(self, module: nn.Module) -> None:
        """
        Build the groups of parameters from the configuration dictionary.
        """
        param_groups = self.config.get("param_groups")

        if param_groups is None:
            # Create a default param group encompassing all model parameters
            self.param_groups = {"main": [param for param in module.named_parameters()]}
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
        """
        Set optimizers.
        """

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
        """
        Set training periods.
        """
        num_epochs = self.config["num_epochs"]
        if self.config.get("periods") is None:
            # Single training period
            self.periods = {
                "main_period": {"range": [0, num_epochs - 1], "freeze": None, "optimizers": self.optimizers.keys()}
            }
        else:
            for epoch_name in self.config.get("periods"):
                epoch_config = self.config["periods"][epoch_name]
                if epoch_config.get("epoch_range") is None or epoch_config.get("optimizers") is None:
                    raise ValueError(f"Missing parameters for training period {epoch_name}")
                self.periods[epoch_name] = epoch_config
                if isinstance(self.periods[epoch_name]["optimizers"], str):
                    self.periods[epoch_name]["optimizers"] = [self.periods[epoch_name]["optimizers"]]
                if isinstance(self.periods[epoch_name].get("freeze"), str):
                    self.periods[epoch_name]["freeze"] = [self.periods[epoch_name]["freeze"]]

                # Sanity checks
                epoch_range = self.periods[epoch_name]["epoch_range"]
                if not isinstance(epoch_range, list) or len(epoch_range) != 2:
                    raise ValueError(f"Invalid epoch range format for training period {epoch_name}: {epoch_range}")
                for group_name in self.periods[epoch_name].get("freeze"):
                    if group_name not in self.param_groups.keys():
                        raise ValueError(f"Unknown parameter group for training period {epoch_name}: {group_name}")
                for optim_name in self.periods[epoch_name]["optimizers"]:
                    if optim_name not in self.optimizers.keys():
                        raise ValueError(f"Unknown optimizers name for training period {epoch_name}: {optim_name}")

    def freeze(self, epoch: int) -> None:
        """Apply parameter freeze depending on current epoch

        Args:
            epoch: current epoch
        """
        groups_to_freeze = []
        for period_name in self.periods:
            period_config = self.periods[period_name]
            min_epoch, max_epoch = period_config["epoch_range"]
            if epoch in range(min_epoch, max_epoch + 1) and period_config.get("freeze") is not None:
                groups_to_freeze += period_config["freeze"]
                logger.info(f"Period {period_name} applies for epoch {epoch}: freezing groups {groups_to_freeze}")

        def _set_requires_grad_on_group(name: str, requires_grad: bool):
            logger.debug(f"Epoch {epoch}: Parameter group {name} is {'trainable' if requires_grad else 'frozen'}")
            for param in self.param_groups[name]:
                param.requires_grad = requires_grad

        for group_to_freeze in groups_to_freeze:
            _set_requires_grad_on_group(name=group_to_freeze, requires_grad=False)
        for group_name in self.param_groups:
            if group_name not in groups_to_freeze:
                _set_requires_grad_on_group(name=group_name, requires_grad=True)

    def zero_grad(self):
        """Reset all optimizer gradients"""
        for name in self.optimizers:
            self.optimizers[name].zero_grad()

    def optimizer_step(self, epoch: int):
        """Apply optimizer step depending on current epoch

        Args:
            epoch: current epoch
        """
        active_period = False  # Is one specific training period active?
        for period_name in self.periods:
            period_config = self.periods[period_name]
            min_epoch, max_epoch = period_config["epoch_range"]
            if epoch in range(min_epoch, max_epoch + 1):
                active_period = True
                for optim_name in period_config["optimizers"]:
                    self.optimizers[optim_name].step()

        # No active period found. By default, update all optimizers
        if not active_period:
            for optim_name in self.optimizers:
                self.optimizers[optim_name].step()

    def scheduler_step(self, epoch: int):
        """Apply learning rate scheduler step depending on current epoch

        Args:
            epoch: current epoch
        """
        active_period = False  # Is one specific training period active?
        for period_name in self.periods:
            period_config = self.periods[period_name]
            min_epoch, max_epoch = period_config["epoch_range"]
            if epoch in range(min_epoch, max_epoch + 1):
                active_period = True
                for optim_name in period_config["optimizers"]:
                    # Not all optimizers are associated with a scheduler
                    if self.schedulers.get(optim_name) is not None:
                        self.schedulers[optim_name].step()

        # No active period found. By default, update all schedulerts
        if not active_period:
            for optim_name in self.schedulers:
                self.schedulers[optim_name].step()

    def to(self, device: str):
        for optim_name in self.optimizers:
            move_optimizer_to(self.optimizers[optim_name], device)

    def state_dict(self) -> dict[str, Any]:
        """Returns the state of the Optimizer manager as a dictionary"""
        state = {"optimizers": {}, "schedulers": {}}
        for optim_name in self.optimizers:
            state["optimizers"][optim_name] = self.optimizers[optim_name].state_dict()
        for optim_name in self.schedulers:
            state["schedulers"][optim_name] = self.schedulers[optim_name].state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for optim_name in self.optimizers:
            self.optimizers[optim_name].load_state_dict(state_dict["optimizers"][optim_name])
        for optim_name in self.schedulers:
            self.schedulers[optim_name].load_state_dict(state_dict["schedulers"][optim_name])
