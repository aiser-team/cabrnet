---
sidebar_label: parser
title: protolib.utils.parser
---

This file contains all the necessary tools to parse the various config files.

#### load\_config

```python
def load_config(config_file: str) -> dict
```

Load a configuration file for ProtoLib.

**Arguments**:

- `config_file` - Path to the configuration file.
  

**Returns**:

  The properly loaded config file.
  

**Raises**:

- `ValueError` - The config file is in an unsupported file format.

#### get\_param\_groups

```python
def get_param_groups(trainer: dict, model: nn.Module) -> dict[str, list[nn.Parameter]]
```

Build the groups of parameters for the training.

**Arguments**:

- `trainer` - Options to build the groups.
- `model` - Model to take the parameters from.
  

**Returns**:

  The groups of parameters.

#### get\_optimizer

```python
def get_optimizer(trainer: dict, param_groups: dict[str, list[nn.Parameter]]) -> optim.Optimizer
```

Build the optimizer.

**Arguments**:

- `trainer` - Options to build the optimizer.
- `param_groups` - Groups of parameters.
  

**Returns**:

  Initialized optimizer.

#### get\_scheduler

```python
def get_scheduler(trainer: dict, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler | None
```

Build the learning rate scheduler for the optimizer.

**Arguments**:

- `trainer` - Training configuration
- `optimizer` - Target optimizer
  

**Returns**:

  Initialized LR scheduler

#### freeze

```python
def freeze(epoch: int, param_groups: dict, trainer: dict) -> None
```

Freeze the parameter groups declared in the training configuration file at the current epoch.

**Arguments**:

- `epoch` - Current epoch.
- `param_groups` - Parameter groups defined in the training configuration file.
- `trainer` - Dictionary from training configuration file.

#### create\_training\_parser

```python
def create_training_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser
```

Create the argument parser for ProtoLib training configuration.

**Arguments**:

- `parser` - Existing parser (if any)

**Returns**:

  The parser itself.

