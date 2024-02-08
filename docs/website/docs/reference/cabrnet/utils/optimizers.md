---
sidebar_label: optimizers
title: cabrnet.utils.optimizers
---

#### move\_optimizer\_to

```python
def move_optimizer_to(optim: torch.optim.Optimizer, device: str) -> None
```

Move optimizer to target device. Solution from https://github.com/pytorch/pytorch/issues/8741

**Arguments**:

- `optim` - Optimizer
- `device` - Target device

## OptimizerManager Objects

```python
class OptimizerManager()
```

#### \_\_init\_\_

```python
def __init__(config_dict: dict, module: nn.Module) -> None
```

Manager in charge of optimizers, learning rate schedulers and freezing parameters

**Arguments**:

- `config_dict` - configuration dictionary
- `module` - target model

#### build\_from\_config

```python
@staticmethod
def build_from_config(config_file: str, model: nn.Module) -> OptimizerManager
```

Build a OptimizerManager object from a YML file

**Arguments**:

- `config_file` - path to configuration file
- `model` - target model
  

**Returns**:

  OptimizerManager

#### get\_active\_periods

```python
def get_active_periods(epoch: int) -> list[str]
```

Get all active periods associated with a given epoch index

**Arguments**:

- `epoch` - current index
  

**Returns**:

  list of period names

#### freeze

```python
def freeze(epoch: int) -> None
```

Apply parameter freeze depending on current epoch

**Arguments**:

- `epoch` - current epoch

#### zero\_grad

```python
def zero_grad()
```

Reset all optimizer gradients

#### optimizer\_step

```python
def optimizer_step(epoch: int)
```

Apply optimizer step depending on current epoch

**Arguments**:

- `epoch` - current epoch

#### scheduler\_step

```python
def scheduler_step(epoch: int)
```

Apply learning rate scheduler step depending on current epoch

**Arguments**:

- `epoch` - current epoch

#### state\_dict

```python
def state_dict() -> dict[str, Any]
```

Returns the state of the Optimizer manager as a dictionary

