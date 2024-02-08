---
sidebar_label: test_prototree_legacy
title: apps.test_prototree_legacy
---

#### compare\_inference

```python
def compare_inference(description: str, ref_model: nn.Module, test_model: nn.Module) -> None
```

Compare two models based on inference

**Arguments**:

- `description` - Test description for logger
- `ref_model` - Reference model
- `test_model` - Model under test

#### compare\_pruning

```python
def compare_pruning(ref_model: nn.Module, test_model: nn.Module) -> None
```

Compare two models based on pruning

**Arguments**:

- `ref_model` - Reference model
- `test_model` - Model under test

#### protolib\_process

```python
def protolib_process(model_config: str, dataset_config: str, training_config: str, visualization_config: str, legacy_state_dict: str | None, seed: int, verbose: bool, root_directory: str, device: str) -> dict
```

Builds, train, prune and perform projection on a ProtoLib tree

**Arguments**:

- `model_config` - Path to model configuration file
- `dataset_config` - Path to dataset configuration file
- `training_config` - Path to training configuration file
- `visualization_config` - Path to prototype visualization configuration file
- `legacy_state_dict` - Optional path to legacy state dictionary
- `seed` - Random seed
- `verbose` - Display progression bars
- `root_directory` - Output directory for prototypes
- `device` - target hardware device

**Returns**:

  dictionary of system states

#### legacy\_process

```python
def legacy_process(model_config: str, dataset_config: str, training_config: str, visualization_config: str, legacy_state_dict: str | None, seed: int, root_directory: str, device: str) -> dict
```

Builds, train, prune and perform projection on a ProtoTree

**Arguments**:

- `model_config` - Path to model configuration file
- `dataset_config` - Path to dataset configuration file
- `training_config` - Path to training configuration file
- `visualization_config` - Path to prototype visualization configuration file
- `legacy_state_dict` - Optional path to legacy state dictionary
- `seed` - Random seed
- `root_directory` - Output directory for prototypes
- `device` - Target hardware device

**Returns**:

  dictionary of system states

#### execute

```python
def execute(args: Namespace) -> None
```

Create Protolib model, then load a state dictionary in legacy ProtoTree form.

**Arguments**:

- `args` - Parsed arguments.

