---
sidebar_label: save
title: cabrnet.utils.save
---

Implement the saving and loading capabilities for a generic ProtoClassifier.

#### save\_checkpoint

```python
def save_checkpoint(directory_path: str,
                    model: ProtoClassifier,
                    model_config: str,
                    optimizer_mngr: OptimizerManager | None,
                    training_config: str | None,
                    dataset_config: str,
                    epoch: int | str,
                    seed: int | None,
                    device: str,
                    stats: dict[str, Any] | None = None) -> None
```

Save everything needed to restart a training process.

**Arguments**:

- `directory_path` - Target location
- `model` - ProtoClassifier
- `model_config` - Path to the model configuration file
- `optimizer_mngr` - Optimizer manager
- `training_config` - Path to the training configuration file
- `dataset_config` - Path to the dataset configuration file
- `epoch` - Current epoch
- `seed` - Initial random seed (recorded for reproducibility)
- `device` - Target hardware device (recorded for reproducibility)
- `stats` - Other optional statistics

#### load\_checkpoint

```python
def load_checkpoint(
        directory_path: str,
        model: ProtoClassifier,
        optimizer_mngr: OptimizerManager | None = None) -> Mapping[str, Any]
```

Restore training process using checkpoint directory.

**Arguments**:

- `directory_path` - Target location
- `model` - ProtoClassifier
- `optimizer_mngr` - Optimizer manager
  

**Returns**:

  dictionary containing auxiliary state information (epoch, seed, device, stats)

