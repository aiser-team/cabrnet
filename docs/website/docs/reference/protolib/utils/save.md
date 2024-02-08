---
sidebar_label: save
title: protolib.utils.save
---

Implement the saving and loading capabilities for a generic ProtoClassifier.

#### save\_checkpoint

```python
def save_checkpoint(directory_path: str, model: ProtoClassifier, model_config: str, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler | None, training_config: str, dataset_config: str, epoch: int, stats: dict[str, Any] | None = None) -> None
```

Save everything needed to restart a training process.

**Arguments**:

- `directory_path` - Target location
- `model` - ProtoClassifier
- `model_config` - Path to the model configuration file
- `optimizer` - Optimizer
- `scheduler` - Scheduler
- `training_config` - Path to the training configuration file
- `dataset_config` - Path to the dataset configuration file
- `epoch` - Current epoch
- `stats` - Other optional statistics

#### load\_checkpoint

```python
def load_checkpoint(directory_path: str) -> Mapping[str, Any]
```

Restore training process using checkpoint directory.

**Arguments**:

- `directory_path` - Target location
  

**Returns**:

  dictionary containing checkpoint state (model, optimizer, scheduler, dataloaders, epoch, stats)

#### save\_model

```python
def save_model(directory_path: str, model: ProtoClassifier, model_config: str, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, training_config: str, dataset_config: str, epoch: int, best_train_acc: float, best_test_acc: float, leaf_labels: dict, checkpoint_frequency: int = 10) -> None
```

Save a tree and all its parameters.

**Arguments**:

- `directory_path` - Target location
- `model` - ProtoClassifier
- `model_config` - Path to the model configuration file
- `optimizer` - Optimizer
- `scheduler` - Scheduler
- `training_config` - Path to the training configuration file
- `dataset_config` - Path to the dataset configuration file
- `epoch` - Current epoch
- `best_train_acc` - Best train accuracy
- `best_test_acc` - Best test accuracy
- `leaf_labels` - Labels of the leaves
- `checkpoint_frequency` - Frequency to which to make checkpoints

#### save\_best\_train\_tree

```python
def save_best_train_tree(directory_path: str, model: ProtoClassifier, model_config: str, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, training_config: str, dataset_config: str, epoch: int, train_acc: float, best_train_acc: float, best_test_acc: float, leaf_labels: dict) -> float
```

Save the best tree at training and all its parameters.

**Arguments**:

- `directory_path` - Target location
- `model` - ProtoClassifier
- `model_config` - Path to the model configuration file
- `optimizer` - Optimizer
- `scheduler` - Scheduler
- `training_config` - Path to the training configuration file
- `dataset_config` - Path to the dataset configuration file
- `epoch` - Current epoch
- `train_acc` - Current train accuracy
- `best_train_acc` - Best train accuracy
- `best_test_acc` - Best test accuracy
- `leaf_labels` - Labels of the leaves

#### save\_best\_test\_tree

```python
def save_best_test_tree(directory_path: str, model: ProtoClassifier, model_config: str, optimizer, scheduler, training_config: str, dataset_config: str, epoch: int, best_train_acc: float, test_acc: float, best_test_acc: float, leaf_labels: dict) -> float
```

Save the best tree at testing and all its parameters.

**Arguments**:

- `directory_path` - Target location
- `model` - ProtoClassifier
- `model_config` - Path to the model configuration file
- `optimizer` - Optimizer
- `scheduler` - Scheduler
- `training_config` - Path to the training configuration file
- `dataset_config` - Path to the dataset configuration file
- `epoch` - Current epoch
- `best_train_acc` - Best train accuracy
- `test_acc` - Current test accuracy
- `best_test_acc` - Best test accuracy
- `leaf_labels` - Labels of the leaves

