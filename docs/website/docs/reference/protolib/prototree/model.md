---
sidebar_label: model
title: protolib.prototree.model
---

## ProtoTree Objects

```python
class ProtoTree(ProtoClassifier)
```

#### \_\_init\_\_

```python
def __init__(extractor: nn.Module, classifier: nn.Module, **kwargs)
```

Build a ProtoTree

**Arguments**:

- `extractor` - Feature extractor
- `classifier` - Classification based on extracted features

#### get\_extra\_state

```python
def get_extra_state() -> Mapping[str, Any]
```

Decision tree architecture to be saved in state_dict.
This is automatically called by state_dict()

#### set\_extra\_state

```python
def set_extra_state(state: Mapping[str, Any]) -> None
```

Rebuild decision tree from architecture information
This is automatically called by load_state_dict()

**Arguments**:

- `state` - information returned by get_extra_state()

#### load\_legacy\_state\_dict

```python
def load_legacy_state_dict(legacy_state: dict) -> None
```

Load state dictionary from legacy format

**Arguments**:

- `legacy_state` - Legacy state dictionary
  

**Raises**:

  ValueError when keys or tensor sizes mismatch.

#### analyse\_leafs

```python
def analyse_leafs(pruning_threshold: float = 0.01) -> None
```

Analyse leaf distributions.

**Arguments**:

- `pruning_threshold` - Expected pruning threshold.

#### loss

```python
def loss(model_output: Any, label: torch.Tensor) -> tuple[torch.Tensor, float]
```

Loss function

**Arguments**:

- `model_output` - Model output, in this case a tuple containing the prediction and the leaf probabilities
- `label` - Batch labels
  

**Returns**:

  loss tensor and batch accuracy

#### train\_epoch

```python
def train_epoch(train_loader: DataLoader, optimizer: Optimizer, device: str = "cuda:0", progress_bar_position: int = 0, epoch_idx: int = 0, verbose: bool = False, max_batches: int | None = None) -> dict[str, float]
```

Train the model for one epoch.

**Arguments**:

- `train_loader` - Dataloader containing training data
- `optimizer` - Learning optimizer
- `device` - Target device
- `progress_bar_position` - Position of the progress bar.
- `epoch_idx` - Epoch index
- `max_batches` - Max number of batches (early stop for small compatibility tests)
- `verbose` - Display progress bar
  

**Returns**:

  dictionary containing learning statistics

#### epilogue

```python
def epilogue(pruning_threshold: float = 0.0) -> None
```

Function called after training, using information from the epilogue
field in the training configuration

**Arguments**:

- `pruning_threshold` - Pruning threshold

#### prune

```python
def prune(pruning_threshold: float = 0.01) -> None
```

Prune decision tree based on threshold.

**Arguments**:

- `pruning_threshold` - Pruning threshold

#### project

```python
def project(data_loader: DataLoader, device: str = "cuda:0", verbose: bool = False, progress_bar_position: int = 0) -> dict[int, dict]
```

Perform prototype projection after training

**Arguments**:

- `data_loader` - Dataloader containing projection data. WARNING: This dataloader must not be shuffled!
- `device` - Target device
- `verbose` - Display progress bar
- `progress_bar_position` - Position of the progress bar.

**Returns**:

  dictionary containing projection information for each prototype

#### explain

```python
def explain(img_path: str, preprocess: Callable, visualizer: SimilarityVisualizer, prototype_dir_path: str, output_dir_path: str, device: str, exist_ok: bool = False, strategy: SamplingStrategy = SamplingStrategy.GREEDY) -> None
```

Explain the decision for a particular image

**Arguments**:

- `img_path` - raw original image
- `preprocess` - preprocessing function
- `visualizer` - prototype visualizer
- `prototype_dir_path` - path to directory containing prototype visualizations
- `output_dir_path` - path to output directory containing the explanation
- `device` - target hardware device
- `exist_ok` - silently overwrite existing explanation if any
- `strategy` - tree sampling strategy

#### explain\_global

```python
def explain_global(prototype_dir_path: str, output_dir_path: str, **kwargs, ,) -> None
```

Explain the global decision-making process

**Arguments**:

- `prototype_dir_path` - path to directory containing prototype visualizations
- `output_dir_path` - path to output directory containing the explanations

