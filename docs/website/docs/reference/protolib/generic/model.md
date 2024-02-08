---
sidebar_label: model
title: protolib.generic.model
---

## ProtoClassifier Objects

```python
class ProtoClassifier(nn.Module)
```

#### \_\_init\_\_

```python
def __init__(extractor: nn.Module, classifier: nn.Module, compatibility_mode: bool = False)
```

Build a generic prototype-based classifier

**Arguments**:

- `extractor` - Feature extractor
- `classifier` - Classification based on extracted features
- `compatibility_mode` - Compatibility mode with legacy architectures. \
  When enabled, batch_norm running parameters are not &quot;properly&quot; frozen, ie they are updated during the
  forward-pass even if the backbone parameters should not be modified.

#### similarities

```python
def similarities(x: Tensor, **kwargs) -> Tensor
```

Return similarity scores

**Arguments**:

- `x` - input tensor
  

**Returns**:

  tensor of similarity scores

#### load\_legacy\_state\_dict

```python
def load_legacy_state_dict(legacy_state: dict) -> None
```

Load state dictionary from legacy format

**Arguments**:

- `legacy_state` - Legacy state dictionary

#### create\_parser

```python
@staticmethod
def create_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser
```

Create the argument parser for a ProtoClassifier.

**Arguments**:

- `parser` - Existing parser (if any)
  

**Returns**:

  The parser itself.

#### build\_from\_config

```python
@staticmethod
def build_from_config(config_file: str, seed: int | None = None, compatibility_mode: bool = False, state_dict_path: str | None = None) -> ProtoClassifier
```

Builds a ProtoClassifier from a YAML configuration file

**Arguments**:

- `config_file` - path to configuration file
- `seed` - random seed (used only to resynchronise random number generators in compatibility tests)
- `compatibility_mode` - compatibility mode with legacy architectures
- `state_dict_path` - path to model state dictionary
  

**Returns**:

  ProtoClassifier

#### loss

```python
def loss(model_output: Any, label: torch.Tensor) -> tuple[torch.Tensor, float]
```

Computes the loss and the accuracy over a batch of model outputs

**Arguments**:

- `model_output` - Model output
- `label` - Batch label
  

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
- `verbose` - Display progress bar
- `max_batches` - Max number of batches (early stop for small compatibility tests)
  

**Returns**:

  dictionary containing learning statistics

#### epilogue

```python
def epilogue(**kwargs) -> None
```

Function called after training, using information from the epilogue
field in the training configuration

#### evaluate

```python
def evaluate(dataloader: DataLoader, device: str = "cuda:0", progress_bar_position: int = 0, verbose: bool = False) -> dict[str, float]
```

Evaluate the model.

**Arguments**:

- `dataloader` - Dataloader containing evaluation data
- `device` - Target device
- `progress_bar_position` - Position of the progress bar
- `verbose` - Display progress bar
  

**Returns**:

  dictionary containing evaluation statistics

#### train

```python
def train(mode: bool = True) -> nn.Module
```

Overwrite train() function to freeze elements if necessary

**Arguments**:

- `mode`: Train (true) or eval (false)

#### project

```python
def project(data_loader: DataLoader, device: str = "cuda:0", verbose: bool = False, progress_bar_position: int = 0) -> dict[int, dict]
```

Perform prototype projection after training

**Arguments**:

- `data_loader` - dataloader containing projection data
- `device` - target device
- `verbose` - display progress bar
- `progress_bar_position` - position of the progress bar.

**Returns**:

  dictionary containing projection information for each prototype

#### extract\_prototypes

```python
def extract_prototypes(dataloader_raw: DataLoader, dataloader: DataLoader, projection_info: dict[int, dict], visualizer: SimilarityVisualizer, dir_path: str, device: str, verbose: bool = False, progress_bar_position: int = 0) -> None
```

Show prototypes based on projection info

**Arguments**:

- `dataloader_raw` - dataloader containing raw projection images (without preprocessing)
- `dataloader` - dataloader containing projection tensors (with preprocessing)
- `projection_info` - projection information (as returned by project method)
- `visualizer` - similarity visualizer
- `dir_path` - destination directory
- `device` - target hardware device
- `verbose` - display progress bar
- `progress_bar_position` - position of the progress bar.

#### explain

```python
def explain(img_path: str, preprocess: Callable, visualizer: SimilarityVisualizer, prototype_dir_path: str, output_dir_path: str, device: str, exist_ok: bool = False, **kwargs, ,) -> None
```

Explain the decision for a particular image

**Arguments**:

- `img_path` - path to raw original image
- `preprocess` - preprocessing function
- `visualizer` - prototype visualizer
- `prototype_dir_path` - path to directory containing prototype visualizations
- `output_dir_path` - path to output directory containing the explanation
- `device` - target hardware device
- `exist_ok` - silently overwrite existing explanation if any

#### explain\_global

```python
def explain_global(prototype_dir_path: str, output_dir_path: str, **kwargs, ,) -> None
```

Explain the global decision-making process

**Arguments**:

- `prototype_dir_path` - path to directory containing prototype visualizations
- `output_dir_path` - path to output directory containing the explanations

