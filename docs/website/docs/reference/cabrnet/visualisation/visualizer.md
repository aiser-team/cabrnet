---
sidebar_label: visualizer
title: cabrnet.visualisation.visualizer
---

## SimilarityVisualizer Objects

```python
class SimilarityVisualizer(nn.Module)
```

#### \_\_init\_\_

```python
def __init__(retrace_fn: Callable,
             view_fn: Callable,
             retrace_params: dict | None = None,
             view_params: dict | None = None,
             config_file: str | None = None,
             *args,
             **kwargs)
```

Init a patch visualizer

**Arguments**:

- `retrace_fn` - visualization function
- `view_fn` - viewing function
- `retrace_params` - optional parameters to retrace function
- `view_params` - optional parameters to viewing function
- `config_file` - optional path to the file used to configure the visualizer

#### forward

```python
def forward(model: nn.Module,
            img: Image.Image,
            img_tensor: Tensor,
            proto_idx: int,
            device: str,
            location: tuple[int, int] | None = None) -> Image.Image
```

Generates a visualization of the most similar patch to a given prototype

**Arguments**:

- `model` - target model
- `img` - original image
- `img_tensor` - image tensor
- `proto_idx` - prototype index
- `device` - target hardware device
- `location` - location inside the similarity map
  

**Returns**:

  patch visualization

#### create\_parser

```python
@staticmethod
def create_parser(
        parser: argparse.ArgumentParser | None = None
) -> argparse.ArgumentParser
```

Create the argument parser for a ProtoVisualizer.

**Arguments**:

- `parser` - Existing parser (if any)
  

**Returns**:

  The parser itself.

#### build\_from\_config

```python
@staticmethod
def build_from_config(config_file: str,
                      target: str | None = None) -> SimilarityVisualizer
```

Builds a ProtoVisualizer from a YAML configuration file

**Arguments**:

- `config_file` - path to configuration file
- `target` - name of target in configuration file
  

**Returns**:

  ProtoVisualizer

