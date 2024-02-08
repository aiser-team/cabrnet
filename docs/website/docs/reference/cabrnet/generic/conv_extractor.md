---
sidebar_label: conv_extractor
title: cabrnet.generic.conv_extractor
---

## ConvExtractor Objects

```python
class ConvExtractor(nn.Module)
```

Class representing the feature extractor.

**Attributes**:

- `arch_name` - Architecture name.
- `weights` - Weights for the network.
- `layer` - Layer to inspect.
- `convnet` - Graph module that represents the intermediate nodes from the given model.
- `add_on` - Add-on layers configuration.
- `output_channels` - Number of output channels of the feature extractor

#### \_\_init\_\_

```python
def __init__(arch: str,
             weights: str | None,
             layer: str,
             add_on: dict[str, dict],
             seed: int | None = None) -> None
```

Initialize ConvExtractor.

**Arguments**:

- `arch` - Architecture name.
- `weights` - Weights for the network.
- `layer` - Layer to inspect.
- `add_on` - Add-on layers configuration.
- `seed` - Random seed (used only to resynchronise random number generators in compatibility tests)

#### forward

```python
def forward(x: torch.Tensor, **kwargs) -> torch.Tensor
```

Define the computation performed at every call.

**Arguments**:

- `x` - Tensor to run the model on.
  

**Returns**:

  The tensor resulting from the inference of the model.

#### create\_add\_on

```python
@staticmethod
def create_add_on(config: dict[str, dict],
                  in_channels: int) -> Tuple[nn.Sequential, int]
```

Build add-on layers based on configuration.

**Arguments**:

- `config` - Add-on layers configuration.
- `in_channels` - Number of input channels (as given by feature extractor).
  

**Returns**:

  Module containing all add-on layers
  

**Raises**:

  ValueError when configuration is invalid

#### build\_from\_dict

```python
@staticmethod
def build_from_dict(config: dict[str, dict],
                    seed: int | None = None) -> nn.Module
```

Builds a ConvExtractor from a configuration dictionary.

**Arguments**:

- `config` - Configuration dictionary
- `seed` - Random seed (used only to resynchronise random number generators in compatibility tests)

**Returns**:

  ConvExtractor
  

**Raises**:

  ValueError when configuration is invalid

