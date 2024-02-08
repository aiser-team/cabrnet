---
sidebar_label: decision
title: protolib.prototree.decision
---

## ProtoTreeClassifier Objects

```python
class ProtoTreeClassifier(nn.Module)
```

#### \_\_init\_\_

```python
def __init__(num_classes: int, depth: int, num_features: int, leaves_init_mode: str = "ZEROS", proto_init_mode: str = "SHIFTED_NORMAL", log_probabilities: bool = False) -> None
```

Create a ProtoTree classifier

**Arguments**:

- `num_classes` - Number of classes
- `depth` - Depth of the binary decision tree
- `num_features` - Number of features (size of each prototype)
- `leaves_init_mode` - Init mode for leaves distributions
- `proto_init_mode` - Init mode for prototypes
- `log_probabilities` - Use log of probabilities

#### num\_prototypes

```python
@property
def num_prototypes() -> int
```

Returns: Total number of prototypes in the decision tree

#### forward

```python
def forward(features: Tensor, strategy: SamplingStrategy = SamplingStrategy.DISTRIBUTED) -> tuple[Tensor, dict] | None
```

Perform classification using decision tree

**Arguments**:

- `features` - Convolutional features from extractor. Shape (N, D, H, W)
- `strategy` - Sampling strategy
  

**Returns**:

  Vector of logits. Shape (N, C)

#### create\_parser

```python
@staticmethod
def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser
```

Add arguments for creating a ProtoTreeClassifier

**Arguments**:

- `parser` - Existing argument parser (if any)
  

**Returns**:

  Parser with arguments

