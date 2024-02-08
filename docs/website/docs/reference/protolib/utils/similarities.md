---
sidebar_label: similarities
title: protolib.utils.similarities
---

## L2Similarities Objects

```python
class L2Similarities(nn.Module)
```

#### \_\_init\_\_

```python
def __init__(num_prototypes: int, num_features: int, log_probabilities: bool = False) -> None
```

Create module for computing similarities based on L2 distance

**Arguments**:

- `num_prototypes` - Number of prototypes
- `num_features` - Size of each prototype
- `log_probabilities` - Return values as log of probabilities

#### forward

```python
def forward(features: Tensor, prototypes: Tensor) -> Tensor
```

Compute similarity based on L2 distance using ||x - y||² = ||x||² + ||y||² - 2 x.y

**Arguments**:

- `features` - Input tensor. Shape (N, D, H, W)
- `prototypes` - Tensor of prototypes. Shape (P, D, 1, 1)
  

**Returns**:

  Tensor of similarities. Shape (N, P, H, W)

