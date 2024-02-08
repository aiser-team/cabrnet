---
sidebar_label: prototypes
title: protolib.utils.prototypes
---

#### init\_prototypes

```python
def init_prototypes(num_prototypes: int, num_features: int, init_mode: str = "SHIFTED_NORMAL") -> Tensor
```

Create tensor of prototypes

**Arguments**:

- `num_prototypes` - Number of prototypes
- `num_features` - Size of each prototype
- `init_mode` - Initialisation mode (default: SHIFTED_NORMAL = N(0.5, 0.1))

