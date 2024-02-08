---
sidebar_label: hacks
title: protolib.utils.hacks
---

#### optimizer\_to

```python
def optimizer_to(optim: Optimizer, device: str) -> None
```

Move optimizer to target device. Solution from https://github.com/pytorch/pytorch/issues/8741

**Arguments**:

- `optim` - Optimizer
- `device` - Target device

