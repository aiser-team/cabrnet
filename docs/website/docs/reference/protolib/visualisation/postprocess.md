---
sidebar_label: postprocess
title: protolib.visualisation.postprocess
---

#### polarity\_and\_collapse

```python
def polarity_and_collapse(array: np.ndarray, polarity: str | None = None, dim: int | None = None) -> np.ndarray
```

Apply polarity filter (optional) followed by average over channels (optional)

**Arguments**:

- `array` - target array
- `polarity` - polarity (positive, negative, absolute)
- `dim` - dimension across which channels are averaged
  

**Returns**:

  modified array

#### normalize\_min\_max

```python
def normalize_min_max(array: np.ndarray) -> np.ndarray
```

Perform min-max normalization of a numpy array

**Arguments**:

- `array` - target array
  

**Returns**:

  normalized array

