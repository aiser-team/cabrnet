---
sidebar_label: postprocess
title: cabrnet.visualisation.postprocess
---

#### polarity\_and\_collapse

```python
def polarity_and_collapse(array: np.ndarray,
                          polarity: str | None = None,
                          dim: int | None = None) -> np.ndarray
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

#### post\_process

```python
def post_process(array: np.ndarray,
                 img: Image.Image,
                 img_tensor: Tensor,
                 resize: bool = True,
                 polarity: str | None = "absolute",
                 gaussian_ksize: int = 5,
                 normalize: bool = False,
                 grads_x_input: bool = False) -> np.array
```

Apply post-processing on numpy array

**Arguments**:

- `array` - source array
- `img` - raw input image
- `img_tensor` - input image tensor
- `resize` - resize array to original image size
- `polarity` - polarity filter (either None, &quot;absolute&quot;, &quot;positive&quot;, or &quot;negative&quot;)
- `gaussian_ksize` - size of gaussian filter kernel size
- `normalize` - perform min-max normalization
- `grads_x_input` - perform element-wise multiplication between gradient and image
  

**Returns**:

  result array

