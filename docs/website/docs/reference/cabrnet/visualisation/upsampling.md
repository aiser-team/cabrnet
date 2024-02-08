---
sidebar_label: upsampling
title: cabrnet.visualisation.upsampling
---

#### cubic\_upsampling

```python
def cubic_upsampling(model: nn.Module,
                     img: Image.Image,
                     img_tensor: Tensor,
                     proto_idx: int,
                     device: str,
                     location: tuple[int, int] | None = None,
                     single_location: bool = True,
                     normalize: bool = False) -> np.array
```

Perform patch visualization using cubic interpolation

**Arguments**:

- `model` - target model
- `img` - raw input image
- `img_tensor` - input image tensor
- `proto_idx` - prototype index
- `device` - target hardware device
- `location` - coordinates of feature vector
- `single_location` - keep only a single location
- `normalize` - perform min-max normalization
  

**Returns**:

  upsampled similarity map

