---
sidebar_label: gradients
title: cabrnet.visualisation.gradients
---

#### smoothgrad

```python
def smoothgrad(model: nn.Module,
               img: Image.Image,
               img_tensor: Tensor,
               proto_idx: int,
               device: str,
               location: tuple[int, int] | None = None,
               single_location: bool = True,
               polarity: str | None = "absolute",
               gaussian_ksize: int = 5,
               normalize: bool = False,
               num_samples: int = 10,
               noise_ratio: float = 0.2,
               grads_x_input: bool = False) -> np.array
```

Perform patch visualization using SmoothGrad (https://arxiv.org/abs/1706.03825)

**Arguments**:

- `model` - target model
- `img` - raw input image
- `img_tensor` - input image tensor
- `proto_idx` - prototype index
- `device` - target hardware device
- `location` - coordinates of feature vector
- `single_location` - keep only a single location
- `polarity` - polarity filter (either None, &quot;absolute&quot;, &quot;positive&quot;, or &quot;negative&quot;)
- `gaussian_ksize` - size of gaussian filter kernel size
- `normalize` - perform min-max normalization
- `img`0 - number of random samples
- `img`1 - noise ratio for random samples
- `img`2 - perform element-wise multiplication between gradient and image
  

**Returns**:

  similarity map

#### randgrad

```python
def randgrad(model: nn.Module,
             img: Image.Image,
             img_tensor: Tensor,
             proto_idx: int,
             device: str,
             location: tuple[int, int] | None = None,
             polarity: str | None = "absolute",
             gaussian_ksize: int = 5,
             normalize: bool = False,
             grads_x_input: bool = False) -> np.array
```

Return random patch visualization (used as a baseline for evaluating properties of other retracing functions)

**Arguments**:

- `model` - target model
- `img` - raw input image
- `img_tensor` - input image tensor
- `proto_idx` - prototype index
- `device` - target hardware device
- `location` - coordinates of feature vector
- `polarity` - polarity filter (either None, &quot;absolute&quot;, &quot;positive&quot;, or &quot;negative&quot;)
- `gaussian_ksize` - size of gaussian filter kernel size
- `normalize` - perform min-max normalization
- `grads_x_input` - perform element-wise multiplication between gradient and image
  

**Returns**:

  similarity map

#### prp

```python
def prp(model: nn.Module,
        img: Image.Image,
        img_tensor: Tensor,
        proto_idx: int,
        device: str,
        location: tuple[int, int] | None = None,
        single_location: bool = True,
        stability_factor: float = 1e-6,
        polarity: str | None = "absolute",
        gaussian_ksize: int = 5,
        normalize: bool = False,
        grads_x_input: bool = False) -> np.array
```

Perform patch visualization using Prototype Relevance Propagation
(https://www.sciencedirect.com/science/article/pii/S0031320322006513#bib0030)

**Arguments**:

- `model` - target model
- `img` - raw input image
- `img_tensor` - input image tensor
- `proto_idx` - prototype index
- `device` - target hardware device
- `location` - coordinates of feature vector
- `single_location` - keep only a single location
- `stability_factor` - LRP stability factor (epsilon)
- `polarity` - polarity filter (either None, &quot;absolute&quot;, &quot;positive&quot;, or &quot;negative&quot;)
- `gaussian_ksize` - size of gaussian filter kernel size
- `img`0 - perform min-max normalization
- `img`1 - perform element-wise multiplication between gradient and image

**Returns**:

  similarity map

