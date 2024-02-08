---
sidebar_label: view
title: cabrnet.visualisation.view
---

#### compute\_bbox

```python
def compute_bbox(array: np.ndarray,
                 threshold: float) -> tuple[int, int, int, int]
```

Returns coordinates of the smallest bounding box such that all values outside the box
are smaller or equal to threshold.

**Arguments**:

- `array` - target array
- `threshold` - threshold value
  

**Returns**:

  coordinates of the bounding box (x_min, x_max, y_min, y_max) where x and y correspond to the first and second
  dimension of the array

#### crop\_to\_percentile

```python
def crop_to_percentile(img: Image.Image, sim_map: np.ndarray,
                       percentile: float, **kwargs) -> Image.Image
```

Crop an image based on a bounding box computed on the similarity map with a given percentile

**Arguments**:

- `img` - original image
- `sim_map` - similarity map, normalized between 0.0 and 1.0
- `percentile` - crop percentile
  

**Returns**:

  cropped image

#### bbox\_to\_percentile

```python
def bbox_to_percentile(img: Image.Image,
                       sim_map: np.ndarray,
                       percentile: float,
                       thickness: int = 4,
                       **kwargs) -> Image.Image
```

Show image with a bounding box computed on the similarity map with a given percentile

**Arguments**:

- `img` - original image
- `sim_map` - similarity map, normalized between 0.0 and 1.0
- `percentile` - crop percentile
- `thickness` - rectangle thickness
  

**Returns**:

  image with bounding box

#### heatmap

```python
def heatmap(img: Image.Image, sim_map: np.ndarray, **kwargs) -> Image.Image
```

Convert a similarity map into a heatmap

**Arguments**:

- `img` - original image
- `sim_map` - similarity map, normalized between 0.0 and 1.0
  

**Returns**:

  image with bounding box

