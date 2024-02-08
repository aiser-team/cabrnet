---
sidebar_label: view
title: protolib.visualisation.view
---

#### compute\_bbox

```python
def compute_bbox(array: np.ndarray, threshold: float) -> tuple[int, int, int, int]
```

Returns coordinates of the smallest bounding box such that all values outside the box are smaller or equal to threshold.

**Arguments**:

- `array` - target array
- `threshold` - threshold value
  

**Returns**:

  coordinates of the bounding box (x_min, x_max, y_min, y_max) where x and y correspond to the first and second
  dimension of the array

#### crop\_to\_percentile

```python
def crop_to_percentile(img: Image.Image, sim_map: np.ndarray, percentile: float) -> Image.Image
```

Crop an image based on a bounding box computed on the similarity map with a given percentile

**Arguments**:

- `img` - original image
- `sim_map` - similarity map
- `percentile` - crop percentile
  

**Returns**:

  cropped image

#### bbox\_to\_percentile

```python
def bbox_to_percentile(img: Image.Image, sim_map: np.ndarray, percentile: float, thickness: int = 2) -> Image.Image
```

Crop an image based on a bounding box computed on the similarity map with a given percentile

**Arguments**:

- `img` - original image
- `sim_map` - similarity map
- `percentile` - crop percentile
- `thickness` - rectangle thickness
  

**Returns**:

  image with bounding box

