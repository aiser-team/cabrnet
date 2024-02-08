---
sidebar_label: explainer
title: protolib.visualisation.explainer
---

## ExplanationGraph Objects

```python
class ExplanationGraph()
```

#### \_\_init\_\_

```python
def __init__(output_dir: str) -> None
```

Init explanation

**Arguments**:

- `output_dir` - path to output directory containing explanation

#### set\_test\_image

```python
def set_test_image(img_path: str) -> None
```

Set test image

**Arguments**:

- `img_path` - path to image

#### render

```python
def render() -> None
```

Generate explanation file

