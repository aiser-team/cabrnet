---
sidebar_label: prp_utils
title: cabrnet.visualisation.prp_utils
---

## L2SimilaritiesLRPWrapper Objects

```python
class L2SimilaritiesLRPWrapper(L2Similarities)
```

Replacement for the L2 similarity layer

**Arguments**:

- `num_prototypes` - Number of prototypes
- `num_features` - Size of each prototype
- `stability_factor` _float_ - epsilon value used for numerical stability

## DecisionLRPWrapper Objects

```python
class DecisionLRPWrapper(nn.Module)
```

Replacement for the decision layer of a ProtoClassifier

**Arguments**:

- `classifier` _nn.Module_ - source decision layer
- `stability_factor` _float_ - epsilon value used for numerical stability

## GradientRule Objects

```python
class GradientRule(PropagationRule)
```

Dummy propagation rule used when ignoring LRP and using gradients directly

## ZBetaLayer Objects

```python
class ZBetaLayer(ABC)
```

Abstract layer for applying Z^Beta rule during relevance backpropagation.
For more information, see https://arxiv.org/pdf/1512.02479.pdf

Use for first (pixel) layer.

**Arguments**:

- `set_bias_to_zero` _bool_ - ignore bias during backpropagation
- `lower_bound` _float_ - smallest admissible pixel value
- `upper_bound` _float_ - largest admissible pixel value
- `stability_factor` _float_ - epsilon value used for numerical stability
  
- `Note` - admissible range for pixel values must take into account input normalization

## ZBetaLinear Objects

```python
class ZBetaLinear(ZBetaLayer, nn.Linear)
```

Replacement layer for applying Z^Beta rule on a nn.Linear operator during relevance backpropagation.
For more information, see https://arxiv.org/pdf/1512.02479.pdf

Use for first (pixel) layer.

**Arguments**:

- `module` _nn.Module_ - layer to be replaced
- `set_bias_to_zero` _bool_ - ignore bias during backpropagation
- `lower_bound` _float_ - smallest admissible pixel value
- `upper_bound` _float_ - largest admissible pixel value
- `stability_factor` _float_ - epsilon value used for numerical stability
  
- `Note` - admissible range for pixel values must take into account input normalization

## ZBetaConv2d Objects

```python
class ZBetaConv2d(ZBetaLayer, nn.Conv2d)
```

Replacement layer for applying Z^Beta rule on a nn.Conv2d operator during relevance backpropagation.
For more information, see https://arxiv.org/pdf/1512.02479.pdf

Use for first (pixel) layer.

**Arguments**:

- `module` _nn.Module_ - layer to be replaced
- `set_bias_to_zero` _bool_ - ignore bias during backpropagation
- `lower_bound` _float_ - smallest admissible pixel value
- `upper_bound` _float_ - largest admissible pixel value
- `stability_factor` _float_ - epsilon value used for numerical stability
  
- `Note` - admissible range for pixel values must take into account input normalization

## Alpha1Beta0Layer Objects

```python
class Alpha1Beta0Layer(ABC)
```

Abstract layer for applying Alpha1-Beta0 rule during relevance backpropagation.
For more information, see https://arxiv.org/pdf/1512.02479.pdf

Use for convolution layers.

**Arguments**:

- `set_bias_to_zero` _bool_ - ignore bias during backpropagation
- `stability_factor` _float_ - epsilon value used for numerical stability

## Alpha1Beta0Linear Objects

```python
class Alpha1Beta0Linear(Alpha1Beta0Layer, nn.Linear)
```

Replacement layer for applying Alpha1-Beta0 rule on a nn.Linear operator during relevance backpropagation.
For more information, see https://arxiv.org/pdf/1512.02479.pdf

Use for convolution layers.

**Arguments**:

- `module` _nn.Module_ - layer to be replaced
- `set_bias_to_zero` _bool_ - ignore bias during backpropagation
- `stability_factor` _float_ - epsilon value used for numerical stability

## Alpha1Beta0Conv2d Objects

```python
class Alpha1Beta0Conv2d(Alpha1Beta0Layer, nn.Conv2d)
```

Replacement layer for applying Alpha1-Beta0 rule on a nn.Conv2d operator during relevance backpropagation.
For more information, see https://arxiv.org/pdf/1512.02479.pdf

Use for convolution layers.

**Arguments**:

- `module` _nn.Module_ - layer to be replaced
- `set_bias_to_zero` _bool_ - ignore bias during backpropagation
- `stability_factor` _float_ - epsilon value used for numerical stability

## StackedSum Objects

```python
class StackedSum(nn.Module)
```

Replacement layer used in order to enforce Epsilon-rule through the addition operator inside residual blocks.
For more information, see https://github.com/AlexBinder/LRP_Pytorch_Resnets_Densenet/blob/master/lrp_general6.py

Use for residual blocks (eg. ResNet).

**Arguments**:

- `stability_factor` _float_ - epsilon value used for numerical stability

#### get\_extractor\_lrp\_composite\_model

```python
def get_extractor_lrp_composite_model(
    model: nn.Module,
    set_bias_to_zero: bool,
    stability_factor: float = 1e-6,
    use_zbeta: bool = True,
    zbeta_lower_bound: float = min(
        [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]),
    zbeta_upper_bound: float = max([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224,
                                    (1 - 0.406) / 0.225])
) -> nn.Module
```

Prepare a feature extractor for composite LRP

**Arguments**:

- `model` - target model
- `set_bias_to_zero` - ignore bias in linear layers
- `stability_factor` _float_ - epsilon value used for numerical stability
- `use_zbeta` - use z-beta rule on first convolution
- `zbeta_lower_bound` _float_ - smallest admissible pixel value (used in z-beta rule)
- `zbeta_upper_bound` _float_ - largest admissible pixel value (used in z-beta rule)
  

**Returns**:

  copy of the model, ready for running Captum LRP

#### get\_cabrnet\_lrp\_composite\_model

```python
def get_cabrnet_lrp_composite_model(
    model: nn.Module,
    set_bias_to_zero: bool = True,
    stability_factor: float = 1e-6,
    use_zbeta: bool = True,
    zbeta_lower_bound: float = min(
        [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]),
    zbeta_upper_bound: float = max([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224,
                                    (1 - 0.406) / 0.225])
) -> nn.Module
```

Prepare a CaBRNet ProtoClassifier model for composite LRP

**Arguments**:

- `model` - target model
- `set_bias_to_zero` - ignore bias in linear layers
- `stability_factor` _float_ - epsilon value used for numerical stability
- `use_zbeta` - use z-beta rule on first convolution
- `zbeta_lower_bound` _float_ - smallest admissible pixel value (used in z-beta rule)
- `zbeta_upper_bound` _float_ - largest admissible pixel value (used in z-beta rule)
  

**Returns**:

  copy of the model, ready for running Captum LRP

