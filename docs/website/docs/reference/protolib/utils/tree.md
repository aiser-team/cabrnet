---
sidebar_label: tree
title: protolib.utils.tree
---

#### log1mexp

```python
def log1mexp(x: Tensor) -> Tensor
```

Numerically accurate evaluation of log(1 - exp(-|x|))
See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf for details.

## TreeNode Objects

```python
class TreeNode(nn.Module)
```

#### \_\_init\_\_

```python
def __init__(node_id: str) -> None
```

Init TreeNode

**Arguments**:

- `node_id` - Node ID

#### num\_prototypes

```python
@property
def num_prototypes() -> int
```

**Returns**:

  Total number of prototypes pointed by this node and all its children

#### num\_nodes

```python
@property
def num_nodes() -> int
```

**Returns**:

  Total size of the subtree, including this node

#### prune\_children

```python
def prune_children(threshold: float = 0.01) -> None
```

Prune children based on threshold.

**Arguments**:

- `threshold` - Pruning threshold

#### size

```python
def size() -> int
```

Alias for self.num_nodes

#### leaves

```python
@property
def leaves() -> Iterator[nn.Module]
```

**Returns**:

  Iterator on all leaves

#### num\_leaves

```python
@property
def num_leaves() -> int
```

**Returns**:

  Total number of leaves

#### export\_arch

```python
def export_arch() -> Mapping[str, Any]
```

Export tree architecture (useful after pruning)

**Returns**:

  tree architecture

#### build\_from\_arch

```python
@staticmethod
def build_from_arch(arch: Mapping[str, Any]) -> TreeNode
```

Builds a decision tree from a configuration mapping

**Arguments**:

- `arch` - architecture mapping
  

**Returns**:

  decision tree

#### extra\_repr

```python
def extra_repr() -> str
```

Overwrite extra_repr from torch.nn.Module.

**Returns**:

  Node ID

## ComparativeNode Objects

```python
class ComparativeNode(TreeNode)
```

#### \_\_init\_\_

```python
def __init__(node_id: str, children: List[TreeNode], proto_idxs: List[int]) -> None
```

Create comparative node

**Arguments**:

- `node_id` - Node ID
- `children` - Node children
- `proto_idxs` - List of prototype indexes

#### forward

```python
def forward(similarities: Tensor, parent_probs: Tensor, conditional_probs: Tensor, greedy_path: Tensor) -> Tuple[Tensor, Dict]
```

Node forward pass using the probability of arriving at this node,
and the similarities to all prototypes.

**Arguments**:

- `similarities` - Tensor of similarities to all prototypes. Shape (N, P)
- `parent_probs` - Absolute (log) probability of reaching this node parent. Shape (N, )
- `conditional_probs` - (Log) probability of reach this node knowing that it reached its parent. Shape (N, )
- `greedy_path` - Keep track of greedy path. Shape (N, )
  

**Returns**:

  Node prediction (shape (N,C)), dictionary of self and children probabilities

#### num\_prototypes

```python
@property
def num_prototypes() -> int
```

**Returns**:

  Total number of prototypes pointed by this node and all its children

#### num\_nodes

```python
@property
def num_nodes() -> int
```

**Returns**:

  Total size of the subtree, including this node

## BinaryNode Objects

```python
class BinaryNode(TreeNode)
```

#### \_\_init\_\_

```python
def __init__(node_id: str, child_sim: TreeNode, child_nsim: TreeNode, proto_idx: int, log_probabilities: bool = False) -> None
```

Create binary node

**Arguments**:

- `node_id` - Node ID
- `child_sim` - Similarity child
- `child_nsim` - Non-similarity child
- `proto_idx` - Single prototype index
- `log_probabilities` - Use log of probabilities

#### forward

```python
def forward(similarities: Tensor, parent_probs: Tensor, conditional_probs: Tensor, greedy_path: Tensor) -> Tuple[Tensor, Dict]
```

Node forward pass using the probability of arriving at this node,
and the similarities to all prototypes.

**Arguments**:

- `similarities` - Tensor of similarities to all prototypes. Shape (N, P)
- `parent_probs` - Absolute (log) probability of reaching this node parent. Shape (N, )
- `conditional_probs` - (Log) probability of reach this node knowing that it reached its parent. Shape (N, )
- `greedy_path` - Keep track of greedy path. Shape (N, )
  

**Returns**:

  Node prediction (shape (N,C)), dictionary of self and children probabilities

#### num\_prototypes

```python
@property
def num_prototypes() -> int
```

**Returns**:

  Total number of prototypes pointed by this node and all its children

#### num\_nodes

```python
@property
def num_nodes() -> int
```

**Returns**:

  Total size of the subtree, including this node

#### create\_binary\_tree

```python
@staticmethod
def create_binary_tree(depth: int, num_classes: int, leaves_init_mode: str = "ZEROS", log_probabilities: bool = False, node_offset: int = 0, depth_offset: int = 0, proto_offset: int = 0) -> TreeNode
```

Create binary tree of a given depth (Prototree)

**Arguments**:

- `depth` - Target depth
- `num_classes` - Number of classes
- `leaves_init_mode` - Init mode for leaves distributions
- `log_probabilities` - Use log of probabilities
- `node_offset` - Index of root node
- `depth_offset` - Current depth
- `proto_offset` - Index of first prototype

**Returns**:

  Binary tree

## LeafNode Objects

```python
class LeafNode(TreeNode)
```

#### \_\_init\_\_

```python
def __init__(node_id: str, num_classes: int, init_mode: str = "ZEROS", log_probabilities: bool = False) -> None
```

Create leaf

**Arguments**:

- `node_id` - Node ID
- `num_classes` - Number of classes
- `log_probabilities` - Use log of probabilities

#### distribution

```python
@property
def distribution() -> Tensor
```

**Returns**:

  Normalised leaf distribution. Shape (1, C)

#### forward

```python
def forward(similarities: Tensor, parent_probs: Tensor, conditional_probs: Tensor, greedy_path: Tensor) -> Tuple[Tensor, Dict]
```

Node forward pass using the probability of arriving at this node,
and the similarities to all prototypes.

**Arguments**:

- `similarities` - Tensor of similarities to all prototypes. Shape (N, P)
- `parent_probs` - Absolute (log) probability of reaching this node parent. Shape (N, )
- `conditional_probs` - (Log) probability of reach this node knowing that it reached its parent. Shape (N, )
- `greedy_path` - Keep track of greedy path. Shape (N, )
  

**Returns**:

  Node prediction (shape (N,C)), dictionary of self and children probabilities

#### num\_prototypes

```python
@property
def num_prototypes() -> int
```

**Returns**:

  Total number of prototypes pointed by this node and all its children

#### num\_nodes

```python
@property
def num_nodes() -> int
```

**Returns**:

  Total size of the subtree, including this node

#### leaves

```python
@property
def leaves() -> Iterator[nn.Module]
```

**Returns**:

  Self

