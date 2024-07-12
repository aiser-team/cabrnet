from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch.nn
import torch.nn as nn
from loguru import logger
from torch import Tensor

leaf_init_modes = ["NORMAL", "ZEROS"]


class MappingMode(Enum):
    r"""Mapping selection between nodes, prototypes and class indices. Used in the get_mapping function of a TreeNode."""

    CLASS_TO_PROTOTYPE = 1
    PROTOTYPE_TO_CLASS = 2
    NODE_TO_PROTOTYPE = 3
    NODE_PATHS = 4
    ID_TO_NODE = 5


def log1mexp(x: Tensor) -> Tensor:
    r"""Returns a numerically accurate evaluation of log(1 - exp(-|x|)).

    See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf for details.

    Args:
        x (tensor): Input tensor.

    Returns:
        log(1 - exp(-|x|)).
    """
    return torch.where(x < np.log(2), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))


class TreeNode(nn.Module):
    r"""Generic class for managing decision trees based on similarity scores with prototypes.

    Attributes:
        node_id: Node identifier.
        proto_idxs: List of prototypes associated with this node.
        log_probabilities: If True, similarity scores are treated as log of probabilities.
    """

    proto_idxs: list | None = None
    log_probabilities: bool = False

    def __init__(self, node_id: str) -> None:
        r"""Initializes a TreeNode.

        Args:
            node_id (str): Node ID.
        """
        super().__init__()
        self.node_id = node_id

    def forward(
        self, similarities: Tensor, parent_probs: Tensor, conditional_probs: Tensor, greedy_path: Tensor
    ) -> Tuple[Tensor, Dict]:
        r"""Performs a forward pass using the probability of arriving at this node, and the similarities to all prototypes.

        Args:
            similarities (tensor): Tensor of similarities to all prototypes. Shape (N, P).
            parent_probs (tensor): Absolute (log) probability of reaching this node parent. Shape (N, ).
            conditional_probs (tensor): Probability (or log probability) of reaching this node,
                knowing that it reached its parent. Shape (N, ).
            greedy_path (tensor): Keep track of greedy path. Shape (N, ).*args, **kwargs) -> None:

        Returns:
            Node prediction (shape (N,C)), dictionary of self and children probabilities.
        """
        raise NotImplementedError

    @property
    def num_prototypes(self) -> int:
        r"""Returns the total number of prototypes pointed by this node and all its children."""
        raise NotImplementedError

    @property
    def num_nodes(self) -> int:
        r"""Returns the total size of the subtree, including this node."""
        raise NotImplementedError

    def prune_children(self, threshold: float = 0.01) -> None:
        r"""Prunes children based on a threshold.

        Args:
            threshold (float, optional): Pruning threshold. Default: 0.01.
        """
        prune_list = []
        for name, child in self.named_children():
            if max([torch.max(leaf.distribution).item() for leaf in child.leaves]) <= threshold:
                prune_list.append(name)
            else:
                # Call pruning function recursively
                child.prune_children(threshold=threshold)
        for child_name in prune_list:
            # Delete this child (and all submodules)
            logger.debug(f"Deleting child {child_name}")
            self.__delattr__(child_name)

        # Replace children with a single child with that child
        singleton_list = []
        for name, child in self.named_children():
            grand_children = [grand_child for grand_child in child.children()]
            if len(grand_children) == 1:
                singleton_list.append((name, grand_children[0]))
        for child_name, grand_child in singleton_list:
            self.__delattr__(child_name)
            # Rename grand child with child's name to maintain consistency
            self.add_module(child_name, grand_child)

    def prune_similar_children(self) -> None:
        r"""Prunes nodes that have the same decision on all children."""
        if self.proto_idxs is None:  # Leaf
            return

        names = ["nsim", "sim"]
        for child_name in names:
            child = self.get_submodule(f"{self.node_id}_child_{child_name}")
            child.prune_similar_children()

        # if a child of `this` is such that
        # its two children are leaves with the same decision,
        # replace child with its second child
        for child_name in names:
            childfullname = f"{self.node_id}_child_{child_name}"
            child = self.get_submodule(childfullname)

            if child.proto_idxs is None:  # child has no children
                continue

            has_great_grand_children = False
            decisions = set()
            grandchild = None
            for grandchild_name in names:
                grandchild = child.get_submodule(f"{child.node_id}_child_{grandchild_name}")
                if grandchild.proto_idxs is None:
                    decision = torch.argmax(grandchild.distribution)
                    decisions.add(decision.item())
                else:
                    has_great_grand_children = True
            if has_great_grand_children:
                continue
            if len(decisions) > 1:
                continue
            logger.debug(f"Merging children of {child.node_id}")
            self.__delattr__(childfullname)
            self.add_module(childfullname, grandchild)

    def size(self) -> int:
        r"""Returns self.num_nodes."""
        return self.num_nodes

    @property
    def leaves(self) -> Iterator[nn.Module]:
        r"""Returns iterator on all leaves."""
        for child in self.children():
            for leaf in child.leaves:
                yield leaf

    @property
    def active_prototypes(self) -> list[int]:
        r"""Returns list of active prototypes."""
        res: list[int] = self.proto_idxs.copy() if self.proto_idxs is not None else []
        res += [proto_idx for child in self.children() for proto_idx in child.active_prototypes]
        return res

    @property
    def num_leaves(self) -> int:
        r"""Returns the total number of leaves."""
        return sum([child.num_leaves for child in self.children()])

    def get_mapping(self, mode: MappingMode) -> dict | None:
        r"""Returns mapping between nodes, prototypes and classes.

        Args:
            mode (MappingMode): Mapping mode.

                - PROTOTYPE_TO_CLASS: Returns the list of class indexes associated with each prototype index.
                - CLASS_TO_PROTOTYPE: Returns the list of prototype indexes associated to each class index.
                - NODE_TO_PROTOTYPE: Returns the list of prototype indexes associated to each node ID.
                - NODE_PATHS: Returns the path (list of node IDs) for each node ID.
                - ID_TO_NODE: Returns the TreeNode for each node ID.
        """
        mapping = dict()
        mapped_classes = set([torch.argmax(leaf.distribution).item() for leaf in self.leaves])
        if mode == MappingMode.PROTOTYPE_TO_CLASS:
            if self.proto_idxs is not None:
                for proto_idx in self.proto_idxs:
                    mapping[proto_idx] = mapped_classes
            for child in self.children():
                mapping.update(child.get_mapping(mode))
        elif mode == MappingMode.CLASS_TO_PROTOTYPE:
            if self.proto_idxs is not None:
                for class_idx in mapped_classes:
                    mapping[class_idx] = self.proto_idxs.copy()
            for child in self.children():
                child_mapping = child.get_mapping(mode)
                for class_idx in child_mapping:
                    mapping[class_idx] += child_mapping[class_idx]
        elif mode == MappingMode.NODE_TO_PROTOTYPE:
            mapping[self.node_id] = self.proto_idxs.copy() if self.proto_idxs is not None else None
            for child in self.children():
                mapping.update(child.get_mapping(mode))
        elif mode == MappingMode.NODE_PATHS:
            mapping[self.node_id] = [self.node_id]
            for child in self.children():
                child_paths = child.get_mapping(mode)
                # Append node_id to all children paths
                for node, node_path in child_paths.items():
                    mapping[node] = [self.node_id] + node_path
        elif mode == MappingMode.ID_TO_NODE:
            mapping[self.node_id] = self
            for child in self.children():
                mapping.update(child.get_mapping(mode))
        else:
            raise NotImplementedError

        return mapping

    def export_arch(self) -> dict[str, Any]:
        r"""Returns the tree architecture (useful after pruning)."""
        arch = {
            "module": self.__class__.__name__,
            "node_id": self.node_id,
            "proto_idxs": self.proto_idxs,
            "log_probabilities": self.log_probabilities,
            "children": [],
        }
        if isinstance(self, BinaryNode):
            for name in ["nsim", "sim"]:  # Order for serialization matters for reconstructing the tree
                arch["children"].append(self.get_submodule(f"{self.node_id}_child_{name}").export_arch())
        else:
            for child in self.children():
                arch["children"].append(child.export_arch())
        return arch

    @staticmethod
    def build_from_arch(arch: dict[str, Any]) -> TreeNode:
        r"""Builds a decision tree from a configuration mapping.

        Args:
            arch (mapping): Architecture mapping.

        Returns:
            Decision tree.
        """
        if arch["module"] == "BinaryNode":
            child_nsim = TreeNode.build_from_arch(arch["children"][0])
            child_sim = TreeNode.build_from_arch(arch["children"][1])
            return BinaryNode(
                node_id=arch["node_id"],
                child_sim=child_sim,
                child_nsim=child_nsim,
                proto_idx=arch["proto_idxs"][0],  # arch["proto_idxs"] is a list of size 1
                log_probabilities=arch["log_probabilities"],
            )
        elif arch["module"] == "LeafNode":
            return LeafNode(
                node_id=arch["node_id"], num_classes=arch["num_classes"], log_probabilities=arch["log_probabilities"]
            )
        raise NotImplementedError(f"Unsupported tree module {arch['module']}")

    def extra_repr(self) -> str:
        r"""Overwrites extra_repr from torch.nn.Module to return the node ID."""
        return self.node_id


class ComparativeNode(TreeNode):
    r"""Comparative node associated with multiple prototypes.

    The probability of reaching a children branch is given by the similarity score with the associated prototype,
    relative to all other similarity scores.

    Attributes:
        node_id: Node identifier.
        proto_idxs: List of prototypes associated with this node.
        log_probabilities: If True, similarity scores are treated as log of probabilities.
    """

    def __init__(self, node_id: str, children: List[TreeNode], proto_idxs: List[int]) -> None:
        r"""Creates a comparative node.

        Args:
            node_id (str): Node ID.
            children (list): Node children.
            proto_idxs (list): List of prototype indexes.
        """
        super().__init__(node_id)
        assert len(children) == len(proto_idxs), "Number of children should match number of prototypes"
        for child in children:
            self.add_module(child.node_id, child)
        self.proto_idxs = proto_idxs

    def forward(
        self, similarities: Tensor, parent_probs: Tensor, conditional_probs: Tensor, greedy_path: Tensor
    ) -> Tuple[Tensor, Dict]:
        r"""Performs a forward pass using the probability of arriving at this node, and the similarities to all prototypes.

        Args:
            similarities (tensor): Tensor of similarities to all prototypes. Shape (N, P).
            parent_probs (tensor): Absolute (log) probability of reaching this node parent. Shape (N, ).
            conditional_probs (tensor): Probability (or log probability) of reaching this node,
                knowing that it reached its parent. Shape (N, ).
            greedy_path (tensor): Keep track of greedy path. Shape (N, ).

        Returns:
            Node prediction (shape (N,C)), dictionary of self and children probabilities.
        """
        # Focus only on relevant prototypes
        selected_similarities = similarities[:, self.proto_idxs]  # Shape (N, p)

        # Absolute probability of reaching this node
        absolute_probs = (
            parent_probs + conditional_probs if self.log_probabilities else parent_probs * conditional_probs
        )

        # Compute children decision probabilities
        children_probs = torch.softmax(selected_similarities, dim=1)  # Shape (N, p)

        # Compute max value per input across all children
        max_conditional_probs = torch.amax(children_probs, dim=1, keepdim=True)  # Shape (N, 1)

        # Compute all children contributions
        children_preds, children_probs_dicts = zip(
            *[
                child(
                    similarities=similarities,
                    parent_probs=absolute_probs,
                    conditional_probs=torch.unsqueeze(children_probs[:, child_idx], dim=1),
                    # For a child to belong to the greedy path, this node must already belong to the path and the
                    # child probability should be maximum among all other children
                    greedy_path=torch.logical_and(
                        greedy_path,
                        torch.eq(torch.unsqueeze(children_probs[:, child_idx], dim=1), max_conditional_probs),
                    ),
                )
                for child_idx, child in enumerate(self.children())
            ]
        )

        # Merge all dictionaries
        prob_dict = {
            self.node_id: {
                "absolute_probability": absolute_probs.detach().cpu(),
                "conditional_probability": conditional_probs.detach().cpu(),
                "greedy_path": greedy_path.detach().cpu(),
            }
        }
        for child_probs_dict in children_probs_dicts:
            prob_dict.update(child_probs_dict)

        # Aggregate predictions FIXME: should take into account the probability of each child
        pred = torch.sum(torch.stack(children_preds, dim=0), dim=0)  # Shape (N, C)
        return pred, prob_dict

    @property
    def num_prototypes(self) -> int:
        r"""Returns the total number of prototypes pointed by this node and all its children."""
        return sum([child.num_prototypes for child in self.children()])

    @property
    def num_nodes(self) -> int:
        r"""Returns the total size of the subtree, including this node."""
        return 1 + sum([child.size() for child in self.children()])


class BinaryNode(TreeNode):
    r"""Binary decision node, associated with a single prototype.

    This node manages two branches (similar and dissimilar) whose respective probability is computed from the similarity
    score with the associated prototype.

    Attributes:
        node_id: Node identifier.
        proto_idxs: List of prototypes associated with this node.
        log_probabilities: If True, similarity scores are treated as log of probabilities.
    """

    def __init__(
        self, node_id: str, child_sim: TreeNode, child_nsim: TreeNode, proto_idx: int, log_probabilities: bool = False
    ) -> None:
        r"""Creates a binary node.

        Args:
            node_id (str): Node ID.
            child_sim (TreeNode): Similarity child.
            child_nsim (TreeNode): Non-similarity child.
            proto_idx (int): Single prototype index.
            log_probabilities (bool, optional): If True, use log of probabilities. Default: False.
        """
        super().__init__(node_id)
        # Add children, keeping track of who is who
        self.add_module(f"{node_id}_child_nsim", child_nsim)
        self.add_module(f"{node_id}_child_sim", child_sim)
        self.proto_idxs = [proto_idx]
        self.log_probabilities = log_probabilities

    def forward(
        self, similarities: Tensor, parent_probs: Tensor, conditional_probs: Tensor, greedy_path: Tensor
    ) -> Tuple[Tensor, Dict]:
        r"""Performs a forward pass using the probability of arriving at this node, and the similarities to all prototypes.

        Args:
            similarities (tensor): Tensor of similarities to all prototypes. Shape (N, P).
            parent_probs (tensor): Absolute (log) probability of reaching this node parent. Shape (N, ).
            conditional_probs (tensor): Probability (or log probability) of reaching this node knowing that it
                reached its parent. Shape (N, ).
            greedy_path (tensor): Keep track of greedy path. Shape (N, ).

        Returns:
            Node prediction (shape (N,C)), dictionary of self and children probabilities.
        """
        # Focus only on relevant prototype
        if self.proto_idxs is None:
            raise ValueError(f"No prototype associated with node {self.node_id}.")

        similarity = similarities[:, [self.proto_idxs[0]]]  # Shape (N, 1)

        # Absolute probability of reaching this node
        absolute_probs = (
            parent_probs + conditional_probs if self.log_probabilities else parent_probs * conditional_probs
        )

        # Compute log(1-similarity) from log(similarity) if necessary
        non_similarity = log1mexp(torch.abs(similarity) + 1e-7) if self.log_probabilities else 1 - similarity

        # Compute max value per input across all children
        max_conditional_probs = torch.amax(
            torch.cat([non_similarity, similarity], dim=1), dim=1, keepdim=True
        )  # Shape (N, 1)

        # Compute all children contributions
        children_preds, children_probs_dicts = zip(
            *[
                self.get_submodule(child_name)(
                    similarities=similarities,
                    parent_probs=absolute_probs,
                    conditional_probs=child_prob,
                    # For a child to belong to the greedy path, this node must already belong to the path and the
                    # child probability should be maximum among all other children
                    greedy_path=torch.logical_and(greedy_path, torch.eq(child_prob, max_conditional_probs)),
                )
                for child_name, child_prob in zip(
                    [f"{self.node_id}_child_nsim", f"{self.node_id}_child_sim"],
                    [non_similarity, similarity],
                )
            ]
        )

        # Merge all dictionaries
        prob_dict = {
            self.node_id: {
                "absolute_probability": absolute_probs.detach().cpu(),
                "conditional_probability": conditional_probs.detach().cpu(),
                "greedy_path": greedy_path.detach().cpu(),
            }
        }
        for child_probs_dict in children_probs_dicts:
            prob_dict.update(child_probs_dict)

        # Aggregate weighted predictions
        if self.log_probabilities:
            # pred = log(exp[log(1-sim) + log(P0)] + exp[log(sim)+log(P1)]) = log( (1-sim) * P0 + sim * P1)
            pred = torch.logsumexp(
                torch.stack([non_similarity + children_preds[0], similarity + children_preds[1]]), dim=0
            )
        else:
            pred = non_similarity * children_preds[0] + similarity * children_preds[1]
        return pred, prob_dict

    @property
    def num_prototypes(self) -> int:
        r"""Returns the total number of prototypes pointed by this node and all its children."""
        return 1 + sum([child.num_prototypes for child in self.children()])

    @property
    def num_nodes(self) -> int:
        r"""Returns the total size of the subtree, including this node."""
        return 1 + sum([child.size() for child in self.children()])

    @staticmethod
    def create_binary_tree(
        depth: int,
        num_classes: int,
        leaves_init_mode: str = "ZEROS",
        log_probabilities: bool = False,
        node_offset: int = 0,
        depth_offset: int = 0,
        proto_offset: int = 0,
    ) -> TreeNode:
        r"""Creates a binary tree of a given depth (Prototree).

        Args:
            depth (int): Target depth.
            num_classes (int): Number of classes.
            leaves_init_mode (str, optional): Init mode for leaves distributions. Default: Distribution of zeros.
            log_probabilities (bool, optional): If True, use log of probabilities. Default: False.
            node_offset (int, optional): Index of root node. Default: 0.
            depth_offset (int, optional): Current depth. Default: 0.
            proto_offset (int, optional): Index of first prototype. Default: 0.

        Returns:
            Binary tree.
        """

        def _create_node(node_depth: int, node_idx: int, proto_idx: int) -> TreeNode:
            if node_depth == depth:
                return LeafNode(
                    node_id=f"leaf_{node_idx}",
                    num_classes=num_classes,
                    init_mode=leaves_init_mode,
                    log_probabilities=log_probabilities,
                )
            else:
                child_nsim = _create_node(node_depth + 1, node_idx + 1, proto_idx + 1)
                child_sim = _create_node(
                    node_depth + 1, node_idx + 1 + child_nsim.num_nodes, proto_idx + 1 + child_nsim.num_prototypes
                )
                return BinaryNode(
                    node_id=f"branch_{node_idx}",
                    child_sim=child_sim,
                    child_nsim=child_nsim,
                    proto_idx=proto_idx,
                    log_probabilities=log_probabilities,
                )

        return _create_node(depth_offset, node_offset, proto_offset)


class LeafNode(TreeNode):
    r"""Leaf node, not associated to any prototype but rather to a probability distribution among all classes.

    Attributes:
        node_id: Node identifier.
        num_classes: Number of classes.
        proto_idxs: List of prototypes associated with this node.
        log_probabilities: If True, similarity scores are treated as log of probabilities.
    """

    def __init__(
        self, node_id: str, num_classes: int, init_mode: str = "ZEROS", log_probabilities: bool = False
    ) -> None:
        r"""Creates a leaf.

        Args:
            node_id (str): Node ID.
            num_classes (int): Number of classes.
            init_mode (str, optional): Initialization mode for the class distribution. Default: "ZEROS".
            log_probabilities (bool, optional): If True, use log of probabilities. Default: False.
        """
        super().__init__(node_id)
        if init_mode not in leaf_init_modes:
            raise ValueError(f"Unknown leaf init mode {init_mode}")
        if init_mode == "ZEROS":
            self._relative_distribution = nn.Parameter(torch.zeros((1, num_classes)), requires_grad=True)
        elif init_mode == "NORMAL":
            self._relative_distribution = nn.Parameter(torch.randn((1, num_classes)), requires_grad=True)
        self.log_probabilities = log_probabilities
        self.num_classes = num_classes

    @property
    def distribution(self) -> Tensor:
        r"""Returns normalized leaf distribution. Shape (1, C)."""

        def stable_softmax(x: Tensor) -> Tensor:
            return torch.softmax(x - torch.max(x, dim=1)[0], dim=1)

        if self.log_probabilities:
            return torch.log_softmax(self._relative_distribution, dim=1)

        return stable_softmax(self._relative_distribution)

    def forward(
        self, similarities: Tensor, parent_probs: Tensor, conditional_probs: Tensor, greedy_path: Tensor
    ) -> Tuple[Tensor, Dict]:
        r"""Performs a forward pass using the probability of arriving at this node, and the similarities to all prototypes.

        Args:
            similarities (tensor): Tensor of similarities to all prototypes. Shape (N, P).
            parent_probs (tensor): Absolute (log) probability of reaching this node parent. Shape (N, ).
            conditional_probs (tensor): Probability (or log probability) of reaching this node knowing that
                it reached its parent. Shape (N, ).
            greedy_path (tensor): Keep track of greedy path. Shape (N, ).

        Returns:
            Node prediction (shape (N,C)), dictionary of self and children probabilities.
        """
        absolute_probs = (
            parent_probs + conditional_probs if self.log_probabilities else parent_probs * conditional_probs
        )
        return self.distribution, {
            self.node_id: {
                "absolute_probability": absolute_probs,  # Keep on device for derivative-free optimization
                "conditional_probability": conditional_probs.detach().cpu(),
                "greedy_path": greedy_path.detach().cpu(),
            }
        }

    @property
    def num_prototypes(self) -> int:
        r"""Returns the total number of prototypes pointed by this node and all its children."""
        return 0

    @property
    def num_nodes(self) -> int:
        r"""Returns the total size of the subtree, including this node."""
        return 1

    @property
    def leaves(self) -> Iterator[nn.Module]:
        r"""Returns self."""
        yield self

    @property
    def num_leaves(self) -> int:
        r"""Returns the total number of leaves (here: 1)."""
        return 1

    def export_arch(self) -> dict[str, Any]:
        r"""Returns the tree architecture (useful after pruning)."""
        arch = {
            "module": self.__class__.__name__,
            "node_id": self.node_id,
            "num_classes": self.num_classes,
            "log_probabilities": self.log_probabilities,
        }
        return arch
