"""Helpers for loading state dictionaries into submodules."""

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch.nn as nn
import torch


def state_dict_for(state_dict: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    r"""Drops a component key from a state dictionary when present.

    The component may be stored either as a nested mapping (``{"extractor":
    {"convnet": state_dict}}``) or as a prefix in flattened state-dict keys
    (``{"extractor.convnet.weight": tensor}``). If neither representation is
    present, the original state dictionary is returned.

    Args:
        state_dict (mapping): Loaded state dictionary or checkpoint mapping.
        key (str): Component key to drop.

    Returns:
        State dictionary for the requested component.

    Raises:
        ValueError: The nested component exists but does not contain a state dictionary.
    """
    if key in state_dict:
        component_state_dict = state_dict[key]
        if not isinstance(component_state_dict, Mapping):
            raise ValueError(f"Checkpoint entry '{key}' does not contain a state dictionary.")
        return component_state_dict

    component_state_dict = state_dict
    for part in key.split("."):
        if part not in component_state_dict:
            break
        component_state_dict = component_state_dict[part]
        if not isinstance(component_state_dict, Mapping):
            break
    else:
        return component_state_dict

    prefix = f"{key}."
    stripped_state_dict = {
        name.removeprefix(prefix): value for name, value in state_dict.items() if name.startswith(prefix)
    }
    return stripped_state_dict or state_dict


def available_module_paths(module: nn.Module, max_depth: int = 3) -> list[str]:
    r"""Lists module paths up to a maximum nesting depth.

    Args:
        module (nn.module): Module to get the names from.
        max_depth (int, optional): Max depth of the extracted module names. Default: 3.

    Returns:
        list of all identifiers of the children modules.
    """
    return [name for name, _ in module.named_modules() if name and name.count(".") < max_depth]


def state_dict_key_with_matching_shape(
    state_dict: Mapping[str, Any], candidate_keys: list[str], key_fragment: str, reference: torch.Tensor
) -> str | None:
    r"""Finds the first candidate state-dictionary key with a matching tensor shape.

    Args:
        state_dict (mapping): Target model state dictionary.
        candidate_keys (list[str]): Still-unmatched keys in the target state dictionary.
        key_fragment (str): Required substring in a candidate key.
        reference (Tensor): Tensor whose shape must match.

    Returns:
        The first matching key, or None when no candidate matches.
    """
    for candidate_key in candidate_keys:
        if key_fragment in candidate_key and state_dict[candidate_key].size() == reference.size():
            return candidate_key
    return None


def remap_state_dict_entry(
    state_dict: MutableMapping[str, Any],
    target_state_dict: Mapping[str, Any],
    remaining_target_keys: list[str],
    source_key: str,
    target_key: str,
) -> None:
    r"""Validates and moves a legacy state-dictionary entry to its target key.

    Args:
        state_dict (mutable mapping): State dictionary being converted.
        target_state_dict (mapping): State dictionary of the target model.
        remaining_target_keys (list[str]): Target keys not yet matched.
        source_key (str): Legacy state-dictionary key.
        target_key (str): Destination key in the target state dictionary.

    Raises:
        ValueError: If the source and target tensor shapes differ.
    """
    if target_state_dict[target_key].size() != state_dict[source_key].size():
        raise ValueError(
            f"Mismatching parameter size for {source_key} and {target_key}. "
            f"Expected {target_state_dict[target_key].size()}, got {state_dict[source_key].size()}"
        )
    state_dict[target_key] = state_dict.pop(source_key)
    remaining_target_keys.remove(target_key)
