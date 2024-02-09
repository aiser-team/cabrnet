import copy
from typing import Any, Callable

import torch
import torch.nn as nn
from cabrnet.generic.model import ProtoClassifier
from cabrnet.utils.optimizers import OptimizerManager
from cabrnet.visualisation.visualizer import SimilarityVisualizer
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm


class ProtoPNet(ProtoClassifier):
    """Class representing a ProtoPNet."""

    def __init__(self, extractor: nn.Module, classifier: nn.Module, **kwargs):
        """Build a ProtoPNet.

        Args:
            extractor: Feature extractor
            classifier: Classification based on extracted features
        """
        super(ProtoPNet, self).__init__(extractor, classifier, **kwargs)

        # Constant tensor for internal computations
        self.register_buffer("_eye", torch.eye(self.classifier.num_classes))

    def load_legacy_state_dict(self, legacy_state: dict) -> None:
        """Load state dictionary from legacy format.

        Args:
            legacy_state: Legacy state dictionary

        Raises:
            ValueError when keys or tensor sizes mismatch.
        """
        legacy_keys = legacy_state.keys()
        final_state = copy.deepcopy(legacy_state)
        cbrn_state = self.state_dict()
        cbrn_keys = list(self.state_dict().keys())
        cbrn_key = "dummy"

        for legacy_key in legacy_keys:
            if legacy_key.startswith("features"):
                # Feature extractor
                cbrn_key = legacy_key.replace("features", "extractor.convnet", 1)
                if cbrn_key not in cbrn_keys:
                    raise ValueError(f"No parameter matching {legacy_key}. Check that model architectures are similar.")
            elif legacy_key.startswith("add_on_layers"):
                # Add-on layers, find matching parameter based on size
                ref_size = legacy_state[legacy_key].size()
                found_match = False
                for cbrn_key in cbrn_keys:
                    if "add_on" in cbrn_key:
                        if cbrn_state[cbrn_key].size() == ref_size:
                            logger.info(f"Matching parameters {cbrn_key} to {legacy_key} based on identical size.")
                            found_match = True
                            break
                if not found_match:
                    raise ValueError(f"No parameter matching {legacy_key}. Check that model architectures are similar.")
            elif legacy_key == "last_layer.weight":
                cbrn_key = "classifier.last_layer.weight"
            elif legacy_key == "prototype_vectors":
                cbrn_key = "classifier.prototypes"
            elif legacy_key == "ones":
                cbrn_key = "classifier.similarity_layer._summation_kernel"
            else:
                final_state[legacy_key] = torch.unsqueeze(final_state[legacy_key], 0)

            # Update state
            if cbrn_state[cbrn_key].size() != final_state[legacy_key].size():
                raise ValueError(
                    f"Mismatching parameter size for {legacy_key} and {cbrn_key}. "
                    f"Expected {cbrn_state[cbrn_key].size()}, got {final_state[legacy_key].size()}"
                )
            final_state[cbrn_key] = final_state.pop(legacy_key)
            cbrn_keys.remove(cbrn_key)
        self.load_state_dict(final_state, strict=False)

    def loss(self, model_output: Any, label: torch.Tensor) -> tuple[torch.Tensor, float, dict[str, float]]:  # type: ignore
        """Loss function.

        Args:
            model_output: Model output, in this case a tuple containing the prediction and the minimum distances.
            label: Batch labels

        Returns:
            loss tensor and batch accuracy
        """
        output, min_distances = model_output

        cross_entropy = torch.nn.functional.cross_entropy(output, label)

        # TODO: store this in a config file and retrieve it from there
        coefs = {"crs_ent": 1, "clst": 0.8, "sep": -0.08, "l1": 1e-4}
        max_dist = 128

        if self._compatibility_mode:
            prototypes_of_correct_class = torch.t(self.classifier.proto_class_map[:, label])
            inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
            cluster_cost = torch.mean(max_dist - inverted_distances)

            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = torch.max(
                (max_dist - min_distances) * prototypes_of_wrong_class, dim=1
            )
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

            l1_mask = 1 - torch.t(self.classifier.proto_class_map)
            l1 = (self.classifier.last_layer.weight * l1_mask).norm(p=1)

        else:
            prototypes_of_correct_class = torch.t(torch.index_select(self.classifier.proto_class_map, 1, label))
            cluster_cost = torch.mean(torch.min(prototypes_of_correct_class * min_distances, dim=1)[0])

            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            separation_cost = -torch.mean(torch.min(prototypes_of_wrong_class * min_distances, dim=1)[0])

            l1_mask = 1 - torch.t(self.classifier.proto_class_map)
            l1 = (self.classifier.last_layer.weight * l1_mask).norm(p=1)

        loss = (
            coefs["crs_ent"] * cross_entropy
            + coefs["clst"] * cluster_cost
            + coefs["sep"] * separation_cost
            + coefs["l1"] * l1
        )

        batch_accuracy = torch.sum(torch.eq(torch.argmax(output, dim=1), label)).item() / len(label)
        stats = {
            "cross_entropy": cross_entropy.item(),
            "cluster_cost": cluster_cost.item(),
            "separation_cost": separation_cost.item(),
            "l1": l1.item(),
        }

        return loss, batch_accuracy, stats

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer_mngr: OptimizerManager,
        device: str = "cuda:0",
        progress_bar_position: int = 0,
        epoch_idx: int = 0,
        verbose: bool = False,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Train the model for one epoch.

        Args:
            train_loader: Dataloader containing training data
            optimizer_mngr: Optimizer manager
            device: Target device
            progress_bar_position: Position of the progress bar.
            epoch_idx: Epoch index
            max_batches: Max number of batches (early stop for small compatibility tests)
            verbose: Display progress bar

        Returns:
            dictionary containing learning statistics
        """
        self.train()
        self.to(device)

        # Training stats
        total_loss = 0.0
        total_acc = 0.0

        # Show progress on progress bar if needed
        train_iter = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            leave=False,
            position=progress_bar_position,
            disable=not verbose,
        )
        batch_num = len(train_loader)

        for batch_idx, (xs, ys) in train_iter:
            # Reset gradients and map the data on the target device
            optimizer_mngr.zero_grad()
            xs, ys = xs.to(device), ys.to(device)

            # Perform inference and compute loss
            ys_pred, distances = self.forward(xs)
            batch_loss, batch_accuracy, _ = self.loss((ys_pred, distances), ys)

            # Compute the gradient and update parameters
            batch_loss.backward()
            optimizer_mngr.optimizer_step(epoch=epoch_idx)

            # Update progress bar
            postfix_str = (
                f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                f"Batch loss: {batch_loss.item():.3f}, Acc: {batch_accuracy:.3f}"
            )
            train_iter.set_postfix_str(postfix_str)

            # Update global metrics
            total_loss += batch_loss.item()
            total_acc += batch_accuracy

            if max_batches is not None and batch_idx == max_batches:
                break

        # Clean gradients after last batch
        optimizer_mngr.zero_grad()

        if max_batches is not None:
            train_info = {
                "avg_loss": total_loss / (max_batches + 1),
                "avg_train_accuracy": total_acc / (max_batches + 1),
            }
        else:
            train_info = {"avg_loss": total_loss / batch_num, "avg_train_accuracy": total_acc / batch_num}
        return train_info

    # TODO: implementation
    def epilogue(self, **kwargs) -> None:
        pass

    # TODO: implementation
    def prune(self, pruning_threshold) -> None:
        pass

    # TODO: implementation
    def project(
        self, data_loader: DataLoader, device: str = "cuda:0", verbose: bool = False, progress_bar_position: int = 0
    ) -> dict[int, dict]:
        return {0: {}}

    # TODO: implementation
    def explain(
        self,
        img_path: str,
        preprocess: Callable,
        visualizer: SimilarityVisualizer,
        prototype_dir_path: str,
        output_dir_path: str,
        device: str,
        exist_ok: bool = False,
        **kwargs,
    ) -> None:
        pass

    # TODO: implementation
    def explain_global(self, prototype_dir_path: str, output_dir_path: str, **kwargs) -> None:
        pass
