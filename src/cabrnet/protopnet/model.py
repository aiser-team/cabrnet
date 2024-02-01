import copy
from typing import Any, Callable

import torch
import torch.nn as nn
from cabrnet.generic.model import ProtoClassifier
from cabrnet.visualisation.visualizer import SimilarityVisualizer
from loguru import logger
from torch.optim import Optimizer
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

    def loss(self, model_output: Any, label: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Loss function.

        Args:
            model_output: Model output, in this case a tuple containing the prediction and the leaf probabilities
            label: Batch labels

        Returns:
            loss tensor and batch accuracy
        """
        ys_pred = model_output
        batch_loss = torch.nn.functional.nll_loss(torch.log(ys_pred), label)
        batch_accuracy = torch.sum(torch.eq(torch.argmax(ys_pred, dim=1), label)).item() / len(label)
        return batch_loss, batch_accuracy

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        device: str = "cuda:0",
        progress_bar_position: int = 0,
        epoch_idx: int = 0,
        verbose: bool = False,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Train the model for one epoch.

        Args:
            train_loader: Dataloader containing training data
            optimizer: Learning optimizer
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
            optimizer.zero_grad()
            xs, ys = xs.to(device), ys.to(device)

            # Perform inference and compute loss
            # ys_pred, info = self.forward(xs)
            # batch_loss, batch_accuracy = self.loss((ys_pred, info), ys)
            ys_pred = self.forward(xs)
            batch_loss, batch_accuracy = self.loss((ys_pred), ys)

            # Compute the gradient and update parameters
            batch_loss.backward()
            optimizer.step()

            # Update progress bar
            postfix_str = (
                f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                f"Batch loss: {batch_loss.item():.3f}, Acc: {batch_accuracy:.3f}"
            )
            train_iter.set_postfix_str(postfix_str)  # type: ignore

            # Update global metrics
            total_loss += batch_loss.item()
            total_acc += batch_accuracy

            if max_batches is not None and batch_idx == max_batches:
                break

        # Clean gradients after last batch
        optimizer.zero_grad()

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
