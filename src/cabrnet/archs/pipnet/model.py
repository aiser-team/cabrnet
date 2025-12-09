import copy
from typing import Any, Callable

import graphviz
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from torch.utils.data import DataLoader

from cabrnet.archs.generic.decision import CaBRNetClassifier
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.optimizers import OptimizerManager
from cabrnet.core.visualization.visualizer import SimilarityVisualizer
from cabrnet.core.utils.image import safe_open_image


class PIPNet(CaBRNet):
    r"""CaBRNet model implementing the PIPNet architecture.

        Attributes:
            extractor: Model used to extract convolutional features from the input image.
            classifier: Model used to compute the classification, based on similarity scores with a set of prototypes.
    #        loss_coefficients: Parameters of the loss function used during training.
    #        projection_config: Parameters of the projection function used during training.
    """

    def __init__(self, extractor: nn.Module, classifier: CaBRNetClassifier, **kwargs):
        r"""Builds a PIPNet.

        Args:
            extractor (Module): Feature extractor.
            classifier (CaBRNetClassifier): Classification based on extracted features.
        """
        super(PIPNet, self).__init__(extractor, classifier, **kwargs)

        # Default training configuration
        self.loss_coefficients = {"alignment_epsilon": 1e-12, "presence_epsilon": 1e-8}

    def _load_legacy_state_dict(self, legacy_state: dict[str, Any]) -> None:
        r"""Loads a state dictionary in legacy format.

        Args:
            legacy_state (state dictionary): Legacy state dictionary.

        Raises:
            ValueError when keys or tensor sizes mismatch.
        """
        legacy_keys = legacy_state.keys()
        final_state = copy.deepcopy(legacy_state)
        cbrn_state = self.state_dict()
        cbrn_keys = list(self.state_dict().keys())
        cbrn_key = "dummy"

        legacy_to_cabrnet = {
            "module._multiplier": None,
            "module._classification.weight": "classifier.last_layer.weight",
            "module._classification.bias": "classifier.last_layer.bias",
            "module._classification.normalization_multiplier": "classifier.normalization_multiplier",
        }

        for legacy_key in legacy_keys:
            if legacy_key.startswith("module._net.features"):
                # Feature extractor
                cbrn_key = legacy_key.replace("module._net.features", "extractor.convnet.features", 1)
                if cbrn_key not in cbrn_keys:
                    raise ValueError(f"No parameter matching {legacy_key}. Check that model architectures are similar.")
            elif legacy_key.startswith("module._add_on"):
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
            elif legacy_key in legacy_to_cabrnet.keys():
                cbrn_key = legacy_to_cabrnet[legacy_key]
            else:
                raise ValueError(f"No parameter matching {legacy_key}. Check that model architectures are similar.")

            if cbrn_key is None:
                # Ignore parameter
                continue

            # Update state
            if cbrn_state[cbrn_key].size() != final_state[legacy_key].size():
                raise ValueError(
                    f"Mismatching parameter size for {legacy_key} and {cbrn_key}. "
                    f"Expected {cbrn_state[cbrn_key].size()}, got {final_state[legacy_key].size()}"
                )
            final_state[cbrn_key] = final_state.pop(legacy_key)
            cbrn_keys.remove(cbrn_key)
        super().load_state_dict(final_state, strict=False)

    def load_state_dict(self, state_dict: dict[str, Any], **kwargs) -> None:  # type:ignore
        r"""Overloads nn.Module load_state_dict to take legacy state dictionaries into account.

        Args:
            state_dict (state dictionary): State dictionary.
        """
        legacy_state = (
            any([key.startswith("module._net.features") for key in state_dict.keys()])
            or "model_state_dict" in state_dict.keys()
        )
        if legacy_state:
            logger.info("Legacy state dictionary detected, performing import.")
            if "model_state_dict" in state_dict.keys():
                state_dict = state_dict["model_state_dict"]
            self._load_legacy_state_dict(state_dict)
        else:
            # Load state dictionary
            super().load_state_dict(state_dict, **kwargs)

    def loss(
        self, model_output: Any, label: torch.Tensor, loss_weights: dict[str, float] | None = None, **kwargs
    ) -> tuple[torch.Tensor, dict[str, float]]:
        r"""Loss function.

        Args:
            model_output (Any): Model output, in this case a tuple containing the prediction and the minimum distances.
            label (tensor): Batch labels.
            loss_weights (dict or None, optional): Optional weights associated with different loss functions.
                Default: None.

        Returns:
            Loss tensor and batch statistics.
        """
        features, prototype_presence, prediction = model_output

        if features.size(0) == label.size(0):  # Evaluation mode
            alignment_loss = torch.zeros(1, device=features.device)
            prototype_presence_loss = -torch.log(
                torch.tanh(torch.sum(prototype_presence, dim=0)) + self.loss_coefficients["presence_epsilon"]
            ).mean()
        else:  # Training mode: input contains twice as many images as the number of labels
            # Recall that the two halves of the batch correspond to two variations on the same input
            features_1, features_2 = features.chunk(2)
            prototype_presence_1, prototype_presence_2 = prototype_presence.chunk(2)
            # Duplicate labels
            label = torch.cat([label, label])

            # Unroll features from (N, P, H, W) to (NxHxW, P) to observe prototype presence across the entire batch
            features_1 = features_1.flatten(start_dim=2).swapaxes(1, 2).flatten(end_dim=1)
            features_2 = features_2.flatten(start_dim=2).swapaxes(1, 2).flatten(end_dim=1)

            def align_loss(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
                r"""Computes alignment loss between two tensors.

                Args:
                    A (tensor): Input tensor.
                    B (tensor): Input tensor.

                Returns:
                    Alignment loss, computed as -log(<A,B> + epsilon)
                """
                res = torch.einsum("nc,nc->n", [A, B])
                res = -torch.log(res + self.loss_coefficients["alignment_epsilon"]).mean()
                return res

            if self._compatibility_mode:
                # May lead to different results if torch.einsum is not commutative
                alignment_loss = (
                    align_loss(features_1, features_2.detach()) + align_loss(features_2, features_1.detach())
                ) / 2.0
            else:
                alignment_loss = align_loss(features_1, features_2)

            # Check for prototype presence inside each batch
            prototype_presence_loss = (
                -(
                    torch.log(
                        torch.tanh(torch.sum(prototype_presence_1, dim=0)) + self.loss_coefficients["presence_epsilon"]
                    ).mean()
                    + torch.log(
                        torch.tanh(torch.sum(prototype_presence_2, dim=0)) + self.loss_coefficients["presence_epsilon"]
                    ).mean()
                )
                / 2.0
            )

        # Classification loss
        normalized_output = torch.log1p(prediction**self.classifier.normalization_multiplier)
        classification_loss = F.nll_loss(
            F.log_softmax(normalized_output, dim=1),
            label,
            reduction="mean",
        )
        batch_accuracy = torch.sum(torch.eq(torch.argmax(prediction, dim=1), label)).item() / len(label)

        if loss_weights is None:
            loss_weights = {}

        loss = (
            alignment_loss * loss_weights.get("alignment", 1.0)
            + prototype_presence_loss * loss_weights.get("prototype_presence", 1.0)
            + classification_loss * loss_weights.get("classification", 1.0)
        )
        stats = {
            "loss": loss.item(),
            "accuracy": batch_accuracy,
            "alignment_loss": alignment_loss.item(),
            "prototype_presence_loss": prototype_presence_loss.item(),
            "classification_loss": classification_loss.item(),
        }
        return loss, stats

    def _training_batch_hook(
        self,
        batch_idx: int,
        batch_num: int,
        dataloaders: dict[str, DataLoader],
        optimizer_mngr: OptimizerManager | torch.optim.Optimizer,
        dataset_name: str = "train_set",
        device: str | torch.device = "cuda:0",
        tqdm_position: int = 0,
        epoch_idx: int = 0,
        verbose: bool = False,
        tqdm_title: str | None = None,
        **kwargs,
    ) -> None:
        r"""Internal function called after each batch in the training loop.

        Args:
            batch_idx (int): Current batch index.
            batch_num (int): Total number of batches.
            dataloaders (dictionary): Dictionary of dataloaders.
            optimizer_mngr (OptimizerManager): Optimizer manager.
            dataset_name (str, optional): Name of the dataset used for training. Default: train_set.
            device (str | device, optional): Hardware device. Default: cuda:0.
            tqdm_position (int, optional): Position of the progress bar. Default: 0.
            epoch_idx (int, optional): Epoch index. Default: 0.
            verbose (bool, optional): Display progress bar. Default: False.
            tqdm_title (str, optional): Progress bar title. Default: "Training epoch {epoch_idx".
        """
        assert isinstance(optimizer_mngr, OptimizerManager), "Unsupported optimizer type {type{optimizer_mngr}}"
        assert optimizer_mngr.schedulers.get("optimizer_classifier"), "Missing LR scheduler for optimizer"

        # In PIPNet, the LR scheduler is called after each batch of data
        if "pretrain" not in optimizer_mngr.get_active_periods(epoch_idx):
            # After pretraining, the epoch index is restarted
            offset = optimizer_mngr.periods["pretrain"]["epoch_range"][1]
            # This weird step value is inherited from the original PIPNet code.
            # (in theory, a scheduler only accepts integer step values).
            optimizer_mngr.schedulers.get("optimizer_classifier").step(
                epoch_idx - 1 - offset + (batch_idx / batch_num)  # type:ignore
            )
            self.classifier.clamp_parameters()

        if "fine_tuning" not in optimizer_mngr.get_active_periods(epoch_idx):
            optimizer_mngr.schedulers.get("optimizer_net").step()

    def train_epoch(
        self,
        dataloaders: dict[str, DataLoader],
        optimizer_mngr: OptimizerManager,
        device: str | torch.device = "cuda:0",
        tqdm_position: int = 0,
        epoch_idx: int = 0,
        verbose: bool = False,
    ) -> dict[str, float]:
        r"""Trains a PIPNet model for one epoch.

        Args:
            dataloaders (dictionary): Dictionary of dataloaders.
            optimizer_mngr (OptimizerManager): Optimizer manager.
            device (str | device, optional): Hardware device. Default: cuda:0.
            tqdm_position (int, optional): Position of the progress bar. Default: 0.
            epoch_idx (int, optional): Epoch index. Default: 0.
            verbose (bool, optional): Display progress bar. Default: False.

        Returns:
            Dictionary containing learning statistics.
        """
        if "pretrain" in optimizer_mngr.get_active_periods(epoch_idx):
            epoch_range = optimizer_mngr.periods["pretrain"]["epoch_range"]
            num_epochs = epoch_range[1] - epoch_range[0] + 1

            # The importance of each loss function varies during training.
            loss_weights = {  # The importance of each loss function varies during training.
                "alignment": (epoch_idx + 1) / num_epochs,
                "prototype_presence": 5.0,
                "classification": 0.0,
            }
            dataset_name = "pretrain_set"

        elif "fine_tuning" in optimizer_mngr.get_active_periods(epoch_idx):
            epoch_range = optimizer_mngr.periods["fine_tuning"]["epoch_range"]
            if epoch_idx == epoch_range[0]:
                # Reset schedulers after pretraining
                optimizer_mngr._set_optimizers()

            loss_weights = {
                "alignment": 0.0,
                "prototype_presence": 0.0,
                "classification": 2.0,
            }
            dataset_name = "train_set"
        else:
            loss_weights = {
                "alignment": 5.0,
                "prototype_presence": 2.0,
                "classification": 2.0,
            }
            dataset_name = "train_set"

        train_info = self._train_epoch(
            dataloaders=dataloaders,
            optimizer_mngr=optimizer_mngr,
            dataset_name=dataset_name,
            device=device,
            tqdm_position=tqdm_position,
            tqdm_title=f"Training epoch {epoch_idx}",
            epoch_idx=epoch_idx,
            verbose=verbose,
            loss_weights=loss_weights,
        )

        # Fix metric names so that they are consistent throughout the training
        if dataset_name == "pretrain_set":
            train_info = {key.replace("pretrain_set", "train_set"): value for key, value in train_info.items()}

        return train_info

    def epilogue(
        self,
        dataloaders: dict[str, DataLoader],
        optimizer_mngr: OptimizerManager,
        output_dir: str,
        num_samples_per_prototype: int = 5,
        device: str | torch.device = "cuda:0",
        verbose: bool = False,
        **kwargs,
    ) -> list[dict]:
        r"""Function called after training, using information from the epilogue field in the training configuration.

        Args:
            dataloaders (dictionary): Dictionary of dataloaders.
            optimizer_mngr (OptimizerManager): Optimizer manager.
            output_dir (str): Path to output directory.
            num_samples_per_prototype (int, optional): Number of samples used to illustrate each prototype. Default: 5.
            device (str | device, optional): Hardware device. Default: cuda:0.
            verbose (bool, optional): Display progress bar. Default: False.

        Returns:
            Projection information.
        """
        projection_info = self.project(
            dataloader=dataloaders["projection_set"],
            update_prototypes=False,
            num_samples_per_prototype=num_samples_per_prototype,
            device=device,
            verbose=verbose,
        )
        return projection_info

    def project(
        self,
        dataloader: DataLoader,
        update_prototypes: bool = False,
        num_samples_per_prototype: int = 5,
        device: str | torch.device = "cuda:0",
        verbose: bool = False,
        tqdm_position: int = 0,
    ) -> list[dict]:
        r"""Performs prototype projection (maximum similarity) after training.
        WARNING: By default, PIPNet does not modify the network but rather finds the most activated images inside
        the projection set.

        Args:
            dataloader (DataLoader): Dataloader containing projection data.
            update_prototypes (bool, optional): If True, update prototypes with their closest vectors. Default: False.
            num_samples_per_prototype (int, optional): Number of samples used to illustrate each prototype. Default: 5.
            device (str | device, optional): Hardware device. Default: cuda:0.
            verbose (bool, optional): Display progress bar. Default: False.
            tqdm_position (int, optional): Position of the progress bar. Default: 0.

        Returns:
            Dictionary containing projection information for each prototype.
        """
        self.eval()
        self.to(device)

        assert not update_prototypes, "PIPNet does not support prototype update during projection"

        # Mapping between classes and prototypes
        proto_class_map = self.prototype_class_mapping
        class_mapping = {c: list(np.nonzero(proto_class_map[:, c])[0]) for c in range(self.classifier.num_classes)}

        # Sanity check to ensure that each class is associated with at least one prototype
        for class_idx in range(self.classifier.num_classes):
            if class_idx not in class_mapping:
                logger.error(f"Inaccessible class {class_idx}!")

        # Show progress on progress bar if needed
        data_iter = tqdm(
            enumerate(dataloader),
            desc="Prototype projection",
            total=len(dataloader),
            leave=False,
            position=tqdm_position,
            disable=not verbose,
        )

        # For each prototype, keep track of:
        #   - the index of the most activated projection images
        #   - the coordinates of the vector inside the latent representation of each image
        #   - the corresponding similarity score
        projection_info = {proto_idx: [] for proto_idx in range(self.num_prototypes)}

        with torch.no_grad():
            for batch_idx, (xs, ys) in data_iter:
                # Map to device and perform inference
                xs = xs.to(device)  # type: ignore
                similarities = self.similarities(xs)  # Shape N x P x H x W
                W = similarities.size(-1)

                # Find location of highest activation scores
                best_score, best_score_loc = torch.max(similarities.flatten(start_dim=2), dim=2)

                for img_idx, label in enumerate(ys):
                    # Some architectures simply ignore the class index
                    class_idx = label.item() if self.classifier.num_classes > 1 else 0

                    for proto_idx in class_mapping[class_idx]:
                        # For each entry, only check prototypes that lead to the corresponding class
                        current_projection_info = projection_info[proto_idx]
                        update_scores = False
                        if len(current_projection_info) == num_samples_per_prototype:
                            current_min_score = min([info["score"] for info in current_projection_info])
                            if current_min_score < best_score[img_idx, proto_idx]:
                                # Remove old score
                                projection_info[proto_idx] = [
                                    info for info in current_projection_info if info["score"] > current_min_score
                                ]
                                update_scores = True
                        else:
                            update_scores = True
                        if update_scores:
                            batch_size = 1 if dataloader.batch_size is None else dataloader.batch_size
                            projection_info[proto_idx].append(
                                {
                                    "img_idx": batch_idx * batch_size + img_idx,
                                    "h": best_score_loc[img_idx, proto_idx].item() // W,
                                    "w": best_score_loc[img_idx, proto_idx].item() % W,
                                    "score": best_score[img_idx, proto_idx].item(),
                                }
                            )

        return sum(
            [
                [{"proto_idx": proto_idx} | entry for entry in projection_info[proto_idx]]
                for proto_idx in projection_info
            ],
            [],
        )

    @staticmethod
    def _build_prototype_representation(
        proto_idx: int,
        prototype_dir: str,
    ) -> graphviz.Digraph:
        r"""Builds the representation of a prototype as graph containing the most activated images from the projection
        set.

        Args:
            proto_idx (int): Prototype index.
            prototype_dir (str): Path to prototype images.

        Returns:
            A graph containing the prototype index surrounded by its representation.
        """
        graph = graphviz.Digraph()
        graph.attr(layout="circo")
        # Default node configuration
        graph.attr("node", label="", fixedsize="True", width="2", height="2", fontsize="25")

        # Build representation of this prototype
        graph.node(
            name=f"P{proto_idx}",
            shape="circle",
            label=f"P{proto_idx}",
            root="True",
        )
        img_path = os.path.abspath(os.path.join(prototype_dir, f"prototype_{proto_idx}.png"))
        index = 0
        while os.path.exists(img_path):
            graph.node(
                name=f"P{proto_idx}_view{index}",
                shape="plaintext",
                image=img_path,
                imagescale="true",
            )
            graph.edge(f"P{proto_idx}", f"P{proto_idx}_view{index}")
            index += 1
            img_path = os.path.abspath(os.path.join(prototype_dir, f"prototype_{proto_idx}_{index}.png"))

        return graph

    def _build_prototype_association(
        self,
        graph: graphviz.Graph,
        target_class: int,
        proto_idx: int,
        prototype_dir: str,
        node_prefix: str = "",
        create_target_class_node: bool = False,
    ):
        r"""Internal function used to build the representation of a prototype, connected to its related classes.

        Args:
            graph (graph): Source graph.
            target_class (int): Target class. The connection to the prototype will be a straight line. All other related
                classes will be connected via dashed lines.
            proto_idx (int): Prototype index.
            prototype_dir (str): Path to prototype images.
            node_prefix (str, optional): Node prefix in the graph. Default: "".
            create_target_class_node (bool, optional): If true, creates a node for the target class. Default: False.
        """
        cluster_path = os.path.abspath(os.path.join(prototype_dir, f"prototype_{proto_idx}_cluster.png"))
        graph.node(
            name=f"{node_prefix}P{proto_idx}",
            shape="plaintext",
            image=cluster_path,
            imagescale="true",
            width="6",  # Increase node size
            height="6",
        )
        # Assume that the target node class has already been defined in the graph.
        if not create_target_class_node:
            graph.edge(f"C{target_class}", f"{node_prefix}P{proto_idx}")

        # Show which classes are connected to this prototype
        proto_class_map = self.prototype_class_mapping
        related_classes = list(np.nonzero(proto_class_map[proto_idx, :])[0])
        for related_class in related_classes:
            if related_class != target_class:
                graph.node(
                    name=f"{node_prefix}P{proto_idx}_C{related_class}",
                    shape="circle",
                    label=f"Class {related_class}",
                    root="True",
                    style="dashed",
                )
                graph.edge(
                    f"{node_prefix}P{proto_idx}",
                    f"{node_prefix}P{proto_idx}_C{related_class}",
                    style="dashed",
                )
            elif create_target_class_node:
                graph.node(
                    name=f"{node_prefix}P{proto_idx}_C{related_class}",
                    shape="circle",
                    label=f"Class {related_class}",
                    root="True",
                    style="filled",
                    fillcolor="lightblue",
                )
                graph.edge(
                    f"{node_prefix}P{proto_idx}",
                    f"{node_prefix}P{proto_idx}_C{related_class}",
                    style="dashed",
                )

    def explain(
        self,
        img: str | Image.Image,
        preprocess: Callable | None,
        visualizer: SimilarityVisualizer,
        prototype_dir: str,
        output_dir: str,
        output_format: str = "pdf",
        device: str | torch.device = "cuda:0",
        exist_ok: bool = False,
        disable_rendering: bool = False,
        num_closest: int = 10,
        class_specific: bool = False,
        **kwargs,
    ) -> list[tuple[int, float, bool]]:
        r"""Explains the decision for a particular image.

        Args:
            img (str or Image): Path to image or image itself.
            preprocess (Callable): Preprocessing function.
            visualizer (SimilarityVisualizer): Similarity visualizer.
            prototype_dir (str): Path to directory containing prototype visualizations.
            output_dir (str): Path to output directory.
            output_format (str, optional): Output file format. Default: pdf.
            device (str | device, optional): Hardware device. Default: cuda:0.
            exist_ok (bool, optional): Silently overwrites existing explanation (if any). Default: False.
            disable_rendering (bool, optional): When True, no visual explanation is generated. Default: False.
            num_closest (int, optional): Number of closest prototypes to display. Default: 10.
            class_specific (bool, optional): If True, only use prototypes from the predicted class. Default: False.

        Returns:
            List of most relevant prototypes for the decision, where each entry is in the form
                (<prototype index>, <similarity score>, <similar>)
            and <similar> indicates whether the prototype is considered similar or dissimilar.
        """
        self.eval()

        with safe_open_image(img, preprocess) as (img, img_tensor):
            # Map to device
            self.to(device)
            img_tensor = img_tensor.to(device)

            # Perform inference
            features, prototype_presence, prediction = self.forward(img_tensor)
            class_idx = torch.argmax(prediction, dim=1)[0].item()

            prototype_presence = prototype_presence[0]
            proto_class_map = self.prototype_class_mapping

            # Remove pruned/non-class specific prototypes
            for proto_idx in range(self.num_prototypes):
                if not self.classifier.prototype_is_active(proto_idx) or (
                    class_specific and proto_class_map[proto_idx, class_idx] == 0
                ):
                    prototype_presence[proto_idx] = 0.0

            # Build explanation
            explanation_graph = graphviz.Graph()
            explanation_graph.attr(layout="circo")
            # Default node configuration
            explanation_graph.attr("node", label="", fixedsize="True", width="2", height="2", fontsize="25")

            img_path = os.path.abspath(os.path.join(output_dir, "original.png"))
            if not disable_rendering:
                os.makedirs(os.path.join(output_dir, "test_patches"), exist_ok=exist_ok)
                # Copy source image
                img.save(img_path)

            most_relevant_prototypes = []  # Keep track of most relevant prototypes

            for _ in range(num_closest):
                proto_idx = int(torch.argmax(prototype_presence).item())
                presence = torch.max(prototype_presence)
                if presence == 0:
                    # Not enough relevant prototypes
                    break
                most_relevant_prototypes.append(
                    (proto_idx, presence, True)
                )  # PIPNet only considers positive similarities

                # Build representation of this prototype, with its class associations
                self._build_prototype_association(
                    graph=explanation_graph,
                    target_class=class_idx,
                    proto_idx=proto_idx,
                    prototype_dir=prototype_dir,
                    node_prefix="",
                    create_target_class_node=True,
                )

                # Generate test image patch
                patch_image_path = os.path.join(output_dir, "test_patches", f"proto_similarity_{proto_idx}.png")
                if not disable_rendering:
                    patch_image = visualizer.forward(img=img, img_tensor=img_tensor, proto_idx=proto_idx, device=device)
                    patch_image.save(patch_image_path)

                explanation_graph.node(
                    name=f"patch_{proto_idx}",
                    shape="plaintext",
                    image=os.path.abspath(patch_image_path),
                    imagescale="true",
                )
                explanation_graph.edge(f"P{proto_idx}", f"patch_{proto_idx}")

                # "Disable" prototype from search
                prototype_presence[proto_idx] = 0

            if not disable_rendering:
                explanation_graph.render(filename=os.path.join(output_dir, f"local_explanation"), format=output_format)
            return most_relevant_prototypes

    def explain_global(
        self,
        prototype_dir: str,
        output_dir: str,
        output_format: str = "pdf",
        **kwargs,
    ) -> None:
        r"""Explains the global decision-making process of a CaBRNet model.

        Args:
            prototype_dir (str): Path to directory containing prototype visualizations.
            output_dir (str): Path to output directory.
            output_format (str, optional): Output file format. Default: pdf.
        """
        proto_class_map = self.prototype_class_mapping
        class_mapping = {c: list(np.nonzero(proto_class_map[:, c])[0]) for c in range(self.classifier.num_classes)}

        # Render one cluster per prototype
        for proto_idx in tqdm(range(self.num_prototypes), desc="Rendering prototype clusters", leave=False):
            if self.prototype_is_active(proto_idx):
                prototype_graph = self._build_prototype_representation(proto_idx=proto_idx, prototype_dir=prototype_dir)
                # Intermediate rendering to preserve layout
                cluster_path = os.path.join(prototype_dir, f"prototype_{proto_idx}_cluster")
                prototype_graph.render(filename=cluster_path, format="png")

        # Disjointed graphs for readability
        def _build_class_node(graph, class_idx):
            graph.node(
                name=f"C{class_idx}",
                shape="circle",
                label=f"Class {class_idx}",
                root="True",
            )
            for proto_idx in class_mapping[class_idx]:
                self._build_prototype_association(
                    graph=graph,
                    target_class=class_idx,
                    proto_idx=proto_idx,
                    prototype_dir=prototype_dir,
                    node_prefix=f"C{class_idx}_",
                )

        explanation_graph = graphviz.Graph()
        explanation_graph.attr(layout="circo")
        # Default node configuration
        explanation_graph.attr("node", label="", fixedsize="True", width="2", height="2", fontsize="25")
        for class_idx in range(self.classifier.num_classes):
            _build_class_node(explanation_graph, class_idx)
        logger.debug(explanation_graph.source)
        explanation_graph.render(filename=os.path.join(output_dir, f"global_explanation"), format=output_format)
