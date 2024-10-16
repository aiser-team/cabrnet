import copy
import os
from typing import Any, Callable

import graphviz
import numpy as np
import torch
import torch.nn as nn

from cabrnet.archs.generic.decision import CaBRNetClassifier
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import batch_mixup
from cabrnet.core.utils.optimizers import OptimizerManager
from cabrnet.core.visualization.explainer import ExplanationGraph
from cabrnet.core.visualization.visualizer import SimilarityVisualizer
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import time


class ProtoPool(CaBRNet):
    r"""CaBRNet model implementing the ProtoPool architecture.

    Attributes:
        extractor: Model used to extract convolutional features from the input image.
        classifier: Model used to compute the classification, based on similarity scores with a set of prototypes.
        training_config: Parameters controlling the training process.
        loss_coefficients: Parameters of the loss function used during training.
    """

    def __init__(self, extractor: nn.Module, classifier: CaBRNetClassifier, **kwargs):
        r"""Builds a ProtoPool.

        Args:
            extractor (Module): Feature extractor.
            classifier (CaBRNetClassifier): Classification based on extracted features.
        """
        super(ProtoPool, self).__init__(extractor, classifier, **kwargs)

        # Additional training configuration
        self.loss_coefficients = {
            "clustering": 0.8,
            "separability": -0.08,
            "regularization": 0.0001,
        }
        self.training_config = {
            "gumbel_min_scale": 1.3,
            "gumbel_max_scale": 10**3,
            "gumbel_epochs": 10,
            "use_mix_up": True,
        }

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
            "last_layer.weight": "classifier.last_layer.weight",
            "prototype_vectors": "classifier.prototypes",
            "proto_presence": "classifier.proto_slot_map",
            "ones": "classifier.similarity_layer._summation_kernel",
            "alfa": None,
        }

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
            elif legacy_key in legacy_to_cabrnet.keys():
                cbrn_key = legacy_to_cabrnet[legacy_key]
            else:
                raise ValueError(f"Unmatched key in legacy state: {legacy_key}")

            if not cbrn_key:
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
        legacy_state = any([key.startswith("features") for key in state_dict.keys()])
        if legacy_state:
            logger.info("Legacy state dictionary detected, performing import.")
            self._load_legacy_state_dict(state_dict)
        else:
            # Load state dictionary
            super().load_state_dict(state_dict, **kwargs)

    def loss(
        self,
        model_output: Any,
        label: torch.Tensor,
        mixed_label: torch.Tensor | None = None,
        mix_percentage: float = 1.0,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        r"""Loss function.

        Args:
            model_output (Any): Model output, in this case a tuple containing the prediction and the minimum distances.
            label (tensor): Original batch labels.
            mixed_label (tensor, optional): Mixed batch labels. Default: None.
            mix_percentage (float, optional): Mix percentage. A value 1.0 indicates that no mix was performed.
                Default: 1.0

        Returns:
            Loss tensor and batch statistics.
        """
        output, min_distances, proto_slot_probs = model_output

        # Cross-entropy loss
        if mixed_label is None:
            mixed_label = label
        cross_entropy = torch.nn.functional.cross_entropy(output, label) * mix_percentage + (
            1 - mix_percentage
        ) * torch.nn.functional.cross_entropy(output, mixed_label)

        # L1 regularization of the parameters of the last layer
        l1_mask = 1 - torch.t(self.classifier.slot_class_map)
        l1 = (self.classifier.last_layer.weight * l1_mask).norm(p=1)

        if self._compatibility_mode:
            # Orthogonal loss, computed using the cosine similarity on the non-normalized proto_slot_map vectors (?)
            orthogonal_loss = torch.nn.functional.cosine_similarity(
                self.classifier.proto_slot_map.unsqueeze(2), self.classifier.proto_slot_map.unsqueeze(-1), dim=1
            ).sum()
            orthogonal_loss = orthogonal_loss / (self.classifier.num_slots_per_class * self.classifier.num_classes) - 1
        else:
            # Orthogonal loss, computed using the cosine similarity on the distribution vectors (after Gumbel-Softmax)
            # Take absolute value of the similarity (to avoid optimization towards -1 !!!)
            mat = torch.nn.functional.cosine_similarity(
                proto_slot_probs.unsqueeze(2), proto_slot_probs.unsqueeze(-1), dim=1
            ).abs()  # Shape (K x S x S)
            # Remove diagonals (all ones) and compute the sum of all pairwise cosine similarity scores
            mat = mat * (1.0 - torch.eye(self.classifier.num_slots_per_class, dtype=mat.dtype, device=mat.device))
            orthogonal_loss = mat.sum()
            # Normalize
            orthogonal_loss = orthogonal_loss / (self.classifier.num_classes * self.classifier.num_slots_per_class**2)

        def distance_loss(dists: torch.Tensor, proto_slot_dists: torch.Tensor, num_prototypes: int):
            r"""Returns the average minimum distance across a batch, based on the *num_prototypes* most relevant
            prototypes for each sample.

            Args:
                dists (tensor): Tensor of distances to all prototypes. Shape: N x P.
                proto_slot_dists (tensor): For each sample, and each slot, probability that the prototype is allocated
                    to that slot. Shape: N x P x S.
                num_prototypes (int): Number of relevant prototypes to consider.
            """
            # Aggregate relevance of each prototype across all slots
            proto_relevance = proto_slot_dists.sum(dim=2).detach()  # Shape N x P

            # Select ONLY the num_prototypes most relevant prototypes (no duplicates allowed)
            indices = torch.topk(input=proto_relevance, k=num_prototypes, dim=1).indices  # Shape N x num_prototypes
            # For each sample, and each prototype, proto_selection is equal to 1 iff the prototype is relevant.
            proto_selection = torch.zeros_like(proto_relevance)
            proto_selection.scatter_(1, src=torch.ones_like(proto_relevance), index=indices)  # Shape N x P

            # Arbitrary high value to select min distances from masked vector
            max_dist = self.classifier.num_features
            # Min distance among the selected prototypes
            inverted_distances = torch.max((max_dist - dists) * proto_selection, dim=1).values
            return torch.mean(max_dist - inverted_distances)

        proto_slot_probs_per_sample = torch.index_select(input=proto_slot_probs, dim=0, index=label)  # type: ignore

        cluster_cost = distance_loss(
            dists=min_distances,
            proto_slot_dists=proto_slot_probs_per_sample,
            num_prototypes=self.classifier.num_slots_per_class,
        )
        separation_cost = distance_loss(
            dists=min_distances,
            proto_slot_dists=1 - proto_slot_probs_per_sample,
            num_prototypes=self.num_prototypes - self.classifier.num_slots_per_class,
        )

        loss = (
            cross_entropy
            + self.loss_coefficients["clustering"] * cluster_cost
            + self.loss_coefficients["separability"] * separation_cost
            + self.loss_coefficients["regularization"] * l1
            + orthogonal_loss
        )

        batch_accuracy = torch.sum(torch.eq(torch.argmax(output, dim=1), label)).item() / len(label)
        stats = {
            "loss": loss.item(),
            "accuracy": batch_accuracy,
            "cross_entropy": cross_entropy.item(),
            "cluster_cost": cluster_cost.item(),
            "separation_cost": separation_cost.item(),
            "orthogonal_loss": orthogonal_loss.item(),
            "l1": l1.item(),
        }

        return loss, stats

    def train_epoch(
        self,
        dataloaders: dict[str, DataLoader],
        optimizer_mngr: OptimizerManager | torch.optim.Optimizer,
        device: str | torch.device = "cuda:0",
        tqdm_position: int = 0,
        epoch_idx: int = 0,
        verbose: bool = False,
    ) -> dict[str, float]:
        r"""Trains a ProtoPool model for one epoch, performing prototype projection and fine-tuning if necessary.

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
        self.train()
        self.to(device)

        # Training stats
        train_info = {}
        nb_inputs = 0

        # Capture data fetch time relative to total batch time to ensure that there is no bottleneck here
        total_batch_time = 0.0
        total_data_time = 0.0

        gumbel_scale = 0
        training_config = self.training_config
        if training_config["gumbel_epochs"]:
            min_scale, max_scale, gumbel_epochs = (
                training_config["gumbel_min_scale"],
                training_config["gumbel_max_scale"],
                training_config["gumbel_epochs"],
            )
            alpha = (max_scale / min_scale) ** 2 / gumbel_epochs
            gumbel_scale = min_scale * np.sqrt(alpha * epoch_idx) if epoch_idx < gumbel_epochs else max_scale

        # Use training dataloader
        train_loader = dataloaders["train_set"]

        # Show progress on progress bar if needed
        train_iter = tqdm(
            enumerate(train_loader),
            desc=f"Training epoch {epoch_idx}",
            total=len(train_loader),
            leave=False,
            position=tqdm_position,
            disable=not verbose,
        )
        ref_time = time.time()
        batch_idx = 0
        for batch_idx, (xs, ys) in train_iter:
            data_time = time.time() - ref_time
            nb_inputs += xs.size(0)

            # Reset gradients and map the data on the target device
            optimizer_mngr.zero_grad()
            xs, ys = xs.to(device), ys.to(device)

            # Mix-up data
            xs, ys_mix, mix_percentage = batch_mixup(
                data=xs, labels=ys, alpha=0.5 if training_config["use_mix_up"] else 0
            )

            # Perform inference and compute loss
            ys_pred, distances, proto_slot_probs = self.forward(xs, gumbel_scale=gumbel_scale)
            batch_loss, batch_stats = self.loss((ys_pred, distances, proto_slot_probs), ys, ys_mix, mix_percentage)

            # Compute the gradient and update parameters
            batch_loss.backward()
            if isinstance(optimizer_mngr, OptimizerManager):
                optimizer_mngr.optimizer_step(epoch=epoch_idx)
            else:  # Simple optimizer
                optimizer_mngr.step()

            # Update progress bar
            batch_accuracy = batch_stats["accuracy"]
            batch_time = time.time() - ref_time
            postfix_str = (
                f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                f"Batch loss: {batch_loss.item():.3f}, Acc: {batch_accuracy:.3f}, "
                f"Batch time: {batch_time:.3f}s (data: {data_time:.3f})"
            )
            train_iter.set_postfix_str(postfix_str)

            # Update all metrics
            if not train_info:
                train_info = batch_stats
            for key, value in batch_stats.items():
                train_info[key] += value * xs.size(0)

            total_batch_time += batch_time
            total_data_time += data_time
            ref_time = time.time()

        # Clean gradients after last batch
        optimizer_mngr.zero_grad()

        train_info = {f"train/{key}": value / nb_inputs for key, value in train_info.items()}

        # Update batch_num with effective value
        batch_num = batch_idx + 1
        train_info.update(
            {
                "time/batch": total_batch_time / batch_num,
                "time/data": total_data_time / batch_num,
            }
        )
        return train_info

    def epilogue(
        self,
        dataloaders: dict[str, DataLoader],
        optimizer_mngr: OptimizerManager,
        output_dir: str,
        device: str | torch.device = "cuda:0",
        verbose: bool = False,
        num_fine_tuning_epochs: int = 25,
        tqdm_position: int = 0,
        **kwargs,
    ) -> dict[int, dict[str, int | float]]:
        r"""Function called after training, using information from the epilogue field in the training configuration.

        Args:
            dataloaders (dictionary): Dictionary of dataloaders.
            optimizer_mngr (OptimizerManager): Optimizer manager.
            output_dir (str): Unused.
            device (str | device, optional): Hardware device. Default: cuda:0.
            verbose (bool, optional): Display progress bar. Default: False.
            num_fine_tuning_epochs (int, optional): Number of fine-tuning epochs to perform after projection.
                Default: 25.
            tqdm_position (int, optional): Position of the progress bar. Default: 0.

        Returns:
            Projection information.
        """
        # Perform projection
        projection_info = self.project(
            dataloader=dataloaders["projection_set"],
            device=device,
            verbose=verbose,
        )
        eval_info = self.evaluate(dataloader=dataloaders["test_set"], device=device, verbose=verbose)
        logger.info(
            f"After projection. Average loss: {eval_info['loss']:.2f}. "
            f"Average accuracy: {eval_info['accuracy']:.2f}."
        )
        if not self._compatibility_mode:
            self.prune(device=device)
            # Freeze all other parameter groups before fine-tuning
            optimizer_mngr.freeze_non_associated_groups("last_layer_optimizer")

        # Last layer fine-tuning
        fine_tuning_progress = tqdm(
            range(num_fine_tuning_epochs),
            desc="Fine-tuning last layer",
            leave=False,
            position=tqdm_position,
            disable=not verbose,
        )
        for _ in fine_tuning_progress:
            self.train_epoch(
                dataloaders=dataloaders,
                optimizer_mngr=optimizer_mngr.optimizers["last_layer_optimizer"],
                device=device,
                tqdm_position=tqdm_position + 1,
                verbose=verbose,
            )

        return projection_info

    def prune(
        self,
        device: str | torch.device = "cuda:0",
    ):
        r"""Performs prototype pruning after training.

        Args:
            device (str | device, optional): Hardware device. Default: cuda:0.

        """
        logger.info("Performing prototype pruning")
        self.eval()
        self.to(device)

        logger.info(f"Model statistics before pruning: {self.num_prototypes} prototypes.")

        with torch.no_grad():
            if self._compatibility_mode:
                # Makes the computation of the class mapping deterministic
                torch.manual_seed(0)
            normalized_proto_slot_map = (
                nn.functional.gumbel_softmax(self.classifier.proto_slot_map * 1e4, tau=0.5, dim=1).detach().cpu()
            )  # Shape C x P x S
            # For each class, keep track of the related prototypes (one for each slot)
            class_mapping = torch.argmax(normalized_proto_slot_map, dim=1).cpu().numpy()  # Shape C x S

        active_prototypes = torch.tensor(list(set(class_mapping.reshape(-1).tolist()))).to(device)

        # Overwrite prototypes and class slots with selected subset (self.num_prototypes is updated automatically)
        self.classifier.prototypes = nn.Parameter(  # type: ignore
            torch.index_select(input=self.classifier.prototypes, dim=0, index=active_prototypes),
            requires_grad=True,
        )
        self.classifier.proto_slot_map = nn.Parameter(
            torch.index_select(input=self.classifier.proto_slot_map, dim=1, index=active_prototypes),
            requires_grad=True,
        )

        # Update shape of similarity layer for computation purposes
        if hasattr(self.classifier.similarity_layer, "_summation_kernel"):
            self.classifier.similarity_layer.register_buffer(
                "_summation_kernel", torch.ones((self.num_prototypes, self.classifier.num_features, 1, 1))
            )

        logger.info(f"Model statistics after pruning: {self.num_prototypes} prototypes.")

    def project(
        self,
        dataloader: DataLoader,
        device: str | torch.device = "cuda:0",
        verbose: bool = False,
        tqdm_position: int = 0,
    ) -> dict[int, dict[str, int | float]]:
        r"""Performs prototype projection after training.

        Args:
            dataloader (DataLoader): Dataloader containing projection data.
            device (str | device, optional): Hardware device. Default: cuda:0.
            verbose (bool, optional): Display progress bar. Default: False.
            tqdm_position (int, optional): Position of the progress bar. Default: 0.

        Returns:
            Dictionary containing projection information for each prototype.
        """
        logger.info("Performing prototype projection")
        self.eval()
        self.to(device)

        # Compute mapping between classes and prototypes
        class_mapping = self.classifier.class_mapping

        with torch.no_grad():
            if not self._compatibility_mode:
                # Hard assignment of prototypes to slots using one-hot distributions
                hard_proto_slot_map = torch.zeros_like(self.classifier.proto_slot_map).to(device)
                for c in range(self.classifier.num_classes):
                    for s in range(self.classifier.num_slots_per_class):
                        hard_proto_slot_map[c, class_mapping[c, s], s] = 1.0
                self.classifier.proto_slot_map.copy_(hard_proto_slot_map)

        # Show progress on progress bar if needed
        data_iter = tqdm(
            enumerate(dataloader),
            desc="Prototype projection",
            total=len(dataloader),
            leave=False,
            position=tqdm_position,
            disable=not verbose,
        )

        # Number of prototypes and prototype length
        num_prototypes, proto_dim = self.num_prototypes, self.classifier.num_features

        # For each prototype, keep track of:
        #   - the index of the closest projection image
        #   - the coordinates of the vector inside the latent representation of that image
        #   - the corresponding distance
        #   - the corresponding vector
        projection_info = {
            proto_idx: {
                "img_idx": -1,
                "h": -1,
                "w": -1,
                "dist": float("inf"),
            }
            for proto_idx in range(num_prototypes)
        }
        projection_vectors = torch.zeros_like(self.classifier.prototypes)

        with torch.no_grad():
            for batch_idx, (xs, ys) in data_iter:
                # Map to device and perform inference
                xs = xs.to(device)
                feats = self.extractor(xs)  # Shape N x D x H x W
                _, W = feats.shape[2], feats.shape[3]
                distances = self.classifier.similarity_layer.distances(
                    feats, self.classifier.prototypes
                )  # Shape (N, P, H, W)
                min_dist, min_dist_idxs = torch.min(distances.view(distances.shape[:2] + (-1,)), dim=2)

                for img_idx, (_, y) in enumerate(zip(xs, ys)):
                    for proto_idx in class_mapping[y.item()]:
                        # For each entry, only check prototypes that lead to the corresponding class
                        if min_dist[img_idx, proto_idx] < projection_info[proto_idx]["dist"]:
                            h, w = (
                                min_dist_idxs[img_idx, proto_idx].item() // W,
                                min_dist_idxs[img_idx, proto_idx].item() % W,
                            )
                            batch_size = 1 if dataloader.batch_size is None else dataloader.batch_size
                            projection_info[proto_idx] = {
                                "img_idx": batch_idx * batch_size + img_idx,
                                "h": h,
                                "w": w,
                                "dist": min_dist[img_idx, proto_idx].item(),
                            }
                            projection_vectors[proto_idx] = feats[img_idx, :, h, w].view(proto_dim, 1, 1).cpu()

            # Update prototype vectors
            self.classifier.prototypes.copy_(projection_vectors)

        return projection_info

    def explain(
        self,
        img: str | Image.Image,
        preprocess: Callable | None,
        visualizer: SimilarityVisualizer,
        prototype_dir: str = "",
        output_dir: str = "",
        output_format: str = "pdf",
        device: str | torch.device = "cuda:0",
        exist_ok: bool = False,
        disable_rendering: bool = False,
        num_closest: int = 10,
        class_specific: bool = True,
        **kwargs,
    ) -> list[tuple[int, float, bool]]:
        r"""Explains the decision for a particular image.

        Args:
            img (str or Image): Path to image or image itself.
            preprocess (Callable): Preprocessing function.
            visualizer (SimilarityVisualizer): Similarity visualizer.
            prototype_dir (str, optional): Path to directory containing prototype visualizations. Default: "".
            output_dir (str, optional): Path to output directory. Default: "".
            output_format (str, optional): Output file format. Default: pdf.
            device (str | device, optional): Hardware device. Default: cuda:0.
            exist_ok (bool, optional): Silently overwrites existing explanation (if any). Default: False.
            disable_rendering (bool, optional): When True, no visual explanation is generated. Default: False.
            num_closest (int, optional): Number of closest prototypes to display. Default: 10.
            class_specific (bool, optional): If True, only use prototypes from the predicted class. Default: True.

        Returns:
            List of most relevant prototypes for the decision, where each entry is in the form
                (<prototype index>, <similarity score>, <similar>)
            and <similar> indicates whether the prototype is considered similar or dissimilar.
        """
        self.eval()

        # Compute mapping between classes and prototypes
        class_mapping = self.classifier.class_mapping

        if isinstance(img, str):
            img = Image.open(img)

        if preprocess is None:
            preprocess = ToTensor()

        img_tensor = preprocess(img)
        if img_tensor.dim() != 4:
            # Fix number of dimensions if necessary
            img_tensor = torch.unsqueeze(img_tensor, dim=0)

        # Map to device
        self.to(device)
        img_tensor = img_tensor.to(device)

        # Perform inference and get minimum distance to each prototype
        prediction, min_distances, _ = self.forward(img_tensor)
        class_idx = torch.argmax(prediction, dim=1)[0]
        min_distances = min_distances[0]

        # Ignore distances to pruned/non-class specific prototypes
        for proto_idx in range(self.num_prototypes):
            if not self.classifier.prototype_is_active(proto_idx) or (
                class_specific and proto_idx not in class_mapping[class_idx]
            ):
                min_distances[proto_idx] = float("inf")

        # Build explanation
        img_path = os.path.join(output_dir, "original.png")
        if not disable_rendering:
            os.makedirs(os.path.join(output_dir, "test_patches"), exist_ok=exist_ok)
            # Copy source image
            img.save(img_path)
        explanation = ExplanationGraph(output_dir=output_dir)
        explanation.set_test_image(img_path=img_path)
        most_relevant_prototypes = []  # Keep track of most relevant prototypes

        for _ in range(num_closest):
            proto_idx = int(torch.argmin(min_distances).item())
            min_distance = torch.min(min_distances)
            if min_distance == float("inf"):
                # Not enough relevant prototypes
                break
            score = self.classifier.similarity_layer.distances_to_similarities(min_distance).item()
            most_relevant_prototypes.append((proto_idx, score, True))  # ProtoPool only considers positive similarities
            # Recover path to prototype image
            prototype_image_path = os.path.join(prototype_dir, f"prototype_{proto_idx}.png")
            # Generate test image patch
            patch_image_path = os.path.join(output_dir, "test_patches", f"proto_similarity_{proto_idx}.png")
            if not disable_rendering:
                patch_image = visualizer.forward(img=img, img_tensor=img_tensor, proto_idx=proto_idx, device=device)
                patch_image.save(patch_image_path)
            in_class_slot = proto_idx in class_mapping[class_idx]
            explanation.add_similarity(
                prototype_img_path=prototype_image_path,
                test_patch_img_path=patch_image_path,
                label=f"Prototype {proto_idx}\n"
                f"({'not ' if not in_class_slot else ''}in class slot, score: {score:.2f})",
                font_color="black" if in_class_slot else "red",
            )
            # "Disable" prototype from search
            min_distances[proto_idx] = float("inf")
        explanation.add_prediction(int(torch.argmax(prediction).item()))
        if not disable_rendering:
            explanation.render(output_format=output_format)
        return most_relevant_prototypes

    def explain_global(self, prototype_dir: str, output_dir: str, output_format: str = "pdf", **kwargs) -> None:
        r"""Explains the global decision-making process of a CaBRNet model.

        Args:
            prototype_dir (str): Path to directory containing prototype visualizations.
            output_dir (str): Path to output directory.
            output_format (str, optional): Output file format. Default: pdf.
        """
        # Compute mapping between classes and prototypes
        class_mapping = self.classifier.class_mapping

        explanation_graph = graphviz.Graph()
        explanation_graph.attr(layout="circo")
        # Default node configuration
        explanation_graph.attr("node", label="", fixedsize="True", width="2", height="2", fontsize="25")

        def _build_class_node(graph, class_idx):
            graph.node(
                name=f"C{class_idx}",
                shape="circle",
                label=f"Class {class_idx}",
                root="True",
            )
            for prototype in class_mapping[class_idx]:
                img_path = os.path.abspath(os.path.join(prototype_dir, f"prototype_{prototype}.png"))
                graph.node(
                    name=f"P{prototype}_C{class_idx}",
                    shape="plaintext",
                    xlabel=f"P{prototype}",
                    image=img_path,
                    imagescale="true",
                )
                graph.edge(f"C{class_idx}", f"P{prototype}_C{class_idx}")
            return graph

        for class_idx in range(self.classifier.num_classes):
            _build_class_node(explanation_graph, class_idx)

        logger.debug(explanation_graph.source)
        explanation_graph.render(filename=os.path.join(output_dir, "global_explanation"), format=output_format)
