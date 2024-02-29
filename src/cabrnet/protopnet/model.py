import copy
import os
from typing import Any, Callable, Mapping

import graphviz
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from cabrnet.generic.model import CaBRNet
from cabrnet.utils.optimizers import OptimizerManager
from cabrnet.visualisation.visualizer import SimilarityVisualizer
from cabrnet.visualisation.explainer import ExplanationGraph
from cabrnet.utils.save import save_checkpoint
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm


class ProtoPNet(CaBRNet):
    """Class representing a ProtoPNet."""

    def __init__(self, extractor: nn.Module, classifier: nn.Module, **kwargs):
        """Build a ProtoPNet.

        Args:
            extractor: Feature extractor
            classifier: Classification based on extracted features
        """
        super(ProtoPNet, self).__init__(extractor, classifier, **kwargs)

        # Default training configuration
        self.loss_coefficients = {
            "clustering": 0.8,
            "separability": -0.08,
            "regularization": 0.0001,
        }
        self.projection_config = {
            "start_epoch": 10,
            "frequency": 10,
            "num_ft_epochs": 20,
        }

    def _load_legacy_state_dict(self, legacy_state: Mapping[str, any]) -> None:
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
        super().load_state_dict(final_state, strict=False)

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        """Overloads nn.Module load_state_dict to take legacy state dictionaries into account"""
        legacy_state = any([key.startswith("features") for key in state_dict.keys()])
        if legacy_state:
            logger.info("Legacy state dictionary detected, performing import.")
            self._load_legacy_state_dict(state_dict)
        else:
            super().load_state_dict(state_dict, **kwargs)

    def register_training_params(self, training_config: dict[str, Any]) -> None:
        """Save additional information from the training configuration directly into the model

        Args:
            training_config: dictionary containing training configuration
        """
        if training_config.get("auxiliary_info") is None:
            logger.warning("Empty auxiliary training configuration. Using default values")
            return
        aux_training_params = training_config["auxiliary_info"]

        if aux_training_params.get("loss_coefficients") is not None:
            self.loss_coefficients = aux_training_params["loss_coefficients"]
        if aux_training_params.get("projection_config") is not None:
            self.projection_config = aux_training_params["projection_config"]

    def similarities(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Return similarity scores
        Args:
            x: input tensor

        Returns:
            tensor of similarity scores
        """
        x = self.extractor(x, **kwargs)
        return self.classifier.similarity_layer(x, self.classifier.prototypes)[0]

    def loss(self, model_output: Any, label: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Loss function.

        Args:
            model_output: Model output, in this case a tuple containing the prediction and the minimum distances.
            label: Batch labels

        Returns:
            loss tensor and batch statistics
        """
        output, min_distances = model_output

        # Cross-entropy loss
        cross_entropy = torch.nn.functional.cross_entropy(output, label)

        # Arbitrary high value to select min distances from masked vector
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
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class

            # Target vector is equal to max_dist everywhere except for the selected prototypes
            cluster_cost = torch.mean(
                torch.min(prototypes_of_correct_class * min_distances + max_dist * prototypes_of_wrong_class, dim=1)[0]
            )
            # Target vector is equal to max_dist for the selected prototypes
            separation_cost = torch.mean(
                torch.min(prototypes_of_wrong_class * min_distances + max_dist * prototypes_of_correct_class, dim=1)[0]
            )

            l1_mask = 1 - torch.t(self.classifier.proto_class_map)
            l1 = (self.classifier.last_layer.weight * l1_mask).norm(p=1)

        loss = (
            cross_entropy
            + self.loss_coefficients["clustering"] * cluster_cost
            + self.loss_coefficients["separability"] * separation_cost
            + self.loss_coefficients["regularization"] * l1
        )

        batch_accuracy = torch.sum(torch.eq(torch.argmax(output, dim=1), label)).item() / len(label)
        stats = {
            "accuracy": batch_accuracy,
            "cross_entropy": cross_entropy.item(),
            "cluster_cost": cluster_cost.item(),
            "separation_cost": separation_cost.item(),
            "l1": l1.item(),
        }

        return loss, stats

    def _train_epoch(
        self,
        dataloaders: dict[str, DataLoader],
        optimizer_mngr: OptimizerManager | torch.optim.Optimizer,
        device: str = "cuda:0",
        progress_bar_position: int = 0,
        epoch_idx: int = 0,
        verbose: bool = False,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Internal function: train the model for exactly one epoch.

        Args:
            dataloaders: Dictionary of dataloaders
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

        # Use training dataloader
        train_loader = dataloaders["train_set"]

        # Show progress on progress bar if needed
        train_iter = tqdm(
            enumerate(train_loader),
            desc=f"Training epoch {epoch_idx}",
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
            batch_loss, batch_stats = self.loss((ys_pred, distances), ys)

            # Compute the gradient and update parameters
            batch_loss.backward()
            if isinstance(optimizer_mngr, OptimizerManager):
                optimizer_mngr.optimizer_step(epoch=epoch_idx)
            else:  # Simple optimizer
                optimizer_mngr.step()

            # Update progress bar
            batch_accuracy = batch_stats["accuracy"]
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

    def train_epoch(
        self,
        dataloaders: dict[str, DataLoader],
        optimizer_mngr: OptimizerManager,
        device: str = "cuda:0",
        progress_bar_position: int = 0,
        epoch_idx: int = 0,
        verbose: bool = False,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Train a ProtoPNet model for one epoch, performing prototype projection and fine-tuning if necessary.

        Args:
            dataloaders: Dictionary of dataloaders
            optimizer_mngr: Optimizer manager
            device: Target device
            progress_bar_position: Position of the progress bar.
            epoch_idx: Epoch index
            max_batches: Max number of batches (early stop for small compatibility tests)
            verbose: Display progress bar

        Returns:
            dictionary containing learning statistics
        """
        # Train for exactly one epoch using the OptimizerManager
        train_info = self._train_epoch(
            dataloaders=dataloaders,
            optimizer_mngr=optimizer_mngr,
            device=device,
            progress_bar_position=progress_bar_position,
            epoch_idx=epoch_idx,
            verbose=verbose,
            max_batches=max_batches,
        )
        # Perform prototype projection if necessary
        if (
            epoch_idx >= self.projection_config["start_epoch"]
            and (epoch_idx - self.projection_config["start_epoch"]) % self.projection_config["frequency"] == 0
        ):
            self.project(
                data_loader=dataloaders["projection_set"],
                device=device,
                verbose=verbose,
                progress_bar_position=progress_bar_position,
            )
            # Freeze all parameters except last layer
            for group in optimizer_mngr.param_groups:
                optimizer_mngr.freeze_group(name=group, freeze=(group != "last_layer"))

            fine_tuning_progress = tqdm(
                range(self.projection_config["num_ft_epochs"]),
                desc="Fine-tuning last layer",
                leave=False,
                position=progress_bar_position,
                disable=not verbose,
            )
            for _ in fine_tuning_progress:
                train_info = self._train_epoch(
                    dataloaders=dataloaders,
                    optimizer_mngr=optimizer_mngr.optimizers["last_layer_optimizer"],  # type: ignore
                    device=device,
                    progress_bar_position=progress_bar_position + 1,
                    epoch_idx=epoch_idx,
                    verbose=verbose,
                    max_batches=max_batches,
                )

        return train_info

    def epilogue(
        self,
        dataloaders: dict[str, DataLoader],
        visualizer: SimilarityVisualizer,
        output_dir: str,
        model_config: str,
        training_config: str,
        dataset_config: str,
        seed: int,
        device: str = "cuda:0",
        verbose: bool = False,
        pruning_threshold: int = 3,
        num_nearest_patches: int = 6,
        **kwargs,
    ) -> None:
        """Function called after training, using information from the epilogue field in the training configuration.

        Args:
            dataloaders: dataloader containing projection data
            visualizer: patch visualizer
            output_dir: output directory
            model_config: path to YML model configuration
            training_config: path to YML training configuration
            dataset_config: path to YML dataset configuration
            seed: initial random seed
            pruning_threshold: Pruning threshold
            num_nearest_patches: Number of patches near the prototype to look at
            device: target device
            verbose: display progress bar
        """
        assert not self._compatibility_mode, "Compatibility mode not supported during epilogue"
        # Perform projection
        projection_info = self.project(data_loader=dataloaders["projection_set"], device=device, verbose=verbose)
        eval_info = self.evaluate(dataloader=dataloaders["test_set"], device=device, verbose=verbose)
        save_checkpoint(
            directory_path=os.path.join(output_dir, f"projected"),
            model=self,
            model_config=model_config,
            optimizer_mngr=None,
            training_config=training_config,
            dataset_config=dataset_config,
            visualization_config=visualizer.config_file,
            epoch="projected",
            seed=seed,
            device=device,
            stats=eval_info,
        )

        # Extract prototypes
        self.extract_prototypes(
            dataloader_raw=dataloaders["projection_set_raw"],
            dataloader=dataloaders["projection_set"],
            projection_info=projection_info,
            visualizer=visualizer,
            dir_path=os.path.join(output_dir, "prototypes"),
            device=device,
            verbose=verbose,
        )

        # Prune weak prototypes
        self.prune(
            data_loader=dataloaders["projection_set"],
            pruning_threshold=pruning_threshold,
            num_nearest_patches=num_nearest_patches,
            device=device,
            verbose=verbose,
            progress_bar_position=0,
        )

    def prune(
        self,
        data_loader: DataLoader,
        pruning_threshold: int = 3,
        num_nearest_patches: int = 6,
        device: str = "cuda:0",
        verbose: bool = False,
        progress_bar_position: int = 0,
    ) -> None:
        """Prune prototypes based on threshold.

        Args:
            data_loader: dataloader containing projection data
            pruning_threshold: Pruning threshold
            num_nearest_patches: Number of patches near the prototype to look at
            device: target device
            verbose: display progress bar
            progress_bar_position: position of the progress bar.
        """
        logger.info("Performing prototypes pruning")
        self.eval()
        self.to(device)

        # map each prototype to its class
        class_mapping = torch.argmax(self.classifier.proto_class_map, dim=1).detach().cpu()

        data_iter = tqdm(
            data_loader,
            total=len(data_loader),
            leave=False,
            position=progress_bar_position,
            disable=not verbose,
            desc="Prototypes pruning",
        )

        prune_info = {proto_idx: [] for proto_idx in range(self.classifier.num_prototypes)}

        with torch.no_grad():
            for xs, ys in data_iter:
                xs = xs.to(device)
                distances = self.l2_distances(xs)
                min_dist, _ = torch.min(distances.view(distances.shape[:2] + (-1,)), dim=2)

                for img_idx, (_, y) in enumerate(zip(xs, ys)):
                    for proto_idx in range(self.classifier.num_prototypes):
                        if len(prune_info[proto_idx]) < num_nearest_patches:
                            prune_info[proto_idx].append(
                                {"dist": min_dist[img_idx, proto_idx].item(), "class": y.item()}
                            )
                            # sort the dictionary by distance every time a new value is added
                            prune_info[proto_idx] = sorted(prune_info[proto_idx], key=lambda d: d["dist"])
                        else:
                            if min_dist[img_idx, proto_idx].item() < prune_info[proto_idx][-1]["dist"]:
                                prune_info[proto_idx][-1] = {
                                    "dist": min_dist[img_idx, proto_idx].item(),
                                    "class": y.item(),
                                }
                            # sort the dictionary by distance every time a new value is added
                            prune_info[proto_idx] = sorted(prune_info[proto_idx], key=lambda d: d["dist"])

        logger.info(f"Model statistics before pruning: {self.classifier.num_prototypes} prototypes.")
        index_prototypes_to_keep = []
        index_prototypes_to_prune = []
        for proto_idx, proto_info in prune_info.items():
            # we count the number of patches of the CORRECT class
            counter = sum([1 for patch in proto_info if patch["class"] == class_mapping[proto_idx]])
            if counter >= pruning_threshold:
                index_prototypes_to_keep.append(proto_idx)
            else:
                index_prototypes_to_prune.append(proto_idx)

        if self._compatibility_mode:
            index_prototypes_to_keep = torch.tensor(index_prototypes_to_keep).to(device)

            # overwrite prototypes with selected subset
            self.classifier.prototypes = nn.Parameter(
                torch.index_select(input=self.classifier.prototypes, dim=0, index=index_prototypes_to_keep),
                requires_grad=True,
            )

            # update last layer
            pruned_last_layer = nn.Linear(
                in_features=self.classifier.num_prototypes, out_features=self.classifier.num_classes, bias=False
            )
            pruned_last_layer.weight.data.copy_(
                torch.index_select(input=self.classifier.last_layer.weight.data, dim=1, index=index_prototypes_to_keep)
            )
            self.classifier.last_layer = pruned_last_layer

            # update shape of similarity layer for computation purposes
            self.classifier.similarity_layer.register_buffer(
                "_summation_kernel", torch.ones((self.classifier.num_prototypes, self.classifier.num_features, 1, 1))
            )

            # update mapping between prototypes and classes
            self.classifier.proto_class_map = torch.index_select(
                input=self.classifier.proto_class_map, dim=0, index=index_prototypes_to_keep
            )
            logger.info(f"Model statistics after pruning: {self.classifier.num_prototypes} prototypes.")
        else:
            last_layer_weights = self.classifier.last_layer.weight.data
            for p_index in index_prototypes_to_prune:
                last_layer_weights[:, p_index] = 0

            self.classifier.last_layer.weight.data.copy_(last_layer_weights)
            logger.info(
                f"Model statistics after pruning: "
                f"{self.classifier.num_prototypes-len(index_prototypes_to_prune)} prototypes."
            )

    def project(
        self,
        data_loader: DataLoader,
        device: str = "cuda:0",
        verbose: bool = False,
        progress_bar_position: int = 0,
    ) -> dict[int, dict]:
        """
        Perform prototype projection after training
        Args:
            data_loader: dataloader containing projection data
            device: target device
            verbose: display progress bar
            progress_bar_position: position of the progress bar.
        Returns:
            dictionary containing projection information for each prototype

        """
        logger.info("Performing prototype projection")
        self.eval()
        self.to(device)

        if data_loader.batch_size != 1:
            logger.warning(
                "Projection results may vary depending on batch size, see issue "
                "https://discuss.pytorch.org/t/cudnn-causing-inconsistent-test-results-depending-on-batch-size/189277"
            )

        # For each class, keep track of the related prototypes
        np_proto_class_map = self.classifier.proto_class_map.detach().cpu().numpy()
        class_mapping = {c: list(np.nonzero(np_proto_class_map[:, c])[0]) for c in range(self.classifier.num_classes)}
        # Sanity check to ensure that each class is associated with at least one prototype
        for class_idx in range(self.classifier.num_classes):
            if class_idx not in class_mapping:
                logger.error(f"Inaccessible class {class_idx}!")

        # Show progress on progress bar if needed
        data_iter = tqdm(
            enumerate(data_loader),
            desc="Prototype projection",
            total=len(data_loader),
            leave=False,
            position=progress_bar_position,
            disable=not verbose,
        )

        # Original number of prototypes (before pruning) and prototype length
        max_num_prototypes, proto_dim = self.classifier.max_num_prototypes, self.classifier.num_features

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
            for proto_idx in range(max_num_prototypes)
        }
        projection_vectors = torch.zeros_like(self.classifier.prototypes)

        with torch.no_grad():
            for batch_idx, (xs, ys) in data_iter:
                # Map to device and perform inference
                xs = xs.to(device)
                feats = self.extractor(xs)  # Shape N x D x H x W
                _, W = feats.shape[2], feats.shape[3]
                distances = self.classifier.similarity_layer.L2_square_distance(
                    feats, self.classifier.prototypes
                )  # Shape (N, P, H, W)
                min_dist, min_dist_idxs = torch.min(distances.view(distances.shape[:2] + (-1,)), dim=2)

                for img_idx, (x, y) in enumerate(zip(xs, ys)):
                    if y.item() not in class_mapping:
                        # Class is not associated with any prototype (this is bad...)
                        continue
                    for proto_idx in class_mapping[y.item()]:
                        # For each entry, only check prototypes that lead to the corresponding class
                        if min_dist[img_idx, proto_idx] < projection_info[proto_idx]["dist"]:
                            h, w = (
                                min_dist_idxs[img_idx, proto_idx].item() // W,
                                min_dist_idxs[img_idx, proto_idx].item() % W,
                            )
                            projection_info[proto_idx] = {
                                "img_idx": batch_idx * data_loader.batch_size + img_idx,  # type: ignore
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
        img_path: str,
        preprocess: Callable,
        visualizer: SimilarityVisualizer,
        prototype_dir_path: str,
        output_dir_path: str,
        device: str,
        exist_ok: bool = False,
        num_closest: int = 10,
        **kwargs,
    ) -> None:
        """Explain the decision for a particular image

        Args:
            img_path: raw original image
            preprocess: preprocessing function
            visualizer: prototype visualizer
            prototype_dir_path: path to directory containing prototype visualizations
            output_dir_path: path to output directory containing the explanation
            device: target hardware device
            exist_ok: silently overwrite existing explanation if any
            num_closest: number of closest prototypes to display
        """
        self.eval()
        img = Image.open(img_path)
        img_tensor = preprocess(img)
        if img_tensor.dim() != 4:
            # Fix number of dimensions if necessary
            img_tensor = torch.unsqueeze(img_tensor, dim=0)

        # Perform inference and get minimum distance to each prototype
        prediction, min_distances = self.forward(img_tensor)
        class_idx = torch.argmax(prediction, dim=1)[0]
        min_distances = min_distances[0]

        # Ignore distances to pruned prototypes
        for proto_idx in range(self.classifier.num_prototypes):
            if int(torch.max(self.classifier.proto_class_map[proto_idx])) == 0:
                min_distances[proto_idx] = float("inf")

        # Build explanation
        os.makedirs(os.path.join(output_dir_path, "test_patches"), exist_ok=exist_ok)
        explanation = ExplanationGraph(output_dir=output_dir_path)
        explanation.set_test_image(img_path=img_path)

        for _ in range(num_closest):
            proto_idx = int(torch.argmin(min_distances).item())
            # Recover path to prototype image
            prototype_image_path = os.path.join(prototype_dir_path, f"prototype_{proto_idx}.png")
            prototype_class_idx = int(torch.argmax(self.classifier.proto_class_map[proto_idx]))
            # Generate test image patch
            patch_image = visualizer.forward(
                model=self, img=img, img_tensor=img_tensor, proto_idx=proto_idx, device=device
            )
            patch_image_path = os.path.join(output_dir_path, "test_patches", f"proto_similarity_{proto_idx}.png")
            patch_image.save(patch_image_path)
            explanation.add_similarity(
                prototype_img_path=prototype_image_path,
                test_patch_img_path=patch_image_path,
                label=f"Prototype {proto_idx} (class {prototype_class_idx})",
                font_color="black" if class_idx == prototype_class_idx else "red",
            )
            # "Disable" prototype from search
            min_distances[proto_idx] = float("inf")
        explanation.render()

    def explain_global(self, prototype_dir_path: str, output_dir_path: str, **kwargs) -> None:
        """Build global explanation_graph.

        Args:
            prototype_dir_path: Path where the prototypes are stored on the system.
            output_dir_path: Path where the result graph will be saved.
        """
        proto_class_map = self.classifier.proto_class_map.detach().cpu().numpy()
        class_mapping = {c: list(np.nonzero(proto_class_map[:, c])[0]) for c in range(self.classifier.num_classes)}

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
                img_path = os.path.relpath(
                    os.path.join(prototype_dir_path, f"prototype_{prototype}.png"), output_dir_path
                )
                graph.node(
                    name=f"P{prototype}",
                    shape="plaintext",
                    xlabel=f"P{prototype}",
                    image=img_path,
                    imagescale="true",
                )
                graph.edge(f"C{class_idx}", f"P{prototype}")
            return graph

        for class_idx in range(self.classifier.num_classes):
            _build_class_node(explanation_graph, class_idx)

        logger.debug(explanation_graph.source)
        explanation_graph.render(filename=os.path.join(output_dir_path, "global_explanation"))
