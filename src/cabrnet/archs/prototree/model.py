import copy
import os
import time
from typing import Any, Callable

import graphviz
import torch
import torch.nn as nn
import torch.nn.functional
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from cabrnet.archs.generic.decision import CaBRNetClassifier
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.archs.prototree.decision import ProtoTreeClassifier, SamplingStrategy
from cabrnet.core.utils.optimizers import OptimizerManager
from cabrnet.core.utils.tree import MappingMode, TreeNode
from cabrnet.core.visualization.explainer import ExplanationGraph
from cabrnet.core.visualization.visualizer import SimilarityVisualizer


class ProtoTree(CaBRNet):
    r"""CaBRNet model implementing the ProtoTree architecture.

    Attributes:
        extractor: Model used to extract convolutional features from the input image.
        classifier: Model used to compute the classification, based on similarity scores with a set of prototypes.
    """

    def __init__(self, extractor: nn.Module, classifier: CaBRNetClassifier, **kwargs):
        r"""Initializes a ProtoTree.

        Args:
            extractor (Module): Feature extractor.
            classifier (CaBRNetClassifier): Classification based on extracted features.
        """
        super(ProtoTree, self).__init__(extractor, classifier, **kwargs)

        # Constant tensor for internal computations
        self.register_buffer("_eye", torch.eye(self.classifier.num_classes))

    def get_extra_state(self) -> dict[str, Any] | None:
        r"""Returns the decision tree architecture to be saved in state_dict.

        This is automatically called by state_dict().
        """
        if isinstance(self.classifier, ProtoTreeClassifier):
            return self.classifier.tree.export_arch()
        # When using PRP visualization with Captum, classifier is no longer a ProtoTreeClassifier
        return None

    def set_extra_state(self, state: dict[str, Any]) -> None:  # type: ignore
        r"""Rebuilds a decision tree from the architecture information.

        This is automatically called by load_state_dict().

        Args:
            state (state dictionary): Information returned by get_extra_state().
        """
        if isinstance(self.classifier, ProtoTreeClassifier):
            # When using PRP visualization with Captum, classifier is no longer a ProtoTreeClassifier
            self.classifier.tree = TreeNode.build_from_arch(state)
            # Update number of active prototypes
            self.classifier._active_prototypes = self.classifier.tree.active_prototypes

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

        for legacy_key in legacy_keys:
            if legacy_key.startswith("_net"):
                # Feature extractor
                cbrn_key = legacy_key.replace("_net", "extractor.convnet")
                if cbrn_key not in cbrn_keys:
                    raise ValueError(f"No parameter matching {legacy_key}. Check that model architectures are similar.")
            elif legacy_key.startswith("_add_on"):
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
            elif legacy_key == "prototype_layer.prototype_vectors":
                cbrn_key = "classifier.prototypes"
            else:
                if not legacy_key.startswith("_root"):
                    raise ValueError(f"Unexpected parameter {legacy_key}")
                # Iterate on letters in the key (first, remove '.')
                symbols = legacy_key[6:].replace(".", "")
                possible_keys = [key for key in cbrn_keys if key.startswith("classifier.tree.")]
                for index, symbol in enumerate(symbols):
                    if symbol == "l":
                        # Keep only keys for which the nsim keyword is present
                        possible_keys = [key for key in possible_keys if "nsim" in key.split(".")[2 + index]]
                    elif symbol == "r":
                        # Keep only keys for which the nsim keyword is not present
                        possible_keys = [key for key in possible_keys if "nsim" not in key.split(".")[2 + index]]
                    else:
                        if len(possible_keys) != 1:
                            raise ValueError(f"Could not match leaf distribution. Candidates: {possible_keys}")
                        break
                cbrn_key = possible_keys[0]
                # Expand dimension of leaf distribution
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

    def load_state_dict(self, state_dict: dict[str, Any], **kwargs):  # type: ignore
        r"""Overloads nn.Module load_state_dict to take legacy state dictionaries into account.

        Args:
            state_dict (state dictionary): State dictionary.
        """
        legacy_state = any([key.startswith("_net") for key in state_dict.keys()])
        if legacy_state:
            logger.info("Legacy state dictionary detected, performing import.")
            self._load_legacy_state_dict(state_dict)
        else:
            super().load_state_dict(state_dict, **kwargs)

    def analyse_leafs(self, pruning_threshold: float = 0.01) -> None:
        r"""Analyses the leaf distributions.

        Args:
            pruning_threshold (float, optional): Expected pruning threshold. Default: 0.01.
        """
        logger.info("Starting leaf analysis.")

        # Find classes that are not covered by any leaf
        covered_classes = [torch.argmax(leaf.distribution).item() for leaf in self.classifier.tree.leaves]
        missing_classes = [label for label in range(self.classifier.num_classes) if label not in covered_classes]
        if missing_classes:
            logger.warning(f"Missing classes from decision tree: {missing_classes}")

        # Percentage of leaves that would remain after pruning based on threshold
        remaining_leaves = [
            leaf for leaf in self.classifier.tree.leaves if torch.max(leaf.distribution).item() > pruning_threshold
        ]
        logger.info(
            f"Remaining leaves after pruning with threshold {pruning_threshold}: {len(remaining_leaves)} "
            f"({len(remaining_leaves)/self.classifier.tree.num_leaves*100:.1f}%)"
        )

    def loss(self, model_output: Any, label: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict[str, float]]:
        r"""Loss function.

        Args:
            model_output (Any): Model output, in this case a tuple containing the prediction and the leaf probabilities.
            label (tensor): Batch labels.

        Returns:
            Loss tensor and batch accuracy.
        """
        ys_pred, _ = model_output
        if self.classifier.log_probabilities:
            # Prediction already given as a log value
            batch_loss = torch.nn.functional.nll_loss(ys_pred, label)
        else:
            batch_loss = torch.nn.functional.nll_loss(torch.log(ys_pred), label)
        batch_accuracy = torch.sum(torch.eq(torch.argmax(ys_pred, dim=1), label)).item() / len(label)
        return batch_loss, {"loss": batch_loss.item(), "accuracy": batch_accuracy}

    def train_epoch(
        self,
        dataloaders: dict[str, DataLoader],
        optimizer_mngr: OptimizerManager,
        device: str | torch.device = "cuda:0",
        tqdm_position: int = 0,
        epoch_idx: int = 0,
        verbose: bool = False,
    ) -> dict[str, float]:
        r"""Trains the model for one epoch.

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

        # Record original leaf distributions
        with torch.no_grad():
            old_dist_params: dict[int, torch.Tensor] = {}
            for leaf in self.classifier.tree.leaves:
                old_dist_params[leaf.node_id] = leaf._relative_distribution.detach().clone()

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
        batch_num, batch_idx = len(train_loader), 0
        ref_time = time.time()

        for batch_idx, (xs, ys) in train_iter:
            data_time = time.time() - ref_time
            nb_inputs += xs.size(0)

            # Reset gradients and map the data on the target device
            optimizer_mngr.zero_grad()
            xs, ys = xs.to(device), ys.to(device)

            # Perform inference and compute loss
            ys_pred, info = self.forward(xs)
            batch_loss, batch_stats = self.loss((ys_pred, info), ys)

            # Compute the gradient and update parameters
            batch_loss.backward()
            optimizer_mngr.optimizer_step(epoch=epoch_idx)

            # Update leaves with derivative-free algorithm
            # Convert integer label into on-hot encoding
            target = self._eye[ys]
            with torch.no_grad():
                for leaf in self.classifier.tree.leaves:
                    # Recover probability to end up in that leaf
                    probs = info[leaf.node_id]["absolute_probability"]  # Shape N x 1
                    if self.classifier.log_probabilities:
                        update = torch.exp(
                            torch.logsumexp(probs + leaf.distribution + torch.log(target) - ys_pred, dim=0)
                        )
                    else:
                        update = torch.sum((probs * leaf.distribution * target) / ys_pred, dim=0)
                    leaf._relative_distribution -= old_dist_params[leaf.node_id] / batch_num
                    torch.nn.functional.relu_(leaf._relative_distribution)
                    leaf._relative_distribution += update

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
        # Update batch_num with effective value
        batch_num = batch_idx + 1
        train_info = {key: value / nb_inputs for key, value in train_info.items()}
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
        pruning_threshold: float = 0.0,
        merge_same_decision: bool = False,
        **kwargs: Any,
    ) -> dict[int, dict[str, int | float]]:
        r"""Function called after training, using information from the epilogue field in the training configuration.

        Args:
            dataloaders (dictionary): Dictionary of dataloaders.
            optimizer_mngr (OptimizerManager): Unused.
            output_dir (str): Unused.
            device (str | device, optional): Hardware device. Default: cuda:0.
            verbose (bool, optional): Display progress bar. Default: False.
            pruning_threshold (float, optional): Pruning threshold. Default: 0.0 (no pruning).
            merge_same_decision (bool, optional): If True, merges branches leading to same top decision. Default: False.

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

        if pruning_threshold <= 0.0:
            logger.warning(f"Leaf pruning disabled (threshold is {pruning_threshold})")
        self.prune(pruning_threshold=pruning_threshold, merge_same_decision=merge_same_decision)

        return projection_info

    def prune(self, pruning_threshold: float = 0.01, merge_same_decision: bool = False) -> None:
        r"""Prunes the decision tree based on a threshold.

        Args:
            pruning_threshold (float, optional): Pruning threshold. Default: 0.01.
            merge_same_decision (bool, optional): If True, merges branches leading to same top decision. Default: False.
        """
        logger.info(f"Pruning tree. Threshold: {pruning_threshold}")
        num_prototypes_before = self.classifier.tree.num_prototypes
        num_leaves_before = self.classifier.tree.num_leaves
        logger.info(
            f"Tree statistics before pruning: {num_leaves_before} leaves, " f"{num_prototypes_before} prototypes."
        )
        self.classifier.tree.prune_children(threshold=pruning_threshold)
        if merge_same_decision:
            self.classifier.tree.prune_similar_children()
        # Update list of active prototypes
        self.classifier._active_prototypes = self.classifier.tree.active_prototypes
        num_prototypes = len(self.classifier._active_prototypes)
        num_leaves = self.classifier.tree.num_leaves
        logger.info(
            f"Tree statistics after pruning: {num_leaves} leaves "
            f"({(num_leaves_before-num_leaves)/num_leaves_before*100:.1f} % pruned), "
            f"{num_prototypes} prototypes "
            f"({(num_prototypes_before-num_prototypes)/num_prototypes_before*100:.1f} % pruned)."
        )

    def project(
        self,
        dataloader: DataLoader,
        device: str | torch.device = "cuda:0",
        verbose: bool = False,
        tqdm_position: int = 0,
    ) -> dict[int, dict]:
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

        # For each class, keep track of the related prototypes
        class_mapping = self.classifier.tree.get_mapping(mode=MappingMode.CLASS_TO_PROTOTYPE)
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

        # Original number of prototypes (before pruning) and prototype length
        num_prototypes, proto_dim = self.classifier.prototypes.shape[0:2]

        # For each prototype, keep track of:
        #   - the index of the closest projection image
        #   - the coordinates of the vector inside the latent representation of that image
        #   - the corresponding similarity score
        #   - the corresponding vector
        projection_info = {
            proto_idx: {
                "img_idx": -1,
                "h": -1,
                "w": -1,
                "score": 0.0,
            }
            for proto_idx in range(num_prototypes)
        }
        projection_vectors = self.classifier.prototypes.clone()

        with torch.no_grad():
            for batch_idx, (xs, ys) in data_iter:
                # Map to device and perform inference
                xs = xs.to(device)
                feats = self.extractor(xs)  # Shape N x D x H x W
                _, W = feats.shape[2], feats.shape[3]
                similarities = self.classifier.similarity_layer(feats, self.classifier.prototypes)  # Shape (N, P, H, W)
                max_sim, max_sim_idxs = torch.max(similarities.view(similarities.shape[:2] + (-1,)), dim=2)

                for img_idx, (_, y) in enumerate(zip(xs, ys)):
                    if y.item() not in class_mapping:
                        # Class is not associated with any prototype (this is bad...)
                        continue
                    for proto_idx in class_mapping[y.item()]:
                        # For each entry, only check prototypes that lead to the corresponding class
                        if max_sim[img_idx, proto_idx] > projection_info[proto_idx]["score"]:
                            h, w = (
                                max_sim_idxs[img_idx, proto_idx].item() // W,
                                max_sim_idxs[img_idx, proto_idx].item() % W,
                            )
                            batch_size = 1 if dataloader.batch_size is None else dataloader.batch_size
                            projection_info[proto_idx] = {
                                "img_idx": batch_idx * batch_size + img_idx,
                                "h": h,
                                "w": w,
                                "score": max_sim[img_idx, proto_idx].item(),
                            }
                            projection_vectors[proto_idx] = feats[img_idx, :, h, w].view(proto_dim, 1, 1).cpu()

                            # Sanity check
                            assert similarities[img_idx, proto_idx, h, w].item() == max_sim[img_idx, proto_idx].item()

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
        strategy: SamplingStrategy = SamplingStrategy.GREEDY,
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
            strategy (SamplingStrategy, optional): Tree sampling strategy. Default: Greedy.


        Returns:
            List of most relevant prototypes for the decision, where each entry is in the form
                (<prototype index>, <similarity score>, <similar>)
            and <similar> indicates whether the prototype is considered similar or dissimilar.
        """
        self.eval()

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

        # Perform inference and get decision leaf
        prediction, tree_info = self.forward(img_tensor, strategy=strategy)
        leaf_id = tree_info["decision_leaf"][0]

        # Compute path to leaf  TODO: This could be done once during model construction, then after pruning
        leaf_path = self.classifier.tree.get_mapping(mode=MappingMode.NODE_PATHS)[leaf_id]

        # Build explanation
        img_path = os.path.join(output_dir, "original.png")
        if not disable_rendering:
            os.makedirs(os.path.join(output_dir, "test_patches"), exist_ok=exist_ok)
            # Copy source image
            img.save(img_path)
        explanation = ExplanationGraph(output_dir=output_dir)
        explanation.set_test_image(img_path=img_path)
        prototype_mapping = self.classifier.tree.get_mapping(mode=MappingMode.NODE_TO_PROTOTYPE)
        node_mapping = self.classifier.tree.get_mapping(mode=MappingMode.ID_TO_NODE)
        parent_id = leaf_path[0]
        proto_idx = prototype_mapping[leaf_path[0]][0]  # Index of the first prototype
        most_relevant_prototypes = []  # Keep track of most relevant prototypes
        for node_id in leaf_path[1:]:
            # Recover path to prototype image
            prototype_image_path = os.path.join(prototype_dir, f"prototype_{proto_idx}.png")
            score = tree_info[node_id]["conditional_probability"].item()
            if node_id == node_mapping[parent_id].get_submodule(f"{parent_id}_child_nsim").node_id:
                # No similarity
                most_relevant_prototypes.append((proto_idx, 1 - score, False))
                if not disable_rendering:
                    explanation.add_similarity(
                        prototype_img_path=prototype_image_path,
                        test_patch_img_path=img_path,
                        label=f"Not similar\n (Score: {1-score:.2f})",
                    )
            else:
                # Similarity
                most_relevant_prototypes.append((proto_idx, score, True))
                patch_image_path = os.path.join(output_dir, "test_patches", f"proto_similarity_{proto_idx}.png")
                if not disable_rendering:
                    patch_image = visualizer.forward(img=img, img_tensor=img_tensor, proto_idx=proto_idx, device=device)
                    patch_image.save(patch_image_path)
                explanation.add_similarity(
                    prototype_img_path=prototype_image_path,
                    test_patch_img_path=patch_image_path,
                    label=f"Similar\n (Score: {score:.2f})",
                )
            parent_id = node_id
            if prototype_mapping[node_id] is None:
                break
            proto_idx = prototype_mapping[node_id][0]  # Update index of prototype associated with next node
        explanation.add_prediction(int(torch.argmax(prediction).item()))
        if not disable_rendering:
            explanation.render(output_format=output_format)
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

        def build_tree_explanation(node: nn.Module, graph: graphviz.Digraph) -> graphviz.Digraph:
            r"""Builds tree explanation recursively.

            Args:
                node (Module): current node
                graph (Digraph): current graph

            Returns:
                Updated graph
            """
            if node.proto_idxs is None:
                # Leaf
                class_idx = torch.argmax(node.distribution).item()
                graph.node(name=f"node_{node.node_id}", label=f"Class {class_idx}", fontsize="25", height="0.5")
            else:
                proto_idx = node.proto_idxs[0]
                img_path = os.path.abspath(os.path.join(prototype_dir, f"prototype_{proto_idx}.png"))
                graph.node(name=f"node_{node.node_id}", image=img_path, imagescale="True")
                for child_name, similarity in zip(["nsim", "sim"], ["not similar", "similar"]):
                    child = node.get_submodule(f"{node.node_id}_child_{child_name}")
                    graph = build_tree_explanation(child, graph)
                    graph.edge(
                        tail_name=f"node_{node.node_id}",
                        head_name=f"node_{child.node_id}",
                        label=similarity,
                        fontsize="25",
                    )
            return graph

        node = self.classifier.tree
        explanation_graph = graphviz.Digraph()
        explanation_graph.attr("node", shape="plaintext", label="", fixedsize="True", width="2", height="2")
        explanation_graph = build_tree_explanation(node, explanation_graph)
        logger.debug(explanation_graph.source)
        explanation_graph.render(filename=os.path.join(output_dir, "global_explanation"), format=output_format)
