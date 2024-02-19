import os
import graphviz
import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import DataLoader
from PIL import Image
from typing import Any, Mapping, Callable
from tqdm import tqdm
from cabrnet.generic.model import CaBRNet
from cabrnet.utils.optimizers import OptimizerManager
from cabrnet.utils.tree import TreeNode, MappingMode
from cabrnet.prototree.decision import SamplingStrategy, ProtoTreeClassifier
from cabrnet.visualisation.visualizer import SimilarityVisualizer
from cabrnet.visualisation.explainer import ExplanationGraph
import copy
from loguru import logger


class ProtoTree(CaBRNet):
    def __init__(self, extractor: nn.Module, classifier: nn.Module, **kwargs):
        """Build a ProtoTree

        Args:
            extractor: Feature extractor
            classifier: Classification based on extracted features
        """
        super(ProtoTree, self).__init__(extractor, classifier, **kwargs)

        # Constant tensor for internal computations
        self.register_buffer("_eye", torch.eye(self.classifier.num_classes))

    def get_extra_state(self) -> Mapping[str, Any] | None:
        """Decision tree architecture to be saved in state_dict.
        This is automatically called by state_dict()"""
        if isinstance(self.classifier, ProtoTreeClassifier):
            return self.classifier.tree.export_arch()
        # When using PRP visualization with Captum, classifier is no longer a ProtoTreeClassifier
        return None

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        """Rebuild decision tree from architecture information
        This is automatically called by load_state_dict()

        Args:
            state: information returned by get_extra_state()
        """
        if isinstance(self.classifier, ProtoTreeClassifier):
            # When using PRP visualization with Captum, classifier is no longer a ProtoTreeClassifier
            self.classifier.tree = TreeNode.build_from_arch(state)

    def load_legacy_state_dict(self, legacy_state: dict) -> None:
        """Load state dictionary from legacy format

        Args:
            legacy_state: Legacy state dictionary

        Raises:
            ValueError when keys or tensor sizes mismatch.
        """
        legacy_keys = legacy_state.keys()
        final_state = copy.deepcopy(legacy_state)
        plib_state = self.state_dict()
        plib_keys = self.state_dict().keys()
        plib_key = "dummy"

        for legacy_key in legacy_keys:
            if legacy_key.startswith("_net"):
                # Feature extractor
                plib_key = legacy_key.replace("_net", "extractor.convnet")
                if plib_key not in plib_keys:
                    raise ValueError(f"No parameter matching {legacy_key}. Check that model architectures are similar.")
            elif legacy_key.startswith("_add_on"):
                # Add-on layers, find matching parameter based on size
                ref_size = legacy_state[legacy_key].size()
                found_match = False
                for plib_key in plib_keys:
                    if "add_on" in plib_key:
                        if plib_state[plib_key].size() == ref_size:
                            logger.info(f"Matching parameters {plib_key} to {legacy_key} based on identical size.")
                            found_match = True
                            break
                if not found_match:
                    raise ValueError(f"No parameter matching {legacy_key}. Check that model architectures are similar.")
            elif legacy_key == "prototype_layer.prototype_vectors":
                plib_key = "classifier.prototypes"
            else:
                if not legacy_key.startswith("_root"):
                    raise ValueError(f"Unexpected parameter {legacy_key}")
                # Iterate on letters in the key (first, remove '.')
                symbols = legacy_key[6:].replace(".", "")
                possible_keys = [key for key in plib_keys if key.startswith("classifier.tree.")]
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
                plib_key = possible_keys[0]
                # Expand dimension of leaf distribution
                final_state[legacy_key] = torch.unsqueeze(final_state[legacy_key], 0)

            # Update state
            if plib_state[plib_key].size() != final_state[legacy_key].size():
                raise ValueError(
                    f"Mismatching parameter size for {legacy_key} and {plib_key}. "
                    f"Expected {plib_state[plib_key].size()}, got {final_state[legacy_key].size()}"
                )
            final_state[plib_key] = final_state.pop(legacy_key)
        self.load_state_dict(final_state, strict=False)

    def analyse_leafs(self, pruning_threshold: float = 0.01) -> None:
        """
        Analyse leaf distributions.
        Args:
            pruning_threshold: Expected pruning threshold.
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

    def loss(self, model_output: Any, label: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Loss function
        Args:
            model_output: Model output, in this case a tuple containing the prediction and the leaf probabilities
            label: Batch labels

        Returns:
            loss tensor and batch accuracy
        """
        ys_pred, info = model_output
        if self.classifier.log_probabilities:
            # Prediction already given as a log value
            batch_loss = torch.nn.functional.nll_loss(ys_pred, label)
        else:
            batch_loss = torch.nn.functional.nll_loss(torch.log(ys_pred), label)
        batch_accuracy = torch.sum(torch.eq(torch.argmax(ys_pred, dim=1), label)).item() / len(label)
        return batch_loss, {"accuracy": batch_accuracy}

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
        """
        Train the model for one epoch.
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

        # Record original leaf distributions
        with torch.no_grad():
            old_dist_params: dict[int, torch.Tensor] = {}
            for leaf in self.classifier.tree.leaves:
                old_dist_params[leaf.node_id] = leaf._relative_distribution.detach().clone()

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
            ys_pred, info = self.forward(xs)
            batch_loss, batch_accuracy = self.loss((ys_pred, info), ys)

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
        optimizer_mngr.zero_grad()

        train_info = {"avg_loss": total_loss / batch_num, "avg_train_accuracy": total_acc / batch_num}
        return train_info

    def epilogue(self, pruning_threshold: float = 0.0) -> None:
        """Function called after training, using information from the epilogue
        field in the training configuration

        Args:
            pruning_threshold: Pruning threshold
        """
        if pruning_threshold <= 0.0:
            logger.warning(f"Leaf pruning disabled (threshold is {pruning_threshold})")
        self.prune(pruning_threshold=pruning_threshold)

    def prune(self, pruning_threshold: float = 0.01) -> None:
        """
        Prune decision tree based on threshold.
        Args:
            pruning_threshold: Pruning threshold
        """
        logger.info(f"Pruning tree. Threshold: {pruning_threshold}")
        num_prototypes_before = self.classifier.tree.num_prototypes
        num_leaves_before = self.classifier.tree.num_leaves
        logger.info(
            f"Tree statistics before pruning: {num_leaves_before} leaves, " f"{num_prototypes_before} prototypes."
        )
        self.classifier.tree.prune_children(threshold=pruning_threshold)
        num_prototypes = self.classifier.tree.num_prototypes
        num_leaves = self.classifier.tree.num_leaves
        logger.info(
            f"Tree statistics after pruning: {num_leaves} leaves "
            f"({(num_leaves_before-num_leaves)/num_leaves_before*100:.1f} % pruned), "
            f"{num_prototypes} prototypes "
            f"({(num_prototypes_before-num_prototypes)/num_prototypes_before*100:.1f} % pruned)."
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
            data_loader: Dataloader containing projection data. WARNING: This dataloader must not be shuffled!
            device: Target device
            verbose: Display progress bar
            progress_bar_position: Position of the progress bar.
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
        class_mapping = self.classifier.tree.get_mapping(mode=MappingMode.CLASS_TO_PROTOTYPE)
        # Sanity check to ensure that each class is associated with at least one prototype
        for class_idx in range(self.classifier.num_classes):
            if class_idx not in class_mapping:
                logger.error(f"Inaccessible class {class_idx}!")

        # Show progress on progress bar if needed
        data_iter = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            leave=False,
            position=progress_bar_position,
            disable=not verbose,
        )

        # Original number of prototypes (before pruning) and prototype length
        max_num_prototypes, proto_dim = self.classifier.prototypes.shape[0:2]

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
                "score": 0,
            }
            for proto_idx in range(max_num_prototypes)
        }
        projection_vectors = self.classifier.prototypes.clone()

        with torch.no_grad():
            for batch_idx, (xs, ys) in data_iter:
                # Map to device and perform inference
                xs = xs.to(device)
                feats = self.extractor(xs)  # Shape N x D x H x W
                H, W = feats.shape[2], feats.shape[3]
                similarities = self.classifier.similarity_layer(feats, self.classifier.prototypes)  # Shape (N, P, H, W)
                max_sim, max_sim_idxs = torch.max(similarities.view(similarities.shape[:2] + (-1,)), dim=2)

                for img_idx, (x, y) in enumerate(zip(xs, ys)):
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
                            projection_info[proto_idx] = {
                                "img_idx": batch_idx * data_loader.batch_size + img_idx,
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
        img_path: str,
        preprocess: Callable,
        visualizer: SimilarityVisualizer,
        prototype_dir_path: str,
        output_dir_path: str,
        device: str,
        exist_ok: bool = False,
        strategy: SamplingStrategy = SamplingStrategy.GREEDY,
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
            strategy: tree sampling strategy
        """
        self.eval()
        img = Image.open(img_path)
        img_tensor = preprocess(img)
        if img_tensor.dim() != 4:
            # Fix number of dimensions if necessary
            img_tensor = torch.unsqueeze(img_tensor, dim=0)

        # Perform inference and get decision leaf
        prediction, tree_info = self.forward(img_tensor, strategy=strategy)
        leaf_id = tree_info["decision_leaf"][0]

        # Compute path to leaf  TODO: This could be done once during model construction, then after pruning
        leaf_path = self.classifier.tree.get_mapping(mode=MappingMode.NODE_PATHS)[leaf_id]

        # Build explanation
        os.makedirs(os.path.join(output_dir_path, "test_patches"), exist_ok=exist_ok)
        explanation = ExplanationGraph(output_dir=output_dir_path)
        explanation.set_test_image(img_path=img_path)
        prototype_mapping = self.classifier.tree.get_mapping(mode=MappingMode.NODE_TO_PROTOTYPE)
        node_mapping = self.classifier.tree.get_mapping(mode=MappingMode.ID_TO_NODE)
        parent_id = leaf_path[0]
        proto_idx = prototype_mapping[leaf_path[0]][0]  # Index of the first prototype
        for node_id in leaf_path[1:]:
            # Recover path to prototype image
            prototype_image_path = os.path.join(prototype_dir_path, f"prototype_{proto_idx}.png")
            score = tree_info[node_id]["conditional_probability"].item()
            if node_id == node_mapping[parent_id].get_submodule(f"{parent_id}_child_nsim").node_id:
                # No similarity
                explanation.add_similarity(
                    prototype_img_path=prototype_image_path,
                    test_patch_img_path=img_path,
                    label=f"Not similar\n (Score: {1-score:.2f})",
                )
            else:
                # Similarity
                patch_image = visualizer.forward(
                    model=self, img=img, img_tensor=img_tensor, proto_idx=proto_idx, device=device
                )
                patch_image_path = os.path.join(output_dir_path, "test_patches", f"proto_similarity_{proto_idx}.png")
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
        explanation.render()

    def explain_global(
        self,
        prototype_dir_path: str,
        output_dir_path: str,
        **kwargs,
    ) -> None:
        """Explain the global decision-making process

        Args:
            prototype_dir_path: path to directory containing prototype visualizations
            output_dir_path: path to output directory containing the explanations
        """

        def build_tree_explanation(node: TreeNode, graph: graphviz.Digraph) -> graphviz.Digraph:
            """Builds tree explanation recursively

            Args:
                node: current node
                graph: current graph

            Returns:
                updated graph
            """
            if node.proto_idxs is None:
                # Leaf
                class_idx = torch.argmax(node.distribution)
                graph.node(name=f"node_{node.node_id}", label=f"Class {class_idx}", fontsize="25", height="0.5")
            else:
                proto_idx = node.proto_idxs[0]
                img_path = os.path.relpath(
                    os.path.join(prototype_dir_path, f"prototype_{proto_idx}.png"), output_dir_path
                )
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
        explanation_graph.render(filename=os.path.join(output_dir_path, "global_explanation"))
