from __future__ import annotations

import argparse
import importlib
import shutil
import os.path
from typing import Any, Callable
import torch
import torch.nn as nn
from loguru import logger
from cabrnet.generic.conv_extractor import ConvExtractor, layer_init_functions
from cabrnet.utils.parser import load_config
from cabrnet.visualisation.visualizer import SimilarityVisualizer
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class ProtoClassifier(nn.Module):
    def __init__(self, extractor: nn.Module, classifier: nn.Module, compatibility_mode: bool = False):
        """Build a generic prototype-based classifier

        Args:
            extractor: Feature extractor
            classifier: Classification based on extracted features
            compatibility_mode: Compatibility mode with legacy architectures. \
                When enabled, batch_norm running parameters are not "properly" frozen, ie they are updated during the
                forward-pass even if the backbone parameters should not be modified.

        """
        super(ProtoClassifier, self).__init__()
        self.extractor = extractor
        self.classifier = classifier
        self._compatibility_mode = compatibility_mode
        if compatibility_mode:
            logger.warning(
                "Compatibility mode enabled. Note: this mode is deprecated in production as it "
                "reproduces bugs and quirks from legacy codes."
            )

    def forward(self, x: Tensor, **kwargs):
        x = self.extractor(x, **kwargs)
        return self.classifier(x, **kwargs)

    def similarities(self, x: Tensor, **kwargs) -> Tensor:
        """
        Return similarity scores
        Args:
            x: input tensor

        Returns:
            tensor of similarity scores
        """
        x = self.extractor(x, **kwargs)
        return self.classifier.similarity_layer(x, self.classifier.prototypes)

    def load_legacy_state_dict(self, legacy_state: dict) -> None:
        """Load state dictionary from legacy format

        Args:
            legacy_state: Legacy state dictionary
        """
        # Specific to legacy architectures
        raise NotImplementedError

    @staticmethod
    def create_parser(
        parser: argparse.ArgumentParser | None = None,
    ) -> argparse.ArgumentParser:
        """Create the argument parser for a ProtoClassifier.
        Args:
            parser: Existing parser (if any)

        Returns:
            The parser itself.
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="Build a ProtoClassifier")
        parser.add_argument(
            "--model-config",
            default="configs/prototree/model.yml",
            metavar="/path/to/file.yml",
            help="Path to the model configuration file",
        )
        parser.add_argument(
            "--model-state-dict",
            required=False,
            metavar="/path/to/model/state.pth",
            help="Path to the model state dictionary",
        )
        return parser

    @staticmethod
    def build_from_config(
        config_file: str,
        seed: int | None = None,
        compatibility_mode: bool = False,
        state_dict_path: str | None = None,
    ) -> ProtoClassifier:
        """
        Builds a ProtoClassifier from a YAML configuration file
        Args:
            config_file: path to configuration file
            seed: random seed (used only to resynchronise random number generators in compatibility tests)
            compatibility_mode: compatibility mode with legacy architectures
            state_dict_path: path to model state dictionary

        Returns:
            ProtoClassifier
        """
        config_dict = load_config(config_file)

        # Sanity checks on mandatory field
        for mandatory_field in ["extractor", "classifier"]:
            if mandatory_field not in config_dict:
                raise ValueError(f"Missing mandatory field {mandatory_field} in configuration")

        add_on_init_mode = None
        if compatibility_mode:
            logger.warning("Compatibility mode: postponing add-on layer initialisation")
            # In compatibility mode with legacy models, postpone add-on layers initialisation
            if "init_mode" in config_dict["extractor"]["add_on"]:
                add_on_init_mode = config_dict["extractor"]["add_on"].pop("init_mode")

        # Build feature extractor
        extractor = ConvExtractor.build_from_dict(config_dict["extractor"], seed=seed)

        # Build classifier
        classifier_config = config_dict["classifier"]
        for mandatory_field in ["module", "name", "params"]:
            if mandatory_field not in classifier_config:
                raise ValueError(f"Missing mandatory field {mandatory_field} in classifier configuration")
        # Check coherency between extractor and classifier
        num_features = extractor.output_channels
        if "num_features" not in classifier_config["params"]:
            logger.warning(
                f"num_features not set in classifier configuration. "
                f"Using value {num_features} inferred from feature extractor"
            )
            classifier_config["params"]["num_features"] = num_features
        elif classifier_config["params"]["num_features"] != num_features:
            raise ValueError(
                f"Mismatching number of channels between extractor and classifier: "
                f"expected {classifier_config['params']['num_features']} "
                f"but feature extractor outputs {num_features} channels"
            )

        # Load classifier module
        classifier_module = importlib.import_module(classifier_config["module"])
        classifier = getattr(classifier_module, classifier_config["name"])(**classifier_config["params"])

        # Load top architecture module if necessary
        if "custom_arch" in config_dict:
            for mandatory_field in ["module", "name"]:
                if mandatory_field not in classifier_config:
                    raise ValueError(f"Missing mandatory field {mandatory_field} in custom architecture configuration")
            arch_config = config_dict["custom_arch"]
            top_arch_module = importlib.import_module(arch_config["module"])
            model = getattr(top_arch_module, arch_config["name"])(
                extractor=extractor, classifier=classifier, compatibility_mode=compatibility_mode
            )
        else:
            model = ProtoClassifier(extractor=extractor, classifier=classifier, compatibility_mode=compatibility_mode)

        # Apply postponed add-on layer initialisation (compatibility mode only)
        if add_on_init_mode is not None:
            model.extractor.add_on.apply(layer_init_functions[add_on_init_mode])

        if state_dict_path is not None:
            logger.info(f"Loading model state from {state_dict_path}")
            model.load_state_dict(state_dict=torch.load(state_dict_path, map_location="cpu"))

        return model

    def loss(self, model_output: Any, label: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Computes the loss and the accuracy over a batch of model outputs
        Args:
            model_output: Model output
            label: Batch label

        Returns:
            loss tensor and batch accuracy
        """
        raise NotImplementedError

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
        """
        Train the model for one epoch.
        Args:
            train_loader: Dataloader containing training data
            optimizer: Learning optimizer
            device: Target device
            progress_bar_position: Position of the progress bar.
            epoch_idx: Epoch index
            verbose: Display progress bar
            max_batches: Max number of batches (early stop for small compatibility tests)

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
            ys_pred, info = self.forward(xs)
            batch_loss, batch_accuracy = self.loss((ys_pred, info), ys)

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

    def epilogue(self, **kwargs) -> None:
        """Function called after training, using information from the epilogue
        field in the training configuration
        """
        pass

    def evaluate(
        self,
        dataloader: DataLoader,
        device: str = "cuda:0",
        progress_bar_position: int = 0,
        verbose: bool = False,
    ) -> dict[str, float]:
        """
        Evaluate the model.
        Args:
            dataloader: Dataloader containing evaluation data
            device: Target device
            progress_bar_position: Position of the progress bar
            verbose: Display progress bar

        Returns:
            dictionary containing evaluation statistics
        """
        logger.info("Evaluating classifier")
        self.eval()
        self.to(device)

        # Training stats
        total_loss = 0.0
        total_acc = 0.0

        # Show progress on progress bar if needed
        data_iter = tqdm(
            dataloader, total=len(dataloader), leave=False, position=progress_bar_position, disable=not verbose
        )
        batch_num = len(dataloader)

        for xs, ys in data_iter:
            xs, ys = xs.to(device), ys.to(device)

            # Perform inference and compute loss
            ys_pred = self.forward(xs)
            batch_loss, batch_accuracy = self.loss(ys_pred, ys)

            # Update global metrics
            total_loss += batch_loss.item()
            total_acc += batch_accuracy

            # Update progress bar
            postfix_str = f"Batch loss: {batch_loss.item():.3f}, Acc: {batch_accuracy:.3f}"
            data_iter.set_postfix_str(postfix_str)  # type: ignore

        return {"avg_loss": total_loss / batch_num, "avg_eval_accuracy": total_acc / batch_num}

    def train(self, mode: bool = True) -> nn.Module:
        """Overwrite train() function to freeze elements if necessary

        :param mode: Train (true) or eval (false)
        """
        self.training = mode
        self.extractor.train(mode)
        self.classifier.train(mode)

        if not self._compatibility_mode:
            # Fix BatchNorm training status
            for name, layer in self.named_modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.train(layer.weight.requires_grad and mode)
        return self

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
        raise NotImplementedError

    def extract_prototypes(
        self,
        dataloader_raw: DataLoader,
        dataloader: DataLoader,
        projection_info: dict[int, dict],
        visualizer: SimilarityVisualizer,
        dir_path: str,
        device: str,
        verbose: bool = False,
        progress_bar_position: int = 0,
    ) -> None:
        """
        Show prototypes based on projection info
        Args:
            dataloader_raw: dataloader containing raw projection images (without preprocessing)
            dataloader: dataloader containing projection tensors (with preprocessing)
            projection_info: projection information (as returned by project method)
            visualizer: similarity visualizer
            dir_path: destination directory
            device: target hardware device
            verbose: display progress bar
            progress_bar_position: position of the progress bar.
        """
        logger.info("Extracting prototype visualization")
        # Create destination directory if necessary
        os.makedirs(dir_path, exist_ok=True)
        # Copy visualizer configuration file
        if os.path.isfile(visualizer.config_file):
            shutil.copyfile(src=visualizer.config_file, dst=os.path.join(dir_path, "visualization.yml"))

        # Show progress on progress bar if needed
        data_iter = tqdm(
            projection_info,
            total=len(projection_info),
            leave=False,
            position=progress_bar_position,
            disable=not verbose,
        )
        for proto_idx in data_iter:
            if projection_info[proto_idx]["img_idx"] == -1:
                # Skip pruned prototype
                continue
            # Original image obtained from dataloader without normalization
            img = dataloader_raw.dataset[projection_info[proto_idx]["img_idx"]][0]
            # Preprocessed image tensor
            img_tensor = dataloader.dataset[projection_info[proto_idx]["img_idx"]][0]
            h, w = projection_info[proto_idx]["h"], projection_info[proto_idx]["w"]
            prototype_part = visualizer.forward(
                model=self, img=img, img_tensor=img_tensor, proto_idx=proto_idx, device=device, location=(h, w)
            )
            img_path = os.path.join(dir_path, f"prototype_{proto_idx}.png")
            prototype_part.save(fp=img_path)

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
        """Explain the decision for a particular image

        Args:
            img_path: path to raw original image
            preprocess: preprocessing function
            visualizer: prototype visualizer
            prototype_dir_path: path to directory containing prototype visualizations
            output_dir_path: path to output directory containing the explanation
            device: target hardware device
            exist_ok: silently overwrite existing explanation if any
        """
        raise NotImplementedError

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
        raise NotImplementedError
