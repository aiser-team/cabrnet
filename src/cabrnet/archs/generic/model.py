from __future__ import annotations

import argparse
import importlib
import os.path
import shutil
from typing import Any, Callable

import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from cabrnet.archs.generic.conv_extractor import ConvExtractor, layer_init_functions
from cabrnet.archs.generic.decision import CaBRNetClassifier
from cabrnet.core.utils.exceptions import check_mandatory_fields
from cabrnet.core.utils.optimizers import OptimizerManager
from cabrnet.core.utils.parser import load_config
from cabrnet.core.visualization.visualizer import SimilarityVisualizer


class CaBRNet(nn.Module):
    r"""Top-module of a Case-Based Reasoning Network (CaBRNet).

    Attributes:
        extractor: Model used to extract convolutional features from the input image.
        classifier: Model used to compute the classification, based on similarity scores with a set of prototypes.
    """

    # Regroups common default file names in a single location
    DEFAULT_MODEL_CONFIG: str = "model_arch.yml"
    DEFAULT_MODEL_STATE: str = "model_state.pth"
    DEFAULT_PROJECTION_INFO: str = "projection_info.csv"

    def __init__(
        self,
        extractor: nn.Module,
        classifier: CaBRNetClassifier,
        compatibility_mode: bool = False,
    ):
        r"""Builds a CaBRNet prototype-based model.

        Args:
            extractor (Module): Feature extractor.
            classifier (CaBRNetClassifier): Classification based on extracted features.
            compatibility_mode (bool, optional): Compatibility mode with legacy architectures. \
                When enabled, batch_norm running parameters are not "properly" frozen, ie they are updated during the
                forward-pass even if the backbone parameters should not be modified.
                Default: False.

        """
        super(CaBRNet, self).__init__()
        self.extractor = extractor
        self.classifier = classifier
        self._compatibility_mode = compatibility_mode
        if compatibility_mode:
            logger.warning(
                "Compatibility mode enabled. Note: this mode is deprecated in production as it "
                "reproduces bugs and quirks from legacy codes."
            )

    def forward(self, x: Tensor, **kwargs):
        r"""Computes model output.

        Args:
            x (tensor): Input tensor.

        Returns:
            Model output.
        """
        x = self.extractor(x, **kwargs)
        return self.classifier(x, **kwargs)

    def similarities(self, x: Tensor, **kwargs) -> Tensor:
        r"""Returns similarity scores.

        Args:
            x (tensor): Input tensor.

        Returns:
            Tensor of similarity scores.
        """
        x = self.extractor(x, **kwargs)
        return self.classifier.similarities(x, **kwargs)

    def distances(self, x: Tensor, **kwargs) -> Tensor:
        r"""Returns pairwise distances between each feature vector and each prototype.

        Args:
            x (tensor): Input tensor.

        Returns:
            Tensor of distances.
        """
        x = self.extractor(x, **kwargs)
        return self.classifier.distances(x, **kwargs)

    @property
    def num_prototypes(self) -> int:
        r"""Returns the number of prototypes."""
        return self.classifier.num_prototypes

    def prototype_is_active(self, proto_idx: int) -> bool:
        r"""Is the prototype *proto_idx* active or disabled?

        Args:
            proto_idx (int): Prototype index.
        """
        return self.classifier.prototype_is_active(proto_idx)

    @property
    def prototype_class_mapping(self) -> np.ndarray:
        r"""Mapping between prototypes and classes.

        Returns:
            Binary array of shape (P, C).
        """
        return self.classifier.prototype_class_mapping

    def _load_legacy_state_dict(self, legacy_state: dict[str, Any]) -> None:
        r"""Loads a state dictionary in legacy format.

        Args:
            legacy_state (state dictionary): Legacy state dictionary.
        """
        # Specific to legacy architectures
        raise NotImplementedError

    def register_training_params(self, training_config: dict[str, Any]) -> None:
        r"""Saves additional information from the training configuration directly into the model.

        Args:
            training_config (dictionary): Dictionary containing training configuration.
        """
        if training_config.get("auxiliary_info") is None:
            logger.warning("Empty auxiliary training configuration. Using default values")
            return
        aux_training_params = training_config["auxiliary_info"]

        for attribute_name, attribute_dict in aux_training_params.items():
            if not hasattr(self, attribute_name):
                raise ValueError(f"Unknown attribute name '{attribute_name}' in training configuration.")
            for key, val in attribute_dict.items():
                # Overwrite default loss coefficients
                if key not in getattr(self, attribute_name).keys():
                    raise ValueError(f"Unknown parameter name '{key}' in attribute '{attribute_name}'")
            getattr(self, attribute_name).update(attribute_dict)

    @staticmethod
    def create_parser(
        parser: argparse.ArgumentParser | None = None,
        mandatory_config: bool = False,
        skip_state_dict: bool = False,
    ) -> argparse.ArgumentParser:
        r"""Creates the argument parser for a CaBRNet model.

        Args:
            parser (ArgumentParser, optional): Existing parser (if any). Default: None.
            mandatory_config (bool, optional): When true, make model configuration mandatory. Default: False.
            skip_state_dict (bool, optional): When true, disable option to load external state dict. Default: False.

        Returns:
            The parser itself.
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="Build a CaBRNet model")
        parser.add_argument(
            "-m",
            "--model-arch",
            required=mandatory_config,
            metavar="/path/to/file.yml",
            help="path to the model configuration file",
        )
        if not skip_state_dict:
            parser.add_argument(
                "-s",
                "--model-state-dict",
                required=mandatory_config,
                metavar="/path/to/model/state.pth",
                help="path to the model state dictionary",
            )
        return parser

    @staticmethod
    def build_from_config(
        config: str | dict[str, Any],
        seed: int | None = None,
        compatibility_mode: bool = False,
        state_dict_path: str | None = None,
    ) -> CaBRNet:
        r"""Builds a CaBRNet model from a YAML configuration file.

        Args:
            config (str|dict): Path to configuration file, or configuration dictionary.
            seed (int, optional): Random seed (used only to resynchronise random number generators in
                compatibility tests). Default: None.
            compatibility_mode (bool, optional): Force compatibility mode with legacy architectures. Default: False.
            state_dict_path (str, optional): Path to model state dictionary. Default: None.

        Returns:
            CaBRNet model.
        """
        if not isinstance(config, (str, dict)):
            raise ValueError(f"Unsupported configuration format: {type(config)}")
        if isinstance(config, str):
            config_dict = load_config(config)
        else:
            config_dict = config

        # Sanity checks on mandatory field
        check_mandatory_fields(
            config_dict=config_dict,
            mandatory_fields=["extractor", "classifier", "top_arch"],
            location="model configuration",
        )
        extractor_config = config_dict["extractor"]
        classifier_config = config_dict["classifier"]
        arch_config = config_dict["top_arch"]

        # Compatibility mode may also be enabled in the classifier configuration
        compatibility_mode = compatibility_mode or config_dict["classifier"].get("params", {}).get(
            "compatibility_mode", False
        )

        # Backward compatibility
        if config_dict.get("similarity") is None:
            logger.warning(
                "Missing explicit similarity function in model configuration, falling back to legacy function."
            )
            if config_dict["classifier"]["name"] == "ProtoPNetClassifier":
                config_dict["similarity"] = {"name": "LegacyProtoPNetSimilarity"}
            elif config_dict["classifier"]["name"] == "ProtoTreeClassifier":
                config_dict["similarity"] = {"name": "LegacyProtoTreeSimilarity"}
            else:
                raise ValueError(
                    f"Unknown default similarity function for classifier {config_dict['classifier']['name']}"
                )
        if extractor_config.get("backbone", {}).get("layer"):
            # Single layer extractor: create a single pipeline called "convnet"
            layer_name = extractor_config["backbone"].pop("layer")
            extractor_config["convnet"] = {"source_layer": layer_name}
            if extractor_config.get("add_on"):
                extractor_config["convnet"]["add_on"] = extractor_config.pop("add_on")

        add_on_init_mode = {}
        if compatibility_mode:
            logger.warning("Compatibility mode: postponing add-on layer initialisation")
            # In compatibility mode with legacy models, postpone add-on layers initialisation
            for pipeline_name, pipeline_config in extractor_config.items():
                if pipeline_name != "backbone" and pipeline_config.get("add_on", {}).get("init_mode"):
                    add_on_init_mode[pipeline_name] = pipeline_config["add_on"].pop("init_mode")

        # Build feature extractor. If state_dict_path is provided, then preloading the "initial" weights of the
        # extractor becomes optional (but should be done if possible). This covers two cases:
        #   1) After publication of the trained model, the original weights of the feature extractor can appear
        #    in the configuration (for transparency) but may not be available to the end-user. In this case, the
        #    library will try to load the original weights but will not crash if they are not available.
        #   2) In some rare cases, the content of state_dict_path may not contain the weights of the feature extractor
        #    (for instance if the feature extractor is left untouched during training). In this particular instance,
        #    the original weights of the feature extractor should be loaded.
        extractor = ConvExtractor(
            config_dict["extractor"], seed=seed, ignore_weight_errors=(state_dict_path is not None)
        )

        # Build classifier
        check_mandatory_fields(
            config_dict=classifier_config,
            mandatory_fields=["module", "name", "params"],
            location="classifier configuration",
        )

        # Check coherency between extractor and classifier
        num_features = extractor.output_channels
        if "num_features" not in classifier_config["params"]:
            logger.warning(
                f"num_features not set in classifier configuration. "
                f"Using value {num_features['convnet']} inferred from feature extractor"
            )
            classifier_config["params"]["num_features"] = num_features["convnet"]
        elif classifier_config["params"]["num_features"] != num_features["convnet"]:
            raise ValueError(
                f"Mismatching number of channels between extractor and classifier: "
                f"expected {classifier_config['params']['num_features']} "
                f"but feature extractor outputs {num_features['convnet']} channels"
            )

        # Update compatibility mode if necessary
        if compatibility_mode:
            classifier_config["params"].update({"compatibility_mode": compatibility_mode})

        # Load classifier module
        classifier_module = importlib.import_module(classifier_config["module"])
        classifier = getattr(classifier_module, classifier_config["name"])(
            similarity_config=config_dict["similarity"], **classifier_config["params"]
        )

        # Load top architecture module
        check_mandatory_fields(
            config_dict=arch_config,
            mandatory_fields=["module", "name"],
            location="top architecture configuration",
        )
        top_arch_module = importlib.import_module(arch_config["module"])
        model = getattr(top_arch_module, arch_config["name"])(
            extractor=extractor, classifier=classifier, compatibility_mode=compatibility_mode
        )

        # Apply postponed add-on layer initialisation (compatibility mode only)
        if add_on_init_mode:
            if model.extractor.num_pipelines == 1:
                model.extractor.add_on.apply(layer_init_functions[add_on_init_mode[next(iter(add_on_init_mode))]])
            else:
                for pipeline_name, init_function in add_on_init_mode.items():
                    model.extractor.add_on[pipeline_name].apply(layer_init_functions[init_function])

        if state_dict_path is not None:
            logger.info(f"Loading model state from {state_dict_path}")
            model.load_state_dict(state_dict=torch.load(state_dict_path, map_location="cpu"))

        return model

    def loss(self, model_output: Any, label: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict[str, float]]:
        r"""Computes the loss and statistics over a batch of model outputs.

        Args:
            model_output (Any): Model output.
            label (Tensor): Batch label.

        Returns:
            Loss tensor and batch statistics.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def epilogue(
        self,
        dataloaders: dict[str, DataLoader],
        optimizer_mngr: OptimizerManager,
        output_dir: str,
        device: str | torch.device = "cuda:0",
        verbose: bool = False,
        **kwargs,
    ) -> dict[int, dict[str, int | float]]:
        r"""Function called after training, using information from the epilogue field in the training configuration.

        Args:
            dataloaders (dictionary): Dictionary of dataloaders.
            optimizer_mngr (OptimizerManager): Optimizer manager.
            output_dir (str): Path to output directory.
            device (str | device, optional): Hardware device. Default: cuda:0.
            verbose (bool, optional): Display progress bar. Default: False.

        Returns:
            Projection information.
        """
        return {}

    def evaluate(
        self,
        dataloader: DataLoader,
        device: str | torch.device = "cuda:0",
        tqdm_position: int = 0,
        verbose: bool = False,
        **kwargs,
    ) -> dict[str, float]:
        r"""Evaluates the model.

        Args:
            dataloader (DataLoader): Dataloader containing evaluation data.
            device (str | device, optional): Hardware device. Default: cuda:0.
            tqdm_position (int, optional): Position of the progress bar. Default: 0.
            verbose (bool, optional): Display progress bar. Default: 0.

        Returns:
            Dictionary containing evaluation statistics.
        """
        logger.info("Evaluating classifier")
        self.eval()
        self.to(device)

        # Global stats
        stats = {}
        nb_inputs = 0

        # Show progress on progress bar if needed
        data_iter = tqdm(
            dataloader,
            desc="Model evaluation",
            total=len(dataloader),
            leave=False,
            position=tqdm_position,
            disable=not verbose,
        )
        with torch.no_grad():
            for xs, ys in data_iter:
                nb_inputs += xs.size(0)
                xs, ys = xs.to(device), ys.to(device)

                # Perform inference and compute loss
                ys_pred = self.forward(xs, **kwargs)
                batch_loss, batch_stats = self.loss(ys_pred, ys)
                batch_accuracy = batch_stats["accuracy"]

                # Update all metrics
                if not stats:
                    stats = batch_stats
                for key, value in batch_stats.items():
                    stats[key] += value * xs.size(0)

                # Update progress bar
                postfix_str = f"Batch loss: {batch_loss.item():.3f}, Acc: {batch_accuracy:.3f}"
                data_iter.set_postfix_str(postfix_str)

        stats = {key: value / nb_inputs for key, value in stats.items()}
        return stats

    def train(self, mode: bool = True) -> nn.Module:
        r"""Overwrites the nn.Module train function to freeze elements if necessary.

        Args:
            mode (bool, optional): Train (True) or eval (False). Default: True.

        Returns:
            Model in updated state.
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
        raise NotImplementedError

    def extract_prototypes(
        self,
        dataloader_raw: DataLoader,
        dataloader: DataLoader,
        projection_info: dict[int, dict],
        visualizer: SimilarityVisualizer,
        dir_path: str,
        device: str | torch.device = "cuda:0",
        verbose: bool = False,
        tqdm_position: int = 0,
    ) -> None:
        r"""Shows prototypes based on projection info.

        Args:
            dataloader_raw (DataLoader): Dataloader containing raw projection images (without preprocessing).
            dataloader (DataLoader): Dataloader containing projection tensors (with preprocessing).
            projection_info (dictionary): Projection information (as returned by project method).
            visualizer (SimilarityVisualizer): Similarity visualizer.
            dir_path (str): Destination directory.
            device (str | device, optional): Hardware device. Default: cuda:0.
            verbose (bool, optional): Display progress bar. Default: 0.
            tqdm_position (int, optional): Position of the progress bar. Default: 0.
        """
        logger.info("Extracting prototype visualization")
        # Create destination directory if necessary
        os.makedirs(dir_path, exist_ok=True)
        # Copy visualizer configuration file
        if visualizer.config_file is not None and os.path.isfile(visualizer.config_file):
            try:
                shutil.copyfile(
                    src=visualizer.config_file,
                    dst=os.path.join(dir_path, SimilarityVisualizer.DEFAULT_VISUALIZATION_CONFIG),
                )
            except shutil.SameFileError:
                logger.warning(f"Ignoring file copy from {visualizer.config_file} to itself.")
                pass

        # Show progress on progress bar if needed
        data_iter = tqdm(
            projection_info,
            desc="Prototype extraction",
            total=len(projection_info),
            leave=False,
            position=tqdm_position,
            disable=not verbose,
        )
        for proto_idx in data_iter:
            if not self.classifier.prototype_is_active(proto_idx):
                # Skip pruned prototype
                continue
            # Original image obtained from dataloader without normalization
            img = dataloader_raw.dataset[projection_info[proto_idx]["img_idx"]][0]
            # Preprocessed image tensor
            img_tensor = dataloader.dataset[projection_info[proto_idx]["img_idx"]][0]
            h, w = projection_info[proto_idx]["h"], projection_info[proto_idx]["w"]
            prototype_part = visualizer.forward(
                img=img,
                img_tensor=img_tensor,
                proto_idx=proto_idx,
                device=device,
                location=(h, w),
            )
            img_path = os.path.join(dir_path, f"prototype_{proto_idx}.png")
            prototype_part.save(fp=img_path)

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

        Returns:
            List of most relevant prototypes for the decision, where each entry is in the form
                (<prototype index>, <similarity score>, <similar>)
            and <similar> indicates whether the prototype is considered similar or dissimilar.
        """
        raise NotImplementedError

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
        raise NotImplementedError
