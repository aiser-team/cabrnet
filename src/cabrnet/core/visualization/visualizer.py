from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, get_args

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch import Tensor
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.attribution.augmentors import GaussianNoiseAugmentor
from cabrnet.core.utils.exceptions import check_mandatory_fields
from cabrnet.core.utils.parser import load_config
from cabrnet.core.visualization.gradients import attribute_prototypes
from cabrnet.core.visualization.postprocess import post_process
from cabrnet.core.visualization.prp_utils import get_cabrnet_lrp_composite_model
from cabrnet.core.visualization.upsampling import cubic_upsampling
from cabrnet.core.visualization.view import SUPPORTED_VIEWING_FUNCTIONS
from cabrnet.core.visualization.depictor import ProtoDepictor

# Type alias for attribution methods
AttributionMethod = Literal["saliency", "smoothgrad", "prp", "randgrad", "cubic"]


def compute_attribution(
    model: CaBRNet,
    attribution_method: AttributionMethod,
    img_size: tuple[int, int],
    img_tensor: Tensor,
    proto_idx: int,
    device: str | torch.device,
    **kwargs,
) -> np.ndarray:
    r"""Computes attribution map using the specified method.

    Args:
        model: Target model.
        attribution_method: Attribution method.
        img_size: Original image size (width, height).
        img_tensor: Image tensor.
        proto_idx: Prototype index.
        device: Hardware device.
        **kwargs: Additional parameters passed to the attribution function.
            For smoothgrad: num_samples and noise_ratio are required.

    Returns:
        Attribution map.
    """
    # Handle smoothgrad: transform into saliency with noise augmentor
    if attribution_method == "smoothgrad":
        num_samples = kwargs.pop("num_samples")
        noise_ratio = kwargs.pop("noise_ratio")
        augmentors = [GaussianNoiseAugmentor(num_samples, noise_ratio)]
        attribution_method = "saliency"
    else:
        augmentors = []

    if attribution_method == "cubic":
        return cubic_upsampling(
            model=model,
            img_size=img_size,
            img_tensor=img_tensor,
            proto_idx=proto_idx,
            device=device,
            # scores between 0 and 1 for visualization
            normalize=True,
            **kwargs,
        )

    grads = attribute_prototypes(
        model=model,
        algorithm=attribution_method,
        input_tensor=img_tensor,
        proto_idx=proto_idx,
        device=device,
        augmentors=augmentors,
    )

    return post_process(
        array=grads,
        img_shape=img_size,
        img_tensor=img_tensor,
        resize=True,
        normalize=True,
        **kwargs,
    )


class SimilarityVisualizer(ProtoDepictor):
    r"""Object used to extract patch visualizations from a model.

    Attributes:
        attribution: Function used to compute the relative importance of each pixel w.r.t. a given similarity score.
        attribution_params: Parameters of the attribution function.
        view: Function used to generate an image from the attribution map.
        view_params: Parameters of the viewing function.
        config_file: Path to the configuration file used to create this object.
        model: Target CaBRNet model.
        transform: Preprocessing transform applied to raw input.
    """

    # Supported attribution methods
    SUPPORTED_ATTRIBUTION_METHODS: tuple[str, ...] = get_args(AttributionMethod)

    @property
    def extension(self) -> str:
        return "png"

    def __init__(
        self,
        model: CaBRNet,
        attribution_method: AttributionMethod,
        view_fn: Callable,
        transform: Callable | None,
        attribution_params: dict | None = None,
        view_params: dict | None = None,
        config_file: Path | None = None,
        *args,
        **kwargs,
    ):
        r"""Initializes a patch visualizer.

        Args:
            model: Attach visualizer to a specific model.
            attribution_method: Attribution method name.
            view_fn: Viewing function.
            transform: Preprocessing transform applied to raw input.
            attribution_params: Parameters to attribution function. Default: None.
            view_params: Parameters to viewing function. Default: None.
            config_file: Path to the file used to configure the visualizer. Default: None.
        """
        super().__init__(*args, **kwargs)
        self.attribution_method: AttributionMethod = attribution_method
        self.attribution_params = attribution_params if attribution_params is not None else {}
        self.view = view_fn
        self.view_params = view_params if view_params is not None else {}
        self.config_file = config_file
        self.transform = transform or (lambda x: x)
        self.model = model
        if self.attribution_method == "prp":
            logger.info("Canonizing model for PRP")
            self.model = get_cabrnet_lrp_composite_model(
                model=model,
                set_bias_to_zero=True,
                stability_factor=self.attribution_params.get("stability_factor", 1e-6),
                use_zbeta=True,
            )

    def forward(
        self,
        img: Image.Image,
        proto_idx: int,
        device: str | torch.device,
        location: tuple[int, int] | str | None = "max",
    ) -> Image.Image:
        r"""Generates a visualization of the most similar patch to a given prototype.

        Args:
            img: Original image.
            img_tensor: Image tensor.
            proto_idx: Prototype index.
            device: Hardware device.
            location: Location inside the similarity map.
                Can be given as an explicit location (tuple) or "max" for the location of maximum similarity.
                Default: max.

        Returns:
            Patch visualization.
        """
        sim_map = self.get_attribution(img=img, proto_idx=proto_idx, device=device, location=location)
        assert not np.any(np.isnan(sim_map)), f"sim map contains nan: {sim_map}"
        return self.view(img=img, sim_map=sim_map, **self.view_params)

    def save(
        self,
        raw_input: Image.Image,
        folder: Path,
        filename: str,
        proto_idx: int,
        device: str | torch.device,
        location: tuple[int, int] | str | None = None,
    ) -> Path:
        r"""Generates and saves a visualization.

        Args:
            raw_input: Raw input image (PIL Image).
            folder: Output directory.
            filename: Filename without extension.
            proto_idx: Prototype index.
            device: Hardware device.
            location: Location inside the similarity map.
                Can be given as an explicit location (tuple) or "max" for the location of maximum similarity.
                Default: None.

        Returns:
            Path to the saved file.
        """
        if self.transform is None:
            raise ValueError("Transform must be set to use save() method")

        visualization = self.forward(img=raw_input, proto_idx=proto_idx, device=device, location=location)
        folder.mkdir(parents=True, exist_ok=True)
        output_path = folder / f"{filename}.{self.extension}"
        visualization.save(output_path)
        return output_path

    def get_attribution(
        self,
        img: Image.Image,
        proto_idx: int,
        device: str | torch.device,
        location: tuple[int, int] | str | None = None,
    ) -> np.ndarray:
        r"""Identifies the most similar pixels to a given prototype.

        Args:
            img (Image): Original image.
            img_tensor (tensor): Image tensor.
            proto_idx (int): Prototype index.
            device (str | device): Hardware device.
            location (tuple[int,int], str or None, optional): Location inside the similarity map.
                Can be given as an explicit location (tuple) or "max" for the location of maximum similarity.
                Default: None.

        Returns:
            Importance map.
        """
        attribution_params = self.attribution_params.copy()
        if location is not None:
            # Overwrite parameter in attribution_params
            attribution_params["location"] = location

        return compute_attribution(
            model=self.model,
            attribution_method=self.attribution_method,
            img_size=(img.width, img.height),
            img_tensor=self.transform(img),
            proto_idx=proto_idx,
            location=location,
            device=device,
            **self.attribution_params,
        )

    @staticmethod
    def build_from_config(
        config: Path | dict[str, Any], model: CaBRNet, dataset_config: dict[str, Any]
    ) -> SimilarityVisualizer:
        r"""Builds a SimilarityVisualizer from a configuration file or dictionary.

        Args:
            config: Path to configuration file or dictionary.
            model: Target model.
            dataset_config: Dataset configuration dictionary.

        Returns:
            SimilarityVisualizer.

        Raises:
            ValueError: If transform cannot be extracted from dataset config.
        """
        transform = DatasetManager.get_dataset_transform(config=dataset_config, dataset="projection_set")
        if transform is None:
            raise ValueError("Could not extract transform from dataset config.")

        if isinstance(config, Path):
            logger.info(f"Loading patch visualizer from {config}.")
            config_dict = load_config(config)
        else:
            # Configuration is given in dictionary form
            config_dict = config

        # Sanity checks on mandatory field
        check_mandatory_fields(
            config_dict=config_dict, mandatory_fields=["attribution", "view"], location="visualizer configuration"
        )

        # Visualization function
        attribution_method = config_dict["attribution"]["type"]
        if attribution_method not in SimilarityVisualizer.SUPPORTED_ATTRIBUTION_METHODS:
            raise NotImplementedError(f"Unknown visualization function {config_dict['attribution']['type']}")
        attribution_params = config_dict["attribution"]["params"] if "params" in config_dict["attribution"] else None

        # Viewing function
        if config_dict["view"]["type"] in SUPPORTED_VIEWING_FUNCTIONS:
            view_fn = SUPPORTED_VIEWING_FUNCTIONS[config_dict["view"]["type"]]
        else:
            raise NotImplementedError(f"Unknown viewing function {config_dict['view']['type']}")
        view_params = config_dict["view"]["params"] if "params" in config_dict["view"] else None

        return SimilarityVisualizer(
            model=model,
            attribution_method=attribution_method,
            view_fn=view_fn,
            transform=transform,
            attribution_params=attribution_params,
            view_params=view_params,
            config_file=config if isinstance(config, Path) else None,
        )
