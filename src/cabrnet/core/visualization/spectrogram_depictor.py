from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, get_args

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch import Tensor

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.attribution.augmentors import BrownNoiseAugmentor
from cabrnet.core.utils.audio_datasets import LabeledSpectroDataset, SpectroConfig
from cabrnet.core.utils.exceptions import check_mandatory_fields
from cabrnet.core.utils.parser import load_config
from cabrnet.core.utils.spectrogram import SpectroToImg
from cabrnet.core.visualization.depictor import ProtoDepictor
from cabrnet.core.visualization.gradients import attribute_prototypes
from cabrnet.core.visualization.prp_utils import get_cabrnet_lrp_composite_model
from cabrnet.core.visualization.view import SUPPORTED_VIEWING_FUNCTIONS

# Type alias for attribution methods
AttributionMethod = Literal["saliency", "smoothgrad", "prp", "randgrad"]


class SpectrogramDepictor(ProtoDepictor):
    r"""Depictor for spectrogram visualizations.

    Takes raw complex spectrum as input, uses BrownNoise augmentor,
    and applies SpectroToImg post-transform before visualization.

    Attributes:
        model: Target CaBRNet model.
        attribution_method: Attribution method name.
        view_fn: Viewing function.
        brown_noise: Brown noise augmentor.
        spectro_to_img: Transform to convert spectrogram to image.
        attribution_params: Parameters of the attribution function.
        view_params: Parameters of the viewing function.
        config_file: Path to the configuration file.
    """

    SUPPORTED_ATTRIBUTION_METHODS: tuple[str, ...] = get_args(AttributionMethod)

    @property
    def extension(self) -> str:
        return "png"

    def __init__(
        self,
        model: CaBRNet,
        attribution_method: AttributionMethod,
        view_fn: Callable,
        spectro_config: SpectroConfig,
        spectro_to_img: SpectroToImg,
        attribution_params: dict,
        view_params: dict,
        config_file: Path | None = None,
        *args,
        **kwargs,
    ):
        r"""Initializes a spectrogram depictor.

        Args:
            model: Attach depictor to a specific model.
            attribution_method: Attribution method name.
            view_fn: Viewing function.
            spectro_config: Spectrogram configuration (FFT parameters).
            spectro_to_img: Transform to convert complex STFT to image tensor.
            attribution_params: Parameters to attribution function.
            view_params: Parameters to viewing function.
            config_file: Path to the file used to configure the depictor. Default: None.

        Raises:
            ValueError: If attribution_params or view_params is None.
        """
        if attribution_params is None:
            raise ValueError("attribution_params must be provided.")
        if view_params is None:
            raise ValueError("view_params must be provided.")

        super().__init__(*args, **kwargs)
        self.attribution_method: AttributionMethod = attribution_method
        self.attribution_params = attribution_params
        self.view_fn = view_fn
        self.view_params = view_params
        self.config_file = config_file
        self.spectro_config = spectro_config
        self.spectro_to_img = spectro_to_img

        # Create BrownNoise augmentor from spectro config
        self.brown_noise_augmentor = BrownNoiseAugmentor(
            n_fft=spectro_config.n_fft,
            width=spectro_config.width,
            num_samples=attribution_params.get("num_samples", 16),
            noise_ratio=attribution_params.get("noise_ratio", 0.1),
            bank_size=attribution_params.get("bank_size", 256),
        )

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
        complex_stft: Tensor,
        proto_idx: int,
        device: str | torch.device,
        location: tuple[int, int] | str | None = None,
    ) -> Image.Image:
        r"""Generates a visualization of the most similar region to a given prototype.

        Args:
            complex_stft: Complex STFT tensor of shape [batch, freq, time].
            proto_idx: Prototype index.
            device: Hardware device.
            location: Location inside the similarity map.
                Can be given as an explicit location (tuple) or "max" for the location of maximum similarity.
                Default: None.

        Returns:
            Spectrogram visualization.
        """
        # Get attribution (augmentation and transform handled internally)
        sim_map = self.get_attribution(
            complex_stft=complex_stft,
            proto_idx=proto_idx,
            device=device,
            location=location,
        )

        # Convert complex STFT to spectrogram image for visualization
        spectrogram_img = self.spectro_to_img(complex_stft)

        # Apply viewing function
        return self.view_fn(spectrogram_img=spectrogram_img, sim_map=sim_map, **self.view_params)

    def save(
        self,
        raw_input: Tensor,
        folder: Path,
        filename: str,
        proto_idx: int,
        device: str | torch.device,
        location: tuple[int, int] | str | None = None,
    ) -> Path:
        r"""Generates and saves a visualization.

        Args:
            raw_input: Complex STFT tensor.
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
        visualization = self.forward(
            complex_stft=raw_input,
            proto_idx=proto_idx,
            device=device,
            location=location,
        )
        folder.mkdir(parents=True, exist_ok=True)
        output_path = folder / f"{filename}.{self.extension}"
        visualization.save(output_path)
        return output_path

    def get_attribution(
        self,
        complex_stft: Tensor,
        proto_idx: int,
        device: str | torch.device,
        location: tuple[int, int] | str | None = None,
    ) -> np.ndarray:
        r"""Identifies the most similar region to a given prototype.

        Args:
            complex_stft: Complex STFT tensor.
            proto_idx: Prototype index.
            device: Hardware device.
            location: Location inside the similarity map.
                Can be given as an explicit location (tuple) or "max" for the location of maximum similarity.
                Default: None.

        Returns:
            Importance map.
        """
        attribution_params = self.attribution_params.copy()

        return attribute_prototypes(
            model=self.model,
            algorithm=self.attribution_method,
            input_tensor=complex_stft,
            proto_idx=proto_idx,
            location=location,
            device=device,
            augmentors=[self.brown_noise_augmentor],
            post_augmentation_transform=self.spectro_to_img,
            **attribution_params,
        )

    @staticmethod
    def build_from_config(
        config: Path | dict[str, Any],
        model: CaBRNet,
        dataset: LabeledSpectroDataset,
    ) -> "SpectrogramDepictor":
        r"""Builds a SpectrogramDepictor from a configuration file or dictionary.

        Args:
            config: Path to configuration file or dictionary.
            model: Target model.
            transform: Preprocessing transform (optional, for compatibility).
            dataset: Dataset to extract spectrogram config from. Must be a LabeledSpectroDataset.

        Returns:
            SpectrogramDepictor.

        Raises:
            ValueError: If dataset is not a LabeledSpectroDataset.
        """
        if dataset is None:
            raise ValueError("Dataset must be provided for SpectrogramDepictor.")

        if not isinstance(dataset, LabeledSpectroDataset):
            raise ValueError(f"Dataset must be an instance of LabeledSpectroDataset, got {type(dataset)}")

        if isinstance(config, Path):
            logger.info(f"Loading spectrogram depictor from {config}.")
            config_dict = load_config(config)
        else:
            config_dict = config

        # Sanity checks on mandatory fields (spectrogram now comes from dataset)
        check_mandatory_fields(
            config_dict=config_dict,
            mandatory_fields=["attribution", "view"],
            location="spectrogram depictor configuration",
        )

        # Attribution method
        attribution_method = config_dict["attribution"]["type"]
        if attribution_method not in SpectrogramDepictor.SUPPORTED_ATTRIBUTION_METHODS:
            raise NotImplementedError(f"Unknown attribution method {attribution_method}")
        attribution_params = config_dict["attribution"].get("params")

        # Viewing function
        view_config = config_dict["view"]
        if view_config["type"] in SUPPORTED_VIEWING_FUNCTIONS:
            view_fn = SUPPORTED_VIEWING_FUNCTIONS[view_config["type"]]
        else:
            raise NotImplementedError(f"Unknown viewing function {view_config['type']}")
        view_params = view_config.get("params")

        # Get spectrogram config from dataset
        spectro_config = dataset.spectro_config

        # Build SpectroToImg from config
        spectro_to_img = SpectroToImg(
            sample_rate=spectro_config.sample_rate,
            n_bands=config_dict.get("n_bands", 64),
            f_min=config_dict.get("f_min", 0),
            f_max=config_dict.get("f_max", spectro_config.sample_rate / 2),
            alpha=config_dict.get("alpha", 0.5),
            flip=config_dict.get("flip", False),
        )

        return SpectrogramDepictor(
            model=model,
            attribution_method=attribution_method,
            view_fn=view_fn,
            spectro_config=spectro_config,
            spectro_to_img=spectro_to_img,
            attribution_params=attribution_params,
            view_params=view_params,
            config_file=config if isinstance(config, Path) else None,
        )
