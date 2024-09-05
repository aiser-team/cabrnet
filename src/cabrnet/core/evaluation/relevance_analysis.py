import csv
import os
import pickle
from typing import Any, Callable

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.exceptions import ArgumentError
from cabrnet.core.utils.parser import load_config
from cabrnet.core.utils.save import load_projection_info
from cabrnet.core.visualization.explainer import PointingGameGraph
from cabrnet.core.visualization.visualizer import SimilarityVisualizer


def get_config(config_file: str) -> dict[str, Any] | None:
    r"""Recovers configuration from YML file.

    Args:
        config_file (str): Path to configuration file.

    Returns:
        Benchmark parameters.
    """
    config = load_config(config_file)
    bench_config = config.get("relevance_analysis", None)
    if bench_config is None:
        return bench_config
    # Check mandatory keys
    supported_extensions = ["pickle", "pkl", "csv"]
    for mandatory_key in ["prototype_info_db", "patch_info_db"]:
        if bench_config.get(mandatory_key) is None:
            raise ArgumentError(f"Missing mandatory parameter {mandatory_key} in bench configuration.")
        if not bench_config[mandatory_key].lower().endswith(tuple(supported_extensions)):
            raise ArgumentError(
                f"Unsupported file extension for {bench_config[mandatory_key]}. "
                f"Supported extensions: {tuple(supported_extensions)}."
            )
    return bench_config


def pg_mask_relevance(attribution: np.ndarray, object_seg: np.ndarray, area_percentage: float) -> float:
    r"""Computes the pointing game relevance of an attribution w.r.t the corresponding segmentation.

    Args:
        attribution (Numpy array): Attribution map.
        object_seg (Numpy array): Segmentation of the object.
        area_percentage (float): Percentage of the most relevant pixels to intersect with segmentation mask.

    Returns:
        Relevance of the attribution w.r.t its size.
    """
    object_seg = np.sum(object_seg, axis=-1)
    sorted_attribution = np.sort(np.reshape(attribution, (-1)))
    threshold = sorted_attribution[int(len(sorted_attribution) * (1 - area_percentage))]
    target_area = attribution > threshold
    num_selected_pixels = np.sum(target_area)
    if num_selected_pixels == 0:
        return 0.0
    relevance_tot = np.sum((object_seg > 0) * target_area)
    return relevance_tot / num_selected_pixels


def pg_energy_relevance(attribution: np.ndarray, object_seg: np.ndarray) -> float:
    r"""Computes the energy-based pointing game relevance of an attribution w.r.t the corresponding segmentation.

        See https://arxiv.org/abs/1910.01279.

    Args:
        attribution (Numpy array): Attribution map.
        object_seg (Numpy array): Segmentation of the object.

    Returns:
        Relevance of the attribution w.r.t its energy.
    """
    object_seg = np.sum(object_seg, axis=-1)
    attr_energy = np.sum(attribution)
    if attr_energy == 0:
        return 0.0
    relevance_tot = np.sum((object_seg > 0) * attribution)
    return relevance_tot / attr_energy


def analyze(
    model: CaBRNet,
    img: Image.Image,
    img_id: int | str,
    seg: Image.Image,
    preprocess: Callable,
    visualizer: SimilarityVisualizer,
    device: str | torch.device,
    area_percentage: float = 0.1,
    num_prototypes: int = 1,
    prototype_location: tuple[int, int, int] | None = None,
    debug_dir: str | None = None,
    debug_format: str = "pdf",
    prototype_dir: str = "",
    **kwargs,
) -> list[dict[str, Any]]:
    r"""Performs pointing game relevance analysis on a single image.

    Args:
        model (Module): CaBRNet model, assumed to be in eval mode and already mapped on the correct device.
        img (Image): Input image.
        img_id (int | str): Image identifier.
        seg (Image): Object segmentation.
        preprocess (Callable): Preprocessing function.
        visualizer (SimilarityVisualizer): Patch visualizer.
        device (str | device): Hardware device.
        area_percentage (float, optional): Area percentage for pointing game mask relevance. Default: 0.1.
        num_prototypes (int, optional): Number of relevant prototypes to analyze. Default: 1.
        prototype_location (tuple, optional): A prototype location, given as (proto_idx, h, w). If not None, focus on
            this location rather than the closest prototypes. Default: None.
        debug_dir (str, optional): Path to debug directory. If given, enables debug mode for visualizing image analysis.
            Default: None.
        debug_format (str, optional): Debug image format. Default: pdf.
        prototype_dir (str, optional): Path to directory containing prototype visualization (required in debug mode).
            Default: ".".

    Returns:
        List of statistics.
    """
    stats = []
    img_tensor = preprocess(img)
    if img_tensor.dim() != 4:
        # Fix number of dimensions if necessary
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # Perform a single inference to capture original similarity maps
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        original_sim_map = model.similarities(img_tensor)[0].cpu().numpy()  # Shape P x H x W

    # Overrides num_prototypes if necessary
    if prototype_location is not None:
        proto_idx = prototype_location[0]
        if not model.prototype_is_active(proto_idx):
            # Ignore inactive prototypes
            return []
        most_relevant_prototypes = [proto_idx]
    else:
        # Perform dummy explanation to capture most relevant prototypes
        most_relevant_prototypes = model.explain(
            img=img,
            preprocess=preprocess,
            visualizer=visualizer,
            prototype_dir="",
            output_dir="",
            device=device,
            disable_rendering=True,
        )
        # Remove dissimilar prototypes and take a subset based on num_prototypes
        most_relevant_prototypes = [proto_idx for (proto_idx, _, similar) in most_relevant_prototypes if similar]
        most_relevant_prototypes = most_relevant_prototypes[:num_prototypes]

    # In debug mode, generate visualization graphs
    debug_mode = debug_dir is not None
    debug_dir = debug_dir or ""
    debug_graph = PointingGameGraph(output_dir=debug_dir)
    if debug_mode:
        os.makedirs(debug_dir, exist_ok=True)

    for proto_idx in most_relevant_prototypes:
        # Compute attribution map
        attribution = visualizer.get_attribution(
            img=img, img_tensor=img_tensor, proto_idx=proto_idx, location="max", device=device
        )

        # Compute pointing game stats
        mask_relevance = pg_mask_relevance(attribution, np.asarray(seg), area_percentage)
        energy_relevance = pg_energy_relevance(attribution, np.asarray(seg))
        stats.append(
            {
                "img_id": img_id,
                "proto_idx": proto_idx,
                "mask_relevance": mask_relevance,
                "energy_relevance": energy_relevance,
            }
        )

        if debug_mode:
            test_patch_img = visualizer.view(img=img, sim_map=attribution, **visualizer.view_params)
            debug_graph.add_block(
                prototype_img_path=os.path.join(prototype_dir, f"prototype_{proto_idx}.png"),
                original_img=img,
                test_patch_img=test_patch_img,
                segmentation=seg,
                attribution=attribution,
                area_percentage=area_percentage,
                img_id=img_id,
                proto_idx=proto_idx,
                sim_score=np.max(original_sim_map[proto_idx]),
                energy_score=energy_relevance,
                mask_score=mask_relevance,
                prototype_mode=(prototype_location is not None),
            )

    if debug_mode:
        debug_graph.render(os.path.join(debug_dir, f"img{img_id}_relevance_analysis"), output_format=debug_format)
    return stats


def patches_relevance_analysis(
    model: CaBRNet,
    dataset_config: str,
    visualization_config: str,
    output_dir: str,
    device: str | torch.device,
    verbose: bool,
    patch_info_db: str,
    sampling_ratio: int = 1,
    tqdm_position: int = 0,
    **kwargs,
) -> None:
    r"""Performs relevance analysis on test patches.

    Args:
        model (Module): CaBRNet model.
        dataset_config (str): Path to dataset configuration file.
        visualization_config (str): Path to visualization configuration file.
        output_dir (str): Path to output directory.
        device (str | device): Hardware device.
        verbose (bool): Verbose mode.
        patch_info_db (str): Path to raw output analysis file.
        sampling_ratio (int, optional): Ratio of test images to use during evaluation (e.g. 10 means only
            one image in ten is used). Default: 1.
        tqdm_position (int, optional): Position of the progress bar. Default: 0.
    """
    logger.info("Starting test patch relevance benchmark")

    model.eval()
    model.to(device)

    # Create dataloaders and visualizer
    datasets = DatasetManager.get_datasets(dataset_config, sampling_ratio=sampling_ratio, load_segmentation=True)
    visualizer = SimilarityVisualizer.build_from_config(config=visualization_config, model=model)

    # Recover preprocessing function
    preprocess = getattr(datasets["test_set"]["dataset"], "transform", ToTensor())
    test_set = datasets["test_set"]["raw_dataset"]
    segmentation_set = datasets["test_set"]["seg_dataset"]

    test_iter = tqdm(
        enumerate(zip(test_set, segmentation_set)),  # type: ignore
        desc="Benchmark on test patches",
        total=len(test_set),  # type: ignore
        leave=False,
        position=tqdm_position,
        disable=not verbose,
    )

    stats = []
    for img_id, ((img, _), (seg, _)) in test_iter:  # type: ignore
        stats += analyze(
            model=model,
            img=img,
            seg=seg,
            preprocess=preprocess,
            visualizer=visualizer,
            device=device,
            img_id=img_id,
            **kwargs,
        )

    output_path = os.path.join(output_dir, patch_info_db)
    if output_path.lower().endswith(("pickle", "pkl")):
        # Save in pickle format
        with open(output_path, "wb") as f:
            pickle.dump(stats, f)
    else:
        # Save as CSV
        with open(output_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=stats[0].keys())
            writer.writeheader()
            writer.writerows(stats)


def proto_relevance_analysis(
    model: CaBRNet,
    dataset_config: str,
    visualization_config: str,
    output_dir: str,
    device: str | torch.device,
    verbose: bool,
    prototype_info_db: str,
    projection_file: str,
    tqdm_position: int = 0,
    **kwargs,
) -> None:
    r"""Performs relevance analysis on prototypes.

    Args:
        model (Module): CaBRNet model.
        dataset_config (str): Path to dataset configuration file.
        visualization_config (str): Path to visualization configuration file.
        output_dir (str): Path to output directory.
        device (str | device): Hardware device.
        verbose (bool): Verbose mode.
        prototype_info_db (str): Path to raw output analysis file.
        projection_file (str): Path to the projection info produced during training.
        tqdm_position (int, optional): Position of the progress bar. Default: 0.
    """
    logger.info("Starting prototype relevance benchmark")

    model.eval()
    model.to(device)

    # Create dataloaders and visualizer
    datasets = DatasetManager.get_datasets(dataset_config, load_segmentation=True)
    visualizer = SimilarityVisualizer.build_from_config(config=visualization_config, model=model)

    # Recover preprocessing function
    preprocess = getattr(datasets["projection_set"]["dataset"], "transform", ToTensor())
    projection_set = datasets["projection_set"]["raw_dataset"]
    segmentation_set = datasets["projection_set"]["seg_dataset"]

    projection_info = load_projection_info(projection_file)

    proto_iter = tqdm(
        projection_info,
        desc="Benchmark on prototypes",
        total=len(projection_info),
        leave=False,
        position=tqdm_position,
        disable=not verbose,
    )

    stats = []

    for proto_idx in proto_iter:
        # Recover source image for the prototype
        img = projection_set[projection_info[proto_idx]["img_idx"]][0]  # type: ignore
        h, w = int(projection_info[proto_idx]["h"]), int(projection_info[proto_idx]["w"])

        seg = segmentation_set[projection_info[proto_idx]["img_idx"]][0]  # type: ignore

        stats += analyze(
            model=model,
            img=img,
            seg=seg,
            preprocess=preprocess,
            visualizer=visualizer,
            device=device,
            img_id=int(projection_info[proto_idx]["img_idx"]),
            prototype_location=(proto_idx, h, w),
            **kwargs,
        )

    output_path = os.path.join(output_dir, prototype_info_db)
    if output_path.lower().endswith(("pickle", "pkl")):
        # Save in pickle format
        with open(output_path, "wb") as f:
            pickle.dump(stats, f)
    else:
        # Save as CSV
        with open(output_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=stats[0].keys())
            writer.writeheader()
            writer.writerows(stats)


def execute(
    model: CaBRNet,
    dataset_config: str,
    visualization_config: str,
    projection_file: str,
    root_dir: str,
    device: str | torch.device,
    verbose: bool,
    prototype_dir: str,
    debug_mode: bool = False,
    **kwargs,
) -> None:
    r"""Performs relevance analysis on prototypes and test patches.

    Args:
        model (Module): CaBRNet model.
        dataset_config (str): Path to dataset configuration file.
        visualization_config (str): Path to visualization configuration file.
        projection_file (str): Path to projection information file.
        root_dir (str): Path to root output directory.
        device (str | device): Hardware device.
        verbose (bool): Verbose mode.
        prototype_dir (str): Path to directory containing a visualization of all prototypes.
        debug_mode (bool, optional): If True, enables debug mode for visualizing image analysis. Default: False.
    """
    output_dir = os.path.join(root_dir, "relevance_analysis")
    os.makedirs(output_dir, exist_ok=True)

    if debug_mode and not os.path.isdir(prototype_dir):
        # Get dataloaders and projection info, then build prototypes
        dataloaders = DatasetManager.get_dataloaders(config=dataset_config)
        projection_info = load_projection_info(projection_file)
        visualizer = SimilarityVisualizer.build_from_config(config=visualization_config, model=model)

        # Avoid generating prototypes if the directory already exists
        model.extract_prototypes(
            dataloader_raw=dataloaders["projection_set_raw"],
            dataloader=dataloaders["projection_set"],
            projection_info=projection_info,
            visualizer=visualizer,
            dir_path=prototype_dir,
            device=device,
            verbose=verbose,
        )

    proto_relevance_analysis(
        model=model,
        dataset_config=dataset_config,
        visualization_config=visualization_config,
        projection_file=projection_file,
        device=device,
        verbose=verbose,
        prototype_dir=prototype_dir,
        debug_dir=os.path.join(output_dir, "debug", "prototype_analysis") if debug_mode else None,
        output_dir=output_dir,
        **kwargs,
    )
    patches_relevance_analysis(
        model=model,
        dataset_config=dataset_config,
        visualization_config=visualization_config,
        projection_file=projection_file,
        device=device,
        verbose=verbose,
        prototype_dir=prototype_dir,
        debug_dir=os.path.join(output_dir, "debug", "testset_analysis") if debug_mode else None,
        output_dir=output_dir,
        **kwargs,
    )
