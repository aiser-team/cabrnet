from cabrnet.generic.model import CaBRNet
from cabrnet.utils.data import DatasetManager
from cabrnet.visualization.visualizer import SimilarityVisualizer
from cabrnet.evaluation.debug_explainer import DebugGraph
from cabrnet.visualization.view import heatmap
from cabrnet.utils.parser import load_config
from cabrnet.utils.exceptions import ArgumentError
import torch
import pandas as pd
from PIL import Image
import csv
import numpy as np
from loguru import logger
from tqdm import tqdm
from typing import Any
import os
import pickle


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


def patches_relevance_analysis(
    model: CaBRNet,
    dataset_config: str,
    visualization_config: str,
    output_dir: str,
    device: str,
    verbose: bool,
    patch_info_db: str,
    area_percentage: float,
    sampling_ratio: int = 1,
    debug_mode: bool = False,
    tqdm_position: int = 0,
    **kwargs,
) -> None:
    r"""Performs relevance analysis on test patches.

    Args:
        model (Module): CaBRNet model.
        dataset_config (str): Path to dataset configuration file.
        visualization_config (str): Path to visualization configuration file.
        output_dir (str): Path to output directory.
        device (str): Target device.
        verbose (bool): Verbose mode.
        patch_info_db (str): Path to raw output analysis file.
        area_percentage (float): Area percentage for pointing game mask relevance.
        sampling_ratio (int, optional): Ratio of test images to use during evaluation (e.g. 10 means only
            one image in ten is used). Default: 1.
        debug_mode (bool, optional): Debug mode. When true, saves visualizations. Default: False.
        tqdm_position (int, optional): Position of the progress bar. Default: 0.
    """
    logger.info("Starting test patch relevance benchmark")

    model.eval()
    model.to(device)

    # Create dataloaders and visualizer
    datasets = DatasetManager.get_datasets(dataset_config, sampling_ratio=sampling_ratio, load_segmentation=True)
    visualizer = SimilarityVisualizer.build_from_config(config_file=visualization_config, model=model)

    # Recover preprocessing function
    preprocess = getattr(datasets["test_set"]["dataset"], "transform", None)
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

    for img_idx, ((img, _), (seg, _)) in test_iter:  # type: ignore
        img_tensor = preprocess(img)  # type: ignore
        if img_tensor.dim() != 4:
            # Fix number of dimensions if necessary
            img_tensor = torch.unsqueeze(img_tensor, dim=0)

        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            original_sim_map = model.similarities(img_tensor)[0].cpu().numpy()  # Shape P x H x W

        # Focus on most similar and active prototype
        while True:
            score = np.max(original_sim_map)
            proto_idx, h_max, w_max = np.where(original_sim_map == score)
            proto_idx, h_max, w_max = proto_idx[0], h_max[0], w_max[0]
            if model.prototype_is_active(proto_idx):  # type:ignore
                break
            # Discard inactive prototype and try again
            original_sim_map[proto_idx] = 0

        attribution = visualizer.get_attribution(
            img=img, img_tensor=img_tensor, proto_idx=proto_idx, device=device  # type: ignore
        )

        mask_relevance = pg_mask_relevance(attribution, np.asarray(seg), area_percentage)
        energy_relevance = pg_energy_relevance(attribution, np.asarray(seg))
        stats.append(
            {
                "img_idx": img_idx,
                "proto_idx": proto_idx,
                "mask_relevance": mask_relevance,
                "energy_relevance": energy_relevance,
            }
        )
        if debug_mode:
            # In debug mode, generate visualization graphs
            explanation = DebugGraph(output_dir=output_dir)
            img_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(img_dir, exist_ok=True)

            def square_resize(img: Image.Image) -> Image.Image:
                # Resize image to square
                final_size = min(img.width, img.height)
                return img.resize((final_size, final_size))

            def save_img(src_img: Image.Image, img_path: str, label: str):
                square_resize(src_img).save(img_path)
                explanation.set_test_image(img_path=img_path, label=label, draw_arrows=False)

            def add_pairs(
                top_img: Image.Image,
                top_img_path: str,
                top_img_label: str,
                bot_img: Image.Image,
                bot_img_path: str,
                bot_img_label: str,
            ):
                square_resize(top_img).save(top_img_path)
                square_resize(bot_img).save(bot_img_path)
                explanation.add_pairs(
                    top_img_path=top_img_path,
                    top_img_label=top_img_label,
                    bot_img_path=bot_img_path,
                    bot_img_label=bot_img_label,
                )

            # Save the original image, the segmentation, the heatmap and their intersection
            save_img(src_img=img, img_path=os.path.join(img_dir, f"img_{img_idx}_src.png"), label="Original image")
            save_img(src_img=seg, img_path=os.path.join(img_dir, f"img_{img_idx}_seg.png"), label="Segmentation")

            sorted_attribution = np.sort(np.reshape(attribution, (-1)))
            threshold = sorted_attribution[int(len(sorted_attribution) * (1 - area_percentage))]
            add_pairs(
                top_img=heatmap(img=img, sim_map=attribution),
                top_img_path=os.path.join(img_dir, f"img_{img_idx}_heatmap.png"),
                top_img_label="Heatmap",
                bot_img=heatmap(img=img, sim_map=attribution > threshold),
                bot_img_path=os.path.join(img_dir, f"img_{img_idx}_heatmap_mask.png"),
                bot_img_label=f"Heatmap ({int(area_percentage*100)}% area)",
            )
            seg_array = np.array(seg)[..., 0]
            add_pairs(
                top_img=heatmap(img=img, sim_map=attribution * (seg_array > 0)),
                top_img_path=os.path.join(img_dir, f"img_{img_idx}_intersect.png"),
                top_img_label=f"{int(energy_relevance*100)} % inside",
                bot_img=heatmap(img=img, sim_map=(attribution > threshold) * (seg_array > 0)),
                bot_img_path=os.path.join(img_dir, f"img_{img_idx}_intersect_mask.png"),
                bot_img_label=f"{int(mask_relevance*100)} % inside",
            )
            add_pairs(
                top_img=heatmap(img=img, sim_map=attribution * (seg_array == 0)),
                top_img_path=os.path.join(img_dir, f"img_{img_idx}_wasted.png"),
                top_img_label=f"{100-int(energy_relevance*100)} % outside",
                bot_img=heatmap(img=img, sim_map=(attribution > threshold) * (seg_array == 0)),
                bot_img_path=os.path.join(img_dir, f"img_{img_idx}_wasted_mask.png"),
                bot_img_label=f"{100-int(mask_relevance*100)} % outside",
            )
            explanation.render(os.path.join(output_dir, f"img_{img_idx}_energy_score"))

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
    device: str,
    verbose: bool,
    prototype_info_db: str,
    area_percentage: float,
    projection_info: str = "projection_info.csv",
    tqdm_position: int = 0,
    **kwargs,
) -> None:
    r"""Performs relevance analysis on prototypes.

    Args:
        model (Module): CaBRNet model.
        dataset_config (str): Path to dataset configuration file.
        visualization_config (str): Path to visualization configuration file.
        output_dir (str): Path to output directory.
        device (str): Target device.
        verbose (bool): Verbose mode.
        prototype_info_db (str): Path to raw output analysis file.
        area_percentage (float): Area percentage for pointing game mask relevance.
        projection_info (str, optional): Path to the projection info produced during training.
            Default: projection_info.csv.
        tqdm_position (int, optional): Position of the progress bar. Default: 0.
    """
    logger.info("Starting prototype relevance benchmark")

    model.eval()
    model.to(device)

    # Create dataloaders and visualizer
    datasets = DatasetManager.get_datasets(dataset_config, load_segmentation=True)
    visualizer = SimilarityVisualizer.build_from_config(config_file=visualization_config, model=model)

    # Recover preprocessing function
    preprocess = getattr(datasets["projection_set"]["dataset"], "transform", None)
    projection_set = datasets["projection_set"]["raw_dataset"]
    segmentation_set = datasets["projection_set"]["seg_dataset"]

    # Seek projection information inside directory of prototypes
    prototype_dir = os.path.join(os.path.dirname(dataset_config), "..", "prototypes")
    projection_info_path = os.path.join(prototype_dir, projection_info)
    if not os.path.isfile(projection_info_path):
        raise FileNotFoundError(f"Could not find projection information file {projection_info_path}")
    if projection_info_path.lower().endswith(tuple(["pickle", "pkl"])):
        with open(projection_info_path, "rb") as file:
            projection_db = pickle.load(file)
    else:
        # CSV format
        projection_list = pd.read_csv(projection_info_path).to_dict(orient="records")
        projection_db = {entry["proto_idx"]: entry for entry in projection_list}

    proto_iter = tqdm(
        projection_db,
        desc="Benchmark on prototypes",
        total=len(projection_db),
        leave=False,
        position=tqdm_position,
        disable=not verbose,
    )

    stats = []

    for proto_idx in proto_iter:
        # Recover source image for the prototype
        img = projection_set[projection_db[proto_idx]["img_idx"]][0]
        img_tensor = preprocess(img)  # type: ignore
        attribution = visualizer.get_attribution(img=img, img_tensor=img_tensor, proto_idx=proto_idx, device=device)
        mask_relevance = pg_mask_relevance(
            attribution, np.asarray(segmentation_set[projection_db[proto_idx]["img_idx"]][0]), area_percentage
        )
        energy_relevance = pg_energy_relevance(
            attribution, np.asarray(segmentation_set[projection_db[proto_idx]["img_idx"]][0])
        )
        stats.append(
            {
                "img_idx": projection_db[proto_idx]["img_idx"],
                "proto_idx": proto_idx,
                "mask_relevance": mask_relevance,
                "energy_relevance": energy_relevance,
            }
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


def execute(root_dir: str, **kwargs) -> None:
    r"""Performs relevance analysis on prototypes and test patches.

    Args:
        root_dir (str): Path to root output directory.
    """
    output_dir = os.path.join(root_dir, "relevance_analysis")
    os.makedirs(output_dir, exist_ok=True)
    proto_relevance_analysis(output_dir=output_dir, **kwargs)
    patches_relevance_analysis(output_dir=output_dir, **kwargs)
