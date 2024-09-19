import csv
import os
import pickle
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torchvision.transforms import ColorJitter, GaussianBlur, ToTensor
from tqdm import tqdm

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.exceptions import ArgumentError
from cabrnet.core.utils.image import square_resize
from cabrnet.core.utils.parser import load_config
from cabrnet.core.utils.save import load_projection_info
from cabrnet.core.visualization.explainer import PerturbationGraph
from cabrnet.core.visualization.view import compute_bbox, heatmap
from cabrnet.core.visualization.visualizer import SimilarityVisualizer


def get_config(config_file: str) -> dict[str, Any] | None:
    r"""Recovers configuration from YML file.

    Args:
        config_file (str): Path to configuration file.

    Returns:
        Benchmark parameters.
    """
    config = load_config(config_file)
    bench_config = config.get("local_perturbation_analysis", None)
    if bench_config is None:
        return bench_config
    # Check mandatory keys
    for mandatory_key in ["info_db", "distribution_img", "global_stats"]:
        if bench_config.get(mandatory_key) is None:
            raise ArgumentError(f"Missing mandatory parameter {mandatory_key} in bench configuration.")
    supported_extensions = ["pickle", "pkl", "csv"]
    if not bench_config["info_db"].lower().endswith(tuple(supported_extensions)):
        raise ArgumentError(
            f"Unsupported file extension for {bench_config['info_db']}. "
            f"Supported extensions: {tuple(supported_extensions)}."
        )
    return bench_config


def _wave_distortion(
    img_array: np.ndarray,
    attribution: np.ndarray,
    num_periods: int = 5,
    amplitude: float = 7,
    direction: str = "vertical",
    percentile: float = 0.7,
) -> np.ndarray:
    r"""Applies a sinusoid distortion to a region of an image.

    Args:
        img_array (Numpy array): Original image in Numpy format.
        attribution (Numpy array): Attribution map.
        num_periods (int, optional): Number of periods for the sinus distortion. Default: 5.
        amplitude (float, optional): Amplitude factor of the sinus distortion. Default: 7.0.
        direction (str, optional): Direction of the perturbation (either "horizontal", "vertical" or "both").
            Default: "both".
        percentile (float, optional): Hard threshold used to calibrate the sinus deformation. Default: 0.7.

    Returns:
        Deformed image.
    """
    if direction not in ["horizontal", "vertical", "both"]:
        raise ValueError(f"Unsupported direction for sinus distortion: {direction}.")

    # Get an estimation of the patch bounding box to calibrate the sinus deformation
    x_min, x_max, y_min, y_max = compute_bbox(array=attribution[..., 0], threshold=1.0 - percentile)
    periods = num_periods / max(x_max - x_min, y_max - y_min)

    def shift(val: float) -> float:
        return amplitude * np.sin(2.0 * np.pi * val * periods)

    distortion = img_array.copy()
    if direction in ["horizontal", "both"]:
        for y in range(distortion.shape[0]):
            distortion[y, :, ...] = np.roll(distortion[y, :, ...], int(shift(y)), axis=0)
    if direction in ["vertical", "both"]:
        for x in range(distortion.shape[1]):
            distortion[:, x, ...] = np.roll(distortion[:, x, ...], int(shift(x)), axis=0)

    return distortion


def _compute_perturbations(
    img: Image.Image,
    img_array: np.ndarray,
    attribution: np.ndarray,
    perturbations: list[str] | None,
    brightness_factor: float = 0.3,
    contrast_factor: float = 0.2,
    saturation_factor: float = 0.3,
    hue_factor: float = 0.2,
    gaussian_blur_ksize: int = 21,
    gaussian_blur_sigma: float = 2.0,
    distortion_periods: int = 5,
    distortion_amplitude: float = 7.0,
    distortion_direction: str = "both",
) -> dict[str, dict[str, Any]]:
    r"""Applies a set of perturbations on an image.

    Args:
        img (Image): Original image in PIL format.
        img_array (Numpy array): Original image in Numpy format.
        attribution (Numpy array): Attribution map.
        perturbations (list[str], optional): List of perturbations to apply (all perturbation if None). Default: None.
        brightness_factor (float, optional): Brightness factor. Default: 0.3.
        contrast_factor (float, optional): Contrast factor. Default: 0.2.
        saturation_factor (float, optional): Saturation factor. Default: 0.3.
        hue_factor (float, optional): Hue factor. Default: 0.2.
        gaussian_blur_ksize (int, optional): Gaussian blur kernel size. Default: 21.
        gaussian_blur_sigma (float, optional): Gaussian blur standard deviation. Default: 2.0.
        distortion_periods (int, optional): Number of periods for the sinus distortion. Default: 5.
        distortion_amplitude (float, optional): Amplitude factor of the sinus distortion. Default: 7.0.
        distortion_direction (str, optional): Direction of the perturbation (either "horizontal", "vertical" or "both").
            Default: "both".

    Returns:
        Dictionary of perturbed images, where the key corresponds to the name of the perturbation.
    """

    def _merge(a: np.ndarray, b: np.ndarray, mask: np.ndarray, name: str):
        mask = mask if a.ndim == b.ndim == 3 else np.squeeze(mask)  # Handle grayscale images
        return np.round(mask * a + (1 - mask) * b).astype(np.uint8)

    perturbed_img_arrays = {}
    if perturbations is None or "brightness" in perturbations:
        perturbed_img_arrays["brightness"] = {
            "focus": _merge(
                np.array(
                    ColorJitter(
                        brightness=(brightness_factor, brightness_factor),
                        contrast=0,
                        saturation=0,
                        hue=0,
                    )(img)
                ),
                img_array,
                mask=attribution,
                name="brightness_focus",
            ),
            "dual": _merge(
                np.array(
                    ColorJitter(
                        brightness=(brightness_factor, brightness_factor),
                        contrast=0,
                        saturation=0,
                        hue=0,
                    )(img)
                ),
                img_array,
                mask=1 - attribution,  # Perturb image outside the attribution mask
                name="brightness_dual",
            ),
            "description": "Brightness reduction",
        }
    if perturbations is None or "contrast" in perturbations:
        perturbed_img_arrays["contrast"] = {
            "focus": _merge(
                np.array(
                    ColorJitter(brightness=0, contrast=(contrast_factor, contrast_factor), saturation=0, hue=0)(img)
                ),
                img_array,
                mask=attribution,
                name="contrast_focus",
            ),
            "dual": _merge(
                np.array(
                    ColorJitter(brightness=0, contrast=(contrast_factor, contrast_factor), saturation=0, hue=0)(img)
                ),
                img_array,
                mask=1 - attribution,  # Perturb image outside the attribution mask
                name="contrast_dual",
            ),
            "description": "Contrast reduction",
        }
    if perturbations is None or "saturation" in perturbations:
        perturbed_img_arrays["saturation"] = {
            "focus": _merge(
                np.array(
                    ColorJitter(
                        brightness=0,
                        contrast=0,
                        saturation=(saturation_factor, saturation_factor),
                        hue=0,
                    )(img)
                ),
                img_array,
                mask=attribution,
                name="saturation_focus",
            ),
            "dual": _merge(
                np.array(
                    ColorJitter(
                        brightness=0,
                        contrast=0,
                        saturation=(saturation_factor, saturation_factor),
                        hue=0,
                    )(img)
                ),
                img_array,
                mask=1 - attribution,  # Perturb image outside the attribution mask
                name="saturation_dual",
            ),
            "description": "Saturation reduction",
        }
    if perturbations is None or "hue" in perturbations:
        perturbed_img_arrays["hue"] = {
            "focus": _merge(
                np.array(ColorJitter(brightness=0, contrast=0, saturation=0, hue=(hue_factor, hue_factor))(img)),
                img_array,
                mask=attribution,
                name="hue_focus",
            ),
            "dual": _merge(
                np.array(ColorJitter(brightness=0, contrast=0, saturation=0, hue=(hue_factor, hue_factor))(img)),
                img_array,
                mask=1 - attribution,  # Perturb image outside the attribution mask
                name="hue_dual",
            ),
            "description": "Hue shift",
        }
    if perturbations is None or "blur" in perturbations:
        perturbed_img_arrays["blur"] = {
            "focus": _merge(
                np.array(GaussianBlur(kernel_size=gaussian_blur_ksize, sigma=gaussian_blur_sigma)(img)),
                img_array,
                mask=attribution,
                name="blur_focus",
            ),
            "dual": _merge(
                np.array(GaussianBlur(kernel_size=gaussian_blur_ksize, sigma=gaussian_blur_sigma)(img)),
                img_array,
                mask=1 - attribution,
                name="blur_dual",
            ),
            "description": "Gaussian blur",
        }
    if perturbations is None or "sin_distortion" in perturbations:
        perturbed_img_arrays["sin_distortion"] = {
            "focus": _merge(
                _wave_distortion(
                    img_array, attribution, distortion_periods, distortion_amplitude, distortion_direction
                ),
                img_array,
                mask=attribution,
                name="dist_focus",
            ),
            "dual": _merge(
                _wave_distortion(
                    img_array, attribution, distortion_periods, distortion_amplitude, distortion_direction
                ),
                img_array,
                mask=1 - attribution,
                name="dist_dual",
            ),
            "description": "Sinus distortion",
        }
    return perturbed_img_arrays


def analyze(
    model: CaBRNet,
    img: Image.Image,
    img_id: int | str,
    preprocess: Callable,
    visualizer: SimilarityVisualizer,
    device: str | torch.device,
    perturbations: list[str] | None = None,
    num_prototypes: int = 1,
    enable_dual_mode: bool = True,
    debug_dir: str | None = None,
    debug_format: str = "pdf",
    prototype_dir: str = "",
    **kwargs,  # Perturbation parameters
) -> list[dict[str, Any]]:
    r"""Performs local similarity analysis on a single image.

    Args:
        model (Module): CaBRNet model, assumed to be in eval mode and already mapped on the correct device.
        img (Image): Input image.
        img_id (int | str): Image identifier.
        preprocess (Callable): Preprocessing function.
        visualizer (SimilarityVisualizer): Patch visualizer.
        device (str | device): Hardware device.
        perturbations (list[str], optional): List of perturbations to apply (all perturbation if None). Default: None.
        num_prototypes (int, optional): Number of relevant prototypes to analyze. Default: 1.
        enable_dual_mode (bool, optional): Enable dual perturbations. Default: True.
        debug_dir (str, optional): Path to debug directory. If given, enables debug mode for visualizing image analysis.
            Default: None.
        debug_format (str, optional): Debug image format. Default: pdf.
        prototype_dir (str, optional): Path to directory containing prototype visualization (required in debug mode).
            Default: ".".
        **kwargs: Perturbation parameters.

    Returns:
        List of statistics for each perturbation and each selected prototype.
    """
    stats = []
    img_array = np.array(img)
    img_tensor = preprocess(img)
    if img_tensor.dim() != 4:
        # Fix number of dimensions if necessary
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

    debug_mode = debug_dir is not None
    debug_dir = debug_dir or ""

    if debug_mode:
        if not os.path.isdir(prototype_dir):
            raise ValueError(f"Prototype directory {prototype_dir} does not exist.")
        for dir_name in ["images", "perturbations"]:
            os.makedirs(os.path.join(debug_dir, dir_name), exist_ok=True)

    # Perform a single inference to capture original similarity maps
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        original_sim_map = model.similarities(img_tensor)[0].cpu().numpy()  # Shape P x H x W

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

    # Initialize the debug graph
    debug_graph = PerturbationGraph(output_dir=debug_dir)

    for proto_idx in most_relevant_prototypes:
        # Compute attribution map and expand to (H x W x 1)
        attribution = visualizer.get_attribution(
            img=img, img_tensor=img_tensor, proto_idx=proto_idx, location="max", device=device
        )
        attribution_heatmap = heatmap(img=img, sim_map=attribution, overlay=True)
        attribution = np.expand_dims(attribution, axis=-1)

        # Recover value and location of highest similarity for this prototype
        score = np.max(original_sim_map[proto_idx])
        h_max, w_max = np.where(original_sim_map[proto_idx] == score)
        h_max, w_max = int(h_max[0]), int(w_max[0])

        # Compute perturbed images
        perturbed_imgs = _compute_perturbations(
            img=img,
            img_array=img_array,
            attribution=attribution,
            perturbations=perturbations,
            **kwargs,  # Pass on perturbation factors
        )

        # Debug information
        patch_img_path = os.path.join(debug_dir, "images", f"img{img_id}_p{proto_idx}_patch.png")
        patch_heatmap_path = os.path.join(debug_dir, "images", f"img{img_id}_p{proto_idx}_heatmap.png")
        proto_path = os.path.join(debug_dir, "images", f"prototype_p{proto_idx}.png")

        if debug_mode:
            # Save images
            patch_img = visualizer.view(img=img, sim_map=attribution[..., 0], **visualizer.view_params)
            square_resize(patch_img).save(patch_img_path)
            square_resize(attribution_heatmap).save(patch_heatmap_path)
            # Reshape and copy prototype
            square_resize(Image.open(os.path.join(prototype_dir, f"prototype_{proto_idx}.png"))).save(proto_path)

        for pert_name in perturbed_imgs:
            perturbation_scores = []
            targets = ["focus"] if not enable_dual_mode else ["focus", "dual"]

            for target in targets:
                pert_img = Image.fromarray(perturbed_imgs[pert_name][target])

                pert_img_tensor = torch.unsqueeze(preprocess(pert_img), dim=0).to(device)
                with torch.no_grad():
                    pert_score = model.similarities(pert_img_tensor)[0, proto_idx, h_max, w_max].item()

                drop_percentage = (score - pert_score) / score
                perturbation_scores.append(pert_score)

                if debug_mode:
                    # Save perturbed image if necessary
                    square_resize(pert_img).save(
                        os.path.join(debug_dir, "images", f"img{img_id}_p{proto_idx}_{pert_name}_{target}.png")
                    )

                logger.debug(
                    f"Similarity between image {img_id} and prototype {proto_idx} "
                    f"sensitivity to {perturbed_imgs[pert_name]['description']}: "
                    f"Score went from {score} to {pert_score} in {target} (drop: {drop_percentage})"
                )
            if debug_mode:
                debug_graph.add_block(
                    perturbation=perturbed_imgs[pert_name]["description"],
                    prototype_label=str(proto_idx),
                    prototype_img_path=proto_path,
                    test_patch_img_path=patch_img_path,
                    test_patch_heatmap_path=patch_heatmap_path,
                    focus_test_patch_img_path=os.path.join(
                        debug_dir, "images", f"img{img_id}_p{proto_idx}_{pert_name}_focus.png"
                    ),
                    dual_test_patch_img_path=os.path.join(
                        debug_dir, "images", f"img{img_id}_p{proto_idx}_{pert_name}_dual.png"
                    )
                    if enable_dual_mode
                    else None,
                    original_sim_score=score,
                    focus_sim_score=perturbation_scores[0],
                    dual_sim_score=perturbation_scores[1] if enable_dual_mode else 0.0,
                )
            # Record drop in similarity
            stats.append(
                {
                    "img_id": img_id,
                    "proto_idx": proto_idx,
                    "original_score": score,
                    "perturbation": pert_name,
                    "score_after_perturbation": perturbation_scores[0],
                    "dual_score_after_perturbation": perturbation_scores[1] if enable_dual_mode else None,
                    "description": perturbed_imgs[pert_name]["description"],
                }
            )
    if debug_mode:
        debug_graph.render(os.path.join(debug_dir, f"img{img_id}_sensitivity"), output_format=debug_format)
    return stats


def execute(
    model: CaBRNet,
    dataset_config: str,
    visualization_config: str,
    root_dir: str,
    device: str | torch.device,
    verbose: bool,
    info_db: str = "local_perturbation_analysis.csv",
    global_stats: str = "global_stats.csv",
    distribution_img: str = "local_perturbation_analysis.png",
    quiet: bool = False,
    enable_dual_mode: bool = True,
    sampling_ratio: int = 1,
    debug_mode: bool = False,
    prototype_dir: str = "",
    projection_file: str = "",
    tqdm_position: int = 0,
    **kwargs,
) -> None:
    r"""Performs local similarity analysis.

    Args:
        model (Module): CaBRNet model.
        dataset_config (str): Path to dataset configuration file.
        visualization_config (str): Path to visualization configuration file.
        root_dir (str): Path to root output directory.
        device (str | device): Hardware device.
        verbose (bool): Verbose mode.
        info_db (str, optional): Path to CSV file containing raw analysis per test image.
            Default: local_perturbation_analysis.csv.
        global_stats (str, optional): Path to CSV file containing general global stats. Default: global_stats.csv.
        distribution_img (str, optional): Path to output image showing distribution of max similarity drops.
            Default: local_perturbation_analysis.png.
        quiet (bool, optional): If True, does not show the distribution image. Default: False.
        enable_dual_mode (bool, optional): Enable dual perturbations. Default: True.
        sampling_ratio (int, optional): Ratio of test images to use during evaluation (e.g. 10 means only
            one image in ten is used). Default: 1.
        debug_mode (bool, optional): If True, enables debug mode for visualizing image analysis. Default: False.
        prototype_dir (str, optional): Path to directory containing a visualization of all prototypes (used in debug
            mode only). Default: "".
        projection_file (str, optional): Path to projection information file (used in debug mode only). Default: "".
        tqdm_position (int, optional): Position of the progress bar. Default: 0.
    """
    logger.info("Starting perturbation benchmark")

    model.eval()
    model.to(device)

    # Create dataloaders and visualizer
    datasets = DatasetManager.get_datasets(dataset_config, sampling_ratio=sampling_ratio)
    visualizer = SimilarityVisualizer.build_from_config(config=visualization_config, model=model)

    # Recover preprocessing function
    preprocess = getattr(datasets["test_set"]["dataset"], "transform", ToTensor())
    dataset = datasets["test_set"]["raw_dataset"]

    test_iter = tqdm(
        enumerate(dataset),  # type: ignore
        desc="Benchmark on test set",
        total=len(dataset),  # type: ignore
        leave=False,
        position=tqdm_position,
        disable=not verbose,
    )

    """Init statistics. For each image in the training set, and the active prototype most similar to that image,
    record the original similarity score and the score after each perturbation
    """
    output_dir = os.path.join(root_dir, "perturbation_analysis")
    os.makedirs(output_dir, exist_ok=True)
    # No need to create the debug directory if debug_mode is disabled
    debug_dir = os.path.join(output_dir, "debug")

    if debug_mode:
        for dir_name in ["images", "perturbations"]:
            os.makedirs(os.path.join(debug_dir, dir_name), exist_ok=True)

        # Get dataloaders and projection info, then build prototypes
        dataloaders = DatasetManager.get_dataloaders(config=dataset_config)
        projection_info = load_projection_info(projection_file)

        # Avoid generating prototypes if the directory already exists
        if not os.path.isdir(prototype_dir):
            model.extract_prototypes(
                dataloader_raw=dataloaders["projection_set_raw"],
                dataloader=dataloaders["projection_set"],
                projection_info=projection_info,
                visualizer=visualizer,
                dir_path=prototype_dir,
                device=device,
                verbose=verbose,
            )

    stats = []
    for img_id, (img, _) in test_iter:  # type: ignore
        stats += analyze(
            model=model,
            img=img,
            img_id=img_id,
            preprocess=preprocess,
            visualizer=visualizer,
            device=device,
            num_prototypes=1,
            enable_dual_mode=enable_dual_mode,
            debug_dir=debug_dir if debug_mode else None,
            prototype_dir=prototype_dir,
            **kwargs,
        )

    output_path = os.path.join(output_dir, info_db)
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

    show_results(
        model=model,
        src_path=os.path.join(output_dir, info_db),
        output_dir=output_dir,
        global_stats=global_stats,
        distribution_img=distribution_img,
        quiet=quiet,
    )


def show_results(
    model: CaBRNet,
    src_path: str,
    output_dir: str,
    global_stats: str,
    distribution_img: str,
    quiet: bool = False,
) -> None:
    r"""Shows results of analysis.

    Args:
        model (Module): Target model.
        src_path (str): Path to input file containing statistics per test image.
        output_dir (str): Output directory.
        global_stats (str): Name of output CSV file containing global statistics.
        distribution_img (str): Mame of output distribution graph.
        quiet (bool, optional): If True, does not display analysis results. Default: False.
    """
    if src_path.lower().endswith(tuple(["pickle", "pkl"])):
        # Open pickle and convert to pandas dataframe
        df = pd.DataFrame(pd.read_pickle(src_path))
    else:
        # CSV format
        df = pd.DataFrame(pd.read_csv(src_path))

    # Recover list of image indexes, prototype indexes and perturbations
    img_indices = df["img_id"].unique()
    perturbations = df["perturbation"].unique()

    # Statistics on images
    max_drop_per_image = []
    for img_id in tqdm(img_indices, desc="Computing image statistics", leave=False):
        img_df = df.query(f"img_id == {img_id}")
        original_score = img_df.iloc[0]["original_score"]
        # Find perturbation that maximizes the similarity drop, ignoring negative values
        max_drop = max((original_score - min(img_df["score_after_perturbation"])) / original_score, 0)
        max_drop_per_image.append(max_drop)
    plt.figure(figsize=(8, 6), dpi=150)
    plt.hist(x=max_drop_per_image, bins=50, density=True)
    plt.xlabel("Maximum similarity drop (%)")
    plt.ylabel("Distribution")
    plt.title("Distribution of maximum similarity drop across the test set")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, distribution_img))
    if not quiet:
        plt.show()

    # Statistics on prototypes: average drop per type of perturbation
    num_analyzed_prototypes = len(df["proto_idx"].unique())
    num_active_prototypes = len(
        [proto_idx for proto_idx in range(model.num_prototypes) if model.prototype_is_active(proto_idx)]
    )
    with open(os.path.join(output_dir, global_stats), "w") as fout:
        writer = csv.writer(fout)
        writer.writerow(["Number of active prototypes", num_active_prototypes])
        writer.writerow(["Number of analyzed prototypes", num_analyzed_prototypes])
        print(f"Number of prototypes analyzed: {num_analyzed_prototypes} / {num_active_prototypes} active prototypes")
        for perturbation in perturbations:
            pert_df = df.query(f"perturbation == '{perturbation}'")
            avg_drop = (pert_df["original_score"] - pert_df["score_after_perturbation"]) / pert_df["original_score"]
            print(f"Average similarity drop for {pert_df.iloc[0]['description']}: {sum(avg_drop)/len(avg_drop):.2f}")
            writer.writerow([perturbation, sum(avg_drop) / len(avg_drop)])
