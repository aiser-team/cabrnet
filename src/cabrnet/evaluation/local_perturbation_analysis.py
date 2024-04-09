from cabrnet.generic.model import CaBRNet
from cabrnet.utils.data import DatasetManager
from cabrnet.visualization.visualizer import SimilarityVisualizer
from cabrnet.visualization.explainer import ExplanationGraph
from cabrnet.visualization.view import compute_bbox
from cabrnet.utils.parser import load_config
from cabrnet.utils.exceptions import ArgumentError
import torch
from torchvision.transforms import ColorJitter, GaussianBlur
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm
from PIL import Image
from typing import Any
import os
import pickle


def get_config(config_file: str) -> dict[str, Any] | None:
    """Recover configuration for YML file

    Args:
        config_file: path to configuration file

    Returns:
        benchmark parameters
    """
    config = load_config(config_file)
    bench_config = config.get("local_perturbation_analysis", None)
    if bench_config is None:
        return bench_config
    # Check mandatory keys
    for mandatory_key in ["info_db", "distribution_img", "prototype_stats"]:
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
    img_array: np.ndarray, attribution: np.ndarray, num_periods: int = 5, amplitude: float = 7, percentile: float = 0.7
) -> np.ndarray:
    # Get an estimation of the patch bounding box to calibrate the sinus deformation
    x_min, x_max, y_min, y_max = compute_bbox(array=attribution[..., 0], threshold=1.0 - percentile)
    periods = num_periods / max(x_max - x_min, y_max - y_min)

    def shift(val: float) -> float:
        return amplitude * np.sin(2.0 * np.pi * val * periods)

    distortion = img_array.copy()
    for y in range(distortion.shape[0]):
        distortion[y, :, ...] = np.roll(distortion[y, :, ...], int(shift(y)), axis=0)
    for x in range(distortion.shape[1]):
        distortion[:, x, ...] = np.roll(distortion[:, x, ...], int(shift(x)), axis=0)

    return distortion


def _compute_perturbations(
    img: Image.Image,
    img_array: np.ndarray,
    attribution: np.ndarray,
    prefix: str | None = None,
    debug: bool = False,
    brightness_factor: float = 0.3,
    contrast_factor: float = 0.2,
    saturation_factor: float = 0.3,
    hue_factor: float = 0.2,
    gaussian_noise_ksize: int = 21,
    gaussian_noise_sigma: float = 2.0,
    distortion_periods: int = 5,
    distortion_amplitude: float = 7.0,
    **kwargs,
) -> dict[str, dict[str, Any]]:
    """Apply a set of perturbations on an image

    Args:
        img: original image in PIL format
        img_array: original image in Numpy format
        attribution: attribution map
        prefix: output prefix in debug mode
        debug: debug mode
        brightness_factor: brightness factor
        contrast_factor: contrast factor
        saturation_factor: saturation factor
        hue_factor: hue factor
        gaussian_noise_ksize: gaussian noise kernel size
        gaussian_noise_sigma: gaussian noise standard deviation
        distortion_periods: number of periods for the sinus distortion
        distortion_amplitude: amplitude factor of the sinus distortion

    Returns:
        dictionary of perturbed images, where the key corresponds to the name of the perturbation
    """

    def _merge(a: np.ndarray, b: np.ndarray, mask: np.ndarray, name: str):
        mask = mask if a.ndim == b.ndim == 3 else np.squeeze(mask)  # Handle grayscale images
        if debug:
            Image.fromarray(a).save(prefix + name + ".png")
            Image.fromarray(np.round(mask * a).astype(np.uint8)).save(prefix + name + "_pos.png")
            Image.fromarray(np.round((1 - mask) * b).astype(np.uint8)).save(prefix + name + "_neg.png")
        return np.round(mask * a + (1 - mask) * b).astype(np.uint8)

    perturbed_img_arrays = {
        "brightness": {
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
        },
        "contrast": {
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
        },
        "saturation": {
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
        },
        "hue": {
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
            "description": "Hue reduction",
        },
        "blur": {
            "focus": _merge(
                np.array(GaussianBlur(kernel_size=gaussian_noise_ksize, sigma=gaussian_noise_sigma)(img)),
                img_array,
                mask=attribution,
                name="blur_focus",
            ),
            "dual": _merge(
                np.array(GaussianBlur(kernel_size=gaussian_noise_ksize, sigma=gaussian_noise_sigma)(img)),
                img_array,
                mask=1 - attribution,
                name="blur_dual",
            ),
            "description": "Gaussian blur",
        },
        "sin_distortion": {
            "focus": _merge(
                _wave_distortion(img_array, attribution, distortion_periods, distortion_amplitude),
                img_array,
                mask=attribution,
                name="dist_focus",
            ),
            "dual": _merge(
                _wave_distortion(img_array, attribution, distortion_periods, distortion_amplitude),
                img_array,
                mask=1 - attribution,
                name="dist_dual",
            ),
            "description": "Sinus distortion",
        },
    }
    return perturbed_img_arrays


def execute(
    model: CaBRNet,
    dataset_config: str,
    visualization_config: str,
    root_dir: str,
    device: str,
    verbose: bool,
    info_db: str,
    enable_dual_mode: bool = True,
    sampling_ratio: int = 1,
    debug_mode: bool = False,
    progress_bar_position: int = 0,
    **kwargs,
) -> None:
    """Perform local similarity analysis

    Args:
            model: CaBRNet model
            dataset_config: path to dataset configuration file
            visualization_config: path to visualization configuration file
            root_dir: path to root output directory
            device: target device
            verbose: verbose mode
            info_db: output file containing individual prototype information (pickle)
            enable_dual_mode: enable dual perturbations
            sampling_ratio: ratio of test images to use during evaluation (e.g. 10 means only one image in ten is used)
            debug_mode: debug mode, save all perturbations
            progress_bar_position: position of the progress bar.
    """
    logger.info("Starting perturbation benchmark")

    model.eval()
    model.to(device)

    # Create dataloaders and visualizer
    datasets = DatasetManager.get_datasets(dataset_config, sampling_ratio=sampling_ratio)
    visualizer = SimilarityVisualizer.build_from_config(config_file=visualization_config, model=model)

    # Infer possible location for prototype directory
    prototype_dir = os.path.join(os.path.dirname(dataset_config), "..", "prototypes")

    # Recover preprocessing function
    preprocess = getattr(datasets["test_set"]["dataset"], "transform", None)
    dataset = datasets["test_set"]["raw_dataset"]

    test_iter = tqdm(
        enumerate(dataset),  # type: ignore
        desc=f"Benchmark on test set",
        total=len(dataset),  # type: ignore
        leave=False,
        position=progress_bar_position,
        disable=not verbose,
    )

    """ Init statistics. For each image in the training set, and the active prototype most similar to that image, 
    record the original similarity score and the score after each perturbation
    """
    stats = []
    output_dir = os.path.join(root_dir, "perturbation_analysis")
    os.makedirs(output_dir, exist_ok=True)
    if debug_mode:
        for dir_name in ["images", "perturbations"]:
            os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)

    for img_idx, (img, label) in test_iter:
        img_array = np.array(img)
        img_tensor = preprocess(img)

        if debug_mode:
            # In debug mode, save original image
            img_path = os.path.join(output_dir, "images", f"img{img_idx}_original.png")
            img.save(img_path)

        if img_tensor.dim() != 4:
            # Fix number of dimensions if necessary
            img_tensor = torch.unsqueeze(img_tensor, dim=0)

        # Perform a single inference to capture original similarity map
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

        # Compute attribution map and expand to (H x W x 1)
        attribution = visualizer.get_attribution(img=img, img_tensor=img_tensor, proto_idx=proto_idx, device=device)  # type: ignore
        attribution = np.expand_dims(attribution, axis=-1)

        # Compute perturbed images
        perturbed_imgs = _compute_perturbations(
            img=img,
            img_array=img_array,
            attribution=attribution,
            prefix=os.path.join(output_dir, "perturbations", f"{img_idx}_{proto_idx}_"),
            debug=debug_mode,
            **kwargs,  # Pass on perturbation factors
        )

        if debug_mode:
            # In debug mode, generate explanation graphs
            explanation = ExplanationGraph(output_dir=output_dir)
            explanation.set_test_image(img_path=img_path)
            patch_img = visualizer.view(img=img, sim_map=attribution[..., 0], **visualizer.view_params)
            patch_img_path = os.path.join(output_dir, "images", f"img{img_idx}_p{proto_idx}_patch.png")
            patch_img.save(patch_img_path)
            explanation.set_test_image(img_path=patch_img_path)

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
                    # Save image if necessary, add bounding box to better locate perturbed part of the image
                    pert_img_path = os.path.join(
                        output_dir, "images", f"img{img_idx}_p{proto_idx}_{pert_name}_{target}.png"
                    )
                    pert_img.save(pert_img_path)

                    explanation.add_similarity(
                        prototype_img_path=os.path.join(prototype_dir, f"prototype_{proto_idx}.png"),
                        test_patch_img_path=pert_img_path,
                        label=f"{perturbed_imgs[pert_name]['description']} ({target})\n"
                        f"Similarity drop\n (from {score:.2f} to {pert_score:.2f})",
                        font_color="blue" if drop_percentage > 0.1 else "red",
                    )

                logger.debug(
                    f"Similarity between image {img_idx} and prototype {proto_idx} "
                    f"sensitivity to {perturbed_imgs[pert_name]['description']}: "
                    f"Score went from {score} to {pert_score} in {target} (drop: {drop_percentage})"
                )

            # Record drop in similarity
            stats.append(
                {
                    "img_idx": img_idx,
                    "proto_idx": proto_idx,
                    "original_score": score,
                    "perturbation": pert_name,
                    "score_after_perturbation": perturbation_scores[0],
                    "dual_score_after_perturbation": perturbation_scores[1] if enable_dual_mode else None,
                    "description": perturbed_imgs[pert_name]["description"],
                }
            )
        if debug_mode:
            explanation.render(os.path.join(output_dir, f"img{img_idx}_p{proto_idx}_sensitivity"))

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

    show_results(model=model, src_path=os.path.join(output_dir, info_db), output_dir=output_dir, **kwargs)


def show_results(
    model: CaBRNet,
    src_path: str,
    output_dir: str,
    prototype_stats: str,
    distribution_img: str,
    quiet: bool = False,
    **kwargs,
) -> None:
    """Show results of analysis

    Args:
        model: target model
        src_path: path to input file containing statistics per test image
        output_dir: output directory
        prototype_stats: name of output CSV file containing prototype statistics
        distribution_img: name of output distribution graph
        quiet: do not display analysis results
    """
    if src_path.lower().endswith(tuple(["pickle", "pkl"])):
        # Open pickle and convert to pandas dataframe
        df = pd.DataFrame.from_dict(pd.read_pickle(src_path))
    else:
        # CSV format
        df = pd.read_csv(src_path)

    # Recover list of image indexes, prototype indexes and perturbations
    img_indices = df["img_idx"].unique()
    perturbations = df["perturbation"].unique()

    # Statistics on images
    max_drop_per_image = []
    for img_idx in tqdm(img_indices, desc="Computing image statistics", leave=False):
        img_df = df.query(f"img_idx == {img_idx}")
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
    with open(os.path.join(output_dir, prototype_stats), "w") as fout:
        writer = csv.writer(fout)
        writer.writerow(["Number of active prototypes", num_active_prototypes])
        writer.writerow(["Number of analyzed prototypes", num_analyzed_prototypes])
        print(f"Number of prototypes analyzed: {num_analyzed_prototypes} / {num_active_prototypes} active prototypes")
        for perturbation in perturbations:
            pert_df = df.query(f"perturbation == '{perturbation}'")
            avg_drop = (pert_df["original_score"] - pert_df["score_after_perturbation"]) / pert_df["original_score"]
            print(f"Average similarity drop for {pert_df.iloc[0]['description']}: {sum(avg_drop)/len(avg_drop):.2f}")
            writer.writerow([perturbation, sum(avg_drop) / len(avg_drop)])
