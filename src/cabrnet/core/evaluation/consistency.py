r"""Consistency is an implementation of the consistency metrics
from 2023 ICCV paper entitled
"Evaluation and Improvement of Interpretability for Self-Explainable Part-Prototype Networks"
by  Qihan Huang et al.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

import cabrnet.core.utils.parts
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.exceptions import ArgumentError
from cabrnet.core.utils.image import safe_open_image
from cabrnet.core.utils.parser import load_config
from cabrnet.core.utils.parts import PartAnnotation
from cabrnet.core.visualization.visualizer import SimilarityVisualizer


def get_config(config_file: Path) -> dict[str, Any] | None:
    r"""Recovers configuration from YML file.

    Args:
        config_file (Path): Path to configuration file.

    Returns:
        Benchmark parameters.
    """
    config = load_config(config_file)
    if "consistency" not in config:
        raise ArgumentError("No configuration for consistency")
    bench_config = config["consistency"]

    absents = []
    for file_parameter in ["image_description", "part_annotations"]:
        if file_parameter not in bench_config:
            absents.append(file_parameter)
            continue
        bench_config[file_parameter] = Path(bench_config[file_parameter])
    for param in ["dataset_name"]:
        if param not in bench_config:
            absents.append(param)

    if absents:
        raise ArgumentError(f"Missing following arguments for consistency check: {absents}")

    for optional_param in ["load_distances", "save_distances"]:
        if optional_param in bench_config:
            bench_config[optional_param] = bool(bench_config[optional_param])
        else:
            bench_config[optional_param] = None

    if "half_size" in bench_config:
        bench_config["half_size"] = float(bench_config["half_size"])

    return bench_config


def image_split_indices(train_test_split: Path) -> tuple[dict[int, int], dict[int, int]]:
    r"""Computes the mapping that associates the indices among the train/test images
    with the complete images.  For instance, the `i`th train image is
    the `j`th image among the complete image set where
    `j = image_split_indices(...)[0][i]`.
    Input file is supposed to be lines of pairs
      `i j`
    where `i` is the index of the image in the complete dataset in increasing order
    and `j` is `1` if this image belongs to the test set.

    Args:
        train_test_split (Path): File that contains the partitioning train / test.

    Returns:
        Mapping from train images to the corresponding index from the complete set.
        Mapping from test images to the corresponding index from the complete set.
    """
    train, test = {}, {}
    img_idx, train_idx, test_idx = 0, 0, 0  # first one starts at 1

    df = pd.read_csv(train_test_split, sep=" ", header=None)
    for idx in df.index:
        img_idx += 1
        if df[1][idx] == 1:
            train_idx += 1
            train[train_idx] = img_idx
        else:
            test_idx += 1
            test[test_idx] = img_idx

    return train, test


def compute_protos_of_class(model: CaBRNet) -> dict[int, list[int]]:
    r"""Computes a mapping from class to prototypes of the class.

    Args:
        model (CaBRNet): Model.

    Returns:
        Mapping that associates each class with a list of its prototypes.
    """
    num_classes = model.classifier.num_classes
    protos_of_class = {k: [] for k in range(num_classes)}
    prototype_class_mapping = model.prototype_class_mapping
    for p in range(model.num_prototypes):
        for k in range(num_classes):
            if prototype_class_mapping[p][k]:
                protos_of_class[k].append(p)
    return protos_of_class


def _save_distances(distances: dict[int, dict[int, list[dict]]], path: Path) -> None:
    df = {}
    first = True
    for _, distance_proto in distances.items():
        for _, distance_proto_part in distance_proto.items():
            for distance in distance_proto_part:
                if first:
                    for param in distance_proto_part[0].keys():
                        df[param] = []
                    first = False
                for param in distance.keys():
                    df[param].append(distance[param])

    df = pd.DataFrame(df)
    df.to_csv(path)


def _add_distance(distances: dict[int, dict[int, list[dict]]], ud: dict) -> None:
    proto_distances = distances.get(ud["prototype_idx"], {})
    if not proto_distances:  # This is a new entry.  It needs to be saved.
        distances[ud["prototype_idx"]] = proto_distances

    proto_part_distances = proto_distances.get(ud["part_idx"], [])
    if not proto_part_distances:  # This is a new entry.  It needs to be saved.
        proto_distances[ud["part_idx"]] = proto_part_distances

    proto_part_distances.append(ud)


def _load_distances(path: Path) -> dict[int, dict[int, list[dict]]]:
    result = {}

    df = pd.read_csv(path)
    for idx in df.index:
        ud = {}
        for param in df.keys()[1:]:
            ud[param] = df[param][idx]
        _add_distance(result, ud)

    return result


def compute_distances(
    model: CaBRNet,
    dataset: Dataset,
    preprocess: Callable,
    visualizer: SimilarityVisualizer,
    annotations: dict[int, dict[int, PartAnnotation]],
    protos_of_class: dict[int, list[int]],
    verbose: bool,
    device: str | torch.device,
) -> dict[int, dict[int, list[dict]]]:
    r"""Computes the distances between the 'centre' of each prototype on the images
    and the centre of each part as provided by the annotations.

    Args:
        model (CaBRNet): Model being evaluated.
        dataset (Dataset): Dataset (test set) used for evaluation.
        preprocess (Callable): Preprocessing method to apply on images.
        visualizer (SimilarityVisualizer): Visualizer used to determine where the prototype is recognized in the image.
        annotations (dict[int, dict[int, PartAnnotation]]: PartAnnotations in the form
            `img_idx -> part_idx -> PartAnnotation`.
        protos_of_class (dict[int, list[int]]): List of prototypes relevant to each class.
        verbose (bool): If true, prints progress of evaluation.
        device (str | torch.device): Device on which the computation is to be performed.

    Returns:
        Dictionary `distances` such that `distances[proto_idx][part_idx]` are the distances
        between the centre of prototype `proto_idx` in the images for which it is relevant
        and the centre of part `part_idx` in these images.
    """
    distances = {}

    data_iter = tqdm(range(len(dataset)), desc="Consistency checking", disable=not verbose)
    for image_idx in data_iter:
        image_filename = dataset.imgs[image_idx][0]
        k = dataset.imgs[image_idx][1]  # class of image
        with safe_open_image(Path(image_filename), preprocess=preprocess) as (img, img_tensor):
            for proto_idx in protos_of_class[k]:
                attrib = visualizer.get_attribution(
                    img=img,
                    proto_idx=proto_idx,
                    device=device,
                    location="max",
                )
                (top_x, top_y) = np.unravel_index(attrib.argmax(), attrib.shape)
                for _, annot in annotations[image_idx].items():
                    if not annot.observed:
                        continue
                    part_idx = annot.part_idx
                    ud = {
                        "prototype_idx": proto_idx,
                        "part_idx": part_idx,
                        "distance_x": abs(top_x - annot.x),
                        "distance_y": abs(top_y - annot.y),
                        "image_width": img.width,
                        "image_height": img.height,
                        "top_x": top_x,
                        "top_y": top_y,
                        "annot_x": annot.x,
                        "annot_y": annot.y,
                    }
                    _add_distance(distances, ud)

    return distances


def execute(
    model: CaBRNet,
    visualization_config: Path | dict[str, Any],
    dataset_config: Path | dict[str, Any],
    dataset_name: str,
    part_parser: str,
    load_distances: bool | None,
    save_distances: bool | None,
    root_dir: Path,
    verbose: bool,
    device: torch.device | str,
    half_size: int | float | None = None,
    threshold: float | None = None,
    **kwargs,
) -> None:
    r"""Compute the consistency score for the model.  See the manual for details on this computation.

    Args:
        model (CaBRNet): Model.
        visualization_config (Path|dict[str, Any]): Visualization configuration used to determine
            the location of the prototypes on the map.
        dataset_config (dict): Dataset configuration.
        dataset_name (str): Name of dataset on which consistency is tested.
        part_parser (str): Name of function used to extract the parts.
        load_distances (bool|None): Whether the distances should be read (if None, tries to read).
        save_distances (bool|None): Whether the distances should be saved
            (if None, saves if the distances have been computed).
        root_dir (Path): Folder where results are to be stored.
        verbose (bool): If True, prints out logging messages.
        device (torch.device | str): Device where computation to be made.
        half_size (int | float | None): Parameter indicating when the location of a prototype
            matches the location of a part.  If None, no computation is performed.
        threshold (int | float | None): Parameter indicating when a prototype is consistent
            with a specific part.  If None, no computation is performed.
    """
    model.to(device)
    model.eval()

    # Get dataset for visualizer
    datasets = DatasetManager.get_datasets(dataset_config)
    projection_dataset = datasets["projection_set"]["dataset"]
    visualizer = SimilarityVisualizer.build_from_config(
        config=visualization_config, model=model, dataset=projection_dataset
    )

    dataloaders = DatasetManager.get_dataloaders(dataset_config)
    dataloader = dataloaders[dataset_name]
    dataset = dataloader.dataset

    annots = getattr(cabrnet.core.utils.parts, part_parser)(dataset, **kwargs)
    protos_of_class = compute_protos_of_class(model)

    # COMPUTES OR LOADS THE DISTANCES
    distances_path = root_dir / "distances.csv"
    distances = None
    if load_distances is None or load_distances:
        # Default behaviour when `load_distances is None` is to try to load it.
        # If that fails, the exception is caught and the distances will be computed instead.
        try:
            if verbose:
                logger.info("Trying to load distances")
            distances = _load_distances(path=distances_path)
            if verbose:
                logger.info("Distances successfully loaded")
        except Exception as e:
            if load_distances is not None:
                raise e
            logger.info(str(e))

    did_compute_distances = False
    if distances is None:
        if verbose:
            logger.info("Computing distances")
        distances = compute_distances(
            model=model,
            dataset=dataset,
            preprocess=transform,
            visualizer=visualizer,
            annotations=annots,
            protos_of_class=protos_of_class,
            verbose=verbose,
            device=device,
        )
        if verbose:
            logger.info("Distances successfully computed")
        did_compute_distances = True

    if (save_distances is None and did_compute_distances) or save_distances:
        if verbose:
            logger.info("Saving distances")
        _save_distances(distances, path=distances_path)
        if verbose:
            logger.info("Distances successfully saved")

    if (half_size is not None) and (threshold is not None):
        # half_size is either a distance (if > 1) or a proportion of image size (if < 1).
        if verbose:
            logger.info("Computing consistency")
        consistencies = {}  # consistency of each prototype
        best_part_of_proto = {}
        num_consistent = 0
        for proto_idx in range(model.num_prototypes):
            consistencies[proto_idx] = 0.0

            for part_idx in distances[proto_idx]:
                distances_pp = distances[proto_idx][part_idx]
                num_in_distances = 0  # number of images where part is near proto
                for d in distances_pp:
                    max_x = half_size if half_size >= 1 else (half_size * d["image_width"])
                    max_y = half_size if half_size >= 1 else (half_size * d["image_height"])
                    if (d["distance_x"] <= max_x) and (d["distance_y"] <= max_y):
                        num_in_distances += 1
                consistency_pp = num_in_distances / len(distances_pp)
                if consistency_pp > consistencies[proto_idx]:
                    consistencies[proto_idx] = consistency_pp
                    best_part_of_proto[proto_idx] = part_idx
            if consistencies[proto_idx] >= threshold:
                num_consistent += 1

        result = {"prototype_idx": [], "consistency": [], "part_idx": []}
        for proto_idx, cons in consistencies.items():
            result["prototype_idx"].append(proto_idx)
            result["consistency"].append(cons)
            result["part_idx"].append(best_part_of_proto[proto_idx])
        result = pd.DataFrame(result)
        result.to_csv(root_dir / "consistencies.csv")

        ave_consistency = sum(consistencies.values()) / len(consistencies)
        logger.info(f"Average consistency: {ave_consistency}")
        logger.info(f"Consistency score: {num_consistent / model.num_prototypes}")
    else:
        if verbose:
            logger.info("Not computing consistency (half-size or threshold not provided)")
