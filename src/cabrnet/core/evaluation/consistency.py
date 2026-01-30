r"""Consistency is an implementation of the consistency metrics
from 2023 ICCV paper entitled
"Evaluation and Improvement of Interpretability for Self-Explainable Part-Prototype Networks"
by  Qihan Huang et al.
"""
import pandas as pd
from pathlib import Path
from loguru import logger
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm
from typing import Any, Callable

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.exceptions import ArgumentError
from cabrnet.core.utils.image import safe_open_image
from cabrnet.core.utils.parser import load_config
from cabrnet.core.utils.parts import Annotation, parse as parse_parts
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
        bench_config["half_size"] = int(bench_config["half_size"])

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


def compute_distances(
    model: CaBRNet,
    dataset: Dataset,
    preprocess: Callable,
    visualizer: SimilarityVisualizer,
    annotations: dict[int, dict[int, Annotation]],
    protos_of_class: dict[int, list[int]],
    verbose: bool,
    device: str | torch.device,
) -> dict[int, dict[int, list[int]]]:
    """Computes the distances between the 'centre' of each prototype on the images
    and the centre of each part as provided by the annotations.

    Args:
        model (CaBRNet): Model being evaluated.
        dataset (Dataset): Dataset (test set) used for evaluation.
        preprocess (Callable): Preprocessing method to apply on images.
        visualizer (SimilarityVisualizer): Visualizer used to determine where the prototype is recognized in the image.
        annotations (dict[int, dict[int, Annotation]]: Annotations in the form `img_idx -> part_idx -> Annotation`.
        protos_of_class (dict[int, list[int]]): List of prototypes relevant to each class.
        verbose (bool): If true, prints progress of evaluation.
        device (str | torch.device): Device on which the computation is to be performed.

    Returns:
        Dictionary `distances` such that `distances[proto_idx][part_idx]` are the distances
        between the centre of prototype `proto_idx` in the images for which it is relevant
        and the centre of part `part_idx` in these images.
    """
    distances = {proto_idx: {} for proto_idx in range(model.num_prototypes)}

    data_iter = tqdm(range(len(dataset)), desc="Consistency checking", disable=not verbose)
    for image_idx in data_iter:
        image_filename = dataset.imgs[image_idx][0]
        k = dataset.imgs[image_idx][1]  # class of image
        with safe_open_image(Path(image_filename), preprocess=preprocess) as (img, img_tensor):
            for proto_idx in protos_of_class[k]:
                attrib = visualizer.get_attribution(
                    img,
                    img_tensor,
                    proto_idx,
                    device=device,
                    location="max",
                )
                (x, y) = np.unravel_index(attrib.argmax(), attrib.shape)
                for _, annot in annotations[image_idx].items():
                    if not annot.observed:
                        continue
                    part_idx = annot.part_idx
                    distances_of_proto = distances[proto_idx]
                    distances_of_proto_part = distances_of_proto.get(part_idx, None)
                    if distances_of_proto_part is None:
                        distances_of_proto_part = []
                        distances_of_proto[part_idx] = distances_of_proto_part
                    distances_of_proto_part.append(max(abs(x - annot.x), abs(y - annot.y)))

    return distances


def execute(
    model: CaBRNet,
    visualization_config: Path,
    dataset_config: dict[str, Any],
    dataset_name: str,
    image_description: Path,
    part_annotations: Path,
    load_distances: bool | None,
    save_distances: bool | None,
    root_dir: Path,
    half_size: int | None,
    verbose: bool,
    device: torch.device | str,
    **kwargs,
) -> None:
    model.to(device)
    visualizer = SimilarityVisualizer.build_from_config(config=visualization_config, model=model)
    dataloaders = DatasetManager.get_dataloaders(dataset_config)
    dataloader = dataloaders[dataset_name]
    dataset = dataloader.dataset
    preprocess = getattr(dataset, "transform", ToTensor())

    annotations = parse_parts(dataset, image_description, part_annotations)
    protos_of_class = compute_protos_of_class(model)

    # data_iter = tqdm(range(len(dataset)), desc="Consistency checking", disable=not verbose)

    # COMPUTES OR LOADS THE DISTANCES
    distances_path = root_dir / "distances.csv"
    distances = None
    if load_distances is None or load_distances:
        # Default behaviour when `load_distances is None` is to try to load it.
        # If that fails, the exception is caught and the distances will be computed instead.
        try:
            if verbose:
                logger.info("Trying to load distances")
            df = pd.read_csv(distances_path)
            distances = {proto_idx: {} for proto_idx in range(model.num_prototypes)}
            for idx in df.index:
                proto_idx = df["proto_idx"][idx]
                part_idx = df["part_idx"][idx]
                distance = df["distance"][idx]
                if part_idx not in distances[proto_idx]:
                    distances[proto_idx][part_idx] = []
                distances[proto_idx][part_idx].append(distance)
            if verbose:
                logger.info("Distances successfully loaded")
        except Exception as e:
            if load_distances is not None:
                raise e

    did_compute_distances = False
    if distances is None:
        if verbose:
            logger.info("Computing distances")
        distances = compute_distances(
            model=model,
            dataset=dataset,
            preprocess=preprocess,
            visualizer=visualizer,
            annotations=annotations,
            protos_of_class=protos_of_class,
            verbose=verbose,
            device=device,
        )
        did_compute_distances = True

    if (save_distances is None and did_compute_distances) or save_distances:
        if verbose:
            logger.info("Saving distances")
        # Saving the distances in a csv file
        result = {
            "proto_idx": [],
            "part_idx": [],
            "distance": [],
        }
        for proto_idx, distance_proto in distances.items():
            for part_idx, distance_proto_part in distance_proto.items():
                for distance in distance_proto_part:
                    result["proto_idx"].append(proto_idx)
                    result["part_idx"].append(part_idx)
                    result["distance"].append(distance)

        result = pd.DataFrame(result)
        result.to_csv(distances_path)

    if half_size:
        if verbose:
            logger.info("Computing consistency")
        consistencies: dict[int, float] = {}  # consistency of each prototype
        for proto_idx in range(model.num_prototypes):
            proto_consistency = 0.0
            for part_idx in distances[proto_idx]:
                distances_pp = distances[proto_idx][part_idx]
                consistency_pp = sum(1 for d in distances_pp if d <= half_size) / len(distances_pp)
                proto_consistency = max(proto_consistency, consistency_pp)
            consistencies[proto_idx] = proto_consistency

        result = {"proto_idx": [], "consistency": []}
        for proto_idx, cons in consistencies.items():
            result["proto_idx"].append(proto_idx)
            result["consistency"].append(cons)
        result = pd.DataFrame(result)
        result.to_csv(root_dir / "consistencies.csv")

        ave_consistency = sum(consistencies.values()) / len(consistencies)
        print(f"Average consistency: {ave_consistency}")
