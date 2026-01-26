"""This file contains all the necessary tools to manipulate annotated parts."""

from dataclasses import dataclass
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset

from cabrnet.core.utils.exceptions import ArgumentError


@dataclass
class Annotation:
    r"""An annotation is a dataclass that contains a single annotation,
    i.e., the fact that a part is located at a given position in a given image
    (or does not appear on the image).

    Specifically, if `observed` is `True`, then the part with index `path_idx`
    appears at location `(x,y)` in the image at position `image_idx` within the dataset.
    Otherwise, it does not appear in this image."""
    image_idx: int  # Index of the image in the current dataset (starting with 0)
    part_idx: int  # Index of the part as defined by `parts.txt`
    x: int
    y: int
    observed: bool  # True if the part is observed in the image
    complete_filename: str | None  # File that contains the image


def parse(
    dataset: Dataset,
    image_description: Path,
    part_annotations: Path,
) -> dict[int, dict[int, Annotation]]:
    r"""Reads the relevant information and computes a dictionary `d`
    such that `d[image_idx][part_idx]` contains the annotation information
    related to image `image_idx` about part `part_idx`.

    Args:
        dataset (Dataset): Dataset that contains the images.
            It is assumed that the filename of the images is accessible
            via `dataset["dataset"].imgs`.
        image_description (Path): Path to a file that contains the description of each image.
            The description is assumed to be a list of lines, each of the form `image_idx filename`.
            The `image_idx` here is a fresh id; it is unrelated to the actual position in the dataset
            (some elements of this file will definitely not appear in the dataset due to the partitioning train/test).
        part_annotations (Path): Path to a file that contains the annotation for each image.
            The annotation is assumed to be a list of lines, each of the form `image_idx part_idx x y observed`
            where `image_idx` refers to the index from the `image_description` file.

    Returns:
        Dictionary that contains all the annotations.
    """
    filename_to_image_idx = {}
    df = pd.read_csv(image_description, sep=" ", header=None)
    for idx in df.index:
        image_idx = df[0][idx]
        filename = df[1][idx]
        filename_to_image_idx[filename] = image_idx

    image_idx_to_dataset_idx: dict[int, tuple[int, str]] = {}
    # image_idx_to_dataset_idx[image_idx] = (dataset_idx, filename)
    for dataset_idx, (complete_name, _) in enumerate(dataset["dataset"].imgs):
        found = False
        for suffix, image_idx in filename_to_image_idx.items():
            if complete_name.endswith(suffix):
                image_idx_to_dataset_idx[image_idx] = (dataset_idx, complete_name)
                found = True
                break
        if not found:
            raise ArgumentError(f"Unknown file with name {complete_name}")

    result = {}
    df = pd.read_csv(part_annotations, sep=" ", header=None)
    for idx in df.index:
        image_idx = df[0][idx]
        if image_idx not in image_idx_to_dataset_idx:
            continue  # Only care about the images from the dataset

        dataset_idx, filename = image_idx_to_dataset_idx[image_idx]
        part_idx = df[1][idx]
        x = int(df[2][idx])
        y = int(df[3][idx])
        observed = df[4][idx] == 1

        img_dict = result.get(dataset_idx)
        if img_dict is None:
            img_dict = {}
            result[dataset_idx] = img_dict

        img_dict[part_idx] = Annotation(
            image_idx=dataset_idx,
            part_idx=part_idx,
            x=x,
            y=y,
            observed=observed,
            complete_filename=filename,
        )
    return result
