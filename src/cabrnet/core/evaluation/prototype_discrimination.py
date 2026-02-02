import numpy
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import torch
from tqdm import tqdm
from typing import Any

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager, DataLoader
from cabrnet.core.utils.exceptions import ArgumentError
from cabrnet.core.utils.parser import load_config


def get_config(config_file: Path) -> dict[str, Any] | None:
    r"""Recovers configuration from YML file.

    Args:
        config_file (Path): Path to configuration file.

    Returns:
        Benchmark parameters.
    """
    bench_config = load_config(config_file).get("prototype_discrimination", None)
    if bench_config is None:
        return bench_config
    missing_params = []
    wrong_types = []

    for mandatory_key, typ in [["methods", list]]:
        param = bench_config.get(mandatory_key)
        if param is None:
            missing_params.append(mandatory_key)
        else:
            if not isinstance(param, typ):
                wrong_types.append((mandatory_key, typ))
    if len(missing_params) > 0:
        raise ArgumentError(f"Missing mandatory parameter(s) {missing_params} in {__file__}")
    if len(wrong_types) > 0:
        raise ArgumentError(f"Wrong type for parameter {wrong_types} in {__file__}")

    return bench_config


def compute_class_of_proto(model: CaBRNet) -> dict[int, list[int]]:
    r"""Computes a mapping from prototype to class.

    Args:
        model (CaBRNet): Model.

    Returns:
        Mapping from prototype to list of classes to which this prototype is relevant.
    """
    num_classes = model.classifier.num_classes
    class_of_proto = {}
    prototype_class_mapping = model.prototype_class_mapping  # Assumes ProtoPNet
    for p in range(model.num_prototypes):
        classes = []
        for cl in range(num_classes):
            if prototype_class_mapping[p][cl]:
                classes.append(cl)
        class_of_proto[p] = classes
    return class_of_proto


def compute_prototypes(model: CaBRNet, relevant_prototypes: dict[str, Any] | None) -> list[int]:
    r"""Computes the list of prototypes for which statistics will be computed.
    The prototypes are assumed to be the list of prototype indices stored in some CSV file.

    Args:
        model (CaBRNet): Model.
        relevant_prototypes (dict[str, Any]|None): Dictionary that indicates which CSV file
        contains the list of prototypes and the CSV key that indicates which column contains the indices;
        all prototypes from model if None.

    Returns: list of prototype indices.
    """
    if relevant_prototypes is None:
        return [p for p in range(model.num_prototypes)]

    filename = Path(relevant_prototypes["file"])
    key = relevant_prototypes["key"]
    with open(filename, "r") as file:
        df = pd.read_csv(file)
        return list({df[key][idx] for idx in df.index})


def gather_statistics(
    model: CaBRNet, dataloader: DataLoader, prototypes: list[int], verbose: bool, device: str | torch.device
) -> tuple[dict, dict]:
    r"""Gathers relevant information for each prototype in the form of two matrices:
    the y_trues (whether a prototype is relevant to an image)
    amd values (how much a prototype is activated on an image).

    Args:
        model (CaBRNet): Model.
        dataloader (DataLoader): Dataloader that contains the list of images.
        prototypes (list): Prototypes for which the statistics are computed.
        verbose (bool): Verbosity.
        device (str): Device on which computation is performed.

    Returns:
         y_trues: matrix such that y_trues[p][img_index] iff prototype p should activate in img_index.
         values: matrix such that values[p][img_index] is the activation of p in img_index.
    """
    num_images = len(dataloader.dataset)
    class_of_proto = compute_class_of_proto(model)

    data_iter = tqdm(dataloader, desc="Model evaluation", total=len(dataloader), disable=not verbose)

    y_trues = {p: numpy.zeros((num_images)) for p in prototypes}
    values = {p: numpy.zeros((num_images)) for p in prototypes}

    with torch.no_grad():
        model.to(device)
        img_idx = -1
        for xs, ys in data_iter:
            xs = xs.to(device)
            sims = model.similarities(xs).flatten(2).amax(dim=2)
            B, P = sims.shape
            for b in range(B):
                img_idx += 1
                cl = ys[b].item()
                for p in range(P):
                    y_trues[p][img_idx] = cl in class_of_proto[p]
                    values[p][img_idx] = sims[b][p].item()

    return y_trues, values


available_methods = {"auroc": roc_auc_score, "auprc": average_precision_score}


def execute(
    model: CaBRNet,
    dataset_config: Path,
    dataset_name: str,
    methods: list,
    root_dir: Path,
    verbose: bool,
    device: str | torch.device,
    relevant_prototypes: dict[str, Any] | None = None,
    **kwargs,
) -> None:
    # Checking method names for early failure
    for method_name in methods:
        if method_name not in available_methods:
            raise ArgumentError(f"Unknown method {method_name}")

    dataloaders = DatasetManager.get_dataloaders(dataset_config)
    dataloader: DataLoader = dataloaders[dataset_name]
    prototypes = compute_prototypes(model, relevant_prototypes)

    y_trues, values = gather_statistics(model, dataloader, prototypes, verbose, device)

    per_method_results = {}
    for method_name in methods:
        method = available_methods[method_name]
        per_method_results[method_name] = [method(y_trues[p], values[p]) for p in prototypes]

    df = pd.DataFrame({"proto_idx": prototypes} | per_method_results)
    with open(root_dir / "stats.csv", "w") as output:
        output.write(df.to_csv())
