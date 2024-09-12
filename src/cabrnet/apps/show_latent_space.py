import torch
import numpy as np
from pacmap import PaCMAP, PCA
from argparse import ArgumentParser, Namespace
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from cabrnet.core.utils.exceptions import ArgumentError
from tqdm import tqdm
import os

description = "visualizes the latent space"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    r"""Creates the argument parser for training a CaBRNet model.

    Args:
        parser (ArgumentParser, optional): Parent parser (if any).
            Default: None.

    Returns:
        The parser itself.
    """
    if parser is None:
        parser = ArgumentParser(description)
    parser = CaBRNet.create_parser(parser)
    parser = DatasetManager.create_parser(parser)
    parser.add_argument(
        "-c",
        "--checkpoint-dir",
        type=str,
        required=False,
        metavar="/path/to/checkpoint/dir",
        help="path to a checkpoint directory (alternative to --model-arch, --model-state-dict, --dataset)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        required=True,
        metavar="path/to/file",
        help="path to output file",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        choices=["tSNE", "PaCMAP", "PCA"],
        metavar="algorithm",
        help="algorithm to use for dimensionality reduction method. Choices are tSNE, PaCMAP and PCA",
    )
    return parser


def check_args(args: Namespace) -> Namespace:
    r"""Checks the validity of the arguments and updates the namespace if necessary.

    Args:
        args (Namespace): Parsed arguments.

    Returns:
        Modified argument namespace.
    """
    if args.checkpoint_dir is not None:
        # Fetch all files from directory
        for param, name in zip(
            [args.model_arch, args.model_state_dict, args.dataset],
            ["--model-arch", "--model-state-dict", "--dataset"],
        ):
            if param is not None:
                raise ArgumentError(f"Cannot specify both options {name} and --checkpoint-dir")
        args.model_arch = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_CONFIG)
        args.model_state_dict = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_STATE)
        args.dataset = os.path.join(args.checkpoint_dir, DatasetManager.DEFAULT_DATASET_CONFIG)

    # Check configuration completeness
    for param, name, option in zip(
        [args.model_arch, args.model_state_dict, args.dataset],
        ["model configuration", "state dictionary", "dataset configuration"],
        ["-m", "-s", "-d"],
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} file (option {option}).")

    return args


def draw_latent(
    model: CaBRNet,
    dataloader: DataLoader,
    output_file: str,
    algorithm: str,
    device: str = "cuda:0",
    seed: int = 42,
    verbose: bool = False,
):
    r"""Plots the latent space of the dataset and prototypes in a 2D plot
    using t-SNE, PaCMAP or PCA for dimensionality reduction.

    Args:
        model (CaBRNet): Target CaBRNet model.
        dataloader (DataLoader): Dataloader containing the dataset.
        output_file (str): Path to the output file.
        algorithm (str): Dimensionality reduction algorithm. Options: tSNE, PaCMAP or PCA.
        device (str, optional): Hardware device. Default: "cuda:0".
        seed (int, optional): Random seed. Default: 42.
        verbose (bool, optional): If True, enables verbose mode. Default: False.

    Raises:
        ValueError: If an invalid dimensionality reduction algorithm is provided.
    """
    logger.info("Plotting latent space")
    model.eval()
    model.to(device)

    # Get the features of the dataset
    features = []  # (N*H*W, D)
    labels = []
    with torch.no_grad():
        data_iter = tqdm(
            dataloader,
            desc="Feature extraction",
            total=len(dataloader),
            leave=False,
            disable=not verbose,
        )
        for xs, ys in data_iter:
            xs, ys = xs.to(device), ys.to(device)
            feats = model.extractor(xs)  # (N, D, H, W)
            feats = torch.transpose(feats, 1, 3)  # (N, W, H, D)
            N, W, H, D = feats.shape
            feats = feats.reshape((N * W * H, D)).cpu().numpy()
            features.append(feats)
            # features.append(feats.view(feats.size(0), -1).cpu().numpy())
            ys = ys.reshape(ys.shape + (1,)).repeat(1, H * W).reshape((ys.shape[0] * H * W,))
            labels.append(ys.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Get the features of the prototypes
    proto_features = model.classifier.prototypes.view(model.num_prototypes, -1).detach().cpu().numpy()

    # Combine features and prototypes
    combined_data = np.vstack((features, proto_features))

    # Perform dimensionality reduction on the combined dataset
    match algorithm:
        case "tSNE":
            projection_method = TSNE(n_components=2, random_state=seed)
        case "PaCMAP":
            projection_method = PaCMAP(n_components=2, MN_ratio=0.5, FP_ratio=2.0, random_state=seed)
        case "PCA":
            projection_method = PCA(n_components=2, random_state=seed)
        case _:
            raise ValueError(f"Invalid dimensionality reduction algorithm: {algorithm}")
    embeddings = projection_method.fit_transform(combined_data)

    # Separate the embeddings back into features and prototypes
    feature_embeddings = embeddings[: len(features)]  # type: ignore
    proto_embeddings = embeddings[len(features) :]  # type: ignore

    # Plot the latent space with features and prototypes
    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    norm = Normalize(vmin=labels.min(), vmax=labels.max())
    for label in unique_labels:
        plt.scatter(
            feature_embeddings[labels == label, 0],
            feature_embeddings[labels == label, 1],
            c=labels[labels == label],
            cmap="viridis",
            norm=norm,
            label=f"Label {label}",
        )
    plt.scatter(proto_embeddings[:, 0], proto_embeddings[:, 1], marker="^", c="red", label="Prototypes")
    plt.title("Latent space")
    plt.legend()
    plt.savefig(output_file)


def execute(args: Namespace) -> None:
    r"""Plot extracted features from a dataset inside the latent space of a CaBRNet model,
    using an algorithm for dimensionality reduction.

    Args:
        args (Namespace): Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    # Build model and load state dictionary
    model = CaBRNet.build_from_config(config=args.model_arch, state_dict_path=args.model_state_dict)

    dataloaders = DatasetManager.get_dataloaders(config=args.dataset, sampling_ratio=args.sampling_ratio)
    output_file = args.output_file

    draw_latent(
        model=model,
        dataloader=dataloaders["train_set"],
        output_file=output_file,
        algorithm=args.algorithm,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose,
    )
