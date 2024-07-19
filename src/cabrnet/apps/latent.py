import torch
import numpy as np
import pacmap
from argparse import ArgumentParser, Namespace
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from cabrnet.utils.exceptions import ArgumentError


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
        "-o",
        "--output-file",
        type=str,
        required=True,
        metavar="path/to/file",
        help="file where the image will be saved",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        required=False,
        metavar="tsne|pacmap",
        help="Dimensionality reduction method (default is 'tsne')",
    )
    return parser


DEFAULT_METHOD = "tsne"


def draw_latent(
    model: CaBRNet, dataloader: DataLoader, output_file: str, method: str = DEFAULT_METHOD, device: str = "cuda:0"
):
    """
    Plots the latent space of the dataset and prototypes in a 2D plot using t-SNE or PaCMAP for dimensionality reduction.

    Args:
        model (CaBRNet): The CaBRNet model.
        dataloader (DataLoader): Dataloader containing the dataset.
        method (str): Dimensionality reduction method. Options: "tsne" (default), "pacmap".
        device (str): Target device. Default: "cuda:0".
        output_file (str): Path to the output file.

    Returns:
        None

    Raises:
        ValueError: If an invalid dimensionality reduction method is provided.
    """
    logger.info("Plotting latent space")
    model.eval()
    model.to(device)

    # Get the features of the dataset
    features = []  # (N*H*W, D)
    labels = []
    with torch.no_grad():
        for xs, ys in dataloader:
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
    if method is None:
        logger.warning(f"Using default method: {DEFAULT_METHOD}")
        method = DEFAULT_METHOD
    if method == "tsne":
        projection_method = TSNE(n_components=2, random_state=0)
    elif method == "pacmap":
        projection_method = pacmap.PaCMAP(n_components=2, MN_ratio=0.5, FP_ratio=2.0)
    else:
        raise ValueError(f"Invalid dimensionality reduction method: {method}")
    embeddings = projection_method.fit_transform(combined_data)

    # Separate the embeddings back into features and prototypes
    feature_embeddings = embeddings[: len(features)]
    proto_embeddings = embeddings[len(features) :]

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
    pass


def execute(args: Namespace) -> None:
    r"""TODO--- Rewrite the comment
    Creates a CaBRNet model, then trains it.

    Args:
        args (Namespace): Parsed arguments.

    """
    device = args.device
    method = str.lower(args.type)

    model = CaBRNet.build_from_config(args.model_config, state_dict_path=args.model_state_dict)
    model.eval()

    dataloaders = DatasetManager.get_dataloaders(config=args.dataset)
    output_file = args.output_file

    if "latent_repr" in dataloaders:
        key = "latent_repr"
    elif "train_set" in dataloaders:
        logger.warning("No `latent_repr' in dataset.yml. Using `train_set' instead.")
        key = "train_set"
    else:
        raise ArgumentError("'latent_repr' not found in dataloaders")
    draw_latent(model=model, dataloader=dataloaders[key], method=method, device=device, output_file=output_file)
