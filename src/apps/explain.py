"""Declare the necessary functions to create an app to explain a classification result."""
import os.path
from pathlib import Path
import sys
from argparse import ArgumentParser, Namespace
from loguru import logger
from cabrnet.generic.model import CaBRNet
from cabrnet.utils.data import create_dataset_parser, get_dataset_transform
from cabrnet.visualisation.visualizer import SimilarityVisualizer

description = "explain the decision of a CaBRNet classifier"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """Create the argument parser for explaining the decision of a CaBRNet classifier.

    Returns:
        The parser itself.
    """
    if parser is None:
        parser = ArgumentParser(description)
    parser = CaBRNet.create_parser(parser)
    # Relies on dataset configuration of the test to deduce the type of preprocessing
    # that needs to be applied on the source image
    parser = create_dataset_parser(parser)
    parser = SimilarityVisualizer.create_parser(parser)
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        metavar="path/to/image",
        help="path to image to be explained",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="path/to/output/directory",
        help="path to output directory",
    )
    parser.add_argument(
        "--prototype-dir",
        type=str,
        required=True,
        metavar="path/to/prototype/directory",
        help="path to directory containing prototype visualizations",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing explanation (if any)",
    )
    return parser


def execute(args: Namespace) -> None:
    """Explain the decision of a cabrnet model.

    Args:
        args: Parsed arguments.

    """
    # Set logger level
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO"}])

    # Build model and load state dictionary
    model: CaBRNet = CaBRNet.build_from_config(config_file=args.model_config, state_dict_path=args.model_state_dict)
    # Init visualizer
    visualizer = SimilarityVisualizer.build_from_config(config_file=args.visualization, target="test_patch")
    # Recover preprocessing function
    preprocess = get_dataset_transform(config_file=args.dataset, dataset="test_set")

    # Generate explanation
    model.explain(
        img_path=args.image,
        preprocess=preprocess,
        visualizer=visualizer,
        prototype_dir_path=args.prototype_dir,
        output_dir_path=os.path.join(args.output_dir, Path(args.image).stem),
        device=args.device,
        exist_ok=args.overwrite,
    )
