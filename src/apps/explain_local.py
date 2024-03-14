"""Declare the necessary functions to create an app to explain a classification result."""
import os.path
from pathlib import Path
from argparse import ArgumentParser, Namespace
from loguru import logger
from cabrnet.generic.model import CaBRNet
from cabrnet.utils.data import DatasetManager
from cabrnet.visualization.visualizer import SimilarityVisualizer
from cabrnet.utils.exceptions import ArgumentError

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
    parser = DatasetManager.create_parser(parser)
    parser = SimilarityVisualizer.create_parser(parser)
    parser.add_argument(
        "-c",
        "--checkpoint-dir",
        type=str,
        required=False,
        metavar="/path/to/checkpoint/dir",
        help="path to a checkpoint directory "
        "(alternative to --model-config, --model-state-dict, --dataset and --visualization)",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        metavar="path/to/image",
        help="path to image to be explained",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        metavar="path/to/output/directory",
        help="path to output directory",
    )
    parser.add_argument(
        "-p",
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


def check_args(args: Namespace) -> Namespace:
    if args.checkpoint_dir is not None:
        # Fetch all files from directory
        for param, name in zip(
            [args.model_config, args.model_state_dict, args.dataset, args.visualization],
            ["--model-config", "--model-state-dict", "--dataset", "--visualization"],
        ):
            if param is not None:
                raise ArgumentError(f"Cannot specify both options {name} and --checkpoint-dir")
        args.model_config = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_CONFIG)
        args.model_state_dict = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_STATE)
        args.dataset = os.path.join(args.checkpoint_dir, DatasetManager.DEFAULT_DATASET_CONFIG)
        args.visualization = os.path.join(args.checkpoint_dir, SimilarityVisualizer.DEFAULT_VISUALIZATION_CONFIG)

    # Check configuration completeness
    for param, name, option in zip(
        [args.model_config, args.model_state_dict, args.dataset, args.visualization],
        ["model configuration", "state dictionary", "dataset configuration", "visualization configuration"],
        ["-m", "-s", "-d", "-z"],
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} file (option {option}).")
    return args


def execute(args: Namespace) -> None:
    """Explain the decision of a cabrnet model.

    Args:
        args: Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    # Build model and load state dictionary
    model: CaBRNet = CaBRNet.build_from_config(config_file=args.model_config, state_dict_path=args.model_state_dict)
    # Init visualizer
    visualizer = SimilarityVisualizer.build_from_config(config_file=args.visualization)
    # Recover preprocessing function
    preprocess = DatasetManager.get_dataset_transform(config_file=args.dataset, dataset="test_set")

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
