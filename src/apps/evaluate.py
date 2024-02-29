"""Declare the necessary functions to create an app to evaluate a CABRNet classifier."""
import os
from argparse import ArgumentParser, Namespace

from loguru import logger

from cabrnet.generic.model import CaBRNet
from cabrnet.utils.data import create_dataset_parser, get_dataloaders

description = "evaluate a CaBRNet classifier"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """Create the argument parser for evaluating a CaBRNet classifier.

    Returns:
        The parser itself.
    """
    if parser is None:
        parser = ArgumentParser(description)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=False,
        metavar="/path/to/checkpoint/dir",
        help="path to a checkpoint directory (alternative to --model-config, --model-state-dict and --dataset)",
    )
    parser = CaBRNet.create_parser(parser)
    parser = create_dataset_parser(parser)
    return parser


def check_args(args: Namespace) -> Namespace:
    if args.checkpoint_dir is not None:
        # Fetch all files from directory
        for param, name in zip(
            [args.model_config, args.model_state_dict, args.dataset],
            ["--model-config", "--model-state-dict", "--dataset"],
        ):
            if param is not None:
                logger.warning(f"Ignoring option {name}: using content pointed by --checkpoint-dir instead")
        args.model_config = os.path.join(args.checkpoint_dir, "model_arch.yml")
        args.model_state_dict = os.path.join(args.checkpoint_dir, "model_state.pth")
        args.dataset = os.path.join(args.checkpoint_dir, "dataset.yml")

    # Check configuration completeness
    for param, name in zip(
        [args.model_config, args.model_state_dict, args.dataset],
        ["model", "state dictionary", "dataset"],
    ):
        if param is None:
            raise AttributeError(f"Missing {name} configuration file.")
    return args


def execute(args: Namespace) -> None:
    """Evaluate a CaBRNet classifier.

    Args:
        args: Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    model = CaBRNet.build_from_config(args.model_config, state_dict_path=args.model_state_dict)
    model.eval()

    # Dataloaders
    dataloaders = get_dataloaders(config_file=args.dataset)
    model.to(args.device)

    stats = model.evaluate(
        dataloader=dataloaders["test_set"], device=args.device, progress_bar_position=0, verbose=args.verbose
    )
    for name, value in stats.items():
        logger.info(f"{name}: {value:.2f}")
