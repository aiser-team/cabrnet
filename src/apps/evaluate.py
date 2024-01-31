"""Declare the necessary functions to create an app to evaluate a CABRNet classifier."""

import os
import sys
import torch
from argparse import ArgumentParser, Namespace

from loguru import logger

from cabrnet.generic.model import ProtoClassifier
from cabrnet.utils.data import create_dataset_parser, get_dataloaders

description = "evaluate a CaBRNet classifier"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """Create the argument parser for evaluating a CaBRNet classifier.

    Returns:
        The parser itself.
    """
    if parser is None:
        parser = ArgumentParser(description)
    parser = ProtoClassifier.create_parser(parser)
    parser = create_dataset_parser(parser)
    parser.add_argument(
        "--legacy-state-dict-dir",
        type=str,
        required=False,
        metavar="path/to/state_dict/directory",
        help="Path to directory containing state dictionary of a legacy model",
    )
    return parser


def execute(args: Namespace) -> None:
    """Evaluate a CaBRNet classifier.

    Args:
        args: Parsed arguments.

    """
    # Set logger level
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO"}])
    model: ProtoClassifier = ProtoClassifier.build_from_config(args.model_config)  # type: ignore
    if args.legacy_state_dict_dir:
        model.load_legacy_state_dict(
            torch.load(os.path.join(args.legacy_state_dict_dir, "model_state.pth"), map_location="cpu")
        )  # type: ignore
    model.eval()

    # Dataloaders
    dataloaders = get_dataloaders(config_file=args.dataset)
    model.to(args.device)

    stats = model.evaluate(dataloader=dataloaders["test_set"], device=args.device, progress_bar_position=0)
    for name, value in stats.items():
        logger.info(f"{name}: {value:.2f}")
