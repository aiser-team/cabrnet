"""Declare the necessary functions to create an app to evaluate a CABRNet classifier."""

import sys
import torch
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
    parser = CaBRNet.create_parser(parser)
    parser = create_dataset_parser(parser)
    return parser


def execute(args: Namespace) -> None:
    """Evaluate a CaBRNet classifier.

    Args:
        args: Parsed arguments.

    """
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
