import os
from argparse import ArgumentParser, Namespace

from loguru import logger

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.exceptions import ArgumentError

description = "evaluates the accuracy of a CaBRNet model"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    r"""Creates the argument parser for evaluating a CaBRNet model.

    Args:
        parser (ArgumentParser, optional): Parent parser (if any).
            Default: None

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
        help="path to a checkpoint directory (alternative to --model-arch, --model-state-dict and --dataset)",
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
                logger.warning(f"Ignoring option {name}: using content pointed by --checkpoint-dir instead")
        args.model_arch = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_CONFIG)
        args.model_state_dict = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_STATE)
        args.dataset = os.path.join(args.checkpoint_dir, DatasetManager.DEFAULT_DATASET_CONFIG)

    # Check configuration completeness
    for param, name in zip(
        [args.model_arch, args.model_state_dict, args.dataset],
        ["model", "state dictionary", "dataset"],
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} configuration file.")
    return args


def execute(args: Namespace) -> None:
    r"""Evaluates the accuracy of a CaBRNet model.

    Args:
        args (Namespace): Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    model = CaBRNet.build_from_config(args.model_arch, state_dict_path=args.model_state_dict)
    model.eval()

    # Dataloaders
    dataloaders = DatasetManager.get_dataloaders(config=args.dataset)
    model.to(args.device)

    stats = model.evaluate(
        dataloader=dataloaders["test_set"], device=args.device, tqdm_position=0, verbose=args.verbose
    )
    for name, value in stats.items():
        logger.info(f"{name}: {value:.2f}")
