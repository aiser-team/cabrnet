from argparse import ArgumentParser, Namespace

import torch

from loguru import logger

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.exceptions import ArgumentError
from cabrnet.core.utils.optimizers import OptimizerManager
from cabrnet.core.utils.parser import load_config

description = "evaluates the accuracy of a CaBRNet model"

alternatives = (
    CaBRNet.ARCHITECTURE_ALTERNATIVE
    + CaBRNet.STATE_ALTERNATIVE
    + DatasetManager.DATASET_ALTERNATIVE
    + OptimizerManager.TRAINING_ALTERNATIVE
)


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    r"""Creates the argument parser for evaluating a CaBRNet model.

    Args:
        parser (ArgumentParser, optional): Parent parser (if any). Default: None.

    Returns:
        The parser itself.
    """
    if parser is None:
        parser = ArgumentParser(description)
    parser = CaBRNet.create_parser(parser)
    parser = DatasetManager.create_parser(parser)
    parser = OptimizerManager.create_parser(parser)
    parser = CaBRNet.create_checkpoint_parser(parser, checkpoint_dest="--checkpoint-dir", alternatives=alternatives)
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=None,
        required=False,
        metavar="dataset-name",
        help="name of the target dataset(s). Default: all declared test sets",
    )
    return parser


def check_args(args: Namespace) -> Namespace:
    r"""Checks the validity of the arguments and updates the namespace if necessary.

    Args:
        args (Namespace): Parsed arguments.

    Returns:
        Modified argument namespace.
    """
    CaBRNet.check_args(args, checkpoint_dest="--checkpoint-dir", alternatives=alternatives)

    # Check configuration completeness
    for param, name, option in zip(
        [args.model_arch, args.model_state_dict, args.dataset, args.training],
        ["model", "state dictionary", "dataset", "training"],
        ["-m", "-s", "-d", "-t"],
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} configuration file (option {option}).")
    return args


def execute(args: Namespace) -> None:
    r"""Evaluates the accuracy of a CaBRNet model.

    Args:
        args (Namespace): Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    model = CaBRNet.build_from_config(args.model_arch, state_dict_path=args.model_state_dict)
    for module_path, weights_path in CaBRNet.parse_load_weights(args.load_weights).items():
        model.load_submodule_state_dict(module_path, torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    # Register auxiliary training parameters (e.g. loss configuration)
    trainer = load_config(args.training)
    model.register_training_params(trainer)

    # Dataloaders
    dataloaders = DatasetManager.get_dataloaders(config=args.dataset, sampling_ratio=args.sampling_ratio)
    model.to(args.device)

    targets = args.targets if args.targets is not None else DatasetManager.test_set_names(dataloaders)
    if not targets:
        raise ArgumentError("No test set declared in the dataset configuration.")

    # Check relevance of all targets first
    for target in targets:
        if not dataloaders.get(target):
            raise ArgumentError(f"Unknown target dataset: {target}")

    for target in targets:
        logger.info(f"Target: {target}")
        stats = model.evaluate(
            dataloaders=dataloaders, dataset_name=target, device=args.device, tqdm_position=0, verbose=args.verbose
        )
        for name, value in stats.items():
            logger.info(f"{name}: {value:.2f}")
