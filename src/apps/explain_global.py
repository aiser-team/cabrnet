import os
from argparse import ArgumentParser, Namespace
from loguru import logger
from cabrnet.generic.model import CaBRNet
from cabrnet.utils.exceptions import ArgumentError

description = "explain the global behaviour of a CaBRNet classifier"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """Create the argument parser for explaining the global behaviour of a CaBRNet classifier.

    Returns:
        The parser itself.
    """
    if parser is None:
        parser = ArgumentParser(description)
    parser = CaBRNet.create_parser(parser)
    parser.add_argument(
        "-c",
        "--checkpoint-dir",
        type=str,
        required=False,
        metavar="/path/to/checkpoint/dir",
        help="path to a checkpoint directory (alternative to --model-config, --model-state-dict)",
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
    return parser


def check_args(args: Namespace) -> Namespace:
    if args.checkpoint_dir is not None:
        # Fetch all files from directory
        for param, name in zip(
            [args.model_config, args.model_state_dict],
            ["--model-config", "--model-state-dict"],
        ):
            if param is not None:
                raise ArgumentError(f"Cannot specify both options {name} and --checkpoint-dir")
        args.model_config = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_CONFIG)
        args.model_state_dict = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_STATE)

    # Check configuration completeness
    for param, name, option in zip(
        [args.model_config, args.model_state_dict],
        ["model configuration", "state dictionary"],
        ["-m", "-s"],
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} file (option {option}).")
    return args


def execute(args: Namespace) -> None:
    """Explain the global behaviour of a CaBRNet classifier

    Args:
        args: Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    # Build model and load state dictionary
    model: CaBRNet = CaBRNet.build_from_config(config_file=args.model_config, state_dict_path=args.model_state_dict)

    # Generate explanation
    try:
        model.explain_global(
            prototype_dir_path=args.prototype_dir,
            output_dir_path=args.output_dir,
        )
    except NotImplementedError:
        logger.error("Global explanation not available for this model.")
