import sys
from argparse import ArgumentParser, Namespace
from loguru import logger
from cabrnet.generic.model import ProtoClassifier

description = "explain the global behaviour of a CaBRNet classifier"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """Create the argument parser for explaining the global behaviour of a CaBRNet classifier.

    Returns:
        The parser itself.
    """
    if parser is None:
        parser = ArgumentParser(description)
    parser = ProtoClassifier.create_parser(parser)
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="path/to/output/directory",
        help="Path to output directory",
    )
    parser.add_argument(
        "--prototype-dir",
        type=str,
        required=True,
        metavar="path/to/prototype/directory",
        help="Path to directory containing prototype visualizations",
    )
    return parser


def execute(args: Namespace) -> None:
    """Explain the global behaviour of a CaBRNet classifier

    Args:
        args: Parsed arguments.

    """
    # Set logger level
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO"}])

    # Build model and load state dictionary
    model: ProtoClassifier = ProtoClassifier.build_from_config(
        config_file=args.model_config, state_dict_path=args.model_state_dict
    )

    # Generate explanation
    try:
        model.explain_global(
            prototype_dir_path=args.prototype_dir,
            output_dir_path=args.output_dir,
        )
    except NotImplementedError:
        logger.error("Global explanation not available for this model.")
