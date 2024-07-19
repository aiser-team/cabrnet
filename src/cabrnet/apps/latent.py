from argparse import ArgumentParser, Namespace
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager

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


def draw_latent(model: CaBRNet, dataloaders: dict, output_file: str):
    pass


def execute(args: Namespace) -> None:
    r"""TODO--- Rewrite the comment
    Creates a CaBRNet model, then trains it.

    Args:
        args (Namespace): Parsed arguments.

    """
    verbose = args.verbose
    device = args.device

    model = CaBRNet.build_from_config(args.model_config, state_dict_path=args.model_state_dict)
    model.eval()

    dataloaders = DatasetManager.get_dataloaders(config=args.dataset)
    output_file = args.output_file

    draw_latent(model, dataloaders, output_file)

    prototypes = model.classifier.prototypes
    # model.extractor

    pass
