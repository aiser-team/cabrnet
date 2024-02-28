from argparse import ArgumentParser, Namespace
from zenodo_get import zenodo_get
import os

description = "download example models"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser(description)
    parser.add_argument(
        "--dir",
        type=str,
        default="examples",
        required=False,
        metavar="path/to/example/directory",
        help="path to destination directory of examples",
    )
    return parser


file_list = [{"record": "10066893", "dir": "pretrained_conv_extractors"}]


def execute(args: Namespace) -> None:
    """Download several pre-trained models from Zenodo

    Args:
        args: Parsed arguments.

    """
    for entry in file_list:
        target_path = os.path.join(args.dir, entry["dir"])
        zenodo_get(["-o", target_path, "-r", entry["record"]])
