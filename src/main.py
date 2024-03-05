"""Main entry point for CaBRNet."""

import importlib
import os
import pathlib
import sys
import random
from argparse import ArgumentParser
from loguru import logger

import numpy as np
import torch


def get_version() -> str:
    with open(os.path.join(pathlib.Path(__file__).parent.resolve(), "..", "VERSION"), "r") as fin:
        return fin.readline()


class ParserWithHelper(ArgumentParser):
    def error(self, message: str):
        """Overrides default error message in argparse to print help menu"""
        self._print_message(f"Error: {message}\n", sys.stderr)
        self.print_help(sys.stderr)
        self.exit(2)


def main():
    """Load the applications and run CaBRNet with them."""
    # Enumerate applications from apps directory
    apps_dir = os.path.join(os.path.dirname(__file__), "apps")
    apps = [os.path.splitext(file)[0] for file in os.listdir(apps_dir) if file.endswith(".py")]

    # Create parser
    parser = ParserWithHelper(description="CaBRNet front-end")
    subparsers = parser.add_subparsers(help="sub-command help", dest="appname")
    subparsers.required = True
    for app_name in apps:
        try:
            module = importlib.import_module(f"apps.{app_name}")
        except ModuleNotFoundError as e:
            logger.warning(f"Skipping application {app_name}. Could not load module: {e}")
            continue
        description = module.description if hasattr(module, "description") else f"help menu for {app_name}"
        if not hasattr(module, "create_parser") or not hasattr(module, "execute"):
            logger.warning(f"Skipping application {app_name} due to missing mandatory function(s)")
            continue
        # Create dedicated sub-parser
        subparser = subparsers.add_parser(app_name, help=description)
        subparser.set_defaults(func=module.execute)
        app_specific_parser = subparser.add_argument_group(description="APPLICATION-SPECIFIC OPTIONS")
        module.create_parser(app_specific_parser)
        common_group = subparser.add_argument_group(description="OTHER OPTIONS")
        common_group.add_argument(
            "--device", type=str, metavar="device", default="cuda:0", help="Target hardware device"
        )
        common_group.add_argument(
            "--seed", type=int, default=42, metavar="value", help="Seed for reproducible experiments"
        )
        # what level of information is stored in the log file
        common_group.add_argument("--logger-level", type=str, metavar="level", default="INFO", help="Logger level")
        common_group.add_argument(
            "--logger-file",
            type=str,
            metavar="path/to/file",
            default=sys.stderr,
            help="Logger file (default: sys.stderr)",
        )
        # print logs and progress bars to the console
        common_group.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-V", "--version", action="version", version=get_version())

    args = parser.parse_args()

    # Set random seeds
    seed = args.seed
    torch.use_deterministic_algorithms(mode=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set logger level
    logger.configure(handlers=[{"sink": args.logger_file, "level": args.logger_level}])

    if not hasattr(args, "func"):
        # Print help menu when no argument is given
        parser.print_help()
    else:
        try:
            args.func(args)
        except AttributeError as e:
            print(e)
            parser.parse_args([args.appname, "-h"])


if __name__ == "__main__":
    main()
