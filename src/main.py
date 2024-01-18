"""Main entry point for CaBRNet."""

import importlib
import os
import sys
import random
from argparse import ArgumentParser
from loguru import logger

import numpy as np
import torch

VERSION = "0.1-alpha"


def main():
    """Load the applications and run CaBRNet with them."""
    # Enumerate applications from apps directory
    apps_dir = os.path.join(os.path.dirname(__file__), "apps")
    apps = [os.path.splitext(file)[0] for file in os.listdir(apps_dir) if file.endswith(".py")]

    # Create parser
    parser = ArgumentParser(description="CaBRNet front-end")
    subparsers = parser.add_subparsers(help="sub-command help")
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
        module.create_parser(subparser)
    parser.add_argument("--version", "-V", action="version", version=VERSION)
    parser.add_argument("--device", type=str, default="cuda:0", help="Target hardware device")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Seed for reproducible experiments")
    # what level of information is stored in the log file
    parser.add_argument("--logger-level", type=str, default="INFO", help="Logger level and verbosity")
    # print logs and progress bars to the console
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Set random seeds
    seed = args.seed
    torch.use_deterministic_algorithms(mode=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set logger level
    logger.configure(handlers=[{"sink": sys.stderr, "level": args.logger_level}])

    if not hasattr(args, "func"):
        # Print help menu when no argument is given
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
