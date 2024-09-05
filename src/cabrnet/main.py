# "Copyright (C) 2024 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
#
# Licensed under the GNU LGPL, Version 2.1 (the "License");
# You may obtain a copy of the License at:
# https://www.gnu.org/licenses/old-licenses/lgpl-2.1-standalone.html
#
# Permission to use, copy, modify, and/or distribute this software for any purpose with or
# without fee is hereby granted, provided that the above copyright notice and this permission
# notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
# SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THIS SOFTWARE.”

"""Main entry point for CaBRNet."""

import importlib
import importlib.metadata
import os
import random
import sys
import traceback
from argparse import ArgumentParser

import numpy as np
import torch
from loguru import logger

from cabrnet.core.utils.exceptions import ArgumentError
from cabrnet.core.utils.system_info import get_hardware_info


class ParserWithHelper(ArgumentParser):
    """Helper class for better parser errors."""

    def error(self, message: str | None = None):
        r"""Overrides default error message in argparse to print help menu."""
        if message is not None:
            self._print_message(f"Error: {message}\n", sys.stderr)
        self.print_help(sys.stderr)
        self.exit(2)


def main():
    r"""Front-end to all applications located in src/cabrnet/apps."""
    # Enumerate applications from apps directory
    apps_dir = os.path.join(os.path.dirname(__file__), "apps")
    apps = [os.path.splitext(file)[0] for file in os.listdir(apps_dir) if file.endswith(".py")]

    # Create parser
    parser = ParserWithHelper(description="CaBRNet front-end")
    subparsers = parser.add_subparsers(help="sub-command help", dest="appname")
    subparsers.required = True
    for app_name in apps:
        try:
            module = importlib.import_module(f"cabrnet.apps.{app_name}")
        except Exception as _:
            logger.warning(f"Skipping application {app_name}. Could not load module: {traceback.format_exc()}")
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
            help="Logger file (default: sys.stderr)",
        )
        # print logs and progress bars to the console
        common_group.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-V", "--version", action="version", version=importlib.metadata.version("cabrnet"))

    args = parser.parse_args()

    # Set random seeds
    seed = args.seed
    torch.use_deterministic_algorithms(mode=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set logger level
    logger.configure(handlers=[{"sink": sys.stderr, "level": args.logger_level}])
    if args.logger_file is not None:
        logger.info(f"Using log file: {args.logger_file}")
        logger.add(sink=args.logger_file, level=args.logger_level)
    logger.info(f"Hardware information: {get_hardware_info(args.device)}")

    if not hasattr(args, "func"):
        # Print help menu when no argument is given
        parser.print_help()
    else:
        try:
            args.func(args)
        except ArgumentError as e:
            print(e)
            parser.parse_args([args.appname, "-h"])


if __name__ == "__main__":
    main()
