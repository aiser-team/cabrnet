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

"""Main entry point for CaBRNet (GUI version)."""

import importlib.metadata
import random
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from loguru import logger

from cabrnet.core.interface.analysis_gui import main as analysis_main
from cabrnet.core.interface.design_gui import main as design_main
from cabrnet.core.utils.system_info import get_hardware_info


class ParserWithHelper(ArgumentParser):
    """Helper class for better parser errors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_argument("-V", "--version", action="version", version=importlib.metadata.version("cabrnet"))

    def error(self, message: str | None = None):
        r"""Overrides default error message in argparse to print help menu."""
        if message is not None:
            self._print_message(f"Error: {message}\n", sys.stderr)
        self.print_help(sys.stderr)
        self.exit(2)


def main():
    r"""Front-end to the CaBRNet GUI."""
    # Create parser
    parser = ParserWithHelper(description="Graphical user interface (GUI) for designing and analysing CaBRNet models")

    parser.add_argument("--device", type=str, metavar="device", default="cuda:0", help="Target hardware device")
    parser.add_argument("--seed", type=int, default=42, metavar="value", help="Seed for reproducible experiments")
    # what level of information is stored in the log file
    parser.add_argument("--logger-level", type=str, metavar="level", default="INFO", help="Logger level")
    parser.add_argument(
        "--logger-file",
        type=str,
        metavar="path/to/file",
        help="Logger file (default: sys.stderr)",
    )
    # print logs and progress bars to the console
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(help="sub-command help", dest="appname")
    subparsers.required = True
    subparsers.add_parser("analysis", help="Analysis dashboard for CaBRNet models").set_defaults(func=analysis_main)
    subparsers.add_parser("design", help="Design dashboard for CaBRNet models").set_defaults(func=design_main)

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

    # Launch GUI
    args.func()


if __name__ == "__main__":
    main()
