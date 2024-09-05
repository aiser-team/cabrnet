import importlib
import os
import shutil
import traceback
from argparse import ArgumentParser, Namespace

from loguru import logger

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.exceptions import ArgumentError
from cabrnet.core.visualization.visualizer import SimilarityVisualizer

description = "computes a set of evaluation metrics on a CaBRNet model"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    r"""Creates the argument parser for evaluating a CaBRNet model using specific metrics.

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
    parser = SimilarityVisualizer.create_parser(parser, mandatory_config=True)
    parser.add_argument(
        "-p",
        "--projection-info",
        type=str,
        required=False,
        metavar="/path/to/projection/info",
        help="path to the CSV file containing the projection information",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-dir",
        type=str,
        required=False,
        metavar="/path/to/checkpoint/dir",
        help="path to a checkpoint directory " "(alternative to --model-arch, --model-state-dict and --dataset)",
    )
    parser.add_argument(
        "-b",
        "--benchmark-configuration",
        type=str,
        required=True,
        metavar="/path/to/configuration/file",
        help="benchmark configuration file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        metavar="path/to/output/directory",
        help="path to output directory containing all analysis results",
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite output directory if necessary")
    parser.add_argument(
        "-y",
        "--prototype-dir",
        type=str,
        required=False,
        metavar="path/to/prototype/directory",
        help="path to directory containing a visualization of all prototypes",
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
            [args.model_arch, args.model_state_dict, args.dataset, args.projection_info],
            ["--model-arch", "--model-state-dict", "--dataset", "--projection-info"],
        ):
            if param is not None:
                logger.warning(f"Ignoring option {name}: using content pointed by --checkpoint-dir instead")
        args.model_arch = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_CONFIG)
        args.model_state_dict = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_STATE)
        args.dataset = os.path.join(args.checkpoint_dir, DatasetManager.DEFAULT_DATASET_CONFIG)
        args.projection_info = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_PROJECTION_INFO)

    # Check configuration completeness
    for param, name, option in zip(
        [args.model_arch, args.model_state_dict, args.dataset, args.projection_info],
        ["model configuration", "state dictionary", "dataset configuration", "projection information"],
        ["-m", "-s", "-d", "-p"],
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} file (option {option}).")
    if os.path.isdir(args.output_dir) and not args.overwrite:
        raise ArgumentError("Output directory already exists. Use --overwrite option to override.")

    # Set-up prototype directory if necessary
    if args.prototype_dir is None:
        args.prototype_dir = os.path.join(os.path.dirname(args.model_arch), "prototypes")

    return args


def execute(args: Namespace) -> None:
    r"""Computes a set of evaluation metrics on a CaBRNet model, based on a configuration file.

    Args:
        args (Namespace): Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    model_arch = args.model_arch
    model_state_dict = args.model_state_dict
    dataset_config = args.dataset
    visualizer_config = args.visualization
    projection_file = args.projection_info
    bench_config = args.benchmark_configuration
    prototype_dir = args.prototype_dir
    verbose = args.verbose
    device = args.device
    output_dir = args.output_dir

    # Build CaBRNet model, then load state dictionary
    model = CaBRNet.build_from_config(model_arch, state_dict_path=model_state_dict)
    model.eval()

    # Create output directory and copy test configuration
    os.makedirs(output_dir, exist_ok=True)  # Check has already been performed in check_args
    shutil.copyfile(src=bench_config, dst=os.path.join(output_dir, os.path.basename(bench_config)))

    # Enumerate all benchmarks
    bench_dir = os.path.join(os.path.dirname(__file__), "..", "core", "evaluation")
    bench_list = [os.path.splitext(file)[0] for file in os.listdir(bench_dir) if file.endswith(".py")]
    for bench_module in bench_list:
        try:
            module = importlib.import_module(f"cabrnet.core.evaluation.{bench_module}")
        except Exception as _:
            logger.warning(f"Skipping benchmark {bench_module}: could not load module. {traceback.format_exc()}")
            continue
        skip_module = False
        for mandatory_fn in ["get_config", "execute"]:
            if not hasattr(module, mandatory_fn):
                logger.warning(f"Skipping benchmark {bench_module}: missing mandatory function {mandatory_fn}")
                skip_module = True
                break

        # Extract bench-specific options
        config = module.get_config(bench_config)
        if config is None:
            logger.warning(f"Skipping benchmark {bench_module}: disabled in configuration file.")
            skip_module = True
        if skip_module:
            continue

        # Benchmark is enabled
        module.execute(
            model=model,
            dataset_config=dataset_config,
            visualization_config=visualizer_config,
            projection_file=projection_file,
            root_dir=output_dir,
            device=device,
            verbose=verbose,
            prototype_dir=prototype_dir,
            **config,
        )
