import os
import shutil
import importlib
import traceback
from loguru import logger
from cabrnet.generic.model import CaBRNet
from cabrnet.utils.data import DatasetManager
from cabrnet.visualization.visualizer import SimilarityVisualizer
from cabrnet.utils.exceptions import ArgumentError
from argparse import ArgumentParser, Namespace


description = "benchmark a CaBRNet model"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser(description)
    parser = CaBRNet.create_parser(parser)
    parser = DatasetManager.create_parser(parser)
    parser = SimilarityVisualizer.create_parser(parser)
    parser.add_argument(
        "-c",
        "--checkpoint-dir",
        type=str,
        required=False,
        metavar="/path/to/checkpoint/dir",
        help="path to a checkpoint directory "
        "(alternative to --model-config, --model-state-dict, --dataset and --visualization)",
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
    return parser


def check_args(args: Namespace) -> Namespace:
    if args.checkpoint_dir is not None:
        # Fetch all files from directory
        for param, name in zip(
            [args.model_config, args.model_state_dict, args.dataset, args.visualization],
            ["--model-config", "--model-state-dict", "--dataset", "--visualization"],
        ):
            if param is not None:
                logger.warning(f"Ignoring option {name}: using content pointed by --checkpoint-dir instead")
        args.model_config = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_CONFIG)
        args.model_state_dict = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_STATE)
        args.dataset = os.path.join(args.checkpoint_dir, DatasetManager.DEFAULT_DATASET_CONFIG)
        args.visualization = os.path.join(args.checkpoint_dir, SimilarityVisualizer.DEFAULT_VISUALIZATION_CONFIG)

    # Check configuration completeness
    for param, name in zip(
        [args.model_config, args.model_state_dict, args.dataset, args.visualization],
        [
            "model configuration",
            "model state",
            "dataset configuration",
            "visualization configuration",
        ],
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} file.")
    if os.path.isdir(args.output_dir) and not args.overwrite:
        raise ArgumentError("Output directory already exists. Use --overwrite option to override.")
    return args


def execute(args: Namespace) -> None:
    """Create CaBRNet model, then load a state dictionary in legacy form.

    Args:
        args: Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    model_config = args.model_config
    model_state_dict = args.model_state_dict
    dataset_config = args.dataset
    visualizer_config = args.visualization
    bench_config = args.benchmark_configuration
    verbose = args.verbose
    device = args.device
    output_dir = args.output_dir

    # Build CaBRNet model, then load state dictionary
    model = CaBRNet.build_from_config(model_config, state_dict_path=model_state_dict)
    model.eval()

    # Create output directory and copy test configuration
    os.makedirs(output_dir, exist_ok=True)  # Check has already been performed in check_args
    shutil.copyfile(src=bench_config, dst=os.path.join(output_dir, os.path.basename(bench_config)))

    # Enumerate all benchmarks
    bench_dir = os.path.join(os.path.dirname(__file__), "..", "cabrnet", "evaluation")
    bench_list = [os.path.splitext(file)[0] for file in os.listdir(bench_dir) if file.endswith(".py")]
    for bench_module in bench_list:
        try:
            module = importlib.import_module(f"cabrnet.evaluation.{bench_module}")
        except Exception as e:
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
            root_dir=output_dir,
            device=device,
            verbose=verbose,
            **config,
        )
