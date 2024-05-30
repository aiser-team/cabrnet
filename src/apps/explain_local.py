import os.path
from pathlib import Path
from argparse import ArgumentParser, Namespace
from cabrnet.generic.model import CaBRNet
from cabrnet.utils.data import DatasetManager
from cabrnet.visualization.visualizer import SimilarityVisualizer
from cabrnet.utils.exceptions import ArgumentError
from cabrnet.utils.save import load_projection_info, safe_copy

description = "explains the decision of a CaBRNet model"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    r"""Creates the argument parser for explaining the decision of a CaBRNet model.

    Args:
        parser (ArgumentParser, optional): Parent parser (if any).
            Default: None

    Returns:
        The parser itself.
    """
    if parser is None:
        parser = ArgumentParser(description)
    parser = CaBRNet.create_parser(parser)
    # Relies on dataset configuration of the test to deduce the type of preprocessing
    # that needs to be applied on the source image
    parser = DatasetManager.create_parser(parser)
    parser = SimilarityVisualizer.create_parser(parser, mandatory_config=True)
    parser.add_argument(
        "-c",
        "--checkpoint-dir",
        type=str,
        required=False,
        metavar="/path/to/checkpoint/dir",
        help="path to a checkpoint directory (alternative to --model-config, --model-state-dict, --dataset)",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        metavar="path/to/image",
        help="path to image to be explained",
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
        "--projection-info",
        type=str,
        metavar="/path/to/projection/info",
        help="path to the CSV file containing the projection informations",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing explanation (if any)",
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
            [args.model_config, args.model_state_dict, args.dataset, args.projection_info],
            ["--model-config", "--model-state-dict", "--dataset", "--projection-info"],
        ):
            if param is not None:
                raise ArgumentError(f"Cannot specify both options {name} and --checkpoint-dir")
        args.model_config = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_CONFIG)
        args.model_state_dict = os.path.join(args.checkpoint_dir, CaBRNet.DEFAULT_MODEL_STATE)
        args.dataset = os.path.join(args.checkpoint_dir, DatasetManager.DEFAULT_DATASET_CONFIG)

    # Check configuration completeness
    for param, name, option in zip(
        [args.model_config, args.model_state_dict, args.dataset, args.projection_info],
        ["model configuration", "state dictionary", "dataset configuration", "projection informations"],
        ["-m", "-s", "-d", "-p"],
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} file (option {option}).")

    output_dir = os.path.join(args.output_dir, Path(args.image).stem)
    if os.path.exists(output_dir) and not args.overwrite:
        raise ArgumentError(
            f"Output directory {output_dir} is not empty. " f"To overwrite existing results, use --overwrite option."
        )

    return args


def execute(args: Namespace) -> None:
    r"""Explains the decision of a CaBRNet model.

    Args:
        args (Namespace): Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    # Build model and load state dictionary
    model: CaBRNet = CaBRNet.build_from_config(config_file=args.model_config, state_dict_path=args.model_state_dict)

    # Init visualizer
    visualizer = SimilarityVisualizer.build_from_config(config_file=args.visualization, model=model)

    # Recover preprocessing function
    preprocess = DatasetManager.get_dataset_transform(config_file=args.dataset, dataset="test_set")

    # Build prototypes
    dataloaders = DatasetManager.get_dataloaders(config_file=args.dataset)
    projection_info = load_projection_info(args.projection_info)
    model.extract_prototypes(
        dataloader_raw=dataloaders["projection_set_raw"],
        dataloader=dataloaders["projection_set"],
        projection_info=projection_info,
        visualizer=visualizer,
        dir_path=os.path.join(args.output_dir, "prototypes"),
        device=args.device,
        verbose=args.verbose,
    )

    # Save visualization config
    safe_copy(
        args.visualization,
        os.path.join(args.output_dir, "prototypes", SimilarityVisualizer.DEFAULT_VISUALIZATION_CONFIG),
    )

    # Generate explanation
    model.explain(
        img=args.image,
        visualizer=visualizer,
        preprocess=preprocess,  # type: ignore
        prototype_dir=os.path.join(args.output_dir, "prototypes"),
        output_dir=os.path.join(args.output_dir, Path(args.image).stem),
        device=args.device,
        exist_ok=args.overwrite,
    )
