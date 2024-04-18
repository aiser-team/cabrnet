import os
from loguru import logger
from cabrnet.generic.model import CaBRNet
from cabrnet.utils.parser import load_config
from cabrnet.utils.optimizers import OptimizerManager
from cabrnet.utils.data import DatasetManager
from cabrnet.utils.save import save_checkpoint
from cabrnet.visualization.visualizer import SimilarityVisualizer
from cabrnet.utils.exceptions import ArgumentError
from argparse import ArgumentParser, Namespace


description = "convert an existing legacy model into a CaBRNet version"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser(description)
    parser = CaBRNet.create_parser(parser)
    parser = OptimizerManager.create_parser(parser)
    parser = DatasetManager.create_parser(parser)
    parser = SimilarityVisualizer.create_parser(parser)
    parser.add_argument(
        "-c",
        "--config-dir",
        type=str,
        required=False,
        metavar="/path/to/config/dir",
        help="path to directory containing all configuration files "
        "(alternative to --model-config, --dataset, --training and --visualization)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        metavar="path/to/output/directory",
        help="path to output directory",
    )
    return parser


def check_args(args: Namespace) -> Namespace:
    if args.config_dir is not None:
        # Fetch all files from directory
        for param, name in zip(
            [args.model_config, args.dataset, args.training, args.visualization],
            ["--model-config", "--dataset", "--training", "--visualization"],
        ):
            if param is not None:
                raise ArgumentError(f"Cannot specify both options {name} and --config_dir")
        args.model_config = os.path.join(args.config_dir, CaBRNet.DEFAULT_MODEL_CONFIG)
        args.dataset = os.path.join(args.config_dir, DatasetManager.DEFAULT_DATASET_CONFIG)
        args.training = os.path.join(args.config_dir, OptimizerManager.DEFAULT_TRAINING_CONFIG)
        args.visualization = os.path.join(args.config_dir, SimilarityVisualizer.DEFAULT_VISUALIZATION_CONFIG)

    # Check configuration completeness
    for param, name, option in zip(
        [args.model_config, args.dataset, args.training, args.visualization, args.model_state_dict],
        [
            "model configuration",
            "dataset configuration",
            "training configuration",
            "visualization configuration",
            "model state",
        ],
        ["-m", "-d", "-z", "-t", "-s"],
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} file (option {option}).")
    return args


def execute(args: Namespace) -> None:
    """Create CaBRNet model, then load a state dictionary in legacy form.

    Args:
        args: Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    model_config = args.model_config
    dataset_config = args.dataset
    training_config = args.training
    legacy_state_dict = args.model_state_dict
    seed = args.seed
    verbose = args.verbose
    root_dir = args.output_dir
    device = args.device

    # Build CaBRNet model, then load legacy state dictionary
    model = CaBRNet.build_from_config(model_config, state_dict_path=legacy_state_dict)
    model.eval()

    dataloaders = DatasetManager.get_dataloaders(dataset_config)

    # Build optimizer manager
    optimizer_mngr = OptimizerManager.build_from_config(config_file=training_config, model=model)

    # Call epilogue
    trainer = load_config(training_config)
    visualizer = SimilarityVisualizer.build_from_config(config_file=args.visualization, model=model)
    model.epilogue(
        dataloaders=dataloaders,
        visualizer=visualizer,
        optimizer_mngr=optimizer_mngr,
        output_dir=root_dir,
        model_config=model_config,
        training_config=training_config,
        dataset_config=dataset_config,
        seed=seed,
        device=device,
        verbose=verbose,
        **trainer.get("epilogue", {}),
    )  # type: ignore

    # Evaluate model
    eval_info = model.evaluate(dataloader=dataloaders["test_set"], device=device, verbose=verbose)
    logger.info(f"Average loss: {eval_info['avg_loss']:.2f}. Average accuracy: {eval_info['avg_eval_accuracy']:.2f}.")
    save_checkpoint(
        directory_path=os.path.join(root_dir, f"imported"),
        model=model,
        model_config=model_config,
        optimizer_mngr=None,
        training_config=training_config,
        dataset_config=dataset_config,
        visualization_config=args.visualization,
        epoch="imported",
        seed=seed,
        device=device,
        stats=eval_info,
    )
