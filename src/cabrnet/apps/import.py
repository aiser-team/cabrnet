import os
from argparse import ArgumentParser, Namespace

from loguru import logger

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.exceptions import ArgumentError
from cabrnet.core.utils.optimizers import OptimizerManager
from cabrnet.core.utils.parser import load_config
from cabrnet.core.utils.save import save_checkpoint

description = "converts an existing legacy model into a CaBRNet model"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    r"""Creates the argument parser for importing a legacy model into a CaBRNet model.

    Args:
        parser (ArgumentParser, optional): Parent parser (if any).
            Default: None

    Returns:
        The parser itself.
    """
    if parser is None:
        parser = ArgumentParser(description)
    parser = CaBRNet.create_parser(parser)
    parser = OptimizerManager.create_parser(parser)
    parser = DatasetManager.create_parser(parser)
    parser.add_argument(
        "-c",
        "--config-dir",
        type=str,
        required=False,
        metavar="/path/to/config/dir",
        help="path to directory containing all configuration files "
        "(alternative to --model-arch, --dataset and --training)",
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
    r"""Checks the validity of the arguments and updates the namespace if necessary.

    Args:
        args (Namespace): Parsed arguments.

    Returns:
        Modified argument namespace.
    """
    if args.config_dir is not None:
        # Fetch all files from directory
        for param, name in zip(
            [args.model_arch, args.dataset, args.training],
            ["--model-arch", "--dataset", "--training"],
        ):
            if param is not None:
                raise ArgumentError(f"Cannot specify both options {name} and --config_dir")
        args.model_arch = os.path.join(args.config_dir, CaBRNet.DEFAULT_MODEL_CONFIG)
        args.dataset = os.path.join(args.config_dir, DatasetManager.DEFAULT_DATASET_CONFIG)
        args.training = os.path.join(args.config_dir, OptimizerManager.DEFAULT_TRAINING_CONFIG)

    # Check configuration completeness
    for param, name, option in zip(
        [args.model_arch, args.dataset, args.training, args.model_state_dict],
        ["model configuration", "dataset configuration", "training configuration", "model state"],
        ["-m", "-d", "-t", "-s"],
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} file (option {option}).")
    return args


def execute(args: Namespace) -> None:
    r"""Creates a CaBRNet model, loads a state dictionary in legacy form, then performs the epilogue.

    Args:
        args (Namespace): Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    model_arch = args.model_arch
    dataset_config = args.dataset
    training_config = args.training
    legacy_state_dict = args.model_state_dict
    seed = args.seed
    verbose = args.verbose
    root_dir = args.output_dir
    device = args.device

    # Build CaBRNet model, then load legacy state dictionary
    model = CaBRNet.build_from_config(model_arch, state_dict_path=legacy_state_dict)
    model.eval()

    dataloaders = DatasetManager.get_dataloaders(dataset_config)

    # Build optimizer manager
    optimizer_mngr = OptimizerManager.build_from_config(config=training_config, model=model)

    # Call epilogue
    trainer = load_config(training_config)
    projection_info = model.epilogue(
        dataloaders=dataloaders,
        optimizer_mngr=optimizer_mngr,
        output_dir=root_dir,
        device=device,
        verbose=verbose,
        **trainer.get("epilogue", {}),
    )

    # Evaluate model
    eval_info = model.evaluate(dataloader=dataloaders["test_set"], device=device, verbose=verbose)
    logger.info(f"Average loss: {eval_info['avg_loss']:.2f}. Average accuracy: {eval_info['avg_accuracy']:.2f}.")
    save_checkpoint(
        directory_path=os.path.join(root_dir, "imported"),
        model=model,
        model_arch=model_arch,
        optimizer_mngr=None,
        training_config=training_config,
        dataset_config=dataset_config,
        projection_info=projection_info,
        epoch="final",
        seed=seed,
        device=device,
        stats=eval_info,
    )
