import os
from argparse import ArgumentParser, Namespace

from loguru import logger
from tqdm import tqdm

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.exceptions import ArgumentError
from cabrnet.core.utils.optimizers import OptimizerManager
from cabrnet.core.utils.parser import load_config
from cabrnet.core.utils.save import load_checkpoint
from cabrnet.core.utils.system_info import get_parent_directory
from cabrnet.core.utils.train import training_loop

description = "trains a CaBRNet model"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    r"""Creates the argument parser for training a CaBRNet model.

    Args:
        parser (ArgumentParser, optional): Parent parser (if any).
            Default: None.

    Returns:
        The parser itself.
    """
    if parser is None:
        parser = ArgumentParser(description)
    parser = CaBRNet.create_parser(parser, skip_state_dict=True)
    parser = DatasetManager.create_parser(parser)
    parser = OptimizerManager.create_parser(parser)
    parser.add_argument(
        "-b",
        "--save-best",
        type=str,
        required=False,
        nargs=2,
        default=["accuracy", "max"],
        metavar=("metric", "min/max"),
        help="save best model based on chosen metric and mode (min or max)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=False,
        metavar="/path/to/output/directory",
        help="path to output directory",
    )
    x_group = parser.add_mutually_exclusive_group(required=False)
    x_group.add_argument(
        "-c",
        "--config-dir",
        type=str,
        required=False,
        metavar="/path/to/config/dir",
        help="path to directory containing all configuration files to start training "
        "(alternative to --model-arch, --dataset and --training)",
    )
    x_group.add_argument(
        "-r",
        "--resume-from",
        type=str,
        metavar="/path/to/checkpoint/directory",
        help="path to existing checkpoint directory to resume training",
    )
    parser.add_argument(
        "-f",
        "--checkpoint-frequency",
        type=int,
        required=False,
        metavar="num_epochs",
        help="checkpoint frequency (in epochs)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow output directory to be overwritten with new results. This option should be enabled when "
        "resuming training from a given checkpoint.",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="check the training pipeline without performing the entire process.",
    )
    parser.add_argument(
        "--epilogue",
        action="store_true",
        help="skip training and go to epilogue.",
    )
    return parser


def check_args(args: Namespace) -> Namespace:
    r"""Checks the validity of the arguments and updates the namespace if necessary.

    Args:
        args (Namespace): Parsed arguments.

    Returns:
        Modified argument namespace.
    """
    for dir_path, option_name in zip([args.resume_from, args.config_dir], ["--resume-from", "--config-dir"]):
        if dir_path is not None:
            for param, name in zip(
                [args.model_arch, args.dataset, args.training], ["--model-arch", "--dataset", "--training"]
            ):
                if param is not None:
                    raise ArgumentError(f"Cannot specify both options {name} and {option_name}")
            args.model_arch = os.path.join(dir_path, CaBRNet.DEFAULT_MODEL_CONFIG)
            # Compatibility with v0.1: will be removed in the future
            if not os.path.isfile(args.model_arch) and os.path.isfile(os.path.join(dir_path, "model.yml")):
                args.model_arch = os.path.join(dir_path, "model.yml")
                logger.warning(
                    f"Using model.yml from {dir_path}: "
                    f"please consider renaming the file to {CaBRNet.DEFAULT_MODEL_CONFIG} to ensure compatibility "
                    f"with future versions"
                )
            args.dataset = os.path.join(dir_path, DatasetManager.DEFAULT_DATASET_CONFIG)
            args.training = os.path.join(dir_path, OptimizerManager.DEFAULT_TRAINING_CONFIG)

    for param, name, option in zip(
        [args.model_arch, args.dataset, args.training], ["model", "dataset", "training"], ["-m", "-d", "-t"]
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} configuration file (option {option}).")

    # Check optimization mode
    if args.save_best[1] not in ["min", "max"]:
        raise ArgumentError(f"Invalid optimization mode '{args.save_best[1]}' in --save-best")

    # In resume mode, specifying the output directory is optional
    if args.output_dir is None and args.resume_from is not None:
        args.output_dir = get_parent_directory(args.resume_from)
        logger.warning(f"Using {args.output_dir} as default output directory based on checkpoint path")
    if args.output_dir is None:
        raise ArgumentError("Missing path to output directory (option --output-dir)")

    # In full training mode (all epochs), or when the output directory is different from the checkpoint parent directory
    # (resume mode), check that the best model directory is available
    best_model_path = os.path.join(args.output_dir, "best")
    if (
        os.path.exists(best_model_path)
        and not args.overwrite
        and not args.epilogue
        and (args.resume_from is None or get_parent_directory(args.resume_from) != args.output_dir)
    ):
        raise ArgumentError(
            f"Output directory {best_model_path} is not empty. "
            f"To overwrite existing results, use --overwrite option."
        )
    final_model_path = os.path.join(args.output_dir, "final")
    if args.epilogue and os.path.exists(final_model_path) and not args.overwrite:
        raise ArgumentError(
            f"Output directory {final_model_path} is not empty. "
            f"To overwrite existing results, use --overwrite option."
        )
    if args.sanity_check and args.sampling_ratio == 1:
        # In sanity check mode, increase the sampling ratio
        args.sampling_ratio = 100

    return args


def execute(args: Namespace) -> None:
    r"""Creates a CaBRNet model, then trains it.

    Args:
        args (Namespace): Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    # Recover options
    verbose = args.verbose
    device = args.device
    training_config = args.training
    model_arch = args.model_arch
    dataset_config = args.dataset
    epilogue_only = args.epilogue
    sanity_check_only = args.sanity_check
    resume_dir = args.resume_from

    model: CaBRNet = CaBRNet.build_from_config(config=model_arch, seed=args.seed)

    # Training configuration
    trainer = load_config(training_config)
    model.register_training_params(trainer)  # Register auxiliary training parameters directly into model
    root_dir = args.output_dir

    # Build optimizer manager
    optimizer_mngr = OptimizerManager.build_from_config(config=training_config, model=model)
    # Dataloaders
    dataloaders = DatasetManager.get_dataloaders(config=dataset_config, sampling_ratio=args.sampling_ratio)
    # By default, process all data batches and all epochs
    start_epoch = 0
    num_epochs = trainer["num_epochs"]
    metric = args.save_best[0]
    maximize = args.save_best[1] == "max"
    best_metric = -float("inf") if maximize else float("inf")
    seed = args.seed

    if resume_dir is not None:
        # Restore state
        state = load_checkpoint(resume_dir, model=model, optimizer_mngr=optimizer_mngr)
        start_epoch = state["epoch"] + 1
        seed = state["seed"]
        train_info = state["stats"]
        best_metric = train_info.get(f"best_{metric}") or train_info.get(metric)
        if best_metric is None:
            raise ArgumentError(
                f"Could not recover best model using metric {metric}: invalid --save-best option? "
                f"Candidates are {list(train_info.keys())}"
            )
        # Remap optimizer to device if necessary
        optimizer_mngr.to(device)

    epoch_select = range(start_epoch, num_epochs)
    if epilogue_only:
        logger.warning(f"{'=' * 20} GOING STRAIGHT TO EPILOGUE {'=' * 20}")
        epoch_select = []
    elif sanity_check_only:
        logger.warning(f"{'='*20} SANITY CHECK MODE: THE TRAINING WILL NOT BE FULLY PERFORMED {'='*20}")
        # Only perform one epoch per training period
        epoch_select = [optimizer_mngr.periods[p_name]["epoch_range"][0] for p_name in optimizer_mngr.periods]
        epoch_select = (
            [epoch for epoch in epoch_select if epoch >= start_epoch]  # Select epochs after start_epoch (if any)
            if start_epoch <= epoch_select[-1]
            else [start_epoch]
        )

    training_loop(
        working_dir=root_dir,
        model=model,
        epoch_range=tqdm(
            epoch_select,
            initial=start_epoch,
            total=num_epochs,
            leave=False,
            desc="Training epochs",
            disable=not verbose,
        ),
        dataloaders=dataloaders,
        optimizer_mngr=optimizer_mngr,
        metric=metric,
        maximize=maximize,
        best_metric=best_metric,
        num_epochs=num_epochs,
        save_final=True,
        checkpoint_frequency=args.checkpoint_frequency,
        model_arch=model_arch,
        training_config=training_config,
        dataset_config=dataset_config,
        resume_dir=resume_dir,
        seed=seed,
        device=device,
        verbose=verbose,
    )
