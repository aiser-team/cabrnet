import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

from cabrnet.generic.model import CaBRNet
from cabrnet.utils.data import DatasetManager
from cabrnet.utils.exceptions import ArgumentError
from cabrnet.utils.optimizers import OptimizerManager
from cabrnet.utils.parser import load_config
from cabrnet.utils.save import load_checkpoint, save_checkpoint
from loguru import logger
from tqdm import tqdm

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
        choices=["acc", "loss"],
        default="acc",
        metavar="metric",
        help="save best model based on accuracy or loss",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=False,
        metavar="path/to/output/directory",
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
        "(alternative to --model-config, --dataset and --training)",
    )
    x_group.add_argument(
        "-r",
        "--resume-from",
        type=str,
        metavar="/path/to/checkpoint/directory",
        help="path to existing checkpoint directory to resume training",
    )
    parser.add_argument(
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


def get_parent_directory(dir_path: str):
    r"""Returns the parent directory of *dir_path*.

    Args:
        dir_path (str): Path to directory.

    Returns:
        Absolute path to parent directory.
    """
    return Path(dir_path).parent.absolute()


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
                [args.model_config, args.dataset, args.training], ["--model-config", "--dataset", "--training"]
            ):
                if param is not None:
                    raise ArgumentError(f"Cannot specify both options {name} and {option_name}")
            args.model_config = os.path.join(dir_path, CaBRNet.DEFAULT_MODEL_CONFIG)
            # Compatibility with v0.1: will be removed in the future
            if not os.path.isfile(args.model_config) and os.path.isfile(os.path.join(dir_path, "model.yml")):
                args.model_config = os.path.join(dir_path, "model.yml")
                logger.warning(
                    f"Using model.yml from {dir_path}: "
                    f"please consider renaming the file to {CaBRNet.DEFAULT_MODEL_CONFIG} to ensure compatibility "
                    f"with future versions"
                )
            args.dataset = os.path.join(dir_path, DatasetManager.DEFAULT_DATASET_CONFIG)
            args.training = os.path.join(dir_path, OptimizerManager.DEFAULT_TRAINING_CONFIG)

    for param, name, option in zip(
        [args.model_config, args.dataset, args.training], ["model", "dataset", "training"], ["-m", "-d", "-t"]
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} configuration file (option {option}).")

    # In resume mode, specifying the output directory is optional
    if args.output_dir is None and args.resume_from is not None:
        args.output_dir = get_parent_directory(args.resume_from)
        logger.warning(f"Using {args.output_dir} as default output directory based on checkpoint path")
    if args.output_dir is None:
        ArgumentError("Missing path to output directory (option --output-dir)")

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

    return args


def metrics_to_str(metrics: dict[str, float]) -> str:
    r"""Converts a dictionary of metrics into a readable string.

    Args:
        metrics (dictionary): Dictionary of batch metrics.

    Returns:
        Readable string representing batch statistics.
    """
    return ", ".join([f"{key}: {value:.3f}" for key, value in metrics.items()])


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
    model_config = args.model_config
    dataset_config = args.dataset
    epilogue_only = args.epilogue
    sanity_check_only = args.sanity_check
    resume_dir = args.resume_from

    model: CaBRNet = CaBRNet.build_from_config(config_file=model_config, seed=args.seed)

    # Training configuration
    trainer = load_config(training_config)
    model.register_training_params(trainer)  # Register auxiliary training parameters directly into model
    root_dir = args.output_dir

    # Build optimizer manager
    optimizer_mngr = OptimizerManager.build_from_config(config_file=training_config, model=model)
    # Dataloaders
    dataloaders = DatasetManager.get_dataloaders(config_file=dataset_config)

    # By default, process all data batches and all epochs
    max_batches = None
    start_epoch = 0
    num_epochs = trainer["num_epochs"]
    best_metric = 0.0 if args.save_best == "acc" else float("inf")
    seed = args.seed

    if resume_dir is not None:
        # Restore state
        state = load_checkpoint(resume_dir, model=model, optimizer_mngr=optimizer_mngr)
        start_epoch = state["epoch"] + 1
        seed = state["seed"]
        train_info = state["stats"]
        best_metric = train_info.get(f"best_avg_train_{args.save_best}")
        if best_metric is None:
            raise ArgumentError(f"Could not recover best model {args.save_best}: invalid --save-best option?")
        # Remap optimizer to device if necessary
        optimizer_mngr.to(device)

    epoch_select = range(start_epoch, num_epochs)
    if epilogue_only:
        logger.warning(f"{'=' * 20} GOING STRAIGHT TO EPILOGUE {'=' * 20}")
        epoch_select = []
    elif sanity_check_only:
        logger.warning(f"{'='*20} SANITY CHECK MODE: THE TRAINING WILL NOT BE FULLY PERFORMED {'='*20}")
        max_batches = 5  # Process only a few batches
        # Only perform one epoch per training period
        epoch_select = [optimizer_mngr.periods[p_name]["epoch_range"][0] for p_name in optimizer_mngr.periods]
        epoch_select = (
            [epoch for epoch in epoch_select if epoch >= start_epoch]  # Select epochs after start_epoch (if any)
            if start_epoch <= epoch_select[-1]
            else [start_epoch]
        )

    for epoch in tqdm(
        epoch_select,
        initial=start_epoch,
        total=num_epochs,
        leave=False,
        desc="Training epochs",
        disable=not verbose,
    ):
        # Freeze parameters if necessary depending on current epoch and parameter group
        optimizer_mngr.freeze(epoch=epoch)
        train_info = model.train_epoch(
            dataloaders=dataloaders,
            optimizer_mngr=optimizer_mngr,
            device=device,
            tqdm_position=1,
            epoch_idx=epoch,
            max_batches=max_batches,
            verbose=verbose,
        )
        # Apply scheduler
        optimizer_mngr.scheduler_step(epoch=epoch)

        save_best_checkpoint = False
        if args.save_best == "acc" and best_metric < train_info["avg_train_accuracy"]:
            best_metric = train_info["avg_train_accuracy"]
            save_best_checkpoint = True
        elif args.save_best == "loss" and best_metric > train_info["avg_loss"]:
            best_metric = train_info["avg_loss"]
            save_best_checkpoint = True
        # Add information regarding current best metric
        train_info[f"best_avg_train_{args.save_best}"] = best_metric
        logger.info(f"Metrics at epoch {epoch}: {metrics_to_str(train_info)}")
        if save_best_checkpoint:
            logger.success(f"Better model found at epoch {epoch}. Saving checkpoint.")
            save_checkpoint(
                directory_path=os.path.join(root_dir, "best"),
                model=model,
                model_config=model_config,
                optimizer_mngr=optimizer_mngr,
                training_config=training_config,
                dataset_config=dataset_config,
                projection_info=None,
                epoch=epoch,
                seed=seed,
                device=device,
                stats=train_info,
            )
        if args.checkpoint_frequency is not None and (epoch % args.checkpoint_frequency == 0):
            save_checkpoint(
                directory_path=os.path.join(root_dir, f"epoch_{epoch}"),
                model=model,
                model_config=model_config,
                optimizer_mngr=optimizer_mngr,
                training_config=training_config,
                dataset_config=dataset_config,
                projection_info=None,
                epoch=epoch,
                seed=seed,
                device=device,
                stats=train_info,
            )

    if not epilogue_only:
        # Seek best model (in epilogue mode, keep the existing model state)
        if os.path.isdir(os.path.join(root_dir, "best")):
            path_to_best = os.path.join(root_dir, "best")
        elif resume_dir is not None and os.path.isdir(os.path.join(get_parent_directory(resume_dir), "best")):
            # Best checkpoint occurred before training was resumed
            path_to_best = os.path.join(get_parent_directory(resume_dir), "best")
        else:
            raise FileNotFoundError("Could not find path to best model. Aborting epilogue.")
        logger.info(f"Loading best model from checkpoint {path_to_best}")
        load_checkpoint(directory_path=path_to_best, model=model, optimizer_mngr=optimizer_mngr)

    # Call epilogue
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
    logger.info(f"Average loss: {eval_info['avg_loss']:.2f}. Average accuracy: {eval_info['avg_eval_accuracy']:.2f}.")
    save_checkpoint(
        directory_path=os.path.join(root_dir, "final"),
        model=model,
        model_config=model_config,
        optimizer_mngr=None,
        training_config=training_config,
        dataset_config=dataset_config,
        projection_info=projection_info,
        epoch=num_epochs,
        seed=seed,
        device=device,
        stats=eval_info,
    )
