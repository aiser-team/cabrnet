"""Declare the necessary functions to create an app to train a CaBRNet classifier."""

import os
from argparse import ArgumentParser, Namespace
from loguru import logger
from tqdm import tqdm
from cabrnet.generic.model import CaBRNet
from cabrnet.utils.optimizers import OptimizerManager
from cabrnet.utils.data import DatasetManager
from cabrnet.utils.parser import load_config
from cabrnet.utils.save import save_checkpoint, load_checkpoint
from cabrnet.visualization.visualizer import SimilarityVisualizer
from cabrnet.utils.exceptions import ArgumentError

description = "train a CaBRNet classifier"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """Create the argument parser for training a CaBRNet classifier.

    Returns:
        The parser itself.
    """
    if parser is None:
        parser = ArgumentParser(description)
    parser = CaBRNet.create_parser(parser, skip_state_dict=True)
    parser = DatasetManager.create_parser(parser)
    parser = OptimizerManager.create_parser(parser)
    parser = SimilarityVisualizer.create_parser(parser)
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
        required=True,
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
        "(alternative to --model-config, --dataset, --training and --visualization)",
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
        "--sanity-check-only",
        action="store_true",
        help="check the training pipeline without performing the entire process.",
    )
    parser.add_argument(
        "--epilogue-only",
        action="store_true",
        help="skip training and go to epilogue.",
    )
    return parser


def check_args(args: Namespace) -> Namespace:
    """Checks for three possible modes:

    - "--resume-from": in this case, ignore options --model-config, --dataset, --training and --visualization
    - "--config-dir": in this case, ignore options --model-config, --dataset, --training and --visualization
    - "--model-config", "--dataset", "--training" and "--visualization": in this case, all options are mandatory
    """
    for dir_path, option_name in zip([args.resume_from, args.config_dir], ["--resume-from", "--config-dir"]):
        if dir_path is not None:
            for param, name in zip(
                [args.model_config, args.dataset, args.training, args.visualization],
                ["--model-config", "--dataset", "--training", "--visualization"],
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
            args.visualization = os.path.join(dir_path, SimilarityVisualizer.DEFAULT_VISUALIZATION_CONFIG)

    for param, name, option in zip(
        [args.model_config, args.dataset, args.training, args.visualization],
        ["model", "dataset", "training", "visualization"],
        ["-m", "-d", "-t", "-z"],
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} configuration file (option {option}).")

    return args


def metrics_to_str(metrics: dict[str, float]) -> str:
    """Controls number of digits when showing batch statistics"""
    return ", ".join([f"{key}: {value:.3f}" for key, value in metrics.items()])


def execute(args: Namespace) -> None:
    """Create a CaBRNet classifier, then train it.

    Args:
        args: Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    # Recover options
    verbose = args.verbose
    device = args.device
    training_config = args.training
    model_config = args.model_config
    dataset_config = args.dataset

    model: CaBRNet = CaBRNet.build_from_config(config_file=model_config, seed=args.seed)

    # Training configuration
    trainer = load_config(training_config)
    model.register_training_params(trainer)  # Register auxiliary training parameters directly into model
    root_dir = args.output_dir

    # Check that output directory is available
    if not args.epilogue_only and not args.overwrite and os.path.exists(os.path.join(root_dir, "best")):
        raise ArgumentError(
            f"Output directory {os.path.join(root_dir, 'best')} is not empty. "
            f"To overwrite existing results, use --overwrite option."
        )

    # Build optimizer manager
    optimizer_mngr = OptimizerManager.build_from_config(config_file=training_config, model=model)
    # Dataloaders
    dataloaders = DatasetManager.get_dataloaders(config_file=dataset_config)

    if args.resume_from is not None:
        # Restore state
        state = load_checkpoint(args.resume_from, model=model, optimizer_mngr=optimizer_mngr)
        start_epoch = state["epoch"] + 1
        seed = state["seed"]
        train_info = state["stats"]
        best_metric = train_info.get(f"best_avg_train_{args.save_best}")
        if best_metric is None:
            raise ArgumentError(f"Could not recover best model {args.save_best}: invalid --save-best option?")
        # Remap optimizer to device if necessary
        optimizer_mngr.to(device)
    else:
        # Start from beginning
        start_epoch = 0
        best_metric = 0.0 if args.save_best == "acc" else float("inf")
        seed = args.seed

    if args.epilogue_only:
        logger.warning(f"{'=' * 20} GOING STRAIGHT TO EPILOGUE {'=' * 20}")
        epoch_select = [None]
    elif args.sanity_check_only:
        logger.warning(f"{'='*20} SANITY CHECK MODE: THE TRAINING WILL NOT BE FULLY PERFORMED {'='*20}")
        max_batches = 5  # Process only a few batches
        # Only perform one epoch per training period
        epoch_select = [optimizer_mngr.periods[p_name]["epoch_range"][0] for p_name in optimizer_mngr.periods]
    else:
        # Process all data batches and all epochs
        max_batches = None
        epoch_select = None

    num_epochs = trainer["num_epochs"]
    for epoch in tqdm(
        range(start_epoch, num_epochs),
        initial=start_epoch,
        total=num_epochs,
        leave=False,
        desc="Training epochs",
        disable=not verbose,
    ):
        if epoch_select is not None and epoch not in epoch_select:
            # Quietly skip epoch (sanity check mode)
            continue

        # Freeze parameters if necessary depending on current epoch and parameter group
        optimizer_mngr.freeze(epoch=epoch)
        train_info = model.train_epoch(
            dataloaders=dataloaders,
            optimizer_mngr=optimizer_mngr,
            device=device,
            progress_bar_position=1,
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

        if save_best_checkpoint:
            logger.info(f"Better model found at epoch {epoch}. Metrics: {metrics_to_str(train_info)}")
            save_checkpoint(
                directory_path=os.path.join(root_dir, "best"),
                model=model,
                model_config=model_config,
                optimizer_mngr=optimizer_mngr,
                training_config=training_config,
                dataset_config=dataset_config,
                visualization_config=args.visualization,
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
                visualization_config=args.visualization,
                epoch=epoch,
                seed=seed,
                device=device,
                stats=train_info,
            )

    # Load best model
    load_checkpoint(directory_path=os.path.join(root_dir, "best"), model=model)

    # Call epilogue
    visualizer = SimilarityVisualizer.build_from_config(config_file=args.visualization, model=model)
    model.epilogue(
        dataloaders=dataloaders,
        visualizer=visualizer,
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
        directory_path=os.path.join(root_dir, "final"),
        model=model,
        model_config=model_config,
        optimizer_mngr=optimizer_mngr,
        training_config=training_config,
        dataset_config=dataset_config,
        visualization_config=args.visualization,
        epoch=num_epochs,
        seed=seed,
        device=device,
        stats=eval_info,
    )
