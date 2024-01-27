"""Declare the necessary functions to create an app to train a CaBRNet classifier."""

import os
import sys
from argparse import ArgumentParser, Namespace
from loguru import logger
from tqdm import tqdm
from cabrnet.generic.model import ProtoClassifier
from cabrnet.utils.data import create_dataset_parser, get_dataloaders
from cabrnet.utils.parser import (
    get_optimizer,
    get_scheduler,
    get_param_groups,
    load_config,
    freeze,
    create_training_parser,
)
from cabrnet.utils.save import save_checkpoint, load_checkpoint
from cabrnet.visualisation.visualizer import SimilarityVisualizer

description = "training a CaBRNet classifier"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """Create the argument parser for training a CaBRNet classifier.

    Returns:
        The parser itself.
    """
    if parser is None:
        parser = ArgumentParser(description)
    parser = ProtoClassifier.create_parser(parser)
    parser = create_dataset_parser(parser)
    parser = create_training_parser(parser)
    parser = SimilarityVisualizer.create_parser(parser)
    return parser


def execute(args: Namespace) -> None:
    """Create a CaBRNet classifier, then train it.

    Args:
        args: Parsed arguments.

    """
    # Set logger level
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO"}])

    # Recover common options
    verbose = args.verbose
    device = args.device

    model: ProtoClassifier = ProtoClassifier.build_from_config(
        config_file=args.model_config, seed=args.seed, state_dict_path=args.model_state_dict
    )

    # Training configuration
    trainer = load_config(args.training)
    root_dir = args.training_dir
    param_groups = get_param_groups(trainer, model)
    optimizer = get_optimizer(trainer, param_groups)
    scheduler = get_scheduler(trainer, optimizer)
    # Dataloaders
    dataloaders = get_dataloaders(config_file=args.dataset)

    num_epochs = trainer["num_epochs"]
    best_metric = 0.0 if args.save_best == "acc" else float("inf")
    for epoch in tqdm(range(num_epochs), total=num_epochs, leave=False, desc="Training epochs"):
        # Freeze parameters if necessary depending on current epoch and parameter group
        freeze(epoch=epoch, param_groups=param_groups, trainer=trainer)
        train_info = model.train_epoch(
            train_loader=dataloaders["train_set"],
            optimizer=optimizer,
            device=device,
            progress_bar_position=1,
            epoch_idx=epoch,
            verbose=verbose,
        )
        # Apply scheduler
        if scheduler is not None:
            scheduler.step()

        if args.save_best == "acc" and best_metric < train_info["avg_train_accuracy"]:
            best_metric = train_info["avg_train_accuracy"]
            save_checkpoint(
                directory_path=os.path.join(root_dir, "best"),
                model=model,
                model_config=args.model_config,
                optimizer=optimizer,
                scheduler=scheduler,
                training_config=args.training,
                dataset_config=args.dataset,
                epoch=epoch,
                seed=args.seed,
                device=device,
                stats=train_info,
            )

    # Load best model
    model = load_checkpoint(directory_path=os.path.join(root_dir, "best"))["model"]

    # Call epilogue
    if trainer.get("epilogue") is not None:
        model.epilogue(**trainer.get("epilogue"))  # type: ignore

    # Perform projection
    projection_info = model.project(data_loader=dataloaders["projection_set"], device=device, verbose=verbose)

    # Extract prototypes
    visualizer = SimilarityVisualizer.build_from_config(config_file=args.visualization, target="prototype")
    model.extract_prototypes(
        dataloader_raw=dataloaders["projection_set_raw"],
        dataloader=dataloaders["projection_set"],
        projection_info=projection_info,
        visualizer=visualizer,
        dir_path=os.path.join(root_dir, "prototypes"),
        device=device,
        verbose=verbose,
    )

    # Evaluate model
    eval_info = model.evaluate(dataloader=dataloaders["test_set"], device=device, verbose=verbose)
    logger.info(f"Average loss: {eval_info['avg_loss']:.2f}. Average accuracy: {eval_info['avg_eval_accuracy']:.2f}.")
