import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import random
from cabrnet.generic.model import CaBRNet
from cabrnet.utils.data import DatasetManager
from cabrnet.utils.exceptions import ArgumentError
from cabrnet.utils.optimizers import OptimizerManager
from cabrnet.utils.parser import load_config
from cabrnet.utils.save import load_checkpoint, save_checkpoint
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from loguru import logger
from typing import Any

description = "performs hyperparameter tuning on a CaBRNet model using Bayesian Optimization"


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
        nargs=4,
        required=True,
        metavar=("metric", "min/max", "path/to/search_space.yml", "num_steps"),
        help="save best hyperparameters based on chosen metric",
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
        "-c",
        "--config-dir",
        type=str,
        required=False,
        metavar="/path/to/config/dir",
        help="path to directory containing all configuration files to start training "
        "(alternative to --model-config, --dataset and --training)",
    )
    parser.add_argument(
        "-r",
        "--resume-from",
        type=str,
        metavar="/path/to/working/directory",
        help="path to existing working directory to resume optimization",
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
    return parser


def check_args(args: Namespace) -> Namespace:
    r"""Checks the validity of the arguments and updates the namespace if necessary.

    Args:
        args (Namespace): Parsed arguments.

    Returns:
        Modified argument namespace.
    """
    if args.config_dir is not None:
        for param, name in zip(
            [args.model_config, args.dataset, args.training], ["--model-config", "--dataset", "--training"]
        ):
            if param is not None:
                raise ArgumentError(f"Cannot specify both options {name} and --config-dir")
        args.model_config = os.path.join(args.config_dir, CaBRNet.DEFAULT_MODEL_CONFIG)
        # Compatibility with v0.1: will be removed in the future
        if not os.path.isfile(args.model_config) and os.path.isfile(os.path.join(args.config_dir, "model.yml")):
            args.model_config = os.path.join(args.config_dir, "model.yml")
            logger.warning(
                f"Using model.yml from {args.config_dir}: "
                f"please consider renaming the file to {CaBRNet.DEFAULT_MODEL_CONFIG} to ensure compatibility "
                f"with future versions"
            )
        args.dataset = os.path.join(args.config_dir, DatasetManager.DEFAULT_DATASET_CONFIG)
        args.training = os.path.join(args.config_dir, OptimizerManager.DEFAULT_TRAINING_CONFIG)

    for param, name, option in zip(
        [args.model_config, args.dataset, args.training], ["model", "dataset", "training"], ["-m", "-d", "-t"]
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} configuration file (option {option}).")

    # Check save-best option
    if args.save_best[1] not in ["min", "max"]:
        raise ArgumentError(f"Unknown optimization directive: {args.save_best[1]}")

    if os.path.exists(args.output_dir) and args.resume_from is None and not args.overwrite:
        raise ArgumentError(
            f"Output directory {args.output_dir} is not empty. "
            f"To overwrite existing results, use --overwrite option."
        )

    return args


def update_configuration(config: dict[str, Any], change_dict: dict[str, Any]) -> dict[str, Any]:
    r"""Recursively updates the keys of a configuration dictionary.

    Args:
        config (dict): Source configuration.
        change_dict (dict): Set of changes to apply.

    Returns:
        Modified configuration.
    """
    for key, value in change_dict.items():
        if key not in config.keys():
            raise ValueError(f"Unknown hyperparameter {key}")
        if isinstance(value, dict):
            config[key] = update_configuration(config.get(key, {}), value)
        else:
            config[key] = value
    return config


def execute(args: Namespace) -> None:
    r"""Creates a CaBRNet model, then trains it.

    Args:
        args (Namespace): Parsed arguments.

    """
    # Check and post-process options
    args = check_args(args)

    # Recover options
    device = args.device

    # Absolute paths are necessary because Optuna launches jobs from another directory
    training_config_file = os.path.abspath(args.training)
    model_config_file = os.path.abspath(args.model_config)
    dataset_config_file = os.path.abspath(args.dataset)
    root_dir = os.path.abspath(args.output_dir)
    sanity_check_only = args.sanity_check
    seed = args.seed

    save_min = args.save_best[1] == "min"
    metric = args.save_best[0]

    # Load initial configuration
    training_config = load_config(training_config_file)
    model_config = load_config(model_config_file)
    dataset_config = load_config(dataset_config_file)

    def evaluate_configuration(config: dict[str, dict]) -> None:
        r"""Evaluates a given hyperparameter configuration.

        Args:
            config (dict): Hyperparameter configuration to be tested.
        """
        nonlocal seed, training_config, model_config, dataset_config

        # Reset RNG states
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Update configuration
        training_config = update_configuration(training_config, config.get("training", {}))
        model_config = update_configuration(model_config, config.get("model", {}))
        dataset_config = update_configuration(dataset_config, config.get("dataset", {}))

        # Build model
        model = CaBRNet.build_from_config(config=model_config, seed=args.seed)
        # Register auxiliary training parameters directly into model
        model.register_training_params(training_config)

        # Build optimizer manager
        optimizer_mngr = OptimizerManager.build_from_config(config=training_config, model=model)
        # Dataloaders
        dataloaders = DatasetManager.get_dataloaders(
            config=dataset_config, sampling_ratio=100 if sanity_check_only else 1
        )

        num_epochs = training_config["num_epochs"]
        best_metric = float("inf") if save_min else 0.0

        for epoch in range(0, num_epochs):
            # Freeze parameters if necessary depending on current epoch and parameter group
            optimizer_mngr.freeze(epoch=epoch)
            train_info = model.train_epoch(
                dataloaders=dataloaders,
                optimizer_mngr=optimizer_mngr,
                device=device,
                epoch_idx=epoch,
            )
            # Apply scheduler
            optimizer_mngr.scheduler_step(epoch=epoch)

            save_best_checkpoint = False
            if (save_min and best_metric > train_info[metric]) or (not save_min and best_metric < train_info[metric]):
                best_metric = train_info[metric]
                save_best_checkpoint = True

            # Add information regarding current best metric
            train_info[f"best_avg_train_{metric}"] = best_metric
            if save_best_checkpoint:
                save_checkpoint(
                    directory_path=os.path.join(root_dir, "best"),
                    model=model,
                    model_config=model_config_file,
                    optimizer_mngr=optimizer_mngr,
                    training_config=training_config_file,
                    dataset_config=dataset_config_file,
                    projection_info=None,
                    epoch=epoch,
                    seed=seed,
                    device=device,
                    stats=train_info,
                )

        # Load best model and call epilogue
        load_checkpoint(directory_path=os.path.join(root_dir, "best"), model=model, optimizer_mngr=optimizer_mngr)
        model.epilogue(
            dataloaders=dataloaders,
            optimizer_mngr=optimizer_mngr,
            output_dir=root_dir,
            device=device,
            **training_config.get("epilogue", {}),
        )

        # Evaluate model
        eval_info = model.evaluate(dataloader=dataloaders["test_set"], device=device)
        train.report(eval_info)

    def tune_search_space(config: dict):
        r"""Recursively update lists of possible parameters into tune format"""
        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = tune_search_space(config.get(key, {}))
            elif isinstance(value, list):
                config[key] = tune.choice(value)
            else:
                raise ValueError(f"Unsupported type in search space definition: {config[key]}")
        return config

    # Load search space
    search_space = tune_search_space(load_config(args.save_best[2]))

    search_alg = OptunaSearch()
    mode = args.save_best[1]
    num_steps = int(args.save_best[3])

    # Set-up hardware resources
    trainable_with_resources = tune.with_resources(evaluate_configuration, {"gpu": 1})

    if args.resume_from is not None:
        # Resume failed/interrupted experiment
        resume_path = os.path.abspath(args.resume_from)
        tuner = tune.Tuner.restore(
            path=resume_path,
            restart_errored=True,
            resume_unfinished=True,
            trainable=trainable_with_resources,
            param_space=search_space,
        )
    else:
        tuner = tune.Tuner(
            trainable=trainable_with_resources,
            tune_config=tune.TuneConfig(
                metric=metric,
                mode=mode,
                search_alg=search_alg,
                num_samples=num_steps,
            ),
            run_config=train.RunConfig(storage_path=root_dir, name="test_experiment"),
            param_space=search_space,
        )

    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
