import copy
import os
import random
import shutil
from argparse import ArgumentParser, Namespace
from typing import Any

import numpy as np
import torch
from loguru import logger
from ray import train, tune
from ray.tune import Trainable
from ray.tune.search.optuna import OptunaSearch

from cabrnet.apps.train import training_loop
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.exceptions import ArgumentError
from cabrnet.core.utils.optimizers import OptimizerManager
from cabrnet.core.utils.parser import load_config

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
        required=False,
        nargs=2,
        default=["avg_accuracy", "max"],
        metavar=("metric", "min/max"),
        help="during training, save best model based on chosen metric and mode (min or max)",
    )
    parser.add_argument(
        "-s",
        "--search-space",
        type=str,
        nargs=4,
        required=True,
        metavar=("metric", "min/max", "path/to/search_space.yml", "num_trials"),
        help="optimize hyperparameters based on chosen metric",
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        required=False,
        default=-1,
        metavar="num_epochs",
        help="stop training if not better model was found during the last X epochs",
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
        "(alternative to --model-arch, --dataset and --training)",
    )
    parser.add_argument(
        "-r",
        "--resume-from",
        type=str,
        metavar="/path/to/working/directory",
        help="path to existing working directory to resume optimization",
    )
    parser.add_argument(
        "-n",
        "--num-resources-per-trial",
        type=int,
        required=False,
        default=1,
        metavar="val",
        help="number of hardware resources allocated to each trial",
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
        "--cleanup",
        action="store_true",
        help="clean working directories after each trial.",
    )
    return parser


def check_args(args: Namespace) -> Namespace:
    r"""Checks the validity of the arguments and updates the namespace if necessary.

    Args:
        args (Namespace): Parsed arguments.

    Returns:
        Modified argument namespace.
    """
    # Check environment variable
    assert (
        os.environ.get("RAY_CHDIR_TO_TRIAL_DIR") == "0"
    ), "Environment variable RAY_CHDIR_TO_TRIAL_DIR should be set to 0"

    if args.config_dir is not None:
        for param, name in zip(
            [args.model_arch, args.dataset, args.training], ["--model-arch", "--dataset", "--training"]
        ):
            if param is not None:
                raise ArgumentError(f"Cannot specify both options {name} and --config-dir")
        args.model_arch = os.path.join(args.config_dir, CaBRNet.DEFAULT_MODEL_CONFIG)
        # Compatibility with v0.1: will be removed in the future
        if not os.path.isfile(args.model_arch) and os.path.isfile(os.path.join(args.config_dir, "model.yml")):
            args.model_arch = os.path.join(args.config_dir, "model.yml")
            logger.warning(
                f"Using model.yml from {args.config_dir}: "
                f"please consider renaming the file to {CaBRNet.DEFAULT_MODEL_CONFIG} to ensure compatibility "
                f"with future versions"
            )
        args.dataset = os.path.join(args.config_dir, DatasetManager.DEFAULT_DATASET_CONFIG)
        args.training = os.path.join(args.config_dir, OptimizerManager.DEFAULT_TRAINING_CONFIG)

    for param, name, option in zip(
        [args.model_arch, args.dataset, args.training], ["model", "dataset", "training"], ["-m", "-d", "-t"]
    ):
        if param is None:
            raise ArgumentError(f"Missing {name} configuration file (option {option}).")

    # Check optimization mode
    if args.save_best[1] not in ["min", "max"]:
        raise ArgumentError(f"Invalid optimization mode '{args.save_best[1]}' in --save-best")
    if args.search_space[1] not in ["min", "max"]:
        raise ArgumentError(f"Invalid optimization mode '{args.search_space[1]}' in --search-space")

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
    model_arch_file = os.path.abspath(args.model_arch)
    dataset_config_file = os.path.abspath(args.dataset)
    root_dir = os.path.abspath(args.output_dir)
    sanity_check_only = args.sanity_check
    seed = args.seed

    # Recover metrics
    train_metric = args.save_best[0]
    train_maximize = args.save_best[1] == "max"
    hp_metric = args.search_space[0]
    hp_mode = args.search_space[1]

    # Load initial configuration
    initial_training_config = load_config(training_config_file)
    initial_model_arch = load_config(model_arch_file)
    initial_dataset_config = load_config(dataset_config_file)

    class CaBRNetTrainable(Trainable):
        model_arch: dict[str, Any]
        training_config: dict[str, Any]
        dataset_config: dict[str, Any]
        working_dir: str

        def setup(self, config: dict[str, Any]):
            r"""Sets up the configuration of a trial.

            Args:
                config (dict): Dictionary containing the hyperparameters that will be tweaked.
            """
            # Reset RNG states
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # Update configuration
            self.training_config = update_configuration(
                copy.deepcopy(initial_training_config), config.get("training", {})
            )
            self.model_arch = update_configuration(copy.deepcopy(initial_model_arch), config.get("model", {}))
            self.dataset_config = update_configuration(copy.deepcopy(initial_dataset_config), config.get("dataset", {}))
            self.working_dir = os.path.join(root_dir, f"trial_{self.trial_name}")

        def step(self) -> dict[str, Any]:
            r"""Returns the statistics for this trial."""
            # Build model
            model = CaBRNet.build_from_config(config=self.model_arch, seed=seed)
            # Register auxiliary training parameters directly into model
            model.register_training_params(self.training_config)

            # Build optimizer manager
            optimizer_mngr = OptimizerManager.build_from_config(config=self.training_config, model=model)
            # Dataloaders
            dataloaders = DatasetManager.get_dataloaders(
                config=self.dataset_config, sampling_ratio=100 if sanity_check_only else 1
            )
            num_epochs = self.training_config["num_epochs"]

            eval_info = training_loop(
                working_dir=self.working_dir,
                model=model,
                epoch_range=range(0, num_epochs),
                dataloaders=dataloaders,
                optimizer_mngr=optimizer_mngr,
                metric=train_metric,
                maximize=train_maximize,
                best_metric=(0.0 if train_maximize else float("inf")),
                num_epochs=num_epochs,
                patience=args.patience,
                save_final=True,
                model_arch=self.model_arch,
                training_config=self.training_config,
                dataset_config=self.dataset_config,
                seed=seed,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                logger_level="INFO",
            )
            return eval_info

        def cleanup(self):
            if args.cleanup:
                logger.info(f"Cleaning up working directory {self.working_dir}")
                # Remove working directory
                shutil.rmtree(self.working_dir)

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
    search_space = tune_search_space(load_config(args.search_space[2]))

    search_alg = OptunaSearch()
    num_trials = int(args.search_space[3])

    # Set-up hardware resources
    resources = {"cpu" if device == "cpu" else "gpu": args.num_resources_per_trial}
    trainable_with_resources = (
        tune.with_resources(CaBRNetTrainable, resources) if args.num_resources_per_trial > 0 else CaBRNetTrainable
    )

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
                metric=hp_metric,
                mode=hp_mode,
                search_alg=search_alg,
                num_samples=num_trials,
            ),
            run_config=train.RunConfig(
                storage_path=root_dir,
                name="test_experiment",
                stop={"training_iteration": 1},  # Each trial configuration is only tested once
                checkpoint_config=train.CheckpointConfig(
                    # Checkpoint management is handled by CaBRNet directly
                    checkpoint_at_end=False
                ),
            ),
            param_space=search_space,
        )

    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
