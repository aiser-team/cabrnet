import os
from loguru import logger
from cabrnet.generic.model import CaBRNet
from cabrnet.utils.parser import load_config
from cabrnet.utils.optimizers import create_training_parser
from cabrnet.utils.data import create_dataset_parser, get_dataloaders
from cabrnet.utils.save import save_checkpoint
from cabrnet.visualisation.visualizer import SimilarityVisualizer
from argparse import ArgumentParser, Namespace


description = "convert an existing legacy model into a CaBRNet version"


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser(description)
    parser = CaBRNet.create_parser(parser)
    parser = create_training_parser(parser)
    parser = create_dataset_parser(parser)
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
                logger.warning(f"Ignoring option {name}: using content pointed by --config-dir instead")
        args.model_config = os.path.join(args.config_dir, "model_arch.yml")
        args.dataset = os.path.join(args.config_dir, "dataset.yml")
        args.training = os.path.join(args.config_dir, "training.yml")
        args.visualization = os.path.join(args.config_dir, "visualization.yml")

    # Check configuration completeness
    for param, name in zip(
        [args.model_config, args.dataset, args.training, args.visualization, args.model_state_dict],
        [
            "model configuration",
            "dataset configuration",
            "training configuration",
            "visualization configuration",
            "model state",
        ],
    ):
        if param is None:
            raise AttributeError(f"Missing {name} file.")
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

    dataloaders = get_dataloaders(dataset_config)

    # Call epilogue
    trainer = load_config(training_config)
    visualizer = SimilarityVisualizer.build_from_config(config_file=args.visualization)
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
