import os
import torch
import torch.nn as nn
import cabrnet.prototree.decision
from cabrnet.generic.model import ProtoClassifier
from cabrnet.utils.parser import (
    load_config,
    get_param_groups,
    get_optimizer,
    get_scheduler,
    freeze,
    create_training_parser,
)
from cabrnet.utils.data import create_dataset_parser, get_dataloaders
from cabrnet.utils.save import save_checkpoint, load_checkpoint
from cabrnet.utils.monitoring import memory_logger
from cabrnet.utils.hacks import optimizer_to
from cabrnet.visualisation.visualizer import SimilarityVisualizer
import legacy.prototree.prototree.prototree as prototree_legacy
import legacy.prototree.prototree.train
import legacy.prototree.util.init
import legacy.prototree.util.analyse
import legacy.prototree.util.log
import legacy.prototree.util.args
import legacy.prototree.util.net
import legacy.prototree.util.data
import legacy.prototree.util.save
import legacy.prototree.prototree.prune
import legacy.prototree.prototree.project
import legacy.prototree.prototree.upsample
from typing import Any
from argparse import ArgumentParser, Namespace
from loguru import logger
import copy
import numpy as np
import random

description = "convert an existing ProtoTree into a CaBRNet version"


class DummyLogger:
    checkpoint_dir: str = ""

    def set_checkpoint_dir(self, dir: str):
        self.checkpoint_dir = dir

    def log_message(self, message: str):
        message = message.strip("\n")
        logger.info(f"[LEGACY] {message}")


dummy_logger = DummyLogger()


def create_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser(description)
    parser = ProtoClassifier.create_parser(parser)
    parser = create_dataset_parser(parser)
    parser = create_training_parser(parser)
    parser = SimilarityVisualizer.create_parser(parser)
    parser.add_argument(
        "--legacy-state-dict",
        "-l",
        required=False,
        type=str,
        metavar="/path/to/state/dict.pth",
        help="Path to the pre-trained legacy state dictionary",
    )
    return parser


def compare_inference(description: str, ref_model: nn.Module, test_model: nn.Module) -> None:
    """
    Compare two models based on inference
    Args:
        description: Test description for logger
        ref_model: Reference model
        test_model: Model under test
    """
    test_model.eval()
    ref_model.eval()

    x_ref = torch.rand(10, 3, 224, 224)
    x_test = x_ref.detach().clone()
    y_ref = ref_model(x_ref)[0]
    y_test = test_model(x_test)[0]

    if torch.all(torch.eq(y_ref, y_test)):
        logger.success(f"{description} successful.")
    else:
        logger.error(f"{description} failed.")


def _compare_generic(ref, test) -> bool:
    res = True
    if type(ref) != type(test):
        logger.error(f"Mismatching types: {type(ref)} v. {type(test)}.")
        res = False
    elif isinstance(ref, dict):
        for key in ref:
            if key not in test:
                logger.error(f"Reference key {key} not found in test state dict")
                res = False
            elif not _compare_generic(ref[key], test[key]):
                logger.error(f"Mismatching values for key {key}.")
                res = False
    elif isinstance(ref, list) or isinstance(ref, tuple):
        if len(ref) != len(test):
            logger.error(f"Mismatching list lengths: {len(ref)} v. {len(test)}.")
            res = False
        for r, t in zip(ref, test):
            res = res and _compare_generic(r, t)
    elif isinstance(ref, torch.Tensor):
        if ref.size() != test.size():
            logger.error(f"Mismatching tensor sizes: {ref.size()} v. {test.size()}")
            res = False
        elif not torch.all(torch.eq(ref, test)):
            logger.error(f"Mismatching tensors")
            res = False
    elif ref != test:
        logger.error(f"Mismatching values: {ref} v. {test}")
        res = False
    return res


def compare_generic(description: str, ref: Any, test: Any) -> bool:
    res = _compare_generic(ref, test)
    if res:
        logger.success(f"{description} successful")
    else:
        logger.error(f"{description} failed")
    return res


def compare_pruning(ref_model: nn.Module, test_model: nn.Module) -> None:
    """
    Compare two models based on pruning
    Args:
        ref_model: Reference model
        test_model: Model under test
    """
    test_model.analyse_leafs(pruning_threshold=0.01)
    test_model.prune(pruning_threshold=0.01)
    legacy.prototree.util.analyse.analyse_leafs(
        ref_model, epoch=0, k=200, leaf_labels={}, threshold=0.01, log=dummy_logger
    )
    legacy.prototree.prototree.prune.prune(ref_model, pruning_threshold_leaves=0.01, log=dummy_logger)
    compare_inference("Model pruning", ref_model=ref_model, test_model=test_model)


def compare_projection_info(ref_dict: dict, test_dict: dict):
    success = True
    for key in ref_dict:
        if test_dict[key]["img_idx"] == -1:
            continue
        if ref_dict[key]["input_image_ix"] != test_dict[key]["img_idx"]:
            logger.error(
                f"Mismatching image index for prototype {key}. "
                f"Expected {ref_dict[key]['input_image_ix']} but found {test_dict[key]['img_idx']}."
            )
            success = False
        W = ref_dict[key]["W"]
        if ref_dict[key]["patch_ix"] != test_dict[key]["h"] * W + test_dict[key]["w"]:
            logger.error(
                f"Mismatching patch coordinates for prototype {key}. "
                f"Expected ({ref_dict[key]['patch_ix']// W}, {ref_dict[key]['patch_ix'] % W}) "
                f"but found ({test_dict[key]['h']}, {test_dict[key]['w']})."
            )
            success = False
    if success:
        logger.success(f"Projection info comparison successful.")
    else:
        logger.error(f"Projection info comparison failed.")


def cabrnet_process(
    model_config: str,
    dataset_config: str,
    training_config: str,
    visualization_config: str,
    legacy_state_dict: str | None,
    seed: int,
    verbose: bool,
    root_directory: str,
    device: str,
) -> dict:
    """
    Builds, train, prune and perform projection on a cabrnet tree
    Args:
        model_config: Path to model configuration file
        dataset_config: Path to dataset configuration file
        training_config: Path to training configuration file
        visualization_config: Path to prototype visualization configuration file
        legacy_state_dict: Optional path to legacy state dictionary
        seed: Random seed
        verbose: Display progression bars
        root_directory: Output directory for prototypes
        device: target hardware device
    Returns:
        dictionary of system states
    """
    # Init random state
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Save system states at various stages
    system_state = {
        "model": None,
        "dataloader_test": None,
        "model_states": [],
        "scheduler_states": [],
        "optimizer_states": [],
    }

    # Build cabrnet tree with RNG resynchronisation and compatibility mode
    model: ProtoClassifier = ProtoClassifier.build_from_config(model_config, seed=seed, compatibility_mode=True)
    system_state["model"] = copy.deepcopy(model)
    # Load data
    dataloaders = get_dataloaders(config_file=dataset_config)
    for xs, ys in dataloaders["train_set"]:
        break
    system_state["dataloader_test"] = xs.cpu().clone(), ys.cpu().clone()
    # Initialize optimizer and LR scheduler
    trainer = load_config(training_config)
    param_groups = get_param_groups(trainer, model)
    optimizer = get_optimizer(trainer, param_groups)
    scheduler = get_scheduler(trainer, optimizer)
    system_state["optimizer_states"].append(copy.deepcopy(optimizer.state_dict()))
    system_state["scheduler_states"].append(copy.deepcopy(scheduler.state_dict()))

    # Training
    num_epochs = trainer["num_epochs"]
    for epoch in range(num_epochs):
        freeze(epoch=epoch, param_groups=param_groups, trainer=trainer)

        model.train_epoch(
            train_loader=dataloaders["train_set"],
            optimizer=optimizer,
            device=device,
            progress_bar_position=0,
            epoch_idx=epoch,
            max_batches=5,
            verbose=verbose,
        )
        # Update scheduler and leaf labels before saving checkpoints
        scheduler.step()
        # Move model and optimizer to CPU before saving, then move back to GPU
        model.to("cpu")
        optimizer_to(optimizer, "cpu")
        system_state["model_states"].append(copy.deepcopy(model.state_dict()))
        system_state["optimizer_states"].append(copy.deepcopy(optimizer.state_dict()))
        system_state["scheduler_states"].append(copy.deepcopy(scheduler.state_dict()))
        model.to(device)
        optimizer_to(optimizer, device)
    if legacy_state_dict is None:
        return system_state

    model.to("cpu")
    model.load_legacy_state_dict(torch.load(legacy_state_dict))
    model.eval()

    # Testing sampling strategies
    system_state["distributed"] = model(
        torch.randn((10, 3, 224, 224)), strategy=cabrnet.prototree.decision.SamplingStrategy.DISTRIBUTED
    )
    system_state["sample_max"] = model(
        torch.randn((10, 3, 224, 224)), strategy=cabrnet.prototree.decision.SamplingStrategy.SAMPLE_MAX
    )
    system_state["greedy"] = model(
        torch.randn((10, 3, 224, 224)), strategy=cabrnet.prototree.decision.SamplingStrategy.GREEDY
    )

    # Prune weak leaves
    model.prune(pruning_threshold=0.01)
    system_state["pruned"] = copy.deepcopy(model)

    # Project prototypes
    projection_info = model.project(data_loader=dataloaders["projection_set"], device=device, verbose=verbose)
    system_state["projected"] = copy.deepcopy(model.to("cpu"))
    system_state["projection_info"] = projection_info
    visualizer = SimilarityVisualizer.build_from_config(config_file=visualization_config, target="prototype")
    model.extract_prototypes(
        dataloader_raw=dataloaders["projection_set_raw"],
        dataloader=dataloaders["projection_set"],
        projection_info=projection_info,
        visualizer=visualizer,
        dir_path=os.path.join(root_directory, "prototypes"),
        device=device,
        verbose=verbose,
    )

    save_checkpoint(
        directory_path=os.path.join(root_directory, "model"),
        model=model,
        model_config=model_config,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=training_config,
        dataset_config=dataset_config,
        epoch=0,
        seed=seed,
        device=device,
        stats=None,
    )
    load_checkpoint(os.path.join(root_directory, "model"), model=model)

    return system_state


def legacy_process(
    model_config: str,
    dataset_config: str,
    training_config: str,
    visualization_config: str,
    legacy_state_dict: str | None,
    seed: int,
    root_directory: str,
    device: str,
) -> dict:
    """
    Builds, train, prune and perform projection on a ProtoTree
    Args:
        model_config: Path to model configuration file
        dataset_config: Path to dataset configuration file
        training_config: Path to training configuration file
        visualization_config: Path to prototype visualization configuration file
        legacy_state_dict: Optional path to legacy state dictionary
        seed: Random seed
        root_directory: Output directory for prototypes
        device: Target hardware device
    Returns:
        dictionary of system states
    """
    # Init random state
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Save system states at various stages
    system_state = {
        "model": None,
        "dataloader_test": None,
        "model_states": [],
        "scheduler_states": [],
        "optimizer_states": [],
    }

    # Configure DummyLogger and create root directory
    dummy_logger.set_checkpoint_dir(root_directory)
    os.makedirs(root_directory, exist_ok=True)

    # Load configurations and simulate ProtoTree args namespace
    config = load_config(model_config)
    args = {}
    # Build feature extractor
    extractor_config = config["extractor"]
    args["num_features"] = [
        extractor_config["add_on"][layer]["params"]["out_channels"]
        for layer in extractor_config["add_on"]
        if layer != "init_mode"
        and "params" in extractor_config["add_on"][layer]
        and "out_channels" in extractor_config["add_on"][layer]["params"]
    ][-1]
    arch = extractor_config["backbone"]["arch"]
    if arch == "resnet50" and "inat" in extractor_config["backbone"]["weights"]:
        # Special ResNet50
        arch = "resnet50_inat"
    args["net"] = arch
    args["disable_pretrained"] = "weights" not in extractor_config["backbone"]
    net, add_on = legacy.prototree.util.net.get_network(
        num_in_channels=0,
        args=Namespace(**args),
        seed=seed,
    )

    # Build ProtoTree
    tree_config = config["classifier"]
    assert tree_config["name"] == "ProtoTreeClassifier", "Invalid configuration file"
    # Update args namespace
    args.update(
        {
            "depth": tree_config["params"]["depth"],
            "disable_derivative_free_leaf_optim": False,
            "kontschieder_normalization": False,
            "kontschieder_train": False,
            "log_probabilities": tree_config["params"]["log_probabilities"],
            "H1": 1,
            "W1": 1,
            "state_dict_dir_tree": "",
            "state_dict_dir_net": "",
        }
    )
    tree = prototree_legacy.ProtoTree(
        num_classes=tree_config["params"]["num_classes"],
        feature_net=net,
        args=Namespace(**args),
        add_on_layers=add_on,
    )
    tree, _ = legacy.prototree.util.init.init_tree(
        tree=tree,
        optimizer=None,
        scheduler=None,
        device="cuda",
        args=Namespace(**args),
    )
    system_state["model"] = copy.deepcopy(tree)

    # Load dataset
    dataset_info = load_config(dataset_config)
    dataset_name = "CARS" if dataset_info["train_set"]["name"] == "StanfordCars" else "CUB-200-2011"
    args.update({"dataset": dataset_name, "batch_size": dataset_info["train_set"]["batch_size"], "disable_cuda": False})
    trainloader, projectloader, testloader, classes, num_channels = legacy.prototree.util.data.get_dataloaders(
        args=Namespace(**args)
    )
    for xs, ys in trainloader:
        break
    system_state["dataloader_test"] = xs.cpu().clone(), ys.cpu().clone()

    # Populate remaining options
    train_config = load_config(training_config)["optimizer"]
    visualization_config = load_config(visualization_config)["prototype"]["view"]
    args.update(
        {
            "optimizer": train_config["name"],
            "net": arch,
            "dataset": dataset_name,
            "lr": train_config["params"]["lr"],
            # "lr_pi": TODO: Add support for training mode with backprop on leaves (ie disable derivative free algorithm)
            "lr_net": train_config["config"]["backbone_to_freeze"]["lr"],
            "lr_block": train_config["config"]["backbone_to_train"]["lr"],
            "weight_decay": train_config["config"]["backbone_to_train"]["weight_decay_rate"],
            "momentum": train_config["params"]["momentum"],
            "disable_derivative_free_leaf_optim": False,
            "log_dir": root_directory,
            "dir_for_saving_images": "upsampling_results",
            "upsample_threshold": visualization_config["params"]["percentile"],
        }
    )
    optimizer, params_to_freeze, params_to_train = legacy.prototree.util.args.get_optimizer(tree, Namespace(**args))
    assert (
        "scheduler" in train_config and train_config["scheduler"]["type"] == "MultiStepLR"
    ), "Invalid LR scheduler configuration"
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, **train_config["scheduler"]["params"])
    system_state["optimizer_states"].append(copy.deepcopy(optimizer.state_dict()))
    system_state["scheduler_states"].append(copy.deepcopy(scheduler.state_dict()))

    # Training
    train_config = load_config(training_config)
    num_epochs = train_config["num_epochs"]
    freeze_epochs = train_config["freeze"]["warmup"]["epoch_range"][1]
    args.update({"freeze_epochs": freeze_epochs + 1})

    for epoch in range(num_epochs):
        legacy.prototree.util.net.freeze(
            tree, epoch + 1, params_to_freeze, params_to_train, Namespace(**args), dummy_logger
        )
        # Train tree
        if tree._kontschieder_train:
            train_info = legacy.prototree.prototree.train.train_epoch_kontschieder(
                tree=tree,
                train_loader=trainloader,
                optimizer=optimizer,
                epoch=epoch,
                disable_derivative_free_leaf_optim=False,
                device=device,
            )
        else:
            train_info = legacy.prototree.prototree.train.train_epoch(
                tree=tree,
                train_loader=trainloader,
                optimizer=optimizer,
                epoch=epoch,
                disable_derivative_free_leaf_optim=False,
                device=device,
                max_batches=5,
            )
        # Update scheduler and leaf labels before saving checkpoints
        scheduler.step()

        # Move model and optimizer to CPU before saving, then move back to GPU
        tree.to("cpu")
        optimizer_to(optimizer, "cpu")
        system_state["model_states"].append(copy.deepcopy(tree.state_dict()))
        system_state["optimizer_states"].append(copy.deepcopy(optimizer.state_dict()))
        system_state["scheduler_states"].append(copy.deepcopy(scheduler.state_dict()))
        tree.to(device)
        optimizer_to(optimizer, device)

    if legacy_state_dict is None:
        return system_state

    tree.to("cpu")
    tree.load_state_dict(torch.load(legacy_state_dict))
    tree.eval()

    # Testing sampling strategies
    system_state["distributed"] = tree(torch.randn((10, 3, 224, 224)), sampling_strategy="distributed")
    system_state["sample_max"] = tree(torch.randn((10, 3, 224, 224)), sampling_strategy="sample_max")
    system_state["greedy"] = tree(torch.randn((10, 3, 224, 224)), sampling_strategy="greedy")

    # Prune weak leaves
    legacy.prototree.prototree.prune.prune(tree=tree, pruning_threshold_leaves=0.01, log=dummy_logger)
    system_state["pruned"] = copy.deepcopy(tree)

    # Project prototypes
    tree.to(device)
    projection_info, _ = legacy.prototree.prototree.project.project_with_class_constraints(
        tree=tree, project_loader=projectloader, device=device, args=Namespace(**args), log=dummy_logger
    )
    system_state["projected"] = copy.deepcopy(tree.to("cpu"))
    system_state["projection_info"] = projection_info

    folder_name = "pruned_and_projected"
    legacy.prototree.util.save.save_tree_description(
        tree=tree, optimizer=optimizer, scheduler=scheduler, description=folder_name, log=dummy_logger
    )

    legacy.prototree.prototree.upsample.upsample(
        tree=tree.to(device),
        project_info=projection_info,
        project_loader=projectloader,
        folder_name=folder_name,
        args=Namespace(**args),
        log=dummy_logger,
    )

    return system_state


def execute(args: Namespace) -> None:
    """Create cabrnet model, then load a state dictionary in legacy ProtoTree form.

    Args:
        args: Parsed arguments.

    """

    # Initialise, then train similar models from the same random seed, and check that the result is identical
    memory_logger.stats()
    system_test = cabrnet_process(
        model_config=args.model_config,
        dataset_config=args.dataset,
        training_config=args.training,
        visualization_config=args.visualization,
        legacy_state_dict=args.legacy_state_dict,
        seed=args.seed,
        verbose=args.verbose,
        root_directory=os.path.join(args.training_dir, "cabrnet"),
        device=args.device,
    )
    memory_logger.stats()
    system_ref = legacy_process(
        model_config=args.model_config,
        dataset_config=args.dataset,
        training_config=args.training,
        visualization_config=args.visualization,
        legacy_state_dict=args.legacy_state_dict,
        seed=args.seed,
        root_directory=os.path.join(args.training_dir, "legacy"),
        device=args.device,
    )
    memory_logger.stats()

    tree_init, model_init = system_ref["model"], system_test["model"]
    compare_inference(
        description="Prototree initialisation",
        ref_model=tree_init,
        test_model=model_init,
    )
    compare_generic(
        description="Comparing image/label batches",
        test=system_test["dataloader_test"],
        ref=system_ref["dataloader_test"],
    )
    for epoch, (ref, test) in enumerate(zip(system_ref["model_states"], system_test["model_states"])):
        tree_init.load_state_dict(ref)
        model_init.load_state_dict(test)
        compare_inference(
            description=f"Prototree matching at epoch {epoch}",
            ref_model=tree_init,
            test_model=model_init,
        )
    for epoch, (ref, test) in enumerate(zip(system_ref["optimizer_states"], system_test["optimizer_states"])):
        compare_generic(
            description=f"Optimizers matching at epoch {epoch-1}",
            ref=ref,
            test=test,
        )
    for epoch, (ref, test) in enumerate(zip(system_ref["scheduler_states"], system_test["scheduler_states"])):
        compare_generic(
            description=f"Schedulers matching at epoch {epoch-1}",
            ref=ref,
            test=test,
        )
    if args.legacy_state_dict is not None:
        for strategy in ["distributed", "sample_max", "greedy"]:
            compare_generic(
                description=f"Comparing strategy {strategy}",
                ref=system_ref[strategy][0],
                test=system_test[strategy][0],
            )
        compare_inference(
            description=f"Model inference matching after pruning",
            ref_model=system_ref["pruned"],
            test_model=system_test["pruned"],
        )
        compare_inference(
            description=f"Model inference matching after projection",
            ref_model=system_ref["projected"],
            test_model=system_test["projected"],
        )

        compare_projection_info(ref_dict=system_ref["projection_info"], test_dict=system_test["projection_info"])
