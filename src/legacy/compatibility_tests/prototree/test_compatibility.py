import unittest
import sys
import os
from typing import Any
from loguru import logger
import numpy as np
import random
from argparse import Namespace
import torch
import torch.nn as nn

from cabrnet.generic.model import CaBRNet
from cabrnet.utils.parser import load_config
from cabrnet.utils.data import get_dataloaders
from cabrnet.utils.optimizers import OptimizerManager
from cabrnet.prototree.decision import SamplingStrategy

import legacy.prototree.prototree.prototree as legacy_prototree
import legacy.prototree.prototree.train as legacy_train
import legacy.prototree.util.net as legacy_net
import legacy.prototree.util.data as legacy_data
import legacy.prototree.util.init as legacy_init
import legacy.prototree.util.args as legacy_args
import legacy.prototree.prototree.prune as legacy_prune
import legacy.prototree.prototree.project as legacy_project


def setup_rng(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class DummyLogger:
    checkpoint_dir: str = ""

    def set_checkpoint_dir(self, dir: str):
        pass

    def log_message(self, message: str):
        pass


def legacy_get_namespace(config_dict: dict[str, str]) -> Namespace:
    """Build legacy compatible namespace from configuration files"""
    args = {}

    # Model information
    config = load_config(config_dict["model_config"])
    extractor_config = config["extractor"]
    tree_config = config["classifier"]
    args["num_features"] = [
        extractor_config["add_on"][layer]["params"]["out_channels"]
        for layer in extractor_config["add_on"]
        if layer != "init_mode"
        and "params" in extractor_config["add_on"][layer]
        and "out_channels" in extractor_config["add_on"][layer]["params"]
    ][-1]
    arch = extractor_config["backbone"]["arch"]
    if arch == "resnet50" and "inat" in extractor_config["backbone"]["weights"]:
        arch = "resnet50_inat"  # Special ResNet50
    args.update(
        {
            "net": arch,
            "disable_pretrained": "weights" not in extractor_config["backbone"],
            "num_classes": tree_config["params"]["num_classes"],
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

    # Dataset information
    dataset_info = load_config(config_dict["dataset_config"])
    dataset_name = "CARS" if dataset_info["train_set"]["name"] == "StanfordCars" else "CUB-200-2011"
    args.update({"dataset": dataset_name, "batch_size": dataset_info["train_set"]["batch_size"], "disable_cuda": False})

    # Training and visualisation information
    train_config = load_config(config_dict["training_config"])["optimizers"]["main_optimizer"]
    # visualization_config = load_config(config_dict["visualization_config"])["prototype"]["view"]
    args.update(
        {
            "optimizer": train_config["type"],
            "net": arch,
            "dataset": dataset_name,
            "lr": train_config["params"]["lr"],
            # "lr_pi": TODO: Add support for training mode with backprop on leaves (ie disable derivative free algorithm)
            "lr_net": train_config["groups"]["backbone_to_freeze"]["lr"],
            "lr_block": train_config["groups"]["backbone_to_train"]["lr"],
            "weight_decay": train_config["groups"]["backbone_to_train"]["weight_decay_rate"],
            "momentum": train_config["params"]["momentum"],
            "disable_derivative_free_leaf_optim": False,
            "freeze_epochs": load_config(config_dict["training_config"])["periods"]["warmup"]["epoch_range"][1] + 1,
        }
    )
    return Namespace(**args)


def legacy_get_model(args: Namespace, seed: int) -> nn.Module:
    # Build feature extractor
    net, add_on = legacy_net.get_network(
        num_in_channels=0,
        args=args,
        seed=seed,
    )
    tree = legacy_prototree.ProtoTree(
        num_classes=args.num_classes,
        feature_net=net,
        args=args,
        add_on_layers=add_on,
    )
    tree, _ = legacy_init.init_tree(
        tree=tree,
        optimizer=None,
        scheduler=None,
        device="cuda",
        args=args,
    )
    return tree


class TestProtoTreeCompatibility(unittest.TestCase):
    def __init__(self, methodName: str = "runTest"):
        super(TestProtoTreeCompatibility, self).__init__(methodName=methodName)

        # Test configuration
        test_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_config_file = os.path.join(test_dir, "model_arch.yml")
        self.dataset_config_file = os.path.join(test_dir, "dataset.yml")
        self.training_config_file = os.path.join(test_dir, "training.yml")
        self.legacy_state_dict = os.path.join(test_dir, "legacy_state.pth")

        # Create a namespace with all legacy options
        self.legacy_params = legacy_get_namespace(
            {
                "model_config": self.model_config_file,
                "dataset_config": self.dataset_config_file,
                "training_config": self.training_config_file,
            }
        )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.seed: int = 42

    def assertTensorEqual(self, expected: torch.Tensor, actual: torch.Tensor):
        if actual.size() != expected.size():
            self.fail(f"Mismatching tensor sizes: {actual.size()} v. {expected.size()}")
        elif not torch.all(torch.eq(expected, actual)):
            self.fail(f"Mismatching tensors (all close? {torch.allclose(expected,actual)})")

    def assertModelEqual(self, expected: nn.Module, actual: nn.Module):
        expected.eval()
        actual.eval()
        expected.to(self.device)
        actual.to(self.device)
        x = torch.rand(10, 3, 224, 224).to(self.device)
        with torch.no_grad():
            y_e = expected(x)
            y_a = actual(x)

        # Only compare logits
        self.assertTensorEqual(y_e[0], y_a[0])

    def assertGenericEqual(self, expected: Any, actual: Any, msg: str | None = None):
        """Generic comparison function for complex data structures"""
        if type(expected) is not type(actual):
            self.fail(f"{msg} Mismatching types ({type(expected)} v. {type(actual)}) for {expected} and {actual}")
        elif isinstance(expected, dict):
            for key in expected:
                if key not in actual:
                    self.fail(f"{msg} Key {key} not found in state dict")
                else:
                    self.assertGenericEqual(expected[key], actual[key], f"Checking key {key}")
        elif isinstance(expected, list) or isinstance(expected, tuple):
            if len(expected) != len(actual):
                self.fail(f"Mismatching list lengths: {len(expected)} v. {len(actual)}.")
            for index, (ex, ac) in enumerate(zip(expected, actual)):
                self.assertGenericEqual(ex, ac, f"Checking list (index {index})")
        elif isinstance(expected, torch.Tensor):
            self.assertTensorEqual(expected, actual)
        elif isinstance(expected, np.ndarray):
            if np.array_equal(expected, actual):
                self.fail(f"Mismatching np arrays (all close? {np.allclose(expected, actual)})")
        else:
            self.assertEqual(expected, actual)

    def assertProjectionInfoEqual(self, expected: dict[Any, Any], actual: dict[Any, Any]):
        for key in expected:
            if actual[key]["img_idx"] == -1:
                continue
            if expected[key]["input_image_ix"] != actual[key]["img_idx"]:
                self.fail(
                    f"Mismatching image index for prototype {key}. "
                    f"Expected {expected[key]['input_image_ix']} but found {actual[key]['img_idx']}."
                )
            W = expected[key]["W"]
            if expected[key]["patch_ix"] != actual[key]["h"] * W + actual[key]["w"]:
                self.fail(
                    f"Mismatching patch coordinates for prototype {key}. "
                    f"Expected ({expected[key]['patch_ix'] // W}, {expected[key]['patch_ix'] % W}) "
                    f"but found ({actual[key]['h']}, {actual[key]['w']})."
                )

    def test_model_init(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(args=self.legacy_params, seed=self.seed)

        self.assertModelEqual(legacy_model, cabrnet_model)

    def test_dataloaders(self):
        # CaBRNet
        setup_rng(self.seed)
        dataloaders = get_dataloaders(config_file=self.dataset_config_file)
        xc_train, yc_train = next(iter(dataloaders["train_set"]))
        xc_test, yc_test = next(iter(dataloaders["test_set"]))
        xc_proj, yc_proj = next(iter(dataloaders["projection_set"]))

        # Legacy
        setup_rng(self.seed)
        train_loader, project_loader, test_loader, _, _ = legacy_data.get_dataloaders(args=self.legacy_params)
        xl_train, yl_train = next(iter(train_loader))
        xl_test, yl_test = next(iter(test_loader))
        xl_proj, yl_proj = next(iter(project_loader))

        self.assertTensorEqual(xl_train, xc_train)
        self.assertTensorEqual(yl_train, yc_train)
        self.assertTensorEqual(xl_test, xc_test)
        self.assertTensorEqual(yl_test, yc_test)
        self.assertTensorEqual(xl_proj, xc_proj)
        self.assertTensorEqual(yl_proj, yc_proj)

    def test_optimizers_init(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)
        optimizer_mngr = OptimizerManager.build_from_config(self.training_config_file, cabrnet_model)

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(args=self.legacy_params, seed=self.seed)
        legacy_optimizer, _, _ = legacy_args.get_optimizer(legacy_model, args=self.legacy_params)
        legacy_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=legacy_optimizer,
            **load_config(self.training_config_file)["optimizers"]["main_optimizer"]["scheduler"]["params"],
        )

        # Compare
        self.assertGenericEqual(legacy_optimizer.state_dict(), optimizer_mngr.optimizers["main_optimizer"].state_dict())
        self.assertGenericEqual(legacy_scheduler.state_dict(), optimizer_mngr.schedulers["main_optimizer"].state_dict())

    def test_load_legacy_state_dict(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)
        cabrnet_model.load_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(args=self.legacy_params, seed=self.seed)
        legacy_model.load_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))

        self.assertModelEqual(legacy_model, cabrnet_model)

    def test_training(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)
        dataloaders = get_dataloaders(config_file=self.dataset_config_file)
        optimizer_mngr = OptimizerManager.build_from_config(self.training_config_file, cabrnet_model)
        trainer = load_config(self.training_config_file)
        num_epochs = trainer["num_epochs"]
        for epoch in range(num_epochs):
            optimizer_mngr.freeze(epoch=epoch)
            cabrnet_model.train_epoch(
                dataloaders=dataloaders,
                optimizer_mngr=optimizer_mngr,
                epoch_idx=epoch,
                device=self.device,
                max_batches=5,
                verbose=True,
            )
            optimizer_mngr.scheduler_step(epoch=epoch)

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(args=self.legacy_params, seed=self.seed)
        train_loader, project_loader, test_loader, _, _ = legacy_data.get_dataloaders(args=self.legacy_params)
        legacy_optimizer, params_to_freeze, params_to_train = legacy_args.get_optimizer(
            legacy_model, args=self.legacy_params
        )
        legacy_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=legacy_optimizer,
            **load_config(self.training_config_file)["optimizers"]["main_optimizer"]["scheduler"]["params"],
        )
        for epoch in range(num_epochs):
            legacy_net.freeze(
                tree=legacy_model,
                epoch=epoch + 1,
                params_to_freeze=params_to_freeze,
                params_to_train=params_to_train,
                args=self.legacy_params,
                log=DummyLogger(),
            )
            _ = legacy_train.train_epoch(
                tree=legacy_model,
                train_loader=train_loader,
                optimizer=legacy_optimizer,
                epoch=epoch,
                disable_derivative_free_leaf_optim=False,
                device=self.device,
                max_batches=5,
            )
            legacy_scheduler.step()

        self.assertModelEqual(legacy_model, cabrnet_model)
        self.assertGenericEqual(legacy_optimizer.state_dict(), optimizer_mngr.optimizers["main_optimizer"].state_dict())
        self.assertGenericEqual(legacy_scheduler.state_dict(), optimizer_mngr.schedulers["main_optimizer"].state_dict())

    def test_sampling(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)
        cabrnet_model.load_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))
        cabrnet_model.eval()
        yc_distributed = cabrnet_model(torch.randn((10, 3, 224, 224)), strategy=SamplingStrategy.DISTRIBUTED)[0]
        yc_sample_max = cabrnet_model(torch.randn((10, 3, 224, 224)), strategy=SamplingStrategy.SAMPLE_MAX)[0]
        yc_greedy = cabrnet_model(torch.randn((10, 3, 224, 224)), strategy=SamplingStrategy.GREEDY)[0]

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(args=self.legacy_params, seed=self.seed)
        legacy_model.load_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))
        legacy_model.eval()
        yl_distributed = legacy_model(torch.randn((10, 3, 224, 224)), sampling_strategy="distributed")[0]
        yl_sample_max = legacy_model(torch.randn((10, 3, 224, 224)), sampling_strategy="sample_max")[0]
        yl_greedy = legacy_model(torch.randn((10, 3, 224, 224)), sampling_strategy="greedy")[0]

        # Compare
        self.assertTensorEqual(yl_distributed, yc_distributed)
        self.assertTensorEqual(yl_sample_max, yc_sample_max)
        self.assertTensorEqual(yl_greedy, yc_greedy)

    def test_pruning(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)
        cabrnet_model.load_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))
        cabrnet_model.prune(pruning_threshold=0.01)

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(args=self.legacy_params, seed=self.seed)
        legacy_model.load_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))
        legacy_prune.prune(tree=legacy_model, pruning_threshold_leaves=0.01, log=DummyLogger())

        self.assertModelEqual(cabrnet_model, legacy_model)

    def test_projection(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)
        dataloaders = get_dataloaders(config_file=self.dataset_config_file)
        cabrnet_model.load_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))
        cabrnet_model.prune(pruning_threshold=0.01)
        cabrnet_projection_info = cabrnet_model.project(
            data_loader=dataloaders["projection_set"], device=self.device, verbose=True
        )

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(args=self.legacy_params, seed=self.seed)
        _, project_loader, _, _, _ = legacy_data.get_dataloaders(args=self.legacy_params)
        legacy_model.load_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))
        legacy_prune.prune(tree=legacy_model, pruning_threshold_leaves=0.01, log=DummyLogger())
        legacy_projection_info, _ = legacy_project.project_with_class_constraints(
            tree=legacy_model.to(self.device),
            project_loader=project_loader,
            device=self.device,
            args=self.legacy_params,
            log=DummyLogger(),
        )

        self.assertModelEqual(legacy_model, cabrnet_model)
        self.assertProjectionInfoEqual(legacy_projection_info, cabrnet_projection_info)


def main():
    torch.use_deterministic_algorithms(mode=True)
    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])
    unittest.main()


if __name__ == "__main__":
    main()
