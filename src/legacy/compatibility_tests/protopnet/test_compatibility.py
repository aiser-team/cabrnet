import unittest
import sys
from typing import Any
from loguru import logger
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from cabrnet.generic.model import ProtoClassifier
from cabrnet.utils.parser import load_config
from cabrnet.utils.data import get_dataloaders
from cabrnet.utils.optimizers import OptimizerManager

import legacy.protopnet.settings as legacy_settings
import legacy.protopnet.preprocess as legacy_preprocess
from legacy.protopnet.model import construct_PPNet


def setup_rng(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class DummyLogger:
    def __call__(self, message: str):
        pass


def legacy_get_model(seed: int) -> nn.Module:
    return construct_PPNet(
        base_architecture=legacy_settings.base_architecture,
        pretrained=True,
        img_size=legacy_settings.img_size,
        prototype_shape=legacy_settings.prototype_shape,
        num_classes=legacy_settings.num_classes,
        prototype_activation_function=legacy_settings.prototype_activation_function,
        add_on_layers_type=legacy_settings.add_on_layers_type,
        seed=seed,
    )


def legacy_get_dataloaders(dataset_config: str) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_info = load_config(dataset_config)
    train_dir = dataset_info["train_set"]["params"]["root"]
    train_push_dir = dataset_info["projection_set"]["params"]["root"]
    test_dir = dataset_info["test_set"]["params"]["root"]
    normalize = transforms.Normalize(mean=legacy_preprocess.mean, std=legacy_preprocess.std)
    legacy_transforms = transforms.Compose(
        [
            transforms.Resize(size=(legacy_settings.img_size, legacy_settings.img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    legacy_transforms_no_normalization = transforms.Compose(
        [
            transforms.Resize(size=(legacy_settings.img_size, legacy_settings.img_size)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = datasets.ImageFolder(train_dir, legacy_transforms)
    train_batch_size = 16  # Reduce train batch size
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=False)
    train_push_dataset = datasets.ImageFolder(train_push_dir, legacy_transforms_no_normalization)
    train_push_loader = DataLoader(
        train_push_dataset,
        batch_size=legacy_settings.train_push_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )
    test_dataset = datasets.ImageFolder(test_dir, legacy_transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=legacy_settings.test_batch_size, shuffle=False, num_workers=4, pin_memory=False
    )
    return train_loader, test_loader, train_push_loader


def legacy_get_optimizers(
    legacy_model: nn.Module,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.lr_scheduler]:
    joint_optimizer_specs = [
        {
            "params": legacy_model.features.parameters(),
            "lr": legacy_settings.joint_optimizer_lrs["features"],
            "weight_decay": 1e-3,
        },
        {
            "params": legacy_model.add_on_layers.parameters(),
            "lr": legacy_settings.joint_optimizer_lrs["add_on_layers"],
            "weight_decay": 1e-3,
        },
        {"params": legacy_model.prototype_vectors, "lr": legacy_settings.joint_optimizer_lrs["prototype_vectors"]},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        joint_optimizer, step_size=legacy_settings.joint_lr_step_size, gamma=0.1
    )

    warm_optimizer_specs = [
        {
            "params": legacy_model.add_on_layers.parameters(),
            "lr": legacy_settings.warm_optimizer_lrs["add_on_layers"],
            "weight_decay": 1e-3,
        },
        {"params": legacy_model.prototype_vectors, "lr": legacy_settings.warm_optimizer_lrs["prototype_vectors"]},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    last_layer_optimizer_specs = [
        {"params": legacy_model.last_layer.parameters(), "lr": legacy_settings.last_layer_optimizer_lr}
    ]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
    return warm_optimizer, joint_optimizer, last_layer_optimizer, joint_lr_scheduler


class TestProtoPNetCompatibility(unittest.TestCase):
    def __init__(self, methodName: str = "runTest"):
        super(TestProtoPNetCompatibility, self).__init__(methodName=methodName)

        # Test configuration
        self.model_config: str = "src/legacy/compatibility_tests/protopnet/model.yml"
        self.dataset_config: str = "src/legacy/compatibility_tests/protopnet/cub200.yml"
        self.training_config: str = "src/legacy/compatibility_tests/protopnet/training.yml"
        self.legacy_state_dict: str | None = "legacy_states/protopnet/protopnet_cub200_vgg19.pth"
        self.seed: int = 42

    def assertTensorEqual(self, expected: torch.Tensor, actual: torch.Tensor):
        if actual.size() != expected.size():
            self.fail(f"Mismatching tensor sizes: {actual.size()} v. {expected.size()}")
        elif not torch.all(torch.eq(expected, actual)):
            self.fail(f"Mismatching tensors (all close? {torch.allclose(expected,actual)})")

    def assertModelEqual(self, expected: nn.Module, actual: nn.Module):
        expected.eval()
        actual.eval()
        x = torch.rand(10, 3, 224, 224)
        with torch.no_grad():
            y_e = expected(x)
            y_a = actual(x)

        self.assertGenericEqual(y_e, y_a)

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
            if not (expected == actual).all():
                self.fail(f"Mismatching np arrays (all close? {np.allclose(expected, actual)})")
        else:
            self.assertEqual(expected, actual)

    def test_model_init(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = ProtoClassifier.build_from_config(self.model_config, seed=self.seed, compatibility_mode=True)

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(self.seed)

        self.assertModelEqual(legacy_model, cabrnet_model)

    def test_dataloaders(self):
        # CaBRNet
        setup_rng(self.seed)
        dataloaders = get_dataloaders(config_file=self.dataset_config)
        xc_train, yc_train = next(iter(dataloaders["train_set"]))
        xc_test, yc_test = next(iter(dataloaders["test_set"]))

        # Legacy
        setup_rng(self.seed)
        train_loader, test_loader, _ = legacy_get_dataloaders(self.dataset_config)
        xl_train, yl_train = next(iter(train_loader))
        xl_test, yl_test = next(iter(test_loader))

        # Do not test push dataloader since normalization is inherently different
        self.assertTensorEqual(xl_train, xc_train)
        self.assertTensorEqual(yl_train, yc_train)
        self.assertTensorEqual(xl_test, xc_test)
        self.assertTensorEqual(yl_test, yc_test)

    def test_optimizers_init(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = ProtoClassifier.build_from_config(self.model_config, seed=self.seed, compatibility_mode=True)
        optimizer_mngr = OptimizerManager.build_from_config(self.training_config, cabrnet_model)

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(self.seed)
        warm_optimizer, joint_optimizer, last_layer_optimizer, joint_lr_scheduler = legacy_get_optimizers(legacy_model)

        # Compare
        self.assertGenericEqual(warm_optimizer.state_dict(), optimizer_mngr.optimizers["warmup_optimizer"].state_dict())
        self.assertGenericEqual(joint_optimizer.state_dict(), optimizer_mngr.optimizers["joint_optimizer"].state_dict())
        self.assertGenericEqual(
            last_layer_optimizer.state_dict(), optimizer_mngr.optimizers["last_layer_optimizer"].state_dict()
        )
        self.assertGenericEqual(
            joint_lr_scheduler.state_dict(), optimizer_mngr.schedulers["joint_optimizer"].state_dict()
        )

    def test_load_legacy_state_dict(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = ProtoClassifier.build_from_config(self.model_config, seed=self.seed, compatibility_mode=True)
        cabrnet_model.load_legacy_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(seed=self.seed)
        legacy_model.load_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))

        self.assertModelEqual(legacy_model, cabrnet_model)


def main():
    torch.use_deterministic_algorithms(mode=True)
    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])
    unittest.main()


if __name__ == "__main__":
    main()
