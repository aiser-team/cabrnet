import unittest
import sys
from typing import Any
from loguru import logger
import numpy as np
import random
from tqdm import tqdm

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
import legacy.protopnet.push as legacy_push
import legacy.protopnet.train_and_test as legacy_tnt


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
    # Reduce all batch sizes
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    train_push_dataset = datasets.ImageFolder(train_push_dir, legacy_transforms_no_normalization)
    train_push_loader = DataLoader(
        train_push_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )
    test_dataset = datasets.ImageFolder(test_dir, legacy_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
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
        self.model_config_file: str = "src/legacy/compatibility_tests/protopnet/model.yml"
        self.dataset_config_file: str = "src/legacy/compatibility_tests/protopnet/cub200.yml"
        self.training_config_file: str = "src/legacy/compatibility_tests/protopnet/training.yml"
        self.legacy_state_dict: str | None = "legacy_states/protopnet/protopnet_cub200_vgg19.pth"
        self.device: str = "cuda:0"
        self.seed: int = 42
        self.verbose: bool = True

    def assertTensorEqual(self, expected: torch.Tensor, actual: torch.Tensor, msg: str | None = None):
        if actual.size() != expected.size():
            self.fail(f"{msg} Mismatching tensor sizes: {actual.size()} v. {expected.size()}.")
        elif not torch.all(torch.eq(expected, actual)):
            # print(expected, actual)
            self.fail(f"{msg} Mismatching tensors (all close? {torch.allclose(expected,actual)}).")

    def assertModelEqual(self, expected: nn.Module, actual: nn.Module):
        expected.eval()
        actual.eval()
        expected.to(self.device)
        actual.to(self.device)
        x = torch.rand(16, 3, 224, 224).to(self.device)
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
                    self.assertGenericEqual(expected[key], actual[key], f"{msg} Checking key {key}.")
        elif isinstance(expected, list) or isinstance(expected, tuple):
            if len(expected) != len(actual):
                self.fail(f"Mismatching list lengths: {len(expected)} v. {len(actual)}.")
            for index, (ex, ac) in enumerate(zip(expected, actual)):
                self.assertGenericEqual(ex, ac, f"{msg} Checking list (index {index}).")
        elif isinstance(expected, torch.Tensor):
            self.assertTensorEqual(expected, actual, msg)
        elif isinstance(expected, np.ndarray):
            if not (expected == actual).all():
                self.fail(f"Mismatching np arrays (all close? {np.allclose(expected, actual)})")
        else:
            self.assertEqual(expected, actual, f"{msg} Checking generic type.")

    def test_model_init(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = ProtoClassifier.build_from_config(
            self.model_config_file, seed=self.seed, compatibility_mode=True
        )

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(self.seed)

        self.assertModelEqual(legacy_model, cabrnet_model)

    def test_dataloaders(self):
        # CaBRNet
        setup_rng(self.seed)
        dataloaders = get_dataloaders(config_file=self.dataset_config_file)
        xc_train, yc_train = next(iter(dataloaders["train_set"]))
        xc_test, yc_test = next(iter(dataloaders["test_set"]))

        # Legacy
        setup_rng(self.seed)
        train_loader, test_loader, _ = legacy_get_dataloaders(self.dataset_config_file)
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
        cabrnet_model = ProtoClassifier.build_from_config(
            self.model_config_file, seed=self.seed, compatibility_mode=True
        )
        optimizer_mngr = OptimizerManager.build_from_config(self.training_config_file, cabrnet_model)

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

    def test_train(self):
        max_batches = 5
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = ProtoClassifier.build_from_config(
            self.model_config_file, seed=self.seed, compatibility_mode=True
        )
        training_config = load_config(self.training_config_file)
        cabrnet_model.register_training_params(training_config)
        optimizer_mngr = OptimizerManager.build_from_config(self.training_config_file, cabrnet_model)
        dataloaders = get_dataloaders(config_file=self.dataset_config_file)
        num_epochs = training_config["num_epochs"]
        for epoch in tqdm(range(num_epochs), desc="Training CaBRNet model", disable=not self.verbose):
            optimizer_mngr.freeze(epoch=epoch)
            _ = cabrnet_model.train_epoch(
                epoch_idx=epoch,
                dataloaders=dataloaders,
                optimizer_mngr=optimizer_mngr,
                device=self.device,
                progress_bar_position=1,
                max_batches=max_batches,
                verbose=self.verbose,
            )
            optimizer_mngr.scheduler_step(epoch=epoch)

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(self.seed)
        warm_optimizer, joint_optimizer, last_layer_optimizer, joint_lr_scheduler = legacy_get_optimizers(legacy_model)
        train_loader, test_loader, train_push_loader = legacy_get_dataloaders(self.dataset_config_file)
        legacy_model_multi = nn.DataParallel(legacy_model)
        push_start = cabrnet_model.projection_config["start_epoch"]
        push_frequency = cabrnet_model.projection_config["frequency"]
        push_epochs = [epoch for epoch in range(push_start, num_epochs) if (epoch - push_start) % push_frequency == 0]
        for epoch in tqdm(range(num_epochs), desc="Training legacy model", disable=not self.verbose):
            if epoch < legacy_settings.num_warm_epochs:
                legacy_tnt.warm_only(model=legacy_model_multi, log=DummyLogger())
                _ = legacy_tnt.train(
                    model=legacy_model_multi,
                    dataloader=train_loader,
                    optimizer=warm_optimizer,
                    class_specific=True,
                    coefs=legacy_settings.coefs,
                    max_batches=max_batches,
                    log=DummyLogger(),
                )
            else:
                legacy_tnt.joint(model=legacy_model_multi, log=DummyLogger())
                _ = legacy_tnt.train(
                    model=legacy_model_multi,
                    dataloader=train_loader,
                    optimizer=joint_optimizer,
                    class_specific=True,
                    coefs=legacy_settings.coefs,
                    log=DummyLogger(),
                    max_batches=max_batches,
                )
                joint_lr_scheduler.step()
                if epoch in push_epochs:
                    legacy_push.push_prototypes(
                        train_push_loader,
                        prototype_network_parallel=legacy_model_multi,
                        class_specific=True,
                        preprocess_input_function=legacy_preprocess.preprocess_input_function,
                        prototype_layer_stride=1,
                        root_dir_for_saving_prototypes=None,
                        epoch_number=None,
                        prototype_img_filename_prefix=None,
                        prototype_self_act_filename_prefix=None,
                        proto_bound_boxes_filename_prefix=None,
                        save_prototype_class_identity=True,
                        log=DummyLogger(),
                    )
                    legacy_tnt.last_only(model=legacy_model_multi, log=DummyLogger())
                    for i in range(cabrnet_model.projection_config["num_ft_epochs"]):
                        _ = legacy_tnt.train(
                            model=legacy_model_multi,
                            dataloader=train_loader,
                            optimizer=last_layer_optimizer,
                            class_specific=True,
                            coefs=legacy_settings.coefs,
                            log=DummyLogger(),
                            max_batches=max_batches,
                        )

        # Compare
        self.assertGenericEqual(warm_optimizer.state_dict(), optimizer_mngr.optimizers["warmup_optimizer"].state_dict())
        self.assertGenericEqual(joint_optimizer.state_dict(), optimizer_mngr.optimizers["joint_optimizer"].state_dict())
        self.assertGenericEqual(
            last_layer_optimizer.state_dict(), optimizer_mngr.optimizers["last_layer_optimizer"].state_dict()
        )
        self.assertGenericEqual(
            joint_lr_scheduler.state_dict(), optimizer_mngr.schedulers["joint_optimizer"].state_dict()
        )
        self.assertModelEqual(legacy_model, cabrnet_model)

    def test_load_legacy_state_dict(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = ProtoClassifier.build_from_config(
            self.model_config_file, seed=self.seed, compatibility_mode=True
        )
        cabrnet_model.load_legacy_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(seed=self.seed)
        legacy_model.load_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))

        self.assertModelEqual(legacy_model, cabrnet_model)

    def test_push_prototypes(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = ProtoClassifier.build_from_config(
            self.model_config_file, seed=self.seed, compatibility_mode=True
        )
        dataloaders = get_dataloaders(config_file=self.dataset_config_file)
        cabrnet_info = cabrnet_model.project(
            data_loader=dataloaders["projection_set"], device=self.device, verbose=self.verbose
        )

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(seed=self.seed)
        _, _, push_loader = legacy_get_dataloaders(self.dataset_config_file)
        legacy_model_multi = nn.DataParallel(legacy_model)
        legacy_push.push_prototypes(
            dataloader=push_loader,
            prototype_network_parallel=legacy_model_multi,
            class_specific=True,
            preprocess_input_function=legacy_preprocess.preprocess_input_function,
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=None,
            epoch_number=None,
            prototype_img_filename_prefix=None,
            prototype_self_act_filename_prefix=None,
            proto_bound_boxes_filename_prefix=None,
            save_prototype_class_identity=True,
            log=DummyLogger(),
        )
        self.assertModelEqual(legacy_model, cabrnet_model)

    def test_compatibility_mode(self):
        # Model with compatibility mode
        compatible_model = ProtoClassifier.build_from_config(
            self.model_config_file, seed=self.seed, compatibility_mode=True
        )
        compatible_model.load_legacy_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))

        cabrnet_model = ProtoClassifier.build_from_config(
            self.model_config_file, seed=self.seed, compatibility_mode=False
        )
        cabrnet_model.load_legacy_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))

        # Get batch of images
        dataloaders = get_dataloaders(config_file=self.dataset_config_file)
        xs, ys = next(iter(dataloaders["train_set"]))

        # Compare outputs and loss values
        expected_logits, expected_min_distances = compatible_model(xs)
        expected_loss, expected_loss_stats = compatible_model.loss((expected_logits, expected_min_distances), ys)
        actual_logits, actual_min_distances = cabrnet_model(xs)
        actual_loss, actual_loss_stats = cabrnet_model.loss((actual_logits, actual_min_distances), ys)
        self.assertGenericEqual(expected_logits, actual_logits, "Checking logits")
        self.assertGenericEqual(expected_min_distances, actual_min_distances, "Checking min distances")
        # Allow some leeway for the delta between loss values
        error_percentage = abs((expected_loss.item() - actual_loss.item()) / expected_loss.item())
        if error_percentage > 0.001:
            self.fail(f"Loss error percentage too great: {error_percentage}")


def main():
    torch.use_deterministic_algorithms(mode=True)
    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])
    unittest.main()


if __name__ == "__main__":
    main()
