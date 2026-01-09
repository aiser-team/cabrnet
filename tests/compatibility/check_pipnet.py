import copy
import unittest
import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import Namespace


from compatibility_tester import CaBRNetCompatibilityTester, setup_rng, SAMPLING_RATIO
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.parser import load_config
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.optimizers import OptimizerManager

import pipnet_legacy.pipnet.pipnet as legacy_pipnet
import pipnet_legacy.pipnet.train as legacy_train
import pipnet_legacy.util.func as legacy_func
import pipnet_legacy.util.data as legacy_data
import pipnet_legacy.util.args as legacy_args


def legacy_get_namespace(config_dict: dict[str, Path], seed: int) -> Namespace:
    """Build legacy compatible namespace from configuration files"""

    # Default parameters
    args = {
        "validation_size": 0.0,
        "batch_size": 16,
        "batch_size_pretrain": 24,
        "epochs": 60,
        "optimizer": "Adam",
        "lr": 0.05,
        "lr_block": 0.0005,
        "lr_net": 0.0005,
        "weight_decay": 0.0,
        "disable_cuda": False,
        "log_dir": "./runs/pipnet_cub_cnext26",
        "num_features": 0,  # Always 0: number of features only depends on the extractor configuration
        "image_size": 224,
        "state_dict_dir_net": "",
        "freeze_epochs": 10,
        "dir_for_saving_images": "Visualization_results",
        "disable_pretrained": False,
        "epochs_pretrain": 10,
        "weighted_loss": False,
        "seed": seed,
        "gpu_ids": "",
        "num_workers": 1,  # To improve compatibility checks
        "bias": False,
        "sampling_ratio": SAMPLING_RATIO // 20,
    }

    # Model information
    config = load_config(config_dict["model_config"])
    extractor_config = config["extractor"]
    arch = extractor_config["backbone"]["arch"]
    if arch == "resnet50" and "inat" in extractor_config["backbone"]["weights"]:
        arch = "resnet50_inat"  # Special ResNet50
    if arch == "convnext_tiny":
        assert extractor_config["backbone"].get("postprocess", {}).get("stride_divider", {}).get("min_channels") in [
            100,
            300,
        ], "Invalid or missing ConvNext subtype"
        if extractor_config["backbone"]["postprocess"]["stride_divider"]["min_channels"] == 100:
            arch = "convnext_tiny_26"
        else:
            arch = "convnext_tiny_13"
    args["net"] = arch
    if extractor_config.get("add_on") is not None:
        args["num_features"] = extractor_config["add_on"]["conv"]["params"]["out_channels"]

    # Dataset information
    dataset_info = load_config(config_dict["dataset_config"])
    match dataset_info["train_set"]["name"]:
        case "StanfordCars":
            dataset_name = "CARS"
        case "pets":
            dataset_name = "pets"
        case _:
            dataset_name = "CUB-200-2011"
    args.update(
        {
            "dataset": dataset_name,
            "batch_size": dataset_info["train_set"]["batch_size"],
            "batch_size_pretrain": dataset_info["pretrain_set"]["batch_size"],
            "disable_cuda": False,
        }
    )

    # Training and visualisation information
    train_config = load_config(config_dict["training_config"])
    num_pretrain_epochs = train_config["periods"]["pretrain"]["num_epochs"]
    args.update(
        {
            "epochs": train_config["num_epochs"] - num_pretrain_epochs,
            "epochs_pretrain": num_pretrain_epochs,
            "lr": train_config["optimizers"]["optimizer_net"]["params"]["lr"],
            "lr_block": train_config["optimizers"]["optimizer_net"]["groups"]["backbone_to_freeze"]["lr"],
            "lr_net": train_config["optimizers"]["optimizer_net"]["groups"]["backbone"]["lr"],
            "weight_decay": train_config["optimizers"]["optimizer_net"]["params"]["weight_decay"],
            "freeze_epochs": train_config["periods"]["fine_tuning"]["num_epochs"]
            + train_config["periods"]["warmup"]["num_epochs"],
        }
    )
    return Namespace(**args)


def legacy_get_model(num_classes: int, args: Namespace, seed: int) -> nn.DataParallel[legacy_pipnet.PIPNet]:
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = legacy_pipnet.get_network(
        num_classes, args, seed
    )
    # Create a PIP-Net
    legacy_model = legacy_pipnet.PIPNet(
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        feature_net=feature_net,
        args=args,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer,
    )
    # Init parameters
    torch.nn.init.normal_(legacy_model._classification.weight, mean=1.0, std=0.1)
    if args.bias:
        torch.nn.init.constant_(legacy_model._classification.bias, val=0.0)
    legacy_model._add_on.apply(legacy_func.init_weights_xavier)
    torch.nn.init.constant_(legacy_model._multiplier, val=2.0)
    legacy_model._multiplier.requires_grad = False
    legacy_model = nn.DataParallel(legacy_model)
    return legacy_model


def legacy_get_optimizers(
    legacy_model: nn.Module, args: Namespace
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    list[torch.nn.parameter.Parameter],
    list[torch.nn.parameter.Parameter],
    list[torch.nn.parameter.Parameter],
]:
    return legacy_args.get_optimizer_nn(net=legacy_model, args=args)  # type: ignore


def legacy_pretrain_model(legacy_model: nn.Module, dataloader: DataLoader, args: Namespace, device: str | torch.device):
    # Define classification loss function and scheduler
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = legacy_get_optimizers(
        legacy_model=legacy_model, args=args
    )
    criterion = nn.NLLLoss(reduction="mean").to(device)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_net,
        T_max=len(dataloader) * args.epochs,
        eta_min=args.lr_block / 100.0,
        last_epoch=-1,
    )

    # PRETRAINING PROTOTYPES PHASE
    for epoch in range(1, args.epochs_pretrain + 1):
        for param in params_to_train:
            param.requires_grad = True
        for param in legacy_model.module._add_on.parameters():
            param.requires_grad = True
        for param in legacy_model.module._classification.parameters():
            param.requires_grad = False
        for param in params_to_freeze:
            param.requires_grad = True
        for param in params_backbone:
            param.requires_grad = False

        # Pretrain prototypes
        legacy_train.train_pipnet(
            legacy_model,
            dataloader,
            optimizer_net,
            optimizer_classifier,
            scheduler_net,
            None,
            criterion,
            epoch,
            args.epochs_pretrain,
            device,
            pretrain=True,
            finetune=False,
        )
    return legacy_model, optimizer_net, optimizer_classifier, scheduler_net


def legacy_train_model(
    legacy_model: nn.Module, dataloader: DataLoader, args: Namespace, device: str | torch.device, num_epochs: int = -1
):
    criterion = nn.NLLLoss(reduction="mean").to(device)
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = legacy_get_optimizers(
        legacy_model=legacy_model, args=args
    )
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_net, T_max=len(dataloader) * args.epochs, eta_min=args.lr_net / 100.0
    )

    # scheduler for the classification layer is restarted, such that the model can re-active zeroed-out prototypes.
    if args.epochs <= 30:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_classifier, T_0=5, eta_min=0.001, T_mult=1
        )
    else:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_classifier, T_0=10, eta_min=0.001, T_mult=1
        )
    for param in legacy_model.module.parameters():
        param.requires_grad = False
    for param in legacy_model.module._classification.parameters():
        param.requires_grad = True

    frozen = True
    num_epochs = args.epochs if num_epochs == -1 else num_epochs
    for epoch in range(1, num_epochs + 1):
        # During finetuning, only train classification layer and freeze rest.
        # Usually done for a few epochs (at least 1, more depends on size of dataset)
        epochs_to_finetune = 3
        if epoch <= epochs_to_finetune and (args.epochs_pretrain > 0 or args.state_dict_dir_net != ""):
            for param in legacy_model.module._add_on.parameters():
                param.requires_grad = False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False
            finetune = True
        else:
            finetune = False
            if frozen:
                # unfreeze backbone
                if epoch > (args.freeze_epochs):
                    for param in legacy_model.module._add_on.parameters():
                        param.requires_grad = True
                    for param in params_to_freeze:
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = True
                    frozen = False
                # freeze first layers of backbone, train rest
                else:
                    for param in params_to_freeze:
                        param.requires_grad = True  # Can be set to False if you want to train fewer layers of backbone
                    for param in legacy_model.module._add_on.parameters():
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = False

        train_info = legacy_train.train_pipnet(
            legacy_model,
            dataloader,
            optimizer_net,
            optimizer_classifier,
            scheduler_net,
            scheduler_classifier,
            criterion,
            epoch,
            args.epochs,
            device,
            pretrain=False,
            finetune=finetune,
        )
    return legacy_model, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier


class Tester(CaBRNetCompatibilityTester):
    def __init__(self, methodName: str = "runTest"):
        super(Tester, self).__init__(arch="pipnet", methodName=methodName)

        self.legacy_params = legacy_get_namespace(
            {
                "model_config": self.model_config_file,
                "dataset_config": self.dataset_config_file,
                "training_config": self.training_config_file,
            },
            seed=self.seed,
        )

    def test_model_init(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(
            num_classes=cabrnet_model.classifier.num_classes, args=self.legacy_params, seed=self.seed
        )
        self.assertModelEqual(legacy_model, cabrnet_model)

    def test_load_legacy(self):
        # CaBRNet
        setup_rng(self.seed + 3)  # Setup with a different seed
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed + 3, compatibility_mode=True)

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(
            num_classes=cabrnet_model.classifier.num_classes, args=self.legacy_params, seed=self.seed
        )

        cabrnet_model.load_state_dict(legacy_model.state_dict())
        self.assertModelEqual(cabrnet_model, legacy_model)

    def test_dataloaders(self):
        # # CaBRNet
        setup_rng(self.seed)
        dataloaders = DatasetManager.get_dataloaders(
            config=self.dataset_config_file, sampling_ratio=SAMPLING_RATIO // 20
        )
        xc_train, yc_train = next(iter(dataloaders["train_set"]))
        xc_pretrain, yc_pretrain = next(iter(dataloaders["pretrain_set"]))
        xc_train_n, yc_train_n = next(iter(dataloaders["train_set_normal"]))
        xc_train_a, yc_train_a = next(iter(dataloaders["train_set_augment"]))
        xc_project, yc_project = next(iter(dataloaders["projection_set"]))
        xc_test, yc_test = next(iter(dataloaders["test_set"]))
        xc_test_project, yc_test_project = next(iter(dataloaders["test_set_projection"]))

        # Legacy
        setup_rng(self.seed)
        (
            train_loader,
            train_loader_pretraining,
            train_loader_normal,
            train_loader_normal_augment,
            project_loader,
            test_loader,
            test_project_loader,
            _,
        ) = legacy_data.get_dataloaders(self.legacy_params, device=None)
        xl_train_1, xl_train_2, yl_train = next(iter(train_loader))
        xl_pretrain_1, xl_pretrain_2, yl_pretrain = next(iter(train_loader_pretraining))
        xl_train_n, yl_train_n = next(iter(train_loader_normal))
        xl_train_a, yl_train_a = next(iter(train_loader_normal_augment))
        xl_project, yl_project = next(iter(project_loader))
        xl_test, yl_test = next(iter(test_loader))
        xl_test_project, yl_test_project = next(iter(test_project_loader))

        xc_train_1, xc_train_2 = xc_train.chunk(2)
        xc_pretrain_1, xc_pretrain_2 = xc_pretrain.chunk(2)
        self.assertTensorEqual(xc_train_1, xl_train_1)
        self.assertTensorEqual(xc_train_2, xl_train_2)
        self.assertTensorEqual(yc_train, yl_train)
        self.assertTensorEqual(xc_pretrain_1, xl_pretrain_1)
        self.assertTensorEqual(xc_pretrain_2, xl_pretrain_2)
        self.assertTensorEqual(yc_pretrain, yl_pretrain)
        self.assertTensorEqual(xc_train_n, xl_train_n)
        self.assertTensorEqual(yc_train_n, yl_train_n)
        self.assertTensorEqual(xc_train_a, xl_train_a)
        self.assertTensorEqual(yc_train_a, yl_train_a)
        self.assertTensorEqual(xc_project, xl_project)
        self.assertTensorEqual(yc_project, yl_project)
        self.assertTensorEqual(xc_test, xl_test)
        self.assertTensorEqual(yc_test, yl_test)
        self.assertTensorEqual(xc_test_project, xl_test_project)
        self.assertTensorEqual(yc_test_project, yl_test_project)

    def test_optimizers_init(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)
        setup_rng(self.seed)  # Mimic PIPNet
        optimizer_mngr = OptimizerManager.build_from_config(self.training_config_file, cabrnet_model)

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(
            num_classes=cabrnet_model.classifier.num_classes, args=self.legacy_params, seed=self.seed
        )
        optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = legacy_get_optimizers(
            legacy_model=legacy_model, args=self.legacy_params
        )
        (
            train_loader,
            train_loader_pretraining,
            train_loader_normal,
            train_loader_normal_augment,
            project_loader,
            test_loader,
            test_project_loader,
            _,
        ) = legacy_data.get_dataloaders(self.legacy_params, device=None)
        scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_net,
            T_max=len(train_loader) * self.legacy_params.epochs,
            eta_min=self.legacy_params.lr_block / 100.0,
            last_epoch=-1,
        )

        # Compare
        self.assertGenericEqual(optimizer_net.state_dict(), optimizer_mngr.optimizers["optimizer_net"].state_dict())
        self.assertGenericEqual(
            optimizer_classifier.state_dict(), optimizer_mngr.optimizers["optimizer_classifier"].state_dict()
        )
        self.assertGenericEqual(scheduler_net.state_dict(), optimizer_mngr.schedulers["optimizer_net"].state_dict())

    def _test_train(self, periods: list[str] | None = None):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)
        training_config = load_config(self.training_config_file)
        cabrnet_model.register_training_params(training_config)
        setup_rng(self.seed)  # Mimic PIPNet
        optimizer_mngr = OptimizerManager.build_from_config(self.training_config_file, cabrnet_model)
        dataloaders = DatasetManager.get_dataloaders(
            config=self.dataset_config_file, sampling_ratio=SAMPLING_RATIO // 20
        )
        if periods is not None:
            num_epochs = sum([training_config["periods"][period]["num_epochs"] for period in periods])
        else:
            # All epochs
            num_epochs = training_config["num_epochs"]

        for epoch in tqdm(range(num_epochs), desc="Training CaBRNet model", disable=not self.verbose):
            optimizer_mngr.freeze(epoch=epoch)
            train_infos = cabrnet_model.train_epoch(
                epoch_idx=epoch,
                dataloaders=dataloaders,
                optimizer_mngr=optimizer_mngr,
                device=self.device,
                tqdm_position=1,
                verbose=self.verbose,
            )
            optimizer_mngr.scheduler_step(epoch=epoch, metric=None)

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(
            num_classes=cabrnet_model.classifier.num_classes, args=self.legacy_params, seed=self.seed
        )
        (
            train_loader,
            train_loader_pretraining,
            train_loader_normal,
            train_loader_normal_augment,
            project_loader,
            test_loader,
            test_project_loader,
            _,
        ) = legacy_data.get_dataloaders(self.legacy_params, device=None)

        # Perform pretraining
        legacy_model, optimizer_net, optimizer_classifier, scheduler_net = legacy_pretrain_model(
            legacy_model, train_loader_pretraining, self.legacy_params, self.device
        )
        num_epochs_pretrain = training_config["periods"]["pretrain"]["num_epochs"]
        scheduler_classifier = None
        if num_epochs > num_epochs_pretrain:
            num_epochs = -1 if periods is None else num_epochs - num_epochs_pretrain
            legacy_model, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier = legacy_train_model(
                legacy_model, train_loader, self.legacy_params, self.device, num_epochs=num_epochs
            )

        converted_model = copy.deepcopy(cabrnet_model)
        converted_model.load_state_dict(legacy_model.state_dict())
        self.assertStateEqual(converted_model.state_dict(), cabrnet_model.state_dict())

        if scheduler_classifier is not None:
            self.assertGenericEqual(
                scheduler_classifier.state_dict(), optimizer_mngr.schedulers["optimizer_classifier"].state_dict()
            )
        self.assertGenericEqual(
            optimizer_classifier.state_dict(), optimizer_mngr.optimizers["optimizer_classifier"].state_dict()
        )
        self.assertGenericEqual(scheduler_net.state_dict(), optimizer_mngr.schedulers["optimizer_net"].state_dict())
        self.assertGenericEqual(optimizer_net.state_dict(), optimizer_mngr.optimizers["optimizer_net"].state_dict())

        self.assertModelEqual(legacy_model, cabrnet_model)

    def test_pretrain(self):
        self._test_train(periods=["pretrain"])

    def test_finetuning(self):
        self._test_train(periods=["pretrain", "fine_tuning"])

    def test_warmup(self):
        self._test_train(periods=["pretrain", "fine_tuning", "warmup"])

    def test_train(self):
        self._test_train()


def main():
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])
    unittest.main()


if __name__ == "__main__":
    main()
