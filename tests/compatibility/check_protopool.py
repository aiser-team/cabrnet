import unittest
import sys
import numpy as np
from tqdm import tqdm
from loguru import logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path


from compatibility_tester import CaBRNetCompatibilityTester, setup_rng, SAMPLING_RATIO
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.archs.protopool.model import ProtoPool
from cabrnet.core.utils.parser import load_config
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.custom_preprocess import batch_mixup
from cabrnet.core.utils.optimizers import OptimizerManager

from protopool_legacy.model import PrototypeChooser
from protopool_legacy.utils import mixup_data
from protopool_legacy.main import dist_loss, update_prototypes_on_batch


def legacy_get_model(model_config_file: Path, seed: int) -> PrototypeChooser:
    model_config = load_config(model_config_file)
    return PrototypeChooser(
        num_prototypes=model_config["classifier"]["params"]["num_prototypes"],
        num_descriptive=model_config["classifier"]["params"]["num_slots_per_class"],
        num_classes=model_config["classifier"]["params"]["num_classes"],
        use_thresh=True,
        arch=model_config["extractor"]["backbone"]["arch"],
        pretrained=model_config["extractor"]["backbone"].get("weights", False),
        add_on_layers_type="log",  # Anything but "bottleneck"
        prototype_activation_function="log",
        proto_depth=model_config["classifier"]["params"]["num_features"],
        use_last_layer=True,
        inat=str(model_config["extractor"]["backbone"].get("weights")).endswith("resnet50_inat.pth"),
        seed=seed,
    )


def legacy_loss_function(
    model: PrototypeChooser,
    data: torch.Tensor,
    label: torch.Tensor,
    gumbel_scalar: int,
    mixup: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Extraction of the ProtoPool loss function, to simplify non-regression tests."""
    criterion = torch.nn.CrossEntropyLoss()
    label_p = label.cpu().numpy().tolist()

    targets_a, targets_b, lam = None, None, 1.0  # Dummy values to prevent pyright warnings

    if mixup:
        data, targets_a, targets_b, lam = mixup_data(data, label, 0.5)

    prob, min_distances, proto_presence = model(data, gumbel_scale=gumbel_scalar)

    if mixup:
        entropy_loss = lam * criterion(prob, targets_a) + (1 - lam) * criterion(prob, targets_b)
    else:
        entropy_loss = criterion(prob, label)

    orthogonal_loss = torch.Tensor([0]).cuda()

    for c in range(0, model.proto_presence.shape[0], 1000):
        orthogonal_loss_p = torch.nn.functional.cosine_similarity(
            model.proto_presence.unsqueeze(2)[c : c + 1000],
            model.proto_presence.unsqueeze(-1)[c : c + 1000],
            dim=1,
        ).sum()
        orthogonal_loss += orthogonal_loss_p
    orthogonal_loss = orthogonal_loss / (model.num_descriptive * model.num_classes) - 1

    proto_presence = proto_presence[label_p]
    inverted_proto_presence = 1 - proto_presence

    clst_loss_val = dist_loss(model, min_distances, proto_presence, model.num_descriptive)
    sep_loss_val = dist_loss(
        model, min_distances, inverted_proto_presence, model.num_prototypes - model.num_descriptive
    )

    l1_mask = 1 - torch.t(model.prototype_class_identity).to(data.device)
    l1 = (model.last_layer.weight * l1_mask).norm(p=1)

    return entropy_loss, orthogonal_loss, clst_loss_val, sep_loss_val, l1


def legacy_get_optimizers(
    legacy_model: nn.Module, base_lr: float = 5e-4
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    warm_optimizer = torch.optim.Adam(
        [
            {
                "params": legacy_model.add_on_layers.parameters(),
                "lr": 3 * base_lr,
                "weight_decay": 1e-3,
            },
            {"params": legacy_model.proto_presence, "lr": 3 * base_lr},
            {"params": legacy_model.prototype_vectors, "lr": 3 * base_lr},
        ]
    )
    joint_optimizer = torch.optim.Adam(
        [
            {
                "params": legacy_model.features.parameters(),
                "lr": base_lr / 10,
                "weight_decay": 1e-3,
            },
            {
                "params": legacy_model.add_on_layers.parameters(),
                "lr": 3 * base_lr,
                "weight_decay": 1e-3,
            },
            {"params": legacy_model.proto_presence, "lr": 3 * base_lr},
            {"params": legacy_model.prototype_vectors, "lr": 3 * base_lr},
        ]
    )
    push_optimizer = torch.optim.Adam(
        [
            {
                "params": legacy_model.last_layer.parameters(),
                "lr": base_lr / 5,
                "weight_decay": 1e-3,
            },
        ]
    )
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=5, gamma=0.1)
    return warm_optimizer, joint_optimizer, push_optimizer, joint_lr_scheduler


def legacy_push_prototypes(model: nn.Module, projection_loader: DataLoader, verbose: bool = False):
    model.eval()
    global_min_proto_dist = np.full(model.num_prototypes, np.inf)
    global_min_fmap_patches = np.zeros(
        [
            model.num_prototypes,
            model.prototype_shape[1],
            model.prototype_shape[2],
            model.prototype_shape[3],
        ]
    )

    proto_rf_boxes = np.full(shape=[model.num_prototypes, 6], fill_value=-1)
    proto_bound_boxes = np.full(shape=[model.num_prototypes, 6], fill_value=-1)

    search_batch_size = projection_loader.batch_size

    for push_iter, (search_batch_input, search_y) in tqdm(
        enumerate(projection_loader), "Legacy projection", disable=not verbose
    ):
        start_index_of_search_batch = push_iter * search_batch_size  # type:ignore

        update_prototypes_on_batch(
            search_batch_input=search_batch_input,
            start_index_of_search_batch=start_index_of_search_batch,
            model=model,
            global_min_proto_dist=global_min_proto_dist,
            global_min_fmap_patches=global_min_fmap_patches,
            proto_rf_boxes=proto_rf_boxes,
            proto_bound_boxes=proto_bound_boxes,
            class_specific=True,
            search_y=search_y,
            prototype_layer_stride=1,
        )

    prototype_update = np.reshape(global_min_fmap_patches, tuple(model.prototype_shape))
    model.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    return global_min_proto_dist


class Tester(CaBRNetCompatibilityTester):
    def __init__(self, methodName: str = "runTest"):
        super(Tester, self).__init__(arch="protopool", methodName=methodName)

    def test_model_init(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model: ProtoPool = CaBRNet.build_from_config(  # type: ignore
            self.model_config_file, seed=self.seed, compatibility_mode=True
        )

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(self.model_config_file, self.seed)

        self.assertModelEqual(legacy_model, cabrnet_model, non_deterministic_inference=True, gumbel_scale=10)

        # Draw a random set of inputs and test loss function
        x = torch.rand(16, 3, 224, 224).to(self.device)
        y = torch.randint(low=0, high=cabrnet_model.classifier.num_classes, size=(16,)).to(self.device)

        setup_rng(self.seed)
        entropy_loss, orthogonal_loss, clst_loss_val, sep_loss_val, l1 = legacy_loss_function(
            model=legacy_model, data=x, label=y, gumbel_scalar=30, mixup=cabrnet_model.training_config["use_mix_up"]
        )

        setup_rng(self.seed)
        mix_percentage, ys_mix = 1.0, None
        if cabrnet_model.training_config["use_mix_up"]:
            x, ys_mix, mix_percentage = batch_mixup(data=x, labels=y, alpha=0.5)
        pred, distances, proto_slot_probs = cabrnet_model.forward(x, gumbel_scale=30)

        _, batch_stats = cabrnet_model.loss(
            (pred, distances, proto_slot_probs), y, mixed_label=ys_mix, mix_percentage=mix_percentage
        )

        self.assertEqual(entropy_loss.item(), batch_stats["cross_entropy"])
        self.assertEqual(clst_loss_val.item(), batch_stats["cluster_cost"])
        self.assertEqual(sep_loss_val.item(), batch_stats["separation_cost"])
        self.assertEqual(l1.item(), batch_stats["l1"])
        self.assertEqual(orthogonal_loss.item(), batch_stats["orthogonal_loss"])

    def test_load_legacy_state_dict(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)
        cabrnet_model.load_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(self.model_config_file, seed=self.seed)
        legacy_model.load_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))

        self.assertModelEqual(legacy_model, cabrnet_model)

    def test_optimizers_init(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)
        optimizer_mngr = OptimizerManager.build_from_config(self.training_config_file, cabrnet_model)

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(self.model_config_file, self.seed)
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
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)
        training_config = load_config(self.training_config_file)
        cabrnet_model.register_training_params(training_config)
        optimizer_mngr = OptimizerManager.build_from_config(self.training_config_file, cabrnet_model)
        dataloaders = DatasetManager.get_dataloaders(config=self.dataset_config_file, sampling_ratio=SAMPLING_RATIO)
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
            optimizer_mngr.scheduler_step(epoch=epoch, metric=train_infos["train_set/accuracy"])

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(self.model_config_file, self.seed)
        warm_optimizer, joint_optimizer, last_layer_optimizer, _ = legacy_get_optimizers(legacy_model)

        # The following code is adapted from https://github.com/gmum/ProtoPool/blob/main/main.py
        legacy_model.features.requires_grad_(False)
        legacy_model.last_layer.requires_grad_(True)
        gumbel_time = cabrnet_model.training_config["gumbel_epochs"]
        start_val = cabrnet_model.training_config["gumbel_min_scale"]
        end_val = cabrnet_model.training_config["gumbel_max_scale"]
        epoch_interval = gumbel_time
        alpha = (end_val / start_val) ** 2 / epoch_interval
        mix_up_data = cabrnet_model.training_config["use_mix_up"]
        warmup_time = optimizer_mngr.periods["warmup"]["epoch_range"][1] + 1
        optimizer = warm_optimizer
        clst_weight, sep_weight = (
            cabrnet_model.loss_coefficients["clustering"],
            cabrnet_model.loss_coefficients["separability"],
        )
        steps, lr_scheduler = False, None
        train_loader = dataloaders["train_set"]

        def lambda1(e):
            return start_val * np.sqrt(alpha * e) if e < epoch_interval else end_val

        for epoch in tqdm(range(num_epochs), desc="Training legacy model", disable=not self.verbose):
            gumbel_scalar = lambda1(epoch)

            if warmup_time == epoch:
                legacy_model.features.requires_grad_(True)
                optimizer = joint_optimizer
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
                steps = True

            legacy_model.train()
            legacy_model.to(self.device)

            trn_tqdm = enumerate(train_loader, 0)
            for i, (data, label) in trn_tqdm:
                data = data.to(self.device)
                label = label.to(self.device)
                entropy_loss, orthogonal_loss, clst_loss_val, sep_loss_val, l1 = legacy_loss_function(
                    model=legacy_model, data=data, label=label, gumbel_scalar=gumbel_scalar, mixup=mix_up_data
                )

                loss = (
                    entropy_loss + clst_loss_val * clst_weight + sep_loss_val * sep_weight + 1e-4 * l1 + orthogonal_loss
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps and lr_scheduler is not None:
                lr_scheduler.step()

        # Compare
        self.assertGenericEqual(warm_optimizer.state_dict(), optimizer_mngr.optimizers["warmup_optimizer"].state_dict())
        self.assertGenericEqual(joint_optimizer.state_dict(), optimizer_mngr.optimizers["joint_optimizer"].state_dict())
        self.assertGenericEqual(
            last_layer_optimizer.state_dict(), optimizer_mngr.optimizers["last_layer_optimizer"].state_dict()
        )
        self.assertIsNotNone(lr_scheduler)
        self.assertGenericEqual(lr_scheduler.state_dict(), optimizer_mngr.schedulers["joint_optimizer"].state_dict())  # type: ignore
        self.assertModelEqual(legacy_model, cabrnet_model)

    def test_push_prototypes(self):
        # CaBRNet
        setup_rng(self.seed)
        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)
        dataloaders = DatasetManager.get_dataloaders(config=self.dataset_config_file, sampling_ratio=SAMPLING_RATIO)
        cabrnet_model.project(
            dataloader=dataloaders["projection_set"],
            device=self.device,
            verbose=self.verbose,
        )

        # Legacy
        setup_rng(self.seed)
        legacy_model = legacy_get_model(self.model_config_file, seed=self.seed)
        legacy_model.to(self.device)
        legacy_push_prototypes(model=legacy_model, projection_loader=dataloaders["projection_set"])

        self.assertModelEqual(legacy_model, cabrnet_model)

    def test_compatibility_mode(self):
        # Model with compatibility mode
        compatible_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=True)
        compatible_model.load_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))

        cabrnet_model = CaBRNet.build_from_config(self.model_config_file, seed=self.seed, compatibility_mode=False)
        cabrnet_model.load_state_dict(torch.load(self.legacy_state_dict, map_location="cpu"))

        # Get batch of images
        dataloaders = DatasetManager.get_dataloaders(config=self.dataset_config_file, sampling_ratio=SAMPLING_RATIO)
        xs, ys = next(iter(dataloaders["train_set"]))

        # Compare outputs and loss values
        # WARNING: Gumbel-Softmax use torch.rand so inference is NON-DETERMINISTIC, ie RNGs must be reset between models
        setup_rng(self.seed)
        expected_logits, expected_min_distances, expected_proto_presence = compatible_model(xs, gumbel_scale=3)
        expected_loss, expected_stats = compatible_model.loss(
            (expected_logits, expected_min_distances, expected_proto_presence), ys
        )

        setup_rng(self.seed)
        actual_logits, actual_min_distances, actual_proto_presence = cabrnet_model(xs, gumbel_scale=3)
        actual_loss, actual_stats = cabrnet_model.loss((actual_logits, actual_min_distances, actual_proto_presence), ys)

        self.assertGenericEqual(expected_min_distances, actual_min_distances, "Checking min distances")
        self.assertGenericEqual(expected_proto_presence, actual_proto_presence, "Checking proto presence")
        # Slight variation in the computation of the average distances between modes of compatibility leads to
        # different logit values
        max_logit_error = torch.max(torch.norm(expected_logits - actual_logits, p=2, dim=1)).item()
        if max_logit_error > 1e-5:
            self.fail(f"Max logit error too great: {max_logit_error}")

        # Remove orthogonal loss from comparison
        expected_stats["orthogonal_loss"] = 0
        actual_stats["orthogonal_loss"] = 0
        self.assertEqual(expected_stats, actual_stats, "Checking individual losses")


def main():
    torch.use_deterministic_algorithms(mode=True)
    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])
    unittest.main()


if __name__ == "__main__":
    main()
