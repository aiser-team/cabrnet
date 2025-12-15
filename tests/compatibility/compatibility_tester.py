import unittest
import torch
import torch.nn as nn
import numpy as np
import random
import os
from typing import Any

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.optimizers import OptimizerManager
from torch.utils.data import Dataset, Subset
from loguru import logger

# Global sampling ratio used to speed up verification
SAMPLING_RATIO = 6000


class DummyLogger:
    checkpoint_dir: str = ""

    def set_checkpoint_dir(self, dir: str):
        pass

    def log_message(self, message: str):
        pass

    def __call__(self, _: str):
        pass


def setup_rng(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_subset(dataset: Dataset, sampling_ratio: int) -> Dataset | Subset:
    if sampling_ratio > 1:
        # Apply data sub-selection
        selected_indices = [idx for idx in range(len(dataset))][::sampling_ratio]  # type: ignore
        return Subset(dataset=dataset, indices=selected_indices)
    return dataset


class CaBRNetCompatibilityTester(unittest.TestCase):
    def __init__(self, arch: str, methodName: str = "runTest"):
        super(CaBRNetCompatibilityTester, self).__init__(methodName=methodName)

        # Test configuration
        test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", arch)
        self.model_config_file = os.path.join(test_dir, CaBRNet.DEFAULT_MODEL_CONFIG)
        self.dataset_config_file = os.path.join(test_dir, DatasetManager.DEFAULT_DATASET_CONFIG)
        self.training_config_file = os.path.join(test_dir, OptimizerManager.DEFAULT_TRAINING_CONFIG)
        self.legacy_state_dict = os.path.join(test_dir, "legacy_state.pth")
        self.output_dir = os.path.join(test_dir, "output")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.seed: int = 42
        self.verbose: bool = True

    def assertTensorEqual(self, expected: torch.Tensor, actual: torch.Tensor, msg: str | None = None):
        if msg is None:
            msg = ""
        if actual.size() != expected.size():
            self.fail(f"{msg} Mismatching tensor sizes: {actual.size()} v. {expected.size()}.")
        elif not torch.all(torch.eq(expected, actual)):
            self.fail(f"{msg} Mismatching tensors (all close? {torch.allclose(expected,actual)}).")

    def assertModelEqual(
        self, expected: nn.Module, actual: nn.Module, non_deterministic_inference: bool = False, **kwargs
    ):
        expected.eval()
        actual.eval()
        expected.to(self.device)
        actual.to(self.device)
        x = torch.rand(16, 3, 224, 224).to(self.device)
        with torch.no_grad():
            if non_deterministic_inference:
                setup_rng(self.seed)
            y_e = expected(x, **kwargs)
            if non_deterministic_inference:
                setup_rng(self.seed)
            y_a = actual(x, **kwargs)
        self.assertGenericEqual(y_e, y_a, msg="Checking model outputs.")

    def assertGenericEqual(self, expected: Any, actual: Any, msg: str | None = None):
        """Generic comparison function for complex data structures"""
        if msg is None:
            msg = ""
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
                self.fail(f"{msg} Mismatching list lengths: {len(expected)} v. {len(actual)}.")
            for index, (ex, ac) in enumerate(zip(expected, actual)):
                self.assertGenericEqual(ex, ac, f"{msg} Checking list (index {index}).")
        elif isinstance(expected, torch.Tensor):
            self.assertTensorEqual(expected, actual, msg)
        elif isinstance(expected, np.ndarray):
            if not (expected == actual).all():
                self.fail(f"{msg} Mismatching np arrays (all close? {np.allclose(expected, actual)})")
        else:
            self.assertEqual(expected, actual, f"{msg} Checking generic type.")

    def assertStateEqual(self, expected: dict[str, Any], actual: dict[str, Any]):
        equal = True
        for key, ref_tensor in expected.items():
            if key not in actual.keys():
                logger.error(f"Missing parameter {key} in state under test")
                equal = False
                continue
            tst_tensor = actual[key]
            if not torch.equal(ref_tensor, tst_tensor):
                logger.error(f"Parameter {key} differ. (all close? {torch.allclose(ref_tensor, tst_tensor)})")
                equal = False
        for key in actual.keys():
            if key not in expected.keys():
                logger.error(f"Extra parameter {key} in state under test")
                equal = False
        self.assertEqual(equal, True, f"State dictionaries do not match")
