"""Unit tests for named dataset collections (multiple test/validation sets)."""

import os
import unittest
from pathlib import Path

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager


class TestMultiDatasetCollections(unittest.TestCase):
    """Tests that named test/validation-set collections work end to end with a freshly built model."""

    def setUp(self) -> None:
        """Loads dataloaders for a config with two named validation sets, plus a freshly built ProtoPNet model."""
        config_dir = Path(os.path.dirname(os.path.realpath(__file__))) / "configs"
        self.dataloaders = DatasetManager.get_dataloaders(config=config_dir / "dataset_multi_val.yml")
        self.model = CaBRNet.build_from_config(config=config_dir / "model_arch_multi_val.yml")
        self.model.eval()

    def test_declares_two_named_validation_sets(self) -> None:
        """Flattens 'val_sets' into two distinct, correctly-prefixed dataloader names."""
        self.assertEqual(
            sorted(DatasetManager.val_set_names(self.dataloaders)), ["val_sets/slice_a", "val_sets/slice_b"]
        )

    def test_declares_one_legacy_test_set(self) -> None:
        """Keeps the legacy single 'test_set' unaffected by the 'val_sets' collection."""
        self.assertEqual(DatasetManager.test_set_names(self.dataloaders), ["test_set"])

    def test_evaluates_each_named_validation_set(self) -> None:
        """Runs a full evaluation pass on each named validation set with an untrained model."""
        for val_set_name in DatasetManager.val_set_names(self.dataloaders):
            stats = self.model.evaluate(dataloaders=self.dataloaders, dataset_name=val_set_name, device="cpu")
            self.assertIn(f"{val_set_name}/loss", stats)
            self.assertIn(f"{val_set_name}/accuracy", stats)


if __name__ == "__main__":
    unittest.main()
