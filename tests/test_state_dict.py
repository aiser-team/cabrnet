"""Tests for loading local, nested, and flattened state dictionaries."""

import unittest
from copy import deepcopy

import torch

from cabrnet.archs.generic.model import CaBRNet
from tests.backbones import TinyBackbone


class TestStateDictLoading(unittest.TestCase):
    """Tests state-dictionary loading for supported architectures."""

    @staticmethod
    def model_config(architecture: str) -> dict:
        """Builds a minimal configuration for an architecture.

        Args:
            architecture (str): Architecture identifier.

        Returns:
            Configuration dictionary for the requested architecture.
        """
        architectures = {
            "protopnet": {
                "top_arch": {"module": "cabrnet.archs.protopnet.model", "name": "ProtoPNet"},
                "similarity": {"name": "ProtoPNetSimilarity"},
                "classifier": {
                    "module": "cabrnet.archs.protopnet.decision",
                    "name": "ProtoPNetClassifier",
                    "params": {"num_classes": 2, "num_proto_per_class": 1},
                },
            },
            "pipnet": {
                "top_arch": {"module": "cabrnet.archs.pipnet.model", "name": "PIPNet"},
                "similarity": {"name": "None"},
                "classifier": {
                    "module": "cabrnet.archs.pipnet.decision",
                    "name": "PIPNetClassifier",
                    "params": {"num_classes": 2},
                },
            },
        }
        return {
            **architectures[architecture],
            "extractor": {
                "backbone": {"module": TinyBackbone.__module__, "arch": "TinyBackbone", "weights": None},
                "convnet": {"source_layer": "features.0"},
            },
        }

    def assert_state_dict_equal(self, expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> None:
        """Asserts equality for every tensor in two state dictionaries.

        Args:
            expected (dict): Expected state dictionary.
            actual (dict): Actual state dictionary.
        """
        self.assertEqual(expected.keys(), actual.keys())
        for key, value in expected.items():
            torch.testing.assert_close(value, actual[key])

    def test_loads_backbone_state_dicts(self) -> None:
        """Tests feature-extractor backbone checkpoint loading.

        Initializes a standalone backbone, and try to load it into a CaBRNet model.
        """
        for architecture in ("protopnet", "pipnet"):
            with self.subTest(architecture=architecture):
                source_backbone = TinyBackbone()
                source_state_dict = source_backbone.state_dict()
                local_state_dict = source_state_dict
                nested_state_dict = {"extractor": {"convnet": source_state_dict}}
                flattened_state_dict = {f"extractor.convnet.{key}": value for key, value in source_state_dict.items()}

                for checkpoint_type, checkpoint in [
                    ("local", local_state_dict),
                    ("nested", nested_state_dict),
                    ("flattened", flattened_state_dict),
                ]:
                    with self.subTest(checkpoint_type=checkpoint_type):
                        target_model = CaBRNet.build_from_config(deepcopy(self.model_config(architecture)))
                        target_model.load_submodule_state_dict("extractor.convnet", checkpoint)
                        self.assert_state_dict_equal(
                            source_backbone.state_dict(), target_model.extractor.convnet.state_dict()
                        )

    def test_loads_complete_model_state_dicts(self) -> None:
        """Tests complete-model checkpoint loading.

        Checks if loading an exported state dict recovers the original model.
        """
        for architecture in ("protopnet", "pipnet"):
            with self.subTest(architecture=architecture):
                source_model = CaBRNet.build_from_config(self.model_config(architecture))
                target_model = CaBRNet.build_from_config(deepcopy(self.model_config(architecture)))

                target_model.load_state_dict(source_model.state_dict())

                self.assert_state_dict_equal(source_model.state_dict(), target_model.state_dict())


if __name__ == "__main__":
    unittest.main()
