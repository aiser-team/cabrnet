from argparse import Namespace
import unittest
import apps.train
import cabrnet.utils.data
import cabrnet.utils.save
from cabrnet.utils.parser import load_config
import os


class TestConfigsLoading(unittest.TestCase):
    def setUp(self):
        self.dataset_config_file = os.path.join("tests", "configs", "dataset.yml")
        self.training_config_file = os.path.join("tests", "configs", "training.yml")
        self.visualization_config_file = os.path.join(
            "tests", "configs", "visualization.yml"
        )
        self.protopnet_config_file = os.path.join(
            "tests", "configs", "protopnet_model_arch.yml"
        )

    def test_dataset_loading(self):
        load_config(self.dataset_config_file)

    def test_training_loading(self):
        load_config(self.training_config_file)

    def test_visualization_loading(self):
        load_config(self.visualization_config_file)

    def test_protopnet_loading(self):
        load_config(self.protopnet_config_file)


class TestTrainApp(unittest.TestCase):
    """
    Tests the train app of cabrnet.
    Launch a shortened training pipeline of supported cabrnet architectures.
    Checks the pipeline launching, early stopping and restoring,
    models loading and saving.
    """

    def setUp(self):
        # TODO: should we generate a config file on the fly using the cabrnet
        # utils, to ensure staying consistent with the configuration schema?
        self.dataset_config_file = os.path.join("tests", "configs", "dataset.yml")
        self.training_config_file = os.path.join("tests", "configs", "training.yml")
        self.visualization_config_file = os.path.join(
            "tests", "configs", "visualization.yml"
        )
        self.protopnet_config_file = os.path.join(
            "tests", "configs", "protopnet_model_arch.yml"
        )
        self.seed = 42
        do_epilogue_only = False
        self.args = Namespace(
            verbose=False,
            device="cpu",
            training=self.training_config_file,
            dataset=self.dataset_config_file,
            visualization=self.visualization_config_file,
            model_config=None,
            epilogue=do_epilogue_only,
            sanity_check=False,
            resume_dir=None,
            resume_from=None,
            config_dir=None,
            overwrite=True,
            output_dir=os.path.relpath("tests/runs/"),
            seed=self.seed,
            checkpoint_frequency=5,
            save_best="acc"
        )

    def test_protopnet_training(self):
        """
        Test the training of a protopnet model based on a torchvision-provided backbone.
        """
        self.args.model_config = self.protopnet_config_file
        apps.train.execute(self.args)
