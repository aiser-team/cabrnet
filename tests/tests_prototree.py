import os
import random
import unittest

import numpy as np
import torch

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.save import load_projection_info
from cabrnet.core.visualization.visualizer import SimilarityVisualizer


def setup_rng(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class TestProtoTree(unittest.TestCase):
    def __init__(self, methodName: str = "runTest"):
        super(TestProtoTree, self).__init__(methodName=methodName)

    def test_train_exists(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        if not os.path.isdir(os.path.join(test_dir, "..", "runs/mnist_prototree/")):
            self.fail(
                f"Missing example ProtoTree folder at {os.path.join(test_dir, '..', 'runs/mnist_prototree/')}."
                "Please run the following command:\n"
                "cabrnet train --device cpu --seed 42 --verbose --logger-level INFO "
                "--model-config configs/prototree/mnist/model_arch.yml "
                "--dataset configs/prototree/mnist/dataset.yml "
                "--training configs/prototree/mnist/training.yml "
                "--output-dir runs/mnist_prototree"
            )

    def test_explain_global(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        run_dir = os.path.join(test_dir, "..", "runs/mnist_prototree")
        model_config_file = os.path.join(run_dir, "final", CaBRNet.DEFAULT_MODEL_CONFIG)
        model_state_dict = os.path.join(run_dir, "final", CaBRNet.DEFAULT_MODEL_STATE)
        projection_info_file = os.path.join(run_dir, "final", CaBRNet.DEFAULT_PROJECTION_INFO)
        dataset_config = os.path.join(run_dir, "final", DatasetManager.DEFAULT_DATASET_CONFIG)
        visualization_config = os.path.join(test_dir, "..", "configs/explanation/mnist_visualization.yml")
        projection_info = load_projection_info(projection_info_file)
        dataloaders = DatasetManager.get_dataloaders(config=dataset_config)
        prototype_dir = os.path.join(run_dir, "prototypes")
        output_dir = os.path.join(run_dir, "global_explainations")

        setup_rng(42)

        model = CaBRNet.build_from_config(config=model_config_file, state_dict_path=model_state_dict)
        model.extract_prototypes(
            dataloader_raw=dataloaders["projection_set_raw"],
            dataloader=dataloaders["projection_set"],
            projection_info=projection_info,
            visualizer=SimilarityVisualizer.build_from_config(config=visualization_config, model=model),
            dir_path=prototype_dir,
            device="cpu",
        )
        model.explain_global(prototype_dir=prototype_dir, output_dir=output_dir, device="cpu")

    def test_explain_local(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        run_dir = os.path.join(test_dir, "..", "runs/mnist_prototree")
        model_config_file = os.path.join(run_dir, "final", CaBRNet.DEFAULT_MODEL_CONFIG)
        model_state_dict = os.path.join(run_dir, "final", CaBRNet.DEFAULT_MODEL_STATE)
        dataset_config = os.path.join(run_dir, "final", DatasetManager.DEFAULT_DATASET_CONFIG)
        visualization_config = os.path.join(test_dir, "..", "configs/explanation/mnist_visualization.yml")
        prototype_dir = os.path.join(run_dir, "prototypes")
        output_dir = os.path.join(run_dir, "global_explainations")

        setup_rng(42)

        model = CaBRNet.build_from_config(config=model_config_file, state_dict_path=model_state_dict)
        model.explain(
            img=os.path.join(test_dir, "..", "examples/images/mnist_sample.png"),
            preprocess=DatasetManager.get_dataset_transform(config=dataset_config, dataset="test_set"),
            visualizer=SimilarityVisualizer.build_from_config(config=visualization_config, model=model),
            prototype_dir=prototype_dir,
            output_dir=output_dir,
            exist_ok=True,
            device="cpu",
        )


def main():
    unittest.main()


if __name__ == "__main__":
    main()
