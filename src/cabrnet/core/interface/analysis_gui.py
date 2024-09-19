import os
from typing import Any, Callable

import gradio as gr
import torch
import yaml
from gradio.components.base import Component
from loguru import logger
from PIL import Image
from torchvision.transforms import ToTensor

import cabrnet.core.evaluation.local_perturbation_analysis
import cabrnet.core.evaluation.relevance_analysis
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.interface.benchmark import (
    create_perturbation_benchmark_gui,
    create_pointing_benchmark_gui,
    get_perturbation_config,
    get_pointing_game_config,
)
from cabrnet.core.interface.utils import (
    create_browse_folder_component,
    gradio_convert_outputs,
)
from cabrnet.core.interface.visualization import (
    create_visualization_gui,
    get_visualization_config,
)
from cabrnet.core.utils.data import DatasetManager
from cabrnet.core.utils.monitoring import metrics_to_str
from cabrnet.core.utils.save import load_projection_info
from cabrnet.core.visualization.visualizer import SimilarityVisualizer


class CaBRNetAnalysisGUI:
    r"""Object in charge of building a GUI for the evaluation and explanation of CaBRNet networks.

    Attributes:
        model: Target model.
        dataloaders: Dataloaders.
        projection_info: Projection information.
        device: Hardware device.
        visualizer: Current patch visualizer.
    """
    DEFAULT_CHECKPOINT_DIR: str = "trained_models/ProtoTree/prototree_cub200_resnet50_depth9_s0_imported/imported/"
    DEFAULT_WORKING_DIR: str = "working_dir"

    def __init__(self):
        r"""Initializes the object."""
        self.model = None
        self.dataloaders = None
        self.projection_info = None
        self.device = "cpu"
        self.visualizer = None
        self._output_dir = CaBRNetAnalysisGUI.DEFAULT_WORKING_DIR
        self._prototype_dir = os.path.join(self._output_dir, "prototypes")

    def checkpoint_callback(self) -> Callable:
        r"""Returns a callback for loading a model and a dataset from a checkpoint directory."""

        def callback(checkpoint_path: str):
            # Set hardware device
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model = None
            self.dataloaders = None
            self.projection_info = None

            try:
                # Load the model
                self.model = CaBRNet.build_from_config(
                    config=os.path.join(checkpoint_path, CaBRNet.DEFAULT_MODEL_CONFIG),
                    state_dict_path=os.path.join(checkpoint_path, CaBRNet.DEFAULT_MODEL_STATE),
                )
                self.model.eval()
                self.model.to(self.device)

                # Load the datasets
                self.dataloaders = DatasetManager.get_dataloaders(
                    config=os.path.join(checkpoint_path, DatasetManager.DEFAULT_DATASET_CONFIG),
                    load_segmentation=True,
                )

                # Load projection info
                self.projection_info = load_projection_info(
                    filename=os.path.join(checkpoint_path, CaBRNet.DEFAULT_PROJECTION_INFO)
                )
                return gr.update(root_dir=self.dataloaders["test_set"].dataset.root)  # type: ignore

            except FileNotFoundError as e:
                logger.warning(e)
                pass

        return callback

    def output_directory_callback(self) -> Callable:
        r"""Returns a callback for controlling the output directory."""

        def callback(output_path: str) -> str:
            self._output_dir = output_path
            return self._output_dir

        return callback

    def evaluate_callback(self) -> Callable:
        r"""Returns a callback for evaluating a model on a test dataset."""

        def callback() -> str | None:
            if self.model is None or self.dataloaders is None:
                logger.warning("No checkpoint loaded (yet).")
                return None
            stats = self.model.evaluate(dataloader=self.dataloaders["test_set"], device=self.device, verbose=True)
            return metrics_to_str(stats)

        return callback

    def load_global_explanation_callback(self) -> Callable:
        r"""Returns a callback for loading an existing global explanation of a model."""

        def callback(checkpoint_path: str | None, current_image_path: str) -> str | None:
            if checkpoint_path is None:
                return None

            if os.path.isdir(os.path.join(checkpoint_path, "prototypes")):
                # Set prototype directory to existing directory
                self._prototype_dir = os.path.join(checkpoint_path, "prototypes")
                logger.info(f"Switching prototype directory to {self._prototype_dir}")

            if os.path.exists(os.path.join(checkpoint_path, "global_explanation.svg")):
                return os.path.join(checkpoint_path, "global_explanation.svg")
            # Do not change current image (if any)
            return current_image_path

        return callback

    def global_explanation_callback(self) -> Callable:
        r"""Returns a callback for generating the global explanation of a model."""

        def callback(gradio_outputs: dict[Component, Any]) -> str | None:
            if self.model is None or self.dataloaders is None or self.projection_info is None:
                logger.warning("No checkpoint loaded (yet).")
                return None

            # Extract configuration
            gradio_config = gradio_convert_outputs(gradio_outputs=gradio_outputs)

            # Builds visualizer
            visualization_config = get_visualization_config(gradio_config=gradio_config)
            visualizer = SimilarityVisualizer.build_from_config(config=visualization_config, model=self.model)

            # Update prototype directory
            self._prototype_dir = os.path.join(self._output_dir, "prototypes")

            # Build prototypes
            self.model.extract_prototypes(
                dataloader_raw=self.dataloaders["projection_set_raw"],
                dataloader=self.dataloaders["projection_set"],
                projection_info=self.projection_info,
                visualizer=visualizer,
                dir_path=self._prototype_dir,
                device=self.device,
                verbose=True,
            )

            # Save visualization config
            with open(
                os.path.join(self._prototype_dir, SimilarityVisualizer.DEFAULT_VISUALIZATION_CONFIG), "w"
            ) as fout:
                yaml.dump(visualization_config, fout)

            # Generate explanation
            self.model.explain_global(
                prototype_dir=self._prototype_dir,
                output_dir=self._output_dir,
                output_format="svg",
            )
            return os.path.join(self._output_dir, "global_explanation.svg")

        return callback

    def local_explanation_callback(self) -> Callable:
        r"""Returns a callback for generating the local explanation of a model."""

        def callback(gradio_outputs: dict[Component, Any]) -> Image.Image | None:
            if self.model is None or self.dataloaders is None or self.projection_info is None:
                logger.warning("No checkpoint loaded (yet).")
                return None

            # Extract configuration
            gradio_config = gradio_convert_outputs(gradio_outputs=gradio_outputs)

            # Builds visualizer
            visualization_config = get_visualization_config(gradio_config=gradio_config)
            visualizer = SimilarityVisualizer.build_from_config(config=visualization_config, model=self.model)

            # Recover preprocessing function
            preprocess = getattr(self.dataloaders["test_set"].dataset, "transform", ToTensor())

            # Dedicated directory for target image
            output_dir = os.path.join(self._output_dir, "local_explanation")

            # Generate explanation
            self.model.explain(
                img=gradio_config["input_image"],
                preprocess=preprocess,
                visualizer=visualizer,
                prototype_dir=self._prototype_dir,
                output_dir=output_dir,
                output_format="png",
                device=self.device,
                exist_ok=True,
            )
            return Image.open(os.path.join(output_dir, "explanation.png"))

        return callback

    def benchmark_callback(self) -> Callable:
        r"""Returns a callback for performing single-shot analysis on an image."""

        def callback(gradio_outputs: dict[Component, Any]) -> Image.Image | None:
            if self.model is None or self.dataloaders is None:
                logger.warning("No checkpoint loaded (yet).")
                return None

            # Extract configuration
            gradio_config = gradio_convert_outputs(gradio_outputs=gradio_outputs)

            if gradio_config["input_image"] is None:
                logger.warning("No image loaded (yet).")
                return None

            # Builds visualizer
            visualization_config = get_visualization_config(gradio_config=gradio_config)
            visualizer = SimilarityVisualizer.build_from_config(config=visualization_config, model=self.model)

            # Recover preprocessing function
            preprocess = getattr(self.dataloaders["test_set"].dataset, "transform", ToTensor())

            # Analyze image
            if gradio_config["benchmark_selection"] == "Local perturbation analysis":
                cabrnet.core.evaluation.local_perturbation_analysis.analyze(
                    model=self.model,
                    img=gradio_config["input_image"],
                    img_id="",
                    preprocess=preprocess,
                    visualizer=visualizer,
                    device=self.device,
                    debug_dir=os.path.join(self._output_dir, "analysis", "local_perturbation"),
                    debug_format="png",
                    prototype_dir=self._prototype_dir,
                    **(get_perturbation_config(gradio_config)["local_perturbation_analysis"]),
                )
                return Image.open(
                    os.path.join(self._output_dir, "analysis", "local_perturbation", "img_sensitivity.png")
                )
            else:
                if gradio_config["segmentation"] is None:
                    logger.warning("Segmentation unavailable.")
                    return None

                cabrnet.core.evaluation.relevance_analysis.analyze(
                    model=self.model,
                    img=gradio_config["input_image"],
                    img_id="",
                    seg=gradio_config["segmentation"],
                    preprocess=preprocess,
                    visualizer=visualizer,
                    device=self.device,
                    debug_dir=os.path.join(self._output_dir, "analysis", "relevance"),
                    debug_format="png",
                    prototype_dir=self._prototype_dir,
                    **(get_pointing_game_config(gradio_config)["relevance_analysis"]),
                )
                return Image.open(os.path.join(self._output_dir, "analysis", "relevance", "img_relevance_analysis.png"))

        return callback

    def create_gui(self) -> gr.Blocks:
        r"""Creates the main GUI for loading, evaluating and explaining a trained model.

        Returns:
            A gradio Block components containing the GUI.
        """
        with gr.Blocks() as block:
            with gr.Group():
                with gr.Row():
                    model_select = create_browse_folder_component(
                        label="Checkpoint directory",
                        default_dir=CaBRNetAnalysisGUI.DEFAULT_CHECKPOINT_DIR,
                        allow_creation=False,
                        key="checkpoint",
                    )
                    stats = gr.Textbox(label="Statistics", interactive=False)
                evaluate_button = gr.Button(value="Evaluate", icon="docs/logos/cabrnet.svg")
                evaluate_button.click(self.evaluate_callback(), outputs=[stats])
                output_select = create_browse_folder_component(
                    label="Working directory",
                    default_dir=CaBRNetAnalysisGUI.DEFAULT_WORKING_DIR,
                    allow_creation=True,
                    key="working_dir",
                )
                output_select.change(self.output_directory_callback(), inputs=output_select)
                # Set output directory to checkpoint directory by default
                model_select.change(self.output_directory_callback(), inputs=model_select, outputs=output_select)

            with gr.Group():
                # Global explanation
                visualization_gui = create_visualization_gui(default_attribution="prp")
                explain_button = gr.Button("Generate global explanation", icon="docs/logos/cabrnet.svg")
                global_explanation = gr.Image(label="Global explanation", interactive=False)
                explain_button.click(
                    self.global_explanation_callback(), inputs=visualization_gui, outputs=global_explanation
                )
                model_select.change(
                    self.load_global_explanation_callback(),
                    inputs=[model_select, global_explanation],
                    outputs=global_explanation,
                )

            with gr.Group():
                with gr.Row():
                    # Input image
                    image_selection = gr.FileExplorer(
                        root_dir="data/CUB_200_2011/images",
                        key="img_path",
                        interactive=True,
                        file_count="single",
                        height=200,
                    )
                    input_image = gr.Image(
                        label="Input image",
                        type="pil",
                        height=200,
                        value="data/CUB_200_2011/images/054.Blue_Grosbeak/Blue_Grosbeak_0037_36794.jpg",
                        key="input_image",
                    )
                    object_segmentation = gr.Image(
                        label="Object segmentation",
                        type="pil",
                        height=200,
                        key="segmentation",
                    )
                model_select.change(self.checkpoint_callback(), inputs=[model_select], outputs=[image_selection])
                image_selection.change(
                    lambda x: Image.open(x) if x is not None else None, inputs=[image_selection], outputs=[input_image]
                )

                # Automatic segmentation retrieval
                def auto_segmentation_retrieval(img_path: str, segmentation: Image.Image) -> Image.Image | None:
                    if img_path is None:
                        return None
                    # Simple search and replace
                    if "dataset/test_full" in img_path:
                        tentative_path = img_path.replace("/dataset/test_full", "/segmentations/")
                        if os.path.isfile(tentative_path):
                            return Image.open(tentative_path)
                        # Try with replacing file extension
                        tentative_path = tentative_path.replace(".jpg", ".png")
                        if os.path.isfile(tentative_path):
                            return Image.open(tentative_path)
                    return None

                image_selection.change(
                    auto_segmentation_retrieval,
                    inputs=[image_selection, object_segmentation],
                    outputs=[object_segmentation],
                )
                explanation = gr.Image(label="Local explanation", height=400, interactive=False)
                explain_button = gr.Button("Generate local explanation", icon="docs/logos/cabrnet.svg")
                local_explanation_inputs: set[Component] = {input_image}
                local_explanation_inputs.update(visualization_gui)
                explain_button.click(
                    self.local_explanation_callback(), inputs=local_explanation_inputs, outputs=explanation
                )

            with gr.Group():
                # Evaluation metrics
                benchmark_selection = gr.Dropdown(
                    label="Benchmark selection",
                    choices=["Local perturbation analysis", "Pointing game analysis"],
                    value="Local perturbation analysis",
                    interactive=True,
                    key="benchmark_selection",
                )
                num_prototype = gr.Slider(
                    label="Number of prototypes",
                    value=2,
                    minimum=1,
                    maximum=10,
                    step=1,
                    interactive=True,
                    key="num_prototypes",
                )
                analyze_button = gr.Button("Analyze", icon="docs/logos/cabrnet.svg")

                # Add configuration for benchmark
                with gr.Row():
                    perturbation_config = create_perturbation_benchmark_gui(
                        benchmark_selection=benchmark_selection, visible=True
                    )
                    pointing_game_config = create_pointing_benchmark_gui(
                        benchmark_selection=benchmark_selection, visible=False
                    )
                    # Output image
                    analysis_output = gr.Image(scale=2)

                benchmark_inputs = {input_image, object_segmentation, benchmark_selection, num_prototype}
                benchmark_inputs.update(visualization_gui)
                benchmark_inputs.update(perturbation_config)
                benchmark_inputs.update(pointing_game_config)
                analyze_button.click(
                    self.benchmark_callback(),
                    inputs=benchmark_inputs,
                    outputs=analysis_output,
                )
        return block


def main():
    r"""Main GUI function."""
    cabrnet_gui = CaBRNetAnalysisGUI()
    block = cabrnet_gui.create_gui()
    block.launch()


if __name__ == "__main__":
    main()
