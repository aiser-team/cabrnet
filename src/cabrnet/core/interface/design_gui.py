import os
from typing import Any, Callable

import gradio as gr
import torchvision.models as torch_models
import yaml
from loguru import logger

from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.interface.utils import create_browse_folder_component
from cabrnet.core.utils.init import layer_init_functions
from cabrnet.core.utils.prototypes import prototype_init_modes
from cabrnet.core.utils.tree import leaf_init_modes

# List and configuration of supported add-on layers. For each parameter, indicate the type of gradio component to use.
supported_addon_layers = {
    "Conv2d": {
        "out_channels": ("number", 0),
        "kernel_size": ("slider", 1, 1, 10),
        "stride": ("slider", 1, 1, 10),
        "padding": ("slider", 0, 0, 10),
        "dilation": ("slider", 1, 0, 10),
        "groups": ("number", 1),
        "bias": ("checkbox", True),
        "padding_mode": ("textbox", "zeros"),
    },
    "AvgPool2d": {"kernel_size": ("slider", 1, 1, 10), "stride": ("slider", 1, 1, 10), "padding": ("slider", 0, 0, 10)},
    "MaxPool2d": {
        "kernel_size": ("slider", 1, 1, 10),
        "stride": ("slider", 1, 1, 10),
        "padding": ("slider", 0, 0, 10),
        "dilation": ("slider", 1, 0, 10),
    },
    "ReLU": {},
    "Sigmoid": {},
    "Tanh": {},
}

# List and configuration of supported architectures. For each parameter, indicate the type of gradio component to use.
supported_architectures = {
    "ProtoPNet": {
        "top_arch": {"module": "cabrnet.archs.protopnet.model", "name": "ProtoPNet"},
        "classifier": {
            "module": "cabrnet.archs.protopnet.decision",
            "name": "ProtoPNetClassifier",
            "params": {
                "num_classes": ("number", 100),
                "num_features": ("number", 128),
                "proto_init_mode": ("dropdown", "SHIFTED_NORMAL", prototype_init_modes),
                "num_proto_per_class": ("slider", 10, 1, 30),
                "incorrect_class_penalty": ("number", -0.5),
                "compatibility_mode": ("checkbox", False),
            },
        },
    },
    "ProtoTree": {
        "top_arch": {"module": "cabrnet.archs.prototree.model", "name": "ProtoTree"},
        "classifier": {
            "module": "cabrnet.archs.prototree.decision",
            "name": "ProtoTreeClassifier",
            "params": {
                "num_classes": ("number", 100),
                "num_features": ("number", 128),
                "proto_init_mode": ("dropdown", "SHIFTED_NORMAL", prototype_init_modes),
                "depth": ("slider", 9, 2, 15),
                "leaves_init_mode": ("dropdown", "ZEROS", leaf_init_modes),
                "log_probabilities": ("checkbox", False),
            },
        },
    },
}


class CaBRNetDesignGUI:
    r"""Object in charge of building a GUI for the design of CaBRNet networks.

    Attributes:
        top_arch_config: Configuration of the top-module.
        extractor_config: Configuration of the feature extractor.
        classifier_config: Configuration of the classifier.
        output_dir: Path to output directory.
        output_file: Output filename.
    """
    DEFAULT_WORKING_DIR: str = "working_dir"
    MAX_ADDON_LAYERS: int = 10

    def __init__(self):
        r"""Initializes the object."""
        self.top_arch_config = {"module": None, "name": None}
        self.extractor_config = {"backbone": {"arch": "resnet50", "layer": None, "weights": None}}
        self.classifier_config = {"module": None, "name": None, "params": {}}
        self.output_dir = self.DEFAULT_WORKING_DIR
        self.output_file = CaBRNet.DEFAULT_MODEL_CONFIG

    def output_selection_callback(self) -> Callable:
        r"""Returns a callback for updating the output directory."""

        def callback(path: str) -> None:
            self.output_dir = path

        return callback

    def output_file_callback(self) -> Callable:
        r"""Returns a callback for updating the output filename."""

        def callback(name: str) -> None:
            self.output_file = name

        return callback

    def load_output_file_callback(self) -> Callable:
        r"""Returns a callback for loading the output file."""

        def callback() -> str | None:
            filepath = os.path.join(self.output_dir, self.output_file)
            if os.path.isfile(filepath):
                return open(filepath, "r").read()
            return None

        return callback

    def save_callback(self) -> Callable:
        r"""Returns a callback for saving the model configuration inside a YAML file."""

        def callback() -> str:
            config = {
                "top_arch": self.top_arch_config,
                "extractor": self.extractor_config,
                "classifier": self.classifier_config,
            }
            logger.debug(f"Model configuration: {config}")
            with open(os.path.join(self.output_dir, self.output_file), "w") as fout:
                yaml.dump(config, fout, sort_keys=False)

            return open(os.path.join(self.output_dir, self.output_file), "r").read()

        return callback

    def backbone_update_callback(self, field: str) -> Callable:
        r"""Returns a callback for modifying the backbone configuration.

        Args:
            field (str): Name of the configuration field.
        """

        def callback(value) -> None:
            self.extractor_config["backbone"][field] = value
            return None

        return callback

    def backbone_weights_callback(self) -> Callable:
        r"""Returns a callback for modifying the backbone weights."""

        def callback(weight_type: str, filepath: str | None) -> None:
            weight_value = weight_type if weight_type != "From file" else filepath
            self.extractor_config["backbone"]["weights"] = weight_value
            return None

        return callback

    def layer_type_callback(self, layer_index: int) -> Callable:
        r"""Returns a callback for changing a layer configuration based on its type.

        Args:
            layer_index (int): Index of the layer.
        """

        def callback(layer_type: str) -> None:
            self.extractor_config["add_on"][f"layer_{layer_index}"]["type"] = layer_type
            if supported_addon_layers[layer_type]:
                # Add default parameters
                self.extractor_config["add_on"][f"layer_{layer_index}"]["params"] = {}
                for key, value in supported_addon_layers[layer_type].items():
                    self.extractor_config["add_on"][f"layer_{layer_index}"]["params"][key] = value[1]

        return callback

    def layer_init_callback(self) -> Callable:
        r"""Returns a callback for changing the initialization mode of a layer."""

        def callback(mode: str) -> None:
            r"""Changes layer init function"""
            self.extractor_config["add_on"]["init_mode"] = mode

        return callback

    def layer_params_callback(self, layer_index: int, param_name: str) -> Callable:
        r"""Returns a callback for changing the parameters of a layer.

        Args:
            layer_index (int): Index of the layer.
            param_name (str): Parameter name.
        """

        def callback(param: str) -> None:
            self.extractor_config["add_on"][f"layer_{layer_index}"]["params"][param_name] = param

        return callback

    def architecture_callback(self) -> Callable:
        r"""Returns a callback for changing the model architecture."""

        def callback(arch: str) -> None:
            r"""Changes the model architecture"""
            self.top_arch_config = supported_architectures[arch]["top_arch"]
            self.classifier_config["module"] = supported_architectures[arch]["classifier"]["module"]
            self.classifier_config["name"] = supported_architectures[arch]["classifier"]["name"]
            self.classifier_config["params"] = {}
            for key, value in supported_architectures[arch]["classifier"]["params"].items():
                self.classifier_config["params"][key] = value[1]

        return callback

    def architecture_param_callback(self, param_name: str) -> Callable:
        """Returns a callback for changing value of individual architecture parameter.

        Args:
            param_name (str): Parameter name.
        """

        def callback(param: str) -> None:
            self.classifier_config["params"][param_name] = param

        return callback

    def addon_configuration_callback(self) -> Callable:
        """Returns a callback for refreshing the add-on configuration."""

        def callback(num_layers: int) -> None:
            r"""Refreshes model configuration based on chosen number of layers"""
            if num_layers == 0:
                if self.extractor_config.get("add_on") is not None:
                    del self.extractor_config["add_on"]
            else:
                if self.extractor_config.get("add_on") is None:
                    self.extractor_config["add_on"] = {}
                for addon_index in range(self.MAX_ADDON_LAYERS):
                    if (
                        addon_index >= num_layers
                        and self.extractor_config["add_on"].get(f"layer_{addon_index}") is not None
                    ):
                        del self.extractor_config["add_on"][f"layer_{addon_index}"]
                    elif addon_index < num_layers:
                        self.extractor_config["add_on"].setdefault(f"layer_{addon_index}", {})

        return callback

    @staticmethod
    def add_generic_selector(selector_config: tuple, param_name: str, visibility: bool = True) -> Any:
        r"""Creates a generic selector based on a configuration tuple.

        Args:
            selector_config (tuple): Configuration of the selector.
            param_name (str): Parameter name.
            visibility (bool, optional): If True, the selector is initially visible. Default: True.
        """
        selector = None
        match selector_config[0]:
            case "slider":
                selector = gr.Slider(
                    minimum=selector_config[2],
                    maximum=selector_config[3],
                    step=1,
                    value=selector_config[1],
                    label=param_name,
                    visible=visibility,
                    interactive=True,
                )
            case "textbox":
                selector = gr.Textbox(value=selector_config[1], label=param_name, visible=visibility, interactive=True)
            case "number":
                selector = gr.Number(value=selector_config[1], label=param_name, visible=visibility, interactive=True)
            case "checkbox":
                selector = gr.Checkbox(value=selector_config[1], label=param_name, visible=visibility, interactive=True)
            case "dropdown":
                selector = gr.Dropdown(
                    value=selector_config[1],
                    choices=selector_config[2],
                    label=param_name,
                    visible=visibility,
                    interactive=True,
                )
            case _:
                logger.error(f"Unsupported selector type: {selector_config[0]}")
        return selector

    def create_gui(self) -> gr.Blocks:
        r"""Creates the main GUI for configuring a CaBRNet model.

        Returns:
            A gradio Block components containing the GUI.
        """
        with gr.Blocks(title="CaBRNet model configuration") as block:
            # Output selection
            with gr.Row():
                output_directory = create_browse_folder_component(
                    label="Output directory", default_dir=".", allow_creation=True
                )
                output_file = gr.Textbox(value="model_arch.yml", label="Output file name", interactive=True)
                output_directory.change(self.output_selection_callback(), inputs=output_directory)
                output_file.change(self.output_file_callback(), inputs=output_file)

            with gr.Row():
                with gr.Column():
                    # Architecture selection
                    arch_selection = gr.Dropdown(label="Architecture", choices=list(supported_architectures.keys()))
                    arch_selection.change(self.architecture_callback(), inputs=[arch_selection])

                    with gr.Accordion("Backbone configuration", open=True):
                        backbone_list = torch_models.list_models()
                        default_backbone_name = self.extractor_config["backbone"]["arch"]
                        default_backbone = torch_models.get_model(default_backbone_name, weights=None)
                        with gr.Row():
                            with gr.Column():
                                backbone_selection = gr.Dropdown(
                                    label="Architecture",
                                    choices=list(backbone_list),
                                    value=default_backbone_name,
                                    interactive=True,
                                )
                                weight_selection = gr.Dropdown(
                                    label="Pretrained weights",
                                    choices=["From file"]
                                    + torch_models.get_model_weights(
                                        default_backbone_name
                                    )._member_names_,  # type:ignore
                                    interactive=True,
                                )
                                layer_selection = gr.Dropdown(
                                    label="Layer name",
                                    choices=[layer[0] for layer in default_backbone.named_modules()],
                                    interactive=True,
                                )

                            # Custom weights pop-up that appears only when selecting the "From file" option
                            weight_file = gr.FileExplorer(
                                root_dir=".",
                                visible=False,
                                label="Custom weight file",
                                height=300,
                                file_count="single",
                                interactive=True,
                            )

                        # Update extractor configuration
                        backbone_selection.change(self.backbone_update_callback("arch"), inputs=[backbone_selection])
                        layer_selection.change(self.backbone_update_callback("layer"), inputs=[layer_selection])
                        weight_selection.change(
                            self.backbone_weights_callback(), inputs=[weight_selection, weight_file]
                        )
                        weight_file.change(self.backbone_weights_callback(), inputs=[weight_selection, weight_file])

                        # Update list of available pretrained weights based on backbone selection
                        backbone_selection.change(
                            lambda choice: gr.update(
                                choices=["From file"] + torch_models.get_model_weights(choice)._member_names_,
                                value=None,
                            ),
                            inputs=[backbone_selection],
                            outputs=[weight_selection],
                        )
                        # Update list of available layers based on backbone selection
                        backbone_selection.change(
                            lambda choice: gr.update(
                                choices=[
                                    layer[0] for layer in torch_models.get_model(choice, weights=None).named_modules()
                                ],
                                value=None,
                            ),
                            inputs=[backbone_selection],
                            outputs=[layer_selection],
                        )
                        # Reveal file explorer when selecting the "From file" option of the weights
                        weight_selection.change(
                            lambda choice: gr.update(visible=(choice == "From file")),
                            inputs=[weight_selection],
                            outputs=[weight_file],
                        )

                    with gr.Accordion("Add-on layers configuration", open=True):
                        num_addon_layers = gr.Slider(
                            label="Number of add-on layers", minimum=0, maximum=self.MAX_ADDON_LAYERS, value=0, step=1
                        )

                        # All possible parameters from supported layers
                        default_parameters = {}
                        for layer_name in supported_addon_layers:
                            for param_name, default_value in supported_addon_layers[layer_name].items():
                                if (
                                    param_name in default_parameters.keys()
                                    and default_parameters[param_name] != default_value
                                ):
                                    logger.error(
                                        f"New default value {default_value} for parameter {param_name} "
                                        f"(was {default_parameters[param_name]})"
                                    )
                                default_parameters[param_name] = default_value

                        addon_init_selection = gr.Dropdown(
                            label="Init mode", visible=False, choices=list(layer_init_functions.keys())
                        )
                        addon_init_selection.change(self.layer_init_callback(), inputs=[addon_init_selection])
                        num_addon_layers.change(
                            lambda x: gr.update(visible=(x > 0)),
                            inputs=[num_addon_layers],
                            outputs=[addon_init_selection],
                        )

                        addon_layers = []

                        def _change_param_visibility(param_name: str):
                            def change_visibility(layer_type: str):
                                return gr.update(
                                    visible=(supported_addon_layers[layer_type].get(param_name) is not None)
                                )

                            return change_visibility

                        with gr.Column():
                            for addon_index in range(self.MAX_ADDON_LAYERS):
                                with gr.Row(visible=False) as addon_config:
                                    # Layer type selection
                                    layer_type = gr.Dropdown(
                                        label="Layer type",
                                        choices=list(supported_addon_layers.keys()),
                                        interactive=True,
                                    )
                                    layer_type.change(
                                        self.layer_type_callback(layer_index=addon_index), inputs=[layer_type]
                                    )

                                    # Add all parameter boxes
                                    for param_name, default_value in default_parameters.items():
                                        selector = self.add_generic_selector(
                                            selector_config=default_value, param_name=param_name, visibility=False
                                        )
                                        if selector:
                                            layer_type.change(
                                                _change_param_visibility(param_name=param_name),
                                                inputs=[layer_type],
                                                outputs=[selector],
                                            )
                                            selector.change(
                                                self.layer_params_callback(
                                                    layer_index=addon_index, param_name=param_name
                                                ),
                                                inputs=[selector],
                                            )
                                addon_layers.append(addon_config)

                        # Change visibility based on the selected number of layers
                        num_addon_layers.change(
                            lambda num_visible_layers: [gr.update(visible=True)] * num_visible_layers
                            + [gr.update(visible=False)] * (self.MAX_ADDON_LAYERS - num_visible_layers),
                            [num_addon_layers],
                            addon_layers,
                        )
                        # Refresh extractor configuration
                        num_addon_layers.change(self.addon_configuration_callback(), inputs=[num_addon_layers])

                    # Classifier configuration
                    with gr.Accordion("Classifier configuration", open=True):
                        for arch in supported_architectures:

                            def _change_arch_param_visibility(arch_name: str):
                                def change_visibility(arch: str):
                                    return gr.update(visible=(arch == arch_name))

                                return change_visibility

                            with gr.Row(visible=False) as arch_params:
                                # Add all parameter boxes
                                for param_name, default_value in supported_architectures[arch]["classifier"][
                                    "params"
                                ].items():
                                    selector = self.add_generic_selector(
                                        selector_config=default_value, param_name=param_name, visibility=True
                                    )
                                    if selector:
                                        selector.change(
                                            self.architecture_param_callback(param_name=param_name), inputs=[selector]
                                        )

                            arch_selection.change(
                                _change_arch_param_visibility(arch), inputs=[arch_selection], outputs=[arch_params]
                            )
                with gr.Column():
                    result = gr.Textbox(interactive=False, label="Output file")
                    output_directory.change(self.load_output_file_callback(), outputs=result)
                    output_file.change(self.load_output_file_callback(), outputs=result)

            # Save YAML file on click
            generate = gr.Button(value="Generate configuration file", icon="docs/logos/cabrnet.svg")
            generate.click(self.save_callback(), outputs=result)

        return block


def main():
    r"""Main GUI function."""
    cabrnet_design = CaBRNetDesignGUI()
    block = cabrnet_design.create_gui()
    block.launch()


if __name__ == "__main__":
    main()
