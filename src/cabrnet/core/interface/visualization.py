from typing import Any

import gradio as gr
from cabrnet.core.interface.utils import change_visibility
from cabrnet.core.visualization.view import supported_viewing_functions
from cabrnet.core.visualization.visualizer import supported_attribution_functions
from gradio.components.base import Component


def create_visualization_gui(
    default_attribution: str = "smoothgrad", default_viewing: str = "bbox_to_percentile"
) -> set[Component]:
    r"""Initializes GUI for the visualization of image patches.

    Args:
        default_attribution (str, optional): Default attribution method. Default: smoothgrad.
        default_viewing (str, optional): Default viewing method. Default: bbox_to_percentile.

    Returns:
        A set of components for configuring the visualization of image patches.
    """
    components: set[Component] = set()

    with gr.Blocks():
        with gr.Row():
            # Select attribution method
            attribution_select = gr.Dropdown(
                choices=list(supported_attribution_functions.keys()),
                value=default_attribution,
                label="Attribution function",
                key="attribution_select",
            )  # type: ignore
            components.add(attribution_select)

            # Add specific options for all attribution methods
            num_samples = gr.Slider(
                minimum=1,
                maximum=50,
                step=1,
                value=10,
                interactive=True,
                visible=(default_attribution == "smoothgrad"),
                label="# random samples",
                key="smoothgrad_num_samples",
            )
            noise_ratio = gr.Number(
                value=0.2,
                interactive=True,
                label="Noise ratio",
                visible=(default_attribution == "smoothgrad"),
                key="smoothgrad_noise_ratio",
            )
            stability_factor = gr.Number(
                value=1e-6,
                interactive=True,
                label="Stability factor",
                visible=(default_attribution == "prp"),
                key="prp_stability_factor",
            )
            components.update([num_samples, noise_ratio, stability_factor])

            # Change visibility depending on the chosen attribution method
            attribution_select.change(change_visibility("smoothgrad"), attribution_select, num_samples)
            attribution_select.change(change_visibility("smoothgrad"), attribution_select, noise_ratio)
            attribution_select.change(change_visibility("prp"), attribution_select, stability_factor)

        grad_methods = ["prp", "smoothgrad", "saliency", "randgrad"]
        with gr.Row(visible=(default_attribution in grad_methods)) as grad_options:
            polarity = gr.Dropdown(
                label="Gradient polarity",
                value="absolute",
                interactive=True,
                choices=["absolute", "positive", "negative", None],  # type: ignore
                key="gradient_polarity",
            )
            gaussian_ksize = gr.Slider(
                minimum=1,
                maximum=20,
                step=1,
                value=5,
                interactive=True,
                label="Gaussian kernel size",
                key="gradient_gaussian_ksize",
            )
            grad_x_input = gr.Checkbox(value=False, interactive=True, label="Gradients x input", key="gradient_x_input")
            similarity_threshold = gr.Number(
                value=0.1,
                interactive=True,
                label="Similarity selection threshold",
                key="gradient_similarity_threshold",
            )
            components.update([polarity, gaussian_ksize, grad_x_input, similarity_threshold])
        attribution_select.change(change_visibility(grad_methods), attribution_select, grad_options)

        # Common attribution options
        single_location = gr.Checkbox(label="Max similarity score only", value=True, key="gradient_max_similarity")
        normalize = gr.Checkbox(label="Normalize", value=True, key="gradient_normalize")
        components.update([single_location, normalize])

        # Select viewing method
        with gr.Row():
            viewing_select = gr.Dropdown(
                choices=list(supported_viewing_functions.keys()),
                value=default_viewing,
                label="Viewing function",
                key="viewing_select",
            )  # type: ignore

            # Common viewing options
            percentile = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.05,
                value=0.7,
                interactive=True,
                label="Selection percentile",
                key="selection_percentile",
            )
            thickness = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                value=2,
                interactive=True,
                label="Bounding box thickness",
                visible=(default_viewing in ["bbox_to_percentile", "heatmap"]),
                key="bbox_thickness",
            )
            viewing_select.change(change_visibility(["bbox_to_percentile", "heatmap"]), viewing_select, thickness)

            overlay = gr.Checkbox(
                label="Overlay", value=False, visible=(default_attribution == "heatmap"), key="heatmap_overlay"
            )
            viewing_select.change(change_visibility(["heatmap"]), viewing_select, overlay)
            components.update([viewing_select, percentile, thickness, overlay])
    return components


def get_visualization_config(gradio_config: dict[str, Any]) -> dict[str, Any]:
    r"""Builds the configuration dictionary for the visualization of image patches.

    Args:
        gradio_config (dictionary): Dictionary of Gradio components outputs.

    Returns:
        Visualization configuration.
    """
    return {
        "attribution": {
            "type": gradio_config["attribution_select"],
            "params": {
                "num_samples": gradio_config["smoothgrad_num_samples"],
                "noise_ratio": gradio_config["smoothgrad_noise_ratio"],
                "stability_factor": gradio_config["prp_stability_factor"],
                "polarity": gradio_config["gradient_polarity"],
                "gaussian_ksize": gradio_config["gradient_gaussian_ksize"],
                "grads_x_input": gradio_config["gradient_x_input"],
                "similarity_threshold": gradio_config["gradient_similarity_threshold"],
                "location": "max" if gradio_config["gradient_max_similarity"] else None,
                "normalize": gradio_config["gradient_normalize"],
            },
        },
        "view": {
            "type": gradio_config["viewing_select"],
            "params": {
                "percentile": gradio_config["selection_percentile"],
                "thickness": gradio_config["bbox_thickness"],
                "overlay": gradio_config["heatmap_overlay"],
            },
        },
    }
