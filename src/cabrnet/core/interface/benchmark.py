from typing import Any

import gradio as gr
from cabrnet.core.interface.utils import change_visibility
from gradio.components.base import Component


def create_perturbation_benchmark_gui(benchmark_selection: gr.Dropdown, visible: bool = False) -> set[Component]:
    r"""Returns a set of components for configuring the local perturbation analysis.

    Args:
        benchmark_selection (gradio component): Component controlling the visibility of the GUI.
        visible (bool, optional): Initial visibility. Default: False.
    """
    with gr.Column(visible=visible) as col:
        perturbation_selection = gr.Dropdown(
            choices=["brightness", "contrast", "saturation", "hue", "blur", "sin_distortion"],
            label="Perturbation selection",
            interactive=True,
            key="perturbation_selection",
            visible=visible,
        )
        dual_mode = gr.Checkbox(value=True, visible=visible, label="Dual mode", key="enable_dual_mode")

        benchmark_selection.change(
            change_visibility("Local perturbation analysis"), benchmark_selection, perturbation_selection
        )

        brightness = gr.Slider(
            label="Brightness",
            value=0.3,
            minimum=0,
            maximum=1,
            step=0.1,
            interactive=True,
            key="brightness",
            visible=False,
        )
        contrast = gr.Slider(
            label="Contrast", value=0.2, minimum=0, maximum=1, step=0.1, interactive=True, key="contrast", visible=False
        )
        saturation = gr.Slider(
            label="Saturation",
            value=0.3,
            minimum=0,
            maximum=1,
            step=0.1,
            interactive=True,
            key="saturation",
            visible=False,
        )
        hue = gr.Slider(
            label="Hue shift",
            value=0.3,
            minimum=-0.5,
            maximum=0.5,
            step=0.1,
            interactive=True,
            key="hue",
            visible=False,
        )
        gaussian_ksize = gr.Slider(
            label="Gaussian blur kernel size",
            value=21,
            minimum=1,
            maximum=29,
            step=2,
            interactive=True,
            key="blur_ksize",
            visible=False,
        )

        gaussian_sigma = gr.Slider(
            label="Gaussian blur sigma",
            value=2.0,
            minimum=0,
            maximum=5,
            step=0.5,
            interactive=True,
            key="blur_sigma",
            visible=False,
        )

        distortion_periods = gr.Slider(
            label="# Distortion periods",
            value=5,
            minimum=0,
            maximum=10,
            step=1,
            interactive=True,
            key="distortion_periods",
            visible=False,
        )

        distortion_amplitude = gr.Slider(
            label="Distortion amplitude",
            value=7.0,
            minimum=0,
            maximum=10,
            step=0.5,
            interactive=True,
            key="distortion_amplitude",
            visible=False,
        )

        distortion_direction = gr.Dropdown(
            label="Distortion direction",
            choices=["horizontal", "vertical", "both"],
            value="both",
            interactive=True,
            key="distortion_direction",
            visible=False,
        )

        perturbation_selection.change(change_visibility("brightness"), perturbation_selection, brightness)
        perturbation_selection.change(change_visibility("contrast"), perturbation_selection, contrast)
        perturbation_selection.change(change_visibility("saturation"), perturbation_selection, saturation)
        perturbation_selection.change(change_visibility("hue"), perturbation_selection, hue)
        perturbation_selection.change(change_visibility("blur"), perturbation_selection, gaussian_ksize)
        perturbation_selection.change(change_visibility("blur"), perturbation_selection, gaussian_sigma)
        perturbation_selection.change(change_visibility("sin_distortion"), perturbation_selection, distortion_periods)
        perturbation_selection.change(change_visibility("sin_distortion"), perturbation_selection, distortion_amplitude)
        perturbation_selection.change(change_visibility("sin_distortion"), perturbation_selection, distortion_direction)

    benchmark_selection.change(change_visibility("Local perturbation analysis"), benchmark_selection, col)
    return {
        perturbation_selection,
        dual_mode,
        brightness,
        contrast,
        saturation,
        hue,
        gaussian_ksize,
        gaussian_sigma,
        distortion_periods,
        distortion_amplitude,
        distortion_direction,
    }


def get_perturbation_config(gradio_config: dict[str, Any]) -> dict[str, Any]:
    r"""Builds the configuration dictionary for the perturbation of image patches.

    Args:
        gradio_config (dictionary): Dictionary of Gradio components outputs.

    Returns:
        Perturbation configuration.
    """
    return {
        "local_perturbation_analysis": {
            "enable_dual_mode": gradio_config["enable_dual_mode"],
            "num_prototypes": gradio_config["num_prototypes"],
            "perturbations": [gradio_config["perturbation_selection"]],
            "brightness_factor": gradio_config["brightness"],
            "contrast_factor": gradio_config["contrast"],
            "saturation_factor": gradio_config["saturation"],
            "hue_factor": gradio_config["hue"],
            "gaussian_blur_ksize": gradio_config["blur_ksize"],
            "gaussian_blur_sigma": gradio_config["blur_sigma"],
            "distortion_periods": gradio_config["distortion_periods"],
            "distortion_amplitude": gradio_config["distortion_amplitude"],
            "distortion_direction": gradio_config["distortion_direction"],
        }
    }


def create_pointing_benchmark_gui(benchmark_selection: gr.Dropdown, visible: bool = False) -> set[Component]:
    r"""Returns a set of components for configuring the pointing game analysis.

    Args:
        benchmark_selection (gradio component): Component controlling the visibility of the GUI.
        visible (bool, optional): Initial visibility. Default: False.
    """
    with gr.Column(visible=visible) as col:
        area_percentage = gr.Slider(
            label="Area percentage",
            value=0.1,
            minimum=0.01,
            maximum=1,
            step=0.01,
            interactive=True,
            key="area_percentage",
            visible=True,
        )
    benchmark_selection.change(change_visibility("Pointing game analysis"), benchmark_selection, col)

    return {area_percentage}


def get_pointing_game_config(gradio_config: dict[str, Any]) -> dict[str, Any]:
    r"""Builds the configuration dictionary for the pointing game analysis of image patches.

    Args:
        gradio_config (dictionary): Dictionary of Gradio components outputs.

    Returns:
        Perturbation configuration.
    """
    return {
        "relevance_analysis": {
            "num_prototypes": gradio_config["num_prototypes"],
            "area_percentage": gradio_config["area_percentage"],
        }
    }
