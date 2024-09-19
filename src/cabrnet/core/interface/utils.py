import os
from tkinter import Tk, filedialog
from typing import Any, Callable

import gradio as gr
from gradio.components.base import Component
from loguru import logger


def create_browse_folder_component(
    label: str, default_dir: str, allow_creation: bool = False, key: str | None = None
) -> gr.Textbox:
    r"""TK-based pop-up window for selecting folders
    Adapted from https://github.com/gradio-app/gradio/issues/2515#issuecomment-1676824284.

    Args:
        label (str): Component label.
        default_dir (str): Default directory for the explorer.
        allow_creation (bool, optional): If True, allow the directory to be created if necessary. Default: False.
        key (str, optional): Component key. Default: None.
    Returns:
        A Gradio TextBox containing the directory path.
    """

    def browse_folder(textbox):
        root = Tk()
        root.attributes("-topmost", True)
        root.withdraw()
        filename = filedialog.askdirectory(initialdir=default_dir, mustexist=not allow_creation)
        if not filename:
            # Revert to default directory
            filename = default_dir
        if not os.path.exists(filename) and allow_creation:
            logger.info(f"Creating directory {filename}")
            os.makedirs(filename)
        root.destroy()
        textbox = filename
        return textbox

    with gr.Column():
        selection = gr.Textbox(
            value=default_dir,
            label=label,
            interactive=True,
            max_lines=1,
            autofocus=False,
            placeholder=f"path/to/{label.lower().replace(' ', '/')}",
            key=key,
        )
        selection.focus(
            browse_folder,
            inputs=[selection],
            outputs=[selection],
            show_progress="hidden",
        )
    return selection


def change_visibility(choices: str | list[str]) -> Callable:
    r"""Returns a callback that changes the visibility of a GradioComponent based on the value of a parameter wrt
        to a list of choices.

    Args:
        choices (str, list[str]): Choices that will trigger the visibility of a component.
    """
    choices = choices if isinstance(choices, list) else [choices]

    def _change_visibility(value: str):
        return gr.update(visible=value in choices)

    return _change_visibility


def gradio_convert_outputs(gradio_outputs: dict[Component, Any]) -> dict[str, Any]:
    r"""Converts outputs given by Gradio components, based on their key.

    Args:
        gradio_outputs (dictionary): Dictionary of Gradio components.

    Returns:
        Dictionary of component keys.
    """
    res = {}
    for component, value in gradio_outputs.items():
        res[component.key] = value
    return res
