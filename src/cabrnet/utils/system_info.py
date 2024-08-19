import torch
from cpuinfo import get_cpu_info
from pathlib import Path


def get_parent_directory(dir_path: str):
    r"""Returns the parent directory of *dir_path*.

    Args:
        dir_path (str): Path to directory.

    Returns:
        Absolute path to parent directory.
    """
    return Path(dir_path).parent.absolute()


def get_hardware_info(device: str | torch.device) -> str:
    r"""Returns the target device hardware information.

    Args:
        device (str | device): Hardware device.
    """
    if device == "cpu":
        return get_cpu_info()["brand_raw"]
    return torch.cuda.get_device_name(device)
