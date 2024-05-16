import torch
from cpuinfo import get_cpu_info


def get_hardware_info(device: str) -> str:
    r"""Returns the target device hardware information.

    Args:
        device (str): Target hardware device.
    """
    if device == "cpu":
        return get_cpu_info()["brand_raw"]
    return torch.cuda.get_device_name(device)
