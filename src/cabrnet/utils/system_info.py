import torch
from cpuinfo import get_cpu_info


def get_hardware_info(device: str) -> str:
    if device == "cpu":
        return get_cpu_info()["brand_raw"]
    return torch.cuda.get_device_name(device)
