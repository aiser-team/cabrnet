import torch
from loguru import logger


def metrics_to_str(metrics: dict[str, float]) -> str:
    r"""Converts a dictionary of metrics into a readable string.

    Args:
        metrics (dictionary): Dictionary of batch metrics.

    Returns:
        Readable string representing batch statistics.
    """
    return ", ".join([f"{key}: {value:.3f}" for key, value in metrics.items()])


class _MemoryLogger:
    r"""Class for CUDA memory monitoring."""

    def __init__(self, show_reserved: bool = False):
        r"""Creates a memory logger.

        Args:
            show_reserved (bool, optional): If True, show reserved memory. Default: False.
        """
        self._allocated_memory = torch.cuda.memory_allocated() // 1024**2
        self._reserved_memory = torch.cuda.memory_reserved() // 1024**2
        self._show_reserved = show_reserved

    def stats(self, message: str | None = None) -> None:
        r"""Displays current memory usage.

        Args:
            message (str, optional): If provided, prefix of the message printed by the module. Default: None.
        """
        message = f"{message}. " if message is not None else ""
        allocated = torch.cuda.memory_allocated() // 1024**2
        reserved = torch.cuda.memory_reserved() // 1024**2
        logger.info(f"{message} Memory allocated: {allocated}MB (delta is {allocated - self._allocated_memory}MB)")
        if self._show_reserved:
            logger.info(f"{message} Memory reserved: {reserved}MB (delta is {reserved - self._reserved_memory}MB)")
        self._allocated_memory = allocated
        self._reserved_memory = reserved


memory_logger = _MemoryLogger(show_reserved=False)
