from loguru import logger
import torch


class _MemoryLogger:
    def __init__(self, show_reserved: bool = False):
        self._allocated_memory = torch.cuda.memory_allocated() // 1024**2
        self._reserved_memory = torch.cuda.memory_reserved() // 1024**2
        self._show_reserved = show_reserved

    def stats(self, message: str | None = None) -> None:
        message = f"{message}. " if message is not None else ""
        allocated = torch.cuda.memory_allocated() // 1024**2
        reserved = torch.cuda.memory_reserved() // 1024**2
        logger.info(f"{message} Memory allocated: {allocated}MB (delta is {allocated - self._allocated_memory}MB)")
        if self._show_reserved:
            logger.info(f"{message} Memory reserved: {reserved}MB (delta is {reserved - self._reserved_memory}MB)")
        self._allocated_memory = allocated
        self._reserved_memory = reserved


memory_logger = _MemoryLogger(show_reserved=False)
