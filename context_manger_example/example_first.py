"""
GPU Memory Cleanup Context Manager

Supports:
- PyTorch CUDA cleanup
- Optional mixed precision inference
- Automatic synchronization
- Exception-safe cleanup
- Garbage collection

Useful for:
- LLM inference
- Batch processing
- Avoiding CUDA OOM issues
"""

from contextlib import ContextDecorator
import gc
import torch


class GPUMemoryManager(ContextDecorator):
    """
    Context manager for safe GPU inference and cleanup.

    Example:
        with GPUMemoryManager():
            output = model(inputs)

    Example with autocast:
        with GPUMemoryManager(mixed_precision=True):
            output = model(inputs)
    """

    def __init__(
        self,
        device: str = "cuda",
        mixed_precision: bool = False,
        clear_cache: bool = True,
        synchronize: bool = True,
    ):
        self.device = device
        self.mixed_precision = mixed_precision
        self.clear_cache = clear_cache
        self.synchronize = synchronize
        self.autocast_context = None

    def __enter__(self):
        if not torch.cuda.is_available():
            return self

        if self.clear_cache:
            torch.cuda.empty_cache()

        # Mixed precision support
        if self.mixed_precision:
            self.autocast_context = torch.cuda.amp.autocast()
            self.autocast_context.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            # Exit autocast safely
            if self.autocast_context:
                self.autocast_context.__exit__(exc_type, exc_val, exc_tb)

            if torch.cuda.is_available():
                if self.synchronize:
                    torch.cuda.synchronize()

                # Run Python garbage collection
                gc.collect()

                # Release unused cached memory
                if self.clear_cache:
                    torch.cuda.empty_cache()

                # Collect interprocess memory
                torch.cuda.ipc_collect()

        except Exception as cleanup_error:
            print(f"GPU cleanup failed: {cleanup_error}")

        # Do not suppress exceptions
        return False