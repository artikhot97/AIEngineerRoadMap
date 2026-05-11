"""
GPU Memory Context Manager
Handles cleanup of GPU memory after model inference with support for
PyTorch, multiple devices, nested contexts, and error recovery.
"""

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GPUMemorySnapshot:
    """Point-in-time GPU memory stats (bytes)."""
    allocated: int
    reserved: int
    peak_allocated: int
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def allocated_mb(self) -> float:
        return self.allocated / 1024 ** 2

    @property
    def reserved_mb(self) -> float:
        return self.reserved / 1024 ** 2

    @property
    def peak_allocated_mb(self) -> float:
        return self.peak_allocated / 1024 ** 2


@dataclass
class GPUMemoryDelta:
    """Memory change between two snapshots."""
    before: GPUMemorySnapshot
    after: GPUMemorySnapshot
    duration_s: float

    @property
    def freed_mb(self) -> float:
        return (self.before.allocated - self.after.allocated) / 1024 ** 2

    @property
    def peak_mb(self) -> float:
        return self.before.peak_allocated_mb

    def __str__(self) -> str:
        return (
            f"Duration: {self.duration_s:.3f}s | "
            f"Freed: {self.freed_mb:+.1f} MB | "
            f"Peak: {self.peak_mb:.1f} MB | "
            f"After (alloc/reserved): "
            f"{self.after.allocated_mb:.1f} / {self.after.reserved_mb:.1f} MB"
        )


# ---------------------------------------------------------------------------
# Core context manager class
# ---------------------------------------------------------------------------

class GPUMemoryManager:
    """
    Context manager that snapshots, monitors, and cleans up GPU memory
    around a block of model-inference code.

    Usage
    -----
    Basic::

        with GPUMemoryManager(device="cuda:0") as mgr:
            outputs = model(inputs)
        print(mgr.delta)

    Return logits to CPU before the block exits so tensors are freed::

        with GPUMemoryManager(device="cuda", move_to_cpu=True) as mgr:
            logits = model(inputs)
            mgr.register_tensor(logits)   # released on exit

    Aggressive (empty cache + full GC)::

        with GPUMemoryManager(device="cuda", aggressive=True):
            ...
    """

    def __init__(
        self,
        device: Union[str, int, None] = "cuda",
        *,
        aggressive: bool = False,
        empty_cache: bool = True,
        run_gc: bool = True,
        reset_peak_stats: bool = True,
        log_level: int = logging.DEBUG,
        raise_on_no_cuda: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        device          : CUDA device string/index, or None to skip CUDA ops.
        aggressive      : Call ``torch.cuda.synchronize`` before cleanup and
                          repeat cache-empty + GC twice for stubborn allocations.
        empty_cache     : Call ``torch.cuda.empty_cache()`` on exit.
        run_gc          : Call ``gc.collect()`` on exit.
        reset_peak_stats: Reset peak-memory counters before the block so
                          ``peak_allocated`` reflects only this block.
        log_level       : Logging level for automatic stats output.
        raise_on_no_cuda: Raise RuntimeError if CUDA is unavailable instead
                          of silently degrading to CPU-only mode.
        """
        self.device = device
        self.aggressive = aggressive
        self.empty_cache = empty_cache
        self.run_gc = run_gc
        self.reset_peak_stats = reset_peak_stats
        self.log_level = log_level
        self.raise_on_no_cuda = raise_on_no_cuda

        self._cuda_available: bool = False
        self._torch_device = None
        self._registered_tensors: list = []
        self._start_time: float = 0.0
        self.before: Optional[GPUMemorySnapshot] = None
        self.after: Optional[GPUMemorySnapshot] = None
        self.delta: Optional[GPUMemoryDelta] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def register_tensor(self, tensor) -> None:
        """
        Register a tensor to be explicitly deleted on context exit.
        Useful for outputs you no longer need after moving them to CPU.
        """
        self._registered_tensors.append(tensor)

    def snapshot(self) -> Optional[GPUMemorySnapshot]:
        """Take a live memory snapshot (returns None if CUDA unavailable)."""
        if not self._cuda_available:
            return None
        import torch
        mem = torch.cuda.memory_stats(self._torch_device)
        return GPUMemorySnapshot(
            allocated=mem.get("allocated_bytes.all.current", 0),
            reserved=mem.get("reserved_bytes.all.current", 0),
            peak_allocated=mem.get("allocated_bytes.all.peak", 0),
        )

    # ------------------------------------------------------------------
    # Context protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "GPUMemoryManager":
        self._cuda_available = self._init_cuda()
        self._start_time = time.monotonic()

        if self._cuda_available:
            import torch
            if self.reset_peak_stats:
                torch.cuda.reset_peak_memory_stats(self._torch_device)
            self.before = self.snapshot()
            logger.log(
                self.log_level,
                "[GPUMemoryManager] enter — allocated: %.1f MB",
                self.before.allocated_mb if self.before else 0,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.monotonic() - self._start_time

        # Release registered tensors first
        for t in self._registered_tensors:
            del t
        self._registered_tensors.clear()

        if self._cuda_available:
            self._cleanup()

        self.after = self.snapshot()
        if self.before and self.after:
            self.delta = GPUMemoryDelta(self.before, self.after, elapsed)
            logger.log(self.log_level, "[GPUMemoryManager] exit — %s", self.delta)

        # Never suppress exceptions
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_cuda(self) -> bool:
        try:
            import torch
        except ImportError:
            logger.warning("[GPUMemoryManager] PyTorch not installed; skipping GPU ops.")
            return False

        if not torch.cuda.is_available():
            if self.raise_on_no_cuda:
                raise RuntimeError("CUDA is not available on this system.")
            logger.debug("[GPUMemoryManager] CUDA unavailable; running in no-op mode.")
            return False

        # Resolve device
        if self.device is None:
            return False
        if isinstance(self.device, int):
            self._torch_device = torch.device(f"cuda:{self.device}")
        else:
            self._torch_device = torch.device(self.device)

        return True

    def _cleanup(self) -> None:
        import torch

        if self.aggressive:
            torch.cuda.synchronize(self._torch_device)

        if self.run_gc:
            gc.collect()

        if self.empty_cache:
            torch.cuda.empty_cache()

        if self.aggressive:
            # Second pass for allocations freed by the first GC run
            gc.collect()
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Functional / decorator helpers
# ---------------------------------------------------------------------------

@contextmanager
def gpu_inference(
    device: Union[str, int, None] = "cuda",
    *,
    aggressive: bool = False,
    empty_cache: bool = True,
    run_gc: bool = True,
    log_level: int = logging.DEBUG,
) -> Generator[GPUMemoryManager, None, None]:
    """
    Lightweight functional wrapper around :class:`GPUMemoryManager`.

    Example::

        with gpu_inference("cuda:1", aggressive=True) as mgr:
            out = model(x)
        print(mgr.delta)
    """
    mgr = GPUMemoryManager(
        device=device,
        aggressive=aggressive,
        empty_cache=empty_cache,
        run_gc=run_gc,
        log_level=log_level,
    )
    with mgr:
        yield mgr


def managed_inference(device: Union[str, int, None] = "cuda", **kwargs):
    """
    Decorator that wraps an inference function with GPU memory cleanup.

    Example::

        @managed_inference(device="cuda", aggressive=True)
        def run_model(model, inputs):
            return model(inputs)
    """
    def decorator(fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kw):
            with GPUMemoryManager(device=device, **kwargs) as mgr:
                result = fn(*args, **kw)
            logger.debug("[managed_inference] %s — %s", fn.__name__, mgr.delta)
            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Multi-device helper
# ---------------------------------------------------------------------------

@contextmanager
def multi_gpu_inference(
    devices: list,
    **kwargs,
) -> Generator[list, None, None]:
    """
    Manage cleanup across several GPUs simultaneously.

    Example::

        with multi_gpu_inference(["cuda:0", "cuda:1"]) as managers:
            # use both GPUs
            ...
        for m in managers:
            print(m.delta)
    """
    managers = [GPUMemoryManager(device=d, **kwargs) for d in devices]
    for m in managers:
        m.__enter__()
    try:
        yield managers
    except Exception:
        for m in managers:
            m.__exit__(*__import__("sys").exc_info())
        raise
    else:
        for m in managers:
            m.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(message)s",
        stream=sys.stdout,
    )

    print("=" * 60)
    print("Demo 1 — basic context manager (CPU-safe no-op when no GPU)")
    print("=" * 60)
    with GPUMemoryManager(device="cuda") as mgr:
        # Simulate work
        data = [i ** 2 for i in range(10_000)]
    print(f"delta: {mgr.delta}\n")

    print("=" * 60)
    print("Demo 2 — functional helper")
    print("=" * 60)
    with gpu_inference("cuda", aggressive=True) as mgr:
        data = list(range(10_000))
    print(f"delta: {mgr.delta}\n")

    print("=" * 60)
    print("Demo 3 — decorator")
    print("=" * 60)

    @managed_inference(device="cuda")
    def fake_inference(n: int) -> list:
        return list(range(n))

    result = fake_inference(5_000)
    print(f"result length: {len(result)}\n")

    print("=" * 60)
    print("Demo 4 — exception safety (error does NOT suppress)")
    print("=" * 60)
    try:
        with GPUMemoryManager(device="cuda") as mgr:
            raise ValueError("Simulated inference error")
    except ValueError as e:
        print(f"Caught expected error: {e}")
        print(f"delta: {mgr.delta}\n")

    print("All demos complete.")