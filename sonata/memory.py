# sonata/memory.py
"""RAM/VRAM budgeting, batch semaphores, and OOM-safe batching.

Heavy SONATA stages touch memory in three risky ways: they build large batches of
graphs, they run eigensolves whose peak allocation is hard to predict, and they
push data through the GPU where a bad batch size means an out-of-memory abort
minutes into a run. This module centralises the defensive discipline:

* probe free RAM and VRAM (:func:`available_ram_bytes`, :func:`available_vram_bytes`);
* size a batch to a fraction of the free budget (:func:`estimate_batch_size`);
* gate concurrent memory-heavy tasks with a counting semaphore (:class:`MemoryGate`);
* iterate in batches that *halve on OOM and retry* (:func:`oom_safe_batches`);
* periodically synchronise + free device caches inside long device loops
  (:func:`sync_every`), matching the "sync every 4--12 SpMV" rule.

All GPU calls go through :mod:`sonata.backends.array`, so this module is
import-safe with no CUDA present.
"""

from __future__ import annotations

import gc
import threading
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence, TypeVar

from .backends import array as ab
from .backends.base import ArrayBackend, capabilities

T = TypeVar("T")

#: Fraction of free memory a batch is allowed to claim (leave headroom for peaks).
DEFAULT_HEADROOM: float = 0.8


# ── probes ────────────────────────────────────────────────────────────────────
def available_ram_bytes() -> int:
    """Free system RAM in bytes (falls back to a conservative 4 GiB guess)."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # pragma: no cover - psutil absent
        return 4 * 1024**3


def available_vram_bytes(backend: ArrayBackend = "torch") -> int | None:
    """Free VRAM in bytes for ``backend``'s device, or ``None`` if not on GPU."""
    if not capabilities().has_cuda:
        return None
    return ab.device_free_bytes(backend)


# ── batch sizing ──────────────────────────────────────────────────────────────
def estimate_batch_size(
    item_bytes: int,
    *,
    backend: ArrayBackend = "cpu",
    headroom: float = DEFAULT_HEADROOM,
    peak_multiplier: float = 3.0,
    min_batch: int = 1,
    max_batch: int | None = None,
) -> int:
    """How many items of ``item_bytes`` fit within the free memory budget.

    Parameters
    ----------
    item_bytes
        Resident size of one item's tensors.
    backend
        ``"cpu"`` sizes against free RAM; a GPU backend sizes against free VRAM.
    headroom
        Keep this fraction of memory in reserve (``0.8`` -> use 80 %).
    peak_multiplier
        Assume transient peaks reach ``peak_multiplier``x the resident size
        (activations, gradients, temporaries).
    min_batch, max_batch
        Clamp the result.

    Returns
    -------
    int
        A safe batch size (at least ``min_batch``).
    """
    if backend == "cpu":
        free = available_ram_bytes()
    else:
        free = available_vram_bytes(backend) or available_ram_bytes()
    budget = int(free * headroom)
    per_item = max(1, int(item_bytes * peak_multiplier))
    n = max(min_batch, budget // per_item)
    if max_batch is not None:
        n = min(n, max_batch)
    return int(n)


# ── concurrency gate ──────────────────────────────────────────────────────────
class MemoryGate:
    """A counting semaphore that caps concurrent memory-heavy sections.

    Use to bound how many workers simultaneously hold a big allocation (e.g. a
    GPU batch), independent of how many CPU workers exist. ``max_concurrent``
    defaults to 1 for a single GPU.

    Examples
    --------
    >>> gate = MemoryGate(max_concurrent=1)
    >>> with gate:            # doctest: +SKIP
    ...     run_gpu_batch()   # only one worker in here at a time
    """

    def __init__(self, max_concurrent: int = 1) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._sem = threading.BoundedSemaphore(max_concurrent)
        self.max_concurrent = max_concurrent

    def __enter__(self) -> "MemoryGate":
        self._sem.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self._sem.release()

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Context-manager alias for acquiring one slot."""
        with self:
            yield


# ── OOM-safe iteration ────────────────────────────────────────────────────────
def _is_oom(exc: BaseException) -> bool:
    """Recognise out-of-memory errors across torch/cupy/numpy."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        "outofmemory" in name
        or "out of memory" in msg
        or "cuda error" in msg and "memory" in msg
        or isinstance(exc, MemoryError)
    )


def run_in_oom_safe_batches(
    fn,
    items: Sequence[T],
    *,
    initial_batch: int | None = None,
    item_bytes: int = 1 << 20,
    backend: ArrayBackend = "cpu",
    min_batch: int = 1,
    free_between: bool = True,
) -> list:
    """Apply a *batched* ``fn(list_of_items) -> list_of_results`` OOM-safely.

    On an out-of-memory error the current batch size is halved and the batch is
    retried; between batches the device cache is optionally freed. Item order is
    preserved.

    Parameters
    ----------
    fn
        Callable taking a list of items and returning a list of results.
    items
        Full sequence of inputs.
    initial_batch
        Starting batch size; if ``None``, estimated from ``item_bytes`` and the
        free budget of ``backend``.
    item_bytes
        Approximate resident bytes per item (for the estimate).
    backend
        Backend whose memory budget governs sizing and whose cache is freed.
    min_batch
        Never shrink below this; if a single item OOMs, the error propagates.
    free_between
        Free the device cache between batches.

    Returns
    -------
    list
        Concatenated results in input order.
    """
    items = list(items)
    if initial_batch is None:
        initial_batch = estimate_batch_size(item_bytes, backend=backend)
    batch = max(min_batch, int(initial_batch))

    results: list = []
    i, n = 0, len(items)
    while i < n:
        chunk = items[i : i + batch]
        try:
            results.extend(fn(chunk))
            i += len(chunk)
            if free_between:
                ab.free_memory(backend)
        except Exception as exc:  # noqa: BLE001 - we re-raise non-OOM below
            if not _is_oom(exc):
                raise
            ab.free_memory(backend)
            gc.collect()
            if batch <= min_batch:
                raise  # a single-item batch still OOMs -> unrecoverable
            batch = max(min_batch, batch // 2)
    return results


# ── periodic device sync inside long loops ────────────────────────────────────
def sync_every(
    iterable: Iterable[T],
    *,
    every: int = 8,
    backend: ArrayBackend = "torch",
    free_cache: bool = False,
) -> Iterator[T]:
    """Yield items, synchronising the device every ``every`` steps.

    Implements the "sync every 4--12 SpMV" guidance for long device loops: it
    bounds the outstanding async work (and thus peak VRAM) without paying a sync
    on every iteration. Optionally frees the cache at each sync point.
    """
    every = max(1, int(every))
    for k, item in enumerate(iterable, start=1):
        yield item
        if k % every == 0:
            ab.synchronize(backend)
            if free_cache:
                ab.free_memory(backend)
    ab.synchronize(backend)


__all__ = [
    "DEFAULT_HEADROOM",
    "available_ram_bytes",
    "available_vram_bytes",
    "estimate_batch_size",
    "MemoryGate",
    "run_in_oom_safe_batches",
    "sync_every",
]
