# sonata/parallel.py
"""Unified CPU parallelism: one ``n_threads`` convention for the whole library.

Every CPU-bound, embarrassingly parallel function in SONATA (per-surface spectral
decomposition, per-subject feature build, bootstrap resampling) takes a single
``n_threads`` argument and routes through :func:`parallel_map`:

* ``n_threads = 1``  -> serial (no joblib overhead, easiest to debug);
* ``n_threads >= 2`` -> joblib with that many worker processes;
* ``n_threads = -1`` -> all usable cores, clamped to ``MAX_CPU_THREADS`` (22).

The subtle-but-critical rule (learned the hard way): when parallelising at the
*item* level, the inner BLAS/OpenMP thread pool of each worker must be pinned to
one thread, or ``N`` workers each spawning ``T`` BLAS threads oversubscribe the
cores and run *slower*. :func:`pin_blas_threads` sets the relevant environment
variables, and :func:`parallel_map` applies it inside each worker automatically.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

from .backends.base import capabilities

T = TypeVar("T")
R = TypeVar("R")

#: Environment variables that control inner numeric thread pools.
_BLAS_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def effective_workers(n_threads: int) -> int:
    """Resolve an ``n_threads`` request to a concrete worker count in ``[1, 22]``."""
    return capabilities().usable_threads(n_threads)


@contextmanager
def pin_blas_threads(n: int = 1) -> Iterator[None]:
    """Temporarily pin inner BLAS/OpenMP pools to ``n`` threads.

    Use around an outer parallel loop so each worker's linear-algebra calls stay
    single-threaded and the machine is not oversubscribed. Restores the previous
    environment on exit.
    """
    previous = {v: os.environ.get(v) for v in _BLAS_ENV_VARS}
    try:
        for v in _BLAS_ENV_VARS:
            os.environ[v] = str(n)
        yield
    finally:
        for v, old in previous.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old


def _pinned_call(fn: Callable[[T], R], item: T) -> R:
    """Worker wrapper: pin BLAS to 1 thread, then run ``fn(item)`` (parallel path)."""
    with pin_blas_threads(1):
        return fn(item)


def _plain_call(fn: Callable[[T], R], item: T) -> R:
    """Worker wrapper without BLAS pinning (module-level so joblib can pickle it)."""
    return fn(item)


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    *,
    n_threads: int = 1,
    backend: str = "loky",
    progress: bool = False,
    description: str = "processing",
    pin_blas: bool = True,
) -> list[R]:
    """Map ``fn`` over ``items`` with SONATA's ``n_threads`` convention.

    Parameters
    ----------
    fn
        Unary callable applied to each item. Must be picklable when
        ``n_threads != 1`` (top-level function, not a lambda/closure).
    items
        Iterable of inputs. Materialised to a list to know the total up front.
    n_threads
        ``1`` serial, ``>=2`` that many workers, ``-1`` all usable cores.
    backend
        joblib backend (``"loky"`` process pool by default; ``"threading"`` for
        I/O-bound work).
    progress
        Show a lightweight progress bar (via :mod:`rich` if available).
    description
        Progress-bar label.
    pin_blas
        Pin inner BLAS threads to 1 in each *parallel* worker (recommended for
        numeric work). Ignored on the serial path, where a single task is free to
        use all BLAS threads.

    Returns
    -------
    list
        Results in input order.
    """
    items = list(items)
    workers = effective_workers(n_threads)

    # Serial path: one task at a time -> no oversubscription, so do NOT pin BLAS
    # (let the single task use every core it can).
    if workers == 1 or len(items) <= 1:
        return [fn(x) for x in _iter_with_progress(items, progress, description)]

    from joblib import Parallel, delayed

    worker = _pinned_call if pin_blas else _plain_call  # module-level -> picklable
    tasks = (delayed(worker)(fn, x) for x in items)
    verbose = 5 if progress else 0
    return Parallel(n_jobs=workers, backend=backend, verbose=verbose)(tasks)


def _iter_with_progress(items: Sequence[T], progress: bool, description: str) -> Iterator[T]:
    """Yield items, optionally wrapped in a rich progress bar (serial path)."""
    if not progress:
        yield from items
        return
    try:
        from rich.progress import track

        yield from track(items, description=description, total=len(items))
    except Exception:  # pragma: no cover - rich absent
        yield from items


__all__ = [
    "effective_workers",
    "pin_blas_threads",
    "parallel_map",
]
