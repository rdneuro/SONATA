# sonata/backends/array.py
"""Uniform array-namespace dispatch across ``cpu`` | ``jax`` | ``cupy`` | ``torch``.

A heavy numeric kernel is written *once* against the namespace returned by
:func:`namespace` and dispatched by name. Host<->device transfer, stream
synchronisation and cache freeing are provided as thin, backend-uniform helpers
so callers never branch on the device themselves.

Only ``cpu`` (NumPy) is imported eagerly; every accelerator is imported lazily on
first use, so this module is import-safe on a machine with no GPU stack.

Typical use::

    from sonata.backends import array as ab

    def kernel(x, backend="auto", n_threads=1):
        be = ab.resolve(backend, n_elements=x.size, n_threads=n_threads)
        xp = ab.namespace(be)
        xg = ab.to_backend(x, be)
        y = xp.sqrt(xg) + 1.0          # written once, runs anywhere
        ab.synchronize(be)
        return ab.to_numpy(y)          # always hand NumPy back to the caller
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .base import ArrayBackend, capabilities, resolve_array_backend

resolve = resolve_array_backend  # re-export under a short local name


# ── namespace resolution (lazy) ───────────────────────────────────────────────
def namespace(backend: ArrayBackend):
    """Return the array module implementing the NumPy-like API for ``backend``.

    ``torch`` is wrapped so that ``.sqrt``, ``.exp``, ``.percentile`` etc. resolve
    against ``torch`` with NumPy-compatible names; the raw ``torch`` module is
    already close enough for the elementwise ops SONATA needs.
    """
    if backend == "cpu":
        return np
    if backend == "jax":
        import jax.numpy as jnp

        return jnp
    if backend == "cupy":
        import cupy

        return cupy
    if backend == "torch":
        import torch

        return torch
    raise ValueError(f"unknown array backend {backend!r}")


# ── transfer helpers ──────────────────────────────────────────────────────────
def to_backend(x: np.ndarray, backend: ArrayBackend, *, dtype=None, device: str | None = None):
    """Move a NumPy array onto ``backend`` (identity for ``cpu``)."""
    if backend == "cpu":
        return np.asarray(x, dtype=dtype)
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.asarray(x, dtype=dtype)
    if backend == "cupy":
        import cupy

        return cupy.asarray(x, dtype=dtype)
    if backend == "torch":
        import torch

        dev = device or ("cuda" if capabilities().has_cuda else "cpu")
        t = torch.as_tensor(np.asarray(x))
        if dtype is not None:
            t = t.to(getattr(torch, np.dtype(dtype).name, t.dtype))
        return t.to(dev)
    raise ValueError(f"unknown array backend {backend!r}")


def to_numpy(x: Any) -> np.ndarray:
    """Bring any backend array back to a NumPy array on the host."""
    if isinstance(x, np.ndarray):
        return x
    mod = type(x).__module__
    if mod.startswith("torch"):
        return x.detach().cpu().numpy()
    if mod.startswith("cupy"):
        import cupy

        return cupy.asnumpy(x)
    if mod.startswith("jax"):
        return np.asarray(x)
    return np.asarray(x)


# ── device lifecycle (used by the memory manager and SpMV loops) ──────────────
def synchronize(backend: ArrayBackend) -> None:
    """Block until queued device work has finished (no-op on CPU/JAX-CPU)."""
    if backend == "torch":
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    elif backend == "cupy":
        import cupy

        cupy.cuda.Stream.null.synchronize()
    elif backend == "jax":
        # JAX is async but has no global device sync; results block on transfer to
        # NumPy (to_numpy) instead, so there is nothing to do here.
        return


def free_memory(backend: ArrayBackend) -> None:
    """Release cached device memory back to the driver (helps avoid OOM)."""
    if backend == "torch":
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif backend == "cupy":
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()


def device_free_bytes(backend: ArrayBackend) -> int | None:
    """Free VRAM in bytes for ``backend``'s device, or ``None`` if not on GPU."""
    if backend == "torch":
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return int(free)
    elif backend == "cupy":
        import cupy

        free, _total = cupy.cuda.runtime.memGetInfo()
        return int(free)
    return None


__all__ = [
    "resolve",
    "namespace",
    "to_backend",
    "to_numpy",
    "synchronize",
    "free_memory",
    "device_free_bytes",
]
