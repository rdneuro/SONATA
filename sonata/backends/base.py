# sonata/backends/base.py
"""Backend registry, lazy capability probing, and the GPU cost-model decision.

SONATA has a handful of computationally heavy kernels (Laplace--Beltrami
eigensolves, edge aggregation, bootstrap resampling, GNN forward/backward) that
can run on several substrates. Rather than hard-wiring a device, every heavy
function accepts ``backend="auto"`` and delegates the *choice* to this module.

Design contract
---------------
* **Nothing here imports an optional heavy library at module load.** ``cupy``,
  ``jax`` and ``torch`` may be absent; probing them is done lazily and cached,
  so ``import sonata`` never fails because a GPU stack is missing.
* **A single source of truth for "what is available".** :func:`capabilities`
  returns a frozen snapshot; callers never re-probe by hand.
* **One cost model.** :func:`should_use_gpu` encodes the rule that GPU is only
  worth it when ``t_gpu_compute + t_transfer < t_cpu(n_threads)``; ``auto``
  resolution uses it so small problems stay on the CPU and avoid transfer
  overhead.

The two concrete dispatchers live next door: :mod:`sonata.backends.array`
(``cpu`` | ``jax`` | ``cupy`` | ``torch``) for numeric kernels, and
:mod:`sonata.backends.bayes` (``pymc`` | ``nutpie`` | ``numpyro`` | ``blackjax``)
for MCMC. Both consult this registry.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

ArrayBackend = Literal["cpu", "jax", "cupy", "torch"]
BayesBackend = Literal["pymc", "nutpie", "numpyro", "blackjax"]

#: Hard cap on CPU worker threads for the cost model (user hardware: 22 usable).
MAX_CPU_THREADS: int = 22


# ── low-level availability probes (cached, import-free) ───────────────────────
def _installed(module: str) -> bool:
    """True if ``module`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def _cuda_via_torch() -> bool:
    """Probe a working CUDA device through torch (the most common path).

    Importing torch is comparatively cheap and is a hard dependency of the GNN,
    so this probe is acceptable; it is cached to a single call per process.
    """
    if not _installed("torch"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive, driver/runtime mismatches
        return False


@lru_cache(maxsize=1)
def _cuda_via_cupy() -> bool:
    if not _installed("cupy"):
        return False
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover
        return False


@lru_cache(maxsize=1)
def _jax_has_gpu() -> bool:
    if not _installed("jax"):
        return False
    try:
        import jax

        return any(d.platform in ("gpu", "cuda", "rocm") for d in jax.devices())
    except Exception:  # pragma: no cover
        return False


# ── capability snapshot ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class Capabilities:
    """Immutable snapshot of what this machine can actually run."""

    array: dict[ArrayBackend, bool]
    bayes: dict[BayesBackend, bool]
    has_cuda: bool
    n_cpus: int = field(default_factory=lambda: os.cpu_count() or 1)

    def array_available(self, name: ArrayBackend) -> bool:
        return self.array.get(name, False)

    def bayes_available(self, name: BayesBackend) -> bool:
        return self.bayes.get(name, False)

    def usable_threads(self, requested: int) -> int:
        """Translate an ``n_threads`` request into an effective worker count.

        ``1`` -> serial, ``-1`` -> all cores, otherwise the request; every
        result is clamped to ``[1, min(n_cpus, MAX_CPU_THREADS)]`` so the cost
        model and joblib never oversubscribe the machine.
        """
        ceiling = min(self.n_cpus, MAX_CPU_THREADS)
        if requested == -1:
            return ceiling
        return max(1, min(int(requested), ceiling))


@lru_cache(maxsize=1)
def capabilities() -> Capabilities:
    """Return a cached snapshot of installed backends and CUDA availability."""
    has_cuda = _cuda_via_torch() or _cuda_via_cupy() or _jax_has_gpu()
    array = {
        "cpu": True,  # numpy is always present
        "jax": _installed("jax"),
        "cupy": _cuda_via_cupy(),  # cupy without a device is useless -> False
        "torch": _installed("torch"),
    }
    bayes = {
        "pymc": _installed("pymc"),
        "nutpie": _installed("nutpie"),
        "numpyro": _installed("numpyro"),
        "blackjax": _installed("blackjax"),
    }
    return Capabilities(array=array, bayes=bayes, has_cuda=has_cuda)


# ── the GPU cost model ────────────────────────────────────────────────────────
def should_use_gpu(
    n_elements: int,
    *,
    n_threads: int = 1,
    flops_per_element: float = 1.0,
    dtype_bytes: int = 8,
    gpu_threshold_elements: int = 200_000,
) -> bool:
    """Decide whether a kernel of ``n_elements`` is worth moving to the GPU.

    Implements the user's rule: prefer the GPU only when the on-device compute
    *plus* the host<->device transfer is cheaper than the CPU time using up to
    ``min(n_threads, MAX_CPU_THREADS)`` workers. Because we cannot time a kernel
    before running it, we use a calibrated size threshold that scales with the
    per-element work and shrinks as more CPU threads become available (more CPU
    threads raise the bar the GPU must clear).

    Parameters
    ----------
    n_elements
        Problem size (e.g. mesh vertices, matrix entries, bootstrap draws).
    n_threads
        CPU workers the caller would otherwise use (``-1`` -> all cores).
    flops_per_element
        Relative arithmetic intensity; heavier kernels favour the GPU sooner.
    dtype_bytes
        Element size in bytes; larger transfers favour the CPU.
    gpu_threshold_elements
        Base break-even size for a light (flops=1, fp64) kernel on 1 CPU thread.

    Returns
    -------
    bool
        ``True`` to use the GPU (only ever ``True`` when CUDA is present).
    """
    caps = capabilities()
    if not caps.has_cuda:
        return False
    threads = caps.usable_threads(n_threads)
    # Effective work; transfer cost scales with bytes moved, so a larger dtype
    # raises the break-even point (penalises the GPU).
    work = n_elements * max(flops_per_element, 1e-6)
    transfer_penalty = dtype_bytes / 8.0
    # More CPU threads make the CPU cheaper -> the GPU must clear a higher bar.
    break_even = gpu_threshold_elements * threads * transfer_penalty
    return work >= break_even


def resolve_array_backend(
    name: ArrayBackend | Literal["auto"],
    *,
    n_elements: int = 0,
    n_threads: int = 1,
    prefer_gpu: tuple[ArrayBackend, ...] = ("torch", "cupy", "jax"),
    **cost_kwargs,
) -> ArrayBackend:
    """Resolve ``"auto"`` (or validate an explicit name) to a usable backend.

    ``auto`` uses :func:`should_use_gpu`; if the GPU wins, the first available
    backend in ``prefer_gpu`` is returned, otherwise ``"cpu"``. An explicit name
    that is unavailable falls back to ``"cpu"`` with the caller free to warn.
    """
    caps = capabilities()
    if name != "auto":
        return name if caps.array_available(name) else "cpu"
    if should_use_gpu(n_elements, n_threads=n_threads, **cost_kwargs):
        for cand in prefer_gpu:
            if caps.array_available(cand):
                return cand
    return "cpu"


def resolve_bayes_backend(
    name: BayesBackend | Literal["auto"],
    *,
    prefer: tuple[BayesBackend, ...] = ("nutpie", "numpyro", "blackjax", "pymc"),
) -> BayesBackend:
    """Resolve ``"auto"`` (or validate a name) to an available sampler backend.

    Default preference favours the fastest compiled NUTS samplers first and
    falls back to the pure-PyMC sampler, which is always present if PyMC is.
    """
    caps = capabilities()
    if name != "auto":
        if caps.bayes_available(name):
            return name
        # explicit-but-absent -> fall through to preference order
    for cand in prefer:
        if caps.bayes_available(cand):
            return cand
    raise RuntimeError(
        "No Bayesian sampler backend available. Install one of: "
        "pymc, nutpie, numpyro, blackjax (e.g. `pip install -e '.[bayes]'`)."
    )


__all__ = [
    "ArrayBackend",
    "BayesBackend",
    "MAX_CPU_THREADS",
    "Capabilities",
    "capabilities",
    "should_use_gpu",
    "resolve_array_backend",
    "resolve_bayes_backend",
]
