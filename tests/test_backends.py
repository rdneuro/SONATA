# tests/test_backends.py
"""Backend registry, cost model, and CPU array-namespace behaviour.

These run anywhere: they exercise the CPU path and the graceful fallbacks that
must hold when no GPU/sampler stack is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from sonata.backends import array as ab
from sonata.backends import base


def test_capabilities_cpu_always_present():
    caps = base.capabilities()
    assert caps.array["cpu"] is True
    assert isinstance(caps.has_cuda, bool)


def test_usable_threads_clamps_to_ceiling():
    caps = base.capabilities()
    ceiling = min(caps.n_cpus, base.MAX_CPU_THREADS)
    assert caps.usable_threads(-1) == ceiling
    assert caps.usable_threads(1) == 1
    assert caps.usable_threads(10_000) == ceiling
    assert caps.usable_threads(1) >= 1


def test_should_use_gpu_false_without_cuda():
    if base.capabilities().has_cuda:
        pytest.skip("CUDA present; the no-CUDA guarantee is not under test here")
    assert base.should_use_gpu(10**9, n_threads=1) is False


def test_resolve_array_backend_falls_back_to_cpu():
    # 'auto' with a huge problem still picks cpu when there is no GPU.
    if not base.capabilities().has_cuda:
        assert base.resolve_array_backend("auto", n_elements=10**9) == "cpu"
    # an explicit, absent backend degrades to cpu
    if not base.capabilities().array_available("torch"):
        assert base.resolve_array_backend("torch") == "cpu"


def test_resolve_bayes_backend_raises_when_none():
    caps = base.capabilities()
    if not any(caps.bayes.values()):
        with pytest.raises(RuntimeError):
            base.resolve_bayes_backend("auto")


def test_array_cpu_roundtrip_and_namespace():
    xp = ab.namespace("cpu")
    x = np.arange(12.0).reshape(3, 4)
    y = xp.sqrt(ab.to_backend(x, "cpu")) + 1.0
    assert np.allclose(ab.to_numpy(y), np.sqrt(x) + 1.0)
    # lifecycle helpers are no-ops on cpu and must not raise
    ab.synchronize("cpu")
    ab.free_memory("cpu")
    assert ab.device_free_bytes("cpu") is None
