# sonata/backends/__init__.py
"""Multi-backend compute layer for SONATA's heavy kernels.

Two dispatchers share one capability registry:

* :mod:`sonata.backends.array` — numeric kernels over ``cpu`` | ``jax`` |
  ``cupy`` | ``torch``.
* :mod:`sonata.backends.bayes` — MCMC over ``pymc`` | ``nutpie`` | ``numpyro`` |
  ``blackjax``.

The registry (:mod:`sonata.backends.base`) probes availability lazily and holds
the GPU cost model, so every heavy function can accept ``backend="auto"`` and let
the library choose the substrate. See each submodule for details.
"""

from __future__ import annotations

from . import array, bayes
from .base import (
    Capabilities,
    capabilities,
    resolve_array_backend,
    resolve_bayes_backend,
    should_use_gpu,
)

__all__ = [
    "array",
    "bayes",
    "capabilities",
    "Capabilities",
    "should_use_gpu",
    "resolve_array_backend",
    "resolve_bayes_backend",
]
