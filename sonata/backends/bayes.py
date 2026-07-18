# sonata/backends/bayes.py
"""MCMC sampler dispatch across ``pymc`` | ``nutpie`` | ``numpyro`` | ``blackjax``.

The Bayesian *attribution* model (regularised-horseshoe block-sparse regression
that recovers per-tract / per-region weights, rather than a point predictor) is
expressed once as a PyMC model. Which NUTS engine draws from it is a backend
choice: PyMC 5 exposes all four through ``pm.sample(nuts_sampler=...)``, and the
compiled samplers (``nutpie``, ``numpyro``, ``blackjax``) are often several times
faster than the default. This module resolves ``"auto"`` to the fastest engine
present and drives sampling with a single, uniform call.

Nothing is imported until :func:`sample` runs, so importing SONATA never pulls in
JAX or a compiler toolchain.
"""

from __future__ import annotations

from typing import Any, Literal

from .base import BayesBackend, resolve_bayes_backend

#: JAX-based engines that benefit from a GPU/TPU when one is present.
_JAX_SAMPLERS: frozenset[str] = frozenset({"numpyro", "blackjax"})


def sample(
    model: Any,
    *,
    backend: BayesBackend | Literal["auto"] = "auto",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int | None = 1234,
    progressbar: bool = True,
    **sample_kwargs,
):
    """Draw from a PyMC ``model`` using the resolved sampler backend.

    Parameters
    ----------
    model
        A PyMC :class:`~pymc.Model` instance (or used inside a ``with model:``).
    backend
        ``"auto"`` picks the fastest available engine (nutpie > numpyro >
        blackjax > pymc); an explicit name falls back to the preference order if
        absent.
    draws, tune, chains, target_accept, random_seed, progressbar
        Standard NUTS controls, forwarded to :func:`pymc.sample`.
    **sample_kwargs
        Extra keyword arguments passed through to :func:`pymc.sample`.

    Returns
    -------
    arviz.InferenceData
        The posterior trace, uniform across backends.
    """
    import pymc as pm

    engine = resolve_bayes_backend(backend)
    kwargs: dict[str, Any] = dict(
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
        progressbar=progressbar,
        nuts_sampler=engine,
        **sample_kwargs,
    )
    # numpyro/blackjax run through JAX and use their own chain method; letting
    # PyMC pick the default chain method keeps the call uniform across engines.
    ctx = model if isinstance(model, pm.Model) else pm.modelcontext(None)
    with ctx:
        return pm.sample(**kwargs)


def resolve(backend: BayesBackend | Literal["auto"] = "auto") -> BayesBackend:
    """Expose the resolved sampler name (thin re-export for callers/tests)."""
    return resolve_bayes_backend(backend)


__all__ = ["sample", "resolve"]
