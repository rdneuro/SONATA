# sonata/attribution.py
"""Bayesian attribution: which tracts/regions carry the structure-function signal.

SONATA's empirical ceiling is a property of the *prediction* problem — no method
beats the group-average FC out of sample. The scientifically productive reframing
(see the project report) is *attribution*: instead of predicting FC, ask which
structural features carry the small individual signal that does exist, with
honest posterior uncertainty. This module implements a **regularized-horseshoe**
(Piironen & Vehtari, 2017) sparse linear model, optionally **group/block-sparse**
so an entire tract's spectral block is shrunk together, and drives sampling
through :mod:`sonata.backends.bayes` (``pymc`` | ``nutpie`` | ``numpyro`` |
``blackjax``).

The horseshoe is the right prior here: a heavy-tailed global--local scale mixture
that shrinks noise coefficients hard toward zero while leaving genuinely large
effects almost unshrunk, so the posterior inclusion structure *is* the
attribution. This module needs PyMC + ArviZ; both are lazy-imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .backends import bayes as bayes_backend


@dataclass
class AttributionResult:
    """Tidy summary of a fitted attribution model."""

    feature_names: list[str]
    mean: np.ndarray          # posterior mean of each coefficient
    hdi_low: np.ndarray       # lower bound of the highest-density interval
    hdi_high: np.ndarray      # upper bound
    prob_direction: np.ndarray  # P(sign matches posterior mean) in [0.5, 1]
    idata: object             # the full arviz.InferenceData (for diagnostics)

    def significant(self, hdi_excludes_zero: bool = True) -> np.ndarray:
        """Boolean mask of features whose HDI excludes zero (credibly non-null)."""
        if hdi_excludes_zero:
            return (self.hdi_low > 0) | (self.hdi_high < 0)
        return self.prob_direction >= 0.975


def build_horseshoe_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: Sequence[int] | None = None,
    expected_nonzero: float = 5.0,
    slab_scale: float = 2.0,
    slab_df: float = 4.0,
    standardize: bool = True,
):
    """Construct a (regularized) horseshoe regression PyMC model.

    Parameters
    ----------
    X
        Design matrix ``(N, D)`` — e.g. per-subject aggregated tract-spectral
        edge features (or their region-level pooling).
    y
        Response ``(N,)`` — e.g. a per-subject FC-prediction score, or a target
        FC edge/strength to attribute.
    groups
        Optional length-``D`` integer group id per column; columns sharing a group
        share a local scale (**group horseshoe**), so an entire tract block is
        included or excluded together. ``None`` gives the element-wise horseshoe.
    expected_nonzero
        Prior guess ``p0`` for the number of relevant features; sets the global
        scale ``tau0 = p0/(D-p0) * sigma/sqrt(N)`` (Piironen & Vehtari).
    slab_scale, slab_df
        Regularizing slab: large effects are shrunk toward a Student-t slab of
        this scale and degrees of freedom, bounding otherwise-unregularized tails.
    standardize
        Z-score columns of ``X`` (recommended so the shared global scale is
        meaningful across features).

    Returns
    -------
    pymc.Model
        The model, ready for :func:`fit_attribution` or manual ``pm.sample``.
    """
    import pymc as pm
    import pytensor.tensor as pt

    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    N, D = X.shape
    if standardize:
        mu = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd

    p0 = float(np.clip(expected_nonzero, 1.0, D - 1))
    with pm.Model() as model:
        Xd = pm.Data("X", X)
        yd = pm.Data("y", y)

        sigma = pm.HalfNormal("sigma", sigma=float(np.std(y) + 1e-6))
        # Global scale: half-Cauchy centred at the Piironen–Vehtari tau0.
        tau0 = (p0 / (D - p0)) * (sigma / np.sqrt(N))
        tau = pm.HalfCauchy("tau", beta=1.0) * tau0

        # Local scales — one per column, or one per group (group horseshoe).
        if groups is None:
            lam = pm.HalfCauchy("lam", beta=1.0, shape=D)
            lam_col = lam
        else:
            groups = np.asarray(groups, int)
            n_groups = int(groups.max()) + 1
            lam_g = pm.HalfCauchy("lam_group", beta=1.0, shape=n_groups)
            lam_col = lam_g[groups]

        # Regularizing slab (Student-t): c^2 ~ InvGamma(slab_df/2, slab_df/2 * s^2).
        c2 = pm.InverseGamma("c2", alpha=slab_df / 2.0,
                             beta=(slab_df / 2.0) * slab_scale**2)
        lam_tilde = pt.sqrt(c2 * lam_col**2 / (c2 + tau**2 * lam_col**2))

        z = pm.Normal("z", 0.0, 1.0, shape=D)
        beta = pm.Deterministic("beta", z * tau * lam_tilde)
        intercept = pm.Normal("intercept", 0.0, 5.0)

        mu_y = intercept + pt.dot(Xd, beta)
        pm.Normal("obs", mu=mu_y, sigma=sigma, observed=yd)
    return model


def fit_attribution(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    groups: Sequence[int] | None = None,
    backend: str = "auto",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    hdi_prob: float = 0.94,
    random_seed: int | None = 1234,
    **model_kwargs,
) -> AttributionResult:
    """Fit the horseshoe attribution model and return a tidy per-feature summary.

    Sampling backend is resolved by :mod:`sonata.backends.bayes` (``"auto"`` picks
    the fastest available NUTS engine). The high ``target_accept`` is deliberate:
    the horseshoe funnel geometry demands it to avoid divergences.

    Returns
    -------
    AttributionResult
        Posterior means, HDIs, direction probabilities, and the full trace.
    """
    import arviz as az

    X = np.asarray(X, float)
    D = X.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(D)]

    model = build_horseshoe_model(X, y, groups=groups, **model_kwargs)
    idata = bayes_backend.sample(
        model, backend=backend, draws=draws, tune=tune, chains=chains,
        target_accept=target_accept, random_seed=random_seed,
    )

    post = idata.posterior["beta"]                       # (chain, draw, D)
    flat = post.stack(sample=("chain", "draw")).values   # (D, S)
    mean = flat.mean(1)
    hdi = az.hdi(idata, var_names=["beta"], hdi_prob=hdi_prob)["beta"].values  # (D, 2)
    # Probability of direction: fraction of draws sharing the posterior-mean sign.
    pd_ = np.maximum((flat > 0).mean(1), (flat < 0).mean(1))

    return AttributionResult(
        feature_names=list(feature_names),
        mean=mean,
        hdi_low=hdi[:, 0],
        hdi_high=hdi[:, 1],
        prob_direction=pd_,
        idata=idata,
    )


__all__ = ["AttributionResult", "build_horseshoe_model", "fit_attribution"]
