# sonata/noninferiority.py
"""Aim-2 non-inferiority: spectral tract descriptors vs. classical diffusion scalars.

Same architecture, same edges, same CV — only the edge feature set differs
('spectral' vs 'diffusion'). We compare per-subject performance with a PAIRED
bootstrap two-one-sided-test (TOST). The margin is PRE-REGISTERED in
``NonInferiorityConfig.margin``; declaring non-inferiority from a non-significant
difference would be invalid (absence of evidence ≠ evidence of absence).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SonataConfig


def paired_bootstrap_tost(metric_a: np.ndarray, metric_b: np.ndarray, margin: float,
                          n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Non-inferiority of A vs B: A is non-inferior if lower (1−2α) CI of (A−B) > −margin."""
    a = np.asarray(metric_a, float); b = np.asarray(metric_b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if d.size == 0:
        return {"mean_diff": np.nan, "non_inferior": False, "equivalent": False}
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(d, d.size, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])           # (1−2α)
    lo2, hi2 = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])  # (1−α)
    return {"mean_diff": float(d.mean()), "ci_lo_1m2a": float(lo), "ci_hi_1m2a": float(hi),
            "ci_lo_1ma": float(lo2), "ci_hi_1ma": float(hi2),
            "margin": float(margin), "n": int(d.size),
            "non_inferior": bool(lo > -margin),
            "equivalent": bool(lo2 > -margin and hi2 < margin)}


def compare_feature_sets(res_spectral: dict, res_diffusion: dict,
                         cfg: SonataConfig) -> tuple[pd.DataFrame, dict]:
    """Align per-subject metrics of the two models and run the TOST."""
    a = res_spectral["metrics"].set_index("subject_id")
    b = res_diffusion["metrics"].set_index("subject_id")
    common = a.index.intersection(b.index)
    col = cfg.noinf_metric if hasattr(cfg, "noinf_metric") else cfg.noninf.metric
    paired = pd.DataFrame({
        "subject_id": common,
        "spectral": a.loc[common, col].to_numpy(),
        "diffusion": b.loc[common, col].to_numpy(),
    })
    paired["diff"] = paired["spectral"] - paired["diffusion"]
    test = paired_bootstrap_tost(paired["spectral"].to_numpy(),
                                 paired["diffusion"].to_numpy(),
                                 margin=cfg.noninf.margin,
                                 n_boot=cfg.noninf.n_bootstrap,
                                 alpha=cfg.noninf.alpha, seed=cfg.train.seed)
    test["metric"] = col
    return paired, test
