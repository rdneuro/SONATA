#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SONATA quickstart — a guided tour in Spyder `# %%` cells.

Each cell runs its stage inline (interleaved figures/tables), so you can step
through the library in Spyder without a define-all-run-at-end script. Cells that
need the full scientific stack (SpectralBrain, torch, PyMC) are marked; the infra
and visualization cells run on the scientific-Python core alone.
"""

# %% [0] What can this machine actually run? ----------------------------------
from sonata import backends

caps = backends.capabilities()
print("CUDA available :", caps.has_cuda)
print("array backends :", {k: v for k, v in caps.array.items() if v})
print("bayes backends :", {k: v for k, v in caps.bayes.items() if v})
print("usable threads :", caps.usable_threads(-1))

# The GPU cost model: small problems stay on the CPU (transfer would dominate).
print("use GPU for 1e4 elems? ", backends.should_use_gpu(10_000))
print("use GPU for 5e6 elems? ", backends.should_use_gpu(5_000_000))


# %% [1] Parallel CPU work with the n_threads convention ----------------------
import numpy as np

from sonata.parallel import parallel_map


def _row_norm(x):                       # top-level -> picklable for joblib
    return float(np.linalg.norm(x))


mats = [np.random.default_rng(i).standard_normal((256, 256)) for i in range(32)]
norms = parallel_map(_row_norm, mats, n_threads=-1, progress=True,
                     description="norms")
print("computed", len(norms), "norms in parallel")


# %% [2] Memory-safe batching for a device loop -------------------------------
from sonata.memory import estimate_batch_size, run_in_oom_safe_batches

batch = estimate_batch_size(item_bytes=64 << 20, backend="cpu")  # 64 MiB items
print("suggested batch size:", batch)

# A batched op that halves its batch and retries if it ever hits OOM.
doubled = run_in_oom_safe_batches(lambda chunk: [c * 2 for c in chunk],
                                  list(range(20)), initial_batch=batch)
print("order-preserving result head:", doubled[:5])


# %% [3] Build features for the whole cohort in parallel ----------------------
# Requires the pipeline extra (nibabel, torch, SpectralBrain) + real data paths.
#
# from sonata import SonataConfig
# from sonata.graph import build_all_subject_features, load_all_cached_features
# from sonata.utils import load_manifest
#
# cfg = SonataConfig(); cfg.ensure()
# manifest = load_manifest(cfg.paths.manifest_csv)
# status = build_all_subject_features(manifest, cfg, n_threads=-1)   # replaces the old script
# feats = load_all_cached_features(cfg)


# %% [4] Visualize the evolution of a matrix across processing stages ---------
from sonata import viz

rng = np.random.default_rng(0)
base = rng.standard_normal((40, 40)); base = (base + base.T) / 2
stages = [((1 - 0.07 * t) * base + 0.07 * t * rng.standard_normal((40, 40)))
          for t in range(12)]
stages = [(m + m.T) / 2 for m in stages]

fig = viz.evolutionary_heatmaps(stages, stage_labels=[f"epoch {5*t}" for t in range(12)],
                                summary="mean_abs", suptitle="FC across training")
viz.save(fig, "quickstart_evolution", "figures")


# %% [5] The one-figure results dashboard -------------------------------------
pred = rng.standard_normal(600)
true = 0.3 * pred + 0.9 * rng.standard_normal(600)
fig = viz.results_dashboard(
    model_labels=["group-mean", "GNN-onehot", "linear", "SONATA"],
    model_scores=[0.62, 0.37, 0.24, 0.07], benchmark=0.62,
    pred=pred, true=true,
    ni_labels=["spectral vs diffusion"], ni_estimate=[-0.119],
    ni_low=[-0.18], ni_high=[-0.05], ni_margin=-0.03,
)
viz.save(fig, "quickstart_dashboard", "figures")


# %% [6] Bayesian attribution: which features carry the signal? ---------------
# Requires the bayes extra (pymc, arviz, and a sampler). Reframes SONATA from
# prediction to attribution — see the project report.
#
# from sonata.attribution import fit_attribution
#
# X = rng.standard_normal((110, 40))       # per-subject aggregated features
# beta_true = np.zeros(40); beta_true[[3, 11, 27]] = [0.8, -0.6, 0.5]
# y = X @ beta_true + 0.5 * rng.standard_normal(110)
# res = fit_attribution(X, y, feature_names=[f"tract_{i}" for i in range(40)],
#                       backend="auto", draws=1000, tune=1000)
# print("credibly non-null features:",
#       [n for n, s in zip(res.feature_names, res.significant()) if s])

print("quickstart complete — see ./figures for the rendered panels")
