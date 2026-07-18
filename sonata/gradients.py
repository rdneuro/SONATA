# sonata/gradients.py
"""Structural connectome gradients — node features, a baseline, and an interpretive axis.

We fit diffusion-map gradients on the TRAIN group-mean SIFT2 matrix (structural,
so no FC leakage), then expose:
- per-ROI gradient coordinates (200 × K) as optional node features;
- a per-edge gradient distance |g_i − g_j| used by the gradient-coupling
  baseline and by the 'coupling along the principal gradient' figure, which is
  where the SC–FC literature locates the decoupling (Preti & Van De Ville 2019).

BrainSpace is used when available; otherwise a transparent NumPy diffusion-map
fallback (normalized-angle affinity → symmetric Laplacian eigenmaps) runs so the
pipeline never blocks.
"""

from __future__ import annotations

import warnings

import numpy as np

N_PARCELS = 200


def rebuild_sc(feat: dict) -> np.ndarray:
    """Dense symmetric SC matrix from a subject's edge list + weights."""
    W = np.zeros((N_PARCELS, N_PARCELS))
    e = feat["edges"]; w = feat["sc_weight"]
    W[e[:, 0], e[:, 1]] = w
    W[e[:, 1], e[:, 0]] = w
    return W


def group_mean_sc(train_feats: list[dict]) -> np.ndarray:
    return np.mean([rebuild_sc(f) for f in train_feats], axis=0)


def fit_gradients(train_feats: list[dict], n_components: int = 5) -> np.ndarray:
    """Return reference gradients (200 × K) fit on the train group-mean SC."""
    W = group_mean_sc(train_feats)
    try:
        from brainspace.gradient import GradientMaps
        gm = GradientMaps(n_components=n_components, approach="dm",
                          kernel="normalized_angle", random_state=0)
        gm.fit(W)
        G = np.asarray(gm.gradients_)
    except Exception as exc:
        warnings.warn(f"BrainSpace unavailable ({exc!r}); NumPy diffusion-map fallback")
        G = _diffusion_map(W, n_components)
    # Replace any all-zero ROI rows (no connections) with column means.
    bad = ~np.isfinite(G).all(axis=1)
    if bad.any():
        G[bad] = np.nanmean(G[~bad], axis=0)
    return G


def _diffusion_map(W: np.ndarray, k: int) -> np.ndarray:
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    cos = (W @ W.T) / (norms * norms.T)
    aff = 1.0 - np.arccos(np.clip(cos, -1, 1)) / np.pi      # normalized-angle affinity
    np.fill_diagonal(aff, 0.0)
    d = aff.sum(1); d[d == 0] = 1.0
    L = (aff / np.sqrt(np.outer(d, d)))
    vals, vecs = np.linalg.eigh(L)
    order = np.argsort(vals)[::-1]
    return vecs[:, order[1:k + 1]] * np.sqrt(np.abs(vals[order[1:k + 1]]))


def edge_gradient_distance(edges: np.ndarray, G: np.ndarray, axis: int = 0) -> np.ndarray:
    """|g_i − g_j| along gradient `axis` for each edge."""
    return np.abs(G[edges[:, 0], axis] - G[edges[:, 1], axis])
