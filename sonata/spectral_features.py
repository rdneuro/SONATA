# sonata/spectral_features.py
"""Fixed-length Laplace–Beltrami spectral descriptors per surface (SpectralBrain).

A node (ROI sub-mesh) or an edge (tract isosurface) is summarized into a single
fixed-length vector so that every graph across every subject has identically
shaped feature matrices. Per-vertex descriptors (HKS, WKS, GPS, BKS) are reduced
across vertices with robust order statistics; ShapeDNA is a fixed-length
eigenvalue fingerprint.

Two design points enforced here
-------------------------------
1. **Scale/size invariance of the truncation (``fixed_k``).** The spectral
   descriptors are only comparable across surfaces if they are truncated at the
   *same* eigen-index ``k``. With a per-mesh ``k = min(n_eigen, n_vertices-2)``
   (the earlier behaviour), small surfaces receive a shorter spectrum padded with
   NaN, so the *effective* descriptor length would depend on mesh **size** — a
   size×shape confound a spectral-geometry reviewer would rightly flag. With
   ``fixed_k=True`` (default) every surface is decomposed at the same ``k`` and a
   surface too small to support it yields an all-NaN vector (flagged downstream
   and imputed inside CV), so the descriptor encodes shape, not size.
2. **Mesh conditioning** (Reuter 2006): Taubin smoothing + a genus guard protect
   the LB spectrum against segmentation-induced handles.

The heavy per-surface work is embarrassingly parallel across surfaces;
:func:`surface_descriptors_batch` maps it with the library-wide ``n_threads``
convention (see :mod:`sonata.parallel`).
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from .config import SpectralConfig
from .parallel import parallel_map

_SUMM_FNS = ("mean", "std", "p10", "p50", "p90")   # 5 robust summaries per channel


def descriptor_length(cfg: SpectralConfig) -> int:
    return len(descriptor_names(cfg))


def descriptor_names(cfg: SpectralConfig) -> list[str]:
    names: list[str] = []
    if "shapedna" in cfg.use_descriptors:
        names += [f"sdna_{i}" for i in range(cfg.n_eigen - 1)]
    if "hks" in cfg.use_descriptors:
        names += [f"hks_{s}_{t}" for t in range(cfg.hks_n_times) for s in _SUMM_FNS]
    if "wks" in cfg.use_descriptors:
        names += [f"wks_{s}_{e}" for e in range(cfg.wks_n_energies) for s in _SUMM_FNS]
    if "gps" in cfg.use_descriptors:
        names += [f"gps_std_{d}" for d in range(10)]
    if "bks" in cfg.use_descriptors:
        names += [f"bks_{s}" for s in _SUMM_FNS]
    return names


def _summarize(arr: np.ndarray) -> np.ndarray:
    """Reduce a (N_vertices, C) per-vertex descriptor to (5*C,) order stats."""
    if arr.ndim == 1:
        arr = arr[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stats = np.vstack([
            np.nanmean(arr, axis=0), np.nanstd(arr, axis=0),
            np.nanpercentile(arr, 10, axis=0), np.nanpercentile(arr, 50, axis=0),
            np.nanpercentile(arr, 90, axis=0),
        ])                                   # (5, C)
    return stats.T.reshape(-1)               # interleave -> [c0_mean,c0_std,...]


def _fit_length(vec: np.ndarray, length: int) -> np.ndarray:
    out = np.full(length, np.nan)
    n = min(len(vec), length)
    out[:n] = vec[:n]
    return out


def _resolve_k(cfg: SpectralConfig, n_vertices: int) -> int | None:
    """Pick the eigen-truncation ``k`` for a surface.

    With ``fixed_k`` (default) return ``n_eigen`` iff the mesh can support it,
    else ``None`` (caller returns an all-NaN descriptor). Otherwise fall back to
    the size-adaptive ``k`` (kept for ablation/back-compat).
    """
    if getattr(cfg, "fixed_k", True):
        k = int(cfg.n_eigen)
        return k if n_vertices >= k + 2 else None
    return int(min(cfg.n_eigen, max(4, n_vertices - 2)))


def surface_descriptor(vertices: np.ndarray, faces: np.ndarray,
                       cfg: SpectralConfig) -> np.ndarray:
    """Compute the fixed-length spectral descriptor for one surface.

    Returns an all-NaN vector of the correct length if the surface is too small,
    cannot support the common truncation ``k`` (when ``fixed_k``), or if its
    Laplace–Beltrami decomposition fails.
    """
    L = descriptor_length(cfg)
    # Cheap guards first — these return before importing the heavy spectral
    # library, so the "surface too small / unsupported k" paths cost nothing.
    if (vertices is None or faces is None or len(vertices) < cfg.min_vertices
            or len(faces) == 0):
        return np.full(L, np.nan)

    n_vertices = int(len(vertices))
    k = _resolve_k(cfg, n_vertices)
    if k is None:  # too small to support the common truncation -> flag as missing
        return np.full(L, np.nan)

    import spectralbrain as sb  # heavy import kept local and after the guards

    try:
        mesh = sb.BrainMesh(np.asarray(vertices, float), np.asarray(faces))
        if cfg.taubin_iter > 0:
            mesh = mesh.taubin_smooth(n_iterations=cfg.taubin_iter,
                                      lambda_=cfg.taubin_lambda, mu=cfg.taubin_mu)
        try:
            g = mesh.genus()
            if g is not None and g > cfg.max_genus:
                warnings.warn(f"high genus={g}; spectrum may be unstable")
        except Exception:
            pass
        # Re-check against the *actual* vertex count after conditioning.
        if getattr(cfg, "fixed_k", True) and mesh.n_vertices < k + 2:
            return np.full(L, np.nan)
        decomp = mesh.decompose(k=k, laplacian_method=cfg.laplacian)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"decompose failed: {exc!r}")
        return np.full(L, np.nan)

    parts: list[np.ndarray] = []
    try:
        if "shapedna" in cfg.use_descriptors:
            sdna = np.asarray(sb.compute_shapedna(
                decomp, normalize=cfg.shapedna_normalize, skip_zero=True), float)
            parts.append(_fit_length(sdna, cfg.n_eigen - 1))
        if "hks" in cfg.use_descriptors:
            hks = np.asarray(sb.compute_hks(decomp, n_times=cfg.hks_n_times), float)
            parts.append(_summarize(hks))
        if "wks" in cfg.use_descriptors:
            wks = np.asarray(sb.compute_wks(decomp, n_energies=cfg.wks_n_energies), float)
            parts.append(_summarize(wks))
        if "gps" in cfg.use_descriptors:
            gps = np.asarray(sb.compute_gps(decomp, skip_zero=True), float)
            std = np.nanstd(gps, axis=0) if gps.ndim == 2 else np.array([np.nanstd(gps)])
            parts.append(_fit_length(std, 10))
        if "bks" in cfg.use_descriptors:
            bks = np.asarray(sb.compute_bks(decomp), float)
            parts.append(_summarize(bks))
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"descriptor computation failed: {exc!r}")
        return np.full(L, np.nan)

    return _fit_length(np.concatenate(parts), L)


def _descriptor_from_item(item: tuple[np.ndarray, np.ndarray], cfg: SpectralConfig) -> np.ndarray:
    """Top-level worker (picklable) for :func:`surface_descriptors_batch`."""
    vertices, faces = item
    return surface_descriptor(vertices, faces, cfg)


def surface_descriptors_batch(
    surfaces: Sequence[tuple[np.ndarray, np.ndarray]],
    cfg: SpectralConfig,
    *,
    n_threads: int = 1,
    progress: bool = False,
) -> np.ndarray:
    """Compute descriptors for many surfaces, optionally in parallel.

    Parameters
    ----------
    surfaces
        Sequence of ``(vertices, faces)`` pairs.
    cfg
        Spectral configuration (shared by every surface).
    n_threads
        ``1`` serial, ``>=2`` joblib workers, ``-1`` all usable cores. Inner BLAS
        threads are pinned to 1 per worker (see :mod:`sonata.parallel`).
    progress
        Show a progress bar.

    Returns
    -------
    numpy.ndarray
        Stacked descriptors of shape ``(len(surfaces), descriptor_length(cfg))``.
    """
    from functools import partial

    fn = partial(_descriptor_from_item, cfg=cfg)
    rows = parallel_map(fn, list(surfaces), n_threads=n_threads,
                        progress=progress, description="spectral descriptors")
    L = descriptor_length(cfg)
    if not rows:
        return np.empty((0, L))
    return np.vstack(rows)


__all__ = [
    "descriptor_length",
    "descriptor_names",
    "surface_descriptor",
    "surface_descriptors_batch",
]
