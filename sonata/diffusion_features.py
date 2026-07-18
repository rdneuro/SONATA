# sonata/diffusion_features.py
"""Per-bundle classical diffusion scalars (FA/MD/AD/RD) — the Aim-2 comparator.

For each TractSeg bundle mask we sample the within-bundle distribution of each
scalar map and reduce it to mean+std. These vectors play exactly the same role
in the edge-feature pipeline as the spectral tract descriptors, so the
non-inferiority test (``noninferiority.py``) compares like with like: same
architecture, same edges, spectral edge features vs. diffusion edge features.
"""

from __future__ import annotations

import warnings

import numpy as np
import nibabel as nib

from .config import TractConfig


def diffusion_length(cfg: TractConfig) -> int:
    return 2 * len(cfg.diffusion_scalars)


def diffusion_names(cfg: TractConfig) -> list[str]:
    out: list[str] = []
    for s in cfg.diffusion_scalars:
        out += [f"{s}_mean", f"{s}_std"]
    return out


def bundle_diffusion_scalars(mask_path, scalar_maps: dict[str, str],
                             cfg: TractConfig) -> np.ndarray:
    """Mean+std of each scalar within one bundle mask.

    Parameters
    ----------
    mask_path : path to a TractSeg bundle mask (.nii.gz).
    scalar_maps : {scalar_name -> path} for the subject's FA/MD/AD/RD maps
        (in the same space/grid as the bundle masks).
    """
    L = diffusion_length(cfg)
    try:
        mask_img = nib.load(str(mask_path))
        mask = np.asarray(mask_img.dataobj) > cfg.iso_level
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"bundle mask load failed: {exc!r}")
        return np.full(L, np.nan)
    if not mask.any():
        return np.full(L, np.nan)

    feats: list[float] = []
    for s in cfg.diffusion_scalars:
        path = scalar_maps.get(s)
        if path is None:
            feats += [np.nan, np.nan]
            continue
        try:
            vol = np.asarray(nib.load(str(path)).dataobj, dtype=float)
        except Exception:
            feats += [np.nan, np.nan]
            continue
        if vol.shape != mask.shape:
            warnings.warn(f"{s}: shape {vol.shape} != mask {mask.shape}; skipping")
            feats += [np.nan, np.nan]
            continue
        vals = vol[mask]
        vals = vals[np.isfinite(vals) & (vals != 0)]
        if vals.size == 0:
            feats += [np.nan, np.nan]
        else:
            feats += [float(vals.mean()), float(vals.std())]
    return np.asarray(feats, float)
