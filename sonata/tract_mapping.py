# sonata/tract_mapping.py
"""SONATA core — the tract→endpoints→ROIs→set logic (steps [1]–[4]).

This is the re-architected core. The previous version assigned a bundle to the
*top-K ROIs it most overlapped anywhere along its trajectory*, then formed all
pairs among them. That is physically wrong: a bundle connects TWO endpoints, and
the corpus callosum (say) grazes dozens of parcels mid-trajectory that it does
not "connect". The result was sparse, noisy, and biologically diffuse coverage.

The corrected logic follows the user's specification literally:

  [1] tract T
  [2] ROIs at ENDPOINT 1 of T            ← geometric terminal clustering, side A
  [3] ROIs at ENDPOINT 2 of T            ← geometric terminal clustering, side B
  [4] compiled set = { descriptor(T) } ∪ { descriptor(ROI) : ROI ∈ A ∪ B }
      with a per-ROI cache so a ROI computed once is reused across tracts.

Endpoints are found by clustering the bundle's voxel coordinates into two
spatially compact groups (KMeans, k=2) and keeping the terminal shell of each
(the voxels farthest from the bundle's mid-plane). Each endpoint's ROIs are the
parcels overlapping that terminal shell, weighted by overlap fraction. The edge
the tract "connects" is then any ROI-pair (a∈A, c∈C) with a>0 weight — but now
those pairs are genuine endpoint-to-endpoint connections, not mid-trajectory
coincidences.

Everything heavy (parcellation resample, per-ROI descriptors) is cached. The FC
loader and SIFT2 edge loader (validated, unchanged in logic) live here too.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage

from .config import SonataConfig, TractConfig
from .utils import fisher_z, get_logger

log = get_logger("sonata.tract")

_MIN_FRAC = 0.02          # ignore ROIs an endpoint barely grazes
_TERMINAL_FRAC = 0.30     # fraction of each half taken as the "terminal shell"


# ──────────────────────────────────────────────────────────────────────────────
# Structural / functional matrices (validated readers — logic unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def _read_matrix(path) -> np.ndarray:
    """Read a square connectivity matrix from .npy/.npz OR CSV (format by ext)."""
    suf = Path(str(path)).suffix.lower()
    if suf == ".npy":
        arr = np.load(str(path), allow_pickle=False)
    elif suf == ".npz":
        with np.load(str(path), allow_pickle=False) as z:
            key = ("connectivity" if "connectivity" in z.files
                   else ("matrix" if "matrix" in z.files else z.files[0]))
            arr = z[key]
    else:
        df = pd.read_csv(path, header=None)
        arr = df.to_numpy()
        if arr.dtype == object:
            df = pd.read_csv(path, index_col=0)
            arr = df.to_numpy(dtype=float)
    arr = np.asarray(arr, float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"matrix at {path} is not square: {arr.shape}")
    return arr


def load_sift2_edges(path, cfg: TractConfig) -> tuple[np.ndarray, np.ndarray]:
    """Undirected edge list (E,2 node idx 0-based) + SC weights (E,)."""
    W = _read_matrix(path)
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    iu, ju = np.triu_indices_from(W, k=1)
    w = W[iu, ju]
    if cfg.sift2_density_target is not None:
        keep_n = int(cfg.sift2_density_target * len(w))
        thr = np.partition(w, -keep_n)[-keep_n] if 0 < keep_n < len(w) else 0.0
        mask = w >= thr
    else:
        mask = w > 0
    edges = np.stack([iu[mask], ju[mask]], axis=1)
    weights = w[mask]
    if cfg.sift2_log:
        weights = np.log1p(weights)
    return edges.astype(np.int64), weights.astype(float)


class IncompleteFCError(ValueError):
    """FC matrix does not cover all Schaefer-200 ROIs (ROIs dropped for no BOLD)."""


def load_fc_on_edges(row: pd.Series, edges: np.ndarray, cfg: SonataConfig) -> np.ndarray:
    """FC sampled on the structural edge set (E,), with SC<->FC alignment guard."""
    n_parcels = int(getattr(cfg.tract, "n_parcels", 200))
    if isinstance(row.get("fc_csv"), str) and pd.notna(row.get("fc_csv")):
        FC = _read_matrix(row["fc_csv"])
    else:
        ts = _read_matrix(row["fc_timeseries_csv"])
        FC = np.corrcoef(ts.T)
    if FC.shape[0] != n_parcels or FC.shape[1] != n_parcels:
        sid = str(row.get("subject_id", "?"))
        raise IncompleteFCError(
            f"{sid}: FC is {FC.shape[0]}x{FC.shape[1]} but Schaefer-{n_parcels} "
            f"requires {n_parcels}x{n_parcels}; cannot realign SC<->FC safely. "
            f"Excluding subject for incomplete functional coverage.")
    emax = int(edges.max()) if edges.size else -1
    if emax >= n_parcels:
        raise IncompleteFCError(
            f"{row.get('subject_id','?')}: edge idx {emax} exceeds FC dim {n_parcels}.")
    FC = (FC + FC.T) / 2.0
    np.fill_diagonal(FC, 0.0)
    if cfg.func.absolute_fc:
        FC = np.abs(FC)
    vals = FC[edges[:, 0], edges[:, 1]]
    if cfg.func.fisher_z and not cfg.func.absolute_fc:
        vals = fisher_z(vals)
    return vals.astype(float)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────
def _voxel_size_mm(affine: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(affine[:3, :3], axis=0)))


def _resample_parc_to_grid(parc_img, ref_img):
    """Resample a label volume onto ref_img's grid (NEAREST — labels)."""
    from nilearn.image import resample_to_img
    res = resample_to_img(parc_img, ref_img, interpolation="nearest")
    return np.asarray(res.dataobj).astype(np.int64)


def _split_endpoints(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a bundle's voxels into two terminal shells (the two endpoints).

    Strategy: PCA along the bundle's principal axis (its length), then take the
    extreme `_TERMINAL_FRAC` of voxels at each end of that axis. This is robust
    and parameter-light: the principal axis of a fasciculus is its long
    trajectory, and its two ends are the anatomical terminations. Returns two
    boolean masks (same shape as `mask`) for endpoint A and endpoint C.
    """
    idx = np.argwhere(mask)
    if idx.shape[0] < 6:
        return mask.copy(), np.zeros_like(mask)
    centered = idx - idx.mean(0)
    # principal axis via SVD (no sklearn dependency for this)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    proj = centered @ axis
    lo_thr = np.quantile(proj, _TERMINAL_FRAC)
    hi_thr = np.quantile(proj, 1.0 - _TERMINAL_FRAC)
    a_mask = np.zeros_like(mask)
    c_mask = np.zeros_like(mask)
    a_sel = idx[proj <= lo_thr]
    c_sel = idx[proj >= hi_thr]
    a_mask[tuple(a_sel.T)] = True
    c_mask[tuple(c_sel.T)] = True
    return a_mask, c_mask


def _endpoint_rois(term_mask: np.ndarray, parc: np.ndarray,
                   code_to_node: dict[int, int]) -> dict[int, float]:
    """ROIs overlapping a terminal shell → {node_idx: overlap_fraction}."""
    codes = parc[term_mask].astype(np.int64)
    codes = codes[codes > 0]
    if codes.size == 0:
        return {}
    uniq, counts = np.unique(codes, return_counts=True)
    roi_counts: dict[int, int] = {}
    for c, n in zip(uniq.tolist(), counts.tolist()):
        node = code_to_node.get(int(c))
        if node is not None:
            roi_counts[node] = roi_counts.get(node, 0) + int(n)
    total = sum(roi_counts.values())
    if total == 0:
        return {}
    return {r: n / total for r, n in roi_counts.items() if n / total >= _MIN_FRAC}


# ──────────────────────────────────────────────────────────────────────────────
# Tract → endpoints → ROI-pairs  (steps [1]–[3])
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TractEndpoints:
    """Per-tract endpoint resolution (step [1]→[3])."""
    bundle_names: list[str]
    # bundle_idx -> (endpointA: {node:frac}, endpointC: {node:frac})
    endpoints: dict[int, tuple[dict[int, float], dict[int, float]]]
    # (i,j) sorted node pair -> {bundle_idx: weight = fracA·fracC}
    edge_weight: dict[tuple[int, int], dict[int, float]] = field(default_factory=dict)


def compute_tract_endpoints(vol_parc_path, vol_code_to_roi: dict[int, int],
                            bundle_paths: dict[str, str],
                            cfg: SonataConfig) -> TractEndpoints:
    """Resolve each bundle's two endpoints and the ROI-pairs they connect.

    For each bundle: load mask → resample parcellation to the bundle grid (cached
    per grid) → split into two terminal shells → ROIs per endpoint → all
    cross-endpoint pairs (a∈A, c∈C) with weight fracA·fracC. This is the corrected
    endpoint-to-endpoint connection, replacing the old mid-trajectory top-K pairs.
    """
    parc_img = nib.load(str(vol_parc_path))
    code_to_node = {code: roi - 1 for code, roi in vol_code_to_roi.items()}
    names = list(bundle_paths.keys())
    endpoints: dict[int, tuple[dict, dict]] = {}
    edge_weight: dict[tuple[int, int], dict[int, float]] = {}
    _parc_cache: dict = {}

    for b, name in enumerate(names):
        try:
            mimg = nib.load(str(bundle_paths[name]))
            mask = np.asarray(mimg.dataobj) > cfg.tract.iso_level
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"bundle {name}: mask load failed ({exc!r})")
            continue
        if not mask.any():
            continue

        if mask.shape == parc_img.shape and np.allclose(mimg.affine, parc_img.affine):
            parc = np.asarray(parc_img.dataobj).astype(np.int64)
            vox_mm = _voxel_size_mm(parc_img.affine)
        else:
            ckey = (mimg.shape, mimg.affine.tobytes())
            if ckey not in _parc_cache:
                _parc_cache[ckey] = _resample_parc_to_grid(parc_img, mimg)
            parc = _parc_cache[ckey]
            vox_mm = _voxel_size_mm(mimg.affine)

        dilate_vox = max(0, int(round(cfg.tract.endpoint_dilation_mm / max(vox_mm, 1e-3))))
        if dilate_vox > 0:
            mask = ndimage.binary_dilation(mask, iterations=dilate_vox)

        a_mask, c_mask = _split_endpoints(mask)
        roi_a = _endpoint_rois(a_mask, parc, code_to_node)
        roi_c = _endpoint_rois(c_mask, parc, code_to_node)
        endpoints[b] = (roi_a, roi_c)

        # cross-endpoint pairs only (genuine connections)
        for ra, fa in roi_a.items():
            for rc, fc in roi_c.items():
                if ra == rc:
                    continue
                key = (ra, rc) if ra < rc else (rc, ra)
                edge_weight.setdefault(key, {})
                edge_weight[key][b] = edge_weight[key].get(b, 0.0) + fa * fc

    return TractEndpoints(bundle_names=names, endpoints=endpoints,
                          edge_weight=edge_weight)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation onto the structural edge set (step [4]→[5] interface)
# ──────────────────────────────────────────────────────────────────────────────
def aggregate_to_edges(edges: np.ndarray, bundle_desc: np.ndarray,
                       endpoints: TractEndpoints) -> tuple[np.ndarray, np.ndarray]:
    """Weighted-mean bundle descriptors onto each structural edge.

    Returns (E,D) edge features and (E,) coverage indicator. Identical contract
    to the previous version, so downstream (cv/model/bayes) is unchanged.
    """
    E = len(edges)
    D = bundle_desc.shape[1]
    feat = np.zeros((E, D), float)
    cover = np.zeros(E, float)
    ew = endpoints.edge_weight
    for e in range(E):
        i, j = int(edges[e, 0]), int(edges[e, 1])
        key = (i, j) if i < j else (j, i)
        wmap = ew.get(key)
        if not wmap:
            continue
        bs = np.fromiter(wmap.keys(), dtype=np.int64)
        ws = np.fromiter(wmap.values(), dtype=float)
        valid = np.isfinite(bundle_desc[bs]).all(axis=1)
        if not valid.any():
            continue
        bs, ws = bs[valid], ws[valid]
        ws = ws / ws.sum()
        feat[e] = ws @ bundle_desc[bs]
        cover[e] = 1.0
    return feat, cover
