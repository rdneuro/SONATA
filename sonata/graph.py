# sonata/graph.py
"""Assemble one geometry-aware graph per subject and convert to a PyG ``Data``.

Two layers, deliberately separated so the heavy feature extraction runs on any
machine (no torch needed) and only the final tensor packing touches PyG:

1. :func:`build_subject_features` — pure NumPy, cached to ``cache/{sid}.npz``.
   Resumable: a present cache file is reused unless ``force=True``.
2. :func:`to_pyg_data` — packs (already imputed/scaled) arrays into a PyG graph.

Node target note: only FC *edges* are supervised. Node functional strength is
recovered downstream as the sum of predicted incident edges, so it is never an
independent label (avoids the node/edge target redundancy from the review).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

import spectralbrain as sb

from .config import SonataConfig
from .diffusion_features import bundle_diffusion_scalars, diffusion_length
from .fs_schaefer import load_subject_parcellation
from .parallel import parallel_map
from .spectral_features import descriptor_length, surface_descriptor
from .roi_cache import ROIDescriptorCache
from .tract_mapping import (
    aggregate_to_edges, compute_tract_endpoints, load_fc_on_edges,
    load_sift2_edges,
)
from .utils import get_logger

log = get_logger("sonata.graph")
N_PARCELS = 200


def _as_vf(obj):
    """Normalize SB bundle/mesh return into (vertices, faces)."""
    if hasattr(obj, "vertices") and hasattr(obj, "faces"):
        return np.asarray(obj.vertices, float), np.asarray(obj.faces)
    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        return np.asarray(obj[0], float), np.asarray(obj[1])
    raise TypeError(f"cannot interpret bundle mesh of type {type(obj)}")


def _voxel_volume(path) -> float:
    img = nib.load(str(path))
    return float(abs(np.linalg.det(img.affine[:3, :3])))


def _scalar_maps_for(row: pd.Series, cfg: SonataConfig) -> dict[str, str]:
    maps: dict[str, str] = {}
    for s in cfg.tract.diffusion_scalars:
        col = f"{s.lower()}_path"
        if col in row and isinstance(row[col], str) and pd.notna(row[col]):
            maps[s] = row[col]
    return maps


# ──────────────────────────────────────────────────────────────────────────────
def build_subject_features(row: pd.Series, cfg: SonataConfig,
                           force: bool = False) -> dict:
    """Extract all raw (un-scaled, NaN-tolerant) features for one subject."""
    sid = str(row["subject_id"])
    cache = cfg.paths.output_dir / "cache" / f"{sid}.npz"
    if cache.exists() and not force:
        d = dict(np.load(cache, allow_pickle=True))
        d["meta"] = d["meta"].item()
        return d

    Dspec = descriptor_length(cfg.spectral)
    Ddiff = diffusion_length(cfg.tract)

    # ── Parcellation → node morphometry + node spectral ──
    parc = load_subject_parcellation(sid, cfg, build_volume=True)
    morph = parc.morphometry.reindex(range(1, N_PARCELS + 1))
    node_morph = morph[["thickness", "area", "volume"]].to_numpy(float)   # (200,3)

    # Per-ROI spectral descriptors via the two-level (memory+disk) cache, so a
    # ROI computed once is reused across every tract that touches it AND across
    # runs. roi_id is the global 1..200 id; we store under (subject_id, roi_id).
    roi_cache = ROIDescriptorCache(cfg.paths.output_dir, sid, cfg.spectral)
    node_spectral = np.full((N_PARCELS, Dspec), np.nan)
    for gid, (v, f) in parc.submeshes.items():
        if 1 <= gid <= N_PARCELS:
            node_spectral[gid - 1] = roi_cache.get_or_compute(gid, v, f)

    # ── Bundles: spectral, diffusion, volume (single ordered pass) ──
    bundle_paths = sb.discover_tractseg_bundles(
        row["tractseg_dir"], bundles=cfg.tract.bundles,
        subdir=cfg.tract.tractseg_subdir)
    if isinstance(bundle_paths, (list, tuple)):
        bundle_paths = {Path(p).stem: str(p) for p in bundle_paths}
    bundle_paths = {k: str(v) for k, v in bundle_paths.items()}
    names = list(bundle_paths.keys())
    B = len(names)

    scalar_maps = _scalar_maps_for(row, cfg)
    bdesc = np.full((B, Dspec), np.nan)
    bdiff = np.full((B, Ddiff), np.nan)
    bvol = np.full((B, 1), np.nan)
    for b, name in enumerate(names):
        path = bundle_paths[name]
        try:
            mesh = sb.load_tractseg_bundle(path, output="mesh", level=cfg.tract.iso_level)
            v, f = _as_vf(mesh)
            bdesc[b] = surface_descriptor(v, f, cfg.spectral)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"bundle {name}: spectral failed ({exc!r})")
        bdiff[b] = bundle_diffusion_scalars(path, scalar_maps, cfg.tract)
        try:
            mimg = nib.load(path)
            n_vox = int((np.asarray(mimg.dataobj) > cfg.tract.iso_level).sum())
            bvol[b, 0] = n_vox * _voxel_volume(path)
        except Exception:
            pass

    # ── Structural edges + functional targets ──
    edges, sc_w = load_sift2_edges(row["sift2_csv"], cfg.tract)
    fc = load_fc_on_edges(row, edges, cfg)

    # ── Steps [1]→[3]: resolve each tract's two endpoints and the ROI-pairs it
    #    connects (endpoint-to-endpoint, not mid-trajectory overlap), then
    #    aggregate tract descriptors onto the structural edge set (step [4]). ──
    endpoints = compute_tract_endpoints(parc.vol_parc_path, parc.vol_code_to_roi,
                                        bundle_paths, cfg)
    edge_spectral, cov_spec = aggregate_to_edges(edges, bdesc, endpoints)
    edge_diffusion, cov_diff = aggregate_to_edges(edges, bdiff, endpoints)
    edge_volume, _ = aggregate_to_edges(edges, bvol, endpoints)

    meta = {"subject_id": sid, "group": str(row.get("group", "NA")),
            "protocol": str(row.get("protocol", "P1")),
            "age": float(row.get("age", np.nan)) if pd.notna(row.get("age", np.nan)) else np.nan,
            "sex": str(row.get("sex", "U")),
            "n_edges": int(len(edges)), "n_bundles": B,
            "coverage_spectral": float(cov_spec.mean()),
            "coverage_diffusion": float(cov_diff.mean())}

    out = dict(
        node_morph=node_morph, node_spectral=node_spectral,
        edges=edges.astype(np.int64), sc_weight=sc_w.astype(np.float32),
        edge_spectral=edge_spectral.astype(np.float32),
        edge_spectral_cover=cov_spec.astype(np.float32),
        edge_diffusion=edge_diffusion.astype(np.float32),
        edge_diffusion_cover=cov_diff.astype(np.float32),
        edge_volume=edge_volume.astype(np.float32),
        fc=fc.astype(np.float32), meta=meta)
    np.savez_compressed(cache, **out)
    _rc = roi_cache.stats()
    log.info("cached %s │ edges=%d bundles=%d spec-cov=%.3f diff-cov=%.3f │ "
             "roi-cache: computed=%d mem_hits=%d disk_hits=%d",
             sid, len(edges), B, cov_spec.mean(), cov_diff.mean(),
             _rc["computed"], _rc["mem_hits"], _rc["disk_hits"])
    return out


# ──────────────────────────────────────────────────────────────────────────────
def build_node_matrix(feat: dict, cfg: SonataConfig) -> np.ndarray:
    """Node feature matrix per ``model.node_feature_mode`` (NaNs tolerated)."""
    if cfg.model.node_feature_mode == "identity":
        return np.eye(N_PARCELS, dtype=np.float32)
    return np.concatenate([feat["node_morph"], feat["node_spectral"]], axis=1).astype(np.float32)


def build_edge_matrix(feat: dict, cfg: SonataConfig) -> np.ndarray:
    """Edge feature matrix per ``model.edge_feature_mode``."""
    sc = feat["sc_weight"][:, None]
    vol = feat["edge_volume"]
    mode = cfg.model.edge_feature_mode
    if mode == "sift2":
        return sc.astype(np.float32)
    if mode == "diffusion":
        return np.concatenate(
            [sc, vol, feat["edge_diffusion"], feat["edge_diffusion_cover"][:, None]],
            axis=1).astype(np.float32)
    # default: spectral (Aim 1)
    return np.concatenate(
        [sc, vol, feat["edge_spectral"], feat["edge_spectral_cover"][:, None]],
        axis=1).astype(np.float32)


def to_pyg_data(feat: dict, x: np.ndarray, edge_attr: np.ndarray, cfg: SonataConfig):
    """Pack finite (imputed/scaled) arrays into a bidirectional PyG ``Data``.

    The undirected edge set is duplicated to both directions for message
    passing; ``edge_id`` maps each directed edge back to its undirected index so
    the readout can average the two orientations (symmetry).
    """
    import torch
    from torch_geometric.data import Data

    edges = feat["edges"]
    E = len(edges)
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_attr2 = torch.tensor(np.concatenate([edge_attr, edge_attr], 0), dtype=torch.float)
    edge_id = torch.tensor(np.concatenate([np.arange(E), np.arange(E)]), dtype=torch.long)

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index, edge_attr=edge_attr2,
        y=torch.tensor(feat["fc"], dtype=torch.float),       # (E,) undirected targets
    )
    data.edge_id = edge_id
    data.num_undirected = E
    data.subject_id = feat["meta"]["subject_id"]
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Subject-level parallel feature build (integrated; replaces parallel_features.py)
# ──────────────────────────────────────────────────────────────────────────────
def _build_one_subject(args: tuple[dict, SonataConfig, bool]) -> dict:
    """Top-level worker (picklable) building one subject's cached features.

    Rebuilds a :class:`pandas.Series` from the row dict so the heavy call is
    identical to the serial path, and returns a small, picklable status record.
    """
    import time

    row_dict, cfg, force = args
    sid = str(row_dict.get("subject_id", "?"))
    t0 = time.time()
    try:
        cache = cfg.paths.output_dir / "cache" / f"{sid}.npz"
        if cache.exists() and not force:
            return {"sid": sid, "status": "cached", "dt": time.time() - t0}
        feat = build_subject_features(pd.Series(row_dict), cfg, force=force)
        meta = feat.get("meta", {}) if isinstance(feat, dict) else {}
        return {"sid": sid, "status": "ok", "dt": time.time() - t0,
                "n_edges": int(meta.get("n_edges", -1))}
    except Exception as exc:  # noqa: BLE001 - status is reported, not raised
        return {"sid": sid, "status": "FAILED", "dt": time.time() - t0,
                "err": f"{exc!r}"}


def build_all_subject_features(
    manifest: pd.DataFrame,
    cfg: SonataConfig,
    *,
    n_threads: int = 1,
    force: bool = False,
    progress: bool = True,
) -> list[dict]:
    """Build (and cache) features for every subject in ``manifest``.

    Embarrassingly parallel across subjects — each writes its own
    ``cache/<sid>.npz`` with no shared state — so this integrates the library's
    ``n_threads`` convention directly, superseding the standalone
    ``parallel_features.py`` driver. Inner BLAS threads are pinned to 1 per worker
    so ``N`` subject workers do not oversubscribe the cores.

    Parameters
    ----------
    manifest
        Subject table (one row per subject; see :func:`sonata.utils.load_manifest`).
    cfg
        Full SONATA configuration.
    n_threads
        ``1`` serial, ``>=2`` joblib workers, ``-1`` all usable cores (capped 22).
    force
        Recompute even if a subject's cache exists.
    progress
        Show a progress bar.

    Returns
    -------
    list of dict
        One status record per subject (``ok`` / ``cached`` / ``FAILED``).
    """
    cfg.ensure()
    items = [(row.to_dict(), cfg, force) for _, row in manifest.iterrows()]
    results = parallel_map(_build_one_subject, items, n_threads=n_threads,
                           progress=progress, description="subject features")
    return results


def load_all_cached_features(cfg: SonataConfig, manifest: "pd.DataFrame | None" = None) -> list[dict]:
    """Load every subject's feature dict, reading the per-subject cache.

    Calls :func:`build_subject_features` per row; because that reuses an existing
    ``cache/<sid>.npz`` when present, this is a fast reload after the (parallel)
    feature build. Subjects whose feature build fails are skipped with a warning.
    """
    from .utils import get_logger, load_manifest

    log = get_logger("sonata.graph")
    if manifest is None:
        manifest = load_manifest(cfg.paths.manifest_csv)
    feats: list[dict] = []
    for _, row in manifest.iterrows():
        try:
            feats.append(build_subject_features(row, cfg))
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s: %r", row.get("subject_id", "?"), exc)
    return feats
