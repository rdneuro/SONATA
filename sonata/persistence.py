# sonata/persistence.py
"""Persist spectra and descriptors per tract and its associated ROIs.

For every subject and every tract bundle we save:
- ``spectra/{sid}/tracts/{bundle}.npz``  → {eigenvalues, eigenvectors}
- ``spectra/{sid}/rois/{roi}.npz``        → {eigenvalues, eigenvectors} (associated ROIs)
- ``spectra/{sid}/pickles/{bundle}.pkl``  → {'tract': {...descriptors+eigvals},
                                             'rois': {roi_id: {...}}}

"Associated ROIs" of a tract = the ROIs in its volumetric footprint (the same
soft assignment used for edge features). Everything is checkpointed: existing
outputs are skipped so the persistence stage is resumable.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

import spectralbrain as sb

from .config import SonataConfig
from .fs_schaefer import load_subject_parcellation
from .graph import _as_vf, _scalar_maps_for
from .spectral_features import descriptor_names
from .tract_mapping import compute_tract_endpoints
from .utils import get_logger, track

log = get_logger("sonata.persist")


def compute_full_spectrum(vertices, faces, cfg: SonataConfig) -> dict:
    """Eigenpairs + all configured descriptors for one surface."""
    out: dict = {"eigenvalues": None, "eigenvectors": None, "descriptors": {}}
    if vertices is None or len(vertices) < cfg.spectral.min_vertices:
        return out
    try:
        mesh = sb.BrainMesh(np.asarray(vertices, float), np.asarray(faces))
        if cfg.spectral.taubin_iter > 0:
            mesh = mesh.taubin_smooth(n_iterations=cfg.spectral.taubin_iter,
                                      lambda_=cfg.spectral.taubin_lambda,
                                      mu=cfg.spectral.taubin_mu)
        k = int(min(cfg.spectral.n_eigen, max(4, mesh.n_vertices - 2)))
        decomp = mesh.decompose(k=k, laplacian_method=cfg.spectral.laplacian)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"spectrum failed: {exc!r}")
        return out
    out["eigenvalues"] = np.asarray(getattr(decomp, "eigenvalues", None))
    out["eigenvectors"] = np.asarray(getattr(decomp, "eigenvectors", None))
    desc = out["descriptors"]
    try:
        if "shapedna" in cfg.spectral.use_descriptors:
            desc["shapedna"] = np.asarray(sb.compute_shapedna(
                decomp, normalize=cfg.spectral.shapedna_normalize, skip_zero=True))
        if "hks" in cfg.spectral.use_descriptors:
            desc["hks"] = np.asarray(sb.compute_hks(decomp, n_times=cfg.spectral.hks_n_times))
        if "wks" in cfg.spectral.use_descriptors:
            desc["wks"] = np.asarray(sb.compute_wks(decomp, n_energies=cfg.spectral.wks_n_energies))
        if "gps" in cfg.spectral.use_descriptors:
            desc["gps"] = np.asarray(sb.compute_gps(decomp, skip_zero=True))
        if "bks" in cfg.spectral.use_descriptors:
            desc["bks"] = np.asarray(sb.compute_bks(decomp))
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"descriptor pack failed: {exc!r}")
    return out


def persist_subject_spectra(row: pd.Series, cfg: SonataConfig,
                            force: bool = False) -> None:
    """Compute & save eigenpairs + descriptor pickles for all tracts of a subject."""
    sid = str(row["subject_id"])
    base = cfg.paths.output_dir / "spectra" / sid
    done_marker = base / ".complete"
    if done_marker.exists() and not force:
        return
    (base / "tracts").mkdir(parents=True, exist_ok=True)
    (base / "rois").mkdir(parents=True, exist_ok=True)
    (base / "pickles").mkdir(parents=True, exist_ok=True)

    parc = load_subject_parcellation(sid, cfg, build_volume=True)
    bundle_paths = sb.discover_tractseg_bundles(
        row["tractseg_dir"], bundles=cfg.tract.bundles, subdir=cfg.tract.tractseg_subdir)
    if isinstance(bundle_paths, (list, tuple)):
        bundle_paths = {Path(p).stem: str(p) for p in bundle_paths}
    bundle_paths = {k: str(v) for k, v in bundle_paths.items()}
    foot = compute_tract_endpoints(parc.vol_parc_path, parc.vol_code_to_roi,
                                   bundle_paths, cfg)

    # Associated ROIs per bundle index (inverse of the edge footprint map).
    assoc: dict[int, set[int]] = {b: set() for b in range(len(foot.bundle_names))}
    for (i, j), wmap in foot.edge_weight.items():
        for b in wmap:
            assoc[b].update((i, j))

    # Cache ROI spectra once (shared across tracts).
    roi_cache: dict[int, dict] = {}

    def roi_spectrum(node_idx: int) -> dict:
        gid = node_idx + 1
        if node_idx in roi_cache:
            return roi_cache[node_idx]
        vf = parc.submeshes.get(gid)
        spec = compute_full_spectrum(*vf, cfg) if vf else {"eigenvalues": None,
                                                           "eigenvectors": None, "descriptors": {}}
        np.savez_compressed(base / "rois" / f"roi_{gid:03d}.npz",
                            eigenvalues=spec["eigenvalues"] if spec["eigenvalues"] is not None else np.array([]),
                            eigenvectors=spec["eigenvectors"] if spec["eigenvectors"] is not None else np.array([]))
        roi_cache[node_idx] = spec
        return spec

    for b, name in track(list(enumerate(foot.bundle_names)),
                         f"  persist spectra {sid}", total=len(foot.bundle_names)):
        tpkl = base / "pickles" / f"{name}.pkl"
        if tpkl.exists() and not force:
            continue
        try:
            mesh = sb.load_tractseg_bundle(bundle_paths[name], output="mesh",
                                           level=cfg.tract.iso_level)
            v, f = _as_vf(mesh)
            tract_spec = compute_full_spectrum(v, f, cfg)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"{name}: tract spectrum failed ({exc!r})")
            tract_spec = {"eigenvalues": None, "eigenvectors": None, "descriptors": {}}

        np.savez_compressed(
            base / "tracts" / f"{name}.npz",
            eigenvalues=tract_spec["eigenvalues"] if tract_spec["eigenvalues"] is not None else np.array([]),
            eigenvectors=tract_spec["eigenvectors"] if tract_spec["eigenvectors"] is not None else np.array([]))

        roi_dicts = {}
        for node_idx in sorted(assoc.get(b, set())):
            spec = roi_spectrum(node_idx)
            roi_dicts[node_idx + 1] = {"eigenvalues": spec["eigenvalues"],
                                       "descriptors": spec["descriptors"]}
        with open(tpkl, "wb") as fh:
            pickle.dump({"subject_id": sid, "tract": name,
                         "tract_spectrum": {"eigenvalues": tract_spec["eigenvalues"],
                                            "descriptors": tract_spec["descriptors"]},
                         "associated_rois": roi_dicts,
                         "descriptor_names": descriptor_names(cfg.spectral)}, fh)
    done_marker.write_text("ok")
    log.info("persisted spectra for %s (%d tracts)", sid, len(foot.bundle_names))
