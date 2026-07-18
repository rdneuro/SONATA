# sonata/fs_schaefer.py
"""FreeSurfer 8.2.0 → Schaefer-200: surface labels, volumetric parcellation,
per-ROI sub-meshes, and per-ROI morphometry.

Why a dedicated step
--------------------
The Schaefer-200 atlas ships as *fsaverage*-space ``.annot`` files. To get
subject-native vertex labels under FreeSurfer 8.2.0 we resample with
``mri_surf2surf``. For the bundle→ROI mapping (``tract_mapping.py``) we also
need the parcellation in the subject's *volume* (T1/diffusion) space, which we
build with ``mri_aparc2aseg``.

Global ROI ids
--------------
We assign a single 1..200 labeling: left-hemisphere parcels → 1..100,
right-hemisphere parcels → 101..200; medial wall / unknown → 0 (ignored).
This matches the row/column order produced by standard Schaefer-200 connectome
pipelines (lh first, then rh), so the SIFT2 and FC matrices line up.

All FreeSurfer calls are checkpointed: if the output already exists they are
skipped, so the pipeline is resumable.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import spectralbrain as sb

from .config import SonataConfig
from .utils import get_logger

log = get_logger("sonata.fs")

# aparc2aseg volumetric convention: lh cortical voxels = 1000 + parcel_index,
# rh = 2000 + parcel_index, where parcel_index is the 1-based index within the
# hemisphere's color table (1..100 for Schaefer-200 per hemi).
_LH_VOL_OFFSET = 1000
_RH_VOL_OFFSET = 2000


# ──────────────────────────────────────────────────────────────────────────────
# Per-subject FreeSurfer dir resolution (from the manifest) + fsaverage linking
# ──────────────────────────────────────────────────────────────────────────────
# The Schaefer annot is in fsaverage space and is resampled to each subject with
# mri_surf2surf. That tool needs SUBJECTS_DIR to contain BOTH the target subject
# AND fsaverage. Our subjects live per-cohort on different disks (ds000221 vs
# covid), recorded per row in the manifest column `fs_subject_dir`. So we set
# SUBJECTS_DIR per subject = parent(fs_subject_dir) and ensure an fsaverage
# symlink exists there, pointing at the canonical fsaverage under
# cfg.paths.subjects_dir. This fixes "could not read .../fsaverage/surf/
# lh.sphere.reg" and the all-subjects-fail SUBJECTS_DIR mismatch.

_MANIFEST_CACHE: dict[str, dict] = {}


def _manifest_lookup(cfg: SonataConfig) -> dict:
    """Map subject_id -> fs_subject_dir from the manifest (cached)."""
    key = str(cfg.paths.manifest_csv)
    if key not in _MANIFEST_CACHE:
        df = pd.read_csv(cfg.paths.manifest_csv, dtype={"subject_id": str})
        if "fs_subject_dir" not in df.columns:
            raise KeyError(
                f"manifest {cfg.paths.manifest_csv} has no 'fs_subject_dir' column "
                f"(found: {list(df.columns)}). Rebuild it with build_manifest.py."
            )
        _MANIFEST_CACHE[key] = dict(zip(df["subject_id"].astype(str),
                                        df["fs_subject_dir"].astype(str)))
    return _MANIFEST_CACHE[key]


def _subject_fs_dir(subject_id: str, cfg: SonataConfig) -> Path:
    """The subject's own FreeSurfer recon dir (.../<SUBJECTS_DIR>/<subject_id>)."""
    fs_dir = _manifest_lookup(cfg).get(str(subject_id))
    if fs_dir is None:
        raise KeyError(f"{subject_id} not in manifest {cfg.paths.manifest_csv}")
    return Path(fs_dir)


def _ensure_fsaverage(subjects_dir: Path, cfg: SonataConfig) -> None:
    """Guarantee a usable 'fsaverage' inside *subjects_dir* (idempotent symlink).

    Points at the canonical fsaverage under cfg.paths.subjects_dir. If a broken
    or incomplete fsaverage is already there (missing lh.sphere.reg), replace it.
    """
    canonical = Path(cfg.paths.subjects_dir) / "fsaverage"
    target = Path(subjects_dir) / "fsaverage"
    needed = target / "surf" / "lh.sphere.reg"
    if needed.exists():
        return  # already good (real dir or valid link)
    # remove a broken link / incomplete dir if present
    try:
        if target.is_symlink() or target.exists():
            if target.is_symlink() or target.is_file():
                target.unlink()
            # if it's a real (incomplete) directory we leave it but warn
            elif target.is_dir():
                log.warning("fsaverage exists but is incomplete in %s "
                            "(no surf/lh.sphere.reg); leaving as-is", subjects_dir)
                return
    except OSError:
        pass
    if not (canonical / "surf" / "lh.sphere.reg").exists():
        raise FileNotFoundError(
            f"Canonical fsaverage incomplete/missing at {canonical} "
            f"(no surf/lh.sphere.reg). Set cfg.paths.subjects_dir to a FreeSurfer "
            f"subjects dir that contains a full fsaverage."
        )
    try:
        target.symlink_to(canonical, target_is_directory=True)
        log.info("linked fsaverage -> %s  (in %s)", canonical, subjects_dir)
    except OSError as exc:
        raise OSError(f"could not create fsaverage symlink in {subjects_dir}: {exc}") from exc


@dataclass
class SubjectParcellation:
    subject_id: str
    # Per-hemi surface geometry with GLOBAL labels (0..200).
    surf: dict[str, dict]                 # 'lh'/'rh' -> {vertices, faces, labels}
    submeshes: dict[int, tuple]           # global_roi_id -> (vertices, faces)
    morphometry: pd.DataFrame             # index global_roi_id; cols thickness/area/volume
    vol_parc_path: Path | None            # volumetric Schaefer parcellation (.mgz)
    vol_code_to_roi: dict[int, int]       # aparc2aseg code -> global roi id


# ──────────────────────────────────────────────────────────────────────────────
def _run_fs(cmd: list[str], cfg: SonataConfig,
            subjects_dir: Path | None = None) -> None:
    """Run a FreeSurfer command with SUBJECTS_DIR set to *subjects_dir*.

    If *subjects_dir* is None, falls back to cfg.paths.subjects_dir (legacy).
    On failure the FreeSurfer stderr/stdout is surfaced in the exception and the
    log — no more silent exit codes.
    """
    env = os.environ.copy()
    env["FREESURFER_HOME"] = str(cfg.paths.freesurfer_home)
    env["SUBJECTS_DIR"] = str(subjects_dir if subjects_dir is not None
                              else cfg.paths.subjects_dir)
    setup = cfg.paths.freesurfer_home / "SetUpFreeSurfer.sh"
    shell_cmd = " ".join(str(c) for c in cmd)
    if setup.exists():
        # source the setup but DON'T discard stderr of the actual command
        shell_cmd = f'source "{setup}" >/dev/null 2>/dev/null; {shell_cmd}'
    log.info("FS │ SUBJECTS_DIR=%s │ %s", env["SUBJECTS_DIR"],
             " ".join(str(c) for c in cmd))
    proc = subprocess.run(["bash", "-lc", shell_cmd], env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = "\n".join(tail[-15:]) if tail else "(no stderr captured)"
        log.error("FS FAILED (exit %d): %s\n%s",
                  proc.returncode, " ".join(str(c) for c in cmd), tail)
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)


def convert_aparc_to_schaefer(subject_id: str, cfg: SonataConfig,
                              build_volume: bool = True) -> Path | None:
    """Resample fsaverage Schaefer-200 to the subject and (optionally) volume.

    Produces ``{subj}/label/?h.Schaefer200.annot`` and, if *build_volume*,
    ``{subj}/mri/Schaefer200+aseg.mgz``. Returns the volumetric path (or None).
    """
    sdir = _subject_fs_dir(subject_id, cfg)         # the subject's own recon dir
    subjects_dir = sdir.parent                        # SUBJECTS_DIR for this cohort
    _ensure_fsaverage(subjects_dir, cfg)              # fsaverage must live alongside
    annot_src = {"lh": cfg.paths.schaefer_annot_lh, "rh": cfg.paths.schaefer_annot_rh}
    for hemi in ("lh", "rh"):
        out_annot = sdir / "label" / f"{hemi}.Schaefer200.annot"
        if out_annot.exists():
            continue
        _run_fs(["mri_surf2surf", "--srcsubject", "fsaverage",
                 "--trgsubject", subject_id, "--hemi", hemi,
                 "--sval-annot", str(annot_src[hemi]),
                 "--tval", str(out_annot)], cfg, subjects_dir=subjects_dir)

    vol_path = sdir / "mri" / "Schaefer200+aseg.mgz"
    if build_volume and not vol_path.exists():
        _run_fs(["mri_aparc2aseg", "--s", subject_id,
                 "--annot", "Schaefer200", "--o", str(vol_path)], cfg,
                subjects_dir=subjects_dir)
    return vol_path if build_volume else None


# ──────────────────────────────────────────────────────────────────────────────
def _hemi_global_labels(labels: np.ndarray, names: list[str], hemi: str
                        ) -> tuple[np.ndarray, dict[int, int]]:
    """Map per-hemi annot indices (0..100, 0=medial wall) to global 0..200.

    Returns the global per-vertex label array and a {hemi_index -> global_id}
    map (used to translate the volumetric aparc2aseg codes later).
    """
    offset = 0 if hemi == "lh" else 100
    glob = np.zeros_like(labels)
    idx_to_global: dict[int, int] = {}
    # Index 0 in a Schaefer '..._order.annot' is the medial wall / Unknown.
    n_parcels = len(names) - 1
    for hemi_idx in range(1, n_parcels + 1):
        gid = offset + hemi_idx
        glob[labels == hemi_idx] = gid
        idx_to_global[hemi_idx] = gid
    return glob, idx_to_global


def load_subject_parcellation(subject_id: str, cfg: SonataConfig,
                              build_volume: bool = True) -> SubjectParcellation:
    """Full per-subject parcellation product: surfaces, sub-meshes, morphometry."""
    vol_path = convert_aparc_to_schaefer(subject_id, cfg, build_volume=build_volume)
    sdir = _subject_fs_dir(subject_id, cfg)

    surf: dict[str, dict] = {}
    submeshes: dict[int, tuple] = {}
    vol_code_to_roi: dict[int, int] = {}
    morph_rows: list[dict] = []

    for hemi in ("lh", "rh"):
        verts, faces = sb.load_freesurfer_surface(sdir / "surf" / f"{hemi}.white")
        labels_idx, _ctab, names = sb.load_freesurfer_annot(
            sdir / "label" / f"{hemi}.Schaefer200.annot")
        thickness = sb.load_freesurfer_morph(sdir / "surf" / f"{hemi}.thickness")

        glob_labels, idx_to_global = _hemi_global_labels(labels_idx, names, hemi)
        surf[hemi] = {"vertices": verts, "faces": faces, "labels": glob_labels}

        # Volumetric code map (aparc2aseg convention).
        vol_off = _LH_VOL_OFFSET if hemi == "lh" else _RH_VOL_OFFSET
        for hidx, gid in idx_to_global.items():
            vol_code_to_roi[vol_off + hidx] = gid

        # Per-ROI sub-meshes (for spectral descriptors) via SpectralBrain.
        parcels = sb.apply_parcellation(verts, faces, glob_labels, ignore_labels=[0])
        submeshes.update(parcels)

        # Per-vertex area for a GM-volume approximation (area × thickness).
        vareas = _vertex_areas(verts, faces)
        for gid in idx_to_global.values():
            mask = glob_labels == gid
            if not mask.any():
                morph_rows.append({"roi": gid, "thickness": np.nan,
                                   "area": np.nan, "volume": np.nan})
                continue
            th = float(np.nanmean(thickness[mask]))
            area = float(vareas[mask].sum())
            morph_rows.append({"roi": gid, "thickness": th, "area": area,
                               "volume": area * th})

    morphometry = (pd.DataFrame(morph_rows).set_index("roi").sort_index())
    return SubjectParcellation(subject_id, surf, submeshes, morphometry,
                               vol_path, vol_code_to_roi)


def _vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Barycentric per-vertex area (one third of incident triangle areas)."""
    v = vertices
    tri = v[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    face_area = 0.5 * np.linalg.norm(cross, axis=1)
    va = np.zeros(len(v))
    for k in range(3):
        np.add.at(va, faces[:, k], face_area / 3.0)
    return va
