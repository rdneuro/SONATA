# sonata/config.py
"""SONATA — Structure-tO-fuNction via spectrAl Tract Attributes.

Central configuration. Everything the pipeline needs to locate data and set
hyperparameters lives here, so at run time you only edit *one* file (or pass an
edited :class:`SonataConfig` to the orchestrator).

Design notes
------------
- Paths are intentionally explicit. Point them at your disk at 05:00 and go.
- ``schaefer_annot`` are the *fsaverage-space* Schaefer-200 .annot files you
  already have; SONATA resamples them to each subject with ``mri_surf2surf``.
- The graph target is functional connectivity on the *support of structural
  connectivity* (edges present in the SIFT2 matrix). Node functional strength
  is NEVER supervised — it is recovered as a deterministic sum of predicted
  incident edges (see ``model.py``), which removes the node/edge target
  redundancy flagged in the design review.
- COHORT ROOTS are the SINGLE SOURCE OF TRUTH (see ``CohortPaths`` below).
  ``build_manifest.py`` imports them from here; nothing is duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Paths:
    """All filesystem locations. Edit these at run time."""

    # Root for everything SONATA writes (cache, checkpoints, figures, tables).
    output_dir: Path = Path("/media/rd/d0/neurocomp/data/output/sonata")

    # Per-cohort subject manifest (CSV) — PRODUCED by build_manifest.py.
    # Columns: subject_id, group, fs_subject_dir, tractseg_dir, sift2_csv,
    #          fc_csv, protocol, age, sex
    manifest_csv: Path = Path("/media/rd/d0/neurocomp/data/output/sonata/info/sonata_manifest.csv")

    # FreeSurfer SUBJECTS_DIR (must contain 'fsaverage'; used only for the
    # surf2surf resample of the Schaefer annot — per-subject FS dirs come from
    # the manifest's fs_subject_dir column, NOT from here).
    subjects_dir: Path = Path("/usr/local/freesurfer/8.2.0/subjects")

    # fsaverage-space Schaefer-200 annotations you already have on disk.
    schaefer_annot_lh: Path = Path("/media/rd/d0/neurocomp/info/annot/fsaverage/lh.Schaefer2018_200Parcels_7Networks_order.annot")
    schaefer_annot_rh: Path = Path("/media/rd/d0/neurocomp/info/annot/fsaverage/rh.Schaefer2018_200Parcels_7Networks_order.annot")

    # FreeSurfer binaries (only needed for the surf2surf / aparc2aseg step).
    freesurfer_home: Path = Path("/usr/local/freesurfer/8.2.0")

    def ensure(self) -> None:
        for sub in ("cache", "checkpoints", "figures", "tables", "graphs", "logs"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Cohort roots  —  SINGLE SOURCE OF TRUTH for where each cohort lives on disk
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CohortPaths:
    """Disk roots + authority CSV per cohort.

    `build_manifest.py` reads this and writes Paths.manifest_csv. To add a new
    cohort later (CAM-CAN, HCP1200), just add another key to ``cohorts`` with
    the same six fields — nothing else changes.

    Notes
    -----
    - ``{sub}`` is a LITERAL placeholder (kept as raw text here, NOT an f-string);
      it is filled in build_manifest.py via ``template.format(sub=subject_id)``.
    - ``authority`` is the CSV that DICTATES inclusion for that cohort
      (e.g. the n=121 intersection list for ds000221). Disk roots only build
      paths; the authority CSV decides who enters.
    - Beware the atlas-folder naming: SC uses ``schaefer200`` (no underscore),
      FC uses ``schaefer_200`` (with underscore). They are deliberately different.
    - FC points at the RAW correlation (``connectivity_correlation.npy``); the
      Fisher-z is applied once by the pipeline (FuncConfig.fisher_z=True).
    """

    cohorts: dict = field(default_factory=lambda: {
        # ── ds000221 — normative / training arm (n≈121, intersection-gated) ──
        "normative": {
            "protocol":  "ds000221",
            "authority": "/media/rd/d0/neurocomp/data/output/sonata/info/sonata_usable_cohort_n121.csv",
            "fs_dir":    "/media/rd/disk4/analysis/ds000221/structural/fs820/{sub}",
            "tractseg":  "/media/rd/disk4/analysis/ds000221/dmri/{sub}/tractseg",
            "dti":       "/media/rd/disk4/analysis/ds000221/dmri/{sub}/dti",
            "sift2":     "/media/rd/disk4/analysis/ds000221/dmri/{sub}/connectivity/schaefer200/connectivity_sift2.csv",
            "fc":        "/media/rd/disk4/analysis/ds000221/rsfmri/connectivity/schaefer_200/acompcor/{sub}/connectivity_correlation.npy",
        },
        # ── COVID-ICU — clinical / transfer arm (n≈24) ──
        "covid": {
            "protocol":  "covid_3T",
            "authority": "/media/rd/d0/neurocomp/data/output/sonata/info/covid_deidentified.csv",
            "fs_dir":    "/media/rd/disk4/analysis/covid/structural/fs820/subjects/{sub}",
            "tractseg":  "/media/rd/disk4/analysis/covid/dmri/v4/{sub}/tractseg",
            "dti":       "/media/rd/disk4/analysis/covid/dmri/v4/{sub}/dti",
            "sift2":     "/media/rd/disk4/analysis/covid/dmri/v4/{sub}/matrices/schaefer200/connectivity_sift2.csv",
            "fc":        "/media/rd/disk4/analysis/covid/rsfmri/fmri/v5/connectivity/schaefer_200/acompcor/{sub}/connectivity_correlation.npy",
        },
        # ── (future) add "camcan": {...}, "hcp": {...} here ──
    })


# ──────────────────────────────────────────────────────────────────────────────
# Spectral feature extraction
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SpectralConfig:
    """Laplace–Beltrami spectral descriptor settings (SpectralBrain)."""

    n_eigen: int = 80           # eigenpairs per surface (k in mesh.decompose)
    # Truncate every surface at the SAME eigen-index so the descriptor encodes
    # shape, not mesh size. A surface too small to support k=n_eigen yields an
    # all-NaN descriptor (flagged + imputed in CV). Set False only for ablation.
    fixed_k: bool = True
    laplacian: str = "robust"      # 'cotangent' | 'robust' (robust tolerates noisy meshes)
    hks_n_times: int = 16          # HKS time samples -> summarized across vertices
    wks_n_energies: int = 16       # WKS energy samples -> summarized across vertices
    shapedna_normalize: str = "area"   # scale-invariant eigenvalue normalization
    use_descriptors: tuple[str, ...] = ("shapedna", "hks", "wks", "gps", "bks")

    # Mesh QC / conditioning before decomposition (LB spectra are sensitive to
    # genus/handles from segmentation noise — Reuter 2006; design review §6).
    taubin_iter: int = 12
    taubin_lambda: float = 0.5
    taubin_mu: float = -0.53
    max_genus: int = 2             # reject/repair surfaces above this genus
    min_vertices: int = 60         # surfaces smaller than this -> NaN features


# ──────────────────────────────────────────────────────────────────────────────
# Tract / connectome handling
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TractConfig:
    """TractSeg bundles, SIFT2 edge weights, bundle→ROI mapping."""

    n_parcels: int = 200
    bundles: tuple[str, ...] | None = None   # None = all TractSeg bundles found
    tractseg_subdir: str | None = "bundle_segmentations"
    iso_level: float = 0.5                    # isosurface threshold for bundle mesh

    # Edge support: keep undirected SC edges with weight above this quantile.
    sift2_density_target: float | None = 0.15  # keep top-10% strongest edges; None=all>0
    sift2_log: bool = True                      # log1p the SIFT2 weights
    endpoint_dilation_mm: float = 3.0           # tolerance when assigning bundle termini to parcels

    # Diffusion scalars sampled per bundle for the Aim-2 baseline.
    diffusion_scalars: tuple[str, ...] = ("FA", "MD", "AD", "RD")


# ──────────────────────────────────────────────────────────────────────────────
# Functional connectivity target
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FuncConfig:
    fisher_z: bool = True          # Fisher-z transform FC before modeling
    absolute_fc: bool = False      # predict |FC| (avoids sign issues post-GSR)
    # Aim-1 residual option: if True, the target becomes FC minus a communication
    # baseline prediction (the SC–FC 'decoupling residual'); if False, raw FC.
    predict_residual: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Model (SONATA edge-conditioned MPNN)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    hidden: int = 64
    n_layers: int = 3
    edge_hidden: int = 64
    dropout: float = 0.15
    # Ablation switch: 'identity' reproduces a Chen/Cui-style one-hot-node baseline
    # (node morphometry/spectra removed); 'full' uses morphometric+spectral nodes.
    node_feature_mode: str = "full"     # 'full' | 'identity'
    # Edge feature set: 'spectral' (SIFT2+volume+tract LB spectra) is Aim-1;
    # 'diffusion' (SIFT2+volume+AD/RD/MD) is the Aim-2 non-inferiority comparator;
    # 'sift2' is the minimal strength-only baseline.
    edge_feature_mode: str = "spectral"  # 'spectral' | 'diffusion' | 'sift2'


# ──────────────────────────────────────────────────────────────────────────────
# Training / cross-validation
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 60             # early-stopping patience (epochs)
    batch_subjects: int = 4        # subjects per mini-batch (one graph each)
    cv: str = "grouped_kfold"     # 'grouped_kfold' (main paper) | 'loso' (covid-only)
    cv_group_col: str = "subject_id"   # leakage-safe: whole subjects per fold
    cv_stratify_col: str = "group"     # balance normative/covid across folds
    n_folds: int = 5               # for grouped_kfold
    inner_val_frac: float = 0.2    # fraction of training subjects held for early stop
    seed: int = 14
    device: str = "cuda"          # 'cuda' | 'cpu'
    harmonize: bool = False        # ComBat on features, fit on train fold only
    harmonize_batch_col: str = "protocol"
    harmonize_covars: tuple[str, ...] = ("age", "gender, timedis")
    n_permutations: int = 1000     # permutation null for the headline metric
    n_bootstrap: int = 10000       # bootstrap CIs for per-subject metrics


# ──────────────────────────────────────────────────────────────────────────────
# Non-inferiority (Aim 2)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class NonInferiorityConfig:
    metric: str = "r"             # 'r' (per-subject Pearson) | 'r2'
    margin: float = 0.03           # PRE-REGISTER this before looking at results
    alpha: float = 0.05
    n_bootstrap: int = 10000


# ──────────────────────────────────────────────────────────────────────────────
# Top-level config
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SonataConfig:
    paths: Paths = field(default_factory=Paths)
    cohorts: CohortPaths = field(default_factory=CohortPaths)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    tract: TractConfig = field(default_factory=TractConfig)
    func: FuncConfig = field(default_factory=FuncConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    noninf: NonInferiorityConfig = field(default_factory=NonInferiorityConfig)
    cohort: str = "ds000221+covid"  # free-text tag used in output filenames

    def ensure(self) -> None:
        self.paths.ensure()
