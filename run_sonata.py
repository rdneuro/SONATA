# run_sonata.py
# =============================================================================
# SONATA — end-to-end orchestrator (Spyder-friendly; run cell by cell with # %%)
#
# Structure-tO-fuNction via spectrAl Tract Attributes.
#
# SINGLE SOURCE OF TRUTH: every path (output_dir, manifest_csv, FreeSurfer
# subjects_dir, Schaefer annots) and every cohort root lives in sonata/config.py.
# This orchestrator instantiates SonataConfig() and reads everything from it and
# from the manifest the manifest builder wrote — there are NO hardcoded paths
# here. Edit a path in ONE place: config.py.
#
# Each stage writes publication tables (CSV + .tex) and figures (PNG+PDF,
# dpi=600, transparent) to <output_dir>, and is resumable (per-subject feature
# cache + per-fold checkpoints + spectra checkpoints).
# =============================================================================

# %% [0] Imports & configuration ---------------------------------------------
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sonata.config import SonataConfig
from sonata.utils import get_logger, load_manifest, seed_everything, track
from sonata import figures as F
from sonata.reporting import save_table, summary_comparison

# All paths and cohort roots come from config.py (single source of truth).
# To change a path, edit sonata/config.py — never here.
cfg = SonataConfig()

# Cross-validation design (see run notes below):
#   'grouped_kfold' = k-fold over BOTH cohorts pooled, grouped by subject
#       (leakage-safe) and stratified by `group` so every fold keeps a similar
#       normative/covid mix. This is the MAIN-PAPER design (Aims 1 & 2): it uses
#       the full n and answers "does spectral geometry beat connectivity-only and
#       is it non-inferior to diffusion?", with `group` as a covariate/stratum.
#   'loso' = leave-one-subject-out (use for the small COVID-only sensitivity run).
# Note: a normative->clinical TRANSFER design (train normative, test covid) is a
# DIFFERENT, secondary question (deviation-from-normative) and is intentionally
# NOT the default — with n=24 and scanner perfectly confounded with group it
# can't cleanly separate disease from acquisition. Kept as a future spin-off.
cfg.train.cv = "grouped_kfold"
cfg.train.cv_group_col = "subject_id"   # leakage-safe: whole subjects per fold
cfg.train.cv_stratify_col = "group"     # balance normative/covid across folds
cfg.train.device = "cuda"
cfg.ensure()

run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
log = get_logger("sonata", level=logging.INFO,
                 logfile=cfg.paths.output_dir / "logs" / f"run_{run_tag}.log")
seed_everything(cfg.train.seed)
F.set_pub_style()
log.info("SONATA %s │ cohort=%s │ cv=%s", run_tag, cfg.cohort, cfg.train.cv)
log.info("output_dir=%s", cfg.paths.output_dir)
log.info("manifest_csv=%s", cfg.paths.manifest_csv)


# %% [1] Manifest -------------------------------------------------------------
# The manifest is the authority for who enters and which cohort each subject is.
# It was written by build_manifest.py from the two authority CSVs (n=121 ds000221
# intersection + covid de-identified) with per-cohort absolute paths resolved.
manifest = load_manifest(cfg.paths.manifest_csv)
log.info("manifest: %d subjects │ groups=%s", len(manifest),
         dict(manifest["group"].value_counts()))
save_table(manifest[["subject_id", "group", "protocol", "age", "sex"]],
           "00_cohort", cfg, caption="Cohort and acquisition summary.")


# %% [2] Build per-subject features (cached / resumable) ----------------------
from sonata.graph import build_subject_features

all_feats = []
for _, row in track(list(manifest.iterrows()), "Building subject graphs",
                    total=len(manifest)):
    try:
        all_feats.append(build_subject_features(row, cfg))
    except Exception as exc:
        log.error("subject %s failed: %r", row["subject_id"], exc)

meta_df = pd.DataFrame([f["meta"] for f in all_feats])
save_table(meta_df, "01_feature_extraction", cfg,
           caption="Per-subject graph construction: edges, bundles, and "
                   "spectral/diffusion edge coverage.")
F.fig_coverage(meta_df, cfg)
log.info("features ready: %d subjects │ mean spectral coverage=%.2f",
         len(all_feats), meta_df["coverage_spectral"].mean())
# FIRST sanity check before any conclusions: inspect 01_feature_extraction —
# a wrong tractseg/sift2/fc path shows up here as low/zero spectral coverage.


# %% [3] Persist eigenpairs + descriptor pickles (resumable) ------------------
from sonata.persistence import persist_subject_spectra

built_ids = set(meta_df["subject_id"]) if len(meta_df) else set()
for _, row in track(list(manifest.iterrows()), "Persisting spectra",
                    total=len(manifest)):
    if row["subject_id"] in built_ids:
        try:
            persist_subject_spectra(row, cfg)
        except Exception as exc:
            log.error("persist %s failed: %r", row["subject_id"], exc)


# %% [4] Structural gradients (interpretation axis) ---------------------------
from sonata.gradients import fit_gradients

G = fit_gradients(all_feats, n_components=5)
np.save(cfg.paths.output_dir / "tables" / "structural_gradients.npy", G)
save_table(pd.DataFrame(G[:, :3], columns=["G1", "G2", "G3"]).assign(roi=range(1, 201)),
           "02_structural_gradients", cfg,
           caption="First three diffusion-map gradients of the group-mean "
                   "structural connectome.")


# %% [5] SONATA — Aim 1 (spectral edge features) ------------------------------
from sonata.train import run_cv

cfg.model.edge_feature_mode = "spectral"
cfg.model.node_feature_mode = "full"
res_spectral = run_cv(all_feats, cfg, tag=f"{cfg.cohort}_spectral")
save_table(res_spectral["metrics"], "03_sonata_spectral_persubject", cfg,
           caption="SONATA (spectral edge features): per-subject FC-prediction "
                   "performance under leakage-safe cross-validation.")
log.info("SONATA spectral │ mean r=%.3f  CI=%.3f–%.3f  p_perm=%.4f",
         res_spectral["mean_r"], *res_spectral["ci_r"], res_spectral["p_perm"])

# Pooled predicted-vs-empirical density + coupling-along-gradient.
ex = all_feats[0]
F.fig_coupling_along_gradient(ex["edges"], ex["fc"], G, cfg)


# %% [6] Baselines + ablations ------------------------------------------------
from sonata.baselines import run_all_baselines

baselines_df = run_all_baselines(all_feats, cfg)

# One-hot-node GNN ablation (Chen/Cui analogue): identity nodes + SIFT2-only edges.
cfg.model.node_feature_mode = "identity"
cfg.model.edge_feature_mode = "sift2"
res_onehot = run_cv(all_feats, cfg, tag=f"{cfg.cohort}_onehot")
onehot_long = res_onehot["metrics"].assign(model="GNN one-hot (SIFT2)")
cfg.model.node_feature_mode = "full"     # restore

spectral_long = res_spectral["metrics"].assign(model="SONATA (spectral)")
per_subject_long = pd.concat(
    [spectral_long[["subject_id", "group", "r", "r2", "model"]],
     onehot_long[["subject_id", "group", "r", "r2", "model"]],
     baselines_df[["subject_id", "group", "r", "r2", "model"]]],
    ignore_index=True)

summary = summary_comparison(res_spectral, baselines_df, cfg)
summary = pd.concat([summary, pd.DataFrame([{
    "model": "GNN one-hot (SIFT2)", "mean_r": res_onehot["mean_r"],
    "ci_low": res_onehot["ci_r"][0], "ci_high": res_onehot["ci_r"][1],
    "mean_r2": float(res_onehot["metrics"]["r2"].mean()),
    "p_perm": res_onehot["p_perm"],
    "n_subjects": int(res_onehot["metrics"]["subject_id"].nunique())}])],
    ignore_index=True).sort_values("mean_r", ascending=False)
save_table(summary, "04_model_comparison", cfg,
           caption="FC-prediction performance: SONATA vs. ablations and baselines "
                   "(mean per-subject Pearson r, 95\\% bootstrap CI, permutation p).")
F.fig_model_vs_baselines(summary, per_subject_long, cfg)


# %% [7] SONATA — Aim 2 comparator (diffusion edge features) ------------------
cfg.model.edge_feature_mode = "diffusion"
res_diffusion = run_cv(all_feats, cfg, tag=f"{cfg.cohort}_diffusion")
cfg.model.edge_feature_mode = "spectral"   # restore
save_table(res_diffusion["metrics"], "05_sonata_diffusion_persubject", cfg,
           caption="SONATA with classical diffusion edge features (AD/RD/MD): "
                   "per-subject FC-prediction performance.")


# %% [8] Non-inferiority (spectral vs diffusion) ------------------------------
from sonata.noninferiority import compare_feature_sets

paired, ni_test = compare_feature_sets(res_spectral, res_diffusion, cfg)
save_table(paired, "06_noninferiority_paired", cfg,
           caption="Paired per-subject performance: spectral vs diffusion edge "
                   "features.")
save_table(pd.DataFrame([ni_test]), "07_noninferiority_test", cfg,
           caption=f"Two-one-sided-test (TOST) non-inferiority of spectral vs "
                   f"diffusion edge features (pre-registered margin "
                   f"{cfg.noninf.margin}).")
F.fig_noninferiority(ni_test, paired, cfg)
log.info("Non-inferiority │ Δ%s=%.3f  CI=%.3f–%.3f  margin=%.3f  → %s",
         ni_test["metric"], ni_test["mean_diff"], ni_test["ci_lo_1m2a"],
         ni_test["ci_hi_1m2a"], cfg.noninf.margin,
         "NON-INFERIOR" if ni_test["non_inferior"] else "not shown")


# %% [9] Showpiece figure — ROIs + tract (spectral & diffusion) ---------------
# Pick a representative bundle for the lead subject and colour the tube by HKS
# (transferred from the tract mesh) and by MD (sampled from the subject's map).
import spectralbrain as sb
from scipy.spatial import cKDTree
from sonata.graph import _as_vf, _scalar_maps_for

lead = manifest.iloc[0]
bundle_paths = sb.discover_tractseg_bundles(
    lead["tractseg_dir"], bundles=cfg.tract.bundles, subdir=cfg.tract.tractseg_subdir)
if isinstance(bundle_paths, (list, tuple)):
    bundle_paths = {Path(p).stem: str(p) for p in bundle_paths}
example_name = next(iter(bundle_paths))
example_mask = bundle_paths[example_name]

# HKS on the tract mesh → nearest-vertex transfer to tube points.
mesh = sb.load_tractseg_bundle(example_mask, output="mesh", level=cfg.tract.iso_level)
mv, mf = _as_vf(mesh)
bm = sb.BrainMesh(mv, mf).taubin_smooth(n_iterations=cfg.spectral.taubin_iter)
decomp = bm.decompose(k=min(cfg.spectral.n_eigen, max(4, bm.n_vertices - 2)),
                      laplacian_method=cfg.spectral.laplacian)
hks = np.asarray(sb.compute_hks(decomp, n_times=cfg.spectral.hks_n_times))[:, cfg.spectral.hks_n_times // 2]
_tree = cKDTree(mv)

def spectral_fn(points):
    return hks[_tree.query(points)[1]]

scalar_maps = _scalar_maps_for(lead, cfg)
md_path = scalar_maps.get("MD")

def diffusion_fn(points):
    return F.sample_nifti_at_points(md_path, points) if md_path else None

# Highlight the two strongest ROIs in this bundle's footprint.
roi_values = {}   # populate with yabplot Schaefer-200 region labels if desired
F.panel_roi_tract(example_mask, roi_values, spectral_fn,
                  diffusion_fn if md_path else None, cfg,
                  tract_name=example_name, spectral_label="HKS", diffusion_label="MD")


# %% [10] Final summary & interpretation --------------------------------------
print("\n" + "=" * 70)
print(f"SONATA — {cfg.cohort} — run {run_tag}")
print("=" * 70)
print(summary.to_string(index=False))
print("-" * 70)
print(f"Aim 1  SONATA (spectral): mean r = {res_spectral['mean_r']:.3f} "
      f"[{res_spectral['ci_r'][0]:.3f}, {res_spectral['ci_r'][1]:.3f}], "
      f"permutation p = {res_spectral['p_perm']:.4f}")
print(f"Aim 2  spectral vs diffusion: Δ{ni_test['metric']} = {ni_test['mean_diff']:+.3f} "
      f"(margin {cfg.noninf.margin}) → "
      f"{'NON-INFERIOR' if ni_test['non_inferior'] else 'not established'}")
beats_onehot = res_spectral["mean_r"] > res_onehot["mean_r"]
print(f"Geometry value: SONATA {'>' if beats_onehot else '<='} one-hot GNN "
      f"({res_spectral['mean_r']:.3f} vs {res_onehot['mean_r']:.3f})")
print("=" * 70)
log.info("DONE. Tables → %s/tables · Figures → %s/figures",
         cfg.paths.output_dir, cfg.paths.output_dir)
