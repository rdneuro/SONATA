# sonata/figures.py
"""Publication figures at every stage (dpi=600, transparent where possible).

Two tiers:
- Statistical panels (matplotlib + seaborn) — always run: feature coverage,
  descriptor distributions, predicted-vs-empirical density, model-vs-baseline
  performance with significance, non-inferiority forest, SC–FC coupling along
  the principal structural gradient.
- 3D renders (tractplots / yabplot / PyVista) — run on a VTK-capable machine:
  strongly-smoothed tract surfaces, a *tube-like* tract view (3D skeleton →
  spline → swept tube), cortical ROI maps, and the SONATA showpiece panel that
  places the involved ROIs beside the tract coloured by spectral descriptors
  AND by AD/RD/MD.

Every 3D helper degrades gracefully if its backend is unavailable, so the
statistical figures never block on VTK.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("VTK_USE_OFFSCREEN", "1")   # headless-safe before VTK import

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import SonataConfig
from .utils import get_logger

log = get_logger("sonata.figures")

# Colourblind-safe palette (Wong 2011 / Okabe-Ito).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def set_pub_style() -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.15)
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.transparent": True,
        "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.9,
        "axes.titleweight": "bold", "legend.frameon": False,
    })


def save_fig(fig, name: str, cfg: SonataConfig) -> Path:
    fdir = cfg.paths.output_dir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    png = fdir / f"{name}.png"
    fig.savefig(png, dpi=600, transparent=True, bbox_inches="tight")
    fig.savefig(fdir / f"{name}.pdf", transparent=True, bbox_inches="tight")
    plt.close(fig)
    log.info("figure → %s (+.pdf)", png.name)
    return png


# ──────────────────────────────────────────────────────────────────────────────
# Statistical panels
# ──────────────────────────────────────────────────────────────────────────────
def fig_coverage(meta_df, cfg: SonataConfig, name="01_edge_coverage"):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    sns.histplot(meta_df["coverage_spectral"], bins=20, ax=ax[0], color=PALETTE[0])
    ax[0].set(title="Spectral edge coverage", xlabel="fraction of SC edges with a bundle")
    sns.histplot(meta_df["coverage_diffusion"], bins=20, ax=ax[1], color=PALETTE[1])
    ax[1].set(title="Diffusion edge coverage", xlabel="fraction of SC edges with a bundle")
    fig.tight_layout()
    return save_fig(fig, name, cfg)


def fig_pred_scatter(pred, true, cfg: SonataConfig, name="03_pred_vs_empirical"):
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    hb = ax.hexbin(true, pred, gridsize=45, cmap="magma", mincnt=1, linewidths=0)
    lo = float(np.nanmin([np.nanmin(true), np.nanmin(pred)]))
    hi = float(np.nanmax([np.nanmax(true), np.nanmax(pred)]))
    ax.plot([lo, hi], [lo, hi], "--", color="white", lw=1.2)
    r = np.corrcoef(np.asarray(true), np.asarray(pred))[0, 1]
    ax.set(xlabel="Empirical FC (Fisher-z)", ylabel="Predicted FC",
           title=f"SONATA edge prediction (r = {r:.2f})")
    fig.colorbar(hb, ax=ax, label="edge count")
    fig.tight_layout()
    return save_fig(fig, name, cfg)


def fig_model_vs_baselines(summary_df, per_subject_long, cfg: SonataConfig,
                           name="04_model_vs_baselines"):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    order = summary_df.sort_values("mean_r", ascending=False)["model"].tolist()
    sns.boxplot(data=per_subject_long, x="model", y="r", order=order, ax=ax,
                palette=PALETTE, width=0.6, fliersize=0)
    sns.stripplot(data=per_subject_long, x="model", y="r", order=order, ax=ax,
                  color="0.2", size=3, alpha=0.5, jitter=0.2)
    ax.set(xlabel="", ylabel="per-subject Pearson r (pred vs empirical FC)",
           title="Functional-connectivity prediction: SONATA vs baselines")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return save_fig(fig, name, cfg)


def fig_noninferiority(test: dict, paired_df, cfg: SonataConfig,
                       name="05_noninferiority_forest"):
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    d, lo, hi = test["mean_diff"], test["ci_lo_1m2a"], test["ci_hi_1m2a"]
    ax.errorbar([d], [0], xerr=[[d - lo], [hi - d]], fmt="o", color=PALETTE[0],
                capsize=4, lw=2, ms=8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.axvline(-test["margin"], color=PALETTE[1], ls="--", lw=1.4,
               label=f"-margin = {-test['margin']:.3f}")
    verdict = "NON-INFERIOR" if test["non_inferior"] else "not shown"
    ax.set(yticks=[], xlabel=f"Δ {test['metric']}  (spectral − diffusion)",
           title=f"Aim 2: spectral vs diffusion edge features — {verdict}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save_fig(fig, name, cfg)


def fig_coupling_along_gradient(edges, fc, G, cfg: SonataConfig,
                                name="06_coupling_along_gradient", n_bins=10):
    g_mid = 0.5 * (G[edges[:, 0], 0] + G[edges[:, 1], 0])
    df = _bin_means(g_mid, np.asarray(fc), n_bins)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(df["x"], df["mean"], "-o", color=PALETTE[2], lw=2)
    ax.fill_between(df["x"], df["mean"] - df["sem"], df["mean"] + df["sem"],
                    color=PALETTE[2], alpha=0.25)
    ax.set(xlabel="principal structural gradient (edge midpoint)",
           ylabel="mean empirical FC (Fisher-z)",
           title="SC–FC coupling along the unimodal–transmodal axis")
    fig.tight_layout()
    return save_fig(fig, name, cfg)


def _bin_means(x, y, n_bins):
    import pandas as pd
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 2:
            continue
        rows.append({"x": float(x[m].mean()), "mean": float(y[m].mean()),
                     "sem": float(y[m].std() / np.sqrt(m.sum()))})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 3D renders (graceful fallback)
# ──────────────────────────────────────────────────────────────────────────────
def tract_smoothed_mesh(mask_path, cfg: SonataConfig, smooth_iter: int = 30,
                        pass_band: float = 0.08):
    """Strongly-smoothed tract isosurface as PyVista PolyData (tractplots if present)."""
    try:
        from tractplots import mask_to_mesh
        return mask_to_mesh(str(mask_path), smooth_iter=smooth_iter, decimate=0.4)
    except Exception:
        return _mask_to_mesh_fallback(mask_path, cfg, smooth_iter, pass_band)


def _mask_to_mesh_fallback(mask_path, cfg, smooth_iter, pass_band):
    import nibabel as nib
    import pyvista as pv
    from skimage import measure
    from scipy.ndimage import gaussian_filter
    img = nib.load(str(mask_path))
    vol = gaussian_filter((np.asarray(img.dataobj) > cfg.tract.iso_level).astype(float), 1.0)
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)
    verts = nib.affines.apply_affine(img.affine, verts)
    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    mesh = pv.PolyData(verts, faces_pv)
    return mesh.smooth_taubin(n_iter=smooth_iter, pass_band=pass_band).compute_normals()


def tract_tube(mask_path, cfg: SonataConfig, radius: float = 1.6):
    """Tube-like tract view: 3D skeleton → ordered centreline → spline → swept tube."""
    try:
        import nibabel as nib
        import pyvista as pv
        from skimage.morphology import skeletonize_3d
        img = nib.load(str(mask_path))
        mask = np.asarray(img.dataobj) > cfg.tract.iso_level
        skel = skeletonize_3d(mask)
        pts = np.argwhere(skel)
        if len(pts) < 4:
            return tract_smoothed_mesh(mask_path, cfg)
        pts_world = nib.affines.apply_affine(img.affine, pts.astype(float))
        ordered = _order_centerline(pts_world)
        spline = pv.Spline(ordered, max(50, len(ordered)))
        return spline.tube(radius=radius, n_sides=24)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"tube view failed ({exc!r}); falling back to smoothed mesh")
        return tract_smoothed_mesh(mask_path, cfg)


def _order_centerline(pts: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbour ordering from the most extreme endpoint (PCA)."""
    from scipy.spatial import cKDTree
    c = pts - pts.mean(0)
    pc1 = np.linalg.svd(c, full_matrices=False)[2][0]
    start = int(np.argmin(c @ pc1))
    remaining = set(range(len(pts)))
    order = [start]; remaining.discard(start)
    tree = cKDTree(pts)
    cur = start
    while remaining:
        dists, idxs = tree.query(pts[cur], k=min(8, len(pts)))
        nxt = next((int(i) for i in np.atleast_1d(idxs) if int(i) in remaining), None)
        if nxt is None:
            nxt = remaining.pop()
        else:
            remaining.discard(nxt)
        order.append(nxt); cur = nxt
    return pts[order]


def render_polydata_scalar(mesh, scalars, cmap, title, cfg: SonataConfig):
    """Offscreen render of a PolyData coloured by a scalar → (H,W,4) array."""
    try:
        import pyvista as pv
        pl = pv.Plotter(off_screen=True, window_size=(900, 700))
        pl.set_background(None)
        if scalars is not None:
            mesh = mesh.copy(); mesh["scalars"] = np.asarray(scalars)
            pl.add_mesh(mesh, scalars="scalars", cmap=cmap, smooth_shading=True,
                        pbr=True, metallic=0.1, roughness=0.45, show_scalar_bar=True,
                        scalar_bar_args={"title": title})
        else:
            pl.add_mesh(mesh, color="#888", smooth_shading=True, pbr=True)
        pl.view_vector((1, 0, 0.3))
        img = pl.screenshot(transparent_background=True, return_img=True)
        pl.close()
        return img
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"render failed ({exc!r})")
        return None


def plot_rois_cortex(values_by_label: dict, cfg: SonataConfig,
                     name="roi_map", cmap="rocket"):
    """Cortical ROI map via yabplot Schaefer-200 template (graceful fallback)."""
    try:
        import yabplot as yab
        out = cfg.paths.output_dir / "figures" / f"{name}.png"
        yab.plot_cortical(data=values_by_label, atlas="schaefer_200",
                          views=["left_lateral", "left_medial",
                                 "right_medial", "right_lateral"],
                          bmesh="midthickness", cmap=cmap, style="matte",
                          display_type="static", export_path=str(out))
        log.info("cortical ROI map → %s", out.name)
        return out
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"yabplot unavailable ({exc!r}); skipping cortical map")
        return None


def sample_nifti_at_points(path, points_world: np.ndarray) -> np.ndarray:
    """Sample a NIfTI scalar map at world (RAS mm) coordinates (nearest voxel)."""
    import nibabel as nib
    img = nib.load(str(path))
    vol = np.asarray(img.dataobj, dtype=float)
    inv = np.linalg.inv(img.affine)
    vox = nib.affines.apply_affine(inv, points_world).round().astype(int)
    vox = np.clip(vox, 0, np.array(vol.shape) - 1)
    return vol[vox[:, 0], vox[:, 1], vox[:, 2]]


def _resolve_scalar(value, points):
    """Allow scalar inputs to be arrays or callables(points)->array."""
    if value is None:
        return None
    if callable(value):
        try:
            return np.asarray(value(points))
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"scalar callable failed ({exc!r})")
            return None
    return np.asarray(value)


def panel_roi_tract(mask_path, roi_values: dict, spectral_scalar, diffusion_scalar,
                    cfg: SonataConfig, name="07_showpiece_roi_tract",
                    tract_name: str = "tract", spectral_label: str = "HKS",
                    diffusion_label: str = "MD"):
    """SONATA showpiece: ROIs + tract coloured by spectral descriptor AND by AD/RD/MD.

    ``spectral_scalar`` / ``diffusion_scalar`` may be arrays aligned to the tube
    points, or callables ``f(points_world) -> array`` (e.g. nearest-vertex HKS
    transfer, or ``sample_nifti_at_points`` for a diffusion map).
    """
    tube = tract_tube(mask_path, cfg)
    pts = np.asarray(tube.points) if hasattr(tube, "points") else None
    spec_vals = _resolve_scalar(spectral_scalar, pts)
    diff_vals = _resolve_scalar(diffusion_scalar, pts)
    img_spec = render_polydata_scalar(tube, spec_vals, "inferno", spectral_label, cfg)
    img_diff = render_polydata_scalar(tube, diff_vals, "viridis", diffusion_label, cfg)
    roi_png = plot_rois_cortex(roi_values, cfg, name=f"{name}_rois")

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for a in ax:
        a.axis("off")
    if roi_png and Path(roi_png).exists():
        ax[0].imshow(plt.imread(str(roi_png)))
    ax[0].set_title(f"Involved ROIs\n({tract_name})", fontweight="bold")
    if img_spec is not None:
        ax[1].imshow(img_spec)
    ax[1].set_title(f"Tract · {spectral_label} (spectral)", fontweight="bold")
    if img_diff is not None:
        ax[2].imshow(img_diff)
    ax[2].set_title(f"Tract · {diffusion_label} (diffusion)", fontweight="bold")
    fig.suptitle(f"SONATA — geometry of edge {tract_name}", fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_fig(fig, name, cfg)
