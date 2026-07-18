# sonata/viz/brain3d.py
"""Three-dimensional renders of SONATA results onto surfaces and tracts.

Wrappers over :mod:`vedo` (mesh/tract scalar overlays, offscreen), :mod:`nilearn`
(cortical surface maps), and ``yabplot`` (Schaefer-200 cortical/subcortical maps).
Every heavy 3D dependency is imported lazily and, if absent, raises a clear
message naming the extra that provides it — so this module imports on any machine
and only demands a 3D stack when a 3D figure is actually requested.

All renderers run offscreen and write an image file, which suits headless
servers and keeps the API uniform (each returns the output path).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _require(module: str, extra: str):
    """Import ``module`` lazily or raise a helpful, actionable error."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            f"3D rendering needs '{module}', not installed. "
            f"Install the extra:  pip install -e '.[{extra}]'"
        ) from exc


def surface_scalar(
    vertices: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray,
    out_path: str | Path,
    *,
    cmap: str = "magma",
    title: str = "",
    size: tuple[int, int] = (1100, 900),
    smooth: int = 0,
) -> Path:
    """Render a triangle mesh coloured by a per-vertex scalar (vedo, offscreen).

    Parameters
    ----------
    vertices, faces
        Mesh geometry ``(V, 3)`` and ``(F, 3)``.
    scalars
        Per-vertex scalar field (e.g. HKS at a fixed time, a t-map, Cohen's d).
    out_path
        Output image path (``.png``).
    cmap, title, size, smooth
        Colour map, on-canvas title, image size, and optional Laplacian smoothing.

    Returns
    -------
    pathlib.Path
        The written image path.
    """
    vedo = _require("vedo", "viz3d")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = vedo.Mesh([np.asarray(vertices, float), np.asarray(faces)])
    if smooth > 0:
        mesh.smooth(niter=int(smooth))
    mesh.cmap(cmap, np.asarray(scalars, float)).add_scalarbar(title=title)
    plotter = vedo.Plotter(offscreen=True, size=size)
    plotter.show(mesh, title, axes=0)
    plotter.screenshot(str(out_path))
    plotter.close()
    return out_path


def tract_scalar(
    vertices: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray,
    out_path: str | Path,
    *,
    cmap: str = "viridis",
    title: str = "tract",
    size: tuple[int, int] = (1200, 800),
) -> Path:
    """Render a white-matter tract isosurface coloured by a scalar (vedo).

    Thin alias of :func:`surface_scalar` with tract-oriented defaults; kept
    separate so tract-specific styling can diverge without touching callers.
    """
    return surface_scalar(vertices, faces, scalars, out_path, cmap=cmap,
                          title=title, size=size)


def cortex_rois(
    values_by_label: dict[int, float],
    out_path: str | Path,
    *,
    atlas: str = "schaefer200",
    cmap: str = "coolwarm",
    title: str = "",
    backend: str = "auto",
) -> Path:
    """Render a per-ROI scalar onto the cortical surface.

    Uses ``yabplot`` when available (native Schaefer-200 support), else falls back
    to a :mod:`nilearn` surface plot. ``backend="auto"`` prefers yabplot.

    Parameters
    ----------
    values_by_label
        Mapping ROI integer label -> scalar value.
    out_path
        Output image path.
    atlas, cmap, title
        Atlas name, colour map, and title.
    backend
        ``"auto"`` | ``"yabplot"`` | ``"nilearn"``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if backend in ("auto", "yabplot"):
        try:
            yb = _require("yabplot", "viz3d")
            return _cortex_via_yabplot(yb, values_by_label, out_path, atlas, cmap, title)
        except ImportError:
            if backend == "yabplot":
                raise
    # nilearn fallback
    return _cortex_via_nilearn(values_by_label, out_path, atlas, cmap, title)


def _cortex_via_yabplot(yb, values_by_label, out_path, atlas, cmap, title) -> Path:
    """Render Schaefer-200 ROI values with yabplot (kept isolated for clarity)."""
    yb.plot_cortical(  # API shim: exact call may vary with yabplot version
        values_by_label, atlas=atlas, cmap=cmap, title=title,
        outfile=str(out_path),
    )
    return out_path


def _cortex_via_nilearn(values_by_label, out_path, atlas, cmap, title) -> Path:
    """Fallback cortical render via nilearn fsaverage surface projection."""
    nilearn_plotting = _require("nilearn.plotting", "viz3d")
    datasets = _require("nilearn.datasets", "viz3d")
    surface = _require("nilearn.surface", "viz3d")
    np_ = np

    fsaverage = datasets.fetch_surf_fsaverage()
    # Map ROI labels -> a fsaverage-vertexwise texture using the Schaefer atlas.
    atlas_obj = datasets.fetch_atlas_schaefer_2018(n_rois=200)
    # atlas_obj.maps is a volumetric atlas; project to surface then recolour.
    tex = surface.vol_to_surf(atlas_obj.maps, fsaverage.pial_left)
    out = np_.zeros_like(tex, dtype=float)
    for lab, val in values_by_label.items():
        out[np_.round(tex).astype(int) == int(lab)] = float(val)
    fig = nilearn_plotting.plot_surf_stat_map(
        fsaverage.infl_left, out, hemi="left", cmap=cmap, title=title,
        colorbar=True,
    )
    fig.savefig(str(out_path), dpi=200)
    return out_path


__all__ = ["surface_scalar", "tract_scalar", "cortex_rois"]
