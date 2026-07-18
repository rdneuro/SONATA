# sonata/viz/heatmaps.py
"""Connectivity heatmaps: one solo labelled map, and an evolutionary panel.

Two public functions share a single drawing primitive (:func:`_draw`):

* :func:`heatmap` — a single, final, labelled matrix (e.g. predicted FC) with an
  attached colourbar and optional network/block dividers.
* :func:`evolutionary_heatmaps` — a grid of matrices captured at successive
  *stages* of processing (e.g. FC across training epochs, or SC->FC coupling
  across pipeline steps), all sharing one value scale and one colourbar, with a
  companion barplot summarising the same values per stage so the reader sees the
  evolution both spatially and as a trend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import style


# ── shared primitive ──────────────────────────────────────────────────────────
def _draw(
    ax,
    mat: np.ndarray,
    *,
    vmin: float | None,
    vmax: float | None,
    cmap: str,
    labels: Sequence[str] | None = None,
    blocks: Sequence[int] | None = None,
    title: str | None = None,
):
    """Render one matrix into ``ax``; return the image handle for a colourbar."""
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
                   interpolation="nearest")
    if title:
        ax.set_title(title)
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if blocks:  # draw dividers between network blocks (cumulative sizes)
        pos = np.cumsum(blocks)[:-1] - 0.5
        for p in pos:
            ax.axhline(p, color="w", lw=0.5)
            ax.axvline(p, color="w", lw=0.5)
    return im


def _symmetric_limits(mats: Sequence[np.ndarray], diverging: bool) -> tuple[float, float]:
    """Compute shared (vmin, vmax) across matrices; symmetric if diverging."""
    stacked = np.concatenate([np.asarray(m, float).ravel() for m in mats])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return 0.0, 1.0
    if diverging:
        a = float(np.nanpercentile(np.abs(finite), 99))
        return -a, a
    return float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99))


# ── public: solo heatmap ──────────────────────────────────────────────────────
def heatmap(
    mat: np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    blocks: Sequence[int] | None = None,
    diverging: bool = True,
    title: str = "connectivity",
    cbar_label: str = "value",
    figsize: tuple[float, float] = (5.2, 4.6),
):
    """Plot a single labelled connectivity matrix with a colourbar.

    Parameters
    ----------
    mat
        Square matrix (e.g. an FC or SC adjacency).
    labels
        Optional per-row/column tick labels (kept small; omit for large atlases).
    blocks
        Optional block sizes (e.g. Schaefer network sizes) to draw white dividers.
    diverging
        Use a symmetric diverging colour map centred at zero (for signed data
        like Fisher-z FC); otherwise a sequential map.
    title, cbar_label, figsize
        Cosmetic controls.

    Returns
    -------
    matplotlib.figure.Figure
    """
    style.set_pub_style()
    cmap = style.DIVERGING_CMAP if diverging else style.SEQ_CMAP
    vmin, vmax = _symmetric_limits([mat], diverging)
    fig, ax = style.plt.subplots(figsize=figsize)
    im = _draw(ax, np.asarray(mat, float), vmin=vmin, vmax=vmax, cmap=cmap,
               labels=labels, blocks=blocks, title=title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    fig.tight_layout()
    return fig


# ── layout helper (reused) ────────────────────────────────────────────────────
def _grid_shape(n: int) -> tuple[int, int]:
    """Pick a pleasant (rows, cols) grid; prefer 3x4 / 4x3 near 12 panels."""
    if n <= 12:
        # honour the 3x4 / 4x3 hint: wider-than-tall for a landscape panel
        for rows in (3, 4, 2):
            cols = -(-n // rows)  # ceil
            if rows * cols >= n and cols >= rows:
                return rows, cols
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


# ── public: evolutionary panel ────────────────────────────────────────────────
def evolutionary_heatmaps(
    mats: Sequence[np.ndarray],
    *,
    stage_labels: Sequence[str] | None = None,
    diverging: bool = True,
    summary: Callable[[np.ndarray], float] | str = "mean_abs",
    summary_label: str = "mean |value|",
    cbar_label: str = "value",
    suptitle: str = "evolution across processing stages",
    panel_size: float = 1.9,
):
    """Grid of matrices across stages, shared scale + colourbar + summary barplot.

    All matrices share one ``(vmin, vmax)`` and one colourbar so panels are
    directly comparable. A companion barplot below the grid shows a scalar
    ``summary`` of each matrix, reading the same values as a trend.

    Parameters
    ----------
    mats
        Sequence of square matrices, one per processing stage (e.g. epochs).
    stage_labels
        Titles for each panel; defaults to ``t0, t1, ...``.
    diverging
        Symmetric diverging map (signed data) vs sequential.
    summary
        Per-matrix scalar for the barplot: a callable, or one of
        ``"mean"``, ``"mean_abs"``, ``"median"``, ``"frobenius"``.
    summary_label, cbar_label, suptitle, panel_size
        Cosmetic controls; ``panel_size`` is the per-heatmap side in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    style.set_pub_style()
    mats = [np.asarray(m, float) for m in mats]
    n = len(mats)
    if n == 0:
        raise ValueError("evolutionary_heatmaps needs at least one matrix")
    if stage_labels is None:
        stage_labels = [f"t{i}" for i in range(n)]

    cmap = style.DIVERGING_CMAP if diverging else style.SEQ_CMAP
    vmin, vmax = _symmetric_limits(mats, diverging)
    rows, cols = _grid_shape(n)

    summ_fn = _resolve_summary(summary)
    summ_vals = np.array([summ_fn(m) for m in mats], float)

    # Layout: (rows) heatmap rows + 1 barplot row spanning all columns.
    fig = style.plt.figure(figsize=(cols * panel_size + 0.8,
                                    rows * panel_size + panel_size + 0.6))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[*([1] * rows), 0.9],
                          hspace=0.35, wspace=0.15)

    im = None
    for k in range(n):
        r, c = divmod(k, cols)
        ax = fig.add_subplot(gs[r, c])
        im = _draw(ax, mats[k], vmin=vmin, vmax=vmax, cmap=cmap,
                   title=str(stage_labels[k]))
        ax.set_title(str(stage_labels[k]), fontsize=8)

    # Shared colourbar down the right side of the heatmap block.
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.42, 0.015, 0.44])
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_label(cbar_label)

    # Companion barplot: same summary value per stage, shared value axis.
    axb = fig.add_subplot(gs[rows, :])
    colors = style.color_cycle(n)
    axb.bar(range(n), summ_vals, color=colors, edgecolor="black", linewidth=0.5)
    axb.set_xticks(range(n))
    axb.set_xticklabels([str(s) for s in stage_labels], rotation=0, fontsize=7)
    axb.set_ylabel(summary_label)
    axb.set_title("per-stage summary", fontsize=8, loc="left")
    axb.margins(x=0.01)

    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=0.99)
    fig.subplots_adjust(right=0.9)
    return fig


def _resolve_summary(summary: Callable[[np.ndarray], float] | str) -> Callable[[np.ndarray], float]:
    """Map a summary name to a nan-robust reducer, or pass a callable through."""
    if callable(summary):
        return summary
    table: dict[str, Callable[[np.ndarray], float]] = {
        "mean": lambda m: float(np.nanmean(m)),
        "mean_abs": lambda m: float(np.nanmean(np.abs(m))),
        "median": lambda m: float(np.nanmedian(m)),
        "frobenius": lambda m: float(np.sqrt(np.nansum(m ** 2))),
    }
    if summary not in table:
        raise ValueError(f"unknown summary {summary!r}; choose {list(table)} or pass a callable")
    return table[summary]


__all__ = ["heatmap", "evolutionary_heatmaps"]
