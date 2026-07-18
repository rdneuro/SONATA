# sonata/viz/panels.py
"""Composition helpers: assemble individual plots into multi-panel figures.

The 2D plotters in :mod:`sonata.viz.metrics` and :mod:`sonata.viz.heatmaps` each
accept an ``ax=`` and return a Figure, so they compose. :func:`grid` lays an
arbitrary list of "draw into this axis" callables into a labelled grid;
:func:`results_dashboard` is the ready-made SONATA summary (benchmark bars +
prediction scatter + non-inferiority forest) built on top of it.
"""

from __future__ import annotations

import string
from typing import Callable, Sequence

import numpy as np

from . import metrics, style


def grid(
    drawers: Sequence[Callable[[object], None]],
    *,
    ncols: int = 2,
    panel_size: tuple[float, float] = (4.2, 3.4),
    letter_panels: bool = True,
    suptitle: str | None = None,
):
    """Composite ``drawers`` (each ``fn(ax)``) into an ``ncols``-wide grid.

    Parameters
    ----------
    drawers
        Callables that draw into a provided matplotlib Axes.
    ncols
        Number of columns; rows are derived from the count.
    panel_size
        Per-panel ``(width, height)`` in inches.
    letter_panels
        Prefix each panel title area with (a), (b), ... for figure references.
    suptitle
        Optional overall title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    style.set_pub_style()
    n = len(drawers)
    nrows = int(np.ceil(n / ncols))
    fig, axes = style.plt.subplots(
        nrows, ncols, figsize=(ncols * panel_size[0], nrows * panel_size[1])
    )
    axes = np.atleast_1d(axes).ravel()
    for i, draw in enumerate(drawers):
        draw(axes[i])
        if letter_panels:
            axes[i].text(-0.08, 1.06, f"({string.ascii_lowercase[i]})",
                         transform=axes[i].transAxes, fontsize=11,
                         fontweight="bold", va="bottom", ha="right")
    for j in range(n, len(axes)):  # hide unused cells
        axes[j].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def results_dashboard(
    *,
    model_labels: Sequence[str],
    model_scores: Sequence[float],
    benchmark: float,
    pred: np.ndarray,
    true: np.ndarray,
    ni_labels: Sequence[str],
    ni_estimate: Sequence[float],
    ni_low: Sequence[float],
    ni_high: Sequence[float],
    ni_margin: float,
    suptitle: str = "SONATA results",
):
    """The canonical SONATA summary figure in three panels.

    (a) benchmark bar comparison, (b) predicted-vs-empirical scatter, and
    (c) the non-inferiority forest — the honest one-figure story of a run.
    """
    def _bars(ax):
        metrics.bars(model_labels, model_scores, reference=benchmark,
                     reference_label="group-mean", ylabel="mean r",
                     title="benchmark", ax=ax)

    def _scatter(ax):
        metrics.scatter(pred, true, title="pred vs. empirical", ax=ax)

    def _forest(ax):
        metrics.forest(ni_labels, ni_estimate, ni_low, ni_high,
                       margin=ni_margin, title="non-inferiority", ax=ax)

    return grid([_bars, _scatter, _forest], ncols=3, panel_size=(3.6, 3.4),
                suptitle=suptitle)


__all__ = ["grid", "results_dashboard"]
