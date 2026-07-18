# sonata/viz/metrics.py
"""Two-dimensional result plots: bar, scatter, volcano, line, forest.

These render the quantitative story of a SONATA run — model comparisons,
predicted-vs-empirical agreement, per-edge/region effects, training curves, and
non-inferiority intervals — with one consistent palette and a Figure return so
callers can composite them into panels (see :mod:`sonata.viz.panels`).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import style


def bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    errors: Sequence[float] | None = None,
    reference: float | None = None,
    reference_label: str = "benchmark",
    ylabel: str = "score",
    title: str = "model comparison",
    figsize: tuple[float, float] = (5.0, 3.4),
    ax=None,
):
    """Grouped bar chart of a scalar metric per model/condition.

    A horizontal ``reference`` line (e.g. the group-average benchmark) makes an
    honest comparison immediate.
    """
    style.set_pub_style()
    fig, ax = _fig_ax(ax, figsize)
    x = np.arange(len(labels))
    colors = style.color_cycle(len(labels))
    ax.bar(x, values, yerr=errors, color=colors, edgecolor="black",
           linewidth=0.6, capsize=3)
    if reference is not None:
        ax.axhline(reference, ls="--", color=style.OKABE_ITO["grey"], lw=1.0)
        ax.text(len(labels) - 0.5, reference, f" {reference_label}", va="bottom",
                ha="right", fontsize=7, color=style.OKABE_ITO["grey"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def scatter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str = "predicted",
    ylabel: str = "empirical",
    title: str = "prediction vs. empirical",
    identity: bool = True,
    annotate_r: bool = True,
    figsize: tuple[float, float] = (4.2, 4.0),
    ax=None,
):
    """Scatter of paired values with an identity line and an ``r`` annotation."""
    style.set_pub_style()
    fig, ax = _fig_ax(ax, figsize)
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[m], y[m], s=8, alpha=0.5, color=style.OKABE_ITO["blue"],
               edgecolors="none")
    if identity and m.any():
        lo = float(min(x[m].min(), y[m].min()))
        hi = float(max(x[m].max(), y[m].max()))
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color=style.OKABE_ITO["grey"])
    if annotate_r and m.sum() > 2:
        r = float(np.corrcoef(x[m], y[m])[0, 1])
        ax.text(0.04, 0.94, f"r = {r:.3f}", transform=ax.transAxes, fontsize=9,
                va="top", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def volcano(
    effect: np.ndarray,
    pvalue: np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    alpha: float = 0.05,
    effect_thresh: float = 0.0,
    xlabel: str = "effect size",
    title: str = "volcano",
    n_annotate: int = 8,
    figsize: tuple[float, float] = (5.0, 4.2),
    ax=None,
):
    """Volcano plot of effect size vs. -log10 p, highlighting significant hits.

    Points passing both ``pvalue < alpha`` and ``|effect| > effect_thresh`` are
    coloured by sign; the strongest ``n_annotate`` are labelled when ``labels``
    are given.
    """
    style.set_pub_style()
    fig, ax = _fig_ax(ax, figsize)
    effect = np.asarray(effect, float).ravel()
    pvalue = np.asarray(pvalue, float).ravel()
    nlp = -np.log10(np.clip(pvalue, 1e-300, 1.0))
    sig = (pvalue < alpha) & (np.abs(effect) > effect_thresh)
    up = sig & (effect > 0)
    dn = sig & (effect < 0)
    ax.scatter(effect[~sig], nlp[~sig], s=7, color=style.OKABE_ITO["grey"],
               alpha=0.4, edgecolors="none", label="ns")
    ax.scatter(effect[up], nlp[up], s=12, color=style.OKABE_ITO["red"],
               edgecolors="none", label="up")
    ax.scatter(effect[dn], nlp[dn], s=12, color=style.OKABE_ITO["blue"],
               edgecolors="none", label="down")
    ax.axhline(-np.log10(alpha), ls="--", lw=0.8, color="black")
    if effect_thresh > 0:
        ax.axvline(effect_thresh, ls=":", lw=0.7, color="black")
        ax.axvline(-effect_thresh, ls=":", lw=0.7, color="black")
    if labels is not None and sig.any():
        order = np.argsort(nlp)[::-1]
        shown = 0
        for i in order:
            if sig[i] and shown < n_annotate:
                ax.annotate(str(labels[i]), (effect[i], nlp[i]), fontsize=6,
                            xytext=(2, 2), textcoords="offset points")
                shown += 1
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$-\log_{10}\,p$")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper right", markerscale=1.2)
    fig.tight_layout()
    return fig


def lines(
    series: dict[str, tuple[Sequence[float], Sequence[float]]],
    *,
    xlabel: str = "epoch",
    ylabel: str = "loss",
    title: str = "training curves",
    logy: bool = False,
    figsize: tuple[float, float] = (5.2, 3.4),
    ax=None,
):
    """Line plot of named ``(x, y)`` series (e.g. train/val learning curves)."""
    style.set_pub_style()
    fig, ax = _fig_ax(ax, figsize)
    colors = style.color_cycle(len(series))
    for (name, (xs, ys)), col in zip(series.items(), colors):
        ax.plot(xs, ys, label=name, color=col)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend()
    fig.tight_layout()
    return fig


def forest(
    labels: Sequence[str],
    estimates: Sequence[float],
    ci_low: Sequence[float],
    ci_high: Sequence[float],
    *,
    margin: float | None = None,
    ref: float = 0.0,
    xlabel: str = "effect (Δ)",
    title: str = "non-inferiority",
    figsize: tuple[float, float] = (5.4, 3.2),
    ax=None,
):
    """Forest plot of point estimates with confidence intervals.

    A dashed non-inferiority ``margin`` and a solid reference (``ref``, default 0)
    make the TOST decision visible: an interval whose lower bound clears the
    margin is non-inferior.
    """
    style.set_pub_style()
    fig, ax = _fig_ax(ax, figsize)
    est = np.asarray(estimates, float)
    lo = np.asarray(ci_low, float)
    hi = np.asarray(ci_high, float)
    y = np.arange(len(labels))[::-1]
    ax.hlines(y, lo, hi, color=style.OKABE_ITO["blue"], lw=2)
    ax.plot(est, y, "o", color=style.OKABE_ITO["blue"], ms=5)
    ax.axvline(ref, color="black", lw=0.8)
    if margin is not None:
        ax.axvline(margin, ls="--", color=style.OKABE_ITO["red"], lw=1.0)
        ax.annotate("margin", xy=(margin, 0.98), xycoords=("data", "axes fraction"),
                    color=style.OKABE_ITO["red"], fontsize=7, va="top", ha="left",
                    xytext=(2, 0), textcoords="offset points")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def _fig_ax(ax, figsize):
    """Return ``(fig, ax)`` creating a new figure only when ``ax`` is None."""
    if ax is None:
        fig, ax = style.plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


__all__ = ["bars", "scatter", "volcano", "lines", "forest"]
