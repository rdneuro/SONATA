# tests/test_viz.py
"""Every 2D figure type renders on synthetic data; 3D degrades gracefully."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from sonata import viz


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _sym(rng, n=20):
    m = rng.standard_normal((n, n))
    return (m + m.T) / 2


def test_solo_heatmap(rng):
    fig = viz.heatmap(_sym(rng), title="fc", blocks=[5, 5, 10])
    assert fig is not None
    viz.style.plt.close(fig)


def test_evolutionary_panel(rng):
    mats = [_sym(rng) for _ in range(12)]
    fig = viz.evolutionary_heatmaps(mats, stage_labels=[f"t{i}" for i in range(12)],
                                    summary="mean_abs")
    # one axis per matrix + 1 barplot + 1 colourbar
    assert len(fig.axes) >= 13
    viz.style.plt.close(fig)


def test_metric_plots(rng):
    for fig in (
        viz.bars(["a", "b", "c"], [0.6, 0.3, 0.1], reference=0.6),
        viz.scatter(rng.standard_normal(100), rng.standard_normal(100)),
        viz.volcano(rng.standard_normal(50), 10 ** (-np.abs(rng.standard_normal(50)))),
        viz.lines({"train": (range(10), rng.random(10))}),
        viz.forest(["x"], [-0.1], [-0.2], [0.0], margin=-0.03),
    ):
        assert fig is not None
        viz.style.plt.close(fig)


def test_results_dashboard(rng):
    fig = viz.results_dashboard(
        model_labels=["a", "b"], model_scores=[0.6, 0.1], benchmark=0.6,
        pred=rng.standard_normal(100), true=rng.standard_normal(100),
        ni_labels=["x"], ni_estimate=[-0.1], ni_low=[-0.2], ni_high=[0.0],
        ni_margin=-0.03,
    )
    assert len(fig.axes) >= 3
    viz.style.plt.close(fig)


def test_connectome_graph(rng):
    A = np.abs(_sym(rng))
    np.fill_diagonal(A, 0)
    fig = viz.graphplot.connectome_graph(A, threshold_quantile=0.8)
    assert fig is not None
    viz.style.plt.close(fig)


def test_brain3d_raises_without_vedo():
    from sonata.backends.base import _installed

    if not _installed("vedo"):
        with pytest.raises(ImportError):
            viz.brain3d.surface_scalar(np.zeros((3, 3)), np.array([[0, 1, 2]]),
                                       np.zeros(3), "/tmp/_x.png")
