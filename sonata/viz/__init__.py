# sonata/viz/__init__.py
"""SONATA visualization package.

A single visual language across four surfaces:

* :mod:`~sonata.viz.style`     — palette, publication rcParams, save helper.
* :mod:`~sonata.viz.heatmaps`  — solo labelled heatmap + evolutionary panel.
* :mod:`~sonata.viz.metrics`   — bar, scatter, volcano, line, forest.
* :mod:`~sonata.viz.panels`    — compose plots into multi-panel figures.
* :mod:`~sonata.viz.graphplot` — NetworkX connectome node-link diagrams.
* :mod:`~sonata.viz.brain3d`   — vedo / nilearn / yabplot 3D renders (lazy).

The 2D modules need only matplotlib/seaborn/networkx; ``brain3d`` imports its 3D
stack lazily so this package is importable anywhere.
"""

from __future__ import annotations

from . import brain3d, graphplot, heatmaps, metrics, panels, style
from .heatmaps import evolutionary_heatmaps, heatmap
from .metrics import bars, forest, lines, scatter, volcano
from .panels import grid, results_dashboard
from .style import save, set_pub_style

__all__ = [
    # submodules
    "style", "heatmaps", "metrics", "panels", "graphplot", "brain3d",
    # flat convenience API
    "set_pub_style", "save",
    "heatmap", "evolutionary_heatmaps",
    "bars", "scatter", "volcano", "lines", "forest",
    "grid", "results_dashboard",
]
