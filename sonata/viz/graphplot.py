# sonata/viz/graphplot.py
"""Connectome node-link diagrams via NetworkX.

Renders a weighted brain graph (nodes = ROIs, edges = SC or predicted FC) as a
2D node-link figure, with node size/colour encoding a per-region scalar (e.g.
predicted functional strength) and edge width/alpha encoding weight. Useful for
inspecting the sparse structure a heatmap flattens.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import style


def connectome_graph(
    adjacency: np.ndarray,
    *,
    node_values: Sequence[float] | None = None,
    node_labels: Sequence[str] | None = None,
    threshold_quantile: float = 0.9,
    layout: str = "spring",
    node_cmap: str = "magma",
    title: str = "connectome",
    figsize: tuple[float, float] = (5.6, 5.2),
    seed: int = 0,
):
    """Draw a thresholded weighted graph from an adjacency matrix.

    Parameters
    ----------
    adjacency
        Square weighted adjacency (SC or FC). The upper triangle is used.
    node_values
        Per-node scalar mapped to node colour and size (default: weighted degree).
    node_labels
        Optional node labels (drawn only for the strongest nodes to avoid clutter).
    threshold_quantile
        Keep only edges above this weight quantile (sparsifies dense matrices).
    layout
        ``"spring"``, ``"circular"``, or ``"kamada_kawai"``.
    node_cmap, title, figsize, seed
        Cosmetic / reproducibility controls.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import networkx as nx

    style.set_pub_style()
    A = np.asarray(adjacency, float).copy()
    A = np.triu(A, 1)
    A[~np.isfinite(A)] = 0.0
    if threshold_quantile > 0:
        nz = A[A > 0]
        if nz.size:
            thr = np.quantile(nz, threshold_quantile)
            A[A < thr] = 0.0

    G = nx.from_numpy_array(A)
    if node_values is None:
        node_values = A.sum(0) + A.sum(1)  # weighted degree
    node_values = np.asarray(node_values, float)

    pos = {
        "spring": lambda: nx.spring_layout(G, seed=seed, weight="weight"),
        "circular": lambda: nx.circular_layout(G),
        "kamada_kawai": lambda: nx.kamada_kawai_layout(G),
    }.get(layout, lambda: nx.spring_layout(G, seed=seed))()

    fig, ax = style.plt.subplots(figsize=figsize)
    weights = np.array([d.get("weight", 1.0) for *_e, d in G.edges(data=True)])
    if weights.size:
        wn = weights / (weights.max() + 1e-12)
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.4 + 2.5 * wn,
                               alpha=0.25 + 0.5 * wn, edge_color=style.OKABE_ITO["grey"])
    sizes = 30 + 220 * _minmax(node_values)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=node_values,
                           cmap=node_cmap, linewidths=0.4, edgecolors="black")
    if node_labels is not None:
        top = np.argsort(node_values)[::-1][:10]
        nx.draw_networkx_labels(G, pos, ax=ax,
                                labels={int(i): str(node_labels[int(i)]) for i in top},
                                font_size=6)
    ax.set_title(title, loc="left")
    ax.axis("off")
    sm = style.plt.cm.ScalarMappable(cmap=node_cmap)
    sm.set_array(node_values)
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label="node value")
    fig.tight_layout()
    return fig


def _minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


__all__ = ["connectome_graph"]
