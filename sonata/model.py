# sonata/model.py
"""SONATA — the geometry-aware edge-conditioned message-passing network.

Design (grounded in the review):
- Base layer is an edge-conditioned MPNN with an *explicit edge update*
  (message = MLP([h_i, h_j, e_ij]); edge embedding refreshed each layer),
  in the spirit of Gilmer 2017 / Simonovsky–Komodakis ECC / Neudorf 2022 —
  not vanilla GraphSAGE, which has no edge-feature pathway.
- The supervised target is the FC weight on each *undirected* structural edge.
  The directed message-passing graph carries both orientations; the readout
  averages the two orientations per undirected edge (symmetry).
- Node functional strength is exposed only as a DERIVED readout (sum of
  predicted incident edges), never as an independent label — removing the
  node/edge target redundancy flagged in the review.

One graph per forward call (200 nodes is tiny); the trainer accumulates
gradients over several subjects per optimizer step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers[:-2])      # drop trailing activation+dropout


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean of ``src`` rows grouped by ``index`` (no torch_scatter dependency)."""
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    cnt = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    cnt.index_add_(0, index, torch.ones(src.size(0), device=src.device, dtype=src.dtype))
    return out / cnt.clamp_min(1.0).unsqueeze(1)


def scatter_mean_1d(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    cnt = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    cnt.index_add_(0, index, torch.ones_like(src))
    return out / cnt.clamp_min(1.0)


class SonataLayer(nn.Module):
    """One edge-conditioned message-passing layer with residual node+edge updates."""

    def __init__(self, h: int, e_h: int, dropout: float):
        super().__init__()
        self.edge_mlp = _mlp([2 * h + e_h, e_h, e_h], dropout)
        self.msg_mlp = _mlp([2 * h + e_h, h, h], dropout)
        self.node_mlp = _mlp([2 * h, h, h], dropout)
        self.enorm = nn.LayerNorm(e_h)
        self.hnorm = nn.LayerNorm(h)

    def forward(self, h: torch.Tensor, e: torch.Tensor,
                edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        h_src, h_dst = h[src], h[dst]
        cat = torch.cat([h_src, h_dst, e], dim=1)
        e = self.enorm(e + self.edge_mlp(cat))
        msg = self.msg_mlp(torch.cat([h_src, h_dst, e], dim=1))
        agg = scatter_mean(msg, dst, dim_size=h.size(0))
        h = self.hnorm(h + self.node_mlp(torch.cat([h, agg], dim=1)))
        return h, e


class Sonata(nn.Module):
    """Full SONATA network: encoders → L edge-conditioned layers → edge head."""

    def __init__(self, n_node_features: int, n_edge_features: int,
                 hidden: int = 64, edge_hidden: int = 64, n_layers: int = 3,
                 dropout: float = 0.15):
        super().__init__()
        self.node_enc = _mlp([n_node_features, hidden, hidden], dropout)
        self.edge_enc = _mlp([n_edge_features, edge_hidden, edge_hidden], dropout)
        self.layers = nn.ModuleList(
            SonataLayer(hidden, edge_hidden, dropout) for _ in range(n_layers))
        # Edge readout from final [h_i, h_j, e_ij] → scalar FC.
        self.readout = _mlp([2 * hidden + edge_hidden, hidden, 1], dropout)

    def forward(self, data, return_strength: bool = False):
        h = self.node_enc(data.x)
        e = self.edge_enc(data.edge_attr)
        ei = data.edge_index
        for layer in self.layers:
            h, e = layer(h, e, ei)
        src, dst = ei[0], ei[1]
        directed = self.readout(torch.cat([h[src], h[dst], e], dim=1)).squeeze(-1)
        # Average the two orientations → one prediction per undirected edge.
        pred = scatter_mean_1d(directed, data.edge_id, dim_size=int(data.num_undirected))
        if not return_strength:
            return pred
        # Derived node strength: sum predicted FC over incident undirected edges.
        edges = data.edge_index[:, : int(data.num_undirected)]  # first half are i<j? see graph.to_pyg_data
        strength = torch.zeros(int(data.x.size(0)), device=pred.device)
        u = data.edge_index[0, : int(data.num_undirected)]
        v = data.edge_index[1, : int(data.num_undirected)]
        strength.index_add_(0, u, pred)
        strength.index_add_(0, v, pred)
        return pred, strength


def build_model(n_node_features: int, n_edge_features: int, cfg) -> Sonata:
    return Sonata(
        n_node_features=n_node_features, n_edge_features=n_edge_features,
        hidden=cfg.model.hidden, edge_hidden=cfg.model.edge_hidden,
        n_layers=cfg.model.n_layers, dropout=cfg.model.dropout)
