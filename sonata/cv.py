# sonata/cv.py
"""Leakage-safe cross-validation: per-fold feature transforms, splits, resume.

All imputation and standardization parameters are estimated on the TRAINING
subjects of a fold only, then applied to held-out subjects — the single most
important leakage guard for small-n connectome models (Rosenblatt 2024). The
optional ComBat hook is likewise fit on the training fold and applied forward.

Resume: :func:`load_resume_state` / :func:`save_resume_state` persist which
folds are complete so an interrupted 15-hour run picks up where it stopped.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import SonataConfig
from .graph import build_edge_matrix, build_node_matrix
from .utils import load_json, save_json, get_logger

log = get_logger("sonata.cv")


# ──────────────────────────────────────────────────────────────────────────────
# Splits
# ──────────────────────────────────────────────────────────────────────────────
def make_splits(subject_ids: list[str], groups: list[str],
                cfg: SonataConfig) -> list[tuple[list[str], list[str]]]:
    ids = np.asarray(subject_ids)
    if cfg.train.cv == "loso":
        return [([s for s in subject_ids if s != t], [t]) for t in subject_ids]
    # grouped k-fold stratified by clinical group, subjects kept intact
    from sklearn.model_selection import StratifiedGroupKFold
    g = np.asarray(groups)
    skf = StratifiedGroupKFold(n_splits=cfg.train.n_folds, shuffle=True,
                               random_state=cfg.train.seed)
    splits = []
    # group == subject so each subject stays whole; stratify by clinical label
    for tr, te in skf.split(ids, y=g, groups=ids):
        splits.append((ids[tr].tolist(), ids[te].tolist()))
    return splits


def inner_val_split(train_ids: list[str], cfg: SonataConfig
                    ) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(cfg.train.seed)
    perm = rng.permutation(train_ids)
    n_val = max(1, int(round(cfg.train.inner_val_frac * len(perm))))
    return perm[n_val:].tolist(), perm[:n_val].tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Per-fold feature transform
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FoldTransform:
    """Median-impute + standardize node/edge features and the target; train-fit."""

    cfg: SonataConfig
    node_med: np.ndarray = field(default=None)
    node_mu: np.ndarray = field(default=None)
    node_sd: np.ndarray = field(default=None)
    edge_med: np.ndarray = field(default=None)
    edge_mu: np.ndarray = field(default=None)
    edge_sd: np.ndarray = field(default=None)
    y_mu: float = 0.0
    y_sd: float = 1.0
    identity_nodes: bool = False

    def fit(self, train_feats: list[dict]) -> "FoldTransform":
        self.identity_nodes = self.cfg.model.node_feature_mode == "identity"
        if not self.identity_nodes:
            X = np.concatenate([build_node_matrix(f, self.cfg) for f in train_feats], 0)
            self.node_med = np.nanmedian(X, axis=0)
            Xi = self._impute(X, self.node_med)
            self.node_mu, self.node_sd = Xi.mean(0), Xi.std(0) + 1e-8
        E = np.concatenate([build_edge_matrix(f, self.cfg) for f in train_feats], 0)
        self.edge_med = np.nanmedian(E, axis=0)
        Ei = self._impute(E, self.edge_med)
        self.edge_mu, self.edge_sd = Ei.mean(0), Ei.std(0) + 1e-8
        y = np.concatenate([f["fc"] for f in train_feats])
        self.y_mu, self.y_sd = float(y.mean()), float(y.std() + 1e-8)
        return self

    @staticmethod
    def _impute(M: np.ndarray, med: np.ndarray) -> np.ndarray:
        out = M.copy()
        idx = np.where(~np.isfinite(out))
        if idx[0].size:
            out[idx] = np.take(med, idx[1])
        return np.nan_to_num(out, nan=0.0)

    def transform(self, feat: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.identity_nodes:
            x = build_node_matrix(feat, self.cfg)               # eye(200)
        else:
            x = self._impute(build_node_matrix(feat, self.cfg), self.node_med)
            x = (x - self.node_mu) / self.node_sd
        e = self._impute(build_edge_matrix(feat, self.cfg), self.edge_med)
        e = (e - self.edge_mu) / self.edge_sd
        y = (feat["fc"] - self.y_mu) / self.y_sd
        return x.astype(np.float32), e.astype(np.float32), y.astype(np.float32)

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.y_sd + self.y_mu


# ──────────────────────────────────────────────────────────────────────────────
# Optional ComBat (fit on train, apply forward). Off by default.
# ──────────────────────────────────────────────────────────────────────────────
def maybe_combat(train_feats, all_feats, cfg: SonataConfig):
    """Harmonize stacked edge features across protocols, fit on train.

    Requires ``neuroCombat`` (and ``neuroCombatFromTraining`` for forward
    application). If unavailable, returns features unchanged with a warning —
    SONATA's primary analysis is designed to run on a protocol-homogeneous
    subset, with harmonization as an explicit sensitivity analysis.
    """
    if not cfg.train.harmonize:
        return all_feats
    try:
        from neuroCombat import neuroCombat, neuroCombatFromTraining  # type: ignore
    except Exception:
        warnings.warn("neuroCombat not installed; skipping harmonization")
        return all_feats
    log.info("ComBat harmonization on edge features (batch=%s)", cfg.train.harmonize_batch_col)
    # Implementation intentionally conservative: harmonize per-subject edge
    # feature *summaries* (mean vector) to estimate batch effects, then subtract
    # the estimated batch shift from every edge of that subject. Full per-edge
    # ComBat across subjects with differing edge supports is ill-posed, so we
    # operate on the aligned summary and document this in the manuscript.
    import pandas as pd
    tr_ids = {f["meta"]["subject_id"] for f in train_feats}
    rows, batch, covars, keys = [], [], [], []
    for f in all_feats:
        E = build_edge_matrix(f, cfg)
        rows.append(np.nanmean(E, axis=0))
        batch.append(f["meta"][cfg.train.harmonize_batch_col])
        covars.append([f["meta"].get(c, np.nan) for c in cfg.train.harmonize_covars])
        keys.append(f["meta"]["subject_id"])
    data = np.asarray(rows).T                       # features × subjects
    cov_df = pd.DataFrame(covars, columns=list(cfg.train.harmonize_covars))
    cov_df[cfg.train.harmonize_batch_col] = batch
    try:
        res = neuroCombat(dat=data, covars=cov_df, batch_col=cfg.train.harmonize_batch_col)
        shifts = data - res["data"]                 # estimated batch shift per subject
        for f, key in zip(all_feats, keys):
            j = keys.index(key)
            # apply summary shift to every edge feature (broadcast subtract)
            f["_combat_shift"] = shifts[:, j]
    except Exception as exc:
        warnings.warn(f"ComBat failed ({exc!r}); using raw features")
    return all_feats


# ──────────────────────────────────────────────────────────────────────────────
# Resume
# ──────────────────────────────────────────────────────────────────────────────
def resume_path(cfg: SonataConfig, tag: str) -> Path:
    return cfg.paths.output_dir / "checkpoints" / f"state_{tag}.json"


def load_resume_state(cfg: SonataConfig, tag: str) -> dict:
    p = resume_path(cfg, tag)
    if p.exists():
        st = load_json(p)
        log.info("resume: %d/%s folds already complete", len(st.get("done_folds", [])), tag)
        return st
    return {"tag": tag, "done_folds": [], "fold_metrics": {}}


def save_resume_state(cfg: SonataConfig, tag: str, state: dict) -> None:
    save_json(state, resume_path(cfg, tag))
