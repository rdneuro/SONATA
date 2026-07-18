# sonata/baselines.py
"""Reference models the GNN must beat (all leakage-safe: fit on train only).

- mean_fc        : per-edge mean empirical FC over train subjects (edge-keyed).
- sc_fc_linear   : Ridge FC ~ log-SIFT2 weight (the trivial coupling baseline).
- comm_gradient  : Ridge FC ~ [SC, communicability, gradient distance].

Each returns a per-subject metrics DataFrame (r, r2, mse) on the test subjects,
directly comparable to the SONATA results. The one-hot-node GNN baseline
(Chen/Cui-style) is produced by re-running ``train.run_cv`` with
``model.node_feature_mode='identity'`` and ``edge_feature_mode='sift2'``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import expm
from sklearn.linear_model import Ridge

from .config import SonataConfig
from .gradients import edge_gradient_distance, fit_gradients, rebuild_sc


def _metrics(pred, true, mask=None) -> dict:
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    if mask is not None:
        mask = np.asarray(mask, bool)
        pred, true = pred[mask], true[mask]
    m = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[m], true[m]
    if pred.size < 3 or pred.std() < 1e-9 or true.std() < 1e-9:
        return {"r": np.nan, "r2": np.nan, "mse": np.nan, "n_eval": int(pred.size)}
    r = float(np.corrcoef(pred, true)[0, 1])
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return {"r": r, "r2": 1 - ss_res / max(ss_tot, 1e-12),
            "mse": float(((true - pred) ** 2).mean()), "n_eval": int(pred.size)}


def _eval_mask(f):
    """Same edge mask as the model uses (imported lazily to avoid a cycle)."""
    from .train import edge_eval_mask
    from .config import SonataConfig
    return edge_eval_mask(f, _eval_mask._cfg)
_eval_mask._cfg = None  # set by run_all_baselines


def _edge_key(e):
    i, j = int(e[0]), int(e[1])
    return (i, j) if i < j else (j, i)


# ──────────────────────────────────────────────────────────────────────────────
def baseline_mean_fc(train_feats, test_feats) -> pd.DataFrame:
    acc: dict[tuple, list] = {}
    for f in train_feats:
        for e, y in zip(f["edges"], f["fc"]):
            acc.setdefault(_edge_key(e), []).append(float(y))
    edge_mean = {k: float(np.mean(v)) for k, v in acc.items()}
    glob = float(np.mean([y for v in acc.values() for y in v])) if acc else 0.0
    rows = []
    for f in test_feats:
        pred = np.array([edge_mean.get(_edge_key(e), glob) for e in f["edges"]])
        r = _metrics(pred, f["fc"], mask=_eval_mask(f)); r.update(
            subject_id=f["meta"]["subject_id"],
            group=f["meta"]["group"], model="mean_fc")
        rows.append(r)
    return pd.DataFrame(rows)


def communicability(W: np.ndarray) -> np.ndarray:
    s = np.sqrt(W.sum(1)); s[s == 0] = 1.0
    return expm(W / np.outer(s, s))


def _design(feat, G):
    W = rebuild_sc(feat)
    C = communicability(W)
    e = feat["edges"]
    sc = feat["sc_weight"]
    comm = C[e[:, 0], e[:, 1]]
    gd = edge_gradient_distance(e, G, axis=0)
    return np.column_stack([sc, comm, gd])


def baseline_sc_fc_linear(train_feats, test_feats) -> pd.DataFrame:
    Xtr = np.concatenate([f["sc_weight"][:, None] for f in train_feats])
    ytr = np.concatenate([f["fc"] for f in train_feats])
    mdl = Ridge(alpha=1.0).fit(Xtr, ytr)
    rows = []
    for f in test_feats:
        pred = mdl.predict(f["sc_weight"][:, None])
        r = _metrics(pred, f["fc"], mask=_eval_mask(f)); r.update(
            subject_id=f["meta"]["subject_id"],
            group=f["meta"]["group"], model="sc_fc_linear")
        rows.append(r)
    return pd.DataFrame(rows)


def baseline_comm_gradient(train_feats, test_feats) -> pd.DataFrame:
    G = fit_gradients(train_feats, n_components=5)
    Xtr = np.concatenate([_design(f, G) for f in train_feats])
    ytr = np.concatenate([f["fc"] for f in train_feats])
    ok = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
    mdl = Ridge(alpha=1.0).fit(Xtr[ok], ytr[ok])
    rows = []
    for f in test_feats:
        pred = mdl.predict(np.nan_to_num(_design(f, G)))
        r = _metrics(pred, f["fc"], mask=_eval_mask(f)); r.update(
            subject_id=f["meta"]["subject_id"],
            group=f["meta"]["group"], model="comm_gradient")
        rows.append(r)
    return pd.DataFrame(rows)


def run_all_baselines(all_feats, cfg: SonataConfig) -> pd.DataFrame:
    """Run every non-GNN baseline under the same CV splits as the GNN."""
    from .cv import make_splits
    _eval_mask._cfg = cfg  # baselines use the SAME edge mask as the model
    feat_by_id = {f["meta"]["subject_id"]: f for f in all_feats}
    ids = list(feat_by_id); groups = [feat_by_id[s]["meta"]["group"] for s in ids]
    splits = make_splits(ids, groups, cfg)
    out = []
    for tr_ids, te_ids in splits:
        tr = [feat_by_id[s] for s in tr_ids]; te = [feat_by_id[s] for s in te_ids]
        out.append(baseline_mean_fc(tr, te))
        out.append(baseline_sc_fc_linear(tr, te))
        out.append(baseline_comm_gradient(tr, te))
    return pd.concat(out, ignore_index=True)
