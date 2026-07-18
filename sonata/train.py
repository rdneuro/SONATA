# sonata/train.py
"""Training and evaluation with leakage-safe CV, checkpoint/resume, and inference.

- One graph per forward; gradients accumulated over ``batch_subjects`` graphs.
- Early stopping on an inner validation split of the training fold.
- Resume at two granularities: completed folds are skipped (metrics restored
  from ``state.json``); an interrupted fold resumes from its last epoch
  checkpoint (``fold_{k}_progress.pt``).
- Headline metric: per-subject Pearson r between predicted and empirical FC on
  the structural edge set, with a within-subject permutation null and a
  subject-level bootstrap CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import SonataConfig
from .cv import (FoldTransform, inner_val_split, load_resume_state, make_splits,
                 maybe_combat, save_resume_state)
from .graph import to_pyg_data
from .model import build_model
from .utils import get_logger, progress, seed_everything

log = get_logger("sonata.train")


def _device(cfg: SonataConfig):
    import torch
    if cfg.train.device == "cuda" and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log.info("GPU: %s │ %.1f GB", p.name, p.total_memory / 1e9)  # total_memory (not total_mem)
        return torch.device("cuda")
    return torch.device("cpu")


def _pack(feat, transform, cfg, device):
    x, e, y = transform.transform(feat)
    data = to_pyg_data(feat, x, e, cfg)
    data = data.to(device)
    import torch
    data.y = torch.tensor(y, dtype=torch.float, device=device)
    return data


def edge_eval_mask(feat: dict, cfg: SonataConfig) -> np.ndarray | None:
    """Boolean edge mask defining the evaluation target (Enquadramento A).

    Driven by ``cfg.train.target_edge_mask``:
      - "all"    : every edge (legacy behaviour; returns None → no masking).
      - "spectral": edges with spectral coverage > 0 (Aim 1 target).
      - "spectral_diffusion_intersection": edges covered by BOTH spectral and
        diffusion features (Aim 2 paired-comparison target).
    The mask is aligned to ``feat["edges"]`` order, so it lines up with the
    per-subject pred/true vectors produced by evaluate()/baselines.
    """
    mode = getattr(cfg.train, "target_edge_mask", "all")
    if mode == "all":
        return None
    cov_s = np.asarray(feat.get("edge_spectral_cover"), float)
    if mode == "spectral":
        return cov_s > 0
    if mode == "spectral_diffusion_intersection":
        cov_d = np.asarray(feat.get("edge_diffusion_cover"), float)
        return (cov_s > 0) & (cov_d > 0)
    raise ValueError(f"unknown target_edge_mask: {mode!r}")


def _metrics(pred: np.ndarray, true: np.ndarray,
             mask: np.ndarray | None = None) -> dict:
    """Per-subject FC-prediction metrics, optionally restricted to a subset of
    edges (Enquadramento A). ``mask`` is a boolean array aligned to ``pred``/
    ``true`` (i.e. to the subject's edge order); only masked-in edges are scored.
    The same mask is applied identically for the model and every baseline, so a
    comparison is always made on one common edge set. ``n_eval`` records how many
    edges entered the metric (provenance for the paper).
    """
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


# ──────────────────────────────────────────────────────────────────────────────
def train_one_fold(train_feats, val_feats, transform, cfg, device,
                   fold_ckpt: Path):
    import torch
    sample = transform.transform(train_feats[0])
    model = build_model(sample[0].shape[1], sample[1].shape[1], cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)
    loss_fn = torch.nn.MSELoss()

    start_epoch, best_val, best_state, bad = 0, np.inf, None, 0
    prog_ckpt = fold_ckpt.with_name(fold_ckpt.stem + "_progress.pt")
    if prog_ckpt.exists():
        st = torch.load(prog_ckpt, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); start_epoch = st["epoch"] + 1
        best_val = st["best_val"]; best_state = st["best_state"]; bad = st["bad"]
        log.info("  resume fold at epoch %d (best_val=%.4f)", start_epoch, best_val)

    train_packed = [_pack(f, transform, cfg, device) for f in train_feats]
    val_packed = [_pack(f, transform, cfg, device) for f in val_feats]

    with progress("  epochs", total=cfg.train.epochs) as adv:
        if start_epoch:
            adv(start_epoch)
        for epoch in range(start_epoch, cfg.train.epochs):
            model.train()
            perm = np.random.permutation(len(train_packed))
            opt.zero_grad()
            for step, idx in enumerate(perm):
                data = train_packed[idx]
                pred = model(data)
                loss = loss_fn(pred, data.y) / cfg.train.batch_subjects
                loss.backward()
                if (step + 1) % cfg.train.batch_subjects == 0 or step == len(perm) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step(); opt.zero_grad()
            sched.step()

            model.eval()
            with torch.no_grad():
                vloss = float(np.mean([loss_fn(model(d), d.y).item() for d in val_packed]))
            if vloss < best_val - 1e-5:
                best_val, bad = vloss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if epoch % 10 == 0 or bad == 0:
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "best_val": best_val, "best_state": best_state, "bad": bad},
                           prog_ckpt)
            adv(1)
            if bad >= cfg.train.patience:
                log.info("  early stop @ epoch %d (best_val=%.4f)", epoch, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model": model.state_dict()}, fold_ckpt)
    if prog_ckpt.exists():
        prog_ckpt.unlink()
    return model


def evaluate(model, test_feats, transform, cfg, device,
             permute: bool = False) -> list[dict]:
    import torch
    model.eval()
    rng = np.random.default_rng(cfg.train.seed)
    rows = []
    with torch.no_grad():
        for f in test_feats:
            data = _pack(f, transform, cfg, device)
            pred = model(data).cpu().numpy()
            true = transform.inverse_y(data.y.cpu().numpy())
            pred = transform.inverse_y(pred)
            if permute:
                true = true[rng.permutation(len(true))]
            row = _metrics(pred, true, mask=edge_eval_mask(f, cfg))
            row["subject_id"] = f["meta"]["subject_id"]
            row["group"] = f["meta"]["group"]
            rows.append(row)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
def run_cv(all_feats: list[dict], cfg: SonataConfig, tag: str) -> dict:
    """Full cross-validation with fold-level resume. Returns metrics + nulls."""
    seed_everything(cfg.train.seed)
    device = _device(cfg)
    ckpt_dir = cfg.paths.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    feat_by_id = {f["meta"]["subject_id"]: f for f in all_feats}
    ids = list(feat_by_id)
    groups = [feat_by_id[s]["meta"]["group"] for s in ids]
    splits = make_splits(ids, groups, cfg)

    state = load_resume_state(cfg, tag)
    per_subject: list[dict] = []
    null_r: list[float] = []

    for k, (tr_ids, te_ids) in enumerate(splits):
        if k in state["done_folds"]:
            per_subject.extend(state["fold_metrics"][str(k)])
            continue
        log.info("[%s] fold %d/%d │ train=%d test=%d", tag, k + 1, len(splits),
                 len(tr_ids), len(te_ids))
        inner_tr, inner_val = inner_val_split(tr_ids, cfg)
        train_feats = [feat_by_id[s] for s in inner_tr]
        val_feats = [feat_by_id[s] for s in inner_val]
        train_feats = maybe_combat(train_feats, train_feats, cfg)

        transform = FoldTransform(cfg).fit([feat_by_id[s] for s in tr_ids])
        model = train_one_fold(train_feats, val_feats, transform, cfg, device,
                               ckpt_dir / f"{tag}_fold{k}.pt")

        test_feats = [feat_by_id[s] for s in te_ids]
        fold_rows = evaluate(model, test_feats, transform, cfg, device)
        for _ in range(max(1, cfg.train.n_permutations // max(len(splits), 1))):
            null_r.extend([r["r"] for r in
                           evaluate(model, test_feats, transform, cfg, device, permute=True)])
        per_subject.extend(fold_rows)
        state["done_folds"].append(k)
        state["fold_metrics"][str(k)] = fold_rows
        save_resume_state(cfg, tag, state)

    metrics = pd.DataFrame(per_subject)
    obs_r = float(metrics["r"].mean())
    null_r = np.asarray([x for x in null_r if np.isfinite(x)], float)
    p_perm = float((np.sum(null_r >= obs_r) + 1) / (len(null_r) + 1)) if null_r.size else np.nan
    ci = bootstrap_ci(metrics["r"].dropna().to_numpy(), cfg.train.n_bootstrap, cfg.train.seed)
    return {"metrics": metrics, "mean_r": obs_r, "p_perm": p_perm,
            "ci_r": ci, "null_r": null_r, "tag": tag}


def bootstrap_ci(x: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))
