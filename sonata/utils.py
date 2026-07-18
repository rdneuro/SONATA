# sonata/utils.py
"""Cross-cutting utilities: logging, seeding, progress, checkpoint/resume, manifest IO.

No torch import at module top level so the non-GNN parts of SONATA (feature
extraction, baselines, stats) import cleanly on machines without CUDA.
"""

from __future__ import annotations

import json
import logging
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

# ── Rich progress with a graceful tqdm/plain fallback ─────────────────────────
try:  # pragma: no cover - environment dependent
    from rich.console import Console
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TextColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    _RICH = True
    _console = Console()
except Exception:  # pragma: no cover
    _RICH = False
    _console = None


def get_logger(name: str = "sonata", level: int = logging.INFO,
               logfile: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s │ %(levelname)-7s │ %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if present) torch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic but not at the cost of throughput on the RTX 3090.
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


@contextmanager
def progress(description: str, total: int | None = None) -> Iterator[Any]:
    """Unified progress context. Yields an ``update(n=1)`` callable."""
    if _RICH:
        cols = [SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                TimeRemainingColumn()]
        with Progress(*cols, console=_console, transient=False) as p:
            task = p.add_task(description, total=total)
            yield lambda n=1: p.advance(task, n)
    else:  # pragma: no cover
        try:
            from tqdm import tqdm
            bar = tqdm(total=total, desc=description)
            yield lambda n=1: bar.update(n)
            bar.close()
        except Exception:
            print(f"[{description}] (no progress backend)")
            yield lambda n=1: None


def track(iterable: Iterable, description: str, total: int | None = None) -> Iterator:
    """Iterate with a progress bar (rich/tqdm/plain)."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = None
    with progress(description, total) as adv:
        for item in iterable:
            yield item
            adv(1)


# ── Checkpoint / resume ───────────────────────────────────────────────────────
def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ── Manifest ──────────────────────────────────────────────────────────────────
REQUIRED_MANIFEST_COLS = ["subject_id", "group", "fs_subject_dir",
                          "tractseg_dir", "sift2_csv"]


def load_manifest(path: Path) -> pd.DataFrame:
    """Load and validate the subject manifest.

    A functional-connectivity source is required: either ``fc_csv`` (a
    precomputed ROI×ROI matrix) or ``fc_timeseries_csv`` (T×ROI BOLD, from which
    SONATA computes Pearson FC). PHI columns such as patient names are dropped
    on load if present.
    """
    df = pd.read_csv(path)
    # Drop any PHI-ish free-text name columns defensively.
    for phi in ("name", "patient_name", "nome", "full_name"):
        if phi in df.columns:
            df = df.drop(columns=[phi])
    missing = [c for c in REQUIRED_MANIFEST_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest {path} missing required columns: {missing}")
    if "fc_csv" not in df.columns and "fc_timeseries_csv" not in df.columns:
        raise ValueError("Manifest needs 'fc_csv' or 'fc_timeseries_csv'.")
    for opt, default in (("protocol", "P1"), ("age", np.nan), ("sex", "U")):
        if opt not in df.columns:
            df[opt] = default
    df["subject_id"] = df["subject_id"].astype(str)
    return df.reset_index(drop=True)


def fisher_z(r: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r = np.clip(r, -1 + eps, 1 - eps)
    return np.arctanh(r)
