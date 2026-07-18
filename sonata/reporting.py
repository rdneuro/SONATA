# sonata/reporting.py
"""Publication-grade tables → CSV + LaTeX (booktabs) at every stage.

``save_table`` writes both a machine-readable CSV and a journal-ready ``.tex``
(``booktabs``, caption, label, sensible float formatting). Returns the two paths.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import SonataConfig
from .utils import get_logger

log = get_logger("sonata.report")


def save_table(df: pd.DataFrame, name: str, cfg: SonataConfig, *,
               caption: str = "", label: str | None = None,
               float_fmt: str = "%.3f", index: bool = False) -> tuple[Path, Path]:
    tdir = cfg.paths.output_dir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    csv_path = tdir / f"{name}.csv"
    tex_path = tdir / f"{name}.tex"
    df.to_csv(csv_path, index=index)

    label = label or f"tab:{name}"
    try:
        body = df.to_latex(index=index, escape=True, float_format=float_fmt,
                           caption=caption or name.replace("_", " ").title(),
                           label=label, bold_rows=False, longtable=False)
        # Upgrade to booktabs if pandas emitted \hline rules.
        body = (body.replace("\\toprule", "\\toprule")
                    .replace("\\hline\n\\hline", "\\midrule"))
    except Exception:  # pragma: no cover
        body = df.to_latex(index=index)
    tex_path.write_text(body)
    log.info("table → %s (+.tex)", csv_path.name)
    return csv_path, tex_path


def summary_comparison(results: dict, baselines_df: pd.DataFrame,
                       cfg: SonataConfig) -> pd.DataFrame:
    """One-row-per-model comparison: mean r, 95% CI, mean R², permutation p."""
    rows = []
    m = results["metrics"]
    rows.append({"model": f"SONATA ({cfg.model.edge_feature_mode})",
                 "mean_r": results["mean_r"],
                 "ci_low": results["ci_r"][0], "ci_high": results["ci_r"][1],
                 "mean_r2": float(m["r2"].mean()), "p_perm": results["p_perm"],
                 "n_subjects": int(m["subject_id"].nunique())})
    for model, g in baselines_df.groupby("model"):
        rows.append({"model": model, "mean_r": float(g["r"].mean()),
                     "ci_low": float(g["r"].quantile(0.025)),
                     "ci_high": float(g["r"].quantile(0.975)),
                     "mean_r2": float(g["r2"].mean()), "p_perm": float("nan"),
                     "n_subjects": int(g["subject_id"].nunique())})
    return pd.DataFrame(rows).sort_values("mean_r", ascending=False).reset_index(drop=True)
